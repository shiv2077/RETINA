"""Unit tests for RedisClient against an in-process Redis double.

These document current behaviour. Where a known bug is in scope elsewhere
(the unacknowledged-entry path, the result-clobbering path), the test
asserts what the code does today rather than what it should do, so this
suite stays green and the behavioural fix lands with its own regression
test rather than silently turning this file red.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from retina_worker.redis_client import (
    JOB_DLQ_STREAM,
    JOB_QUEUE_STREAM,
    KEY_PREFIX,
    WORKER_GROUP,
)
from retina_worker.schemas import (
    InferenceJob,
    InferenceResult,
    JobStatus,
    ModelType,
    PipelineStage,
)


def _job(job_id: str = "job1") -> InferenceJob:
    return InferenceJob(
        job_id=job_id,
        image_id=job_id,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
        image_path=f"/tmp/{job_id}.png",
    )


def _enqueue(server_client, job: InferenceJob) -> str:
    return server_client.xadd(
        JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()}
    )


class TestConsumerGroup:
    def test_group_is_created_on_init(self, redis_client):
        groups = redis_client.client.xinfo_groups(JOB_QUEUE_STREAM)
        assert [g["name"] for g in groups] == [WORKER_GROUP]

    def test_init_is_idempotent(self, redis_client, settings):
        # Constructing a second client against the same server must not raise
        # on BUSYGROUP.
        from retina_worker.redis_client import RedisClient

        second = RedisClient(settings)
        assert second.health_check() is True

    def test_health_check_true_when_reachable(self, redis_client):
        assert redis_client.health_check() is True

    def test_scaled_out_replicas_register_as_distinct_consumers(
        self, patched_redis, settings
    ):
        """F7: `docker compose up --scale worker=4` must not collide every
        replica onto one consumer identity."""
        from retina_worker.redis_client import RedisClient

        a = RedisClient(settings.model_copy(update={"consumer_name": "worker-a"}))
        b = RedisClient(settings.model_copy(update={"consumer_name": "worker-b"}))
        _enqueue(a.client, _job("j1"))
        _enqueue(a.client, _job("j2"))
        a.read_job(block_ms=1)
        b.read_job(block_ms=1)

        consumers = a.client.xinfo_consumers(JOB_QUEUE_STREAM, WORKER_GROUP)

        assert sorted(c["name"] for c in consumers) == ["worker-a", "worker-b"]

    def test_consumer_name_defaults_to_the_hostname(self, monkeypatch):
        from retina_worker.config import Settings

        monkeypatch.setattr("socket.gethostname", lambda: "retina-worker-3")
        assert Settings().consumer_name == "retina-worker-3"

    def test_consumer_name_env_override(self, monkeypatch):
        from retina_worker.config import Settings

        monkeypatch.setenv("WORKER_CONSUMER_NAME", "custom-1")
        assert Settings().consumer_name == "custom-1"


class TestReadAndAcknowledge:
    def test_read_job_returns_none_on_empty_stream(self, redis_client):
        assert redis_client.read_job(block_ms=1) is None

    def test_read_job_roundtrips_an_enqueued_job(self, redis_client):
        _enqueue(redis_client.client, _job("abc123"))

        got = redis_client.read_job(block_ms=1)

        assert got is not None
        entry_id, job = got
        assert job.job_id == "abc123"
        assert job.image_path == "/tmp/abc123.png"
        assert entry_id

    def test_unacknowledged_job_stays_pending(self, redis_client):
        _enqueue(redis_client.client, _job())
        redis_client.read_job(block_ms=1)

        pending = redis_client.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)

        assert pending["pending"] == 1

    def test_acknowledge_clears_the_pending_entry(self, redis_client):
        _enqueue(redis_client.client, _job())
        entry_id, _ = redis_client.read_job(block_ms=1)

        assert redis_client.acknowledge_job(entry_id) is True
        assert (
            redis_client.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)["pending"]
            == 0
        )

    def test_a_second_read_does_not_redeliver_an_unacked_entry(self, redis_client):
        """">" delivers only new messages, so an unacked entry is not
        redelivered without an explicit XCLAIM — which nothing calls."""
        _enqueue(redis_client.client, _job())
        first = redis_client.read_job(block_ms=1)
        second = redis_client.read_job(block_ms=1)

        assert first is not None
        assert second is None


class TestPoisonedEntries:
    """F2: an unparseable entry must not be stranded in the PEL forever."""

    def _pending(self, redis_client) -> int:
        return redis_client.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)["pending"]

    def _dlq(self, redis_client) -> int:
        return redis_client.client.xlen(JOB_DLQ_STREAM)

    def test_unparseable_json_is_dead_lettered(self, redis_client):
        redis_client.client.xadd(JOB_QUEUE_STREAM, {"job_data": "{not json at all"})

        assert redis_client.read_job(block_ms=1) is None
        assert self._pending(redis_client) == 0
        assert self._dlq(redis_client) == 1

    def test_schema_violation_is_dead_lettered(self, redis_client):
        redis_client.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": json.dumps({"job_id": "x"})}
        )

        assert redis_client.read_job(block_ms=1) is None
        assert self._pending(redis_client) == 0
        assert self._dlq(redis_client) == 1

    def test_missing_job_data_field_is_dead_lettered(self, redis_client):
        redis_client.client.xadd(JOB_QUEUE_STREAM, {"nonsense": "1"})

        assert redis_client.read_job(block_ms=1) is None
        assert self._pending(redis_client) == 0
        assert self._dlq(redis_client) == 1

    def test_dlq_entry_records_reason_and_timestamp(self, redis_client):
        redis_client.client.xadd(JOB_QUEUE_STREAM, {"job_data": "{not json"})
        redis_client.read_job(block_ms=1)

        _entry_id, fields = redis_client.client.xrange(JOB_DLQ_STREAM)[0]

        assert fields["reason"]
        assert fields["failed_at"]
        assert fields["job_data"] == "{not json"


class TestReclaim:
    """F2: nothing else in the codebase claims stale pending entries."""

    def _abandon(self, redis_client, settings, job_id="stale"):
        """Read an entry as another consumer and never acknowledge it."""
        from retina_worker.redis_client import RedisClient

        other = RedisClient(settings.model_copy(update={"consumer_name": "dead-1"}))
        _enqueue(other.client, _job(job_id))
        other.read_job(block_ms=1)

    def test_stale_entry_is_reclaimed_and_returned(self, redis_client, settings):
        self._abandon(redis_client, settings)

        reclaimed = redis_client.reclaim_stale_jobs(min_idle_ms=0, max_deliveries=5)

        assert [job.job_id for _entry_id, job in reclaimed] == ["stale"]

    def test_fresh_entry_is_left_alone(self, redis_client, settings):
        self._abandon(redis_client, settings)

        assert redis_client.reclaim_stale_jobs(
            min_idle_ms=60_000, max_deliveries=5
        ) == []

    def test_entry_over_the_delivery_cap_is_dead_lettered(
        self, redis_client, settings
    ):
        self._abandon(redis_client, settings)

        reclaimed = redis_client.reclaim_stale_jobs(min_idle_ms=0, max_deliveries=0)

        assert reclaimed == []
        assert redis_client.client.xlen(JOB_DLQ_STREAM) == 1
        assert (
            redis_client.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)["pending"] == 0
        )


class TestStoreResult:
    def _result(self, job_id: str = "job1", status=JobStatus.COMPLETED):
        return InferenceResult(
            job_id=job_id,
            image_id=job_id,
            status=status,
            created_at=datetime.utcnow(),
            stage=PipelineStage.UNSUPERVISED,
            anomaly_score=0.42,
            is_anomaly=False,
            confidence=0.9,
            product_confidence=0.8,
            vlm_api_cost_estimate_usd=0.0,
            stage2_confidence=None,
        )

    def test_result_is_readable_after_store(self, redis_client):
        redis_client.store_result(self._result())

        stored = redis_client.get_result("job1")

        assert stored is not None
        assert stored["job_id"] == "job1"
        assert stored["anomaly_score"] == pytest.approx(0.42)

    def test_result_key_gets_a_seven_day_ttl(self, redis_client):
        redis_client.store_result(self._result())

        ttl = redis_client.client.ttl(f"{KEY_PREFIX}:results:job1")

        assert 0 < ttl <= 7 * 24 * 60 * 60

    def test_store_result_writes_the_image_to_job_mapping(self, redis_client):
        redis_client.store_result(self._result())

        mapping = redis_client.client.hgetall(f"{KEY_PREFIX}:images:job1")

        assert mapping["latest_job_id"] == "job1"

    def test_job_status_hash_expires_with_its_result(self, redis_client):
        """F18: retina:jobs:* used to live forever while retina:results:*
        expired after 7 days."""
        redis_client.update_job_status("job1", JobStatus.PROCESSING)

        ttl = redis_client.client.ttl(f"{KEY_PREFIX}:jobs:job1")

        assert 0 < ttl <= 7 * 24 * 60 * 60


class TestLabelingPool:
    def test_sample_is_scored_by_uncertainty_not_anomaly_score(self, redis_client):
        redis_client.add_to_labeling_pool(
            image_id="img1", anomaly_score=0.95, uncertainty_score=0.10
        )

        pool = redis_client.client.zrange(
            f"{KEY_PREFIX}:al:pool", 0, -1, withscores=True
        )

        assert pool == [("img1", pytest.approx(0.10))]

    def test_sample_metadata_is_stored_as_json(self, redis_client):
        redis_client.add_to_labeling_pool(
            image_id="img1", anomaly_score=0.95, uncertainty_score=0.10
        )

        raw = redis_client.client.get(f"{KEY_PREFIX}:al:samples:img1")
        meta = json.loads(raw)

        assert meta["image_id"] == "img1"
        assert meta["anomaly_score"] == pytest.approx(0.95)

    def test_pool_is_trimmed_to_max_size(self, redis_client, settings):
        max_size = settings.al_pool_max_size
        for i in range(max_size + 5):
            redis_client.add_to_labeling_pool(
                image_id=f"img{i}", anomaly_score=0.5, uncertainty_score=i / 1000
            )

        assert redis_client.client.zcard(f"{KEY_PREFIX}:al:pool") == max_size

    def test_evicted_samples_take_their_metadata_keys_with_them(
        self, redis_client, settings
    ):
        """F18: trimming the sorted set used to orphan the matching
        retina:al:samples:* strings, which have no TTL."""
        max_size = settings.al_pool_max_size
        for i in range(max_size + 5):
            redis_client.add_to_labeling_pool(
                image_id=f"img{i}", anomaly_score=0.5, uncertainty_score=i / 1000
            )

        remaining = redis_client.client.keys(f"{KEY_PREFIX}:al:samples:*")

        assert len(remaining) == max_size
        assert f"{KEY_PREFIX}:al:samples:img0" not in remaining


class TestStats:
    def test_increment_completed_jobs_counts_up(self, redis_client):
        assert redis_client.increment_completed_jobs() == 1
        assert redis_client.increment_completed_jobs() == 2
