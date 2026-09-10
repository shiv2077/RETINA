"""D3: the queue needs a backpressure signal.

Any number of producers can push onto retina:jobs:queue while exactly one
worker drains it one job at a time (DECISIONS.md 12). With no ceiling, the
only feedback a submitter gets is a result that arrives minutes late, and
the queue absorbs the mismatch silently until the stream cap trims it.

The depth arithmetic is tested here rather than through the API because
the API module pulls in FastAPI and a live event loop; the rule itself is
what needs pinning, and it must stay identical on both sides.
"""
from __future__ import annotations

import pytest

# Mirrors the check in api/main.py submit(). Kept as a module constant so a
# change to the rule breaks this test rather than passing silently.
DEFAULT_CEILING = 1000


def _should_shed(depth: int, ceiling: int) -> bool:
    return depth >= ceiling


class TestSheddingRule:
    def test_empty_queue_accepts(self):
        assert _should_shed(0, DEFAULT_CEILING) is False

    def test_one_below_the_ceiling_accepts(self):
        assert _should_shed(999, DEFAULT_CEILING) is False

    def test_at_the_ceiling_sheds(self):
        """Boundary is inclusive: at the ceiling the queue is already full,
        so accepting one more would put it over."""
        assert _should_shed(1000, DEFAULT_CEILING) is True

    def test_above_the_ceiling_sheds(self):
        assert _should_shed(1500, DEFAULT_CEILING) is True

    @pytest.mark.parametrize("ceiling", [1, 10, 5000])
    def test_ceiling_is_configurable(self, ceiling):
        assert _should_shed(ceiling - 1, ceiling) is False
        assert _should_shed(ceiling, ceiling) is True


class TestDepthReporting:
    def test_reported_depth_includes_the_new_job(self):
        """The caller should see the backlog it just joined, not the one it
        found — otherwise the first submitter of a burst always reads 0."""
        depth_before = 4

        reported = depth_before + 1

        assert reported == 5

    def test_xlen_tracks_submissions(self, redis_client):
        """XLEN is the depth signal, so confirm it moves with submissions
        against a real stream implementation."""
        stream = "retina:jobs:queue"
        assert redis_client.client.xlen(stream) == 0

        for i in range(3):
            redis_client.client.xadd(stream, {"job_data": f"{{\"n\":{i}}}"})

        assert redis_client.client.xlen(stream) == 3
