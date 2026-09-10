"""F16: an image id must resolve correctly on both sides of a mount boundary.

The bug being locked out: api/main.py wrote a host-absolute path
(`/home/.../data/uploads/x.png`) into the job payload, and the worker
opened it verbatim. Natively that worked, because both processes saw the
same filesystem. With the worker in a container the path does not exist,
so the worker had an image id it could not turn into pixels — failing
silently, since a missing image just produced None.

The property under test is that the id plus each process's own root is
enough, and that the relative layout below the root is identical.
"""
from __future__ import annotations

import hashlib

import pytest

from retina_worker.config import DEFAULT_IMAGE_ROOT, Settings, image_relpath
from retina_worker.schemas import InferenceJob


def _id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestImageRelpath:
    def test_is_relative(self):
        assert not image_relpath(_id_for(b"x")).is_absolute()

    def test_is_sharded_on_the_first_two_characters(self):
        image_id = _id_for(b"x")

        rel = image_relpath(image_id)

        assert rel.parts[0] == image_id[:2]
        assert rel.name == f"{image_id}.png"

    def test_same_bytes_give_the_same_location(self):
        assert image_relpath(_id_for(b"same")) == image_relpath(_id_for(b"same"))

    def test_different_bytes_give_different_locations(self):
        assert image_relpath(_id_for(b"a")) != image_relpath(_id_for(b"b"))

    @pytest.mark.parametrize(
        "bad_id",
        ["../etc/passwd", "a/b", "a\\b", "x", "", "..", "/abs"],
    )
    def test_separators_and_stubs_are_rejected(self, bad_id):
        """A traversal attempt must fail loudly, not be joined onto a root."""
        with pytest.raises(ValueError):
            image_relpath(bad_id)


class TestCrossBoundaryResolution:
    def test_two_roots_agree_on_the_relative_layout(self, tmp_path):
        """The actual cross-boundary property: a host process and a
        containerized one resolve the same id to the same place under
        their own different roots."""
        image_id = _id_for(b"pixels")
        host = Settings(image_root=str(tmp_path / "host" / "data" / "images"))
        container = Settings(image_root="/data/images")

        host_path = host.image_path(image_id)
        container_path = container.image_path(image_id)

        assert host_path != container_path
        assert host_path.relative_to(host.resolved_image_root()) == (
            container_path.relative_to(container.resolved_image_root())
        )

    def test_worker_reads_what_the_submitter_wrote(self, tmp_path):
        """Simulates the two sides sharing one directory through different
        mount points, which is what the bind mount provides."""
        shared = tmp_path / "shared"
        submitter = Settings(image_root=str(shared))
        worker = Settings(image_root=str(shared))
        data = b"\x89PNG fake bytes"
        image_id = _id_for(data)

        dest = submitter.image_path(image_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

        assert worker.image_path(image_id).read_bytes() == data

    def test_env_var_overrides_the_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RETINA_IMAGE_ROOT", str(tmp_path))

        assert Settings().resolved_image_root() == tmp_path.resolve()

    def test_default_root_is_absolute_and_not_cwd_relative(self):
        assert Settings().resolved_image_root() == DEFAULT_IMAGE_ROOT
        assert DEFAULT_IMAGE_ROOT.is_absolute()


class TestJobPayloadCarriesNoPath:
    def test_schema_has_no_image_path_field(self):
        """The field is gone, so a host path cannot be serialized even by
        accident — the regression is structurally impossible, not just
        avoided by convention."""
        assert "image_path" not in InferenceJob.model_fields

    def test_serialized_job_contains_no_filesystem_path(self):
        job = InferenceJob(job_id="j1", image_id=_id_for(b"x"))

        payload = job.model_dump_json()

        assert "/" not in payload.split('"image_id"')[1].split(",")[0]
        assert "image_path" not in payload
