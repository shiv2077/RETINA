"""F22b: checkpoint resolution must not depend on the working directory.

PatchCoreRegistry used to hardcode `Path("./checkpoints")`, so a worker
started from anywhere other than the repo root looked for checkpoints in a
directory that did not exist and reported zero trained categories — the
same class of bug as storing an absolute host path in the job payload,
just in the other direction.

These live in the default (non-integration) suite on purpose: the
resolution logic sits on Settings, so it is testable without importing
torch or anomalib.
"""
from __future__ import annotations

import os
from pathlib import Path

from retina_worker.config import DEFAULT_CHECKPOINT_DIR, Settings


class TestResolvedCheckpointDir:
    def test_default_is_absolute(self):
        assert Settings().resolved_checkpoint_dir().is_absolute()

    def test_default_points_at_the_repo_checkpoints_dir(self):
        assert Settings().resolved_checkpoint_dir() == DEFAULT_CHECKPOINT_DIR
        assert DEFAULT_CHECKPOINT_DIR.name == "checkpoints"

    def test_default_is_identical_from_any_working_directory(self, tmp_path):
        """The regression itself: same answer from the repo root and from /tmp."""
        original = Path.cwd()
        try:
            os.chdir(original)
            from_repo = Settings().resolved_checkpoint_dir()
            os.chdir(tmp_path)
            from_elsewhere = Settings().resolved_checkpoint_dir()
        finally:
            os.chdir(original)

        assert from_repo == from_elsewhere

    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATCHCORE_CHECKPOINT_PATH", str(tmp_path))

        assert Settings().resolved_checkpoint_dir() == tmp_path.resolve()

    def test_env_override_is_absolutised(self, monkeypatch):
        monkeypatch.setenv("PATCHCORE_CHECKPOINT_PATH", "relative/checkpoints")

        resolved = Settings().resolved_checkpoint_dir()

        assert resolved.is_absolute()
        assert resolved.parts[-2:] == ("relative", "checkpoints")

    def test_the_repo_checkpoints_actually_resolve(self):
        """Guards the parents[3] arithmetic: if the package is ever moved,
        the default silently points somewhere wrong and this catches it."""
        expected = Path(__file__).resolve().parents[2] / "checkpoints"

        assert DEFAULT_CHECKPOINT_DIR == expected
