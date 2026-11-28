"""Resume an interrupted classification run (pipeline.run_full_pipeline)
from where it left off, instead of starting over.

Each expensive stage saves its result under ``outputs/checkpoints/``:

- the fitted feature pipeline (cleaning + Boruta/RFECV/... selection)
- each Round-1 candidate's validation score
- every Optuna study, trial by trial (SQLite storage) -- an interrupted
  search continues with the remaining trials, completed ones are kept
- the fitted tabular finalist, the train+validation
  refit, and the production refit

The checkpoint folder is tied to a fingerprint of the run's configuration
and input data files: if any setting or data file changed since the
checkpoints were written, they are discarded and the run starts fresh
(resuming with different settings would mix results from two different
runs). A run that finishes successfully deletes its checkpoints, so they
only ever exist for an interrupted run.

Code changes are NOT part of the fingerprint -- after editing the code,
start the next run fresh (config.resume_from_checkpoint = False / the
``--fresh-start``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import joblib

from .config import Config

logger = logging.getLogger(__name__)

# Settings that control checkpointing itself, not the run's results.
_FINGERPRINT_EXCLUDED_FIELDS = {"resume_from_checkpoint"}


def _file_signature(path) -> Optional[list]:
    p = Path(path) if path else None
    if p is None or not p.exists():
        return None
    stat = p.stat()
    return [str(p), stat.st_size, stat.st_mtime_ns]


def run_fingerprint(config: Config) -> str:
    """Hash of every config field + the input data files' size/mtime."""
    fields = {k: v for k, v in dataclasses.asdict(config).items() if k not in _FINGERPRINT_EXCLUDED_FIELDS}
    payload = {
        "config": fields,
        "compounds": _file_signature(config.compounds_path),
        "class_dictionary": _file_signature(config.class_dictionary_path),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


class Checkpointer:
    """Save/load named stage results for one run. When disabled, every
    ``has`` is False and ``save`` does nothing, so callers never branch on
    whether checkpointing is on.
    """

    def __init__(self, config: Config, enabled: bool = True, fresh: bool = False):
        self.enabled = enabled
        self.dir = config.outputs_dir / "checkpoints"
        if not enabled:
            return
        fingerprint = run_fingerprint(config)
        fp_path = self.dir / "fingerprint.txt"
        previous = fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else None
        if fresh and previous is not None:
            logger.info("Starting fresh: discarding previous checkpoints in %s.", self.dir)
            self.clear()
        elif previous is not None and previous != fingerprint:
            logger.info("Settings or data changed since the last interrupted run; discarding its checkpoints.")
            self.clear()
        elif previous is not None:
            logger.info("Resuming from checkpoints in %s (same settings and data as the interrupted run).", self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        fp_path.write_text(fingerprint, encoding="utf-8")
        # Unique per (non-resumed) run and prefixed to every Optuna study
        # name: on Windows the SQLite file can still be locked when clear()
        # runs, and a leftover optuna.db must never be mistaken for this
        # run's studies by a later fresh run with identical settings.
        run_id_path = self.dir / "run_id.txt"
        if not run_id_path.exists():
            run_id_path.write_text(uuid.uuid4().hex[:12], encoding="utf-8")
        self.run_id = run_id_path.read_text(encoding="utf-8").strip()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.joblib"

    def has(self, key: str) -> bool:
        return self.enabled and self._path(key).exists()

    def load(self, key: str) -> Any:
        logger.info("Checkpoint: reusing saved '%s' (skipping recomputation).", key)
        return joblib.load(self._path(key))

    def save(self, key: str, obj: Any) -> None:
        if not self.enabled:
            return
        # Write-then-rename, so a crash mid-save never leaves a truncated
        # file that a later resume would try (and fail) to load.
        tmp = self.dir / f"{key}.joblib.tmp"
        joblib.dump(obj, tmp)
        tmp.replace(self._path(key))

    def optuna_storage(self) -> Optional[str]:
        """SQLite URL for Optuna studies (trial-level resume), or None when disabled."""
        if not self.enabled:
            return None
        return f"sqlite:///{(self.dir / 'optuna.db').as_posix()}"

    def study_name(self, name: str) -> Optional[str]:
        """This run's Optuna study name for ``name``, or None when disabled."""
        return f"{self.run_id}_{name}" if self.enabled else None

    def clear(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
