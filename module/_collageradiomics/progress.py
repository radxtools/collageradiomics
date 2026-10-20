"""Throttled logging and callbacks for the Python, CLI, and Slicer frontends."""

import logging
import time
from collections.abc import Callable

ProgressCallback = Callable[[str, int, int], None]


class ProgressReporter:
    def __init__(self, callback: ProgressCallback | None = None) -> None:
        if callback is not None and not callable(callback):
            raise TypeError("progress_callback must be callable")
        self.callback = callback
        self.stage = ""
        self.last_percent = -10
        self.last_update = 0.0
        self.logger = logging.getLogger("collageradiomics")

    def __call__(self, stage: str, completed: int, total: int) -> None:
        changed = stage != self.stage or completed == 0
        if changed:
            self.stage, self.last_percent = stage, -10
        percent = int(100 * completed / total)
        if changed or percent >= self.last_percent + 10 or completed == total:
            self.logger.info("%s: %s/%s (%s%%)", stage, completed, total, percent)
            self.last_percent = percent
        now = time.monotonic()
        if self.callback and (changed or completed == total or now - self.last_update >= 0.05):
            self.callback(stage, completed, total)
            self.last_update = now
