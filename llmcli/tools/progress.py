"""Animated one-line stderr progress bar."""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional


def _unicode_ok() -> bool:
    try:
        "⠋█░".encode(sys.stderr.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class Progress:
    """Animated one-line stderr progress bar.

    Silent unless stderr is a TTY, so piping or redirecting output stays clean.
    A daemon thread redraws on a timer, which keeps the spinner moving while
    the main thread is blocked on a slow request.
    """

    def __init__(self, label: str, total: int, enabled: bool = True):
        self.label = label
        self.total = max(total, 1)
        self.done = 0
        self.enabled = enabled and sys.stderr.isatty()
        self._fancy = _unicode_ok()
        self._frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if self._fancy else "|/-\\"
        self._fill, self._empty = ("█", "░") if self._fancy else ("#", "-")
        self._frame = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = time.monotonic()

    def __enter__(self) -> "Progress":
        self._started = time.monotonic()
        if self.enabled:
            self._draw()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def advance(self, n: int = 1) -> None:
        with self._lock:
            self.done += n

    def _draw(self) -> None:
        with self._lock:
            done = min(self.done, self.total)
        pct = done / self.total
        width = 24
        filled = int(pct * width)
        bar = self._fill * filled + self._empty * (width - filled)
        spin = self._frames[self._frame % len(self._frames)]
        secs = time.monotonic() - self._started
        sys.stderr.write(
            f"\r\033[2K{spin} {self.label} {bar} {pct * 100:3.0f}%  {done}/{self.total}  {secs:4.1f}s"
        )
        sys.stderr.flush()

    def _spin(self) -> None:
        while not self._stop.wait(0.1):
            self._frame += 1
            self._draw()

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self.enabled:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()
        return False
