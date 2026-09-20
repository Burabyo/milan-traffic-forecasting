"""Memory measurement utilities.

Task 1 of the assignment asks for *evidence* of memory usage before and after
optimisation. Two complementary numbers are recorded:

* ``deep_memory_mb`` - the size of the pandas object itself
  (``DataFrame.memory_usage(deep=True)``), which isolates the effect of dtype
  and schema choices.
* ``peak_rss_mb`` - the peak resident set size of the whole process during the
  operation, sampled with ``psutil``. This captures transient allocations that
  the object-level number hides (parsing buffers, intermediate copies), which
  is what actually decides whether the job fits in RAM.

Both are reported because either alone is misleading.
"""
from __future__ import annotations

import gc
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import pandas as pd
import psutil

_MB = 1024 ** 2


def deep_memory_mb(obj: pd.DataFrame | pd.Series) -> float:
    """Size of a pandas object in MiB, including Python-object contents."""
    usage = obj.memory_usage(deep=True)
    total = float(usage.sum()) if hasattr(usage, "sum") else float(usage)
    return total / _MB


@dataclass
class MemoryReport:
    label: str
    peak_rss_mb: float = 0.0
    baseline_rss_mb: float = 0.0
    delta_rss_mb: float = 0.0
    wall_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("extra"))
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"[{self.label}] peak RSS {self.peak_rss_mb:8.1f} MiB | "
            f"delta {self.delta_rss_mb:8.1f} MiB | {self.wall_seconds:6.2f} s"
        )


class _RSSSampler(threading.Thread):
    """Poll process RSS in the background so transient peaks are not missed."""

    def __init__(self, interval: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0.0
        self._stop_evt = threading.Event()
        self._proc = psutil.Process()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                rss = self._proc.memory_info().rss / _MB
            except psutil.Error:  # pragma: no cover
                break
            self.peak = max(self.peak, rss)
            time.sleep(self.interval)

    def stop(self) -> float:
        self._stop_evt.set()
        self.join(timeout=1.0)
        return self.peak


@contextmanager
def measure(label: str, report_list: list | None = None):
    """Measure peak RSS and wall time of the enclosed block.

    >>> with measure("naive load") as rep:
    ...     df = pd.read_csv(path)
    >>> print(rep)
    """
    gc.collect()
    proc = psutil.Process()
    baseline = proc.memory_info().rss / _MB
    sampler = _RSSSampler()
    sampler.peak = baseline
    sampler.start()
    report = MemoryReport(label=label, baseline_rss_mb=baseline)
    t0 = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_seconds = time.perf_counter() - t0
        report.peak_rss_mb = sampler.stop()
        report.delta_rss_mb = report.peak_rss_mb - baseline
        if report_list is not None:
            report_list.append(report)


def reports_to_frame(reports: list[MemoryReport]) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in reports])
