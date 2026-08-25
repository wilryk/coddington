"""A tiny in-process job registry, so long traces can report progress.

A full day is dozens of timesteps and a field is hundreds of heliostats, so
a day sweep is minutes of work. That is far too long for an HTTP request to
sit open with nothing to show, and a browser that shows nothing for four
minutes is indistinguishable from one that has hung. So the work runs on a
background thread, the caller gets a job id immediately, and progress is
polled.

Deliberately in-process and in-memory: this app is a local tool serving one
person, and jobs are worth exactly as much as the browser tab that started
them. Nothing here is a queue, a scheduler, or a database, and it should not
grow into one — if a run is worth keeping, it belongs in a stored run
(:mod:`heliostat.store`) written by the CLI.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

#: Finished jobs are kept so a slow browser can still collect its result,
#: but not forever: each carries a whole day of per-step numbers.
MAX_FINISHED_JOBS = 8


@dataclass
class Job:
    """One background run and everything the poller needs to know about it."""

    id: str
    total: int
    label: str = ""
    done: int = 0
    state: str = "running"  # running | done | error | cancelled
    detail: str = ""
    error: str | None = None
    result: Any = None
    #: Binary payloads too large or too non-JSON to belong in ``result`` --
    #: e.g. a day sweep's per-timestep flux PNGs -- keyed however the caller
    #: likes. Lives and dies with the job exactly like ``result`` does:
    #: nothing here persists past eviction.
    blobs: dict[str, bytes] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    #: Optional cost-weighted alternative to ``done``/``total`` steps, for a
    #: job whose steps cost wildly different amounts of wall time (a field
    #: trace's outer-ring heliostats can cost several times an inner-ring
    #: one -- see ``heliostat.web.app._heliostat_progress_weights``). Left at
    #: 0 (the default), ``eta_s`` and ``snapshot``'s ``frac`` fall back to
    #: plain step counting, so a job that never sets these behaves exactly
    #: as before. A caller that does track weighted progress still reports
    #: ``done``/``total`` as the true step count -- only the fraction/ETA
    #: derived from these gets cost-weighted.
    weight_done: float = 0.0
    weight_total: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return end - self.started_at

    @property
    def _progress_frac(self) -> float | None:
        """Fraction of work done, cost-weighted where a weight total was
        supplied, else a plain step count -- shared by ``eta_s`` and
        ``snapshot`` so the bar and the estimate never disagree."""
        if self.total <= 0:
            return None
        if self.weight_total > 0:
            return self.weight_done / self.weight_total
        if self.done <= 0:
            return None
        return self.done / self.total

    @property
    def eta_s(self) -> float | None:
        """Seconds remaining, extrapolated from the cost-weighted fraction
        of work done so far when one is available, else from the plain
        step count.

        ``None`` until a step has finished, because an estimate from zero
        samples is a guess dressed as information.
        """
        if self.state != "running" or self.done <= 0 or self.done >= self.total:
            return None
        frac = self._progress_frac
        if not frac:
            return None
        return (self.elapsed_s / frac) * (1.0 - frac)

    def snapshot(self) -> dict:
        frac = self._progress_frac
        return {
            "job_id": self.id,
            "state": self.state,
            "label": self.label,
            "done": self.done,
            "total": self.total,
            "detail": self.detail,
            "frac": None if frac is None else round(frac, 4),
            "elapsed_s": round(self.elapsed_s, 2),
            "eta_s": None if self.eta_s is None else round(self.eta_s, 1),
            "error": self.error,
        }

    def cancelled(self) -> bool:
        return self._cancel.is_set()


class JobRegistry:
    """Start jobs, look them up, and forget the oldest finished ones."""

    def __init__(self, max_finished: int = MAX_FINISHED_JOBS):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_finished = max_finished

    def start(self, total: int, work: Callable[[Job], Any], label: str = "") -> Job:
        """Run ``work(job)`` on a background thread.

        ``work`` reports progress by setting ``job.done`` and ``job.detail``,
        and should check :meth:`Job.cancelled` between steps so a cancel is
        acted on rather than merely recorded.
        """
        job = Job(id=uuid.uuid4().hex[:12], total=int(total), label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()

        def runner() -> None:
            try:
                job.result = work(job)
                # A cancel during the last step still means cancelled: the
                # result is partial, and calling it "done" would hand back a
                # short day as if it were a whole one.
                job.state = "cancelled" if job.cancelled() else "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.finished_at = time.monotonic()

        threading.Thread(target=runner, daemon=True, name=f"heliostat-job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.state != "running":
            return False
        job._cancel.set()
        return True

    def _evict_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.state != "running"]
        if len(finished) <= self._max_finished:
            return
        finished.sort(key=lambda j: j.finished_at or 0.0)
        for job in finished[: len(finished) - self._max_finished]:
            self._jobs.pop(job.id, None)
