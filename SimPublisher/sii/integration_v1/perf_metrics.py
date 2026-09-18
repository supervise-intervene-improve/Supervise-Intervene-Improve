"""Streaming performance aggregation for the VR runtime.

`runtime_impl.py` already measures per-frame render/build times and keeps cumulative
per-topic publish counters, but it only ever *prints* them one line at a time. This
module turns those same numbers into windowed statistics
(avg / p50 / p75 / p90 / p95 / p99) and writes one JSONL row per window so a run can be
analysed after the fact instead of grepped.

Schema note: the row key is `latency_ms` for historical reasons, but `observe()` is unit-
agnostic and the row also carries non-latency series. `acc_risk` (the raw, unscaled ACC
score in [0, 1]) lives there so the OOD auto-pause threshold can be calibrated from the
observed p90 instead of guessed — see `RiskPublisherThread._run` in `runtime_impl.py`.

Design constraints:
  * The hot path must stay cheap — `observe()` is an append to a bounded deque plus a
    few float updates. No formatting, no locking on the publish thread's critical path.
  * Nothing here may raise into the main loop. A metrics failure must never take down a
    session, so the file sink swallows and reports write errors once.
  * Wall-clock (`time.time()`) is the shared timebase with the Quest-side collector, so
    the two logs can be joined later. Durations use `time.monotonic()` internally.
"""

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Nearest-rank percentile. `sorted_vals` must be sorted and non-empty."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class StreamStat:
    """Windowed distribution of a scalar (milliseconds, in practice).

    Keeps a bounded sample deque for percentiles plus exact count/mean/min/max over the
    same window. `max_samples` caps memory: at 90 Hz a 4096-sample window covers ~45 s,
    which is far longer than the default 10 s summary window.
    """

    __slots__ = ("name", "_samples", "_sum", "_min", "_max", "_count", "_lifetime_count")

    def __init__(self, name: str, max_samples: int = 4096):
        self.name = name
        self._samples = deque(maxlen=max_samples)
        self._sum = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._count = 0
        self._lifetime_count = 0

    def observe(self, value: float) -> None:
        v = float(value)
        self._samples.append(v)
        self._sum += v
        self._count += 1
        self._lifetime_count += 1
        if v < self._min:
            self._min = v
        if v > self._max:
            self._max = v

    @property
    def count(self) -> int:
        return self._count

    @property
    def lifetime_count(self) -> int:
        return self._lifetime_count

    def snapshot(self) -> Optional[dict]:
        if self._count == 0:
            return None
        ordered = sorted(self._samples)
        return {
            "n": self._count,
            "avg": self._sum / self._count,
            "p50": _percentile(ordered, 50.0),
            # p75/p90 are here for distribution shaping (picking an OOD risk threshold from
            # observed acc_risk), not for latency triage. The list is already sorted, so
            # they cost nothing beyond two extra lookups per window.
            "p75": _percentile(ordered, 75.0),
            "p90": _percentile(ordered, 90.0),
            "p95": _percentile(ordered, 95.0),
            "p99": _percentile(ordered, 99.0),
            "min": self._min,
            "max": self._max,
        }

    def reset_window(self) -> None:
        self._samples.clear()
        self._sum = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._count = 0


class RateMeter:
    """Hz of a cumulative counter, from deltas between successive reads.

    Used for the per-topic publish counters that `IsolatedSensorPublisher` already
    maintains — we read them, we do not add new ones. A counter that goes backwards
    (socket rebuild, publisher restart) re-baselines instead of reporting a negative
    rate.
    """

    __slots__ = ("name", "_last_value", "_last_t", "_window_delta", "_window_t0")

    def __init__(self, name: str):
        self.name = name
        self._last_value: Optional[float] = None
        self._last_t: Optional[float] = None
        self._window_delta = 0.0
        self._window_t0: Optional[float] = None

    def update(self, cumulative_value: float, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        value = float(cumulative_value)
        if self._last_value is None or value < self._last_value:
            # First sample, or the counter was reset underneath us — re-baseline.
            self._last_value = value
            self._last_t = now
            if self._window_t0 is None:
                self._window_t0 = now
            return
        self._window_delta += value - self._last_value
        self._last_value = value
        self._last_t = now
        if self._window_t0 is None:
            self._window_t0 = now

    def snapshot(self, now: Optional[float] = None) -> Optional[dict]:
        if self._window_t0 is None:
            return None
        now = time.monotonic() if now is None else now
        elapsed = max(now - self._window_t0, 1e-9)
        return {"count": self._window_delta, "hz": self._window_delta / elapsed}

    def reset_window(self, now: Optional[float] = None) -> None:
        self._window_delta = 0.0
        self._window_t0 = time.monotonic() if now is None else now


class MetricsCollector:
    """Owns the named stats/rates for one session and emits windowed rows.

    Typical use from the main loop:

        metrics.observe("render_ms", ms)
        ...
        row = metrics.maybe_emit()        # None until the window elapses

    `maybe_emit()` does all the expensive work (sorting, formatting, file IO) and only
    once per `window_s`, so the per-frame cost stays at `observe()`.
    """

    def __init__(
        self,
        *,
        out_path: Optional[str] = None,
        window_s: float = 10.0,
        print_summary: bool = False,
        session_index: Optional[int] = None,
        mode: str = "unknown",
        max_samples: int = 4096,
    ):
        self.window_s = max(float(window_s), 0.5)
        self.print_summary = bool(print_summary)
        self.session_index = session_index
        self.mode = str(mode)
        self.max_samples = int(max_samples)

        self._stats: Dict[str, StreamStat] = {}
        self._rates: Dict[str, RateMeter] = {}
        self._counters: Dict[str, float] = {}
        self._labels: Dict[str, object] = {}
        self._lock = threading.Lock()

        self._window_t0 = time.monotonic()
        self._window_wall_t0 = time.time()
        self._session_t0 = time.monotonic()
        self._window_index = 0

        self._out_path: Optional[Path] = Path(out_path) if out_path else None
        self._fh = None
        self._write_error_logged = False
        # Lifetime accumulators so the shutdown summary covers the whole run rather
        # than only the final (usually partial) window.
        self._lifetime: Dict[str, dict] = {}

        if self._out_path is not None:
            self._open_sink()

    # ------------------------------------------------------------------ sink

    def _open_sink(self) -> None:
        try:
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._out_path, "a", encoding="utf-8", buffering=1)
            print(f"[Metrics] writing rows to {self._out_path}")
        except Exception as exc:  # pragma: no cover - disk/permission dependent
            print(f"[Metrics][WARN] could not open {self._out_path}: {exc}")
            self._fh = None

    def _write_row(self, row: dict) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception as exc:  # pragma: no cover
            if not self._write_error_logged:
                print(f"[Metrics][WARN] row write failed: {exc}")
                self._write_error_logged = True

    # ------------------------------------------------------------- hot path

    def observe(self, name: str, value_ms: float) -> None:
        stat = self._stats.get(name)
        if stat is None:
            stat = StreamStat(name, max_samples=self.max_samples)
            self._stats[name] = stat
        stat.observe(value_ms)

    def observe_many(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            self.observe(name, value)

    def update_rate(self, name: str, cumulative_value: float) -> None:
        meter = self._rates.get(name)
        if meter is None:
            meter = RateMeter(name)
            self._rates[name] = meter
        meter.update(cumulative_value)

    def update_rates(self, values: Dict[str, float], prefix: str = "") -> None:
        for name, value in values.items():
            self.update_rate(f"{prefix}{name}", value)

    def incr(self, name: str, amount: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    def set_label(self, name: str, value) -> None:
        """Attach a scalar/string to every subsequent row (mode, fps, peers, ...)."""
        self._labels[name] = value

    # ------------------------------------------------------------- emission

    def due(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self._window_t0) >= self.window_s

    def maybe_emit(self, now: Optional[float] = None) -> Optional[dict]:
        now = time.monotonic() if now is None else now
        if (now - self._window_t0) < self.window_s:
            return None
        return self.emit(now=now)

    def emit(self, now: Optional[float] = None, final: bool = False) -> Optional[dict]:
        now = time.monotonic() if now is None else now
        with self._lock:
            elapsed = max(now - self._window_t0, 1e-9)
            stats = {}
            for name, stat in self._stats.items():
                snap = stat.snapshot()
                if snap is not None:
                    stats[name] = snap
                    self._accumulate_lifetime(name, snap)
            rates = {}
            for name, meter in self._rates.items():
                snap = meter.snapshot(now=now)
                if snap is not None and snap["count"] > 0:
                    rates[name] = snap

            if not stats and not rates and not self._counters and not final:
                # Nothing happened this window (idle session) — still advance so we do
                # not emit a huge catch-up window later.
                self._reset_window(now)
                return None

            row = {
                "wall_t": time.time(),
                "window_index": self._window_index,
                "window_s": elapsed,
                "session_index": self.session_index,
                "mode": self.mode,
                "uptime_s": now - self._session_t0,
                "final": bool(final),
                "labels": dict(self._labels),
                "latency_ms": stats,
                "rates_hz": rates,
                "counters": dict(self._counters),
            }
            self._window_index += 1
            self._write_row(row)
            if self.print_summary:
                print(self.format_row(row))
            self._reset_window(now)
            return row

    def _accumulate_lifetime(self, name: str, snap: dict) -> None:
        acc = self._lifetime.get(name)
        if acc is None:
            acc = {"n": 0, "sum": 0.0, "min": float("inf"), "max": 0.0}
            self._lifetime[name] = acc
        acc["n"] += snap["n"]
        acc["sum"] += snap["avg"] * snap["n"]
        acc["min"] = min(acc["min"], snap["min"])
        acc["max"] = max(acc["max"], snap["max"])

    def _reset_window(self, now: float) -> None:
        for stat in self._stats.values():
            stat.reset_window()
        for meter in self._rates.values():
            meter.reset_window(now=now)
        self._counters.clear()
        self._window_t0 = now
        self._window_wall_t0 = time.time()

    # ------------------------------------------------------------ rendering

    @staticmethod
    def format_row(row: dict) -> str:
        parts = [
            f"[PerfSummary] window={row['window_index']} "
            f"mode={row.get('mode')} dur={row['window_s']:.1f}s"
        ]
        labels = row.get("labels") or {}
        if labels:
            parts.append(
                "  labels: "
                + " ".join(f"{k}={v}" for k, v in sorted(labels.items()))
            )
        latency = row.get("latency_ms") or {}
        for name in sorted(latency):
            s = latency[name]
            parts.append(
                f"  {name:<16} n={s['n']:<6d} avg={s['avg']:7.2f} p50={s['p50']:7.2f} "
                f"p95={s['p95']:7.2f} p99={s['p99']:7.2f} max={s['max']:7.2f} ms"
            )
        rates = row.get("rates_hz") or {}
        if rates:
            rendered = " ".join(
                f"{name}={rates[name]['hz']:.1f}" for name in sorted(rates)
            )
            parts.append(f"  rate_hz: {rendered}")
        counters = row.get("counters") or {}
        if counters:
            rendered = " ".join(
                f"{k}={int(v) if float(v).is_integer() else v}"
                for k, v in sorted(counters.items())
            )
            parts.append(f"  counters: {rendered}")
        return "\n".join(parts)

    def format_lifetime_summary(self) -> str:
        if not self._lifetime:
            return "[PerfSummary][final] no samples recorded"
        lines = [
            f"[PerfSummary][final] session={self.session_index} mode={self.mode} "
            f"uptime={time.monotonic() - self._session_t0:.1f}s"
        ]
        for name in sorted(self._lifetime):
            acc = self._lifetime[name]
            if acc["n"] <= 0:
                continue
            lines.append(
                f"  {name:<16} n={acc['n']:<8d} avg={acc['sum'] / acc['n']:7.2f} "
                f"min={acc['min']:7.2f} max={acc['max']:7.2f} ms"
            )
        return "\n".join(lines)

    def close(self) -> None:
        try:
            self.emit(final=True)
        except Exception as exc:  # pragma: no cover
            print(f"[Metrics][WARN] final emit failed: {exc}")
        try:
            print(self.format_lifetime_summary())
        except Exception:
            pass
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None


def default_metrics_path(
    log_dir: str = "session_logs",
    session_index: Optional[int] = None,
    stamp: Optional[str] = None,
) -> str:
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    idx = "XX" if session_index is None else f"{int(session_index):02d}"
    return os.path.join(log_dir, f"metrics_S{idx}_{stamp}.jsonl")


def iter_rows(path) -> Iterable[dict]:
    """Read back a metrics JSONL, skipping malformed lines (partial final write)."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
