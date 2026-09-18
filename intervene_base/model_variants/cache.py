"""Bounded model-variant cache with a background compiler.

Compiling a cups variant costs ~1.3 s cold / ~0.56 s warm and ~404 MB resident, so a
variant must never be built on a hot path. This cache gives callers three access levels:

    get(desc)           non-blocking. `None` on a miss. Safe in a render loop.
    prefetch(desc)      enqueue a background build. Never blocks, never raises.
    get_or_build(desc)  blocking. Only ever called at an episode boundary.

Structure deliberately mirrors `PCWorkerThread` in `SimPublisher/sii/integration_v1/
runtime_impl.py`: one daemon worker, a latest-wins pending dict keyed by variant key, a
semaphore counting un-grabbed jobs. Reviewing one teaches you the other. One worker only --
`spec.compile()` is CPU-bound and holds the GIL for long stretches, so a second would just
contend.

WHY NO MJB DISK CACHE: `mj_saveModel` writes 217 MB per variant and loads in 89 ms, but the
file is MuJoCo-version-specific (the policy runs 3.4.0, the VR runtime 3.3.7) and 50
variants would be 10.8 GB. Since the speculative precompiler moves the build off the
critical path anyway, a disk cache would only make BACKGROUND work faster. Not worth the
failure modes.

The base model is pinned: it is the fallback every consumer degrades to, so evicting it
would turn a cache-capacity problem into a correctness problem.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from model_variants.builder import VariantIncompatible, build_model
from model_variants.descriptor import BASE_KEY, variant_key


class VariantCache:
    def __init__(self, xml_path, reference_model=None, *, capacity: int = 4,
                 on_evict: Optional[Callable[[str, object], None]] = None,
                 label: str = "", enable_worker: bool = True):
        self.xml_path = str(xml_path)
        self.reference_model = reference_model
        self.capacity = max(1, int(capacity))
        self.label = label or "VariantCache"
        self._on_evict = on_evict

        self._models: Dict[str, object] = {}
        self._used_at: Dict[str, float] = {}
        self._descriptors: Dict[str, Optional[dict]] = {}
        # Keys a caller currently renders from. Never evicted, whatever the LRU says.
        self._retained: Dict[str, int] = {}
        self._lock = threading.RLock()

        # Keys that failed to build, with the reason. Cached so a broken descriptor is not
        # retried once per episode forever; the reason is what gets stamped on the episode.
        self.failures: Dict[str, str] = {}

        self._pending: Dict[str, Optional[dict]] = {}
        self._pending_lock = threading.Lock()
        self._work_sem = threading.Semaphore(0)
        # Signalled whenever a build finishes, so get_or_build can wait on an in-flight one
        # instead of compiling the same variant a second time.
        self._built = threading.Condition(threading.Lock())
        self._inflight: set = set()

        self.stats_counters = {
            "hit": 0, "miss": 0, "build": 0, "build_fail": 0,
            "evict": 0, "prefetch": 0, "wait_inflight": 0,
        }
        self.build_ms_total = 0.0
        self.build_ms_last = 0.0

        self._stop = threading.Event()
        self._thread = None
        if enable_worker:
            self._thread = threading.Thread(
                target=self._run, daemon=True, name=f"VariantBuild-{label or 'x'}"
            )
            self._thread.start()

    # ------------------------------------------------------------------ base model

    def install_base(self, model) -> None:
        """Register the already-compiled base model. Pinned; never evicted or rebuilt."""
        with self._lock:
            self._models[BASE_KEY] = model
            self._descriptors[BASE_KEY] = None
            self._used_at[BASE_KEY] = time.time()
            self._retained[BASE_KEY] = self._retained.get(BASE_KEY, 0) + 1
            if self.reference_model is None:
                self.reference_model = model

    # ------------------------------------------------------------------- accessors

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._models

    def get(self, descriptor_or_key):
        """Non-blocking lookup. `None` on a miss. Accepts a descriptor or a key."""
        key = descriptor_or_key if isinstance(descriptor_or_key, str) else variant_key(descriptor_or_key)
        with self._lock:
            model = self._models.get(key)
            if model is not None:
                self._used_at[key] = time.time()
                self.stats_counters["hit"] += 1
                return model
            self.stats_counters["miss"] += 1
            return None

    def failure_reason(self, descriptor_or_key) -> str:
        key = descriptor_or_key if isinstance(descriptor_or_key, str) else variant_key(descriptor_or_key)
        with self._lock:
            return self.failures.get(key, "")

    def retain(self, key: str) -> None:
        with self._lock:
            self._retained[key] = self._retained.get(key, 0) + 1

    def release(self, key: str) -> None:
        with self._lock:
            n = self._retained.get(key, 0) - 1
            if n <= 0:
                self._retained.pop(key, None)
            else:
                self._retained[key] = n

    def prefetch(self, descriptor: Optional[dict]) -> None:
        """Queue a background build. Idempotent, never blocks, never raises."""
        if descriptor is None:
            return
        key = variant_key(descriptor)
        with self._lock:
            if key in self._models or key in self.failures:
                return
        with self._pending_lock:
            if key in self._pending:
                return
            self._pending[key] = descriptor
        self.stats_counters["prefetch"] += 1
        self._work_sem.release()

    def get_or_build(self, descriptor: Optional[dict]):
        """Blocking. Returns the model, or raises `VariantIncompatible`.

        Only ever called at an episode boundary. If the worker is already building this
        exact variant, wait for it rather than compiling a second copy.
        """
        if descriptor is None:
            model = self.get(BASE_KEY)
            if model is not None:
                return model
        key = variant_key(descriptor)

        model = self.get(key)
        if model is not None:
            return model
        with self._lock:
            reason = self.failures.get(key)
        if reason:
            raise VariantIncompatible(reason)

        with self._built:
            if key in self._inflight:
                self.stats_counters["wait_inflight"] += 1
                while key in self._inflight and not self._stop.is_set():
                    self._built.wait(timeout=30.0)
                model = self.get(key)
                if model is not None:
                    return model
                with self._lock:
                    reason = self.failures.get(key)
                if reason:
                    raise VariantIncompatible(reason)

        return self._build_now(key, descriptor)

    # -------------------------------------------------------------------- internals

    def _build_now(self, key: str, descriptor: Optional[dict]):
        with self._built:
            if key in self._inflight:
                # Another caller won the race between the check above and here.
                while key in self._inflight and not self._stop.is_set():
                    self._built.wait(timeout=30.0)
                model = self.get(key)
                if model is not None:
                    return model
            self._inflight.add(key)
        try:
            t0 = time.perf_counter()
            model = build_model(self.xml_path, descriptor, reference_model=self.reference_model)
            dt_ms = (time.perf_counter() - t0) * 1e3
            with self._lock:
                self._models[key] = model
                self._descriptors[key] = descriptor
                self._used_at[key] = time.time()
                self.stats_counters["build"] += 1
                self.build_ms_last = dt_ms
                self.build_ms_total += dt_ms
                # Protect the entry we just built: it is the one the caller asked for, and
                # at capacity it would otherwise be the newest-but-only evictable entry and
                # get dropped immediately -- handing back a model the cache no longer knows
                # about, and firing on_evict for a slot that is about to be used.
                self._evict_locked(protect=key)
            print(f"[{self.label}] built variant {key} in {dt_ms:.0f} ms "
                  f"(cached={len(self._models)}/{self.capacity})")
            return model
        except VariantIncompatible as exc:
            with self._lock:
                self.failures[key] = str(exc)
                self.stats_counters["build_fail"] += 1
            print(f"[{self.label}][ERROR] variant {key} refused: {exc}")
            raise
        except Exception as exc:                          # never let a build kill a session
            with self._lock:
                self.failures[key] = f"unexpected:{exc}"
                self.stats_counters["build_fail"] += 1
            print(f"[{self.label}][ERROR] variant {key} build raised: {exc}")
            raise VariantIncompatible(f"unexpected:{exc}") from exc
        finally:
            with self._built:
                self._inflight.discard(key)
                self._built.notify_all()

    def _evict_locked(self, protect: Optional[str] = None) -> None:
        """LRU, skipping the base, anything a caller is rendering from, and `protect`.

        `capacity` COUNTS the pinned base slot, which is why the launcher sizes it
        `OOD_MAX_STATES + 1`.

        If everything is protected we deliberately do NOT evict: exceeding the cap by one
        model costs 404 MB, while thrashing an `MjrContext` rebuild every frame would cost
        the render loop. The caller degrades instead (see the grid's badged fallback).
        """
        while len(self._models) > self.capacity:
            candidates = [
                (t, k) for k, t in self._used_at.items()
                if k != BASE_KEY and k != protect and k not in self._retained and k in self._models
            ]
            if not candidates:
                return
            _, key = min(candidates)
            model = self._models.pop(key, None)
            self._used_at.pop(key, None)
            self._descriptors.pop(key, None)
            self.stats_counters["evict"] += 1
            if self._on_evict is not None:
                try:
                    self._on_evict(key, model)
                except Exception as exc:
                    print(f"[{self.label}][WARN] on_evict({key}) raised: {exc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._work_sem.acquire(timeout=0.5):
                continue
            if self._stop.is_set():
                return
            with self._pending_lock:
                if not self._pending:
                    continue
                key, descriptor = self._pending.popitem()
            with self._lock:
                if key in self._models or key in self.failures:
                    continue
            try:
                self._build_now(key, descriptor)
            except VariantIncompatible:
                pass                                       # already recorded and logged

    def stats(self) -> dict:
        with self._lock:
            return {
                **self.stats_counters,
                "cached": len(self._models),
                "capacity": self.capacity,
                "retained": len(self._retained),
                "failures": len(self.failures),
                "build_ms_last": round(self.build_ms_last, 1),
                "build_ms_total": round(self.build_ms_total, 1),
            }

    def close(self) -> None:
        self._stop.set()
        self._work_sem.release()
        with self._built:
            self._built.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
