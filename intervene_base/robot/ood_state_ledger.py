"""Cross-process agreement on how many grid sessions are running an OOD scene.

WHY THIS EXISTS
---------------
A multi-window run is N independent `app.py` processes. "Between MIN and MAX of them are
running an out-of-distribution scene" is a property of the WHOLE GRID, but each process
only knows about itself, so something has to be shared.

Deliberately NOT a synchronized reshuffle. Every session decides alone, at its own episode
boundary, and only for itself; the OOD positions migrate around the grid as a side effect.
Nothing is ever reset grid-wide — session identity has to stay spatially stable for the
desktop grid and the VR panels, and simultaneous resets across N sessions are suspected of
having caused point-cloud problems.

THE RULE (fill-to-MAX)
----------------------
    others = live OOD sessions, excluding me
      I am OOD :  stay OOD  iff others <  MIN     # leave only while the floor stays covered
      I am ID  :  become OOD iff others <  MAX    # fill up toward MAX

The asymmetry is what produces rotation. The obvious rule (`others < MIN` for everyone)
pins the count at exactly MIN but never moves it: an incumbent sees MIN-1 others and
re-elects itself forever, so the same two cells stay OOD for the whole run and the other
seven never see an OOD scene. Simulated over 600 episode completions on a 9-grid,
fill-to-MAX keeps the count within [MIN, MAX] at all times and rotates every session
through OOD.

FAILURE MODE
------------
Any error -> log once, report ID. ID is the no-change default; a broken ledger must never
be able to turn every session OOD at once.

The file layout mirrors `robot_ownership_lock.py`: a JSON data file written atomically via
tmp+rename, plus a small lock file used only for the read-decide-write critical section.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple


class OodStateLedger:
    """Shared OOD-count ledger for one launch cohort."""

    def __init__(
        self,
        *,
        session_index: int,
        ledger_key: str,
        min_states: int,
        max_states: int,
        ttl_s: float = 1800.0,
    ):
        safe_key = "".join(
            c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(ledger_key)
        )
        self.session_index = int(session_index)
        self.ledger_key = str(ledger_key)
        self.min_states = int(min_states)
        self.max_states = max(int(max_states), int(min_states))
        self.ttl_s = float(ttl_s)
        self.path = Path("/tmp") / f"iilar_ood_{safe_key}.json"
        self.lock_path = Path("/tmp") / f"iilar_ood_{safe_key}.lock"
        self._pid = os.getpid()
        self._warned = False

    # ------------------------------------------------------------------ enabled

    @property
    def enabled(self) -> bool:
        """Off iff MIN <= 0. There is deliberately no boolean flag: a stale OOD_ENABLED=1
        left over from the retired auto-pause supervisor must not switch on a feature with
        completely different semantics."""
        return self.min_states > 0

    # -------------------------------------------------------------------- utils

    def _warn_once(self, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            print(f"[OOD][WARN] ledger unavailable ({exc}); this session stays IN-distribution.")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _read(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(errors="ignore"))
        except FileNotFoundError:
            return {}
        except Exception:
            # One retry: a reader can lose a race with os.replace on some filesystems.
            try:
                payload = json.loads(self.path.read_text(errors="ignore"))
            except Exception as exc:
                self._warn_once(exc)
                return {}
        sessions = payload.get("sessions")
        return sessions if isinstance(sessions, dict) else {}

    def _write(self, sessions: dict) -> bool:
        """Returns False if the entry could not be published.

        The caller MUST treat a failed write as "I am ID": an OOD session the others
        cannot see does not occupy a slot in their count, so they would fill up to MAX
        around it and the real total would exceed MAX.
        """
        tmp = self.path.with_suffix(f".json.tmp.{self._pid}")
        try:
            tmp.write_text(json.dumps({
                "version": 1,
                "ledger_key": self.ledger_key,
                "sessions": sessions,
            }))
            os.replace(tmp, self.path)     # atomic: readers never see a torn file
            return True
        except Exception as exc:
            self._warn_once(exc)
            try:
                tmp.unlink()
            except Exception:
                pass
            return False

    def _live_entries(self, sessions: dict) -> dict:
        """Drop entries whose process is gone or whose stamp is older than the TTL.

        The pid check is the real liveness test and is exact; the TTL only guards against
        pid reuse. It must stay far longer than an episode, because a session re-stamps
        only at its own episode boundary — a short TTL would make a session in a long
        episode vanish from its neighbours' count and let a third one go OOD.
        """
        now = time.time()
        live = {}
        for key, entry in sessions.items():
            if not isinstance(entry, dict):
                continue
            try:
                pid = int(entry.get("pid", 0))
                stamped = float(entry.get("wall_t", 0.0))
            except (TypeError, ValueError):
                continue
            if not self._pid_alive(pid):
                continue
            if self.ttl_s > 0.0 and (now - stamped) > self.ttl_s:
                continue
            live[str(key)] = entry
        return live

    def _acquire_lock(self, *, budget_s: float = 0.5) -> Optional[int]:
        """Best-effort mutex for the read-decide-write section.

        Returns the fd, or None if it could not be taken in budget. A None result does NOT
        abort the decision — an episode boundary must never block on a lock — it just
        means the decision races, which is exactly what MAX absorbs.
        """
        deadline = time.time() + budget_s
        while time.time() < deadline:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(fd, f"pid={self._pid}\n".encode())
                return fd
            except FileExistsError:
                try:
                    owner = int(self.lock_path.read_text(errors="ignore").split("=", 1)[1])
                except Exception:
                    owner = 0
                if owner and not self._pid_alive(owner):
                    try:
                        self.lock_path.unlink()
                        continue
                    except Exception:
                        pass
                time.sleep(0.02)
            except Exception as exc:
                self._warn_once(exc)
                return None
        return None

    def _release_lock(self, fd: Optional[int]) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # ------------------------------------------------------------------ the API

    @staticmethod
    def decide(*, currently_ood: bool, others_ood: int, min_states: int, max_states: int) -> bool:
        """The fill-to-MAX rule, isolated so it can be tested without any file I/O."""
        if currently_ood:
            return others_ood < min_states
        return others_ood < max_states

    def register_initial(self, ood: bool) -> bool:
        """Record a starting value chosen by the launcher, without consulting the band.

        The launcher pre-assigns the initial set so grid position is not correlated with
        condition. If sessions instead band-filled at startup they would decide in launch
        order (they are staggered for CUDA), pinning OOD to S00/S01 every run.
        """
        if not self.enabled:
            return False
        fd = self._acquire_lock()
        try:
            sessions = self._live_entries(self._read())
            sessions[str(self.session_index)] = self._entry(bool(ood), episode=0)
            if not self._write(sessions):
                return False       # unpublished OOD is worse than no OOD; see _write
        except Exception as exc:
            self._warn_once(exc)
            return False
        finally:
            self._release_lock(fd)
        return bool(ood)

    def decide_for_new_episode(self, *, currently_ood: bool, episode: int,
                               scenario_id: str = "") -> Tuple[bool, int]:
        """Decide this session's OOD-ness for a new episode. Returns (ood, others_ood)."""
        if not self.enabled:
            return False, 0
        fd = self._acquire_lock()
        degraded = fd is None
        try:
            sessions = self._live_entries(self._read())
            others = sum(
                1 for key, entry in sessions.items()
                if str(key) != str(self.session_index) and bool(entry.get("ood"))
            )
            ood = self.decide(
                currently_ood=bool(currently_ood),
                others_ood=others,
                min_states=self.min_states,
                max_states=self.max_states,
            )
            if ood and others >= self.max_states:
                # With the lock held the count is exact, so this should be unreachable.
                # If it fires, the lock path degraded and MAX is doing real work.
                print(f"[OOD][WARN] MAX clamp bound (others={others} >= {self.max_states}); "
                      "the ledger lock likely degraded.")
                ood = False
            sessions[str(self.session_index)] = self._entry(
                ood, episode=episode, scenario_id=scenario_id
            )
            if not self._write(sessions) and ood:
                # Could not publish. An OOD session the others cannot see does not hold a
                # slot in their count, so they would fill to MAX around it. Stay ID.
                print("[OOD][WARN] could not publish this session's OOD state; staying "
                      "IN-distribution so the grid count cannot exceed the maximum.")
                ood = False
        except Exception as exc:
            self._warn_once(exc)
            return False, 0
        finally:
            self._release_lock(fd)

        if degraded:
            print("[OOD][WARN] decided without the ledger lock; count may race.")
        return bool(ood), int(others)

    def _entry(self, ood: bool, *, episode: int, scenario_id: str = "") -> dict:
        return {
            "index": self.session_index,
            "ood": bool(ood),
            "pid": self._pid,
            "wall_t": time.time(),
            "episode": int(episode),
            "scenario_id": str(scenario_id),
        }

    def release(self) -> None:
        """Drop this session's entry so the others stop counting it immediately."""
        if not self.enabled:
            return
        fd = self._acquire_lock(budget_s=0.25)
        try:
            sessions = self._live_entries(self._read())
            sessions.pop(str(self.session_index), None)
            self._write(sessions)
        except Exception:
            pass
        finally:
            self._release_lock(fd)
