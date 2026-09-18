"""Cross-process exclusive ownership of a physical robot arm.

WHY THIS EXISTS
---------------
A multi-window run starts N independent `app.py` policy processes (9-15 of them), and
EVERY one of them builds its own robot adapter for the same `INTERVENE_ROBOT_KEY`. Nothing
stopped two of them from calling `robot_adapter.connect()` on the same arm: the second
connect would switch control mode underneath the first, so an operator mid-takeover could
have the arm yanked into HYBRID_JOINT_IMPEDANCE by a different window. There is one
physical arm; ownership has to be exclusive.

The VR runtime already solved exactly this with `RobotOwnershipLock` in
`SimPublisher/sii/integration_v1/runtime_impl.py`. This is a deliberate re-implementation
rather than an import, because `intervene_base` and `SimPublisher` are separate packages
with separate interpreters (polymetis py3.10 vs the VR .venv py3.13) and importing across
them would drag the whole VR runtime into the policy process.

CRITICAL: the lock file path and payload format MUST stay byte-compatible with that class
(`/tmp/iilar_robot_<key>.lock`, `pid=<int>` on the first line). The VR runtime and the
policy processes have to contend for the *same* file, or the lock arbitrates nothing.
Changing either side alone silently reintroduces the double-connect bug.

YIELD REQUESTS
--------------
`acquire()` is strictly non-blocking, which is correct while a human is holding the arm but
wrong while the holder is only driving it back to its initial pose: an operator who wanted
to intervene on another session was refused outright with `robot_busy` and had to wait for
a movement they did not care about.

The yield protocol adds ONE sidecar file, `/tmp/iilar_robot_<key>.yield`, and never touches
the lock file — so the VR runtime, which knows nothing about yielding, keeps working
unchanged (it simply never yields, and the requester times out exactly as it does today).

The holder decides. A requester only ASKS; it never takes the lock away. The holder honours
the request only while it is returning home — never while a human has the arm — stops
commanding, and releases the lock through its normal path. That ordering is what guarantees
two processes never command the arm at once.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


class RobotOwnershipLock:
    """Exclusive lock for one robot key, shared with the VR runtime's implementation."""

    def __init__(self, robot_key: str):
        safe_key = "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in str(robot_key)
        )
        self.robot_key = str(robot_key)
        # Must match runtime_impl.RobotOwnershipLock exactly — see module docstring.
        self.path = Path("/tmp") / f"iilar_robot_{safe_key}.lock"
        # Sidecar. Deliberately NOT the lock file: that one has a compatibility contract.
        self.yield_path = Path("/tmp") / f"iilar_robot_{safe_key}.yield"
        self._fd: Optional[int] = None
        self._pid = os.getpid()
        self._yield_warned = False

    @property
    def held(self) -> bool:
        return self._fd is not None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Is `pid` a process that can still be holding the arm?

        `os.kill(pid, 0)` is NOT sufficient on its own: a ZOMBIE (state Z) has already
        exited but keeps its process-table entry until the parent reaps it, so the
        signal succeeds and the PID looks alive forever. The launcher force-kills the
        nine policy processes on shutdown (`still running after 10s; force-killing`),
        which is exactly how a zombie is produced -- and the lock it left behind then
        made EVERY later intervention fail with `robot_busy`
        ("Intervention unavailable - robot in use") while no robot was in use at all.
        Observed on p3: `/tmp/iilar_robot_p3.lock pid=1427125` -> `[python] <defunct>`.
        """
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Exists but owned by another user — still alive, so do not steal the lock.
            return True
        except OSError:
            return False
        return not RobotOwnershipLock._pid_is_zombie(pid)

    @staticmethod
    def _pid_is_zombie(pid: int) -> bool:
        """True when /proc says the process has exited and is awaiting reaping.

        Linux-only by construction; anything unreadable is reported as NOT a zombie so
        the lock errs toward leaving a live owner alone.
        """
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="ignore") as fh:
                stat = fh.read()
        except (OSError, ValueError):
            return False
        # Field 3 is the state. comm (field 2) is parenthesised and may itself contain
        # spaces or ')', so split after the LAST ')' rather than on whitespace.
        tail = stat.rpartition(")")[2].split()
        return bool(tail) and tail[0] == "Z"

    def _read_owner_pid(self) -> Optional[int]:
        try:
            text = self.path.read_text(errors="ignore")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        for line in text.splitlines():
            if line.startswith("pid="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
        return None

    def acquire(self) -> Tuple[bool, str]:
        """Try to take ownership. Returns (acquired, human-readable message)."""
        if self._fd is not None:
            return True, f"already owns {self.robot_key}"
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner_pid = self._read_owner_pid()
                # A crashed owner must not lock the arm out forever, but only reclaim when
                # the pid is provably gone — never on a timeout, which would race a live
                # takeover.
                if owner_pid is not None and not self._pid_alive(owner_pid):
                    try:
                        self.path.unlink()
                        print(f"[RobotLock] Removed stale lock {self.path} from pid={owner_pid}.")
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        return False, f"stale lock unlink failed: {exc}"
                    continue
                return False, (
                    f"held by pid={owner_pid if owner_pid is not None else 'unknown'} "
                    f"({self.path})"
                )
            except Exception as exc:
                return False, f"could not create lock {self.path}: {exc}"

            payload = (
                f"pid={self._pid}\n"
                f"robot_key={self.robot_key}\n"
                f"created_unix={time.time():.3f}\n"
                f"argv={' '.join(sys.argv)}\n"
            )
            try:
                os.write(fd, payload.encode("utf-8", errors="replace"))
            except Exception:
                pass
            self._fd = fd
            return True, f"acquired {self.path}"
        return False, f"could not acquire {self.path}"

    # ---------------------------------------------------------------- yielding

    def _yield_warn_once(self, exc: Exception) -> None:
        """Yielding is an optimisation; a broken sidecar must never break the lock."""
        if not self._yield_warned:
            self._yield_warned = True
            print(f"[RobotLock][WARN] yield sidecar unavailable ({exc}); "
                  "falling back to non-blocking acquire.")

    def read_yield_request(self) -> Optional[dict]:
        try:
            return json.loads(self.yield_path.read_text(errors="ignore"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            self._yield_warn_once(exc)
            return None

    def request_yield(self, owner_pid: Optional[int], reason: str = "intervene") -> None:
        """Ask the current holder to stop early. Best effort; never raises."""
        payload = {
            "requester_pid": self._pid,
            "owner_pid": int(owner_pid) if owner_pid else 0,
            "requested_unix": time.time(),
            "reason": str(reason),
        }
        tmp = self.yield_path.with_suffix(f".yield.tmp.{self._pid}")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.yield_path)   # atomic: readers never see a torn file
        except Exception as exc:
            self._yield_warn_once(exc)
            try:
                tmp.unlink()
            except Exception:
                pass

    def clear_yield_request(self, *, only_mine: bool = True) -> None:
        try:
            if only_mine:
                req = self.read_yield_request()
                if req is not None and int(req.get("requester_pid", 0)) != self._pid:
                    return
            self.yield_path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            self._yield_warn_once(exc)

    def yield_requested_of_me(self, *, ttl_s: float = 10.0) -> Optional[dict]:
        """Return a live request addressed to THIS process, else None.

        Three guards, all necessary: `owner_pid` stops us honouring a request aimed at a
        previous holder; the TTL stops a file left behind by a crashed requester from
        cancelling every future return-home; the liveness check stops a requester that
        gave up from still being obeyed.
        """
        req = self.read_yield_request()
        if not req:
            return None
        try:
            if int(req.get("owner_pid", 0)) != self._pid:
                return None
            if ttl_s > 0.0 and (time.time() - float(req.get("requested_unix", 0.0))) > ttl_s:
                return None
            if not self._pid_alive(int(req.get("requester_pid", 0))):
                return None
        except (TypeError, ValueError):
            return None
        return req

    def acquire_with_yield(
        self, *, wait_s: float = 2.5, poll_s: float = 0.1, reason: str = "intervene"
    ) -> Tuple[bool, str]:
        """`acquire()`, but ask a busy holder to yield and retry for up to `wait_s`.

        Falls through to exactly today's refusal if the holder does not yield — a holder
        with a human on the arm is *supposed* to refuse.
        """
        acquired, msg = self.acquire()
        if acquired:
            self.clear_yield_request(only_mine=True)
            return True, msg
        if wait_s <= 0.0:
            return False, msg

        owner_pid = self._read_owner_pid()
        self.request_yield(owner_pid, reason=reason)
        print(f"[RobotLock] Arm held by pid={owner_pid}; requested yield, waiting up to {wait_s:.1f}s.")

        deadline = time.time() + float(wait_s)
        while time.time() < deadline:
            time.sleep(max(0.01, float(poll_s)))
            acquired, msg = self.acquire()
            if acquired:
                waited = wait_s - (deadline - time.time())
                self.clear_yield_request(only_mine=True)
                return True, f"{msg} (after {waited:.2f}s yield wait)"

        # Do not leave the request behind: a later, unrelated return-home by that same
        # holder would otherwise be cancelled by a requester that has already given up.
        self.clear_yield_request(only_mine=True)
        return False, msg

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        try:
            owner_pid = self._read_owner_pid()
            # Only unlink if we still own it; a reclaimed-stale lock may now belong to
            # someone else and deleting it would hand the arm to a third process.
            if owner_pid in (None, self._pid):
                self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[RobotLock][WARN] Could not release {self.path}: {exc}")
