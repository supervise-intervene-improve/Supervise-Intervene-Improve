#!/usr/bin/env python3
"""Add current wire-compatibility guards to a detached historical consumer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXIT_OLD = """        resume_after_release = has_robot_control() or state != STATE_REPLAY_RUNNING
        release_robot_control(\"EXIT_SINGLE\", resume_sim=resume_after_release)
"""

EXIT_NEW = """        # Forensic compatibility: policy frame indices do not index the local replay.
        resume_after_release = (
            not policy_state_mirror.enabled
            and (has_robot_control() or state != STATE_REPLAY_RUNNING)
        )
        release_robot_control(\"EXIT_SINGLE\", resume_sim=resume_after_release)
"""

MIRROR_INIT_OLD = """        self.last_episode_id: Optional[str] = None
        self._stale_warned = False
"""

MIRROR_INIT_NEW = """        self.last_episode_id: Optional[str] = None
        self.latest_frame_idx: int = 0
        self._stale_warned = False
"""

MIRROR_APPLY_OLD = """        self.last_applied_seq = seq
        self.last_episode_id = episode_id
        self.latest_acc_risk = float(message.get(\"acc_risk\", 0.0))
"""

MIRROR_APPLY_NEW = """        self.last_applied_seq = seq
        self.last_episode_id = episode_id
        self.latest_frame_idx = int(message.get(\"frame_idx\", self.latest_frame_idx))
        self.latest_acc_risk = float(message.get(\"acc_risk\", 0.0))
"""

MIRROR_LOOP_OLD = """                    if applied is not None:
                        current_idx = int(applied[\"frame_idx\"])
                        if applied[\"first_apply\"] or applied[\"episode_changed\"]:
"""

MIRROR_LOOP_NEW = """                    if applied is not None:
                        # The mirrored frame belongs to the policy episode, not the local
                        # fallback trajectory. Keep current_idx local and bounded.
                        if applied[\"first_apply\"] or applied[\"episode_changed\"]:
"""

STATUS_OLD = """                    sensor_node.publish(
                        \"SimPub/Status/paused\",
                        b\"1\" if paused_now else b\"0\",
                    )
"""

STATUS_NEW = """                    sensor_node.publish(
                        \"SimPub/Status/paused\",
                        b\"1\" if paused_now else b\"0\",
                    )
                    # The current APK gates point clouds on this identity heartbeat.
                    # This is transport compatibility only; the historical render and
                    # point-cloud paths remain unchanged.
                    sensor_node.publish(
                        \"SimPub/Status/session_state\",
                        json.dumps({
                            \"version\": 1,
                            \"session_index\": int(getattr(args, \"session_index\", 0)),
                            \"topic_port\": int(args.topic_port),
                            \"episode_id\": str(policy_state_mirror.last_episode_id or \"\"),
                            \"policy_seq\": policy_state_mirror.last_applied_seq,
                            \"frame_idx\": int(policy_state_mirror.latest_frame_idx),
                            \"mode\": \"policy_mirror\" if policy_state_mirror.enabled else str(state),
                            \"paused\": paused_now,
                            \"intervention_live\": False,
                            \"intervention_phase\": \"policy_resumed\",
                            \"wall_t\": _now_paused_pub,
                        }, separators=(\",\", \":\")).encode(\"utf-8\"),
                    )
"""


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {count}")
    return source.replace(old, new, 1)


def prepare(runtime_path: Path) -> dict:
    source = runtime_path.read_text(encoding="utf-8")
    original_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if "SimPub/Status/session_state" in source:
        raise RuntimeError("historical consumer already publishes session_state")
    source = _replace_once(source, EXIT_OLD, EXIT_NEW, "EXIT_SINGLE")
    source = _replace_once(source, MIRROR_INIT_OLD, MIRROR_INIT_NEW, "mirror frame identity")
    source = _replace_once(source, MIRROR_APPLY_OLD, MIRROR_APPLY_NEW, "mirror frame update")
    source = _replace_once(source, MIRROR_LOOP_OLD, MIRROR_LOOP_NEW, "local replay index isolation")
    source = _replace_once(source, STATUS_OLD, STATUS_NEW, "paused heartbeat")
    runtime_path.write_text(source, encoding="utf-8")
    patched_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return {
        "runtime": str(runtime_path),
        "original_sha256": original_sha,
        "patched_sha256": patched_sha,
        "session_identity_heartbeat": True,
        "policy_mirror_exit_guard": True,
        "policy_frame_index_isolated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = prepare(args.runtime)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
