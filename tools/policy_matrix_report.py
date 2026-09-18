#!/usr/bin/env python3
"""Summarize four policy producer/consumer forensic verdicts and choose the next gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = ("old-old", "current-old", "old-current", "current-current")


def _find_case(root, case, group):
    candidates = []
    for verdict in root.glob("*/verdict.json"):
        name = verdict.parent.name
        if f"_{case}_" not in name:
            continue
        if group and f"_{group}_" not in name:
            continue
        candidates.append(verdict)
    return sorted(candidates)[-1] if candidates else None


def _case_summary(verdict, delivery=None):
    sessions = verdict.get("sessions", {})
    snapshots = [item.get("snapshot_hz") for item in sessions.values()]
    distinct = [item.get("distinct_hz") for item in sessions.values()]
    max_gaps = [item.get("max_distinct_gap_s") for item in sessions.values()]
    delivery_passed = bool(delivery.get("passed")) if delivery else None
    selection_counts = []
    if delivery:
        selection_counts = [
            item.get("enters", 0) for item in delivery.get("sessions", {}).values()
        ]
    return {
        "passed": bool(verdict.get("passed")) and delivery_passed is not False,
        "policy_passed": bool(verdict.get("passed")),
        "delivery_passed": delivery_passed,
        "quest_measured": bool(delivery and delivery.get("quest", {}).get("measured")),
        "selection_visits_min": min(selection_counts) if selection_counts else 0,
        "sessions": len(sessions),
        "snapshot_hz_avg": sum(snapshots) / len(snapshots) if snapshots else None,
        "distinct_hz_avg": sum(distinct) / len(distinct) if distinct else None,
        "max_distinct_gap_s": max(max_gaps) if max_gaps else None,
    }


def diagnose(cases):
    passed = {name: value.get("passed", False) for name, value in cases.items()}
    policy_passed = {
        name: value.get("policy_passed", value.get("passed", False))
        for name, value in cases.items()
    }
    if not policy_passed.get("old-old", False):
        return "historical_or_environment", "The exact historical pair fails; inspect machine/runtime state before editing code."
    if not policy_passed.get("current-old", False) and not policy_passed.get("current-current", False):
        return "producer", "The current policy producer stalls with both consumers; bisect intervene_base."
    if all(passed.get(name, False) for name in CASES):
        if all(cases[name].get("delivery_passed") is True for name in CASES):
            return "validated_baseline", "All combinations pass state and delivery gates; isolate normal-launcher features next."
        return "scale_or_transition", "Policy state passes, but delivery was not validated in every case."
    if not passed.get("old-old", False):
        return "historical_delivery_or_environment", "Historical policy state passes but historical delivery fails; inspect Quest, network, and machine state before editing current code."
    if not passed.get("current-current", False):
        if not passed.get("current-old", False) and passed.get("old-current", False):
            return "producer_delivery", "Delivery fails with the current producer and both consumers; inspect producer-to-runtime state interaction."
        if not passed.get("old-current", False) and passed.get("current-old", False):
            return "consumer", "Current SimPublisher delivery fails with both producers; bisect activation, scheduling, and worker handoff."
        if passed.get("current-old", False) and passed.get("old-current", False):
            return "interaction", "Both mixed pairs pass but current/current fails; trace the cross-version contract."
    return "multiple_boundaries", "More than one pair fails; inspect per-case stalls before choosing a bisect boundary."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.root)
    cases = {}
    missing = []
    paths = {}
    for case in CASES:
        path = _find_case(root, case, args.group)
        if path is None:
            missing.append(case)
            continue
        verdict = json.loads(path.read_text(encoding="utf-8"))
        delivery_path = path.parent / "delivery_verdict.json"
        delivery = (
            json.loads(delivery_path.read_text(encoding="utf-8"))
            if delivery_path.exists() else None
        )
        cases[case] = _case_summary(verdict, delivery)
        paths[case] = str(path)
    boundary, recommendation = diagnose(cases) if not missing else (
        "incomplete", f"Missing verdicts: {', '.join(missing)}"
    )
    result = {
        "complete": not missing, "group": args.group, "missing": missing,
        "boundary": boundary, "recommendation": recommendation,
        "paths": paths, "cases": cases,
    }
    result["quest_measured_all"] = bool(cases) and all(
        item.get("quest_measured", False) for item in cases.values()
    )
    if not result["quest_measured_all"]:
        result["limitations"] = [
            "Quest receive/draw/upload and display cadence were not measured in every case."
        ]
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
