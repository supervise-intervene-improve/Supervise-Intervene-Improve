import argparse
from pathlib import Path

from multi_app import MultiApp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open multiple MuJoCo replay trajectories, or run ACT policies in the multi viewer."
    )
    parser.add_argument("xml_path", help="Path to the MuJoCo XML model.")
    parser.add_argument(
        "npz_paths",
        nargs="+",
        help="Replay trajectory NPZs. In policy mode these are used as reset NPZs.",
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "policy"),
        default="replay",
        help="Use replay for NPZ playback, or policy to roll out an ACT checkpoint per tile.",
    )
    parser.add_argument(
        "--view",
        choices=("free", "cameras"),
        default="free",
        help="Initial viewer mode. 'free' uses the rollout free camera; 'cameras' shows front/left/right.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="ACT pretrained_model checkpoint directory.")
    parser.add_argument("--policy_hz", type=float, default=10.0)
    parser.add_argument("--action_mode", choices=("queue", "replan"), default="queue")
    parser.add_argument(
        "--arm_action_mode",
        choices=("absolute", "ctrl_delta", "qpos_error"),
        default="absolute",
    )
    parser.add_argument("--gripper_action_mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--arm_delta_clip", type=float, default=None)
    parser.add_argument("--realtime_factor", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=0, help="Policy mode only. 0 means run until stopped.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--robot_key", default="p1")
    return parser.parse_args()


def main():
    args = parse_args()

    xml_path = Path(args.xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    npz_paths = [Path(path) for path in args.npz_paths]
    for path in npz_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    policy_configs = None
    if args.mode == "policy":
        from playback.policy_player import PolicyPlayerConfig

        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode policy")
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        policy_configs = [
            PolicyPlayerConfig(
                checkpoint=args.checkpoint,
                reset_npz=path,
                policy_hz=args.policy_hz,
                action_mode=args.action_mode,
                arm_action_mode=args.arm_action_mode,
                gripper_action_mode=args.gripper_action_mode,
                arm_delta_clip=args.arm_delta_clip,
                realtime_factor=args.realtime_factor,
                max_steps=args.max_steps,
            )
            for path in npz_paths
        ]

    app = MultiApp(
        str(xml_path),
        [str(path) for path in npz_paths],
        width=args.width,
        height=args.height,
        robot_key=args.robot_key,
        player_mode=args.mode,
        policy_configs=policy_configs,
        view_mode=args.view,
    )
    app.run()


if __name__ == "__main__":
    main()
