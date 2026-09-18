import argparse
from pathlib import Path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Replay a MuJoCo trajectory, or run an ACT policy in the same intervention viewer."
    )
    parser.add_argument("xml_path", help="Path to the MuJoCo XML model.")
    parser.add_argument(
        "npz_path",
        help="Path to the replay trajectory NPZ. In policy mode this is used as the reset NPZ.",
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "policy"),
        default="replay",
        help="Use replay for NPZ playback, or policy to roll out an ACT checkpoint.",
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
    parser.add_argument(
        "--dump_camera_images",
        type=parse_bool,
        default=False,
        help="Save top_rgb.png/top_depth.png each frame for debugging.",
    )
    parser.add_argument(
        "--cmd_bind",
        default="127.0.0.1",
        help="Bind address for optional policy command receiver.",
    )
    parser.add_argument(
        "--cmd_port",
        type=int,
        default=0,
        help="If >0, bind a ZMQ PULL socket for Quest/VR-forwarded policy commands.",
    )
    parser.add_argument(
        "--state_bind",
        default="127.0.0.1",
        help="Bind address for optional live policy-state publisher.",
    )
    parser.add_argument(
        "--state_port",
        type=int,
        default=0,
        help="If >0, bind a ZMQ PUB socket for live MuJoCo state mirroring.",
    )
    parser.add_argument(
        "--state_hz",
        type=float,
        default=30.0,
        help="Maximum live state publish rate when --state_port is enabled.",
    )
    args = parser.parse_args()

    xml_path = args.xml_path
    npz_path = args.npz_path

    if not Path(xml_path).exists():
        raise FileNotFoundError(xml_path)
    if not Path(npz_path).exists():
        raise FileNotFoundError(npz_path)
    if args.mode == "policy":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --mode policy")
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)

    from app import App
    from playback.policy_player import PolicyPlayerConfig

    policy_config = None
    if args.mode == "policy":
        policy_config = PolicyPlayerConfig(
            checkpoint=args.checkpoint,
            reset_npz=Path(npz_path),
            policy_hz=args.policy_hz,
            action_mode=args.action_mode,
            arm_action_mode=args.arm_action_mode,
            gripper_action_mode=args.gripper_action_mode,
            arm_delta_clip=args.arm_delta_clip,
            realtime_factor=args.realtime_factor,
            max_steps=args.max_steps,
        )

    app = App(
        xml_path,
        npz_path,
        view_mode=args.view,
        player_mode=args.mode,
        policy_config=policy_config,
        dump_camera_images=args.dump_camera_images,
        cmd_bind=args.cmd_bind,
        cmd_port=args.cmd_port,
        state_bind=args.state_bind,
        state_port=args.state_port,
        state_hz=args.state_hz,
    )
    app.run()


if __name__ == "__main__":
    main()
