from data_io.stitch import stitch_npz, make_replanned_output_path


class ReplanController:
    def __init__(self, planner_adapter=None):
        self.planner = planner_adapter
        self.in_progress = False

    def set_planner(self, planner_adapter):
        self.planner = planner_adapter

    def _get_player_grip_seed(self, player):
        cut_idx = int(player.frame_idx)

        if getattr(player.log, "grip_real", None) is not None:
            return float(player.log.grip_real[cut_idx])

        if player.model.nu >= 8:
            finger = float(player.data.ctrl[7])
            return float(max(0.0, min(0.08, 2.0 * finger)))

        return None

    def replan_from_player(self, player, mirror_controller=None):
        if self.in_progress:
            print("[REPLAN] Already in progress.")
            return None

        if self.planner is None:
            print("[REPLAN] No planner adapter configured yet.")
            return None

        self.in_progress = True
        try:
            cut_idx = int(player.frame_idx)
            original_path = str(player.npz_path)
            grip_seed = self._get_player_grip_seed(player)

            reuse_live_robot = False
            robot_adapter = None

            if (
                mirror_controller is not None
                and mirror_controller.is_connected()
                and mirror_controller.current_robot_key() == getattr(self.planner, "robot_key", None)
            ):
                print("[REPLAN] Reusing active mirror robot connection for replanning.")
                robot_adapter = mirror_controller.detach_for_reuse()
                reuse_live_robot = True
            elif mirror_controller is not None and mirror_controller.is_connected():
                print("[REPLAN] Disabling mirror before replanning.")
                mirror_controller.disable()

            suffix_path = self.planner.run_replan(
                player,
                robot_adapter=robot_adapter,
                already_connected=reuse_live_robot,
            )

            out_path = make_replanned_output_path(original_path)
            stitch_npz(original_path, suffix_path, cut_idx, out_path)
            print(f"[REPLAN] New replay written to: {out_path}")

            if reuse_live_robot and mirror_controller is not None and robot_adapter is not None:
                mirror_controller.attach_reused_robot(
                    robot_adapter,
                    grip_width_seed=grip_seed,
                    force_reconnect=True,
                )

            return out_path

        except Exception as e:
            print(f"[REPLAN] Failed: {e}")
            return None
        finally:
            self.in_progress = False