import logging
import os
import time
import warnings
from dataclasses import dataclass

import av
import draccus
import numpy as np
import pygame
import rerun as rr
import torch
from tensordict import TensorDict

from imikit.datasets.base import ImikitSample, ImikitTransition
from imikit.datasets.h5_dataset import H5DatasetWriter
from imikit.datasets.lerobot import (
    LeRobotDatasetWriter,
    create_canonical_lerobot_features,
)
from imikit.robot_hardware.factr.factr import FACTRGravityCompensation
from imikit.robot_hardware.factr.factr_controller import FactrPDController
from imikit.robot_hardware.panda.panda_env import PandaEnv, PandaEnvConfig
from imikit.robot_hardware.panda.panda_teleop import PandaTeleop

log = logging.getLogger(__name__)


@dataclass
class TeleopRecordingConfig:
    follower: PandaEnvConfig
    lerobot_dataset_id: str
    lerobot_dataset_root: str
    h5_dataset_root: str | None = None
    obs_filter: list[str] | None = None
    lerobot_codec: str = "hevc"


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", module="torch.jit")
    av.logging.set_level(av.logging.VERBOSE)

    # to disable SVT logging see https://github.com/huggingface/lerobot/issues/2162
    # there's no way to disable x265 logging without modifying lerobot

    cfg = draccus.parse(config_class=TeleopRecordingConfig)
    log.info(cfg)

    # rr.init("Franka Panda")
    # rr.spawn(memory_limit="25%")

    follower_env = PandaEnv(cfg.follower)

    factr = FACTRGravityCompensation("config/robot/factr/leader.yaml")
    factr_ctrl = FactrPDController(factr, 
                                   kp=np.array([0.25, 1.0, 0.75, 1.0, 0.5, 0.5, 0.25]), 
                                   kd=np.array([0.025, 0.1, 0.075, 0.1, 0.05, 0.05, 0.025])
    )
    # factr.start(follower_env)

    assert not os.path.exists(cfg.lerobot_dataset_root)
    lerobot_kwargs = dict(
        repo_id=cfg.lerobot_dataset_id,
        fps=follower_env.spec.fps,
        root=cfg.lerobot_dataset_root,
        video_backend="pyav",
    )
    lerobot_features, lerobot_feature_mappings = create_canonical_lerobot_features(
        follower_env.spec,
        task_key="task",
        task_per_step=False,
        obs_keys=cfg.obs_filter,
    )
    lerobot_dataset = LeRobotDatasetWriter(
        lerobot_features,
        lerobot_feature_mappings,
        background_encode=True,
        dataset_kwargs=lerobot_kwargs,
        codec=cfg.lerobot_codec,
        crf=22,
    )

    if cfg.h5_dataset_root is not None:
        assert not os.path.exists(cfg.h5_dataset_root)
        h5_dataset = H5DatasetWriter(
            follower_env.spec,
            cfg.h5_dataset_root,
            filter_obs_keys=cfg.obs_filter,
        )
    else:
        h5_dataset = None

    pygame.init()
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    pygame.display.set_caption("Robot Control")
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

    clock = pygame.time.Clock()

    log.info("Resetting Panda. Please wait...")
    ep_obs = follower_env.reset()
    log.info("Panda Reset done.")

    log.info("Resetting Factr. Please wait...")
    initial_q = follower_env.get_obs()["proprio.joint_pos"].numpy().astype(np.float64)
    print("q", initial_q)
    factr_ctrl.set_position(initial_q)
    factr_ctrl.start(env=follower_env)

    log.info("Robot Control:")
    log.info("  Space:  Start new episode")
    log.info("  Return: Save current episode")
    log.info("  Delete: Drop current episode")
    log.info("  Escape: Finalize dataset and exit")

    try:
        log.info("Ready")

        paused = True
        transitions = []
        prev_obs = None
        while True:
            for event in pygame.event.get():
                if not event.type == pygame.KEYDOWN:
                    continue

                if paused and event.key == pygame.K_SPACE:
                    log.info("Starting new episode")
                    paused = False
                    prev_obs = follower_env.get_obs()

                elif not paused and event.key == pygame.K_RETURN:
                    log.info("Saving current episode")
                    log.info(f"  Transitions: {len(transitions)}")

                    episode = ImikitSample(
                        transition=torch.stack(transitions, dim=0), episode=ep_obs
                    )

                    lerobot_dataset.add_episode(episode)
                    if h5_dataset is not None:
                        h5_dataset.add_episode(episode)

                    paused = True
                    transitions = []
                    prev_obs = None
                    ep_obs = follower_env.reset()
                    log.info("Dataset statistics:")
                    log.info(f"  Episodes: {lerobot_dataset.n_episodes}")
                    log.info(f"  Total Transitions: {lerobot_dataset.n_transitions}")
                    log.info("Ready")

                elif not paused and event.key == pygame.K_DELETE:
                    log.info("Dropping current episode")
                    paused = True
                    transitions = []
                    prev_obs = None
                    ep_obs = follower_env.reset()
                    log.info("Dataset statistics:")
                    log.info(f"  Episodes: {lerobot_dataset.n_episodes}")
                    log.info(f"  Total Transitions: {lerobot_dataset.n_transitions}")
                    log.info("Ready")

                elif event.key == pygame.K_ESCAPE:
                    if len(transitions) > 0:
                        log.info(
                            "You have unsaved episode data. Save or drop it before exiting."
                        )
                        continue

                    return

            if not paused:
                leader_state = factr.get_state()
                if leader_state is None:
                    log.warning("No leader state available (yet)")

                leader_q, leader_qd, leader_gripper = leader_state
                action = TensorDict(
                    {
                        "joint_pos": torch.from_numpy(leader_q).to(dtype=torch.float32),
                        "gripper": torch.from_numpy(leader_gripper).to(
                            dtype=torch.float32
                        ),
                    }
                )

                print()
                print(prev_obs["proprio.joint_pos"])
                print(action["joint_pos"])
                # print(action["gripper"])

                if len(transitions) < 60:
                    delta = action["joint_pos"] - prev_obs["proprio.joint_pos"]
                    delta = torch.clip(delta, min=-2e-2, max=2e-2)
                    action["joint_pos"] = prev_obs["proprio.joint_pos"] + delta

                next_obs = follower_env.step(action)[0]
                # next_obs = follower_env.get_obs()

                # for k, v in next_obs.items():
                #     if k.startswith("images") and len(v.shape) == 3:
                #         rr.log(
                #             f"obs/{k}",
                #             rr.Image(
                #                 v.permute((1, 2, 0)).numpy(),
                #                 color_model="RGB",
                #             ),
                #         )

                transitions.append(
                    ImikitTransition(observation=prev_obs, action=action)
                )
                prev_obs = next_obs

            clock.tick(follower_env.spec.fps)
    finally:
        log.info(
            "Finalizing dataset. This may take a while as it has to wait for file encoding workers."
        )
        lerobot_dataset.finalize()
        if h5_dataset is not None:
            h5_dataset.finalize()
        log.info("Dataset finalized")

        log.info("Shutting down hardware...")
        factr_ctrl.stop()
        factr.shutdown()
        follower_env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
