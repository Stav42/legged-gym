from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.svan_m1_flat.svan_m1_flat_config import SvanM1FlatCfg, SvanM1FlatCfgPPO
from legged_gym.envs.base.legged_robot import LeggedRobot
import os

import isaacgym
from legged_gym.utils import  get_args, export_policy_as_jit, Logger
from task_registry import task_registry


import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    print("Debugger 1")

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    print("Debugger 2")
    # load policy

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print("Debugger 3")

    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    obs = torch.ones((48, 1))
    obs = torch.t(obs)
    print("Observations: ", obs)
    actions = policy(obs)
    print(actions)
    print("Debugger 4")


if __name__ == '__main__':

    task_registry.register("svan_m1_flat", LeggedRobot, SvanM1FlatCfg(), SvanM1FlatCfgPPO() )

    args = get_args()
    print("\n\n\n Args are: ", args)
    play(args)