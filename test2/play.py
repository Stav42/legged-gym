from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
import isaacgym
import torch
from torch import Tensor
from collections import deque
import statistics

import os
from helpers import  get_args
from task_registry import task_registry

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

import numpy as np
import torch


class OnPolicyRunner:

    def __init__(self,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = 'cpu'
        self.num_envs=1
        self.num_privileged_obs=None
        self.num_obs=48
        num_critic_obs=48
        self.num_actions=12
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.num_obs,
                                                        num_critic_obs,
                                                        self.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.num_envs, self.num_steps_per_env, [self.num_obs], [self.num_privileged_obs], [self.num_actions])

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path,  map_location=torch.device('cpu'))
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference



def make_alg_runner(name=None, args=None, log_root="default") -> OnPolicyRunner:
    # if no args passed get command line arguments
    if args is None:
        args = get_args()

    log_dir = "./"

    train_cfg_dict =  {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, \
    'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True, 'value_loss_coef': 1.0},\
     'init_member_classes': {}, 'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0},\
      'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'Flat_svan_m1', 'load_run': -1, 'max_iterations': 1500, 'num_steps_per_env': 24, \
      'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 'runner_class_name': 'OnPolicyRunner', 'seed': 1}

    runner = OnPolicyRunner(train_cfg_dict, log_dir, device=args.rl_device)
    resume = True
    if resume:
        resume_path = './model_0.pt'
        runner.load(resume_path)
    return runner

def play(args):
    ppo_runner = make_alg_runner(name=args.task, args=args)
    policy = ppo_runner.get_inference_policy(device='cpu')
    obs = torch.ones((48, 1))
    obs = torch.t(obs)
    actions = policy(obs)
    print(actions)


if __name__ == '__main__':
    args = get_args()
    play(args)