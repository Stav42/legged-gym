import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
# from rsl_rl.runners import OnPolicyRunner
from on_policy_runner import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, parse_sim_params

class TaskRegistry():
    def __init__(self):
        self.train_cfgs = {}

    def make_alg_runner(self, name=None, args=None, log_root="default") -> OnPolicyRunner:
        # if no args passed get command line arguments
        if args is None:
            args = get_args()

        log_dir = "./"
        

        train_cfg_dict =  {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 'init_member_classes': {}, 'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'Flat_svan_m1', 'load_run': -1, 'max_iterations': 1500, 'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 'runner_class_name': 'OnPolicyRunner', 'seed': 1}
    
        runner = OnPolicyRunner(train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = True
        if resume:
            resume_path = './model_0.pt'
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner

# make global task registry
task_registry = TaskRegistry()