# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymapi
from isaacgym import gymtorch

from pynput import keyboard

from pynput import keyboard

import numpy as np
import torch

import matplotlib.pyplot as plt


plt.ion()
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 16))
axes = [ax1, ax2, ax3, ax4, ax5, ax6]
# line, = ax.plot([], [], 'r-')  # Red line plot
lines_lin_vel = [ax1.plot([], [], label=f'Base Linear Velocity {i+1}')[0] for i in range(3)]  # Initialize 5 line plots
lines_ang_vel = [ax2.plot([], [], label=f'Base Angular Velocity {i+1}')[0] for i in range(3)]  # Initialize 5 line plots
lines_command = [ax3.plot([], [], label=f'Command {i+1}')[0] for i in range(3)]  # Initialize 5 line plots
lines_joint1 = [ax4.plot([], [], label=f'Joint Position {i+1}')[0] for i in range(6)]  # Initialize 5 line plots
lines_joint2 = [ax5.plot([], [], label=f'Joint Position {6+i+1}')[0] for i in range(6)]  # Initialize 5 line plots
lines_target_position = [ax6.plot([], [], label=f'Target Joint Position {i+1}')[0] for i in range(6)]  # Initialize 5 line plots
lines = [lines_lin_vel, lines_ang_vel, lines_command, lines_joint1, lines_joint2, lines_target_position]

def on_press(key):
    print("Key Pressed")
    update_command(key)

for ax in axes:
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 1.5)

command = [0, 0, 0]
pause = False

def update_command(key):
    global command
    global pause
    try:
        if key == keyboard.Key.up:
            command[0] += 0.2
        elif key == keyboard.Key.down:
            command[0] -= 0.2
        elif key == keyboard.Key.right:
            command[1] += 0.1
        elif key == keyboard.Key.left:
            command[1] -= 0.1
        elif key.char == 'y':
            command[2] += 0.1
        elif key.char == 't':
            command[2] -= 0.1
        elif key.char == 'z':
            pause = not pause

    except AttributeError:
        pass

def on_press(key):
    print("Key Pressed")
    update_command(key)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    print("initial Observation 1: ", obs)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    print("initial Observation 2: ", obs)

    policy = ppo_runner.get_inference_policy(device=env.device)
    print("initial Observation 3: ", obs)
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    print("initial Observation 4: ", obs)
    plot = False
    global pause

    obs_ind = [[0, 1, 2], [3, 4, 5], [9, 10, 11], [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23], [36, 37, 38, 39, 40, 41]]

    for i in range(10*int(env.max_episode_length)):

        if pause:
            while True:
                # print("On Break")
                if pause == False:
                    break
        
        default_joint_angles =  [0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000, -1.5000, -0.1000,  1.0000, -1.5000]
        # default_joint_angles =  [0.4000,  0.4000, 0.40000, 0.40000,  0.40000, 0.40000,  0.40000,  0.40000, 0.4000, 0.40000,  0.40000, 0.4000]


        actions = policy(obs.detach())
        if i>0:
            actions = actions
            # print("Model Activated")
        else:
            actions = 0*actions
        # for i in range(12):
        #     actions[0][i] = -5 * default_joint_angles[i]
        # _dof_states = env.gym.acquire_dof_state_tensor(env.sim)
        # dof_states = gymtorch.wrap_tensor(_dof_states)
        # actions[0] = 4*torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # actions = 0 * actions
        print("actions: ", actions)

        # if env.gym.get_sim_time(env.sim)>0:
        #     for j in range(0, 12):
        #         dof_states[j][0] = default_joint_angles[j]+0.5*np.sin(env.gym.get_sim_time(env.sim)/2)
                # dof_state_list[j][0] = dof_state_list[j][0]+0.0*np.sin(env.gym.get_sim_time(env.sim)/2)
                # env.gym.set_dof_target_position(env.envs[0], j, dof_state_list[j][0])

        # env_ids = torch.tensor([0]).to(dtype=torch.int32)
        # env.gym.set_dof_state_tensor_indexed(env.sim,
        #                                       gymtorch.unwrap_tensor(dof_states),
        #                                       gymtorch.unwrap_tensor(env_ids), 1)

        obs, _, rews, dones, infos = env.step(actions.detach())

        # for i in range(1):
            # env.gym.set_rigid_linear_velocity(env.envs[0], env.gym.get_rigid_handle(env.envs[0], "m2", env.gym.get_actor_rigid_body_names(env.envs[0], 0)[i]), gymapi.Vec3(0.0, 0, 0.5))
            # env.gym.set_rigid_angular_velocity(env.envs[0], env.gym.get_rigid_handle(env.envs[0], "m2", env.gym.get_actor_rigid_body_names(env.envs[0], 0)[i]), gymapi.Vec3(0.1, 0, 0))
        # dof_state_list = env.gym.get_actor_dof_states(env.envs[0], 0, 1)

        # print(dof_state_list[0][0])
        

            # env.gym.set_dof_target_positions(env.envs[0], 0, dof_state_list ,1)
        # print("DOF StateList New: ", env.gym.get_actor_dof_states(env.envs[0], 0, 1))
        # _dof_states = env.gym.acquire_dof_state_tensor(env.sim)
        # dof_states = gymtorch.wrap_tensor(_dof_states)
        # print("DOF State New: ", dof_states)
        
        obs[0, 9] = command[0]
        obs[0, 10] = command[1]
        obs[0, 11] = command[2]
        action_init =  [0.1000,  0.8000, -1.5000, -0.1000,  0.8000, -1.5000,  0.1000,  1.0000, -1.5000, -0.1000,  1.0000, -1.5000]
        action_list = [float(act) for itr, act in enumerate(actions.detach()[0, :])]
        obs_list = [float(act) for act in obs.detach()[0, :]]
        print("Observation: ", obs)
        if plot:
            for index, line in enumerate(lines):
                for ind in range(len(obs_ind[index])):
                    x_data = np.append(line[ind].get_xdata(), i)
                    x_data_clipped = x_data[-100:]
                    if index == 5:
                        y_data = np.append(line[ind].get_ydata(), action_list[ind])
                    else:
                        y_data = np.append(line[ind].get_ydata(), obs_list[obs_ind[index][ind]])
                    y_data_clipped = y_data[-100:]

                    line[ind].set_xdata(x_data_clipped)
                    line[ind].set_ydata(y_data_clipped)

                axes[index].set_xlim(np.min(x_data_clipped), np.max(x_data_clipped))
                axes[index].relim()
                axes[index].autoscale_view()

            plt.pause(0.01)
            plt.show()


        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    args = get_args()
    play(args)
    plt.ioff()
