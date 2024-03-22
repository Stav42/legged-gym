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
# contributors may be used to endorse or promote products derived FRom
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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class M2FlatCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]


        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class commands( LeggedRobotCfg.commands ):
        stance_int = 400
        stance_dur = 225
        stance_env_num_den = 192
        pacing_offset = False
    
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'plane'
        # measure_heights = False
        mesh_type = "trimesh"
        measure_heights = False
        svan_terrain = True
        svan_curriculum = True
        svan_dyn_random = False
        # curriculum = True
        terrain_length = 45
        restitution = 0
        # max_init_terrain_level = 0
        static_friction = 20
        dynamic_friction = 1
        max_terrain_level = 12
        visualize_force = False
        terrain_width = 45
        num_rows = 1 # number of terrain rows (levels)
        num_cols = 4 # number of terrain cols (types)

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # stiffness = {'FR_hip_joint': 10, 'FR_thigh_joint': 5, 'FR_calf_joint': 2, 'FL_hip_joint': 10, 'FL_thigh_joint': 5, 'FL_calf_joint': 2,
        #             'RR_hip_joint': 10, 'RR_thigh_joint': 5, 'RR_calf_joint': 2, 'RL_hip_joint': 10, 'RL_thigh_joint': 5, 'RL_calf_joint': 2,
        # }  # [N*m/rad]
        # damping = {'FR_hip_joint': 8.94, 'FR_thigh_joint': 6.32, 'FR_calf_joint': 2.82, 'FL_hip_joint': 8.94, 'FL_thigh_joint': 6.32, 'FL_calf_joint': 2.82,
        #             'RR_hip_joint': 8.94, 'RR_thigh_joint': 6.32, 'RR_calf_joint': 2.82, 'RL_hip_joint': 8.94, 'RL_thigh_joint': 6.32, 'RL_calf_joint': 2.82,
        # }     # [N*m*s/rad]

        #60 and 0.5
        stiffness = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':100,   # [rad]
            'RL_hip_joint':100,  # [rad]
            'FR_hip_joint':100,  # [rad]
            'RR_hip_joint':100,   # [rad]

            'FL_thigh_joint':45,     # [rad]
            'RL_thigh_joint':45,   # [rad]
            'FR_thigh_joint':45,     # [rad]
            'RR_thigh_joint':45,  # [rad]

            'FL_calf_joint':45,   # [rad]
            'RL_calf_joint':45,    # [rad]
            'FR_calf_joint':45,  # [rad]
            'RR_calf_joint':45,    # [rad]
        }

        damping = { # = target angles [rad] when action = 0.0
            'FL_hip_joint':0.5,   # [rad]
            'RL_hip_joint':0.5,  # [rad]
            'FR_hip_joint':0.5,  # [rad]
            'RR_hip_joint':0.5,   # [rad]

            'FL_thigh_joint':0.5,     # [rad]
            'RL_thigh_joint':0.5,   # [rad]
            'FR_thigh_joint':0.5,     # [rad]
            'RR_thigh_joint':0.5,  # [rad]

            'FL_calf_joint':0.5,   # [rad]
            'RL_calf_joint':0.5,    # [rad]
            'FR_calf_joint':0.5,  # [rad]
            'RR_calf_joint':0.5,    # [rad]
        }
        # stiffness = {'joint': 50}
        # damping = {'joint': 0.5} 
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.20
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/m2/urdf/SVANM2_URDF_inertia_change.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        mcp_running = False
        name = "m2"
        foot_name = "foot"
        fix_base_link = False
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip"]
        # terminate_after_contacts_on = []
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand( LeggedRobotCfg.domain_rand ):
        push_robots = True
        randomize_base_mass = True
        randomize_friction = True
        added_mass_range = [-0.5, 0.5]
        lag_timesteps = 6
        randomize_lag_timesteps = False
        randomize_rigids_after_start = False
        randomize_friction_indep = False
        randomize_friction = False
        friction_range = [0.9, 3.0]
        randomize_restitution = False
        restitution_range = [0.0, 0.2]
        randomize_base_mass = False
        # added_mass_range = [-1.0, 3.0]
        randomize_gravity = False
        gravity_range = [0.8, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        randomize_ground_friction = False
        ground_friction_range = [15.0, 20.0]
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = False
        motor_offset_range = [-0.02, 0.02]
        push_robots = True
        randomize_Kp_factor = False
        randomize_Kd_factor = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.3
        only_positive_rewards = True
        gait_force_sigma = 100.
        gait_vel_sigma = 10
        # base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
        sigma_rew_neg = 0.02
        kappa_gait_probs = 0.07
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0005
            # dof_pos_limits = -10.0
            termination = -10.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 2.0
            lin_vel_z = -0.05
            ang_vel_xy = -0.05
            orientation = -0.05
            dof_vel = -0.0005
            dof_acc = -2.5e-7
            base_height = -0.5
            # feet_air_time =  1.00
            collision = -0.
            dof_vel_limits = -1.0
            dof_pos_limits = -1.0
            torque_limits = -1.0
            feet_stumble = -0.0 
            action_rate = -0.0
            # stand_still = -0.5
            stance_selective = -1.5
            tracking_contacts_shaped_force = 10.0
            tracking_contacts_shaped_vel = 10.0
            feet_clearance_cmd_linear = -11.0

        penalty_level = {
            'action_rate_selective': 0,
            'orientation_selective': 0,
            'dof_vel_selective': 0,
            'dof_acc_selective': 0,
            'lin_vel_z_selective': 0, 
            'ang_vel_xy_selective': 0, 
        }

        stance_penalty = {'stance_selective': 0}

class M2FlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'Flat_m2'
        max_iterations = 2500


## Checking ssh cases
  