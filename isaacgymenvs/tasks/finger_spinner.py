# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSSGPS/GPRS/GSM Shield, SIM808, Arduino Development Boards OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import *


class FingerSpinnerMujoco(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500

        self.cfg["env"]["numObservations"] = 6 + 9 + 1  # dof + rb pos + rb dist
        self.cfg["env"]["numActions"] = 2

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render
        )

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.full_state = self.finger_num_dof + self.spinner_num_dof
        self.actions_tensor = torch.zeros(self.num_envs * self.full_state, device=self.device, dtype=torch.float)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # Assuming self.num_dof is the sum of all each dof for all robots
        # Each FS in this case however has 2 dofs hence the 4
        self.dof_pos = self.dof_state.view(self.num_envs, self.full_state, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.full_state, 2)[..., 1]
        self.dof_vel[:] = 0
        self.dof_pos[:, 0] = torch.rand((self.num_envs), device=self.device)
        self.dof_pos[:, 1] = torch.rand((self.num_envs), device=self.device)
        self.dof_pos[:, 2] = 1.3 * torch.ones((self.num_envs), device=self.device)

        env_ids_int32 = torch.arange(0, self.num_envs, device=self.device).to(dtype=torch.int32)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.time = 1

        self.proximal_id, self.distal_id, self.spinner_id = 1, 2, 4

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        finger_asset_file = "mjcf/finger.xml"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            finger_asset_file = self.cfg["env"]["asset"].get("assetFileName", finger_asset_file)

        asset_path = os.path.join(asset_root, finger_asset_file)
        asset_root = os.path.dirname(asset_path)
        finger_asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        finger_asset = self.gym.load_asset(self.sim, asset_root, finger_asset_file, asset_options)
        self.finger_num_dof = self.gym.get_asset_dof_count(finger_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(finger_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        spinner_asset_file = "mjcf/spinner.xml"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            spinner_asset_file = self.cfg["env"]["asset"].get("assetFileName", spinner_asset_file)

        asset_path = os.path.join(asset_root, spinner_asset_file)
        asset_root = os.path.dirname(asset_path)
        spinner_asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        spinner_asset = self.gym.load_asset(self.sim, asset_root, spinner_asset_file, asset_options)
        self.spinner_num_dof = self.gym.get_asset_dof_count(spinner_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.finger_handle = []
        self.spinner_handle = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            finger_handle = self.gym.create_actor(env_ptr, finger_asset, pose, "finger_isaac", i, -1, 0)
            spinner_handle = self.gym.create_actor(env_ptr, spinner_asset, pose, "spinner_isaac", i, -1, 0)

            # Actuation level
            finger_dof_prop = self.gym.get_actor_dof_properties(env_ptr, finger_handle)
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            # Joint damping and stiffness
            finger_dof_prop['stiffness'][:] = 0.0
            finger_dof_prop['damping'][:] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, finger_handle, finger_dof_prop)
            self.envs.append(env_ptr)

            self.spinner_handle.append(spinner_handle)
            self.envs.append(env_ptr)
            self.finger_handle.append(finger_handle)

    def compute_reward(self):
        # retrieve environment observations from buffer

        finger_1_theta = self.obs_buf[:, 0]
        finger_1_dtheta = self.obs_buf[:, 1]
        finger_2_theta = self.obs_buf[:, 2]
        finger_2_dtheta = self.obs_buf[:, 3]
        spinner_theta = self.obs_buf[:, 4]
        spinner_dtheta = self.obs_buf[:, 5]
        action_1 = self.actions_tensor[:, 0]
        action_2 = self.actions_tensor[:, 1]
        distal_spinner_dist = self.obs_buf[:, 15]

        self.rew_buf[:], self.reset_buf[:] = conpute_finger_spinner_reward(
            finger_1_theta, finger_2_theta, spinner_theta, finger_1_dtheta,
            finger_2_dtheta, spinner_dtheta, action_1, action_2, distal_spinner_dist,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()
        self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()
        self.obs_buf[env_ids, 4] = self.dof_pos[env_ids, 2].squeeze()
        self.obs_buf[env_ids, 5] = self.dof_vel[env_ids, 2].squeeze()
        self.obs_buf[env_ids, 6:9] = self.rigid_body_state[env_ids, self.proximal_id, 0:3].squeeze() #proximal pos
        self.obs_buf[env_ids, 9:12] = self.rigid_body_state[env_ids, self.distal_id, 0:3].squeeze() #distal pos
        self.obs_buf[env_ids, 12:15] = self.rigid_body_state[env_ids, self.spinner_id, 0:3].squeeze() #spinner pos

        self.obs_buf[env_ids, 15] = torch.cdist(
            self.rigid_body_state[env_ids, self.spinner_id, 0:3].view(len(env_ids), -1, 3),
            self.rigid_body_state[env_ids, self.distal_id, 0:3].view(len(env_ids), -1, 3)
        ).squeeze()

        return self.obs_buf

    def reset_idx(self, env_ids):

        self.dof_pos[env_ids, 0] = torch.rand(len(env_ids), device=self.device)
        self.dof_pos[env_ids, 1] = torch.rand(len(env_ids), device=self.device)
        self.dof_pos[env_ids, 2] = 1.3 * torch.ones(len(env_ids), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        full_state = self.finger_num_dof + self.spinner_num_dof
        actions_tensor = torch.zeros(self.num_envs * full_state, device=self.device, dtype=torch.float)
        actions_tensor[0::full_state] = actions[:, 0].to(self.device).squeeze() * self.max_push_effort * self.joint_gears[0]
        actions_tensor[1::full_state] = actions[:, 1].to(self.device).squeeze() * self.max_push_effort * self.joint_gears[1]
        self.actions_tensor = actions.clone().to(self.device)
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def conpute_finger_spinner_reward(finger_1_theta, finger_2_theta, spinner_theta, finger_1_dtheta, finger_2_dtheta,
                                  spinner_d_theta, action_1, action_2, distal_spinner_dist, reset_buf, progress_buf, max_episode_length):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving

    reward_pos = -1 * spinner_theta ** 2 * 1000
    reward_dist = -1 * distal_spinner_dist ** 2 * 100 * -(reward_pos) * 0.01
    reward_vel = -1 * spinner_d_theta ** 2 * 10
    reg = -1 * (action_1 ** 2 + action_2 ** 2) * 50
    # print(f"dist **2 {distal_spinner_dist ** 2}")
    # print(f"spinner_pos {spinner_theta}, dist {distal_spinner_dist}\n")
    # print(f"R_dist {reward_dist}, R_pos {reward_pos}, R_vel {reward_vel}, R_reg {reg}\n")
    reward = reg + reward_pos + reward_dist + reward_vel
    reset_buf = torch.where(torch.abs(spinner_theta[:]) < 0.01, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset_buf
