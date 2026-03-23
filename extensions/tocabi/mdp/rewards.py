# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# * --- Regularization Rewards --- * #
def _idx(asset, name: str) -> int:
    """Strict joint index lookup with a helpful error."""
    try:
        return asset.data.joint_names.index(name)
    except ValueError as e:
        # 에러 메시지에 현재 joint 목록 일부를 보여주면 디버깅이 훨씬 쉬움
        jn = asset.data.joint_names
        preview = jn[:50]
        raise ValueError(
            f"Joint '{name}' not found in asset.data.joint_names. "
            f"len={len(jn)} preview(first50)={preview}"
        ) from e

def _slice_action_by_joint_names(env: ManagerBasedRLEnv, action_name: str, joint_names_expr: list[str] | None):
    action = env.action_manager.action[action_name]
    if not joint_names_expr:
        return action
    term = env.action_manager.get_term(action_name)
    joint_names = getattr(term, "_joint_names", None) if term is not None else None
    if joint_names is None:
        return action
    idx, _ = string_utils.resolve_matching_names(joint_names_expr, list(joint_names), preserve_order=True)
    return action[:, idx]

def survival_reward(env: ManagerBasedRLEnv):
    return torch.ones(env.num_envs, device=env.device)

def action_smoothness1(env: ManagerBasedRLEnv, action_name: str, joint_names_expr: list[str] | None = None):
    action = _slice_action_by_joint_names(env, action_name, joint_names_expr)
    prev_action = env.action_manager.prev_action[action_name]
    if joint_names_expr:
        term = env.action_manager.get_term(action_name)
        joint_names = getattr(term, "_joint_names", None) if term is not None else None
        if joint_names is not None:
            idx, _ = string_utils.resolve_matching_names(joint_names_expr, list(joint_names), preserve_order=True)
            prev_action = prev_action[:, idx]
    # clamp actions to avoid explosion from stale/invalid values
    action = torch.clamp(action, -1.0, 1.0)
    prev_action = torch.clamp(prev_action, -1.0, 1.0)
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - prev_action)/dt2
    return -torch.sum(error, dim=1)

def action_smoothness2(env: ManagerBasedRLEnv, action_name: str, joint_names_expr: list[str] | None = None):
    action = _slice_action_by_joint_names(env, action_name, joint_names_expr)
    prev_action = env.action_manager.prev_action[action_name]
    prev_prev_action = env.action_manager.prev_prev_action[action_name]
    if joint_names_expr:
        term = env.action_manager.get_term(action_name)
        joint_names = getattr(term, "_joint_names", None) if term is not None else None
        if joint_names is not None:
            idx, _ = string_utils.resolve_matching_names(joint_names_expr, list(joint_names), preserve_order=True)
            prev_action = prev_action[:, idx]
            prev_prev_action = prev_prev_action[:, idx]
    # clamp actions to avoid explosion from stale/invalid values
    action = torch.clamp(action, -1.0, 1.0)
    prev_action = torch.clamp(prev_action, -1.0, 1.0)
    prev_prev_action = torch.clamp(prev_prev_action, -1.0, 1.0)
    dt2 = (env.step_dt*env.cfg.decimation)**2
    error = torch.square(action - 2*prev_action + prev_prev_action)/dt2
    return -torch.sum(error, dim=1)

def joint_position_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return -torch.sum(torch.square(error), dim=1) * _negsqrd_exp(torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1), scale=1.)

def joint_torque_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str | None = None):
    asset: Articulation = env.scene[asset_cfg.name]
    if action_name is not None:
        term = env.action_manager._terms.get(action_name)
        if term is not None and hasattr(term, "_last_pd_torque_full"):
            pd_torque = term._last_pd_torque_full
            return -torch.sum(torch.square(pd_torque[:, asset_cfg.joint_ids]), dim=1)
    return -torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.joint_vel[:,asset_cfg.joint_ids]), dim=1)

def base_lin_vel_z_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.square(asset.data.root_lin_vel_b[:, 2])

def base_ang_vel_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def joint_pos_limits_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    margin: float = 0.10,
    outside_weight: float = 1.0,
    exempt_joint_names: list[str] | None = None,
):
    """Penalize near-limit (inside) and out-of-limit joint positions.

    - Inside barrier region: quadratic penalty when distance-to-limit < margin.
    - Outside limits: linear excess penalty (weighted by outside_weight).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    q_min = asset.data.soft_joint_pos_limits[:, :, 0]
    q_max = asset.data.soft_joint_pos_limits[:, :, 1]

    dist_lower = q - q_min
    dist_upper = q_max - q

    # outside-limit excess (same spirit as previous implementation)
    out_lower = torch.relu(-dist_lower)
    out_upper = torch.relu(-dist_upper)
    outside = out_lower + out_upper

    # CBF-like inside barrier near limits
    if margin > 0.0:
        in_lower = torch.relu(margin - dist_lower) * (dist_lower >= 0.0).to(q.dtype)
        in_upper = torch.relu(margin - dist_upper) * (dist_upper >= 0.0).to(q.dtype)
        inside = torch.square(in_lower / margin) + torch.square(in_upper / margin)
    else:
        inside = torch.zeros_like(q)

    penalty = inside + outside_weight * outside

    # Optional per-joint exemption (e.g., joints whose nominal pose is near limits).
    if exempt_joint_names:
        ex_ids = []
        for name in exempt_joint_names:
            try:
                ex_ids.append(asset.data.joint_names.index(name))
            except ValueError:
                continue
        if ex_ids:
            penalty[:, ex_ids] = 0.0

    return -torch.sum(penalty[:, asset_cfg.joint_ids], dim=1)

def joint_torque_limits_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    # Penalize joint torques too close to the limit
    out_of_limits = (torch.abs(asset.data.applied_torque) - asset.data.soft_joint_torque_limits).clip(min=0.)
    return -torch.sum(out_of_limits[:,asset_cfg.joint_ids], dim=1)

def joint_regularization_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    error = 0.0

    # --- yaw joints regularization around default ---
    right_hip_yaw_idx = _idx(asset, "R_HipYaw_Joint")
    left_hip_yaw_idx  = _idx(asset, "L_HipYaw_Joint")
    error += _negsqrd_exp(
        asset.data.joint_pos[:, right_hip_yaw_idx] - asset.data.default_joint_pos[:, right_hip_yaw_idx]
    )
    error += _negsqrd_exp(
        asset.data.joint_pos[:, left_hip_yaw_idx] - asset.data.default_joint_pos[:, left_hip_yaw_idx]
    )

    # --- hip ab/ad (roll) regularization around default ---
    right_hip_abad_idx = _idx(asset, "R_HipRoll_Joint")
    left_hip_abad_idx  = _idx(asset, "L_HipRoll_Joint")
    error += _negsqrd_exp(
        asset.data.joint_pos[:, right_hip_abad_idx] - asset.data.default_joint_pos[:, right_hip_abad_idx]
    )
    error += _negsqrd_exp(
        asset.data.joint_pos[:, left_hip_abad_idx] - asset.data.default_joint_pos[:, left_hip_abad_idx]
    )

    return error / 4.0

def hip_yaw_negative_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    l_idx = _idx(asset, "L_HipYaw_Joint")
    r_idx = _idx(asset, "R_HipYaw_Joint")
    diff_l = torch.abs(asset.data.joint_pos[:, l_idx])
    diff_r = torch.abs(asset.data.joint_pos[:, r_idx])
    # penalize negative (L-R < 0)
    return -torch.relu(diff_l) - torch.relu(diff_r)


def hip_yaw_sum_zero_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    l_idx = _idx(asset, "L_HipYaw_Joint")
    r_idx = _idx(asset, "R_HipYaw_Joint")
    s = asset.data.joint_pos[:, l_idx] + asset.data.joint_pos[:, r_idx]
    # reward when sum is near zero
    return _negsqrd_exp(s)


def hip_roll_threshold_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_min: float = -0.06,
    right_max: float = 0.06,
):
    # Discontinuous penalty with threshold crossing jump:
    # each violating side contributes (1 + relu(distance)).
    asset: Articulation = env.scene[asset_cfg.name]
    l_idx = _idx(asset, "L_HipRoll_Joint")
    r_idx = _idx(asset, "R_HipRoll_Joint")

    l_q = asset.data.joint_pos[:, l_idx]
    r_q = asset.data.joint_pos[:, r_idx]

    l_relu = torch.relu(left_min - l_q)   # active when l_q < left_min
    r_relu = torch.relu(r_q - right_max)  # active when r_q > right_max

    l_term = (l_relu > 0.0).float() * (1.0 + l_relu)
    r_term = (r_relu > 0.0).float() * (1.0 + r_relu)

    return -(l_term + r_term)


def foot_yaw_frame_y_distance_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float
):
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for foot distance penalty.")
    foot_ids = env.body_feet_ids
    foot_pos_w = asset.data.body_pos_w[:, foot_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    rel_pos_w = foot_pos_w - root_pos_w
    root_quat = asset.data.root_quat_w
    quat_yaw = math_utils.yaw_quat(root_quat)
    num_feet = len(foot_ids)
    quat_yaw_rep = quat_yaw.repeat_interleave(num_feet, dim=0)
    rel_pos_yaw = math_utils.quat_rotate_inverse(
        quat_yaw_rep, rel_pos_w.reshape(-1, 3)
    ).reshape(rel_pos_w.shape)
    y_dist = torch.abs(rel_pos_yaw[:, 0, 1] - rel_pos_yaw[:, 1, 1])
    return -torch.relu(threshold - y_dist)


def arm_joint_regularization_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    error = 0.0

    # tocabi shoulder axis order:
    # yaw (base) -> pitch (base) -> roll (arm axis)
    r_sh_yaw   = _idx(asset, "R_Shoulder1_Joint")
    l_sh_yaw   = _idx(asset, "L_Shoulder1_Joint")
    r_sh_pitch = _idx(asset, "R_Shoulder2_Joint")
    l_sh_pitch = _idx(asset, "L_Shoulder2_Joint")
    r_sh_roll  = _idx(asset, "R_Shoulder3_Joint")
    l_sh_roll  = _idx(asset, "L_Shoulder3_Joint")

    # 어깨 3축은 default 기준으로 regularization (0 기준보다 안정적)
    error += _negsqrd_exp(asset.data.joint_pos[:, r_sh_yaw]   - asset.data.default_joint_pos[:, r_sh_yaw])
    error += _negsqrd_exp(asset.data.joint_pos[:, l_sh_yaw]   - asset.data.default_joint_pos[:, l_sh_yaw])

    error += _negsqrd_exp(asset.data.joint_pos[:, r_sh_pitch] - asset.data.default_joint_pos[:, r_sh_pitch])
    error += _negsqrd_exp(asset.data.joint_pos[:, l_sh_pitch] - asset.data.default_joint_pos[:, l_sh_pitch])

    error += _negsqrd_exp(asset.data.joint_pos[:, r_sh_roll]  - asset.data.default_joint_pos[:, r_sh_roll])
    error += _negsqrd_exp(asset.data.joint_pos[:, l_sh_roll]  - asset.data.default_joint_pos[:, l_sh_roll])

    # elbow
    r_elbow = _idx(asset, "R_Elbow_Joint")
    l_elbow = _idx(asset, "L_Elbow_Joint")
    error += _negsqrd_exp(asset.data.joint_pos[:, r_elbow] - asset.data.default_joint_pos[:, r_elbow])
    error += _negsqrd_exp(asset.data.joint_pos[:, l_elbow] - asset.data.default_joint_pos[:, l_elbow])

    # 총 8항(어깨 6 + 팔꿈치 2)
    return error / 8.0



# * --- Floating base rewards --- * #

def base_height_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, base_height_target: float):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired base height
    base_height = asset.data.root_pos_w[:, 2]
    error = (base_height_target - base_height).flatten()
    return _negsqrd_exp(error)

def base_height_band_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
    max_height: float,
):
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    out_low = torch.relu(min_height - base_height)
    out_high = torch.relu(base_height - max_height)
    return -(out_low + out_high)

def base_heading_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired base heading
    base_heading = asset.data.heading_w
    commands = env.command_manager.get_command(command_name)
    command_heading = torch.atan2(commands[:, 1], commands[:, 0])
    base_heading_error = torch.abs(math_utils.wrap_to_pi(command_heading - base_heading))
    return _neg_exp(base_heading_error, scale=torch.pi/2)

def base_z_orientation_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking upright orientation
    error = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return _negsqrd_exp(error, scale=0.2)


def base_projected_gravity_lateral_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.0,
):
    """Reward lateral projected-gravity (y) being near zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = asset.data.projected_gravity_b[:, 1]
    return _negsqrd_exp(error, scale=scale)


def base_ang_vel_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.0,
):
    """Reward small base angular velocity magnitude (xy)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    error = torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    return _negsqrd_exp(error, scale=scale)

def tracking_lin_vel_world_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired linear velocity command in the world frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, :2] - asset.data.root_lin_vel_w[:, :2]
    error *= 1./(1. + torch.abs(commands[:, :2]))
    return _negsqrd_exp(error, scale=1.).sum(dim=1)

def tracking_lin_vel_xy_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired linear velocity command in the base frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, :2] - asset.data.root_lin_vel_b[:, :2]
    error *= 1./(1. + torch.abs(commands[:, :2]))
    return _negsqrd_exp(error, scale=1.).sum(dim=1)

def tracking_lin_vel_x_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired x linear velocity command in the base frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, 0] - asset.data.root_lin_vel_b[:, 0]
    error *= 1./(1. + torch.abs(commands[:, 0]))
    return _negsqrd_exp(error, scale=0.3)

def tracking_lin_vel_y_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired y linear velocity command in the base frame
    commands = env.command_manager.get_command(command_name)
    error = commands[:, 1] - asset.data.root_lin_vel_b[:, 1]
    error *= 1./(1. + torch.abs(commands[:, 1]))
    return _negsqrd_exp(error, scale=0.25)

def tracking_yaw_vel_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired yaw velocity command
    commands = env.command_manager.get_command(command_name)
    error = commands[:, 2] - asset.data.root_ang_vel_b[:, 2]
    error *= 1./(1. + torch.abs(commands[:, 2]))
    return _negsqrd_exp(error, scale=0.3)

def CAM_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    sigma = 1.0
    return -torch.sum(torch.square(env.CM_bf[:, 3:5]/sigma), dim=1)

def dCAM_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Penalize the change in centroidal angular momentum in the xy plane
    return -torch.clamp_min(torch.sum(env.CM_bf[:, 3:5] * env.dCM_bf[:, 3:5], dim=1), 0.0)  # only positive build-up

# def dCAM_xy_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
#     asset: RigidObject = env.scene[asset_cfg.name]

#     # x = k · k_dot  (xy-plane)
#     x = torch.sum(env.CM_bf[:, 3:5] * env.dCM_bf[:, 3:5], dim=1)

#     # image params
#     a = 3.0
#     c = 10.0

#     # f = x / (c + |x|)
#     f = x / (c + torch.abs(x))

#     # penalty = 1/2 * ( (1 - e^{a f}) / (1 + e^{a f}) - 1 )
#     ef = torch.exp(a * f)
#     return 0.5 * ((1.0 - ef) / (1.0 + ef) - 1.0)

def tracking_CAM_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Reward tracking desired centroidal angular momentum (humanoid config parity).
    error = env.CM_des[:, 5] - env.CM[:, 5]
    error *= 1.0 / (1.0 + torch.abs(env.CM_des[:, 5]))
    return _negsqrd_exp(error, scale=1.0)

def tracking_CAM_x_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Track desired centroidal angular momentum x in body frame.
    error = env.CM_des_bf[:, 3] - env.CM_bf[:, 3]
    error *= 1.0 / (1.0 + torch.abs(env.CM_des_bf[:, 3]))
    return _negsqrd_exp(error, scale=0.1)

def tracking_CAM_y_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Track desired centroidal angular momentum y in body frame.
    error = env.CM_des_bf[:, 4] - env.CM_bf[:, 4]
    error *= 1.0 / (1.0 + torch.abs(env.CM_des_bf[:, 4]))
    return _negsqrd_exp(error, scale=0.1)

def tracking_CAM_z_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    asset: RigidObject = env.scene[asset_cfg.name]
    # Track desired CAM angular component around z-axis in body frame.
    error = env.CM_des_bf[:, 5] - env.CM_bf[:, 5]
    error *= 1.0 / (1.0 + torch.abs(env.CM_des_bf[:, 5]))
    return _negsqrd_exp(error, scale=0.1)

def CAM_compensation_reward(env: ManagerBasedRLEnv):
    """ Compensating the centroidal angular momentum generated by leg joint by arm joint """
    return _negsqrd_exp(env.CM_leg[:, -1] + env.CM_arm[:, -1])

# * --- Stepping rewards --- * #


def _effective_contact_schedule(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return contact schedule optionally signed by per-episode first-foot convention."""
    sched = env.contact_schedule.squeeze(1)
    if hasattr(env, "first_foot_sign"):
        sign = env.first_foot_sign.squeeze(1).to(sched.dtype)
        sched = sched * sign
    return sched

def contact_schedule_reward(env: ManagerBasedRLEnv):
    """ Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing) """ 
    schedule = _effective_contact_schedule(env)
    return (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * schedule

def contact_schedule_stepper_reward(env: ManagerBasedRLEnv):
    """ Alternate right and left foot contacts
        First, right foot contacts (left foot swing), then left foot contacts (right foot swing) 
        Contact should be made at the target foot position """ 
    schedule = _effective_contact_schedule(env)
    contact_rewards = (env.foot_contact[:,0].int() - env.foot_contact[:,1].int()) * schedule
    tracking_rewards = 3 * _neg_exp(env.step_location_offset[~env.foot_on_motion])
    return contact_rewards * tracking_rewards

# * --- Termination rewards --- * #
def termination_penalty(env: ManagerBasedRLEnv, group_name: str):
    # Penalize termination (except for time_outs)
    return -env.termination_manager.terminated[group_name].float()/env.step_dt


# * ######################### HELPER FUNCTIONS ############################## * #

REWARD_TRACKING_SIGMA = 0.25

def _neg_exp(x, scale=1):
    """ shorthand helper for negative exponential e^(-x/scale)
        scale: range of x
    """
    return torch.exp(-(x/scale)/REWARD_TRACKING_SIGMA)

def _negsqrd_exp(x, scale=1):
    """ shorthand helper for negative squared exponential e^(-(x/scale)^2)
        scale: range of x
    """
    return torch.exp(-torch.square(x/scale)/REWARD_TRACKING_SIGMA)


def foot_contact_flat_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Penalize foot roll/pitch (yaw ignored), regardless of contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for foot orientation penalty.")
    foot_ids = env.body_feet_ids

    foot_quat = asset.data.body_quat_w[:, foot_ids, :]  # (N, 2, 4)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(foot_quat.reshape(-1, 4))
    roll = roll.view(env.num_envs, -1)
    pitch = pitch.view(env.num_envs, -1)
    max_angle = torch.pi / 2
    roll = torch.clamp(roll, -max_angle, max_angle)
    pitch = torch.clamp(pitch, -max_angle, max_angle)
    error = roll * roll + pitch * pitch

    return -torch.sum(error, dim=1)


def grf_limit_penalty(env: ManagerBasedRLEnv, threshold: float = 1500.0):
    """Penalize vertical GRF above a threshold for each foot."""
    if not hasattr(env, "contact_forces") or not hasattr(env, "feet_ids"):
        raise AttributeError("Expected env.contact_forces and env.feet_ids for GRF limit penalty.")
    grf_z = env.contact_forces[:, env.feet_ids, 2]
    excess = torch.clamp(grf_z - threshold, min=0.0)
    penalty = torch.square(excess / threshold)
    return -torch.sum(penalty, dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_force_threshold: float = 1.0,
    vel_scale: float = 0.5,
):
    """Penalize foot horizontal speed while in contact with ground.

    The penalty is dense and only active when vertical contact force exceeds
    ``contact_force_threshold``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for foot slip penalty.")
    if not hasattr(env, "contact_forces") or not hasattr(env, "feet_ids"):
        raise AttributeError("Expected env.contact_forces and env.feet_ids for foot slip penalty.")

    foot_ids = env.body_feet_ids  # body ids for velocity lookup
    foot_vel_xy = asset.data.body_state_w[:, foot_ids, 7:9]  # (N, 2, 2)
    foot_speed_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # (N, 2)

    # sensor feet_ids order is [right, left], same convention used across this task.
    contact_mask = (env.contact_forces[:, env.feet_ids, 2] > contact_force_threshold).float()
    penalty = contact_mask * (foot_speed_sq / max(vel_scale * vel_scale, 1e-6))
    return -torch.sum(penalty, dim=1)


def illegal_contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    """Penalize contact force magnitude above threshold (dense)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    term_contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1)
    excess = torch.clamp(term_contact - threshold, min=0.0)
    penalty = torch.sum(excess / threshold, dim=1)
    return -penalty


def knee_negative_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_knee_pos: float = 0.6,
):
    """Penalize knee angles below a minimum threshold with ReLU."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return -torch.sum(torch.relu(min_knee_pos - q), dim=1)


def shoulder1_action_penalty(
    env: ManagerBasedRLEnv,
    action_name: str,
    left_name: str = "L_Shoulder1_Joint",
    right_name: str = "R_Shoulder1_Joint",
):
    """Penalize shoulder1 action magnitude to keep it near zero."""
    action = env.action_manager.action[action_name]
    term = env.action_manager.get_term(action_name)
    joint_names = getattr(term, "_joint_names", None)
    if joint_names is None:
        return torch.zeros(action.shape[0], device=action.device)
    try:
        idx_l = joint_names.index(left_name)
        idx_r = joint_names.index(right_name)
    except ValueError as e:
        preview = joint_names[:50]
        raise ValueError(
            f"Shoulder1 joints not found in action term '{action_name}'. "
            f"left='{left_name}' right='{right_name}' preview(first50)={preview}"
        ) from e
    err = torch.square(action[:, idx_l]) + torch.square(action[:, idx_r])
    return -err


def swing_foot_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.05,
    max_height: float | None = 0.25,
    schedule_based: bool = True,
):
    """Penalize swing foot height (world z) below a minimum threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for swing foot height reward.")
    foot_ids = env.body_feet_ids
    foot_z = asset.data.body_pos_w[:, foot_ids, 2]
    shortfall = torch.clamp(min_height - foot_z, min=0.0)
    if max_height is not None:
        shortfall = torch.clamp(shortfall, max=max_height)
    if schedule_based:
        if not hasattr(env, "contact_schedule"):
            raise AttributeError("Expected env.contact_schedule for schedule-based swing foot height reward.")
        schedule = env.contact_schedule.squeeze(1)
        right_swing = (schedule < 0.0).float()
        left_swing = (schedule > 0.0).float()
        weight = torch.abs(schedule).unsqueeze(1)
        swing_mask = torch.stack([right_swing, left_swing], dim=1) * weight
    else:
        if not hasattr(env, "foot_contact"):
            raise AttributeError("Expected env.foot_contact for swing foot height reward.")
        swing_mask = (~env.foot_contact).float()
    return -torch.sum(shortfall * swing_mask, dim=1)


def swing_foot_height_cubic_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_height: float = 0.19,
    min_height: float = 0.14,
    rise_end: float = 0.24,
    hold_end: float = 0.25,
    swing_end: float = 0.5,
):
    """Penalize deviation from a cubic swing-foot height profile.

    Swing profile (per foot, in swing-phase s in [0, 0.5]):
      - 0.0~rise_end: cubic rise from min_height to max_height
      - rise_end~hold_end: hold max_height
      - hold_end~swing_end: cubic fall to min_height
    Support foot gets zero weight (no penalty).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for swing foot height reward.")
    if not hasattr(env, "phase"):
        raise AttributeError("Expected env.phase for swing foot height reward.")

    foot_ids = env.body_feet_ids  # order: [right, left]
    foot_z = asset.data.body_pos_w[:, foot_ids, 2]

    phase = env.phase
    if phase.dim() > 1:
        phase = phase.squeeze(-1)

    def smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def swing_profile(swing_phase: torch.Tensor) -> torch.Tensor:
        rise_t = smoothstep(swing_phase / rise_end)
        fall_t = smoothstep((swing_phase - hold_end) / (swing_end - hold_end))
        amp = max_height - min_height
        rise_val = min_height + amp * rise_t
        fall_val = min_height + amp * (1.0 - fall_t)
        return torch.where(
            swing_phase < rise_end,
            rise_val,
            torch.where(
                swing_phase < hold_end,
                torch.full_like(swing_phase, max_height),
                torch.where(
                    swing_phase < swing_end,
                    fall_val,
                    torch.full_like(swing_phase, min_height),
                ),
            ),
        )

    left_mask = phase < 0.5
    right_mask = ~left_mask

    left_phase = phase
    right_phase = phase - 0.5

    left_target = torch.where(left_mask, swing_profile(left_phase), torch.zeros_like(phase))
    right_target = torch.where(right_mask, swing_profile(right_phase), torch.zeros_like(phase))

    desired_z = torch.stack([right_target, left_target], dim=1)
    swing_mask = torch.stack([right_mask.float(), left_mask.float()], dim=1)
    shortfall = torch.clamp(desired_z - foot_z, min=0.0)
    penalty = shortfall * swing_mask
    return -torch.sum(penalty, dim=1)


def termination_penalty_term(env: ManagerBasedRLEnv, group_name: str, term_name: str):
    """Penalize a specific termination term only (excluding time_outs)."""
    term = env.termination_manager.get_term(group_name, term_name)
    return -term.float() / env.step_dt


# * ######################### HELPER FUNCTIONS ############################## * #

REWARD_TRACKING_SIGMA = 0.25


def wrist_height_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    """Penalize high wrist height (world z)."""
    asset: Articulation = env.scene[asset_cfg.name]
    wrist_ids = asset_cfg.body_ids
    if wrist_ids is None:
        raise AttributeError("Expected asset_cfg.body_ids for wrist height penalty.")
    z = asset.data.body_pos_w[:, wrist_ids, 2]
    return -torch.sum(torch.square(z), dim=1)
