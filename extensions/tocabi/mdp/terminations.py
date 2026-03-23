# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


"""
Root terminations.
"""

def base_termination(
    env: ManagerBasedRLEnv, max_lin_vel: float = None, max_ang_vel: float = None, max_tilting: float = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")                
) -> torch.Tensor:
    """Terminate when the asset's linear velocity exceeds the maximum linear velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if max_lin_vel is not None:
        terminated |= torch.any(torch.norm(asset.data.root_lin_vel_b, dim=-1, keepdim=True) > max_lin_vel, dim=1)
    if max_ang_vel is not None:
        terminated |= torch.any(torch.norm(asset.data.root_ang_vel_b, dim=-1, keepdim=True) > max_ang_vel, dim=1)
    if max_tilting is not None:
        terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 0:1]) > max_tilting, dim=1)
        terminated |= torch.any(torch.abs(asset.data.projected_gravity_b[:, 1:2]) > max_tilting, dim=1)
    return terminated


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w
    # check if any contact force exceeds the threshold
    term_contact = torch.norm(net_contact_forces[:, sensor_cfg.body_ids, :], dim=-1)
    terminated = torch.any((term_contact > threshold), dim=1)
    return terminated

def joint_pos_out_of_bounds(
    env: ManagerBasedRLEnv,
    bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when joint positions are outside the specified bounds."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    lower, upper = bounds
    return torch.any((q < lower) | (q > upper), dim=1)


def foot_crossing_yaw_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_y_dist: float = 0.0
) -> torch.Tensor:
    """Terminate when feet cross in yaw-only frame (left y <= right y + min_y_dist)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "body_feet_ids"):
        raise AttributeError("Expected env.body_feet_ids for foot crossing termination.")
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
    # env.body_feet_ids order: [right, left]
    right_y = rel_pos_yaw[:, 0, 1]
    left_y = rel_pos_yaw[:, 1, 1]
    return left_y <= (right_y + min_y_dist)


def feet_collision(env: ManagerBasedRLEnv, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
    """Terminate on physics contact between filtered foot pairs only."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
    # If filter_prim_paths_expr is used, force_matrix_w is pairwise contact force.
    force_matrix = getattr(contact_sensor.data, "force_matrix_w", None)
    if force_matrix is not None:
        mag = torch.norm(force_matrix, dim=-1)
        reduce_dims = tuple(range(1, mag.ndim))
        return torch.any(mag > threshold, dim=reduce_dims)
    net_forces = contact_sensor.data.net_forces_w
    mag = torch.norm(net_forces, dim=-1)
    return torch.any(mag > threshold, dim=1)
