# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""

def base_lin_vel_world(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w

def base_heading(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root heading in the world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.heading_w.unsqueeze(-1)

def base_ang_vel_lpf(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lpf_cutoff_hz: float = 25.0,
    lpf_enabled: bool = True,
    lpf_key: str | None = None,
) -> torch.Tensor:
    """Root angular velocity in base frame with optional LPF (applied before noise)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b
    if not lpf_enabled:
        return w

    state_key = (asset_cfg.name, "base_ang_vel", w.shape[1], lpf_key)
    dt = _get_obs_dt(env)
    alpha = _compute_lpf_alpha(float(lpf_cutoff_hz), dt)

    if not hasattr(env, "_obs_base_ang_vel_lpf_state"):
        env._obs_base_ang_vel_lpf_state = {}
    state = env._obs_base_ang_vel_lpf_state.get(state_key, None)
    if state is None or state.shape != w.shape:
        filt = w.clone()
    else:
        filt = state + alpha * (w - state)
    env._obs_base_ang_vel_lpf_state[state_key] = filt
    return filt


"""
Joint state.
"""

def _compute_lpf_alpha(cutoff_hz: float, dt: float) -> float:
    if cutoff_hz <= 0.0 or dt <= 0.0:
        return 1.0
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return float(dt / (rc + dt))


def _get_obs_dt(env: ManagerBasedEnv) -> float:
    base_dt = float(getattr(env, "step_dt", 0.0) or 0.0)
    decimation = int(getattr(getattr(env, "cfg", None), "decimation", 1) or 1)
    dt = base_dt * decimation if base_dt > 0.0 else 0.0
    return dt if dt > 0.0 else (1.0 / 60.0)


def joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    episodic_bias_enabled: bool = False,
    episodic_bias_sigma: float = 0.0,
    episodic_bias_key: str | None = None,
) -> torch.Tensor:
    """The joint positions of the asset.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    Optionally adds per-episode Gaussian bias (constant within episode, resampled at reset).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]

    if (not episodic_bias_enabled) or episodic_bias_sigma <= 0.0:
        return q

    # Prefer shared all-joint episodic bias so observations and PD torque feedback use identical bias.
    all_bias = getattr(env, "_joint_pos_episode_bias_all", None)
    if isinstance(all_bias, torch.Tensor) and all_bias.ndim == 2 and all_bias.shape[0] == q.shape[0]:
        joint_ids = getattr(asset_cfg, "joint_ids", None)
        if isinstance(joint_ids, slice):
            bias = all_bias[:, joint_ids]
        elif joint_ids is None:
            bias = all_bias
        else:
            idx = torch.as_tensor(list(joint_ids), device=all_bias.device, dtype=torch.long)
            bias = all_bias.index_select(1, idx)
        if bias.shape == q.shape:
            return q + bias

    key_joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(key_joint_ids, slice):
        key_joint_ids = ("slice", key_joint_ids.start, key_joint_ids.stop, key_joint_ids.step)
    elif key_joint_ids is None:
        key_joint_ids = "all"
    else:
        key_joint_ids = tuple(int(i) for i in key_joint_ids)
    state_key = (asset_cfg.name, key_joint_ids, q.shape[1], episodic_bias_key)

    if not hasattr(env, "_obs_joint_pos_episode_bias"):
        env._obs_joint_pos_episode_bias = {}
    if not hasattr(env, "_obs_joint_pos_episode_bias_sigma"):
        env._obs_joint_pos_episode_bias_sigma = {}

    bias = env._obs_joint_pos_episode_bias.get(state_key, None)
    sigma = float(episodic_bias_sigma)
    if bias is None or bias.shape != q.shape:
        bias = torch.randn_like(q) * sigma
        env._obs_joint_pos_episode_bias[state_key] = bias
    env._obs_joint_pos_episode_bias_sigma[state_key] = sigma

    return q + bias


def joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lpf_cutoff_hz: float = 25.0,
    lpf_enabled: bool = True,
    lpf_key: str | None = None,
):
    """The joint velocities of the asset.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if not lpf_enabled:
        return qd

    key_joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(key_joint_ids, slice):
        key_joint_ids = ("slice", key_joint_ids.start, key_joint_ids.stop, key_joint_ids.step)
    elif key_joint_ids is None:
        key_joint_ids = "all"
    else:
        key_joint_ids = tuple(int(i) for i in key_joint_ids)
    state_key = (asset_cfg.name, key_joint_ids, qd.shape[1], lpf_key)

    dt = _get_obs_dt(env)
    alpha = _compute_lpf_alpha(float(lpf_cutoff_hz), dt)

    if not hasattr(env, "_obs_joint_vel_lpf_state"):
        env._obs_joint_vel_lpf_state = {}
    state = env._obs_joint_vel_lpf_state.get(state_key, None)
    if state is None or state.shape != qd.shape:
        filt = qd.clone()
    else:
        filt = state + alpha * (qd - state)
    env._obs_joint_vel_lpf_state[state_key] = filt
    return filt


def last_action_by_joint_names(
    env: ManagerBasedRLEnv,
    action_name: str,
    joint_names_expr: list[str],
) -> torch.Tensor:
    """Slice last action for specific joints from an action term."""
    term = env.action_manager.get_term(action_name)
    if term is None:
        raise AttributeError(f"Action term '{action_name}' not found.")
    # Prefer clipped raw actions if available (after term-level clamping)
    action = getattr(term, "raw_actions", None)
    if action is None:
        action = env.action_manager.action[action_name]
    joint_names = getattr(term, "_joint_names", None)
    if joint_names is None:
        return action
    idx, _ = string_utils.resolve_matching_names(joint_names_expr, list(joint_names), preserve_order=True)
    return action[:, idx]

def centroidal_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset."""
    return env.CM

def centroidal_momentum_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset in the base frame."""
    return env.CM_bf

def centroidal_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des

def centroidal_momentum_des_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset in the base frame."""
    return env.CM_des_bf

def centroidal_linear_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal linear momentum of the asset."""
    return env.CM[:, :3]

def centroidal_linear_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, :3]

def centroidal_angular_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset."""
    return env.CM[:, 3:]

def centroidal_angular_momentum_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset in the base frame."""
    return env.CM_bf[:, 3:]

def centroidal_angular_momentum_mixed(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    return torch.hstack([env.CM_bf[:, 3:5], env.CM[:, 5:6]])

def centroidal_angular_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, 3:]

def centroidal_angular_momentum_des_bf(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset in the base frame."""
    return env.CM_des_bf[:, 3:]

def centroidal_angular_momentum_des_mixed(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal angular momentum of the asset in the mixed frame. \
       x, y axis in the base frame, z axis in the world frame."""
    return torch.hstack([env.CM_des_bf[:, 3:5], env.CM_des[:, 5:6]])

def centroidal_momentum_xy_linear_z_angular(env: ManagerBasedEnv) -> torch.Tensor:
    """Mixed centroidal momentum in body frame: linear x,y and angular z."""
    return torch.hstack([env.CM_bf[:, 0:2], env.CM_bf[:, 5:6]])

def centroidal_momentum_des_xy_linear_z_angular(env: ManagerBasedEnv) -> torch.Tensor:
    """Desired mixed centroidal momentum in body frame: linear x,y and angular z."""
    return torch.hstack([env.CM_des_bf[:, 0:2], env.CM_des_bf[:, 5:6]])

def centroidal_yaw_momentum(env: ManagerBasedEnv) -> torch.Tensor:
    """The centroidal momentum of the asset."""
    return env.CM[:, 5:6]

def centroidal_yaw_momentum_des(env: ManagerBasedEnv) -> torch.Tensor:
    """The desired centroidal momentum of the asset."""
    return env.CM_des[:, 5:6]

"""
Robot state.
"""

def foot_states_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right foot states relative to the base frame."""
    
    return env.foot_states_right

def foot_states_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left foot states relative to the base frame."""
    
    return env.foot_states_left

"""
Commands.
"""

def phase_sin(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sin of the phase variable."""
    return env.phase_sin

def phase_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cos of the phase variable."""
    return env.phase_cos

def joint_effort_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The target joint efforts."""
    return env.joint_effort_target.float()

def step_commands_right(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right foot step commands relative to the base frame."""
    return env.step_commands_right

def step_commands_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left foot step commands relative to the base frame."""
    return env.step_commands_left
