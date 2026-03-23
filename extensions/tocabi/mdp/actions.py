from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import math

import omni.log
import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction, JointPositionAction


class JointPositionActionClipped(JointPositionAction):
    """Joint position action with default offset and clipping to joint limits.

    This implements:
        q_des = clip(scale * action + q_ref, q_min, q_max)
    """

    def __init__(self, cfg: "JointPositionActionClippedCfg", env):
        super().__init__(cfg, env)
        self._joint_pos_limits = None
        if getattr(self.cfg, "joint_pos_limits", None) is not None:
            if isinstance(self.cfg.joint_pos_limits, dict):
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.joint_pos_limits, self._joint_names
                )
                limits = torch.zeros(self.action_dim, 2, device=self.device)
                limits[:, 0] = -1.0
                limits[:, 1] = 1.0
                limits[index_list] = torch.tensor(value_list, device=self.device)
                self._joint_pos_limits = limits
            else:
                limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)
                if limits.shape[0] != self.action_dim or limits.shape[1] != 2:
                    raise ValueError(
                        f"joint_pos_limits must be (action_dim, 2), got {tuple(limits.shape)}"
                    )
                self._joint_pos_limits = limits

    def process_actions(self, actions: torch.Tensor):
        if self._joint_pos_limits is not None:
            actions = math_utils.unscale_transform(
                actions, self._joint_pos_limits[:, 0], self._joint_pos_limits[:, 1]
            )
        super().process_actions(actions)
        # clip to the articulation's soft joint limits
        lower = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        upper = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]
        self._processed_actions = torch.clamp(self._processed_actions, min=lower, max=upper)


@configclass
class JointPositionActionClippedCfg(JointPositionActionCfg):
    """Configuration for JointPositionActionClipped."""

    class_type: type[ActionTerm] = JointPositionActionClipped
    # Optional per-joint action->position limits (same shape/order as action joints)
    joint_pos_limits: dict[str, tuple[float, float]] | list[tuple[float, float]] | None = None


class TocabiAction(ActionTerm):
    """Tocabi action that applies explicit PD torques to lower + upper joints."""

    cfg: "TocabiActionCfg"
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor

    def __init__(self, cfg: "TocabiActionCfg", env):
        super().__init__(cfg, env)

        # resolve joint names (keep cfg order for q_init/action mapping)
        self._joint_ids, _resolved_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        self._joint_names = list(self.cfg.lower_joint_names)
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._upper_joint_names = list(self.cfg.upper_joint_names)
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]
        self._default_lower_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._num_joints = len(self._joint_ids)
        # precompute arm indices in action vector (by joint name)
        self._arm_action_idx = [
            i
            for i, name in enumerate(self._joint_names)
            if ("Shoulder" in name or "Elbow" in name)
        ]
        self._fixed_zero_action_idx: list[int] = []
        fixed_zero_names = getattr(self.cfg, "fixed_zero_action_joint_names", None)
        if fixed_zero_names:
            for name in fixed_zero_names:
                if name in self._joint_names:
                    self._fixed_zero_action_idx.append(self._joint_names.index(name))
        self._upper_wrist1_zero_idx: list[int] = []
        for wrist_name in ("L_Wrist1_Joint", "R_Wrist1_Joint"):
            if wrist_name in self._upper_joint_names:
                self._upper_wrist1_zero_idx.append(self._upper_joint_names.index(wrist_name))

        self._p_gains = torch.tensor(self.cfg.p_gains, device=self.device)
        self._d_gains = torch.tensor(self.cfg.d_gains, device=self.device)
        self._torque_limits = torch.tensor(self.cfg.torque_limits, device=self.device)
        self._rand_torque_inj_range = torch.tensor(self.cfg.rand_torque_inj_range, device=self.device)
        self._rand_motor_scale_range = torch.tensor(self.cfg.rand_motor_scale_range, device=self.device)
        self._rand_torque_episode_range = torch.tensor(self.cfg.rand_torque_episode_range, device=self.device)
        self._rand_torque_tick_range = torch.tensor(self.cfg.rand_torque_tick_range, device=self.device)
        self._episode_torque_bias = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._last_pd_torque_full = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)
        self._lower_joint_pos_limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)
        self._dbg_step = 0
        self._joint_vel_lpf_state = torch.zeros(
            self.num_envs, len(self._joint_ids) + len(self._upper_joint_ids), device=self.device
        )
        self._joint_vel_lpf_initialized = False

        # q_init override (per joint name) for non-controlled joints
        self._q_init_lower_pos = self._default_lower_joint_pos.clone()
        self._q_init_upper_pos = self._default_upper_joint_pos.clone()
        self._q_init_dict: dict[str, float] | None = None
        self._init_q_init_buffers(getattr(self.cfg, "q_init", None))
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._processed_actions = actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if not hasattr(self, "_dbg_disable_once"):
            self._dbg_disable_once = False
        if not self._dbg_disable_once:
            scale_val = (
                float(self._scale.abs().max().item()) if isinstance(self._scale, torch.Tensor) else float(self._scale)
            )
            print(
                f"[ACT_DBG] term={self.__class__.__name__} "
                f"disable_arm_action={getattr(self.cfg, 'disable_arm_action', None)} "
                f"scale={scale_val}",
                flush=True,
            )
            self._dbg_disable_once = True
        # If arm action is disabled, keep only arm joints at q_init (others still follow mapped actions).
        scale_disabled = False
        if isinstance(self._scale, torch.Tensor):
            scale_disabled = bool((self._scale.abs().max().item() == 0.0))
        elif isinstance(self._scale, (float, int)):
            scale_disabled = float(self._scale) == 0.0
        if scale_disabled:
            self._raw_actions.zero_()
            self._processed_actions[:] = self._q_init_lower_pos
            return
        if self.cfg.pd_control:
            if getattr(self.cfg, "force_zero_raw_action", False):
                actions = torch.zeros_like(self._processed_actions)
            else:
                actions = self._processed_actions.clamp(-1.0, 1.0)
            if self._fixed_zero_action_idx:
                actions[:, self._fixed_zero_action_idx] = 0.0
            self._raw_actions[:] = actions
            actions = math_utils.unscale_transform(
                actions,
                self._lower_joint_pos_limits[:, 0],
                self._lower_joint_pos_limits[:, 1],
            )
            self._processed_actions[:] = actions
            if getattr(self.cfg, "disable_arm_action", False) and self._arm_action_idx:
                self._processed_actions[:, self._arm_action_idx] = self._q_init_lower_pos[:, self._arm_action_idx]
                self._raw_actions[:, self._arm_action_idx] = 0.0

    def apply_actions(self):
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids
        qd_all = torch.cat(
            [
                self._asset.data.joint_vel[:, self._joint_ids],
                self._asset.data.joint_vel[:, self._upper_joint_ids],
            ],
            dim=1,
        )
        cutoff_hz = float(getattr(self.cfg, "joint_vel_lpf_cutoff_hz", 25.0))
        if cutoff_hz > 0.0:
            dt = float(getattr(self._env, "physics_dt", 0.0) or 0.0)
            if dt <= 0.0:
                dt = float(getattr(self._env, "step_dt", 0.0) or 0.0)
            if dt <= 0.0:
                dt = 1.0 / 60.0
            rc = 1.0 / (2.0 * math.pi * cutoff_hz)
            alpha = float(dt / (rc + dt))
            if (not self._joint_vel_lpf_initialized) or self._joint_vel_lpf_state.shape != qd_all.shape:
                self._joint_vel_lpf_state = qd_all.clone()
                self._joint_vel_lpf_initialized = True
            else:
                self._joint_vel_lpf_state = self._joint_vel_lpf_state + alpha * (qd_all - self._joint_vel_lpf_state)
            qd_all_used = self._joint_vel_lpf_state
        else:
            qd_all_used = qd_all

        if self.cfg.pd_control:
            upper_target = self._q_init_upper_pos
            # When arm control is enabled, keep wrist1 q_des fixed at zero.
            if (not getattr(self.cfg, "disable_arm_action", False)) and self._upper_wrist1_zero_idx:
                upper_target = upper_target.clone()
                upper_target[:, self._upper_wrist1_zero_idx] = 0.0
            target_pos = torch.cat([self.processed_actions, upper_target], dim=1)
            lower_count = len(self._joint_ids)
            q_lower = self._asset.data.joint_pos[:, self._joint_ids]
            q_upper = self._asset.data.joint_pos[:, self._upper_joint_ids]
            # Apply same episodic encoder bias to PD feedback if available.
            bias_all = getattr(self._env, "_joint_pos_episode_bias_all", None)
            if isinstance(bias_all, torch.Tensor) and bias_all.ndim == 2 and bias_all.shape[0] == self.num_envs:
                q_lower = q_lower + bias_all[:, self._joint_ids]
                q_upper = q_upper + bias_all[:, self._upper_joint_ids]
            lower_body_torque = (
                self._p_gains[:lower_count] / 9.0
                * (target_pos[:, :lower_count] - q_lower)
                + self._d_gains[:lower_count] / 3.0 * (-qd_all_used[:, :lower_count])
            )
            upper_body_torque = (
                self._p_gains[lower_count:]
                * (target_pos[:, lower_count:] - q_upper)
                + self._d_gains[lower_count:] * (-qd_all_used[:, lower_count:])
            )
            pd_torque = torch.cat([lower_body_torque, upper_body_torque], dim=1)
            target_effort = pd_torque
            # store PD torque (before bias/clipping) in full joint order
            self._last_pd_torque_full.zero_()
            joint_ids_ordered = self._joint_ids + self._upper_joint_ids
            self._last_pd_torque_full[:, joint_ids_ordered] = pd_torque
        else:
            rand_motor_scale = math_utils.sample_uniform(
                self._rand_motor_scale_range[0],
                self._rand_motor_scale_range[1],
                self.processed_actions.shape,
                device=self.device,
            )
            lower_count = len(self._joint_ids)
            lower_body_torque = self.processed_actions * self._torque_limits[:lower_count] * rand_motor_scale
            q_upper = self._asset.data.joint_pos[:, self._upper_joint_ids]
            bias_all = getattr(self._env, "_joint_pos_episode_bias_all", None)
            if isinstance(bias_all, torch.Tensor) and bias_all.ndim == 2 and bias_all.shape[0] == self.num_envs:
                q_upper = q_upper + bias_all[:, self._upper_joint_ids]
            upper_body_torque = (
                self._p_gains[lower_count:] / 9.0
                * (self._default_upper_joint_pos - q_upper)
                + self._d_gains[lower_count:] / 3.0 * (-qd_all_used[:, lower_count:])
            )
            lower_rand_torque_inj = math_utils.sample_uniform(
                self._rand_torque_inj_range[0],
                self._rand_torque_inj_range[1],
                lower_body_torque.shape,
                device=self.device,
            )
            target_effort = torch.cat([lower_body_torque + lower_rand_torque_inj, upper_body_torque], dim=1)
            # no PD torque for lower joints in this mode
            self._last_pd_torque_full.zero_()

        lower_count = len(self._joint_ids)
        # episode bias + tick bias (lower only)
        if self.cfg.enable_torque_bias and lower_count > 0:
            tick_mean = float(self._rand_torque_tick_range[0])
            tick_std = float(self._rand_torque_tick_range[1])
            if tick_std != 0.0 or tick_mean != 0.0:
                tick_torque = torch.randn(
                    (self.num_envs, lower_count), device=self.device, dtype=self._episode_torque_bias.dtype
                ) * tick_std + tick_mean
            else:
                tick_torque = torch.zeros(
                    self.num_envs, lower_count, device=self.device, dtype=self._episode_torque_bias.dtype
                )
            target_effort[:, :lower_count] += self._episode_torque_bias[:, :lower_count] + tick_torque


        # torque clamp
        total = target_effort.shape[1]
        if total > len(self._torque_limits):
            upper_lim = torch.full((total - len(self._torque_limits),), float(self._torque_limits[-1]), device=self.device)
            torque_limits = torch.cat([self._torque_limits, upper_lim], dim=0)
        else:
            torque_limits = self._torque_limits[:total]
        torque_limits = torque_limits.view(1, -1).expand(self.num_envs, -1)
        target_effort = torch.clamp(target_effort, min=-torque_limits, max=torque_limits)

        self._asset.set_joint_effort_target(target_effort, joint_ids=joint_ids_ordered)

        # Debug: print arm action stats (periodic)
        if self._arm_action_idx and getattr(self.cfg, "log_arm_debug", False):
            if not hasattr(self, "_dbg_arm_step"):
                self._dbg_arm_step = 0
            self._dbg_arm_step += 1
            if (self._dbg_arm_step % 2) == 0:
                arm_raw = self._raw_actions[:, self._arm_action_idx]
                arm_proc = self._processed_actions[:, self._arm_action_idx]
                # q_des for arm joints (processed actions) and q_init (configured)
                arm_q_init = self._q_init_lower_pos[:, self._arm_action_idx]
                arm_q_des = arm_proc
                msg = (
                    f"[ARM_ACT_RAW] mean={arm_raw.mean().item():.4f} "
                    f"min={arm_raw.min().item():.4f} max={arm_raw.max().item():.4f} "
                    f"zero_frac={(arm_raw.abs() < 1e-6).float().mean().item():.3f} | "
                    f"[ARM_ACT_PROC] mean={arm_proc.mean().item():.4f} "
                    f"min={arm_proc.min().item():.4f} max={arm_proc.max().item():.4f}"
                )
                omni.log.info(msg)
                print(msg, flush=True)
                # dump first env arm q_des vs q_init with joint names
                qdes0 = arm_q_des[0].detach().cpu().tolist()
                qinit0 = arm_q_init[0].detach().cpu().tolist()
                arm_names = [self._joint_names[i] for i in self._arm_action_idx]
                print(f"[ARM_QDES] {qdes0}", flush=True)
                print(f"[ARM_QINIT] {qinit0}", flush=True)
                print(f"[ARM_NAMES] {arm_names}", flush=True)
                named = [f"{n}: q_des={qd:.4f} q_init={qi:.4f}" for n, qd, qi in zip(arm_names, qdes0, qinit0)]
                print("[ARM_QDES_QINIT] " + " | ".join(named), flush=True)

    def _init_q_init_buffers(self, q_init: dict[str, float] | None):
        """Build q_init tensors from dict (if provided)."""
        if not isinstance(q_init, dict):
            self._q_init_dict = None
            self._q_init_lower_pos = self._default_lower_joint_pos.clone()
            self._q_init_upper_pos = self._default_upper_joint_pos.clone()
            return
        self._q_init_dict = dict(q_init)
        lower_vals = []
        for name in self._joint_names:
            if name in q_init:
                lower_vals.append(float(q_init[name]))
            else:
                lower_vals.append(0.0)
        upper_vals = []
        for name in self.cfg.upper_joint_names:
            if name in q_init:
                upper_vals.append(float(q_init[name]))
            else:
                upper_vals.append(0.0)
        if len(lower_vals) == len(self._joint_names):
            lower_t = torch.tensor(lower_vals, device=self.device).view(1, -1).repeat(self.num_envs, 1)
            mask = torch.tensor(
                [name in q_init for name in self._joint_names],
                device=self.device,
                dtype=torch.bool,
            ).view(1, -1)
            self._q_init_lower_pos = torch.where(mask, lower_t, self._default_lower_joint_pos)
        if len(upper_vals) == len(self.cfg.upper_joint_names):
            upper_t = torch.tensor(upper_vals, device=self.device).view(1, -1).repeat(self.num_envs, 1)
            mask = torch.tensor(
                [name in q_init for name in self.cfg.upper_joint_names],
                device=self.device,
                dtype=torch.bool,
            ).view(1, -1)
            self._q_init_upper_pos = torch.where(mask, upper_t, self._default_upper_joint_pos)

    def set_q_init(self, q_init: dict[str, float] | None):
        """Update q_init buffers after init (e.g., from env cfg post_init)."""
        self._init_q_init_buffers(q_init)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        if env_ids is None:
            env_ids = slice(None)
        try:
            qd_all = torch.cat(
                [
                    self._asset.data.joint_vel[:, self._joint_ids],
                    self._asset.data.joint_vel[:, self._upper_joint_ids],
                ],
                dim=1,
            )
            self._joint_vel_lpf_state[env_ids] = qd_all[env_ids]
            self._joint_vel_lpf_initialized = True
        except Exception:
            self._joint_vel_lpf_initialized = False
        if self.cfg.enable_torque_bias and self._num_joints > 0:
            epi_mean = float(self._rand_torque_episode_range[0])
            epi_std = float(self._rand_torque_episode_range[1])
            if epi_std != 0.0 or epi_mean != 0.0:
                self._episode_torque_bias[env_ids] = (
                    torch.randn(
                        (self._episode_torque_bias[env_ids].shape[0], self._num_joints),
                        device=self.device,
                        dtype=self._episode_torque_bias.dtype,
                    )
                    * epi_std
                    + epi_mean
                )
            else:
                self._episode_torque_bias[env_ids] = 0.0


class ArmPdAction(ActionTerm):
    """PD torque action for arm joints with explicit gains/limits."""

    cfg: "ArmPdActionCfg"
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor

    def __init__(self, cfg: "ArmPdActionCfg", env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self._joint_names]
        self._num_joints = len(self._joint_ids)

        self._p_gains = torch.tensor(self.cfg.p_gains, device=self.device)
        self._d_gains = torch.tensor(self.cfg.d_gains, device=self.device)
        self._torque_limits = torch.tensor(self.cfg.torque_limits, device=self.device)
        self._joint_pos_limits = torch.tensor(self.cfg.joint_pos_limits, device=self.device)

        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._joint_vel_lpf_state = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._joint_vel_lpf_initialized = False

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._processed_actions = actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        actions = self._processed_actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = actions
        actions = math_utils.unscale_transform(
            actions,
            self._joint_pos_limits[:, 0],
            self._joint_pos_limits[:, 1],
        )
        self._processed_actions[:] = actions

    def apply_actions(self):
        target_pos = self.processed_actions
        q = self._asset.data.joint_pos[:, self._joint_ids]
        # Apply same episodic encoder bias to PD feedback if available.
        bias_all = getattr(self._env, "_joint_pos_episode_bias_all", None)
        if isinstance(bias_all, torch.Tensor) and bias_all.ndim == 2 and bias_all.shape[0] == self.num_envs:
            q = q + bias_all[:, self._joint_ids]
        qd = self._asset.data.joint_vel[:, self._joint_ids]
        cutoff_hz = float(getattr(self.cfg, "joint_vel_lpf_cutoff_hz", 25.0))
        if cutoff_hz > 0.0:
            dt = float(getattr(self._env, "physics_dt", 0.0) or 0.0)
            if dt <= 0.0:
                dt = float(getattr(self._env, "step_dt", 0.0) or 0.0)
            if dt <= 0.0:
                dt = 1.0 / 60.0
            rc = 1.0 / (2.0 * math.pi * cutoff_hz)
            alpha = float(dt / (rc + dt))
            if (not self._joint_vel_lpf_initialized) or self._joint_vel_lpf_state.shape != qd.shape:
                self._joint_vel_lpf_state = qd.clone()
                self._joint_vel_lpf_initialized = True
            else:
                self._joint_vel_lpf_state = self._joint_vel_lpf_state + alpha * (qd - self._joint_vel_lpf_state)
            qd_used = self._joint_vel_lpf_state
        else:
            qd_used = qd
        pd_torque = (
            self._p_gains / 9.0 * (target_pos - q)
            + self._d_gains / 3.0 * (-qd_used)
        )
        torque_limits = self._torque_limits.view(1, -1).expand(self.num_envs, -1)
        pd_torque = torch.clamp(pd_torque, min=-torque_limits, max=torque_limits)
        self._asset.set_joint_effort_target(pd_torque, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        if env_ids is None:
            env_ids = slice(None)
        try:
            qd = self._asset.data.joint_vel[:, self._joint_ids]
            self._joint_vel_lpf_state[env_ids] = qd[env_ids]
            self._joint_vel_lpf_initialized = True
        except Exception:
            self._joint_vel_lpf_initialized = False


class JointPositionActionWithUpper(JointAction):
    """Joint position action that also enforces default upper-joint targets."""

    cfg: "TocabiJointPositionActionCfg"

    def __init__(self, cfg: "TocabiJointPositionActionCfg", env):
        super().__init__(cfg, env)
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.lower_joint_names)
        self._joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.lower_joint_names]
        self._upper_joint_ids = [self._asset.data.joint_names.index(joint_name) for joint_name in self.cfg.upper_joint_names]
        self._default_upper_joint_pos = self._asset.data.default_joint_pos[:, self._upper_joint_ids]

        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")

        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    def apply_actions(self):
        target_pos = torch.cat([self.processed_actions, self._default_upper_joint_pos], dim=1)
        joint_ids_ordered = self._joint_ids + self._upper_joint_ids
        self._asset.set_joint_position_target(target_pos, joint_ids=joint_ids_ordered)


@configclass
class TocabiActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = TocabiAction

    lower_joint_names: list[str] = MISSING
    upper_joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    rescale_to_limits: bool = True
    pd_control: bool = True
    joint_pos_limits: list[tuple[float, float]] = MISSING
    torque_limits: list[float] = MISSING
    p_gains: list[float] = MISSING
    d_gains: list[float] = MISSING
    enable_torque_bias: bool = False
    rand_torque_inj_range: tuple[float, float] = (0.0, 0.0)
    rand_torque_episode_range: tuple[float, float] = (0.0, 0.0)
    rand_torque_tick_range: tuple[float, float] = (0.0, 0.0)
    rand_motor_scale_range: tuple[float, float] = (1.0, 1.0)
    disable_arm_action: bool = False
    log_arm_debug: bool = False
    force_zero_raw_action: bool = False
    q_init: dict[str, float] | None = None
    joint_vel_lpf_cutoff_hz: float = 25.0
    fixed_zero_action_joint_names: list[str] | None = None


@configclass
class ArmPdActionCfg(ActionTermCfg):
    """Configuration for ArmPdAction."""

    class_type: type[ActionTerm] = ArmPdAction

    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    clip: dict[str, tuple[float, float]] | None = None
    joint_pos_limits: list[tuple[float, float]] = MISSING
    torque_limits: list[float] = MISSING
    p_gains: list[float] = MISSING
    d_gains: list[float] = MISSING
    joint_vel_lpf_cutoff_hz: float = 25.0


@configclass
class TocabiJointPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = JointPositionActionWithUpper

    use_default_offset: bool = True
    joint_names: list[str] = MISSING
    lower_joint_names: list[str] = MISSING
    upper_joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
