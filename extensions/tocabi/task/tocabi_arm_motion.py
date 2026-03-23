from __future__ import annotations

import torch
from collections.abc import Sequence
import time

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
import carb

from ..utils import VanillaKeyboard, TocabiFullVanillaRecorder
import extensions.tocabi.utils as brl_utils
from extensions.tocabi.assets.tocabi import ARM_LOCKED_JOINT_EXPR, TRUNK_JOINT_EXPR
from .tocabi_vanilla import TocabiVanillaEnv


class TocabiArmMotionEnv(TocabiVanillaEnv):
    """
    - leg_joint_pos + arm_joint_pos 액션 텀 사용 (leg/arm concat)
    - action_full (num_envs, num_joints) 에 scatter
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Backward-compat for wrappers expecting "joint_pos" without changing total_action_dim.
        am = getattr(self, "action_manager", None)
        if am is not None:
            for attr in ("_terms", "terms", "_term_dict"):
                term_dict = getattr(am, attr, None)
                if isinstance(term_dict, dict) and "joint_pos" in term_dict and "leg_joint_pos" in term_dict:
                    term_dict.pop("joint_pos", None)
            if hasattr(am, "_action"):
                am._action.pop("joint_pos", None)
            if hasattr(am, "_prev_action"):
                am._prev_action.pop("joint_pos", None)
            if hasattr(am, "_prev_prev_action"):
                am._prev_prev_action.pop("joint_pos", None)

            if hasattr(am, "get_term"):
                _orig_get_term = am.get_term

                def _get_term_compat(name: str):
                    if name == "joint_pos":
                        return _orig_get_term("leg_joint_pos")
                    return _orig_get_term(name)

                am.get_term = _get_term_compat

        # ✅ 여기 시점에는 managers가 로드된 상태여야 함(IsaacLab 기본 플로우 기준)
        self._resolve_action_joint_ids_and_buffers()

        # ✅ isaac joint order -> pin joint order permutation 생성(반드시 33개)
        self._build_joint_perm_isaac_to_pin()

        # lock non-controlled joints to default targets
        self._resolve_locked_joint_ids()

        # per-step timing (arm motion override)
        self._step_timing_count = 0
        self._step_timing_total_s = 0.0
        self._step_timing_grf_s = 0.0
        self._step_timing_footvel_s = 0.0
        self._step_timing_every = 50
        self._enable_step_timing = False
        self._debug_logging = render_mode is not None
        self._setup_scheduled_push()

    # ❌ 여기서 action_manager 만지면 안됨
    def _init_buffers(self):
        super()._init_buffers()
        self.leg_joint_ids = None
        self.arm_joint_ids = None
        self.action_full = None

        # logging용 RS 버퍼 (root pos 3 + euler 3 + joints 33)
        self.gen_coord_pin_RS = torch.zeros(self.num_envs, 6 + self.robot.num_joints, device=self.device)
        # scheduled push one-shot buffers (env-progress based)
        self._push_trigger_frac_1 = torch.zeros(self.num_envs, device=self.device)
        self._push_trigger_frac_2 = torch.zeros(self.num_envs, device=self.device)
        self._push_fired_1 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_fired_2 = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # +1/-1 sign to select first-foot convention per episode (for symmetric schedule rewards)
        self.first_foot_sign = torch.ones(self.num_envs, 1, device=self.device)

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        seed: int | None = None,
        options: dict | None = None,
        **kwargs,
    ):
        """Reset env and randomize phase start to 0 or 0.5 with equal probability."""
        out = super().reset(env_ids=env_ids, seed=seed, options=options, **kwargs)
        self._randomize_phase_start(env_ids)
        self._resample_joint_pos_obs_bias(env_ids)
        # Debug: print phase start for env 0 (play)
        try:
            phase0 = (self.episode_length_buf[0] % self.full_step_period[0]) / self.full_step_period[0]
            print(f"[PHASE_RESET] env0 phase={phase0.item():.3f}", flush=True)
        except Exception:
            pass
        return out

    def _reset_idx(self, env_ids: Sequence[int]):
        """Internal per-episode reset path used during training rollout."""
        super()._reset_idx(env_ids)
        self._randomize_phase_start(env_ids)
        self._reset_scheduled_push(env_ids)
        self._resample_joint_pos_obs_bias(env_ids)

    def _randomize_phase_start(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        half = (self.full_step_period[env_ids_t] // 2).to(torch.long).squeeze(-1)
        zeros = torch.zeros_like(half)
        choose_half = torch.rand(half.shape[0], device=self.device) < 0.5
        self.episode_length_buf[env_ids_t] = torch.where(choose_half, half, zeros)
        # Keep schedule sign neutral; phase(0/0.5) already determines first-foot order.
        self.first_foot_sign[env_ids_t, 0] = 1.0

    def _resample_joint_pos_obs_bias(self, env_ids: Sequence[int] | None):
        # Backward-compat: resample per-term bias states if they exist.
        if hasattr(self, "_obs_joint_pos_episode_bias"):
            bias_dict = getattr(self, "_obs_joint_pos_episode_bias", None)
            sigma_dict = getattr(self, "_obs_joint_pos_episode_bias_sigma", None)
            if isinstance(bias_dict, dict) and isinstance(sigma_dict, dict):
                if env_ids is None:
                    env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                else:
                    env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
                if env_ids_t.numel() > 0:
                    for key, bias in bias_dict.items():
                        sigma = float(sigma_dict.get(key, 0.0))
                        if sigma <= 0.0:
                            continue
                        if not isinstance(bias, torch.Tensor) or bias.ndim != 2 or bias.shape[0] != self.num_envs:
                            continue
                        bias[env_ids_t] = torch.randn((env_ids_t.numel(), bias.shape[1]), device=self.device) * sigma

        # Shared all-joint bias used by both observations and PD torque feedback.
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        enabled = bool(getattr(self.cfg, "enable_obs_joint_pos_episode_bias", False))
        sigma = float(getattr(self.cfg, "obs_joint_pos_episode_bias_sigma", 0.0))
        num_joints = int(self.robot.num_joints)
        if (not hasattr(self, "_joint_pos_episode_bias_all")
            or self._joint_pos_episode_bias_all.shape != (self.num_envs, num_joints)):
            self._joint_pos_episode_bias_all = torch.zeros((self.num_envs, num_joints), device=self.device)

        if enabled and sigma > 0.0:
            self._joint_pos_episode_bias_all[env_ids_t] = torch.randn((env_ids_t.numel(), num_joints), device=self.device) * sigma
        else:
            self._joint_pos_episode_bias_all[env_ids_t] = 0.0

    def _setup_scheduled_push(self):
        self._scheduled_push_enabled = bool(getattr(self.cfg, "enable_scheduled_push", False))
        self._push_force_range = tuple(getattr(self.cfg, "push_force_range", (-15.0, 15.0)))
        self._push_lateral_scale = float(getattr(self.cfg, "push_lateral_scale", 0.6))
        self._push_torque_range = tuple(getattr(self.cfg, "push_torque_range", (-1.5, 1.5)))
        self._push_window_1 = tuple(getattr(self.cfg, "push_window_1", (0.2, 0.4)))
        self._push_window_2 = tuple(getattr(self.cfg, "push_window_2", (0.6, 0.8)))
        # epoch-based curriculum for push strength
        self._push_curriculum_start_epoch = int(getattr(self.cfg, "push_curriculum_start_epoch", 3000))
        self._push_curriculum_end_epoch = int(getattr(self.cfg, "push_curriculum_end_epoch", 15000))
        self._push_curriculum_steps_per_epoch = max(int(getattr(self.cfg, "push_curriculum_steps_per_epoch", 24)), 1)

        # pulse half-span (center to edge) in ticks, sampled from closed interval [min, max].
        self._push_halfspan_min_steps = max(int(getattr(self.cfg, "push_pulse_halfspan_min_steps", 0)), 0)
        self._push_halfspan_max_steps = max(int(getattr(self.cfg, "push_pulse_halfspan_max_steps", 19)), self._push_halfspan_min_steps)

        self._push_body_ids = None
        try:
            body_ids, _ = self.robot.find_bodies("Pelvis_Link")
            if isinstance(body_ids, list) and len(body_ids) > 0:
                self._push_body_ids = [int(body_ids[0])]
            elif isinstance(body_ids, torch.Tensor) and body_ids.numel() > 0:
                self._push_body_ids = [int(body_ids.flatten()[0].item())]
        except Exception:
            self._push_body_ids = None

        num_push_bodies = len(self._push_body_ids) if self._push_body_ids is not None else 1

        # Per-window pulse parameters (sampled per-episode/env)
        self._push_center_step_1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._push_center_step_2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._push_duration_steps_1 = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._push_duration_steps_2 = torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        # Per-window impulse samples (N*s). Force is computed from impulse / (duration * dt).
        self._push_impulse_1 = torch.zeros((self.num_envs, num_push_bodies, 3), device=self.device)
        self._push_impulse_2 = torch.zeros((self.num_envs, num_push_bodies, 3), device=self.device)
        self._push_torque_impulse_1 = torch.zeros((self.num_envs, num_push_bodies, 3), device=self.device)
        self._push_torque_impulse_2 = torch.zeros((self.num_envs, num_push_bodies, 3), device=self.device)

    def _reset_scheduled_push(self, env_ids: Sequence[int] | None):
        if not getattr(self, "_scheduled_push_enabled", False):
            return
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        self._push_fired_1[env_ids_t] = False
        self._push_fired_2[env_ids_t] = False

        max_ep = max(int(self.max_episode_length), 1)
        w1_start = int(self._push_window_1[0] * max_ep)
        w1_end = max(w1_start + 1, int(self._push_window_1[1] * max_ep))  # exclusive
        w2_start = int(self._push_window_2[0] * max_ep)
        w2_end = max(w2_start + 1, int(self._push_window_2[1] * max_ep))  # exclusive

        num_push_bodies = self._push_impulse_1.shape[1]

        size = (env_ids_t.numel(), num_push_bodies, 3)
        self._push_impulse_1[env_ids_t] = math_utils.sample_uniform(*self._push_force_range, size, self.device)
        self._push_impulse_2[env_ids_t] = math_utils.sample_uniform(*self._push_force_range, size, self.device)
        # Elliptical XY push distribution: keep X-axis range, shrink Y-axis by configurable scale.
        self._push_impulse_1[env_ids_t, :, 1] *= self._push_lateral_scale
        self._push_impulse_2[env_ids_t, :, 1] *= self._push_lateral_scale
        self._push_torque_impulse_1[env_ids_t] = math_utils.sample_uniform(*self._push_torque_range, size, self.device)
        self._push_torque_impulse_2[env_ids_t] = math_utils.sample_uniform(*self._push_torque_range, size, self.device)

        # Sample per-window pulse center and duration.
        for env_id in env_ids_t.tolist():
            # window 1: sample half-span h uniformly from [min,max] (closed interval),
            # then use odd duration (2h+1) so left/right are exactly symmetric.
            win_len = max(w1_end - w1_start, 1)
            h_cap = (win_len - 1) // 2
            h_min = min(self._push_halfspan_min_steps, h_cap)
            h_max = min(self._push_halfspan_max_steps, h_cap)
            if h_max < h_min:
                h_max = h_min
            h = int(torch.randint(h_min, h_max + 1, (1,), device=self.device).item())
            dur = 2 * h + 1
            cmin = w1_start + h
            cmax = (w1_end - 1) - h
            center = int(torch.randint(cmin, cmax + 1, (1,), device=self.device).item())
            self._push_duration_steps_1[env_id] = dur
            self._push_center_step_1[env_id] = center

            # window 2
            win_len = max(w2_end - w2_start, 1)
            h_cap = (win_len - 1) // 2
            h_min = min(self._push_halfspan_min_steps, h_cap)
            h_max = min(self._push_halfspan_max_steps, h_cap)
            if h_max < h_min:
                h_max = h_min
            h = int(torch.randint(h_min, h_max + 1, (1,), device=self.device).item())
            dur = 2 * h + 1
            cmin = w2_start + h
            cmax = (w2_end - 1) - h
            center = int(torch.randint(cmin, cmax + 1, (1,), device=self.device).item())
            self._push_duration_steps_2[env_id] = dur
            self._push_center_step_2[env_id] = center

    def _apply_scheduled_push(self):
        if not getattr(self, "_scheduled_push_enabled", False):
            return
        if self._push_body_ids is None:
            return

        # Curriculum scale by policy epoch: 0 before start, linear ramp to 1 at end, then hold.
        curr_step = float(getattr(self, "common_step_counter", 0))
        curr_epoch = curr_step / float(self._push_curriculum_steps_per_epoch)
        if curr_epoch <= self._push_curriculum_start_epoch:
            push_scale = 0.0
        elif curr_epoch >= self._push_curriculum_end_epoch:
            push_scale = 1.0
        else:
            push_scale = (curr_epoch - self._push_curriculum_start_epoch) / float(
                self._push_curriculum_end_epoch - self._push_curriculum_start_epoch
            )

        num_bodies = self.robot.num_bodies
        body_ids = self._push_body_ids
        num_push_bodies = len(body_ids)

        ep_step = self.episode_length_buf

        # window-1 active mask based on sampled center/duration
        left1 = (self._push_duration_steps_1 - 1) // 2
        start1 = self._push_center_step_1 - left1
        end1 = start1 + self._push_duration_steps_1  # exclusive
        active1 = (~self._push_fired_1) & (ep_step >= start1) & (ep_step < end1)
        done1 = (~self._push_fired_1) & (ep_step >= end1)

        # window-2 active mask based on sampled center/duration
        left2 = (self._push_duration_steps_2 - 1) // 2
        start2 = self._push_center_step_2 - left2
        end2 = start2 + self._push_duration_steps_2  # exclusive
        active2 = (~self._push_fired_2) & (ep_step >= start2) & (ep_step < end2)
        done2 = (~self._push_fired_2) & (ep_step >= end2)

        external_force_b = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        external_torque_b = torch.zeros_like(external_force_b)

        if torch.any(active1):
            env_ids = torch.nonzero(active1, as_tuple=False).squeeze(-1)
            dur_s = (self._push_duration_steps_1[env_ids].float() * self.physics_dt).view(-1, 1, 1)
            forces = (self._push_impulse_1[env_ids] * push_scale) / dur_s
            torques = (self._push_torque_impulse_1[env_ids] * push_scale) / dur_s
            indices = torch.tensor(body_ids, dtype=torch.long, device=self.device).repeat(env_ids.numel(), 1)
            indices = indices + env_ids.unsqueeze(1) * num_bodies
            indices = indices.view(-1)
            external_force_b.flatten(0, 1)[indices] += forces.flatten(0, 1)
            external_torque_b.flatten(0, 1)[indices] += torques.flatten(0, 1)

        if torch.any(active2):
            env_ids = torch.nonzero(active2, as_tuple=False).squeeze(-1)
            dur_s = (self._push_duration_steps_2[env_ids].float() * self.physics_dt).view(-1, 1, 1)
            forces = (self._push_impulse_2[env_ids] * push_scale) / dur_s
            torques = (self._push_torque_impulse_2[env_ids] * push_scale) / dur_s
            indices = torch.tensor(body_ids, dtype=torch.long, device=self.device).repeat(env_ids.numel(), 1)
            indices = indices + env_ids.unsqueeze(1) * num_bodies
            indices = indices.view(-1)
            external_force_b.flatten(0, 1)[indices] += forces.flatten(0, 1)
            external_torque_b.flatten(0, 1)[indices] += torques.flatten(0, 1)

        if torch.any(torch.abs(external_force_b) > 0) or torch.any(torch.abs(external_torque_b) > 0):
            self.robot.root_physx_view.apply_forces_and_torques_at_position(
                force_data=external_force_b.view(-1, 3),
                torque_data=external_torque_b.view(-1, 3),
                position_data=None,
                indices=torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                is_global=False,
            )

        self._push_fired_1[done1] = True
        self._push_fired_2[done2] = True

    def _resolve_locked_joint_ids(self):
        joint_ids = []
        for expr_list in (ARM_LOCKED_JOINT_EXPR, TRUNK_JOINT_EXPR):
            ids, _ = self.robot.find_joints(expr_list, preserve_order=True)
            joint_ids.extend(ids)
        if len(joint_ids) == 0:
            raise RuntimeError("[locked] No joints matched ARM_LOCKED_JOINT_EXPR/TRUNK_JOINT_EXPR.")
        self.locked_joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    # ---------------------------
    # joint permutation builder
    # ---------------------------
    def _get_pin_joint_names(self) -> list[str]:
        # brl_utils에 export 돼있으면 그걸 우선 사용
        if hasattr(brl_utils, "PIN_JOINT_NAMES"):
            return list(brl_utils.PIN_JOINT_NAMES)

        # 아니면 constants.py에서 직접 가져오기
        from extensions.tocabi.utils.constants import PIN_JOINT_NAMES  # noqa
        return list(PIN_JOINT_NAMES)

    def _build_joint_perm_isaac_to_pin(self):
        isaac_names = list(self.robot.joint_names)  # IsaacLab/MJCF order (33)
        pin_names = self._get_pin_joint_names()     # 우리가 "pin order"로 정의한 리스트 (33)

        name_to_isaac = {n: i for i, n in enumerate(isaac_names)}

        missing = [n for n in pin_names if n not in name_to_isaac]
        if missing:
            raise RuntimeError(
                f"[joint_perm] PIN_JOINT_NAMES has names not in robot.joint_names: {missing}\n"
                f"robot.joint_names: {isaac_names}"
            )

        perm = [name_to_isaac[n] for n in pin_names]

        if len(perm) != len(isaac_names):
            raise RuntimeError(
                f"[joint_perm] length mismatch: perm={len(perm)} vs robot joints={len(isaac_names)}\n"
                f"PIN_JOINT_NAMES must include ALL actuated joints (same count as robot.joint_names)."
            )

        self.joint_perm_isaac_to_pin = torch.tensor(perm, device=self.device, dtype=torch.long)

    # ✅ managers 로드 이후에 호출될 함수
    def _resolve_action_joint_ids_and_buffers(self):
        leg_term = self._get_action_term_strict("leg_joint_pos")
        arm_term = self._get_action_term_strict("arm_joint_pos")

        leg_joint_ids = list(getattr(leg_term, "_joint_ids", []))
        arm_joint_ids = list(getattr(arm_term, "_joint_ids", []))
        if not leg_joint_ids:
            raise RuntimeError("[actions] leg_joint_pos term has no joint ids.")
        if not arm_joint_ids:
            raise RuntimeError("[actions] arm_joint_pos term has no joint ids.")

        self.leg_joint_ids = torch.tensor(leg_joint_ids, device=self.device, dtype=torch.long)
        self.arm_joint_ids = torch.tensor(arm_joint_ids, device=self.device, dtype=torch.long)

        # full target buffer (per-joint)
        self.action_full = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

    def _get_action_term_strict(self, name: str):
        am = getattr(self, "action_manager", None)
        if am is None:
            raise RuntimeError("action_manager is not initialized yet. Do not access it inside _init_buffers().")

        if hasattr(am, "get_term"):
            try:
                return am.get_term(name)
            except Exception:
                pass

        for attr in ("terms", "_terms", "_term_dict"):
            d = getattr(am, attr, None)
            if isinstance(d, dict) and name in d:
                return d[name]

        keys = []
        for attr in ("terms", "_terms", "_term_dict"):
            d = getattr(am, attr, None)
            if isinstance(d, dict):
                keys = sorted(list(d.keys()))
                break
        raise KeyError(f"Action term '{name}' not found. Available terms: {keys}")

    def _set_CasADi_urdf_name(self):
        self.urdf_name = "dyros_tocabi"

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboard(self)

    def _setup_recorder(self):
        self.recorder = TocabiFullVanillaRecorder(
            fps=int(1 / self.step_dt),
            total_weight=float(self.total_weight[0]),
            robot_cfg=self.robot.cfg,
        )

    def _post_physics_step_callback(self):
        t_step0 = time.perf_counter() if self._enable_step_timing else None
        self.progress_within_step = self.episode_length_buf.unsqueeze(1) % self.full_step_period
        self.phase = self.progress_within_step / self.full_step_period

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.contact_forces = self.contact_sensor.data.net_forces_w
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        self.contact_schedule = self.smooth_sqr_wave(self.phase)
        self.vel_command = self.command_manager.get_command("base_velocity")
        self._apply_scheduled_push()

        self._compute_generalized_coordinates()


        if self._enable_step_timing:
            t0 = time.perf_counter()
            self._calculate_GRF()
            self._step_timing_grf_s += time.perf_counter() - t0
        else:
            self._calculate_GRF()

        self._calculate_CoM()
        self._calculate_centroidal_momentum()

        if self._enable_step_timing:
            t1 = time.perf_counter()
            self._calculate_foot_velocity()
            self._step_timing_footvel_s += time.perf_counter() - t1
        else:
            self._calculate_foot_velocity()

        # NaN guardrail for debugging
        if torch.isnan(self.gen_coord_pin).any():
            raise RuntimeError("[NaN] gen_coord_pin has NaNs")
        if torch.isnan(self.gen_vel_body_pin).any():
            raise RuntimeError("[NaN] gen_vel_body_pin has NaNs")
        if torch.isnan(self.CM).any():
            raise RuntimeError("[NaN] CM has NaNs")

        if self._enable_step_timing:
            self._step_timing_total_s += time.perf_counter() - t_step0
            self._step_timing_count += 1
            if self._step_timing_count % self._step_timing_every == 0:
                cnt = self._step_timing_count
                total_ms = (self._step_timing_total_s / cnt) * 1000.0
                grf_ms = (self._step_timing_grf_s / cnt) * 1000.0
                foot_ms = (self._step_timing_footvel_s / cnt) * 1000.0
                carb.log_warn(
                    f"[STEP_TIMING] avg per step: total={total_ms:.3f} ms | GRF={grf_ms:.3f} ms | foot_vel={foot_ms:.3f} ms"
                )

        # debug: dCAM_xy magnitude (arm) every N steps
        if self._debug_logging:
            if not hasattr(self, "_dcam_debug_every"):
                self._dcam_debug_every = 200
                self._dcam_debug_count = 0
            self._dcam_debug_count += 1
            if self._dcam_debug_count % self._dcam_debug_every == 0:
                # CM_arm: [N,6] -> angular components [3:5] are CAM (x,y)
                dcam_xy = self.dCM_arm[:, 3:5]
                max_abs = dcam_xy.abs().max().item()
                mean_abs = dcam_xy.abs().mean().item()
                carb.log_warn(f"[DCAM_DEBUG] |dCAM_xy| mean={mean_abs:.3e} max={max_abs:.3e}")

    def _pre_physics_step_callback(self):
        # keep non-controlled arm joints locked at default positions (q_ref)
        locked_targets = self.robot.data.default_joint_pos[:, self.locked_joint_ids]
        self.robot.set_joint_position_target(locked_targets, joint_ids=self.locked_joint_ids)

    def _compute_generalized_coordinates(self):
        """
        ✅ 핵심 수정:
        - convert_* 함수들에 self.joint_perm_isaac_to_pin 전달
        - gen_acc는 vel 변환 함수로 돌리지 말고 안전하게 reorder만 수행
        - RS(euler)는 convert_gen_coord로 돌리지 말고 직접 구성
        """
        sim_nj = self.num_sim_joints
        casadi_nj = self.num_casadi_joints

        # ---------------- gen coord (world) ----------------
        gen_coord_sim = torch.hstack(
            (self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w, self.robot.data.joint_pos)
        )
        gen_coord_pin_sim = brl_utils.convert_gen_coord_from_isaaclab_to_pin(
            gen_coord_sim,
            self.joint_perm_isaac_to_pin,
        )

        # self.gen_coord_pin: (num_envs, 7 + casadi_nj) 로 vanilla에서 만들어졌다고 가정
        self.gen_coord_pin.zero_()
        n_fill = min(7 + sim_nj, 7 + casadi_nj)
        self.gen_coord_pin[:, :n_fill] = gen_coord_pin_sim[:, :n_fill]

        # ---------------- gen vel (body) ----------------
        gen_vel_body_sim = torch.hstack(
            (self.robot.data.root_link_lin_vel_b, self.robot.data.root_link_ang_vel_b, self.robot.data.joint_vel)
        )
        gen_vel_body_pin_sim = brl_utils.convert_gen_vel_from_isaaclab_to_pin(
            gen_vel_body_sim,
            self.joint_perm_isaac_to_pin,
        )

        self.gen_vel_body_pin.zero_()
        n_fill_v = min(self.num_base_joints + sim_nj, self.num_base_joints + casadi_nj)
        self.gen_vel_body_pin[:, :n_fill_v] = gen_vel_body_pin_sim[:, :n_fill_v]

        # ---------------- gen acc (body) ----------------
        # ⚠️ vel 변환 함수로 acc 넣지 말고, base(6) + joints reorder만 해서 안전하게 사용
        gen_acc_body_sim = torch.hstack(
            (self.robot.data.root_link_lin_acc_b, self.robot.data.root_link_ang_acc_b, self.robot.data.joint_acc)
        )
        base_acc = gen_acc_body_sim[:, : self.num_base_joints]
        joints_acc = gen_acc_body_sim[:, self.num_base_joints :]
        joints_acc_pin = joints_acc.index_select(1, self.joint_perm_isaac_to_pin)
        gen_acc_body_pin_sim = torch.hstack((base_acc, joints_acc_pin))

        self.gen_acc_body_pin.zero_()
        self.gen_acc_body_pin[:, :n_fill_v] = gen_acc_body_pin_sim[:, :n_fill_v]

        # desired base vel (pin body vel convention)
        self.gen_vel_body_pin_des.zero_()
        self.gen_vel_body_pin_des[:, 0] = self.vel_command[:, 0]
        self.gen_vel_body_pin_des[:, 1] = self.vel_command[:, 1]
        self.gen_vel_body_pin_des[:, 5] = self.vel_command[:, 2]

        # ---------------- logging RS (root pos + euler + joints_pin) ----------------
        root_euler = torch.stack(math_utils.euler_xyz_from_quat(self.robot.data.root_link_quat_w), dim=1)
        joints_pin = self.robot.data.joint_pos.index_select(1, self.joint_perm_isaac_to_pin)
        self.gen_coord_pin_RS[:] = torch.hstack((self.robot.data.root_link_pos_w, root_euler, joints_pin))

    def _get_log_data(self):
        robot_idx = 0

        self.action_full.zero_()
        # action_manager.action contains processed targets per term
        leg_action = self.action_manager.action["leg_joint_pos"]
        arm_action = self.action_manager.action["arm_joint_pos"]
        # start from defaults and overwrite controlled joints
        self.action_full[:, :] = self.robot.data.default_joint_pos
        self.action_full[:, self.leg_joint_ids] = leg_action
        self.action_full[:, self.arm_joint_ids] = arm_action

        q_pin = brl_utils.convert_joint_order_from_isaaclab_to_pin(
            self.robot.data.joint_pos,
            self.joint_perm_isaac_to_pin,
        )
        dq_pin = brl_utils.convert_joint_order_from_isaaclab_to_pin(
            self.robot.data.joint_vel,
            self.joint_perm_isaac_to_pin,
        )
        tau_pin = brl_utils.convert_joint_order_from_isaaclab_to_pin(
            self.robot.data.applied_torque,
            self.joint_perm_isaac_to_pin,
        )
        qdes_pin = brl_utils.convert_joint_order_from_isaaclab_to_pin(
            self.action_full,
            self.joint_perm_isaac_to_pin,
        )

        return {
            "root_lin_vel_b": self.robot.data.root_lin_vel_b[robot_idx],
            "root_ang_vel_b": self.robot.data.root_ang_vel_b[robot_idx],
            "vel_command": self.vel_command[robot_idx],

            "CM": self.CM[robot_idx],
            "CM_base": self.CM_base[robot_idx],
            "CM_leg": self.CM_leg[robot_idx],
            "CM_arm": self.CM_arm[robot_idx],
            "CM_des": self.CM_des[robot_idx],

            "dCM": self.dCM[robot_idx],
            "dCM_base": self.dCM_base[robot_idx],
            "dCM_leg": self.dCM_leg[robot_idx],
            "dCM_arm": self.dCM_arm[robot_idx],
            "dCM_des": self.dCM_des[robot_idx],

            "rf_toe_GRF": self.rf_toe_GRF[robot_idx],
            "rf_heel_GRF": self.rf_heel_GRF[robot_idx],
            "lf_toe_GRF": self.lf_toe_GRF[robot_idx],
            "lf_heel_GRF": self.lf_heel_GRF[robot_idx],
            "rf_GRM": self.rf_GRM[robot_idx],
            "lf_GRM": self.lf_GRM[robot_idx],

            "joint_pos_pin": q_pin[robot_idx],
            "joint_pos_target_pin": qdes_pin[robot_idx],
            "joint_vel_pin": dq_pin[robot_idx],
            "joint_torque_pin": tau_pin[robot_idx],

            "phase": self.phase[robot_idx],
            "foot_contact": self.foot_contact[robot_idx],
            "contact_schedule": self.contact_schedule[robot_idx],
            "foot_velocity": self.foot_velocity[robot_idx],

            "gen_coord": self.gen_coord_pin_RS[robot_idx],
            "gen_vel_body": self.gen_vel_body_pin[robot_idx],
        }

    # ---------------------------------------------------------
    # GRF: (너가 준 humanoid 원본 그대로) - 이 파일에 반드시 있어야 함
    # ---------------------------------------------------------
    # def _calculate_GRF(self):
    #     """Update foot contact forces"""
    #     if not self.has_GRF_sensor:
    #         return

    #     # * Update foot states
    #     rfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[0], :7]
    #     lfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[1], :7]
    #     rfoot_height_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_height_vec)
    #     rfoot_to_toe_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_to_toe_vec)
    #     rfoot_to_heel_vec_in_world = math_utils.quat_apply(rfoot_state[:, 3:7], self.foot_to_heel_vec)
    #     lfoot_height_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_height_vec)
    #     lfoot_to_toe_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_to_toe_vec)
    #     lfoot_to_heel_vec_in_world = math_utils.quat_apply(lfoot_state[:, 3:7], self.foot_to_heel_vec)

    #     self.foot_states[:, 0, :3] = rfoot_state[:, :3] + rfoot_height_vec_in_world
    #     self.foot_states[:, 0, 3:7] = rfoot_state[:, 3:7]
    #     self.foot_states[:, 1, :3] = lfoot_state[:, :3] + lfoot_height_vec_in_world
    #     self.foot_states[:, 1, 3:7] = lfoot_state[:, 3:7]
    #     self.toe_heel_pos[:, 0, :] = self.foot_states[:, 0, :3] + rfoot_to_toe_vec_in_world
    #     self.toe_heel_pos[:, 1, :] = self.foot_states[:, 0, :3] + rfoot_to_heel_vec_in_world
    #     self.toe_heel_pos[:, 2, :] = self.foot_states[:, 1, :3] + lfoot_to_toe_vec_in_world
    #     self.toe_heel_pos[:, 3, :] = self.foot_states[:, 1, :3] + lfoot_to_heel_vec_in_world

    #     # * Calculate GRF
    #     tol = 1e-2

    #     rel_idx = torch.arange(
    #         self.rf_contact_sensor.cfg.max_contact_data_count_per_env, device=self.device
    #     ).unsqueeze(0)
    #     rf_valid_mask = rel_idx < self.rf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)
    #     lf_valid_mask = rel_idx < self.lf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)

    #     rf_toe_diff = (self.toe_heel_pos[:, 0].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
    #     rf_toe_dist = rf_toe_diff.norm(dim=-1)
    #     rf_toe_mask = (rf_toe_dist < tol) * rf_valid_mask

    #     rf_heel_diff = (self.toe_heel_pos[:, 1].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
    #     rf_heel_dist = rf_heel_diff.norm(dim=-1)
    #     rf_heel_mask = (rf_heel_dist < tol) * rf_valid_mask

    #     lf_toe_diff = (self.toe_heel_pos[:, 2].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
    #     lf_toe_dist = lf_toe_diff.norm(dim=-1)
    #     lf_toe_mask = (lf_toe_dist < tol) * lf_valid_mask

    #     lf_heel_diff = (self.toe_heel_pos[:, 3].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
    #     lf_heel_dist = lf_heel_diff.norm(dim=-1)
    #     lf_heel_mask = (lf_heel_dist < tol) * lf_valid_mask

    #     self.rf_toe_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_toe_mask.unsqueeze(2)).sum(dim=1)
    #     self.rf_heel_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_heel_mask.unsqueeze(2)).sum(dim=1)
    #     self.lf_toe_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_toe_mask.unsqueeze(2)).sum(dim=1)
    #     self.lf_heel_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_heel_mask.unsqueeze(2)).sum(dim=1)

    #     # * GRM = (p_toe - p_foot) x GRF_toe + (p_heel - p_foot) x GRF_heel
    #     self.rf_GRM = torch.linalg.cross(rfoot_to_toe_vec_in_world, self.rf_toe_GRF) + torch.linalg.cross(
    #         rfoot_to_heel_vec_in_world, self.rf_heel_GRF
    #     )
    #     self.lf_GRM = torch.linalg.cross(lfoot_to_toe_vec_in_world, self.lf_toe_GRF) + torch.linalg.cross(
    #         lfoot_to_heel_vec_in_world, self.lf_heel_GRF
    #     )
