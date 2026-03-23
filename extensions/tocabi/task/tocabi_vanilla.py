from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

# ---- project utils ----
from ..utils import VanillaKeyboard, TocabiVanillaRecorder

# ---- tocabi utils ----
import extensions.tocabi.utils as brl_utils

# ---- casadi / cusadi ----
from casadi import Function
from cusadi import CusadiFunction

import pinocchio as pin

import time
import carb  # ✅ 추가

try:
    from extensions.tocabi.dynamics import PINOCCHIO_CASADI_FUNCTIONS_DIR
except Exception as e:
    raise ImportError(
        "extensions.tocabi.dynamics 에서 PINOCCHIO_CASADI_FUNCTIONS_DIR를 못 찾음."
    ) from e


class TocabiVanillaEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.num_base_joints = 6

        super().__init__(cfg, render_mode, **kwargs)

        # super 이후에만 가능한 것들
        self._resolve_action_joint_ids()

        # ✅ (중요) casadi/cusadi 로딩이 실제로 필요하면 여기서 반드시 호출
        # env 생성 시 1회만 호출되도록 가드
        if not hasattr(self, "_casadi_loaded"):
            self._casadi_loaded = False

        if not self._casadi_loaded:
            carb.log_warn("[TOCABI] calling _load_casadi_functions_and_infer_dims() ...")
            self._load_casadi_functions_and_infer_dims()
            self._casadi_loaded = True
            carb.log_warn("[TOCABI] casadi/cusadi load done.")

        carb.log_warn(f"[TOCABI] num_joints: {self.robot.num_joints}")
        # 너무 길면 로그가 터질 수 있으니 필요시만
        # carb.log_warn(f"[TOCABI] joint_names: {self.robot.joint_names}")

        # Cusadi timing (per-step accumulate + periodic print)
        self._cusadi_timing_count = 0
        self._cusadi_timing_com_s = 0.0
        self._cusadi_timing_cmm_s = 0.0
        self._cusadi_timing_every = 50
        self._cusadi_timing_sync = True

        # Per-step timing for other heavy sections
        self._step_timing_count = 0
        self._step_timing_total_s = 0.0
        self._step_timing_grf_s = 0.0
        self._step_timing_footvel_s = 0.0
        self._step_timing_every = 50

        # Disable sync/timing logs in headless training to avoid slowdowns.
        if render_mode is None:
            self._cusadi_timing_sync = False
            self._cusadi_timing_every = 1_000_000
            self._step_timing_every = 1_000_000
    # ---------------------------- internal helpers ----------------------------

    @staticmethod
    def _find_body_indices(name_list: list[str], candidates: list[str]) -> int:
        # 1) exact
        for cand in candidates:
            if cand in name_list:
                return name_list.index(cand)
        # 2) substring
        for i, nm in enumerate(name_list):
            for cand in candidates:
                if cand in nm:
                    return i
        raise ValueError(f"Cannot find any of candidates={candidates} in name_list={name_list[:80]}...")

    def _resolve_feet_names(self):
        body_names = list(self.robot.data.body_names)
        sensor_body_names = list(self.contact_sensor.body_names)

        right_candidates = ["R_Foot_Link", "R_Foot", "right_foot", "RightFoot", "r_foot", "R_AnkleRoll", "R_Ankle"]
        left_candidates  = ["L_Foot_Link", "L_Foot", "left_foot", "LeftFoot", "l_foot", "L_AnkleRoll", "L_Ankle"]

        r_body = self._find_body_indices(body_names, right_candidates)
        l_body = self._find_body_indices(body_names, left_candidates)
        r_sens = self._find_body_indices(sensor_body_names, right_candidates)
        l_sens = self._find_body_indices(sensor_body_names, left_candidates)

        self.body_feet_ids = [r_body, l_body]
        self.feet_ids = [r_sens, l_sens]

    def _resolve_action_joint_ids(self):
        """Resolve joint ids for leg/arm action terms without using has_term()."""

        def _get_term(name: str):
            am = getattr(self, "action_manager", None)
            if am is None:
                return None

            if hasattr(am, "get_term"):
                try:
                    return am.get_term(name)
                except Exception:
                    pass

            # IsaacLab 버전에 따라 내부 dict명이 다를 수 있음
            for attr in ("terms", "_terms", "_term_dict", "_term_configs"):
                d = getattr(am, attr, None)
                if isinstance(d, dict) and name in d:
                    return d[name]
            return None

        self.leg_joint_ids = None
        self.arm_joint_ids = None

        leg_term = _get_term("joint_pos")
        arm_term = _get_term("arm_joint_pos")

        if leg_term is not None:
            self.leg_joint_ids = leg_term._joint_ids
        if arm_term is not None:
            self.arm_joint_ids = arm_term._joint_ids

        # fallback: joint_pos 없고 leg_joint_pos 있으면 그걸 leg로 취급
        if self.leg_joint_ids is None:
            jp_term = _get_term("leg_joint_pos")
            if jp_term is not None:
                self.leg_joint_ids = jp_term._joint_ids

        self.num_base_joints = 6
        self.num_sim_joints = self.robot.num_joints
        self.num_leg_joints = 0 if self.leg_joint_ids is None else len(self.leg_joint_ids)
        self.num_arm_joints = 0 if self.arm_joint_ids is None else len(self.arm_joint_ids)

        self.action_full = torch.zeros(self.num_envs, self.num_sim_joints, device=self.device)

    # ---------------------------- CasADi loading (FIXED) ----------------------------

    def _set_CasADi_urdf_name(self):
        self.urdf_name = "dyros_tocabi"

    @staticmethod
    def _deserialize_function_from_file(p: Path) -> Function:
        b = p.read_bytes()
        fn = Function.deserialize(b.decode("latin1"))
        return fn

    def _load_casadi_functions_and_infer_dims(self):
        self._set_CasADi_urdf_name()

        casdir = Path(PINOCCHIO_CASADI_FUNCTIONS_DIR)

        def list_dir_hint() -> str:
            if not casdir.exists():
                return f"(casadi dir not found: {casdir})"
            files = sorted([x.name for x in casdir.glob("*.casadi")])
            return "available files:\n  - " + "\n  - ".join(files)

        def load_required(name: str) -> Function:
            p = casdir / f"{name}_{self.urdf_name}.casadi"
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing casadi file: {p}\n"
                    f"{list_dir_hint()}\n"
                    f"Fix: generate script must create *{name}_{self.urdf_name}.casadi*"
                )
            carb.log_warn(f"[TOCABI][CasADi] loading {p}")
            fn = self._deserialize_function_from_file(p)
            if fn.is_null():
                raise RuntimeError(
                    f"CasADi deserialize returned NULL: {p}\n"
                    f"Fix: file is corrupted OR not saved via Function.serialize() latin1 bytes."
                )
            return fn

        def load_optional(name: str) -> Function | None:
            p = casdir / f"{name}_{self.urdf_name}.casadi"
            if not p.exists():
                return None
            fn = self._deserialize_function_from_file(p)
            return None if fn.is_null() else fn

        # ---- required ----
        self.M_fn = load_required("M")
        self.CMM_fn = load_required("CMM")
        self.dCMM_fn = load_required("dCMM")
        self.CM_fn = load_required("CM")
        self.dCM_fn = load_required("dCM")
        self.CoM_fn = load_required("CoM")

        # ---- optional ----
        self.base_pos_fn = load_optional("base_pos")
        self.base_rot_fn = load_optional("base_rot")

        nq = int(self.M_fn.size1_in(0))
        if nq < 7:
            raise ValueError(f"Invalid casadi nq={nq}. M_{self.urdf_name}.casadi input size is weird.")

        self.num_casadi_joints = nq - 7
        self.num_casadi_total_vel = self.num_base_joints + self.num_casadi_joints

        # cusadi
        carb.log_warn(f"[TOCABI] creating CusadiFunction(s) | num_envs={self.num_envs} | precision=double")

        t0 = time.perf_counter()
        try:
            self.cusadi_CMM_fn  = CusadiFunction(self.CMM_fn,  num_instances=self.num_envs, precision="double")
            self.cusadi_dCMM_fn = CusadiFunction(self.dCMM_fn, num_instances=self.num_envs, precision="double")
            self.cusadi_CM_fn   = CusadiFunction(self.CM_fn,   num_instances=self.num_envs, precision="double")
            self.cusadi_dCM_fn  = CusadiFunction(self.dCM_fn,  num_instances=self.num_envs, precision="double")
            self.cusadi_CoM_fn  = CusadiFunction(self.CoM_fn,  num_instances=self.num_envs, precision="double")
        except Exception as e:
            carb.log_error("[TOCABI] CusadiFunction creation failed: " + repr(e))
            raise

        t1 = time.perf_counter()
        carb.log_warn(f"[TOCABI] CusadiFunction 생성 시간: {(t1 - t0):.4f} sec")

        # buffers sized by casadi model (not sim)
        self.gen_coord_pin = torch.zeros(self.num_envs, 7 + self.num_casadi_joints, device=self.device)
        self.gen_vel_body_pin = torch.zeros(self.num_envs, self.num_casadi_total_vel, device=self.device)
        self.gen_acc_body_pin = torch.zeros(self.num_envs, self.num_casadi_total_vel, device=self.device)
        self.gen_vel_body_pin_des = torch.zeros(self.num_envs, self.num_casadi_total_vel, device=self.device)
    # ---------------------------- joint-name permutations ----------------------------

    @staticmethod
    def _quat_wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
        # torchscript 문제(튜플 인덱싱) 피하려고 list 사용
        return q_wxyz[:, [1, 2, 3, 0]]

    def _prepare_pin_permutations(self):
        """Create joint order permutations: IsaacLab joint order <-> Pin joint order."""

        # --- 1) IsaacLab joint name list 얻기 ---
        if hasattr(self.robot, "data") and hasattr(self.robot.data, "joint_names"):
            isaac_joint_names = list(self.robot.data.joint_names)
        elif hasattr(self.robot, "joint_names"):
            isaac_joint_names = list(self.robot.joint_names)
        else:
            raise RuntimeError("Cannot find joint names from self.robot (joint_names).")

        # --- 2) Pin joint name list: 반드시 pin order로 준비되어 있어야 함 ---
        if not hasattr(brl_utils, "PIN_JOINT_NAMES"):
            raise RuntimeError(
                "brl_utils.PIN_JOINT_NAMES가 없음. "
                "extensions/tocabi/utils/constants.py 등에 PIN_JOINT_NAMES를 정의해서 export 해야 함."
            )
        pin_joint_names = list(brl_utils.PIN_JOINT_NAMES)
        if len(pin_joint_names) == 0:
            raise RuntimeError(
                "brl_utils.PIN_JOINT_NAMES가 비어있음. "
                "pinocchio(URDF) 기준 actuated joint name 순서를 채워야 함."
            )

        # --- 3) permutation 만들기 ---
        isaac_map = {n: i for i, n in enumerate(isaac_joint_names)}

        pin_to_isaac = []
        missing = []
        for n in pin_joint_names:
            if n not in isaac_map:
                missing.append(n)
            else:
                pin_to_isaac.append(isaac_map[n])
        if len(missing) > 0:
            raise RuntimeError(f"PIN_JOINT_NAMES 중 IsaacLab joint에 없는 이름: {missing}")

        # isaac -> pin (length = pin joints).  isaac_vec.index_select(1, perm) 하면 pin order 됨.
        self.joint_perm_isaac_to_pin = torch.tensor(pin_to_isaac, device=self.device, dtype=torch.long)

        # pin -> isaac (length = pin joints).  pin_vec.index_select(1, inv) 같은 건 보통 필요없고,
        # 필요하면 scatter로 복원하는 게 맞음. 여기서는 “pin index -> isaac index”를 그대로 저장.
        self.joint_perm_pin_to_isaac = torch.tensor(pin_to_isaac, device=self.device, dtype=torch.long)

        # convenience
        self.num_pin_joints = len(pin_joint_names)

    def _joints_isaac_to_pin(self, x_isaac: torch.Tensor) -> torch.Tensor:
        """x_isaac: [N, sim_nj] -> [N, pin_nj]"""
        return x_isaac.index_select(1, self.joint_perm_isaac_to_pin)

    # ---------------------------- overrides ----------------------------

    def _init_buffers(self):
        super()._init_buffers()

        # managers/sensors는 super 이후에만 안전
        self.contact_sensor = self.scene.sensors["contact_forces"]
        self.rf_contact_sensor = self.scene.sensors.get("rf_GRF", None)
        self.lf_contact_sensor = self.scene.sensors.get("lf_GRF", None)
        self.has_GRF_sensor = self.rf_contact_sensor is not None and self.lf_contact_sensor is not None

        # feet resolve needs contact_sensor + robot bodies
        self._resolve_feet_names()

        # pin permutations needs robot joint names
        self._prepare_pin_permutations()

        self.phase = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase_sin = torch.zeros(self.num_envs, 1, device=self.device)
        self.phase_cos = torch.zeros(self.num_envs, 1, device=self.device)
        self.contact_schedule = torch.zeros(self.num_envs, 1, device=self.device)

        self.foot_contact = torch.zeros(self.num_envs, len(self.feet_ids), dtype=torch.bool, device=self.device)
        self.foot_states = torch.zeros(self.num_envs, len(self.feet_ids), 7, device=self.device)
        self.foot_velocity = torch.zeros(self.num_envs, len(self.feet_ids), 3, device=self.device)
        self.toe_heel_pos = torch.zeros(self.num_envs, 2 * len(self.feet_ids), 3, device=self.device)

        self.foot_height_vec = torch.tensor([0.0, 0.0, -0.04], device=self.device).repeat(self.num_envs, 1)
        self.foot_to_toe_vec = torch.tensor([0.105, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.foot_to_heel_vec = torch.tensor([-0.045, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.vel_command = torch.zeros(self.num_envs, 3, device=self.device)

        self.rf_toe_GRF = torch.zeros(self.num_envs, 3, device=self.device)
        self.rf_heel_GRF = torch.zeros(self.num_envs, 3, device=self.device)
        self.lf_toe_GRF = torch.zeros(self.num_envs, 3, device=self.device)
        self.lf_heel_GRF = torch.zeros(self.num_envs, 3, device=self.device)
        self.rf_GRM = torch.zeros(self.num_envs, 3, device=self.device)
        self.lf_GRM = torch.zeros(self.num_envs, 3, device=self.device)

        if self.cfg.commands.step_command.step_period_range is not None:
            lo, hi = self.cfg.commands.step_command.step_period_range
            step_vals = torch.linspace(lo, hi, self.num_envs, device=self.device)
            step_vals = torch.round(step_vals).clamp(min=1).to(torch.long)
            self.step_period = step_vals.unsqueeze(1)
        else:
            if self.cfg.commands.step_command.step_period is None:
                raise ValueError("step_period_range or step_period must be set.")
            self.step_period = self.cfg.commands.step_command.step_period * torch.ones(
                self.num_envs, 1, device=self.device, dtype=torch.long
            )
        self.full_step_period = 2 * self.step_period.clone()

        self.CoM = torch.zeros(self.num_envs, 3, device=self.device)

        self.CM = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_base = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_leg = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_arm = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_des = torch.zeros(self.num_envs, 6, device=self.device)

        self.dCM = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_base = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_leg = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_arm = torch.zeros(self.num_envs, 6, device=self.device)

        self.rot_matrix_block = torch.zeros(self.num_envs, 6, 6, device=self.device)

        self.CM_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_base_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_leg_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_arm_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.CM_des_bf = torch.zeros(self.num_envs, 6, device=self.device)

        self.dCM_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_base_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_leg_bf = torch.zeros(self.num_envs, 6, device=self.device)
        self.dCM_arm_bf = torch.zeros(self.num_envs, 6, device=self.device)

        # casadi funcs + dims
        self._load_casadi_functions_and_infer_dims()

    def _post_physics_step_callback(self):
        t_step0 = time.perf_counter()
        self.progress_within_step = self.episode_length_buf.unsqueeze(1) % self.full_step_period
        self.phase = self.progress_within_step / self.full_step_period

        self.phase_sin = torch.sin(2 * torch.pi * self.phase)
        self.phase_cos = torch.cos(2 * torch.pi * self.phase)

        self.contact_forces = self.contact_sensor.data.net_forces_w
        self.foot_contact = torch.gt(self.contact_forces[:, self.feet_ids, 2], 0)

        self.contact_schedule = self.smooth_sqr_wave(self.phase)
        self.vel_command = self.command_manager.get_command("base_velocity")

        self._compute_generalized_coordinates()

        t0 = time.perf_counter()
        self._calculate_GRF()
        self._step_timing_grf_s += time.perf_counter() - t0

        self._calculate_CoM()
        self._calculate_centroidal_momentum()

        t1 = time.perf_counter()
        self._calculate_foot_velocity()
        self._step_timing_footvel_s += time.perf_counter() - t1

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

    def _compute_generalized_coordinates(self):
        """Build q,v,a in Pin convention WITHOUT assuming 17/25 sizes."""
        sim_nj = self.num_sim_joints
        casadi_nj = self.num_casadi_joints

        # ----- q (Pin): [pos(3), quat_xyzw(4), joints(pin_order)] -----
        q_pos = self.robot.data.root_link_pos_w.clone()
        q_pos[:, 2] = 0.91
        q_quat_xyzw = self._quat_wxyz_to_xyzw(self.robot.data.root_link_quat_w)  # Isaac: wxyz -> xyzw
        q_joints_pin = self._joints_isaac_to_pin(self.robot.data.joint_pos)

        gen_coord_pin_sim = torch.hstack((q_pos, q_quat_xyzw, q_joints_pin))

        self.gen_coord_pin.zero_()
        n_fill = min(7 + self.num_pin_joints, 7 + casadi_nj)
        self.gen_coord_pin[:, :n_fill] = gen_coord_pin_sim[:, :n_fill]

        # ----- v (Pin): [v(3), w(3), dq(pin_order)] -----
        v_lin = self.robot.data.root_link_lin_vel_b
        v_ang = self.robot.data.root_link_ang_vel_b
        dq_pin = self._joints_isaac_to_pin(self.robot.data.joint_vel)

        gen_vel_body_pin_sim = torch.hstack((v_lin, v_ang, dq_pin))

        self.gen_vel_body_pin.zero_()
        n_fill_v = min(self.num_base_joints + self.num_pin_joints, self.num_base_joints + casadi_nj)
        self.gen_vel_body_pin[:, :n_fill_v] = gen_vel_body_pin_sim[:, :n_fill_v]

        # ----- a (Pin): [a_lin(3), a_ang(3), ddq(pin_order)] -----
        a_lin = self.robot.data.root_link_lin_acc_b
        a_ang = self.robot.data.root_link_ang_acc_b
        ddq_pin = self._joints_isaac_to_pin(self.robot.data.joint_acc)

        gen_acc_body_pin_sim = torch.hstack((a_lin, a_ang, ddq_pin))

        self.gen_acc_body_pin.zero_()
        self.gen_acc_body_pin[:, :n_fill_v] = gen_acc_body_pin_sim[:, :n_fill_v]

        # desired
        self.gen_vel_body_pin_des.zero_()
        self.gen_vel_body_pin_des[:, 0] = self.vel_command[:, 0]
        self.gen_vel_body_pin_des[:, 1] = self.vel_command[:, 1]
        self.gen_vel_body_pin_des[:, 5] = self.vel_command[:, 2]

    def _calculate_GRF(self):
        """ Update foot contact forces """
        if not self.has_GRF_sensor:
            return
        
        # * Update foot states
        rfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[0], :7]
        lfoot_state = self.robot.data.body_state_w[:, self.body_feet_ids[1], :7]
        rfoot_height_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_height_vec)
        rfoot_to_toe_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_to_toe_vec)
        rfoot_to_heel_vec_in_world = math_utils.quat_apply(rfoot_state[:,3:7], self.foot_to_heel_vec)
        lfoot_height_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_height_vec)
        lfoot_to_toe_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_to_toe_vec)
        lfoot_to_heel_vec_in_world = math_utils.quat_apply(lfoot_state[:,3:7], self.foot_to_heel_vec)

        self.foot_states[:,0,:3] = rfoot_state[:,:3] + rfoot_height_vec_in_world
        self.foot_states[:,0,3:7] = rfoot_state[:,3:7]
        self.foot_states[:,1,:3] = lfoot_state[:,:3] + lfoot_height_vec_in_world
        self.foot_states[:,1,3:7] = lfoot_state[:,3:7]
        self.toe_heel_pos[:,0,:] = self.foot_states[:,0,:3] + rfoot_to_toe_vec_in_world
        self.toe_heel_pos[:,1,:] = self.foot_states[:,0,:3] + rfoot_to_heel_vec_in_world
        self.toe_heel_pos[:,2,:] = self.foot_states[:,1,:3] + lfoot_to_toe_vec_in_world
        self.toe_heel_pos[:,3,:] = self.foot_states[:,1,:3] + lfoot_to_heel_vec_in_world

        # * Calculate GRF
        tol = 1e-2

        rel_idx = torch.arange(self.rf_contact_sensor.cfg.max_contact_data_count_per_env, device=self.device).unsqueeze(0)  # Shape: (1, max_contacts)
        rf_valid_mask = rel_idx < self.rf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)
        lf_valid_mask = rel_idx < self.lf_contact_sensor.data.GRF_count_buffer.unsqueeze(1)

        rf_toe_diff = (self.toe_heel_pos[:,0].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
        rf_toe_dist = rf_toe_diff.norm(dim=-1)
        rf_toe_mask = (rf_toe_dist < tol) * rf_valid_mask
        rf_heel_diff = (self.toe_heel_pos[:,1].unsqueeze(1) - self.rf_contact_sensor.data.GRF_points_buffer)
        rf_heel_dist = rf_heel_diff.norm(dim=-1)
        rf_heel_mask = (rf_heel_dist < tol) * rf_valid_mask
        lf_toe_diff = (self.toe_heel_pos[:,2].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
        lf_toe_dist = lf_toe_diff.norm(dim=-1)
        lf_toe_mask = (lf_toe_dist < tol) * lf_valid_mask
        lf_heel_diff = (self.toe_heel_pos[:,3].unsqueeze(1) - self.lf_contact_sensor.data.GRF_points_buffer)
        lf_heel_dist = lf_heel_diff.norm(dim=-1)
        lf_heel_mask = (lf_heel_dist < tol) * lf_valid_mask

        self.rf_toe_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_toe_mask.unsqueeze(2)).sum(dim=1)
        self.rf_heel_GRF = (self.rf_contact_sensor.data.GRF_forces_buffer * rf_heel_mask.unsqueeze(2)).sum(dim=1)
        self.lf_toe_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_toe_mask.unsqueeze(2)).sum(dim=1)
        self.lf_heel_GRF = (self.lf_contact_sensor.data.GRF_forces_buffer * lf_heel_mask.unsqueeze(2)).sum(dim=1)

        # * Calculate the Ground Reaction Moment (GRM) = (p_toe - p_foot) x GRF_toe + (p_heel - p_foot) x GRF_heel
        self.rf_GRM = torch.linalg.cross(rfoot_to_toe_vec_in_world, self.rf_toe_GRF) + torch.linalg.cross(rfoot_to_heel_vec_in_world, self.rf_heel_GRF)
        self.lf_GRM = torch.linalg.cross(lfoot_to_toe_vec_in_world, self.lf_toe_GRF) + torch.linalg.cross(lfoot_to_heel_vec_in_world, self.lf_heel_GRF)

    def _calculate_CoM(self):
        """ Calculate the CoM position of the robot """

        # (num_envs, 40) 가 들어가야 함 (CusadiFunction num_instances = num_envs)
        x = self.gen_coord_pin
        # Cusadi built in double precision expects double inputs
        if getattr(self.cusadi_CoM_fn, "precision", "double") == "double" and x.dtype != torch.double:
            x = x.to(torch.double)

        # 안전장치
        assert x.is_cuda, f"CoM input must be CUDA tensor, got {x.device}"
        if not x.is_contiguous():
            x = x.contiguous()

        # nnz_in=40 확인했으니 고정
        assert x.dim() == 2, f"expect (num_envs, dof) but got {tuple(x.shape)}"
        assert x.shape[0] == self.num_envs, f"x.shape[0] must be num_envs={self.num_envs}, got {x.shape[0]}"
        assert x.shape[1] == 40, f"CoM expects 40, got {x.shape[1]}"

        # 입력은 "리스트에 텐서 1개"로 batch 통째로 넣는다
        if self._cusadi_timing_sync and x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.cusadi_CoM_fn.evaluate([x])
        if self._cusadi_timing_sync and x.is_cuda:
            torch.cuda.synchronize()
        self._cusadi_timing_com_s += time.perf_counter() - t0
        # outputs_sparse[0] shape: (num_envs, nnz_out=3)
        self.CoM = self.cusadi_CoM_fn.outputs_sparse[0].to(self.device).float()

    def _calculate_centroidal_momentum(self):
        """ Calculate the centroidal momelntum of the robot
            IsaacLab's mass matrix is projected to the root link's CoM frame
            We need to convert it to whole body's CoM frame
            Convert from root_link_CoM (world frame) -> root_link (base frame) -> CoM (world frame)
        """

        # ---- 1) Cusadi 입력은 self.device로 통일 + contiguous 보장 ----
        q = self.gen_coord_pin
        if q.device != self.device:
            q = q.to(self.device)
        if not q.is_contiguous():
            q = q.contiguous()

        v = self.gen_vel_body_pin
        if v.device != self.device:
            v = v.to(self.device)
        if not v.is_contiguous():
            v = v.contiguous()

        a = self.gen_acc_body_pin
        if a.device != self.device:
            a = a.to(self.device)
        if not a.is_contiguous():
            a = a.contiguous()

        v_des = self.gen_vel_body_pin_des
        if v_des.device != self.device:
            v_des = v_des.to(self.device)
        if not v_des.is_contiguous():
            v_des = v_des.contiguous()

        # ---- 1.6) Cusadi expects double inputs when built in double precision ----
        if getattr(self.cusadi_CMM_fn, "precision", "double") == "double":
            if q.dtype != torch.double:
                q = q.to(torch.double)
            if v.dtype != torch.double:
                v = v.to(torch.double)

        # ---- 1.5) Debug: report extreme state once to diagnose blow-ups ----
        if not hasattr(self, "_state_debug_count"):
            self._state_debug_count = 0
        if self._state_debug_count < 2:
            q_abs = q.abs()
            v_abs = v.abs()
            a_abs = a.abs()
            q_max = float(q_abs.max().item()) if q_abs.numel() > 0 else 0.0
            v_max = float(v_abs.max().item()) if v_abs.numel() > 0 else 0.0
            a_max = float(a_abs.max().item()) if a_abs.numel() > 0 else 0.0

            if (not torch.isfinite(q).all()) or (not torch.isfinite(v).all()) or (not torch.isfinite(a).all()) or q_max > 50.0 or v_max > 100.0 or a_max > 1000.0:
                # map max indices to joint names where possible
                pin_joint_names = list(getattr(brl_utils, "PIN_JOINT_NAMES", []))
                base_q_names = ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
                base_v_names = ["base_vx", "base_vy", "base_vz", "base_wx", "base_wy", "base_wz"]

                q_flat_idx = int(q_abs.view(-1).argmax().item()) if q_abs.numel() > 0 else 0
                v_flat_idx = int(v_abs.view(-1).argmax().item()) if v_abs.numel() > 0 else 0
                a_flat_idx = int(a_abs.view(-1).argmax().item()) if a_abs.numel() > 0 else 0

                q_env = q_flat_idx // q.shape[1]
                q_idx = q_flat_idx % q.shape[1]
                v_env = v_flat_idx // v.shape[1]
                v_idx = v_flat_idx % v.shape[1]
                a_env = a_flat_idx // a.shape[1]
                a_idx = a_flat_idx % a.shape[1]

                if q_idx < 7:
                    q_name = base_q_names[q_idx] if q_idx < len(base_q_names) else f"base_q[{q_idx}]"
                else:
                    j = q_idx - 7
                    q_name = pin_joint_names[j] if j < len(pin_joint_names) else f"joint[{j}]"

                if v_idx < 6:
                    v_name = base_v_names[v_idx] if v_idx < len(base_v_names) else f"base_v[{v_idx}]"
                else:
                    j = v_idx - 6
                    v_name = pin_joint_names[j] if j < len(pin_joint_names) else f"joint[{j}]"

                if a_idx < 6:
                    a_name = base_v_names[a_idx] if a_idx < len(base_v_names) else f"base_a[{a_idx}]"
                else:
                    j = a_idx - 6
                    a_name = pin_joint_names[j] if j < len(pin_joint_names) else f"joint[{j}]"

                carb.log_warn(
                    f"[STATE_DEBUG] max|q|={q_max:.3e} at env={q_env}, idx={q_idx} ({q_name}) | "
                    f"max|v|={v_max:.3e} at env={v_env}, idx={v_idx} ({v_name}) | "
                    f"max|a|={a_max:.3e} at env={a_env}, idx={a_idx} ({a_name})"
                )

                # action scale quick peek (if available)
                action_dict = getattr(getattr(self, "action_manager", None), "action", None)
                if isinstance(action_dict, dict):
                    action_stats = []
                    for k, v_act in action_dict.items():
                        if torch.is_tensor(v_act):
                            action_stats.append(f"{k}:max|a|={float(v_act.abs().max().item()):.3e}")
                    if action_stats:
                        carb.log_warn("[STATE_DEBUG] action " + " | ".join(action_stats))

                self._state_debug_count += 1

        # ---- 2) Cusadi evaluate (여기서 나오는 출력 dtype이 기준이 됨: 보통 double) ----
        if self._cusadi_timing_sync and q.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # ensure outputs are zeroed to avoid stale values when kernels underwrite
        try:
            self.cusadi_CMM_fn.outputs_sparse[0].zero_()
        except Exception:
            pass
        self.cusadi_CMM_fn.evaluate([q])
        CMM = (
            self.cusadi_CMM_fn.outputs_sparse[0]
            .reshape(self.num_envs, -1, self.num_base_joints)
            .permute(0, 2, 1)
            .to(self.device)
        )

        try:
            self.cusadi_dCMM_fn.outputs_sparse[0].zero_()
        except Exception:
            pass
        self.cusadi_dCMM_fn.evaluate([q, v])
        if self._cusadi_timing_sync and q.is_cuda:
            torch.cuda.synchronize()
        self._cusadi_timing_cmm_s += time.perf_counter() - t0
        dCMM = (
            self.cusadi_dCMM_fn.outputs_sparse[0]
            .reshape(self.num_envs, -1, self.num_base_joints)
            .permute(0, 2, 1)
            .to(self.device)
        )

        # NaN/Inf guardrails for CM inputs/outputs
        if not torch.isfinite(q).all():
            raise RuntimeError("[NaN/Inf] gen_coord_pin has non-finite values")
        if not torch.isfinite(v).all():
            raise RuntimeError("[NaN/Inf] gen_vel_body_pin has non-finite values")
        if not torch.isfinite(a).all():
            raise RuntimeError("[NaN/Inf] gen_acc_body_pin has non-finite values")
        if not torch.isfinite(CMM).all():
            raise RuntimeError("[NaN/Inf] CMM has non-finite values")
        if not torch.isfinite(dCMM).all():
            raise RuntimeError("[NaN/Inf] dCMM has non-finite values")

        # Fallback: if some envs have absurd CMM, recompute those envs with casadi and overwrite
        cmm_abs_env = CMM.abs().amax(dim=(1, 2))
        if cmm_abs_env.numel() > 0 and float(cmm_abs_env.max().item()) > 1.0e6:
            if not hasattr(self, "_cmm_fallback_count"):
                self._cmm_fallback_count = 0
            bad_idx = (cmm_abs_env > 1.0e6).nonzero(as_tuple=False).squeeze(1)
            # always include the worst env, then take a batch of bad envs
            max_env = int(cmm_abs_env.argmax().item())
            idx_list = [max_env] + bad_idx.tolist()
            # unique, keep order
            seen = set()
            uniq_idx = []
            for idx in idx_list:
                if idx not in seen:
                    uniq_idx.append(idx)
                    seen.add(idx)
            # limit work per step
            for idx in uniq_idx[:32]:
                try:
                    q_i = q[idx].detach().cpu().numpy()
                    v_i = v[idx].detach().cpu().numpy()
                    cmm_cpu = np.array(self.CMM_fn(q_i))
                    dcmm_cpu = np.array(self.dCMM_fn(q_i, v_i))
                    CMM[idx] = torch.as_tensor(cmm_cpu, device=CMM.device, dtype=CMM.dtype)
                    dCMM[idx] = torch.as_tensor(dcmm_cpu, device=dCMM.device, dtype=dCMM.dtype)
                    self._cmm_fallback_count += 1
                    carb.log_warn(f"[CMM_FALLBACK] env={idx} replaced cusadi CMM/dCMM with casadi output")
                except Exception as e:
                    carb.log_warn(f"[CMM_FALLBACK] env={idx} casadi fallback failed: {e!r}")

        # Debug: track extreme CMM across envs (helps isolate a single bad env)
        if not hasattr(self, "_cmm_env_debug_count"):
            self._cmm_env_debug_count = 0
        if self._cmm_env_debug_count < 2:
            cmm_abs_env = CMM.abs().amax(dim=(1, 2))
            cmm_max = float(cmm_abs_env.max().item()) if cmm_abs_env.numel() > 0 else 0.0
            if cmm_max > 1.0e6:
                env_idx = int(cmm_abs_env.argmax().item())
                q_env = q[env_idx].abs()
                v_env = v[env_idx].abs()
                a_env = a[env_idx].abs()
                q_max = float(q_env.max().item()) if q_env.numel() > 0 else 0.0
                v_max = float(v_env.max().item()) if v_env.numel() > 0 else 0.0
                a_max = float(a_env.max().item()) if a_env.numel() > 0 else 0.0

                # map q max index to joint name
                pin_joint_names = list(getattr(brl_utils, "PIN_JOINT_NAMES", []))
                q_idx = int(q_env.argmax().item()) if q_env.numel() > 0 else 0
                if q_idx < 7:
                    q_name = ["root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"][q_idx]
                else:
                    j = q_idx - 7
                    q_name = pin_joint_names[j] if j < len(pin_joint_names) else f"joint[{j}]"

                # compare cusadi vs casadi for the problematic env (CPU)
                try:
                    q_env0 = q[env_idx].detach().cpu().numpy()
                    cmm_cpu = np.array(self.CMM_fn(q_env0))
                    cmm_cpu_abs = float(np.max(np.abs(cmm_cpu)))
                    cmm_gpu_abs = float(CMM[env_idx].abs().max().item())
                    carb.log_warn(
                        f"[CMM_ENV_DEBUG] env={env_idx} max|CMM|={cmm_max:.3e} | "
                        f"cusadi={cmm_gpu_abs:.3e}, casadi={cmm_cpu_abs:.3e} | "
                        f"max|q|={q_max:.3e} ({q_name}) | max|v|={v_max:.3e} | max|a|={a_max:.3e}"
                    )
                except Exception as e:
                    carb.log_warn(
                        f"[CMM_ENV_DEBUG] env={env_idx} max|CMM|={cmm_max:.3e} | "
                        f"max|q|={q_max:.3e} ({q_name}) | max|v|={v_max:.3e} | max|a|={a_max:.3e} | "
                        f"casadi compare failed: {e!r}"
                    )
                self._cmm_env_debug_count += 1

        # One-time CMM sanity check: compare cusadi vs casadi (CPU)
        if not hasattr(self, "_cmm_debugged"):
            self._cmm_debugged = True
            try:
                q0 = q[0].detach().cpu().numpy()
                cmm_cpu = np.array(self.CMM_fn(q0))
                cmm_cpu_abs = float(np.max(np.abs(cmm_cpu)))
                cmm_gpu_abs = float(CMM[0].abs().max().item())
                carb.log_warn(
                    f"[TOCABI][CMM_DEBUG] max|CMM| cusadi={cmm_gpu_abs:.3e}, casadi={cmm_cpu_abs:.3e}"
                )
            except Exception as e:
                carb.log_warn(f"[TOCABI][CMM_DEBUG] casadi compare failed: {e!r}")

        # ---- 3) 앞으로 matmul 터지는 원인: dtype mismatch 방지 ----
        # CMM/dCMM dtype(대개 float64)에 맞춰서 v, a, v_des를 캐스팅
        dtype = CMM.dtype
        v_d     = v.to(dtype=dtype)
        a_d     = a.to(dtype=dtype)
        vdes_d  = v_des.to(dtype=dtype)

        # ---- 4) CM / dCM 계산 (연산은 dtype로, 최종 저장은 float32로 통일) ----
        CM_d = (CMM @ v_d.unsqueeze(2)).squeeze(2)
        self.CM = CM_d.to(torch.float32)
        if not torch.isfinite(self.CM).all():
            cmm_abs = CMM.abs().max().item()
            v_abs = v_d.abs().max().item()
            a_abs = a_d.abs().max().item()
            cm_abs = CM_d.abs().max().item()
            q_abs = q.abs().max().item()
            # extra debug: compare cusadi vs casadi CMM when CM explodes
            try:
                q0 = q[0].detach().cpu().numpy()
                cmm_cpu = np.array(self.CMM_fn(q0))
                cmm_cpu_abs = float(np.max(np.abs(cmm_cpu)))
                cmm_gpu_abs = float(CMM[0].abs().max().item())
                carb.log_warn(
                    f"[TOCABI][CMM_DEBUG_ERR] max|CMM| cusadi={cmm_gpu_abs:.3e}, casadi={cmm_cpu_abs:.3e}"
                )
            except Exception as e:
                carb.log_warn(f"[TOCABI][CMM_DEBUG_ERR] casadi compare failed: {e!r}")
            raise RuntimeError(
                "[NaN/Inf] CM has non-finite values "
                f"(max|CMM|={cmm_abs:.3e}, max|v|={v_abs:.3e}, max|a|={a_abs:.3e}, "
                f"max|CM|={cm_abs:.3e}, max|q|={q_abs:.3e})"
            )

        # base/leg/arm도 동일 dtype로 계산해야 안전함
        nb = self.num_base_joints
        nl = self.num_leg_joints

        CM_base_d = (CMM[:, :, :nb] @ v_d[:, :nb].unsqueeze(2)).squeeze(2)
        CM_leg_d  = (CMM[:, :, nb:nb+nl] @ v_d[:, nb:nb+nl].unsqueeze(2)).squeeze(2)
        CM_arm_d  = (CMM[:, :, nb+nl:] @ v_d[:, nb+nl:].unsqueeze(2)).squeeze(2)
        CM_des_d  = (CMM @ vdes_d.unsqueeze(2)).squeeze(2)

        self.CM_base = CM_base_d.to(torch.float32)
        self.CM_leg  = CM_leg_d.to(torch.float32)
        self.CM_arm  = CM_arm_d.to(torch.float32)
        self.CM_des  = CM_des_d.to(torch.float32)

        if not torch.isfinite(self.CM).all():
            raise RuntimeError("[NaN/Inf] CM has non-finite values")
        if not torch.isfinite(self.CM_base).all():
            raise RuntimeError("[NaN/Inf] CM_base has non-finite values")
        if not torch.isfinite(self.CM_leg).all():
            raise RuntimeError("[NaN/Inf] CM_leg has non-finite values")
        if not torch.isfinite(self.CM_arm).all():
            raise RuntimeError("[NaN/Inf] CM_arm has non-finite values")
        if not torch.isfinite(self.CM_des).all():
            raise RuntimeError("[NaN/Inf] CM_des has non-finite values")

        # dCM = dA(q) * qdot + A(q) * qddot
        dCM_d = (dCMM @ v_d.unsqueeze(2)).squeeze(2) + (CMM @ a_d.unsqueeze(2)).squeeze(2)
        self.dCM = dCM_d.to(torch.float32)

        dCM_base_d = (dCMM[:, :, :nb] @ v_d[:, :nb].unsqueeze(2)).squeeze(2) + (CMM[:, :, :nb] @ a_d[:, :nb].unsqueeze(2)).squeeze(2)
        dCM_leg_d  = (dCMM[:, :, nb:nb+nl] @ v_d[:, nb:nb+nl].unsqueeze(2)).squeeze(2) + (CMM[:, :, nb:nb+nl] @ a_d[:, nb:nb+nl].unsqueeze(2)).squeeze(2)
        dCM_arm_d  = (dCMM[:, :, nb+nl:] @ v_d[:, nb+nl:].unsqueeze(2)).squeeze(2) + (CMM[:, :, nb+nl:] @ a_d[:, nb+nl:].unsqueeze(2)).squeeze(2)

        self.dCM_base = dCM_base_d.to(torch.float32)
        self.dCM_leg  = dCM_leg_d.to(torch.float32)
        self.dCM_arm  = dCM_arm_d.to(torch.float32)

        # ---- 5) 회전변환도 dtype 맞추기 (rot_matrix_block이 double이면 여기서 또 터짐) ----
        rot_matrix = math_utils.matrix_from_quat(self.robot.data.root_link_quat_w).permute(0, 2, 1)  # world->base
        if not torch.isfinite(rot_matrix).all():
            raise RuntimeError("[NaN/Inf] rot_matrix has non-finite values")
        # rot_matrix는 보통 float32이므로 CM도 float32로 맞춰서 연산 (이미 CM들은 float32)
        if self.rot_matrix_block.dtype != rot_matrix.dtype:
            self.rot_matrix_block = self.rot_matrix_block.to(rot_matrix.dtype)

        self.rot_matrix_block[:, :3, :3] = rot_matrix
        self.rot_matrix_block[:, 3:, 3:] = rot_matrix

        self.CM_bf      = (self.rot_matrix_block @ self.CM.unsqueeze(2)).squeeze(2)
        self.CM_base_bf = (self.rot_matrix_block @ self.CM_base.unsqueeze(2)).squeeze(2)
        self.CM_leg_bf  = (self.rot_matrix_block @ self.CM_leg.unsqueeze(2)).squeeze(2)
        self.CM_arm_bf  = (self.rot_matrix_block @ self.CM_arm.unsqueeze(2)).squeeze(2)
        self.CM_des_bf  = (self.rot_matrix_block @ self.CM_des.unsqueeze(2)).squeeze(2)

        if not torch.isfinite(self.CM_bf).all():
            raise RuntimeError("[NaN/Inf] CM_bf has non-finite values")

        self.dCM_bf      = (self.rot_matrix_block @ self.dCM.unsqueeze(2)).squeeze(2)
        self.dCM_base_bf = (self.rot_matrix_block @ self.dCM_base.unsqueeze(2)).squeeze(2)
        self.dCM_leg_bf  = (self.rot_matrix_block @ self.dCM_leg.unsqueeze(2)).squeeze(2)
        self.dCM_arm_bf  = (self.rot_matrix_block @ self.dCM_arm.unsqueeze(2)).squeeze(2)

        # periodic timing report
        self._cusadi_timing_count += 1
        if self._cusadi_timing_count % self._cusadi_timing_every == 0:
            cnt = self._cusadi_timing_count
            com_ms = (self._cusadi_timing_com_s / cnt) * 1000.0
            cmm_ms = (self._cusadi_timing_cmm_s / cnt) * 1000.0
            total_ms = com_ms + cmm_ms
            carb.log_warn(
                f"[CUSADI_TIMING] avg per step: CoM={com_ms:.3f} ms | CMM/dCMM={cmm_ms:.3f} ms | total={total_ms:.3f} ms"
            )

    def _calculate_foot_velocity(self):
        """ Calculate the foot velocity in world frame """
        self.foot_velocity[:,0] = self.robot.data.body_state_w[:, self.body_feet_ids[0], 7:10]
        self.foot_velocity[:,1] = self.robot.data.body_state_w[:, self.body_feet_ids[1], 7:10]

    def _setup_keyboard_interface(self):
        self.keyboard_interface = VanillaKeyboard(self)

    def _setup_recorder(self):
        self.recorder = TocabiVanillaRecorder(
            fps=int(1 / self.step_dt),
            total_weight=self.total_weight[0],
            robot_cfg=self.robot.cfg,
        )

    def _get_log_data(self):
        robot_idx = 0
        self.action_full.zero_()

        # ActionManager.has_term() 제거: action dict 키로 판단
        am = getattr(self, "action_manager", None)
        action_dict = getattr(am, "action", {}) if am is not None else {}

        if self.leg_joint_ids is not None and "joint_pos" in action_dict:
            self.action_full[:, self.leg_joint_ids] = action_dict["joint_pos"]
        if self.arm_joint_ids is not None and "arm_joint_pos" in action_dict:
            self.action_full[:, self.arm_joint_ids] = action_dict["arm_joint_pos"]

        # fallback leg_joint_pos
        if "leg_joint_pos" in action_dict and self.leg_joint_ids is not None:
            self.action_full[:, self.leg_joint_ids] = action_dict["leg_joint_pos"]

        # 로그용 pin joint order (name-based perm)
        q_pin = self._joints_isaac_to_pin(self.robot.data.joint_pos)[robot_idx]
        dq_pin = self._joints_isaac_to_pin(self.robot.data.joint_vel)[robot_idx]
        tau_pin = self._joints_isaac_to_pin(self.robot.data.applied_torque)[robot_idx]

        qdes_pin = self._joints_isaac_to_pin(self.robot.data.default_joint_pos + self.action_full)[robot_idx]

        return {
            "root_lin_vel_b": self.robot.data.root_lin_vel_b[robot_idx],
            "root_ang_vel_b": self.robot.data.root_ang_vel_b[robot_idx],
            "vel_command": self.vel_command[robot_idx],

            "CM": self.CM[robot_idx],
            "CM_base": self.CM_base[robot_idx],
            "CM_leg": self.CM_leg[robot_idx],
            "CM_arm": self.CM_arm[robot_idx],
            "CM_des": self.CM_des[robot_idx],

            "CM_bf": self.CM_bf[robot_idx],
            "CM_base_bf": self.CM_base_bf[robot_idx],
            "CM_leg_bf": self.CM_leg_bf[robot_idx],
            "CM_arm_bf": self.CM_arm_bf[robot_idx],
            "CM_des_bf": self.CM_des_bf[robot_idx],

            "dCM": self.dCM[robot_idx],
            "dCM_base": self.dCM_base[robot_idx],
            "dCM_leg": self.dCM_leg[robot_idx],
            "dCM_arm": self.dCM_arm[robot_idx],

            "dCM_bf": self.dCM_bf[robot_idx],
            "dCM_base_bf": self.dCM_base_bf[robot_idx],
            "dCM_leg_bf": self.dCM_leg_bf[robot_idx],
            "dCM_arm_bf": self.dCM_arm_bf[robot_idx],

            "rf_toe_GRF": self.rf_toe_GRF[robot_idx],
            "rf_heel_GRF": self.rf_heel_GRF[robot_idx],
            "lf_toe_GRF": self.lf_toe_GRF[robot_idx],
            "lf_heel_GRF": self.lf_heel_GRF[robot_idx],
            "rf_GRM": self.rf_GRM[robot_idx],
            "lf_GRM": self.lf_GRM[robot_idx],

            "joint_pos_pin": q_pin,
            "joint_pos_target_pin": qdes_pin,
            "joint_vel_pin": dq_pin,
            "joint_torque_pin": tau_pin,

            "phase": self.phase[robot_idx],
            "foot_contact": self.foot_contact[robot_idx],
            "contact_schedule": self.contact_schedule[robot_idx],
            "foot_velocity": self.foot_velocity[robot_idx],
        }

    # ---------------------------- helper functions ----------------------------

    def smooth_sqr_wave(self, phase, eps=1.0):
        p = 2.0 * torch.pi * phase
        return torch.sin(p) / torch.sqrt(torch.sin(p) ** 2 + eps**2)

    def smooth_sqr_wave_advanced(self, phase, eps=0.2, a=0.2):
        t = (1 - a) / 2
        r = 2.0 * torch.pi * phase / (1 - a)
        r_out = torch.sin(r) / torch.sqrt(torch.sin(r) ** 2 + eps**2)

        l = 2.0 * torch.pi * (phase - (0.5 - t)) / (1 - a)
        l_out = torch.sin(l) / torch.sqrt(torch.sin(l) ** 2 + eps**2)

        mask_r = phase < t
        mask_l = (0.5 <= phase) & (phase < 0.5 + t)

        return r_out * mask_r + l_out * mask_l
