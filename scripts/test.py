#!/usr/bin/env python3
# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

# Force local IsaacLab/isaaclab_rl over editable installs in other workspaces.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_ISAACLAB = os.path.join(_REPO_ROOT, "IsaacLab", "source")
if _LOCAL_ISAACLAB not in sys.path:
    sys.path.insert(0, _LOCAL_ISAACLAB)
sys.path = [p for p in sys.path if "rl_ws/src/LearningHumanoidArmMotion-RAL2025-Code" not in p]
# Drop ROS Python site-packages to avoid ABI/version-mismatched pinocchio shadowing.
sys.path = [p for p in sys.path if not p.startswith("/opt/ros/")]


def _force_local_pkg(pkg_name: str, local_root: str) -> None:
    for key in list(sys.modules):
        if key == pkg_name or key.startswith(pkg_name + "."):
            del sys.modules[key]
    if local_root not in sys.path:
        sys.path.insert(0, local_root)
    __import__(pkg_name)
    mod = sys.modules[pkg_name]
    local_pkg_path = os.path.join(local_root, pkg_name)
    if os.path.isdir(local_pkg_path):
        mod.__path__ = [local_pkg_path]


_force_local_pkg("isaaclab", _LOCAL_ISAACLAB)
_force_local_pkg("isaaclab_rl", _LOCAL_ISAACLAB)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_RSL_RL = os.path.join(REPO_ROOT, "rsl_rl")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if REPO_RSL_RL not in sys.path:
    sys.path.insert(0, REPO_RSL_RL)
sys.path = [
    p
    for p in sys.path
    if not (
        p.endswith("/LearningHumanoidArmMotion-RAL2025-Code")
        or p.endswith("/LearningHumanoidArmMotion-RAL2025-Code/rsl_rl")
    )
]
sys.path = [p for p in sys.path if not p.startswith("/opt/ros/")]

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Grid test a modular policy and save velocity tracking errors to CSV.")
parser.add_argument("--task", type=str, default="tocabi_arm_motion", help="Task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Seed.")
parser.add_argument("--enable_arm", type=int, default=None, help="Override arm control enable (1/0).")
parser.add_argument("--grid_n", type=int, default=5, help="Number of grid points per axis.")
parser.add_argument("--warmup_steps", type=int, default=200, help="Warmup steps per command.")
parser.add_argument("--measure_steps", type=int, default=400, help="Measured steps per command.")
parser.add_argument("--csv_path", type=str, default=None, help="Output CSV path. Default: checkpoint run folder.")
parser.add_argument("--vx_min", type=float, default=None)
parser.add_argument("--vx_max", type=float, default=None)
parser.add_argument("--vy_min", type=float, default=None)
parser.add_argument("--vy_max", type=float, default=None)
parser.add_argument("--wz_min", type=float, default=None)
parser.add_argument("--wz_max", type=float, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_experience = (
    f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    if args_cli.headless
    else f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import ModularOnPolicyRunner

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlModularOnPolicyRunnerCfg, RslRlModularVecEnvWrapper

import extensions.tocabi.task  # noqa: F401
import extensions.humanoid  # noqa: F401
import isaaclab_tasks  # noqa: F401


def _build_grid(lo: float, hi: float, n: int) -> list[float]:
    return torch.linspace(float(lo), float(hi), int(n)).cpu().tolist()


def _resolve_ranges(env_cfg) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    r = env_cfg.commands.base_velocity.ranges
    vx = (args_cli.vx_min if args_cli.vx_min is not None else r.lin_vel_x[0],
          args_cli.vx_max if args_cli.vx_max is not None else r.lin_vel_x[1])
    vy = (args_cli.vy_min if args_cli.vy_min is not None else r.lin_vel_y[0],
          args_cli.vy_max if args_cli.vy_max is not None else r.lin_vel_y[1])
    wz = (args_cli.wz_min if args_cli.wz_min is not None else r.ang_vel_z[0],
          args_cli.wz_max if args_cli.wz_max is not None else r.ang_vel_z[1])
    return vx, vy, wz


def _set_cmd(env, vx: float, vy: float, wz: float) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command[:, 0] = vx
    term.vel_command[:, 1] = vy
    term.vel_command[:, 2] = wz


def _get_policies(env, agent_cfg: RslRlModularOnPolicyRunnerCfg):
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    leg_policy, arm_policy = runner.get_inference_policy(device=env.unwrapped.device)
    return runner, leg_policy, arm_policy, resume_path


def _step_once(env, leg_policy, arm_policy, leg_obs, arm_obs):
    with torch.inference_mode():
        leg_actions = leg_policy(leg_obs)
        arm_actions = arm_policy(arm_obs)
    actions = torch.cat((leg_actions, arm_actions), dim=1)
    obs_dict, _, _, _, _, _ = env.step(actions)
    return obs_dict["leg_actor"], obs_dict["arm_actor"]


def _ensure_app_running() -> None:
    if not simulation_app.is_running():
        raise RuntimeError("simulation_app stopped during test loop (Isaac Sim closed).")


def _checkpoint_epoch_tag(resume_path: str) -> str:
    stem = Path(resume_path).stem  # e.g., model_5000, model_best
    if stem == "model_best":
        return "best"
    if stem.startswith("model_"):
        return stem.split("model_", 1)[1]
    return stem


def _default_csv_path(resume_path: str) -> Path:
    ckpt_path = Path(resume_path)
    epoch_tag = _checkpoint_epoch_tag(resume_path)
    return ckpt_path.parent / f"velocity_grid_eval_{epoch_tag}.csv"


def _write_rows_csv(out_path: Path, rows: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "cmd_vx", "cmd_vy", "cmd_wz", "avg_vel_x", "avg_vel_y", "avg_vel_z",
        "avg_error_x", "avg_error_y", "avg_error_z", "avg_error_xy",
        "num_envs", "warmup_steps", "measure_steps",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if args_cli.enable_arm is not None:
        if hasattr(agent_cfg, "enable_arm"):
            agent_cfg.enable_arm = bool(args_cli.enable_arm)
        if hasattr(agent_cfg, "arm_update_enabled"):
            agent_cfg.arm_update_enabled = bool(args_cli.enable_arm)
        if hasattr(env_cfg, "enable_arm"):
            env_cfg.enable_arm = bool(args_cli.enable_arm)
    elif hasattr(env_cfg, "enable_arm"):
        env_cfg.enable_arm = bool(getattr(agent_cfg, "enable_arm", True))
    if hasattr(env_cfg, "_apply_arm_enable"):
        env_cfg._apply_arm_enable()
    if hasattr(agent_cfg, "enable_actor_history") and hasattr(env_cfg, "enable_actor_history"):
        env_cfg.enable_actor_history = bool(getattr(agent_cfg, "enable_actor_history"))
    if hasattr(env_cfg, "_apply_actor_history"):
        env_cfg._apply_actor_history()

    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
        agent_cfg.seed = int(args_cli.seed)
    else:
        env_cfg.seed = agent_cfg.seed

    env_cfg.record = False
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlModularVecEnvWrapper(env)

    try:
        am = env.unwrapped.action_manager
        env.num_actions = {
            "leg": am.get_term("leg_joint_pos").action_dim,
            "arm": am.get_term("arm_joint_pos").action_dim,
        }
    except Exception:
        pass

    runner, leg_policy, arm_policy, resume_path = _get_policies(env, agent_cfg)
    epoch_tag = _checkpoint_epoch_tag(resume_path)

    obs_dict = env.get_observations()
    leg_obs, arm_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]

    vx_rng, vy_rng, wz_rng = _resolve_ranges(env_cfg)
    vx_grid = _build_grid(*vx_rng, args_cli.grid_n)
    vy_grid = _build_grid(*vy_rng, args_cli.grid_n)
    wz_grid = _build_grid(*wz_rng, args_cli.grid_n)
    total_cases = len(vx_grid) * len(vy_grid) * len(wz_grid)
    print(f"[INFO] Grid size: {len(vx_grid)}x{len(vy_grid)}x{len(wz_grid)} = {total_cases}")
    print(f"[INFO] Ranges: vx={vx_rng}, vy={vy_rng}, wz={wz_rng}")

    rows: list[dict[str, float | int]] = []
    case_idx = 0
    device = env.unwrapped.device
    nenv = int(env.unwrapped.num_envs)
    out_path = Path(args_cli.csv_path) if args_cli.csv_path else _default_csv_path(resume_path)
    print(f"[INFO] CSV path: {out_path}")

    for vx in vx_grid:
        for vy in vy_grid:
            for wz in wz_grid:
                case_idx += 1
                print(f"[TEST] {case_idx}/{total_cases} cmd=({vx:.3f}, {vy:.3f}, {wz:.3f})", flush=True)
                t_case0 = time.perf_counter()

                obs_dict = env.reset()
                if isinstance(obs_dict, tuple):
                    obs_dict = obs_dict[0]
                if not isinstance(obs_dict, dict):
                    obs_dict = env.get_observations()
                leg_obs, arm_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]

                for _ in range(int(args_cli.warmup_steps)):
                    _ensure_app_running()
                    _set_cmd(env, vx, vy, wz)
                    leg_obs, arm_obs = _step_once(env, leg_policy, arm_policy, leg_obs, arm_obs)

                sum_abs_x = torch.zeros((), device=device)
                sum_abs_y = torch.zeros((), device=device)
                sum_abs_z = torch.zeros((), device=device)
                sum_abs_xy = torch.zeros((), device=device)
                sum_vel_x = torch.zeros((), device=device)
                sum_vel_y = torch.zeros((), device=device)
                sum_vel_z = torch.zeros((), device=device)
                count = 0

                for _ in range(int(args_cli.measure_steps)):
                    _ensure_app_running()
                    _set_cmd(env, vx, vy, wz)
                    leg_obs, arm_obs = _step_once(env, leg_policy, arm_policy, leg_obs, arm_obs)

                    root_lin_b = env.unwrapped.robot.data.root_lin_vel_b[:, :2]
                    root_wz_b = env.unwrapped.robot.data.root_ang_vel_b[:, 2]
                    err_x = torch.abs(root_lin_b[:, 0] - vx)
                    err_y = torch.abs(root_lin_b[:, 1] - vy)
                    err_z = torch.abs(root_wz_b - wz)

                    sum_abs_x += err_x.sum()
                    sum_abs_y += err_y.sum()
                    sum_abs_z += err_z.sum()
                    sum_abs_xy += torch.sqrt(err_x * err_x + err_y * err_y).sum()
                    sum_vel_x += root_lin_b[:, 0].sum()
                    sum_vel_y += root_lin_b[:, 1].sum()
                    sum_vel_z += root_wz_b.sum()
                    count += nenv

                rows.append(
                    {
                        "cmd_vx": float(vx),
                        "cmd_vy": float(vy),
                        "cmd_wz": float(wz),
                        "avg_vel_x": float((sum_vel_x / count).item()),
                        "avg_vel_y": float((sum_vel_y / count).item()),
                        "avg_vel_z": float((sum_vel_z / count).item()),
                        "avg_error_x": float((sum_abs_x / count).item()),
                        "avg_error_y": float((sum_abs_y / count).item()),
                        "avg_error_z": float((sum_abs_z / count).item()),
                        "avg_error_xy": float((sum_abs_xy / count).item()),
                        "num_envs": nenv,
                        "warmup_steps": int(args_cli.warmup_steps),
                        "measure_steps": int(args_cli.measure_steps),
                    }
                )
                _write_rows_csv(out_path, rows)
                t_case1 = time.perf_counter()
                print(
                    f"[TEST_DONE] {case_idx}/{total_cases} "
                    f"elapsed={t_case1 - t_case0:.2f}s "
                    f"saved_rows={len(rows)} -> {out_path}",
                    flush=True,
                )

    print(f"[DONE] Saved CSV: {out_path} ({len(rows)} rows)")

    runner.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        raise
    finally:
        simulation_app.close()
