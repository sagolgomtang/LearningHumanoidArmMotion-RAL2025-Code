#!/usr/bin/env python3
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

parser = argparse.ArgumentParser(description="Evaluate local policy sensitivity (Lipschitz-like) on on-policy states.")
parser.add_argument("--task", type=str, default="tocabi_arm_motion")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--enable_arm", type=int, default=None)
parser.add_argument("--actor", type=str, default="leg", choices=["leg", "arm"], help="Which actor to evaluate.")
parser.add_argument("--rollout_warmup_steps", type=int, default=200, help="Warmup before collecting valid states.")
parser.add_argument("--collect_steps", type=int, default=200, help="How many env steps to collect on-policy states.")
parser.add_argument("--max_obs_samples", type=int, default=4096, help="Max sampled observations for metric computation.")
parser.add_argument("--fd_eps", type=float, default=1e-2, help="Per-dimension finite-difference epsilon.")
parser.add_argument("--rand_eps_l2", type=float, default=5e-2, help="Random-direction perturbation L2 norm.")
parser.add_argument("--rand_dirs_per_obs", type=int, default=4, help="Random perturbation directions per sampled obs.")
parser.add_argument("--dim_chunk", type=int, default=64, help="Obs-dim chunk size for per-dim evaluation.")
parser.add_argument("--csv_prefix", type=str, default=None, help="Optional output file prefix. Default: run folder + epoch.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
print(f"[LIP_TEST_SCRIPT] file={__file__} debug_ver=2026-02-24b", flush=True)

app_experience = (
    f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    if args_cli.headless
    else f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import ModularOnPolicyRunner

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg, set_registry_to_original_files
from isaaclab_rl.rsl_rl import RslRlModularOnPolicyRunnerCfg, RslRlModularVecEnvWrapper

import extensions.tocabi.task  # noqa: F401
import extensions.humanoid  # noqa: F401
import isaaclab_tasks  # noqa: F401


def _ensure_app_running() -> None:
    if not simulation_app.is_running():
        raise RuntimeError("simulation_app stopped during lipschitz_test.")


def _checkpoint_epoch_tag(resume_path: str) -> str:
    stem = Path(resume_path).stem
    if stem == "model_best":
        return "best"
    if stem.startswith("model_"):
        return stem.split("model_", 1)[1]
    return stem


def _default_prefix(resume_path: str, actor_name: str) -> Path:
    p = Path(resume_path)
    epoch = _checkpoint_epoch_tag(resume_path)
    return p.parent / f"lipschitz_eval_{actor_name}_{epoch}"


def _get_policies(env, agent_cfg: RslRlModularOnPolicyRunnerCfg):
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    leg_policy, arm_policy = runner.get_inference_policy(device=env.unwrapped.device)
    return runner, leg_policy, arm_policy, resume_path


def _actor_expected_obs_dim(runner, actor_name: str) -> int | None:
    try:
        actor_mod = runner.leg_alg.actor_critic.actor if actor_name == "leg" else runner.arm_alg.actor_critic.actor
        norm_mod = getattr(actor_mod, "obs_normalizer", None)
        mean_buf = getattr(norm_mod, "_mean", None)
        if mean_buf is not None:
            return int(mean_buf.shape[-1])
    except Exception:
        return None
    return None


def _print_obs_layout_debug(env) -> None:
    om = getattr(env.unwrapped, "observation_manager", None)
    if om is None:
        print("[WARN] No observation_manager on env.unwrapped", flush=True)
        return
    for group_name in ("leg_actor", "arm_actor", "leg_critic", "arm_critic"):
        try:
            names = list(om.active_terms[group_name])
            dims = [tuple(d) for d in om.group_obs_term_dim[group_name]]
            total = om.group_obs_dim[group_name]
            print(f"[OBS_LAYOUT] {group_name} total={total} terms={list(zip(names, dims))}", flush=True)
        except Exception as e:
            print(f"[OBS_LAYOUT] failed for {group_name}: {e}", flush=True)
    # Print resolved joint ordering for key terms (helps map obs_dim -> joint_name).
    try:
        robot = None
        robot_joint_names = None
        try:
            robot = env.unwrapped.scene["robot"]
            robot_joint_names = list(getattr(robot, "joint_names", []))
        except Exception:
            robot_joint_names = None
        for group_name in ("leg_actor", "arm_actor"):
            term_names = list(om.active_terms[group_name])
            term_cfgs = list(getattr(om, "_group_obs_term_cfgs", {}).get(group_name, []))
            for term_name, term_cfg in zip(term_names, term_cfgs):
                if term_name not in {"joint_pos", "joint_vel"}:
                    continue
                params = getattr(term_cfg, "params", None) or {}
                asset_cfg = params.get("asset_cfg", None)
                joint_names = getattr(asset_cfg, "joint_names", None) if asset_cfg is not None else None
                joint_ids = getattr(asset_cfg, "joint_ids", None) if asset_cfg is not None else None
                resolved_joint_names = None
                if isinstance(joint_ids, list) and robot_joint_names:
                    try:
                        resolved_joint_names = [robot_joint_names[j] for j in joint_ids]
                    except Exception:
                        resolved_joint_names = None
                print(
                    f"[OBS_JOINT_ORDER] {group_name}.{term_name} "
                    f"joint_names={joint_names} joint_ids={joint_ids} resolved_joint_names={resolved_joint_names}",
                    flush=True,
                )
    except Exception as e:
        print(f"[OBS_JOINT_ORDER] failed: {e}", flush=True)


def _build_obs_dim_metadata(env, actor_key: str) -> list[dict[str, str | int]]:
    """Build per-dimension metadata for an actor observation group."""
    om = getattr(env.unwrapped, "observation_manager", None)
    if om is None:
        return []
    try:
        group_name = actor_key
        term_names = list(om.active_terms[group_name])
        term_dims = [tuple(d) for d in om.group_obs_term_dim[group_name]]
        term_cfgs = list(getattr(om, "_group_obs_term_cfgs", {}).get(group_name, []))
    except Exception:
        return []

    robot_joint_names = None
    try:
        robot_joint_names = list(getattr(env.unwrapped.scene["robot"], "joint_names", []))
    except Exception:
        robot_joint_names = None

    # Collect resolved joint names for terms that use asset_cfg joint_ids.
    resolved_joint_names_by_term: dict[str, list[str] | None] = {}
    for term_name, term_cfg in zip(term_names, term_cfgs):
        params = getattr(term_cfg, "params", None) or {}
        asset_cfg = params.get("asset_cfg", None)
        joint_ids = getattr(asset_cfg, "joint_ids", None) if asset_cfg is not None else None
        resolved = None
        if isinstance(joint_ids, list) and robot_joint_names:
            try:
                resolved = [robot_joint_names[j] for j in joint_ids]
            except Exception:
                resolved = None
        resolved_joint_names_by_term[term_name] = resolved

    meta: list[dict[str, str | int]] = []
    obs_dim = 0
    for term_name, dims in zip(term_names, term_dims):
        flat_dim = 1
        for x in dims:
            flat_dim *= int(x)
        resolved_joint_names = resolved_joint_names_by_term.get(term_name)
        for local_idx in range(flat_dim):
            row = {
                "obs_dim": int(obs_dim),
                "term_name": term_name,
                "term_local_idx": int(local_idx),
                "joint_name": "",
            }
            if resolved_joint_names is not None and local_idx < len(resolved_joint_names):
                row["joint_name"] = resolved_joint_names[local_idx]
            meta.append(row)
            obs_dim += 1
    return meta


def _extract_actor_obs(env, obs_dict, actor_key: str, expected_dim: int | None = None) -> torch.Tensor:
    obs = obs_dict[actor_key]
    got = int(obs.shape[-1])
    if expected_dim is None or got == expected_dim:
        return obs
    om = getattr(env.unwrapped, "observation_manager", None)
    if om is None:
        return obs
    try:
        live = om.compute()[actor_key]
    except Exception as e:
        print(f"[WARN] observation_manager.compute() failed for {actor_key}: {e}", flush=True)
        return obs
    live_dim = int(live.shape[-1])
    if live_dim == expected_dim:
        print(
            f"[WARN] Using observation_manager.compute()['{actor_key}'] ({live_dim}) "
            f"instead of env return ({got}) due to dim mismatch.",
            flush=True,
        )
        return live
    return obs


def _step_once(env, leg_policy, arm_policy, leg_obs, arm_obs, expected_leg_dim=None, expected_arm_dim=None):
    with torch.inference_mode():
        leg_actions = leg_policy(leg_obs)
        arm_actions = arm_policy(arm_obs)
    actions = torch.cat((leg_actions, arm_actions), dim=1)
    obs_dict, _, _, _, _, _ = env.step(actions)
    leg_next = _extract_actor_obs(env, obs_dict, "leg_actor", expected_leg_dim)
    arm_next = _extract_actor_obs(env, obs_dict, "arm_actor", expected_arm_dim)
    return leg_next, arm_next


def _write_csv(path: Path, rows: list[dict], default_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else default_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sample_on_policy_obs(env, leg_policy, arm_policy, actor_key: str, expected_leg_dim=None, expected_arm_dim=None) -> torch.Tensor:
    obs_dict = env.reset()
    if isinstance(obs_dict, tuple):
        obs_dict = obs_dict[0]
    if not isinstance(obs_dict, dict):
        obs_dict = env.get_observations()
    leg_obs = _extract_actor_obs(env, obs_dict, "leg_actor", expected_leg_dim)
    arm_obs = _extract_actor_obs(env, obs_dict, "arm_actor", expected_arm_dim)

    for _ in range(int(args_cli.rollout_warmup_steps)):
        _ensure_app_running()
        leg_obs, arm_obs = _step_once(
            env, leg_policy, arm_policy, leg_obs, arm_obs, expected_leg_dim=expected_leg_dim, expected_arm_dim=expected_arm_dim
        )

    chunks: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for k in range(int(args_cli.collect_steps)):
        _ensure_app_running()
        leg_obs, arm_obs = _step_once(
            env, leg_policy, arm_policy, leg_obs, arm_obs, expected_leg_dim=expected_leg_dim, expected_arm_dim=expected_arm_dim
        )
        obs_cur = leg_obs if actor_key == "leg_actor" else arm_obs
        chunks.append(obs_cur.detach().clone())
        if (k + 1) % max(1, int(args_cli.collect_steps) // 5) == 0:
            print(f"[COLLECT] {k+1}/{args_cli.collect_steps} steps", flush=True)
    obs_all = torch.cat(chunks, dim=0)
    dt = time.perf_counter() - t0
    print(f"[INFO] Collected on-policy obs: {tuple(obs_all.shape)} in {dt:.2f}s")
    return obs_all


def _subsample_obs(obs_all: torch.Tensor, max_samples: int) -> torch.Tensor:
    n = obs_all.shape[0]
    if n <= max_samples:
        return obs_all
    idx = torch.randperm(n, device=obs_all.device)[:max_samples]
    return obs_all.index_select(0, idx)


def _actor_forward_mean(policy_fn, obs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return policy_fn(obs)


def _compute_random_dir_metric(policy_fn, obs_samples: torch.Tensor) -> dict[str, float]:
    base = _actor_forward_mean(policy_fn, obs_samples)
    n, d = obs_samples.shape
    k = int(args_cli.rand_dirs_per_obs)
    eps_l2 = float(args_cli.rand_eps_l2)
    if k <= 0 or eps_l2 <= 0.0:
        return {"rand_ratio_mean": 0.0, "rand_ratio_p50": 0.0, "rand_ratio_p90": 0.0, "rand_ratio_max": 0.0}

    obs_rep = obs_samples.repeat_interleave(k, dim=0)
    base_rep = base.repeat_interleave(k, dim=0)
    delta = torch.randn((n * k, d), device=obs_samples.device, dtype=obs_samples.dtype)
    delta = delta / (torch.linalg.norm(delta, dim=1, keepdim=True) + 1e-8)
    delta = delta * eps_l2
    pert = obs_rep + delta
    act_pert = _actor_forward_mean(policy_fn, pert)
    ratios = torch.linalg.norm(act_pert - base_rep, dim=1) / (torch.linalg.norm(delta, dim=1) + 1e-8)
    return {
        "rand_ratio_mean": float(ratios.mean().item()),
        "rand_ratio_p50": float(torch.quantile(ratios, 0.5).item()),
        "rand_ratio_p90": float(torch.quantile(ratios, 0.9).item()),
        "rand_ratio_max": float(ratios.max().item()),
    }


def _compute_per_dim_metric(policy_fn, obs_samples: torch.Tensor) -> list[dict]:
    n, d = obs_samples.shape
    h = float(args_cli.fd_eps)
    if h <= 0.0:
        raise ValueError("--fd_eps must be > 0")
    base = _actor_forward_mean(policy_fn, obs_samples)
    rows: list[dict] = []
    chunk = max(1, int(args_cli.dim_chunk))

    obs_mean = obs_samples.mean(dim=0)
    obs_std = obs_samples.std(dim=0, unbiased=False)
    for j0 in range(0, d, chunk):
        _ensure_app_running()
        j1 = min(d, j0 + chunk)
        m = j1 - j0
        # Build (+) and (-) perturb batches in one shot.
        obs_rep = obs_samples.unsqueeze(1).repeat(1, m, 1).reshape(n * m, d)
        base_rep = base.unsqueeze(1).repeat(1, m, 1).reshape(n * m, base.shape[1])
        idx = torch.arange(j0, j1, device=obs_samples.device).repeat(n)
        obs_plus = obs_rep.clone()
        obs_minus = obs_rep.clone()
        obs_plus[torch.arange(n * m, device=obs_samples.device), idx] += h
        obs_minus[torch.arange(n * m, device=obs_samples.device), idx] -= h
        act_plus = _actor_forward_mean(policy_fn, obs_plus)
        act_minus = _actor_forward_mean(policy_fn, obs_minus)
        ratio_plus = torch.linalg.norm(act_plus - base_rep, dim=1) / h
        ratio_minus = torch.linalg.norm(act_minus - base_rep, dim=1) / h
        ratio = 0.5 * (ratio_plus + ratio_minus)
        ratio = ratio.view(n, m)

        for local in range(m):
            vals = ratio[:, local]
            j = j0 + local
            rows.append(
                {
                    "obs_dim": int(j),
                    "ratio_mean": float(vals.mean().item()),
                    "ratio_p50": float(torch.quantile(vals, 0.5).item()),
                    "ratio_p90": float(torch.quantile(vals, 0.9).item()),
                    "ratio_max": float(vals.max().item()),
                    "obs_mean": float(obs_mean[j].item()),
                    "obs_std": float(obs_std[j].item()),
                }
            )
        print(f"[PER_DIM] processed dims {j0}:{j1}/{d}", flush=True)
    return rows


def main():
    if getattr(args_cli, "load_files", False):
        set_registry_to_original_files(args_cli.task, args_cli.load_run)
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
        if hasattr(agent_cfg, "arm_update_enabled"):
            agent_cfg.arm_update_enabled = bool(getattr(agent_cfg, "enable_arm", True))
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
    _print_obs_layout_debug(env)
    actor_key = "leg_actor" if args_cli.actor == "leg" else "arm_actor"
    policy_fn = leg_policy if args_cli.actor == "leg" else arm_policy
    obs_dim_meta_rows = _build_obs_dim_metadata(env, actor_key)
    expected_leg_dim = _actor_expected_obs_dim(runner, "leg")
    expected_arm_dim = _actor_expected_obs_dim(runner, "arm")
    print(f"[INFO] Checkpoint expected dims: leg={expected_leg_dim}, arm={expected_arm_dim}", flush=True)

    prefix = Path(args_cli.csv_prefix) if args_cli.csv_prefix else _default_prefix(resume_path, args_cli.actor)
    per_dim_csv = prefix.parent / f"{prefix.name}_per_dim.csv"
    summary_csv = prefix.parent / f"{prefix.name}_summary.csv"
    print(f"[INFO] Output summary CSV: {summary_csv}")
    print(f"[INFO] Output per-dim CSV: {per_dim_csv}")

    obs_all = _sample_on_policy_obs(
        env,
        leg_policy,
        arm_policy,
        actor_key,
        expected_leg_dim=expected_leg_dim,
        expected_arm_dim=expected_arm_dim,
    )
    obs_samples = _subsample_obs(obs_all, int(args_cli.max_obs_samples))
    print(f"[INFO] Using obs samples: {tuple(obs_samples.shape)}")

    # Early shape sanity check (common failure source: history on/off mismatch vs checkpoint).
    try:
        out_chk = _actor_forward_mean(policy_fn, obs_samples[:2])
    except RuntimeError as e:
        msg = str(e)
        if "size of tensor a" in msg and "size of tensor b" in msg:
            expected = None
            try:
                actor_mod = runner.leg_alg.actor_critic.actor if args_cli.actor == "leg" else runner.arm_alg.actor_critic.actor
                norm_mod = getattr(actor_mod, "obs_normalizer", None)
                mean_buf = getattr(norm_mod, "_mean", None)
                if mean_buf is not None:
                    expected = int(mean_buf.shape[-1])
            except Exception:
                pass
            got = int(obs_samples.shape[1])
            hint = (
                f"Observation dim mismatch for {args_cli.actor} actor: env gives {got}, "
                f"checkpoint expects {expected if expected is not None else 'unknown'}."
            )
            hint += " Likely config drift (e.g., actor history on/off). Try `--load_files` for that run."
            raise RuntimeError(hint) from e
        raise

    rand_stats = _compute_random_dir_metric(policy_fn, obs_samples)
    per_dim_rows = _compute_per_dim_metric(policy_fn, obs_samples)
    if obs_dim_meta_rows and len(obs_dim_meta_rows) == len(per_dim_rows):
        meta_by_dim = {int(r["obs_dim"]): r for r in obs_dim_meta_rows}
        for row in per_dim_rows:
            m = meta_by_dim.get(int(row["obs_dim"]))
            if m is None:
                continue
            row["term_name"] = str(m.get("term_name", ""))
            row["term_local_idx"] = int(m.get("term_local_idx", -1))
            row["joint_name"] = str(m.get("joint_name", ""))

    # Global summary from per-dim rows.
    ratio_means = torch.tensor([r["ratio_mean"] for r in per_dim_rows], device=obs_samples.device)
    ratio_p90s = torch.tensor([r["ratio_p90"] for r in per_dim_rows], device=obs_samples.device)
    summary_rows = [
        {
            "actor": args_cli.actor,
            "checkpoint": str(Path(resume_path).name),
            "epoch": _checkpoint_epoch_tag(resume_path),
            "obs_dim": int(obs_samples.shape[1]),
            "action_dim": int(_actor_forward_mean(policy_fn, obs_samples[:1]).shape[1]),
            "num_envs": int(env.unwrapped.num_envs),
            "rollout_warmup_steps": int(args_cli.rollout_warmup_steps),
            "collect_steps": int(args_cli.collect_steps),
            "collected_obs": int(obs_all.shape[0]),
            "used_obs_samples": int(obs_samples.shape[0]),
            "fd_eps": float(args_cli.fd_eps),
            "rand_eps_l2": float(args_cli.rand_eps_l2),
            "rand_dirs_per_obs": int(args_cli.rand_dirs_per_obs),
            "per_dim_ratio_mean_mean": float(ratio_means.mean().item()),
            "per_dim_ratio_mean_p90dims": float(torch.quantile(ratio_means, 0.9).item()),
            "per_dim_ratio_p90_mean": float(ratio_p90s.mean().item()),
            "per_dim_ratio_maxdim": float(ratio_means.max().item()),
            **rand_stats,
        }
    ]

    _write_csv(
        summary_csv,
        summary_rows,
        default_fields=list(summary_rows[0].keys()),
    )
    _write_csv(
        per_dim_csv,
        per_dim_rows,
        default_fields=[
            "obs_dim",
            "term_name",
            "term_local_idx",
            "joint_name",
            "ratio_mean",
            "ratio_p50",
            "ratio_p90",
            "ratio_max",
            "obs_mean",
            "obs_std",
        ],
    )

    print(f"[DONE] Saved summary CSV: {summary_csv}")
    print(f"[DONE] Saved per-dim CSV: {per_dim_csv}")

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
