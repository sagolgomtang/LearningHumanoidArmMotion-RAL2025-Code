# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import os
import sys
import sysconfig

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ISAACLAB_SOURCE = os.path.join(_REPO_ROOT, "IsaacLab", "source")
_PKG_ROOTS = {
    "isaaclab": os.path.join(_ISAACLAB_SOURCE, "isaaclab"),
    "isaaclab_rl": os.path.join(_ISAACLAB_SOURCE, "isaaclab_rl"),
    "isaaclab_tasks": os.path.join(_ISAACLAB_SOURCE, "isaaclab_tasks"),
}


def _prepend_once(path: str) -> None:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _sanitize_runtime_env() -> None:
    """Avoid ROS-shipped C++ libs shadowing this env's cmeel/pinocchio stack."""
    ros_prefixes = ("/opt/ros/",)

    def _filter_env_list(var_name: str) -> None:
        raw = os.environ.get(var_name, "")
        if not raw:
            return
        kept = [p for p in raw.split(":") if p and not p.startswith(ros_prefixes)]
        os.environ[var_name] = ":".join(kept)

    _filter_env_list("LD_LIBRARY_PATH")
    _filter_env_list("PYTHONPATH")

    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        cmeel_lib = os.path.join(purelib, "cmeel.prefix", "lib")
        if os.path.isdir(cmeel_lib):
            ld_parts = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
            if cmeel_lib not in ld_parts:
                os.environ["LD_LIBRARY_PATH"] = ":".join([cmeel_lib, *ld_parts])


_prepend_once(_REPO_ROOT)
_prepend_once(os.path.join(_REPO_ROOT, "rsl_rl"))
for _pkg_root in _PKG_ROOTS.values():
    _prepend_once(_pkg_root)
_sanitize_runtime_env()

# Force local IsaacLab/isaaclab_rl over editable installs in other workspaces.
sys.path = [
    p for p in sys.path
    if "rl_ws/src/LearningHumanoidArmMotion-RAL2025-Code" not in p
    or _REPO_ROOT in p
]
sys.path = [p for p in sys.path if not p.startswith("/opt/ros/")]


def _force_local_pkg(pkg_name: str, pkg_root: str) -> None:
    for key in list(sys.modules):
        if key == pkg_name or key.startswith(pkg_name + "."):
            del sys.modules[key]
    _prepend_once(pkg_root)
    __import__(pkg_name)
    mod = sys.modules.get(pkg_name)
    mod_file = getattr(mod, "__file__", "")
    mod_paths = list(getattr(mod, "__path__", []))
    local_pkg_path = os.path.join(pkg_root, pkg_name)
    if os.path.isdir(local_pkg_path):
        mod.__path__ = [local_pkg_path]
        mod_paths = list(mod.__path__)
    elif os.path.isdir(pkg_root):
        mod.__path__ = [pkg_root]
        mod_paths = list(mod.__path__)
    if mod_file and pkg_root not in mod_file:
        raise RuntimeError(f"{pkg_name} imported from non-local path: {mod_file}")
    if mod_paths and any(pkg_root not in p for p in mod_paths):
        raise RuntimeError(f"{pkg_name} namespace paths are non-local: {mod_paths}")


_force_local_pkg("isaaclab", _PKG_ROOTS["isaaclab"])
_force_local_pkg("isaaclab_rl", _PKG_ROOTS["isaaclab_rl"])
import time
from typing import Any, Dict

# Ensure this repo is first on sys.path to avoid mixing with other clones.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_RSL_RL = os.path.join(REPO_ROOT, "rsl_rl")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Force local rsl_rl ahead of other clones (namespace package picks first path)
if REPO_RSL_RL not in sys.path:
    sys.path.insert(0, REPO_RSL_RL)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Remove other same-named clone paths while keeping this repo on sys.path.
sys.path = [
    p for p in sys.path
    if "LearningHumanoidArmMotion-RAL2025-Code" not in p
    or p == REPO_ROOT
    or p == REPO_RSL_RL
    or p == SCRIPT_DIR
]
sys.path = [p for p in sys.path if not p.startswith("/opt/ros/")]

# # low-level debug logger to file + stderr (avoids buffering issues)
# def _dbg(msg: str) -> None:
#     try:
#         ts = time.strftime("%Y-%m-%d %H:%M:%S")
#         line = f"[TRAIN_DEBUG] {ts} {msg}\n"
#         with open("/tmp/train_modular_debug.log", "a", encoding="utf-8") as f:
#             f.write(line)
#         os.write(2, line.encode())
#     except Exception:
#         pass
#
# _dbg(f"boot: cwd={os.getcwd()} argv={' '.join(sys.argv)} py={sys.executable}")
# _dbg(f"boot: sys.path[0:5]={sys.path[:5]}")

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--enable_arm",
    type=int,
    default=None,
    help="Override arm control enable (1=on, 0=off). If unset, use config.",
)
parser.add_argument(
    "--enable_leg_update",
    type=int,
    default=None,
    help="Override leg policy update enable (1=on, 0=off). If unset, use config.",
)
parser.add_argument(
    "--leg_onnx_path",
    type=str,
    default=None,
    help="Path to frozen leg ONNX policy used for inference when leg update is disabled.",
)
# benchmark options
parser.add_argument("--bench", action="store_true", default=False, help="Run env.step benchmark only and exit.")
parser.add_argument("--bench_steps", type=int, default=50, help="Benchmark steps (timed).")
parser.add_argument("--bench_warmup", type=int, default=10, help="Warmup steps (not timed).")
parser.add_argument("--bench_sync", action="store_true", default=False, help="Sync CUDA around timing.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

# launch omniverse app
# _dbg("pre AppLauncher()")
app_launcher = AppLauncher(args_cli)
# _dbg("post AppLauncher()")
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch
import traceback
from datetime import datetime

import carb
from rsl_rl.runners import ModularOnPolicyRunner
import rsl_rl
import rsl_rl.runners.modular_on_policy_runner as mopr

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from utils import local_code_save_helper

# Import extensions to set up environment tasks
import extensions.humanoid  # noqa: F401
import extensions.tocabi.task  # noqa: F401
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlModularOnPolicyRunnerCfg, RslRlModularVecEnvWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# quick sanity: show which rsl_rl is actually imported
# _dbg(f"rsl_rl import: {rsl_rl.__file__}")
# _dbg(f"rsl_rl.__path__: {list(getattr(rsl_rl, '__path__', []))}")
# _dbg(f"modular_on_policy_runner: {mopr.__file__}")


def _maybe_sync(sync: bool):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_num_envs(env) -> int:
    for attr in ["num_envs", "num_env"]:
        if hasattr(env, attr):
            v = getattr(env, attr)
            if isinstance(v, int):
                return v
    if args_cli.num_envs is not None:
        return int(args_cli.num_envs)
    return 1


def _make_dummy_action(env, device: str) -> Any:
    space = getattr(env, "action_space", None)
    nenv = _get_num_envs(env)

    if space is None:
        if hasattr(env, "num_actions"):
            act_dim = int(env.num_actions)
            return torch.zeros((nenv, act_dim), device=device, dtype=torch.float32)
        raise RuntimeError("Cannot infer action shape: env has no action_space/num_actions")

    try:
        import gymnasium.spaces as spaces
    except Exception:
        spaces = None

    if spaces is not None and isinstance(space, spaces.Box):
        act_shape = space.shape
        if len(act_shape) == 1:
            return torch.zeros((nenv, act_shape[0]), device=device, dtype=torch.float32)
        if len(act_shape) >= 2 and act_shape[0] == nenv:
            return torch.zeros(act_shape, device=device, dtype=torch.float32)
        return torch.zeros((nenv, *act_shape), device=device, dtype=torch.float32)

    if spaces is not None and isinstance(space, spaces.Dict):
        out: Dict[str, Any] = {}
        for k, sub in space.spaces.items():
            if not isinstance(sub, spaces.Box):
                raise RuntimeError(f"Unsupported Dict sub-space type: {type(sub)} for key={k}")
            sh = sub.shape
            if len(sh) == 1:
                out[k] = torch.zeros((nenv, sh[0]), device=device, dtype=torch.float32)
            elif len(sh) >= 2 and sh[0] == nenv:
                out[k] = torch.zeros(sh, device=device, dtype=torch.float32)
            else:
                out[k] = torch.zeros((nenv, *sh), device=device, dtype=torch.float32)
        return out

    raise RuntimeError(f"Unsupported action_space type: {type(space)}")


def bench_env_step(env, device: str, warmup: int, steps: int, sync: bool):
    """Benchmark env.step throughput only (includes physics + env computation)."""
    print("[BENCH] Reset env...")
    env.reset()

    actions = _make_dummy_action(env, device=device)

    print(f"[BENCH] Warmup: {warmup} steps (not timed)")
    for _ in range(int(warmup)):
        env.step(actions)

    _maybe_sync(sync)
    steps = int(steps)
    print(f"[BENCH] Timed: {steps} steps")
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    _maybe_sync(sync)
    t1 = time.perf_counter()

    nenv = _get_num_envs(env)
    elapsed = t1 - t0
    per_step = elapsed / max(steps, 1)
    steps_per_s = (nenv * steps) / elapsed if elapsed > 0 else float("inf")

    print("\n[BENCH RESULT]")
    print(f"  num_envs        : {nenv}")
    print(f"  steps (timed)   : {steps}")
    print(f"  elapsed         : {elapsed:.3f} s")
    print(f"  avg env.step    : {per_step*1000.0:.3f} ms/step (wall)")
    print(f"  throughput      : {steps_per_s:,.0f} steps/s  (num_envs * steps / elapsed)")
    print("  note            : includes physics + obs/reward/reset logic + (CUSADI if inside env)\n")


def main():
    """Train with RSL-RL agent."""
    # _dbg(f"main entry: {__file__}")
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # On resume, start scheduled-push curriculum at max strength immediately.
    if getattr(agent_cfg, "resume", False):
        if hasattr(env_cfg, "enable_scheduled_push") and bool(getattr(env_cfg, "enable_scheduled_push", False)):
            if hasattr(env_cfg, "push_curriculum_start_epoch"):
                env_cfg.push_curriculum_start_epoch = 0
            if hasattr(env_cfg, "push_curriculum_end_epoch"):
                env_cfg.push_curriculum_end_epoch = 0
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
    if args_cli.enable_leg_update is not None and hasattr(agent_cfg, "leg_update_enabled"):
        agent_cfg.leg_update_enabled = bool(args_cli.enable_leg_update)
    if args_cli.leg_onnx_path is not None:
        agent_cfg.leg_frozen_onnx_path = args_cli.leg_onnx_path
    if hasattr(env_cfg, "_apply_arm_enable"):
        env_cfg._apply_arm_enable()
    if hasattr(agent_cfg, "enable_actor_history") and hasattr(env_cfg, "enable_actor_history"):
        env_cfg.enable_actor_history = bool(getattr(agent_cfg, "enable_actor_history"))
    if hasattr(env_cfg, "_apply_actor_history"):
        env_cfg._apply_actor_history()
    # hard-disable code state logging to avoid long git diff step
    # Ensure code snapshot is saved under logs/.../files
    agent_cfg.store_code_state = True

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env_cfg.seed = agent_cfg.seed
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # _dbg("env created")
    # wrap for video recording
    if agent_cfg.enable_logging and args_cli.video:
        env.metadata["render_fps"] = int(1/env.step_dt)
        video_interval = int(agent_cfg.num_steps_per_env*agent_cfg.save_interval)
        video_length = int(env_cfg.video_length_s/env.step_dt)

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlModularVecEnvWrapper(env)
    # Ensure modular action dims reflect leg+arm terms (in case wrapper is stale).
    try:
        am = env.unwrapped.action_manager
        leg_dim = am.get_term("leg_joint_pos").action_dim
        arm_dim = am.get_term("arm_joint_pos").action_dim
        env.num_actions = {"leg": leg_dim, "arm": arm_dim}
    except Exception:
        pass
    # _dbg("env wrapped")

    if args_cli.bench:
        base_env = getattr(env, "unwrapped", None)
        if base_env is not None and hasattr(base_env, "_enable_step_timing"):
            base_env._enable_step_timing = True
        bench_env_step(
            env,
            device=str(agent_cfg.device),
            warmup=args_cli.bench_warmup,
            steps=args_cli.bench_steps,
            sync=args_cli.bench_sync,
        )
        env.close()
        return

    # create runner from rsl-rl
    runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # _dbg("runner created")
    runner.cfg["store_code_state"] = True
    # fallback for older rsl_rl versions missing train/eval helpers
    if not hasattr(runner, "train_mode"):
        runner.train_mode = lambda: (runner.leg_alg.actor_critic.train(), runner.arm_alg.actor_critic.train())
    if not hasattr(runner, "eval_mode"):
        runner.eval_mode = lambda: (runner.leg_alg.actor_critic.eval(), runner.arm_alg.actor_critic.eval())
    if not hasattr(runner, "save"):
        def _save(path, infos=None):
            payload = {
                "leg_model_state_dict": runner.leg_alg.actor_critic.state_dict(),
                "arm_model_state_dict": runner.arm_alg.actor_critic.state_dict(),
                "leg_optimizer_state_dict": runner.leg_alg.optimizer.state_dict(),
                "arm_optimizer_state_dict": runner.arm_alg.optimizer.state_dict(),
                "iter": runner.current_learning_iteration,
                "infos": infos or {},
            }
            torch.save(payload, path)
        runner.save = _save
    # write git state to logs (optional in some rsl_rl versions)
    if hasattr(runner, "add_git_repo_to_log"):
        runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if agent_cfg.enable_logging:
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        if getattr(agent_cfg, "store_code_state", True):
            local_code_save_helper.log_and_save(log_dir)

    # _dbg("starting learn()")
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # _dbg("learn() finished")

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
