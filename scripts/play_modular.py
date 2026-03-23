# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import os
import sys

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


_prepend_once(_REPO_ROOT)
_prepend_once(os.path.join(_REPO_ROOT, "rsl_rl"))
for _pkg_root in _PKG_ROOTS.values():
    _prepend_once(_pkg_root)


import argparse
import os
import sys
import csv

# Force local IsaacLab/isaaclab_rl over editable installs in other workspaces.
sys.path = [
    p for p in sys.path
    if "rl_ws/src/LearningHumanoidArmMotion-RAL2025-Code" not in p
    or _REPO_ROOT in p
]


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
import math
import numpy as np
from typing import Optional

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--export_only",
    action="store_true",
    help="Export policy and exit without running the play loop.",
)
parser.add_argument(
    "--enable_arm",
    type=int,
    default=None,
    help="Override arm control enable (1=on, 0=off). If unset, use checkpoint config.",
)
parser.add_argument(
    "--leg_pt_path",
    type=str,
    default=None,
    help="Optional frozen leg TorchScript policy (.pt) path to override checkpoint leg policy.",
)
parser.add_argument(
    "--dump_arm_csv",
    type=int,
    default=0,
    choices=[0, 1],
    help="Dump arm obs/action to CSV during play (1=true, 0=false). Default: 0",
)
parser.add_argument(
    "--print_pos_limits",
    type=int,
    default=0,
    choices=[0, 1],
    help="Print joints near/outside soft joint position limits during play (1=true, 0=false). Default: 0",
)
parser.add_argument(
    "--print_camz",
    type=int,
    default=0,
    choices=[0, 1],
    help="Print env0 CAM z and desired CAM z during play (1=true, 0=false). Default: 0",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import cusadi
import extensions

print('[PATH_CHECK] PYTHONPATH:', os.environ.get('PYTHONPATH'))
print('[PATH_CHECK] sys.path[:5]:', sys.path[:5])
print('[PATH_CHECK] cusadi:', cusadi.__file__)
print('[PATH_CHECK] extensions:', extensions.__file__)

import torch
import traceback

import carb
from rsl_rl.modules.actor import Actor
from rsl_rl.runners import ModularOnPolicyRunner

# Import extensions to set up environment tasks
import extensions.humanoid  # noqa: F401
import extensions.tocabi.task  # noqa: F401
import isaaclab_tasks  # noqa: F401
from extensions import ISAACLAB_BRL_ROOT_DIR

from extensions.humanoid.utils import ContactForcesLivePlotter, ScreenShotter
try:
    from extensions.tocabi.utils.plotter import HipYawLivePlotter
except Exception:
    HipYawLivePlotter = None

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg, set_registry_to_original_files
from isaaclab_rl.rsl_rl import (
    RslRlModularOnPolicyRunnerCfg,
    RslRlModularVecEnvWrapper,
)


def _load_modular_runner_checkpoint(runner: ModularOnPolicyRunner, path: str) -> None:
    try:
        loaded_dict = torch.load(path)
    except Exception:
        import sys

        sys.modules["learning"] = sys.modules["rsl_rl"]
        sys.modules["learning.storage"] = sys.modules["rsl_rl.storage"]
        loaded_dict = torch.load(path)

    runner.leg_alg.actor_critic.load_state_dict(loaded_dict["leg_model_state_dict"])
    runner.arm_alg.actor_critic.load_state_dict(loaded_dict["arm_model_state_dict"])
    if "leg_optimizer_state_dict" in loaded_dict:
        runner.leg_alg.optimizer.load_state_dict(loaded_dict["leg_optimizer_state_dict"])
    if "arm_optimizer_state_dict" in loaded_dict:
        runner.arm_alg.optimizer.load_state_dict(loaded_dict["arm_optimizer_state_dict"])
    runner.current_learning_iteration = loaded_dict.get("iter", 0)


def _build_actor_from_state_dict(actor_state: dict[str, torch.Tensor]) -> Actor:
    weight_items: list[tuple[int, torch.Tensor]] = []
    for k, v in actor_state.items():
        if k.startswith("mean_NN.") and k.endswith(".weight"):
            try:
                idx = int(k.split(".")[1])
            except Exception:
                continue
            weight_items.append((idx, v))
    if not weight_items:
        raise RuntimeError("No actor mean_NN weights found in state dict.")
    weight_items.sort(key=lambda x: x[0])
    first_w = weight_items[0][1]
    last_w = weight_items[-1][1]
    num_obs = int(first_w.shape[1])
    num_actions = int(last_w.shape[0])
    hidden_dims = [int(w.shape[0]) for _, w in weight_items[:-1]]
    normalize_obs = any(k.startswith("obs_normalizer.") for k in actor_state.keys())
    actor = Actor(
        num_obs=num_obs,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        normalize_obs=normalize_obs,
        custom_initialization=False,
    )
    actor.load_state_dict(actor_state, strict=False)
    actor.eval()
    return actor


def _export_from_checkpoint_only(resume_path: str, experiment_name: str) -> None:
    loaded = torch.load(resume_path, map_location="cpu")

    if "leg_model_state_dict" in loaded:
        leg_model_state = loaded["leg_model_state_dict"]
    elif "model_state_dict" in loaded:
        leg_model_state = loaded["model_state_dict"]
    else:
        raise RuntimeError("Checkpoint does not contain leg model state dict.")

    arm_model_state = loaded.get("arm_model_state_dict", None)

    def _extract_actor_state(model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in model_state.items():
            if k.startswith("actor."):
                out[k[len("actor."):]] = v
        if not out:
            # older checkpoints may already be actor-only
            out = {k: v for k, v in model_state.items() if k.startswith("mean_NN.") or k.startswith("obs_normalizer.") or k == "std" or k.startswith("log_std_NN.")}
        return out

    leg_actor_state = _extract_actor_state(leg_model_state)
    if not leg_actor_state:
        raise RuntimeError("Could not extract leg actor state from checkpoint.")
    leg_actor = _build_actor_from_state_dict(leg_actor_state)

    path = os.path.join(ISAACLAB_BRL_ROOT_DIR, "logs", "rsl_rl", experiment_name, "exported")
    token = resume_path.split("/")
    model_name = f'{token[-2]}_{token[-1].split(".")[0]}'
    leg_actor.export(path, model_name + "_leg")

    if arm_model_state is not None:
        arm_actor_state = _extract_actor_state(arm_model_state)
        if arm_actor_state:
            arm_actor = _build_actor_from_state_dict(arm_actor_state)
            arm_actor.export(path, model_name + "_arm")

    print(f'Exported policy model "{model_name}" to: ', path)


def main():
    """Play with RSL-RL agent."""
    if args_cli.load_files:
        set_registry_to_original_files(args_cli.task, args_cli.load_run)
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlModularOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Export-only path: avoid creating env/sim assets entirely.
    if args_cli.export_only:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        _export_from_checkpoint_only(resume_path, agent_cfg.experiment_name)
        return

    if args_cli.enable_arm is not None:
        if hasattr(agent_cfg, "enable_arm"):
            agent_cfg.enable_arm = bool(args_cli.enable_arm)
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

    dump_arm_csv_enabled = bool(args_cli.dump_arm_csv)
    print_pos_limits_enabled = bool(args_cli.print_pos_limits)
    print_camz_enabled = bool(args_cli.print_camz)
    arm_csv_header: list[str] | None = None
    arm_csv_path: str | None = None
    arm_csv_file = None
    arm_csv_writer = None
    camz_csv_path: str | None = None
    camz_csv_file = None
    camz_csv_writer = None

    # create isaac environment
    env_cfg.seed = agent_cfg.seed
    env_cfg.record = RECORD
    if USE_JOYSTICK:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (-JOY_MAX_X, JOY_MAX_X)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (-JOY_MAX_Y, JOY_MAX_Y)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (-JOY_MAX_YAW, JOY_MAX_YAW)
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlModularVecEnvWrapper(env)
    # Force action dims from action_manager to avoid zero-dim arm loading.
    try:
        am = env.unwrapped.action_manager
        env.num_actions = {
            "leg": am.get_term("leg_joint_pos").action_dim,
            "arm": am.get_term("arm_joint_pos").action_dim,
        }
    except Exception:
        pass

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # load previously trained model
    ppo_runner = ModularOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if hasattr(ppo_runner, "load"):
        ppo_runner.load(resume_path)
    else:
        _load_modular_runner_checkpoint(ppo_runner, resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    if hasattr(ppo_runner, "get_inference_policy"):
        leg_policy, arm_policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    else:
        ppo_runner.leg_alg.actor_critic.eval()
        ppo_runner.arm_alg.actor_critic.eval()
        ppo_runner.leg_alg.actor_critic.to(env.unwrapped.device)
        ppo_runner.arm_alg.actor_critic.to(env.unwrapped.device)
        leg_policy = ppo_runner.leg_alg.actor_critic.act_inference
        arm_policy = ppo_runner.arm_alg.actor_critic.act_inference

    # Optional: override leg policy with a frozen exported TorchScript actor.
    if args_cli.leg_pt_path:
        leg_ts = torch.jit.load(args_cli.leg_pt_path, map_location=env.unwrapped.device)
        leg_ts.eval()
        print(f"[INFO] Overriding leg policy with TorchScript: {args_cli.leg_pt_path}")

        def _leg_policy_ts(obs):
            with torch.inference_mode():
                out = leg_ts(obs)
                if out.ndim == 1:
                    out = out.unsqueeze(0)
                return out

        leg_policy = _leg_policy_ts

    # * export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY or args_cli.export_only:
        path = os.path.join(ISAACLAB_BRL_ROOT_DIR, 'logs', 'rsl_rl', agent_cfg.experiment_name, 'exported')
        token = resume_path.split('/')
        model_name = f'{token[-2]}_{token[-1].split(".")[0]}'
        ppo_runner.export(path, model_name)
        print(f'Exported policy model "{model_name}" to: ', path)
    if args_cli.export_only:
        env.close()
        return

    # initialize custom functionality for logging
    log_states = ["root_lin_vel_w"] # "step_length", "step_width", "base_lin_vel", "base_pos", "dof_pos", 
    log_commands = ["vel_command"]
    robot_index = 0  # which robot is used for logging

    screenshotter = ScreenShotter(env.unwrapped, resume_path)
    if LIVE_PLOT:
        contactForcesLivePlotter = ContactForcesLivePlotter(env.unwrapped)
        hipYawPlotter = HipYawLivePlotter(env.unwrapped)

    # joystick (optional)
    joystick = None
    keyboard_fallback = False
    pygame_mod = None
    prev_buttons = None
    if USE_JOYSTICK:
        try:
            import pygame as pygame_mod  # type: ignore
        except Exception as exc:
            raise RuntimeError("pygame is required for joystick input. Install it in your env.") from exc
        pygame_mod.init()
        pygame_mod.joystick.init()
        if pygame_mod.joystick.get_count() == 0:
            keyboard_fallback = True
            print("[WARN] No joystick detected. Falling back to numpad cmd input (KP8/KP2/KP4/KP6/KP7/KP9, KP0=zero).")
        else:
            joystick = pygame_mod.joystick.Joystick(0)
            joystick.init()
            print(
                f"[INFO] Joystick connected: {joystick.get_name()} "
                f"(axes={joystick.get_numaxes()}, buttons={joystick.get_numbuttons()})"
            )
            prev_buttons = [0] * joystick.get_numbuttons()

    # cache for external force application (force mode)
    asset = env.unwrapped.scene["robot"]
    joint_names = list(getattr(asset.data, "joint_names", []))
    pos_limit_print_every = 20
    pos_limit_margin = 0.05
    camz_print_every = 20
    force_body_ids, _ = env.unwrapped.robot.find_bodies(["Pelvis_Link"])
    env_ids_all = torch.arange(env.unwrapped.scene.num_envs, device=env.unwrapped.device)

    # camera control (optional)
    vcc = getattr(env.unwrapped, "viewport_camera_controller", None)
    cam_control_available = vcc is not None
    if cam_control_available:
        cam_lookat = np.array(vcc.default_cam_lookat, dtype=float)
        cam_eye = np.array(vcc.default_cam_eye, dtype=float)
        cam_offset = cam_eye - cam_lookat
        cam_radius = float(np.linalg.norm(cam_offset)) if np.linalg.norm(cam_offset) > 1.0e-6 else 1.0
        cam_yaw = math.atan2(cam_offset[1], cam_offset[0])
        cam_pitch = math.atan2(cam_offset[2], math.sqrt(cam_offset[0] ** 2 + cam_offset[1] ** 2))
    else:
        cam_lookat = None
        cam_radius = None
        cam_yaw = None
        cam_pitch = None

    # mirror debug (play)
    mirror_leg = {
        "enabled": True,
        "obs_idx": torch.tensor(ppo_runner.leg_alg._mirror_obs_indices, device=env.unwrapped.device) if hasattr(ppo_runner.leg_alg, "_mirror_obs_indices") and ppo_runner.leg_alg._mirror_obs_indices is not None else None,
        "obs_sign": torch.tensor(ppo_runner.leg_alg._mirror_obs_signs, device=env.unwrapped.device) if hasattr(ppo_runner.leg_alg, "_mirror_obs_signs") and ppo_runner.leg_alg._mirror_obs_signs is not None else None,
        "act_idx": torch.tensor(ppo_runner.leg_alg._mirror_action_indices, device=env.unwrapped.device) if hasattr(ppo_runner.leg_alg, "_mirror_action_indices") and ppo_runner.leg_alg._mirror_action_indices is not None else None,
        "act_sign": torch.tensor(ppo_runner.leg_alg._mirror_action_signs, device=env.unwrapped.device) if hasattr(ppo_runner.leg_alg, "_mirror_action_signs") and ppo_runner.leg_alg._mirror_action_signs is not None else None,
    }
    mirror_arm = {
        "enabled": True,
        "obs_idx": torch.tensor(ppo_runner.arm_alg._mirror_obs_indices, device=env.unwrapped.device) if hasattr(ppo_runner.arm_alg, "_mirror_obs_indices") and ppo_runner.arm_alg._mirror_obs_indices is not None else None,
        "obs_sign": torch.tensor(ppo_runner.arm_alg._mirror_obs_signs, device=env.unwrapped.device) if hasattr(ppo_runner.arm_alg, "_mirror_obs_signs") and ppo_runner.arm_alg._mirror_obs_signs is not None else None,
        "act_idx": torch.tensor(ppo_runner.arm_alg._mirror_action_indices, device=env.unwrapped.device) if hasattr(ppo_runner.arm_alg, "_mirror_action_indices") and ppo_runner.arm_alg._mirror_action_indices is not None else None,
        "act_sign": torch.tensor(ppo_runner.arm_alg._mirror_action_signs, device=env.unwrapped.device) if hasattr(ppo_runner.arm_alg, "_mirror_action_signs") and ppo_runner.arm_alg._mirror_action_signs is not None else None,
    }

    # reset environment
    obs_dict = env.get_observations()
    phase0 = env.unwrapped.phase[0].item()
    print(f"[PHASE_PLAY] env0 phase={phase0:.3f}", flush=True)
    leg_actor_obs, arm_actor_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]
    mode = "move"
    cmd_lock = False
    locked_cmd = (0.0, 0.0, 0.0)
    # simulate environment
    try:
        while simulation_app.is_running():
            if USE_JOYSTICK and joystick is not None:
                # pump events and update base command from joystick
                try:
                    if pygame_mod is None:
                        raise RuntimeError("pygame not initialized")
                    pygame_mod.event.pump()
                    term = env.unwrapped.command_manager.get_term("base_velocity")
                    num_axes = joystick.get_numaxes()
                    x = -float(joystick.get_axis(JOY_AXIS_X)) if JOY_AXIS_X < num_axes else 0.0
                    y = -float(joystick.get_axis(JOY_AXIS_Y)) if JOY_AXIS_Y < num_axes else 0.0
                    yaw = -float(joystick.get_axis(JOY_AXIS_YAW)) if JOY_AXIS_YAW < num_axes else 0.0
                    # deadzone
                    def _dz(v: float, z: float) -> float:
                        return 0.0 if abs(v) < z else v
                    x, y, yaw = _dz(x, JOY_DEADZONE), _dz(y, JOY_DEADZONE), _dz(yaw, JOY_DEADZONE)
                    cmd_x = x * JOY_MAX_X
                    cmd_y = y * JOY_MAX_Y
                    cmd_yaw = yaw * JOY_MAX_YAW
                    # Update command inside inference mode to allow in-place on inference tensors.
                    with torch.inference_mode():
                        if mode == "move":
                            use_x, use_y, use_yaw = locked_cmd if cmd_lock else (cmd_x, cmd_y, cmd_yaw)
                            term.vel_command[:, 0] = use_x
                            term.vel_command[:, 1] = use_y
                            term.vel_command[:, 2] = use_yaw
                        else:
                            term.vel_command[:, 0] = 0.0
                            term.vel_command[:, 1] = 0.0
                            term.vel_command[:, 2] = 0.0
                    if False and env.unwrapped.common_step_counter % 60 == 0:
                        axes_dump = [float(joystick.get_axis(i)) for i in range(joystick.get_numaxes())]
                        buttons_dump = [int(joystick.get_button(i)) for i in range(joystick.get_numbuttons())]
                        print(
                            f"[JOY] axes: x={x:.2f} y={y:.2f} yaw={yaw:.2f} "
                            f"cmd={[cmd_x, cmd_y, cmd_yaw]} raw={axes_dump} buttons={buttons_dump}"
                        )
                    # buttons (edge-triggered)
                    buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
                    if prev_buttons is None:
                        prev_buttons = buttons
                    # button 0: move mode / lock toggle
                    if len(buttons) > 0 and buttons[0] and not prev_buttons[0]:
                        if mode != "move":
                            mode = "move"
                            print("[JOY] Mode set: move")
                        else:
                            cmd_lock = not cmd_lock
                            if cmd_lock:
                                locked_cmd = (cmd_x, cmd_y, cmd_yaw)
                                print(f"[JOY] Command locked: {locked_cmd}")
                            else:
                                print("[JOY] Command unlocked")
                    # button 3: force mode
                    if len(buttons) > 3 and buttons[3] and not prev_buttons[3]:
                        mode = "force"
                        print("[JOY] Mode set: force")
                    # camera yaw (buttons 4/5) and zoom (buttons 6/7)
                    if cam_control_available and cam_lookat is not None:
                        cam_changed = False
                        if len(buttons) > 4 and buttons[4]:
                            cam_yaw += JOY_CAM_YAW_STEP
                            cam_changed = True
                        if len(buttons) > 5 and buttons[5]:
                            cam_yaw -= JOY_CAM_YAW_STEP
                            cam_changed = True
                        if len(buttons) > 6 and buttons[6]:
                            cam_radius = max(JOY_CAM_RADIUS_MIN, cam_radius * JOY_CAM_ZOOM_IN)
                            cam_changed = True
                        if len(buttons) > 7 and buttons[7]:
                            cam_radius = min(JOY_CAM_RADIUS_MAX, cam_radius * JOY_CAM_ZOOM_OUT)
                            cam_changed = True
                        elev_axis = float(joystick.get_axis(JOY_AXIS_CAM_ELEV)) if JOY_AXIS_CAM_ELEV < num_axes else 0.0
                        if abs(elev_axis) <= JOY_DEADZONE and joystick.get_numhats() > 0:
                            # fallback to d-pad vertical if axis is not providing elevation
                            elev_axis = float(joystick.get_hat(0)[1])
                        if abs(elev_axis) > JOY_DEADZONE:
                            cam_pitch += elev_axis * JOY_CAM_PITCH_RATE
                            cam_pitch = max(-JOY_CAM_PITCH_MAX, min(JOY_CAM_PITCH_MAX, cam_pitch))
                            cam_changed = True
                        if cam_changed:
                            cam_offset = np.array(
                                [
                                    cam_radius * math.cos(cam_pitch) * math.cos(cam_yaw),
                                    cam_radius * math.cos(cam_pitch) * math.sin(cam_yaw),
                                    cam_radius * math.sin(cam_pitch),
                                ],
                                dtype=float,
                            )
                            cam_eye = cam_lookat + cam_offset
                            vcc.update_view_location(eye=cam_eye, lookat=cam_lookat)
                    # apply force once when button 3 is pressed in force mode
                    if mode == "force" and len(buttons) > 3 and buttons[3] and not prev_buttons[3]:
                        fx = cmd_x * JOY_FORCE_X
                        fy = cmd_y * JOY_FORCE_Y
                        tz = cmd_yaw * JOY_FORCE_TORQUE
                        force_vec = torch.tensor([fx, fy, 0.0], device=env.unwrapped.device) / env.unwrapped.physics_dt
                        torque_vec = torch.tensor([0.0, 0.0, tz], device=env.unwrapped.device) / env.unwrapped.physics_dt
                        _external_force_b = torch.zeros(
                            (env.unwrapped.scene.num_envs, asset.num_bodies, 3), device=env.unwrapped.device
                        )
                        _external_torque_b = torch.zeros_like(_external_force_b)
                        indices = (
                            torch.tensor(force_body_ids, dtype=torch.long, device=env.unwrapped.device).repeat(len(env_ids_all), 1)
                            + env_ids_all.unsqueeze(1) * asset.num_bodies
                        ).view(-1)
                        _external_force_b.flatten(0, 1)[indices] = force_vec
                        _external_torque_b.flatten(0, 1)[indices] = torque_vec
                        asset.root_physx_view.apply_forces_and_torques_at_position(
                            force_data=_external_force_b.view(-1, 3),
                            torque_data=_external_torque_b.view(-1, 3),
                            position_data=None,
                            indices=env_ids_all,
                            is_global=False,
                        )
                    # button 9: termination
                    if len(buttons) > 9 and buttons[9] and not prev_buttons[9]:
                        term_mgr = getattr(env.unwrapped, "termination_manager", None)
                        with torch.inference_mode():
                            if term_mgr is not None:
                                term_mgr._terminated_buf[:] = True
                                term_mgr._reset_buf[:] = True
                            env.reset()
                        print("[JOY] Termination triggered.")
                    # button 10: exit sim
                    if len(buttons) > 10 and buttons[10] and not prev_buttons[10]:
                        print("[JOY] Exit requested.")
                        env.close()
                        simulation_app.close()
                        return
                    prev_buttons = buttons
                except Exception as exc:
                    print(f"[WARN] Joystick update failed: {exc}")
            elif USE_JOYSTICK and keyboard_fallback and pygame_mod is not None:
                try:
                    pygame_mod.event.pump()
                    keys = pygame_mod.key.get_pressed()
                    term = env.unwrapped.command_manager.get_term("base_velocity")

                    # Accept both keypad and top-row digits (NumLock/layout variance).
                    x = (
                        float(keys[pygame_mod.K_KP8] or keys[pygame_mod.K_8])
                        - float(keys[pygame_mod.K_KP2] or keys[pygame_mod.K_2])
                    )
                    y = (
                        float(keys[pygame_mod.K_KP4] or keys[pygame_mod.K_4])
                        - float(keys[pygame_mod.K_KP6] or keys[pygame_mod.K_6])
                    )
                    yaw = (
                        float(keys[pygame_mod.K_KP7] or keys[pygame_mod.K_7])
                        - float(keys[pygame_mod.K_KP9] or keys[pygame_mod.K_9])
                    )

                    cmd_x = x * JOY_MAX_X
                    cmd_y = y * JOY_MAX_Y
                    cmd_yaw = yaw * JOY_MAX_YAW

                    if keys[pygame_mod.K_KP0] or keys[pygame_mod.K_0]:
                        cmd_x, cmd_y, cmd_yaw = 0.0, 0.0, 0.0

                    with torch.inference_mode():
                        if mode == "move":
                            term.vel_command[:, 0] = cmd_x
                            term.vel_command[:, 1] = cmd_y
                            term.vel_command[:, 2] = cmd_yaw
                        else:
                            term.vel_command[:, 0] = 0.0
                            term.vel_command[:, 1] = 0.0
                            term.vel_command[:, 2] = 0.0
                    if abs(cmd_x) > 0.0 or abs(cmd_y) > 0.0 or abs(cmd_yaw) > 0.0:
                        print(f"[KEY_CMD] x={cmd_x:.3f} y={cmd_y:.3f} yaw={cmd_yaw:.3f}", flush=True)
                except Exception as exc:
                    print(f"[WARN] Keyboard cmd update failed: {exc}")
            # run everything in inference mode
            with torch.inference_mode():
                # Snapshot current obs for CSV so we store aligned pair: (obs_t, act_t).
                obs0_for_csv = None
                step_for_csv = None
                if dump_arm_csv_enabled:
                    obs0_for_csv = arm_actor_obs[0].detach().cpu().numpy()
                    step_for_csv = float(env.unwrapped.common_step_counter)
                # agent stepping
                leg_actions = leg_policy(leg_actor_obs)
                arm_actions = arm_policy(arm_actor_obs)
                actions = torch.cat((leg_actions, arm_actions), dim=1)
            # debug: force q_ref tracking (zero action -> q_ref) 
            # actions = torch.zeros_like(actions)

            # debug: print arm actions periodically
            if not hasattr(ppo_runner, "_arm_action_dbg_every"):
                ppo_runner._arm_action_dbg_every = 50
                ppo_runner._arm_action_dbg_count = 0
            ppo_runner._arm_action_dbg_count += 1
            if ppo_runner._arm_action_dbg_count % ppo_runner._arm_action_dbg_every == 0:
                a0 = arm_actions[0].detach().cpu().numpy().tolist()

            # mirror debug print (play): compare a(M(o)) vs M(a(o))
            if env.unwrapped.common_step_counter % 60 == 0:
                if mirror_leg["obs_idx"] is not None and mirror_leg["act_idx"] is not None:
                    obs_m = leg_actor_obs[:, mirror_leg["obs_idx"]] * mirror_leg["obs_sign"]
                    mu_m = ppo_runner.leg_alg.actor_critic.actor.mean_NN(
                        ppo_runner.leg_alg.actor_critic.actor.norm_obs(obs_m)
                    )
                    mu_mirror = leg_actions[:, mirror_leg["act_idx"]] * mirror_leg["act_sign"]
                    diff = (mu_m - mu_mirror).abs()
                    print(
                        f"[MIRROR][PLAY][LEG] mean|diff|={diff.mean().item():.4e} "
                        f"max|diff|={diff.max().item():.4e}"
                    )
                if mirror_arm["obs_idx"] is not None and mirror_arm["act_idx"] is not None:
                    obs_m = arm_actor_obs[:, mirror_arm["obs_idx"]] * mirror_arm["obs_sign"]
                    mu_m = ppo_runner.arm_alg.actor_critic.actor.mean_NN(
                        ppo_runner.arm_alg.actor_critic.actor.norm_obs(obs_m)
                    )
                    mu_mirror = arm_actions[:, mirror_arm["act_idx"]] * mirror_arm["act_sign"]
                    diff = (mu_m - mu_mirror).abs()
                    print(
                        f"[MIRROR][PLAY][ARM] mean|diff|={diff.mean().item():.4e} "
                        f"max|diff|={diff.max().item():.4e}"
                    )

            # env stepping
            obs_dict, _, _, _, _, _ = env.step(actions)
            leg_actor_obs, arm_actor_obs = obs_dict["leg_actor"], obs_dict["arm_actor"]

            if print_pos_limits_enabled and env.unwrapped.common_step_counter % pos_limit_print_every == 0:
                q = asset.data.joint_pos[0]
                q_min = asset.data.soft_joint_pos_limits[0, :, 0]
                q_max = asset.data.soft_joint_pos_limits[0, :, 1]
                near_lower = (q >= q_min) & ((q - q_min) < pos_limit_margin)
                near_upper = (q <= q_max) & ((q_max - q) < pos_limit_margin)
                out_lower = q < q_min
                out_upper = q > q_max
                active = near_lower | near_upper | out_lower | out_upper
                if torch.any(active):
                    entries = []
                    active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1).tolist()
                    for j in active_ids:
                        status = []
                        if bool(out_lower[j]):
                            status.append("OUT_LO")
                        if bool(out_upper[j]):
                            status.append("OUT_HI")
                        if bool(near_lower[j]):
                            status.append("NEAR_LO")
                        if bool(near_upper[j]):
                            status.append("NEAR_HI")
                        entries.append(
                            f"{joint_names[j]}: q={q[j].item():.4f} lim=[{q_min[j].item():.4f}, {q_max[j].item():.4f}] {'/'.join(status)}"
                        )
                    print("[POS_LIM] " + " | ".join(entries), flush=True)

            if print_camz_enabled and env.unwrapped.common_step_counter % camz_print_every == 0:
                cm_z = float(env.unwrapped.CM[0, 5].item())
                cm_des_z = float(env.unwrapped.CM_des[0, 5].item())
                cm_bf_z = float(env.unwrapped.CM_bf[0, 5].item()) if hasattr(env.unwrapped, "CM_bf") else float("nan")
                print(
                    f"[CAM_Z] CMz={cm_z:.6f} CMz_des={cm_des_z:.6f} CMz_bf={cm_bf_z:.6f}",
                    flush=True,
                )

            if print_camz_enabled:
                cm_z = float(env.unwrapped.CM[0, 5].item())
                cm_des_z = float(env.unwrapped.CM_des[0, 5].item())
                cm_bf_z = float(env.unwrapped.CM_bf[0, 5].item()) if hasattr(env.unwrapped, "CM_bf") else float("nan")
                cm_base_z = float(env.unwrapped.CM_base[0, 5].item()) if hasattr(env.unwrapped, "CM_base") else float("nan")
                cm_arm_z = float(env.unwrapped.CM_arm[0, 5].item()) if hasattr(env.unwrapped, "CM_arm") else float("nan")
                cm_leg_z = float(env.unwrapped.CM_leg[0, 5].item()) if hasattr(env.unwrapped, "CM_leg") else float("nan")
                cm_rest_z = cm_z - cm_arm_z
                cm_base_bf_z = float(env.unwrapped.CM_base_bf[0, 5].item()) if hasattr(env.unwrapped, "CM_base_bf") else float("nan")
                cm_arm_bf_z = float(env.unwrapped.CM_arm_bf[0, 5].item()) if hasattr(env.unwrapped, "CM_arm_bf") else float("nan")
                cm_leg_bf_z = float(env.unwrapped.CM_leg_bf[0, 5].item()) if hasattr(env.unwrapped, "CM_leg_bf") else float("nan")
                cm_rest_bf_z = cm_bf_z - cm_arm_bf_z
                vel_cmd_x = float(env.unwrapped.vel_command[0, 0].item())
                vel_cmd_y = float(env.unwrapped.vel_command[0, 1].item())
                vel_cmd_yaw = float(env.unwrapped.vel_command[0, 2].item())
                if camz_csv_writer is None:
                    export_dir = os.path.join(log_root_path, "play_csv")
                    os.makedirs(export_dir, exist_ok=True)
                    run_tag = os.path.basename(os.path.dirname(resume_path))
                    ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]
                    camz_csv_path = os.path.join(export_dir, f"{run_tag}_{ckpt_tag}_camz.csv")
                    camz_csv_file = open(camz_csv_path, "w", newline="", encoding="utf-8")
                    camz_csv_writer = csv.writer(camz_csv_file)
                    camz_csv_writer.writerow(
                        [
                            "step",
                            "cm_z",
                            "cm_des_z",
                            "cm_bf_z",
                            "cm_base_z",
                            "cm_leg_z",
                            "cm_arm_z",
                            "cm_rest_z",
                            "cm_base_bf_z",
                            "cm_leg_bf_z",
                            "cm_arm_bf_z",
                            "cm_rest_bf_z",
                            "vel_command_x",
                            "vel_command_y",
                            "vel_command_yaw",
                        ]
                    )
                camz_csv_writer.writerow(
                    [
                        float(env.unwrapped.common_step_counter),
                        cm_z,
                        cm_des_z,
                        cm_bf_z,
                        cm_base_z,
                        cm_leg_z,
                        cm_arm_z,
                        cm_rest_z,
                        cm_base_bf_z,
                        cm_leg_bf_z,
                        cm_arm_bf_z,
                        cm_rest_bf_z,
                        vel_cmd_x,
                        vel_cmd_y,
                        vel_cmd_yaw,
                    ]
                )
                if camz_csv_file is not None:
                    camz_csv_file.flush()

            if dump_arm_csv_enabled:
                if obs0_for_csv is None:
                    obs0_for_csv = arm_actor_obs[0].detach().cpu().numpy()
                obs0 = obs0_for_csv
                act0 = arm_actions[0].detach().cpu().numpy()
                if arm_csv_header is None:
                    arm_csv_header = (
                        ["step"]
                        + [f"arm_obs_{i}" for i in range(obs0.shape[0])]
                        + [f"arm_act_{i}" for i in range(act0.shape[0])]
                    )
                    export_dir = os.path.join(log_root_path, "play_csv")
                    os.makedirs(export_dir, exist_ok=True)
                    run_tag = os.path.basename(os.path.dirname(resume_path))
                    ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]
                    arm_csv_path = os.path.join(export_dir, f"{run_tag}_{ckpt_tag}_arm_obs_action.csv")
                    arm_csv_file = open(arm_csv_path, "w", newline="", encoding="utf-8")
                    arm_csv_writer = csv.writer(arm_csv_file)
                    arm_csv_writer.writerow(arm_csv_header)
                if arm_csv_writer is not None:
                    arm_csv_writer.writerow(
                        [step_for_csv if step_for_csv is not None else float(env.unwrapped.common_step_counter)]
                        + [float(x) for x in obs0.tolist()]
                        + [float(x) for x in act0.tolist()]
                    )
                    # persist even if play is interrupted
                    if arm_csv_file is not None:
                        arm_csv_file.flush()

            # debug logs disabled (Q_DES / ARM stats)
    
            if env.unwrapped.screenshot:
                image = env.viewport_camera_image
                screenshotter.screenshot(image)
                env.unwrapped.screenshot = False
    
            if CUSTOM_COMMANDS:
                if env.unwrapped.common_step_counter == 1:
                    env.unwrapped.vel_command[:, 0] = 1.0
                    env.unwrapped.vel_command[:, 1] = 0.
                    env.unwrapped.vel_command[:, 2] = 0.
                elif env.unwrapped.common_step_counter == 200:
                    env.unwrapped.vel_command[:, 0] = 1.0
                    env.unwrapped.vel_command[:, 1] = 0.
                    env.unwrapped.vel_command[:, 2] = 1.0
                elif env.unwrapped.common_step_counter == 400:
                    env.unwrapped.vel_command[:, 0] = 0.
                    env.unwrapped.vel_command[:, 1] = 0.
                    env.unwrapped.vel_command[:, 2] = 1.0
                elif env.unwrapped.common_step_counter == 600:
                    env.unwrapped.record_done = True
    
            if LIVE_PLOT:
                # custom plotting
                contactForcesLivePlotter.log(env.unwrapped.contact_forces[robot_index])
                contactForcesLivePlotter.plot()
                hipYawPlotter.log()
                hipYawPlotter.plot()
    
            if RECORD:
                if env.unwrapped.record_done:
                    env.unwrapped.recorder.save(resume_path)
                    simulation_app.close()

    finally:
        if arm_csv_file is not None:
            arm_csv_file.close()
        if camz_csv_file is not None:
            camz_csv_file.close()
            if arm_csv_path is not None:
                print(f"[INFO] Saved arm obs/action CSV: {arm_csv_path}")
        env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True # True, False
    CUSTOM_COMMANDS = False # True, False
    LIVE_PLOT = False # True, False
    RECORD = False # True, False
    USE_JOYSTICK = True # True, False
    JOY_AXIS_X = 1      # left stick vertical (forward/back)
    JOY_AXIS_Y = 0      # left stick horizontal (left/right)
    JOY_AXIS_YAW = 3    # right stick horizontal
    JOY_AXIS_CAM_ELEV = 7
    JOY_DEADZONE = 0.02
    JOY_MAX_X = 1.0
    JOY_MAX_Y = 0.5
    JOY_MAX_YAW = 0.6
    JOY_FORCE_X = 100.0
    JOY_FORCE_Y = 100.0
    JOY_FORCE_TORQUE = 30.0
    JOY_CAM_YAW_STEP = 0.08
    JOY_CAM_PITCH_RATE = 0.03
    JOY_CAM_PITCH_MAX = 1.2
    JOY_CAM_ZOOM_IN = 0.94
    JOY_CAM_ZOOM_OUT = 1.06
    JOY_CAM_RADIUS_MIN = 0.5
    JOY_CAM_RADIUS_MAX = 20.0

    # args_cli.load_files = True # True, False
    # args_cli.load_run = '2025-05-23_00-34-58'
    # args_cli.checkpoint = 'model_1000.pt'
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
