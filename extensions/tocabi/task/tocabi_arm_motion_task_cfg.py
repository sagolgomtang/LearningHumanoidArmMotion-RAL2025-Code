# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # noqa: F401, F403
from .. import mdp as brl_mdp
from ..assets import TOCABI_CFG
from ..terrains import COBBLESTONE_ROAD_CFG

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane", # "plane" or "generator"
        terrain_generator=ROUGH_TERRAINS_CFG, # ROUGH_TERRAINS_CFG, COBBLESTONE_ROAD_CFG
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # static_friction=1.0,
            # dynamic_friction=1.0,
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = TOCABI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")
    # Foot-to-foot collision only (exclude ground).
    rf_foot_to_lf_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_Foot_Link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/L_Foot_Link"],
        history_length=1,
        debug_vis=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # # camera
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = brl_mdp.VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        reference="body",
        ranges=brl_mdp.VelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.6, 0.6),
        ),
    )

    # """ Stepping Commands """
    step_command = brl_mdp.StepCommandCfg(
        step_period_range=(35.0, 45.0),
    )


LEG_ACTION_JOINT_NAMES = [
        "L_HipYaw_Joint",
        "L_HipRoll_Joint",
        "L_HipPitch_Joint",
        "L_Knee_Joint",
        "L_AnklePitch_Joint",
        "L_AnkleRoll_Joint",
        "R_HipYaw_Joint",
        "R_HipRoll_Joint",
        "R_HipPitch_Joint",
        "R_Knee_Joint",
        "R_AnklePitch_Joint",
        "R_AnkleRoll_Joint",
]
ARM_ACTION_JOINT_NAMES = [
        "L_Shoulder1_Joint",
        "L_Shoulder2_Joint",
        "L_Shoulder3_Joint",
        "L_Elbow_Joint",
        "R_Shoulder1_Joint",
        "R_Shoulder2_Joint",
        "R_Shoulder3_Joint",
        "R_Elbow_Joint",
]
UPPER_JOINT_NAMES = [
        "Waist1_Joint",
        "Waist2_Joint",
        "Upperbody_Joint",
        "L_Armlink_Joint",
        "L_Forearm_Joint",
        "L_Wrist1_Joint",
        "L_Wrist2_Joint",
        "Neck_Joint",
        "Head_Joint",
        "R_Armlink_Joint",
        "R_Forearm_Joint",
        "R_Wrist1_Joint",
        "R_Wrist2_Joint",
]

# Joint-parameter blocks (order: legs(12) + arms(8) + upper(13))
_LEG_COUNT = 12
_ARM_COUNT = 8
_UPPER_COUNT = len(UPPER_JOINT_NAMES)

_FULL_P_GAINS = [
    2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
    2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
    400.0, 1000.0, 400.0, 400.0,
    400.0, 1000.0, 400.0, 400.0,
    6000.0, 10000.0, 10000.0,
    400.0, 400.0, 100.0, 100.0,
    100.0, 100.0,
    400.0, 400.0, 100.0, 100.0,
]
_FULL_D_GAINS = [
    15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
    15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
    10.0, 28.0, 10.0, 10.0,
    10.0, 28.0, 10.0, 10.0,
    200.0, 100.0, 100.0,
    10.0, 10.0, 3.0, 3.0,
    2.0, 2.0,
    10.0, 10.0, 3.0, 3.0,
]
_FULL_TORQUE_LIMITS = [
    333, 232, 263, 289, 222, 166,
    333, 232, 263, 289, 222, 166,
    64, 64, 64, 23,
    64, 64, 64, 23,
    303, 303, 303,
    64, 23, 10, 10,
    10, 10,
    64, 23, 10, 10,
]

_LEG_P_GAINS = _FULL_P_GAINS[:_LEG_COUNT]
_ARM_P_GAINS = _FULL_P_GAINS[_LEG_COUNT:_LEG_COUNT + _ARM_COUNT]
_UPPER_P_GAINS = _FULL_P_GAINS[_LEG_COUNT + _ARM_COUNT:]
_LEG_D_GAINS = _FULL_D_GAINS[:_LEG_COUNT]
_ARM_D_GAINS = _FULL_D_GAINS[_LEG_COUNT:_LEG_COUNT + _ARM_COUNT]
_UPPER_D_GAINS = _FULL_D_GAINS[_LEG_COUNT + _ARM_COUNT:]
_LEG_TORQUE_LIMITS = _FULL_TORQUE_LIMITS[:_LEG_COUNT]
_ARM_TORQUE_LIMITS = _FULL_TORQUE_LIMITS[_LEG_COUNT:_LEG_COUNT + _ARM_COUNT]
_UPPER_TORQUE_LIMITS = _FULL_TORQUE_LIMITS[_LEG_COUNT + _ARM_COUNT:]

_FULL_JOINT_POS_LIMITS = [
    (-0.3, 0.3),
    (-1.0, 1.0),
    (-1.0, 0.5),
    (-0.3, 1.5),
    (-0.8, 0.5),
    (-0.6, 0.6),  # left legs
    
    (-0.3, 0.3),
    (-1.0, 1.0),
    (-1.0, 0.5),
    (-0.3, 1.5),
    (-0.8, 0.5),
    (-0.6, 0.6),  # right legs
    
    # (-0.3, 0.3),
    # (-0.5, 0.5),
    # (-1.0, 0.5),
    # (-0.3, 1.5),
    # (-0.8, 0.5),
    # (-0.6, 0.6),  # left legs
    
    # (-0.3, 0.3),
    # (-0.5, 0.5),
    # (-1.0, 0.5),
    # (-0.3, 1.5),
    # (-0.8, 0.5),
    # (-0.6, 0.6),  # right legs
    
    (-0.1, 0.7),
    (-0.2, 0.8),
    (1.4, 1.6),
    (-1.5, -0.5),  # left arms
    
    (-0.7, 0.1),
    (-0.8, 0.2),
    (-1.6, -1.4),
    (0.5, 1.5),  # right arms
]
_LEG_JOINT_POS_LIMITS = _FULL_JOINT_POS_LIMITS[:_LEG_COUNT]
_ARM_JOINT_POS_LIMITS = _FULL_JOINT_POS_LIMITS[_LEG_COUNT:_LEG_COUNT + _ARM_COUNT]


@configclass
class ActionsCfg:
    leg_joint_pos = brl_mdp.TocabiActionCfg(
        asset_name="robot",
        clip={".*": (-1.0, 1.0)},
        lower_joint_names=LEG_ACTION_JOINT_NAMES,
        upper_joint_names=UPPER_JOINT_NAMES,
        pd_control=True,
        p_gains=_LEG_P_GAINS + _UPPER_P_GAINS,
        d_gains=_LEG_D_GAINS + _UPPER_D_GAINS,
        torque_limits=_LEG_TORQUE_LIMITS + _UPPER_TORQUE_LIMITS,
        joint_pos_limits=_LEG_JOINT_POS_LIMITS,
        rand_torque_inj_range=(0.0, 0.0),
        rand_torque_episode_range=(0.0, 3.0),
        rand_torque_tick_range=(0.0, 5.0),
        rand_motor_scale_range=(0.8, 1.2),
        scale=1.0,
        disable_arm_action=False,
        joint_vel_lpf_cutoff_hz=25.0,
    )
    arm_joint_pos = brl_mdp.TocabiActionCfg(
        asset_name="robot",
        clip={".*": (-1.0, 1.0)},
        lower_joint_names=ARM_ACTION_JOINT_NAMES,
        fixed_zero_action_joint_names=["L_Shoulder1_Joint", "R_Shoulder1_Joint"],
        upper_joint_names=[],
        pd_control=True,
        p_gains=_ARM_P_GAINS,
        d_gains=_ARM_D_GAINS,
        torque_limits=_ARM_TORQUE_LIMITS,
        joint_pos_limits=_ARM_JOINT_POS_LIMITS,
        rand_torque_inj_range=(0.0, 0.0),
        rand_torque_episode_range=(0.0, 3.0),
        rand_torque_tick_range=(0.0, 5.0),
        rand_motor_scale_range=(0.8, 1.2),
        scale=0.0,
        disable_arm_action=True,
        force_zero_raw_action=False,
        joint_vel_lpf_cutoff_hz=25.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LegActorCfg(ObsGroup):
        """Observations for leg actor. (order preserved)"""

        # base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        # base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(
            func=brl_mdp.base_ang_vel_lpf,
            noise=Gnoise(std=0.10),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "leg_actor_base_ang_vel",
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)

        # tocabi legs only (order preserved by joint_names_expr)
        joint_pos = ObsTerm(
            func=brl_mdp.joint_pos,
            noise=Gnoise(std=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr)},
        )
        joint_vel = ObsTerm(
            func=brl_mdp.joint_vel,
            noise=Gnoise(std=1.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "leg_actor_joint_vel",
            },
        )

        last_leg_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "leg_joint_pos",
                "joint_names_expr": LEG_ACTION_JOINT_NAMES,
            },
            noise=Gnoise(std=0.1),
        )
        # CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LegCriticCfg(ObsGroup):
        """Observations for leg critic. (order preserved)"""
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(
            func=brl_mdp.base_ang_vel_lpf,
            noise=Gnoise(std=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "leg_critic_base_ang_vel",
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)

        # NOTE: critic은 보통 전체 관절을 쓰는 경우가 많아서 기존 유지
        # brl_mdp.joint_pos/vel이 내부에서 어떤 joint set을 기본으로 쓰는지에 따라
        # 여기에도 legs joint_names_expr를 넣을지 결정해야 함.
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05))
        joint_vel = ObsTerm(
            func=brl_mdp.joint_vel,
            noise=Gnoise(std=1.0),
            params={
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "leg_critic_joint_vel",
            },
        )

        last_leg_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "leg_joint_pos",
                "joint_names_expr": LEG_ACTION_JOINT_NAMES,
            },
        )
        last_arm_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "arm_joint_pos",
                "joint_names_expr": ARM_ACTION_JOINT_NAMES,
            },
        )
        # CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        # CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ArmActorCfg(ObsGroup):
        """Observations for arm actor. (order preserved)"""
        # base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        # base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(
            func=brl_mdp.base_ang_vel_lpf,
            noise=Gnoise(std=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "arm_actor_base_ang_vel",
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))

        # tocabi arms only (order preserved by joint_names_expr)
        joint_pos = ObsTerm(
            func=brl_mdp.joint_pos,
            noise=Gnoise(std=0.05),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        joint_vel = ObsTerm(
            func=brl_mdp.joint_vel,
            noise=Gnoise(std=1.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "arm_actor_joint_vel",
            },
        )

        last_arm_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "arm_joint_pos",
                "joint_names_expr": ARM_ACTION_JOINT_NAMES,
            },
            noise=Gnoise(std=0.1),
        )
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ArmCriticCfg(ObsGroup):
        """Observations for arm critic. (order preserved)"""
        base_height = ObsTerm(func=mdp.base_pos_z, noise=Gnoise(std=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05))
        base_heading = ObsTerm(func=brl_mdp.base_heading, noise=Gnoise(std=0.01))
        base_ang_vel = ObsTerm(
            func=brl_mdp.base_ang_vel_lpf,
            noise=Gnoise(std=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "arm_critic_base_ang_vel",
            },
        )
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gnoise(std=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        phase_sin = ObsTerm(func=brl_mdp.phase_sin)
        phase_cos = ObsTerm(func=brl_mdp.phase_cos)

        # NOTE: 여기 역시 critic 기본 유지
        joint_pos = ObsTerm(func=brl_mdp.joint_pos, noise=Gnoise(std=0.05))
        joint_vel = ObsTerm(
            func=brl_mdp.joint_vel,
            noise=Gnoise(std=1.0),
            params={
                "lpf_enabled": True,
                "lpf_cutoff_hz": 25.0,
                "lpf_key": "arm_critic_joint_vel",
            },
        )

        last_leg_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "leg_joint_pos",
                "joint_names_expr": LEG_ACTION_JOINT_NAMES,
            },
        )
        last_arm_action = ObsTerm(
            func=brl_mdp.last_action_by_joint_names,
            params={
                "action_name": "arm_joint_pos",
                "joint_names_expr": ARM_ACTION_JOINT_NAMES,
            },
        )
        CAM = ObsTerm(func=brl_mdp.centroidal_angular_momentum_mixed, noise=Gnoise(std=0.1))
        CAM_des = ObsTerm(func=brl_mdp.centroidal_angular_momentum_des_mixed)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    leg_actor: LegActorCfg = LegActorCfg()
    leg_critic: LegCriticCfg = LegCriticCfg()
    arm_actor: ArmActorCfg = ArmActorCfg()
    arm_critic: ArmCriticCfg = ArmCriticCfg()



@configclass
class EventsCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Pelvis_Link"),
            "mass_distribution_params": (-1.0, 1.0),
            "operation": "add",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x":     (-0., 0.), 
                "y":     (-0., 0.), 
                "z":     (-0., 0.),
                "roll":  (-torch.pi/20, torch.pi/20), 
                "pitch": (-torch.pi/20, torch.pi/20), 
                "yaw":   (-torch.pi, torch.pi)
            },
            "velocity_range": {
                "x":     (-.1, .1),
                "y":     (-.1, .1),
                "z":     (-.1, .1),
                "roll":  (-.1, .1),
                "pitch": (-.1, .1),
                "yaw":   (-.1, .1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_range,
        mode="reset",
        params={
            "position_range": torch.tensor([[-0.1, 0.1], # right_hip_yaw
                                            [-0.1, 0.1], # left_hip_yaw
                                            [-0.1, 0.1], # right_hip_abad
                                            [-0.1, 0.1], # left_hip_abad
                                            [-0.2, 0.2], # right_hip_pitch
                                            [-0.2, 0.2], # left_hip_pitch
                                            [ 0.6, 0.7], # right_knee
                                            [ 0.6, 0.7], # left_knee
                                            [-0.3, 0.0], # right_ankle
                                            [-0.3, 0.0], # left_ankle
                                            [-0.1, 0.1], # right_shoulder_pitch
                                            [-0.1, 0.1], # left_shoulder_pitch
                                            [-0.1, 0.1], # right_shoulder_abad
                                            [-0.1, 0.1], # left_shoulder_abad
                                            [-0.1, 0.1], # right_shoulder_yaw
                                            [-0.1, 0.1], # left_shoulder_yaw
                                            [-0.1, 0.1], # right_elbow
                                            [-0.1, 0.1], # left_elbow
                                            ]),
            "velocity_range": torch.tensor([[-0.1, 0.1], # right_hip_yaw
                                            [-0.1, 0.1], # left_hip_yaw
                                            [-0.1, 0.1], # right_hip_abad
                                            [-0.1, 0.1], # left_hip_abad
                                            [-0.1, 0.1], # right_hip_pitch
                                            [-0.1, 0.1], # left_hip_pitch
                                            [-0.1, 0.1], # right_knee
                                            [-0.1, 0.1], # left_knee
                                            [-0.1, 0.1], # right_ankle
                                            [-0.1, 0.1], # left_ankle
                                            [-0.1, 0.1], # right_shoulder_pitch
                                            [-0.1, 0.1], # left_shoulder_pitch
                                            [-0.1, 0.1], # right_shoulder_abad
                                            [-0.1, 0.1], # left_shoulder_abad
                                            [-0.1, 0.1], # right_shoulder_yaw
                                            [-0.1, 0.1], # left_shoulder_yaw
                                            [-0.1, 0.1], # right_elbow
                                            [-0.1, 0.1], # left_elbow
                                            ]),
        },
    )

    # reset_ball = EventTerm(
    #     func=brl_mdp.throw_ball_uniform,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.0),
    #     params={
    #         "pose_range": {
    #             "x":     (1., 1.), 
    #             "y":     (0., 0.), 
    #             "z":     (1., 1.),
    #             "roll":  (-0., 0.), 
    #             "pitch": (-0., 0.), 
    #             "yaw":   (-0., 0.)
    #         },
    #         "velocity_range": {
    #             "x":     (-2., -2.),
    #             "y":     (-0., 0.),
    #             "z":     (-.5, .5),
    #             "roll":  (-0., 0.),
    #             "pitch": (-0., 0.),
    #             "yaw":   (-0., 0.),
    #         },
    #     },
    # )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_xy_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class EventsDeployCfg:
    """Configuration for randomization by events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    if hasattr(mdp, "randomize_rigid_body_com"):
        randomize_rigid_body_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
    else:
        randomize_rigid_body_com = None

    randomize_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    if hasattr(mdp, "randomize_joint_parameters"):
        randomize_armature_scale = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "armature_distribution_params": (0.5, 1.5),
                "operation": "scale",
            },
        )
        randomize_armature_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (-0.3, 0.3),
                "operation": "add",
            },
        )
    else:
        randomize_armature_scale = None
        randomize_armature_friction = None

    if hasattr(mdp, "randomize_actuator_gains"):
        randomize_damping = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.0, 0.0),
                "damping_distribution_params": (-0.9, 3.0),
                "operation": "add",
            },
        )
    else:
        randomize_damping = None

    # reset
    reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    @configclass
    class LegRewardsCfg:
        """Reward terms for the leg actor-critic."""
        survival = RewTerm(func=brl_mdp.survival_reward, weight=1.0)
        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=brl_mdp.action_smoothness1,
            weight=2e-3,
            params={
                "action_name": "leg_joint_pos",
                "joint_names_expr": LEG_ACTION_JOINT_NAMES,
            },
        )
        action_smoothness2 = RewTerm(
            func=brl_mdp.action_smoothness2,
            weight=2e-4,
            params={
                "action_name": "leg_joint_pos",
                "joint_names_expr": LEG_ACTION_JOINT_NAMES,
            },
        )
        joint_torque = RewTerm(
            func=brl_mdp.joint_torque_penalty,
            weight=1e-4,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr),
                "action_name": "leg_joint_pos",
            },
        )
        joint_velocity = RewTerm(
            func=brl_mdp.joint_velocity_penalty,
            weight=2e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr)},
        )
        foot_flat_contact = RewTerm(
            func=brl_mdp.foot_contact_flat_orientation_penalty,
            weight=100.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        swing_foot_height_legacy = RewTerm(
            func=brl_mdp.swing_foot_height_reward,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_height": 0.20,
                "max_height": 0.20,
                "schedule_based": True,
            },
        )
        swing_foot_height = RewTerm(
            func=brl_mdp.swing_foot_height_cubic_reward,
            weight=500.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_height": 0.18,
                "min_height": 0.14,
                "rise_end": 0.24,
                "hold_end": 0.25,
                "swing_end": 0.5,
            },
        )
        base_lin_vel_z = RewTerm(
            func=brl_mdp.base_lin_vel_z_penalty,
            weight=1e-1,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_ang_vel_xy = RewTerm(
            func=brl_mdp.base_ang_vel_xy_penalty,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_height_band = RewTerm(
            func=brl_mdp.base_height_band_penalty,
            weight=0.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_height": 0.84,
                "max_height": 0.86,
            },
        )
        joint_pos_limits = RewTerm(
            func=brl_mdp.joint_pos_limits_penalty,
            weight=10,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr),
                "margin": 0.10,
                "outside_weight": 1.0,
            },
        )
        joint_torque_limits = RewTerm(
            func=brl_mdp.joint_torque_limits_penalty,
            weight=1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["legs"].joint_names_expr)},
        )
        joint_regularization = RewTerm(
            func=brl_mdp.joint_regularization_penalty,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        knee_negative = RewTerm(
            func=brl_mdp.knee_negative_penalty,
            weight=100.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["L_Knee_Joint", "R_Knee_Joint"]),
                "min_knee_pos": 0.6,
            },
        )
        hip_yaw_negative = RewTerm(
            func=brl_mdp.hip_yaw_negative_penalty,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        hip_yaw_sum_zero = RewTerm(
            func=brl_mdp.hip_yaw_sum_zero_reward,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        hip_roll_threshold = RewTerm(
            func=brl_mdp.hip_roll_threshold_penalty,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "left_min": -0.06,
                "right_max": 0.06,
            },
        )
        illegal_contact = RewTerm(
            func=brl_mdp.illegal_contact_penalty,
            weight=0.0,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "Pelvis_Link",
                        "L_Thigh_Link",
                        "L_Knee_Link",
                        "L_AnkleRoll_Link",
                        "R_Thigh_Link",
                        "R_Knee_Link",
                        "R_AnkleRoll_Link",
                        "R_HipRoll_Link",
                        "L_HipRoll_Link",
                    ],
                ),
            },
        )
        foot_yaw_frame_y_dist = RewTerm(
            func=brl_mdp.foot_yaw_frame_y_distance_penalty,
            weight=10.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "threshold": 0.18,
            },
        )
        grf_limit = RewTerm(
            func=brl_mdp.grf_limit_penalty,
            weight=50.0,
            params={"threshold": 1500.0},
        )
        foot_slip = RewTerm(
            func=brl_mdp.foot_slip_penalty,
            weight=20.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "contact_force_threshold": 1.0,
                "vel_scale": 0.5,
            },
        )

        # * Floating base rewards * #
        base_z_orientation = RewTerm(
            func=brl_mdp.base_z_orientation_reward,
            weight=11.5,  # 1.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        tracking_lin_vel_x = RewTerm(
            func=brl_mdp.tracking_lin_vel_x_reward,
            weight=7.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
        )
        tracking_lin_vel_y = RewTerm(
            func=brl_mdp.tracking_lin_vel_y_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
        )
        tracking_yaw_vel = RewTerm(
            func=brl_mdp.tracking_yaw_vel_reward,
            weight=10.0,  # 1.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
        )

        # * Stepping rewards * #
        contact_schedule = RewTerm(
            func=brl_mdp.contact_schedule_reward,
            weight=30.0,
        )

        # * Termination rewards * #
        termination = RewTerm(
            func=brl_mdp.termination_penalty,
            weight=20.0,
            params={"group_name": "leg"},
        )

    @configclass
    class ArmRewardsCfg:
        """Reward terms for the arm actor-critic."""

        # * Regularization rewards * #
        action_smoothness1 = RewTerm(
            func=brl_mdp.action_smoothness1,
            weight=1e-3,
            params={"action_name": "arm_joint_pos"},
        )
        action_smoothness2 = RewTerm(
            func=brl_mdp.action_smoothness2,
            weight=1e-4,
            params={"action_name": "arm_joint_pos"},
        )
        joint_torque = RewTerm(
            func=brl_mdp.joint_torque_penalty,
            weight=7.5e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        joint_velocity = RewTerm(
            func=brl_mdp.joint_velocity_penalty,
            weight=5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        joint_position = RewTerm(
            func=brl_mdp.joint_position_penalty,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        joint_pos_limits = RewTerm(
            func=brl_mdp.joint_pos_limits_penalty,
            weight=10,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        joint_torque_limits = RewTerm(
            func=brl_mdp.joint_torque_limits_penalty,
            weight=1e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=TOCABI_CFG.actuators["arms"].joint_names_expr)},
        )
        dCAM_xy = RewTerm(
            func=brl_mdp.dCAM_xy_penalty,
            weight=5e-2,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        tracking_CAM_reward = RewTerm(
            func=brl_mdp.tracking_CAM_reward,
            weight=10.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity"},
        )

        # * Termination rewards * #
        termination_illegal_contact = RewTerm(
            func=brl_mdp.termination_penalty_term,
            weight=0.0,
            params={"group_name": "arm", "term_name": "illegal_contact"},
        )
        termination = RewTerm(
            func=brl_mdp.termination_penalty,
            weight=10.0,
            params={"group_name": "arm"},
        )

    leg: LegRewardsCfg = LegRewardsCfg()
    arm: ArmRewardsCfg = ArmRewardsCfg()



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    @configclass
    class LegTerminationCfg:
        illegal_contact = DoneTerm(
            func=brl_mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "Pelvis_Link",
                        "L_Thigh_Link",
                        "L_Knee_Link",
                        "L_AnkleRoll_Link",
                        "R_Thigh_Link",
                        "R_Knee_Link",
                        "R_AnkleRoll_Link",
                        "R_HipRoll_Link",
                        "L_HipRoll_Link",
                    ],
                ),
                "threshold": 1.0,
            },
        )

        base_termination = DoneTerm(
            func=brl_mdp.base_termination,
            params={
                "max_lin_vel": 15.0,
                "max_ang_vel": 10.0,
                "max_tilting": 0.8,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        hip_yaw_limits = DoneTerm(
            func=brl_mdp.joint_pos_out_of_bounds,
            params={
                "bounds": (-0.5, 0.5),
                "asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "R_HipYaw_Joint"]),
            },
        )
        foot_crossing = DoneTerm(
            func=brl_mdp.foot_crossing_yaw_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "min_y_dist": 0.16,
            },
        )
        # System-level foot-to-foot collision termination (physics contact only).
        foot_foot_collision = DoneTerm(
            func=brl_mdp.feet_collision,
            params={
                "sensor_name": "rf_foot_to_lf_contact",
                "threshold": 1.0,
            },
        )

    @configclass
    class ArmTerminationCfg:
        illegal_contact = DoneTerm(
            func=brl_mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "L_Armlink_Link",
                        "L_Elbow_Link",
                        "L_Wrist1_Link",
                        "R_Armlink_Link",
                        "R_Elbow_Link",
                        "R_Wrist1_Link",
                        "L_Forearm_Link",
                        "R_Forearm_Link",
                    ],
                ),
                # effectively disable arm illegal_contact termination for now
                "threshold": 1.0e9,
            },
        )

    @configclass
    class TimeOutTerminationCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    leg: LegTerminationCfg = LegTerminationCfg()
    arm: ArmTerminationCfg = ArmTerminationCfg()
    time_out: TimeOutTerminationCfg = TimeOutTerminationCfg()



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    pass


##
# Environment configuration
##


@configclass
class TocabiArmMotionEnvCfg(ManagerBasedRLEnvCfg):
    DEPLOYMENT_TRAINING = True
    ADD_BALL = False
    enable_randomization = True
    enable_torque_bias = True

    viewer = ViewerCfg(eye=(2.0, -2.0, 0.5), origin_type="asset_root", asset_name="robot")

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    if DEPLOYMENT_TRAINING:
        events: EventsDeployCfg = EventsDeployCfg()
        scene.robot.init_state.pos = (0.0, 0.0, 0.91)

        # ✅ tocabi joint name 기준으로 재작성
        scene.robot.init_state.joint_pos = {
            # legs
            "L_HipYaw_Joint": 0.0,
            "L_HipRoll_Joint": 0.0,
            "L_HipPitch_Joint": -0.24,
            "L_Knee_Joint": 0.6,
            "L_AnklePitch_Joint": -0.36,
            "L_AnkleRoll_Joint": 0.0,

            "R_HipYaw_Joint": 0.0,
            "R_HipRoll_Joint": 0.0,
            "R_HipPitch_Joint": -0.24,
            "R_Knee_Joint": 0.6,
            "R_AnklePitch_Joint": -0.36,
            "R_AnkleRoll_Joint": 0.0,

            # arms (일단 "전부 사용" 방침이면 전부 넣어주는 게 깔끔)
            "L_Shoulder1_Joint": 0.3,
            "L_Shoulder2_Joint": 0.3,
            "L_Shoulder3_Joint": 1.5,
            "L_Armlink_Joint": -1.27,
            "L_Elbow_Joint": -1.0,
            "L_Forearm_Joint": 0.0,
            "L_Wrist1_Joint": -1.0,
            "L_Wrist2_Joint": 0.0,
            "Neck_Joint": 0.0,
            "Head_Joint": 0.0,

            "R_Shoulder1_Joint": -0.3,
            "R_Shoulder2_Joint": -0.3,
            "R_Shoulder3_Joint": -1.5,
            "R_Armlink_Joint": 1.27,
            "R_Elbow_Joint": 1.0,
            "R_Forearm_Joint": 0.0,
            "R_Wrist1_Joint": 1.0,
            "R_Wrist2_Joint": 0.0,

            # trunk (원하면 추가)
            "Waist1_Joint": 0.0,
            "Waist2_Joint": 0.0,
            "Upperbody_Joint": 0.0,
        }
    else:
        events: EventsCfg = EventsCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    # whether arm action/update is enabled (wired from runner cfg)
    enable_arm: bool = True
    # actor observation history toggle/length (wired from runner cfg for enable only)
    enable_actor_history: bool = False
    actor_history_length: int = 40
    # base_ang_vel observation LPF (applied before gaussian noise in observation manager)
    enable_obs_base_ang_vel_lpf: bool = True
    obs_base_ang_vel_lpf_cutoff_hz: float = 60.0
    # joint_vel observation LPF (applied before gaussian noise in observation manager)
    enable_obs_joint_vel_lpf: bool = True
    obs_joint_vel_lpf_cutoff_hz: float = 25.0
    # episodic Gaussian bias on joint_pos observations (sampled at each reset)
    enable_obs_joint_pos_episode_bias: bool = False
    obs_joint_pos_episode_bias_sigma: float = 0.02
    # scheduled push disturbance (episode progress-based, one-shot per window)
    enable_scheduled_push: bool = True
    # joint init randomization at reset (all joints)
    enable_joint_init_randomization: bool = False
    joint_init_position_range: tuple[float, float] = (-0.1, 0.1)
    joint_init_velocity_range: tuple[float, float] = (-0.1, 0.1)
    push_force_range: tuple[float, float] = (-50.0, 50.0)
    # lateral (Y) axis scale for elliptical push distribution; 1.0 keeps isotropic XY
    push_lateral_scale: float = 1.0
    push_torque_range: tuple[float, float] = (-1.5, 1.5)
    # pulse half-span h (center->edge ticks) sampled uniformly from closed interval [min, max].
    # actual duration is (2*h + 1) ticks, so left/right are always symmetric.
    push_pulse_halfspan_min_steps: int = 0
    push_pulse_halfspan_max_steps: int = 2
    push_window_1: tuple[float, float] = (0.2, 0.4)
    push_window_2: tuple[float, float] = (0.6, 0.8)
    # push curriculum: linearly ramp scheduled push strength from start->end epoch, then keep max.
    push_curriculum_start_epoch: int = 3000
    push_curriculum_end_epoch: int = 10000
    # RSL-RL on-policy default update horizon (env steps per policy iteration)
    push_curriculum_steps_per_epoch: int = 24

    def _apply_arm_enable(self):
        if hasattr(self.actions, "leg_joint_pos"):
            # Legs should always follow mapped actions; never gate them with arm flag.
            self.actions.leg_joint_pos.disable_arm_action = False
        if hasattr(self.actions, "arm_joint_pos"):
            self.actions.arm_joint_pos.scale = 1.0 if self.enable_arm else 0.0
            self.actions.arm_joint_pos.disable_arm_action = not self.enable_arm

    def _apply_actor_history(self):
        """Apply optional history to actor observations, excluding last-action terms."""
        hist_len = int(self.actor_history_length) if self.enable_actor_history else 0
        if hist_len < 0:
            hist_len = 0
        one_step_terms = {
            "phase_sin",
            "phase_cos",
            "velocity_commands",
            "last_leg_action",
            "last_arm_action",
        }
        for group_name in ("leg_actor", "arm_actor"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            # Do not use group-level override because last_action must stay single-step.
            if hasattr(group, "history_length"):
                group.history_length = None
            if hasattr(group, "flatten_history_dim"):
                group.flatten_history_dim = True
            for term_name, term_cfg in vars(group).items():
                if term_name in {"enable_corruption", "concatenate_terms", "history_length", "flatten_history_dim"}:
                    continue
                if not hasattr(term_cfg, "func"):
                    continue
                use_hist = 0 if term_name in one_step_terms else hist_len
                if hasattr(term_cfg, "history_length"):
                    term_cfg.history_length = use_hist
                if hasattr(term_cfg, "flatten_history_dim"):
                    term_cfg.flatten_history_dim = True

    def _apply_obs_joint_pos_episode_bias(self):
        """Apply episodic joint-pos bias toggle/sigma to all joint_pos observation terms."""
        for group_name in ("leg_actor", "leg_critic", "arm_actor", "arm_critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            term_cfg = getattr(group, "joint_pos", None)
            if term_cfg is None or not hasattr(term_cfg, "params"):
                continue
            params = dict(getattr(term_cfg, "params", {}) or {})
            params["episodic_bias_enabled"] = bool(self.enable_obs_joint_pos_episode_bias)
            params["episodic_bias_sigma"] = float(self.obs_joint_pos_episode_bias_sigma)
            params.setdefault("episodic_bias_key", f"{group_name}_joint_pos")
            term_cfg.params = params

    def _apply_obs_joint_vel_lpf(self):
        """Apply LPF toggle/cutoff to all joint_vel observation terms."""
        for group_name in ("leg_actor", "leg_critic", "arm_actor", "arm_critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            term_cfg = getattr(group, "joint_vel", None)
            if term_cfg is None or not hasattr(term_cfg, "params"):
                continue
            params = dict(getattr(term_cfg, "params", {}) or {})
            params["lpf_enabled"] = bool(self.enable_obs_joint_vel_lpf)
            params["lpf_cutoff_hz"] = float(self.obs_joint_vel_lpf_cutoff_hz)
            params.setdefault("lpf_key", f"{group_name}_joint_vel")
            term_cfg.params = params

    def _apply_obs_base_ang_vel_lpf(self):
        """Apply LPF toggle/cutoff to all base_ang_vel observation terms."""
        for group_name in ("leg_actor", "leg_critic", "arm_actor", "arm_critic"):
            group = getattr(self.observations, group_name, None)
            if group is None:
                continue
            term_cfg = getattr(group, "base_ang_vel", None)
            if term_cfg is None or not hasattr(term_cfg, "params"):
                continue
            params = dict(getattr(term_cfg, "params", {}) or {})
            params["asset_cfg"] = params.get("asset_cfg", SceneEntityCfg("robot"))
            params["lpf_enabled"] = bool(self.enable_obs_base_ang_vel_lpf)
            params["lpf_cutoff_hz"] = float(self.obs_base_ang_vel_lpf_cutoff_hz)
            params.setdefault("lpf_key", f"{group_name}_base_ang_vel")
            term_cfg.params = params

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20
        self.video_length_s = 3

        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # one resample in the middle of each episode
        self.commands.base_velocity.resampling_time_range = (
            self.episode_length_s * 0.4,
            self.episode_length_s * 0.6,
        )
        # torque bias toggle for leg/arm action
        for term_name in ("leg_joint_pos", "arm_joint_pos"):
            term = getattr(self.actions, term_name, None)
            if term is not None and hasattr(term, "enable_torque_bias"):
                term.enable_torque_bias = self.enable_torque_bias

        # arm enable/disable toggles (scale + arm action masking)
        self._apply_arm_enable()
        self._apply_actor_history()
        self._apply_obs_base_ang_vel_lpf()
        self._apply_obs_joint_pos_episode_bias()
        self._apply_obs_joint_vel_lpf()
        print(
            f"[CFG] enable_arm={self.enable_arm} "
            f"leg.disable_arm_action={getattr(self.actions.leg_joint_pos, 'disable_arm_action', None)} "
            f"arm.scale={getattr(self.actions.arm_joint_pos, 'scale', None)} "
            f"arm.disable_arm_action={getattr(self.actions.arm_joint_pos, 'disable_arm_action', None)} "
            f"enable_actor_history={self.enable_actor_history} "
            f"actor_history_length={self.actor_history_length} "
            f"obs_base_ang_vel_lpf={self.enable_obs_base_ang_vel_lpf} "
            f"obs_base_ang_vel_lpf_cutoff_hz={self.obs_base_ang_vel_lpf_cutoff_hz} "
            f"obs_joint_vel_lpf={self.enable_obs_joint_vel_lpf} "
            f"obs_joint_vel_lpf_cutoff_hz={self.obs_joint_vel_lpf_cutoff_hz}",
            flush=True,
        )
        # pass q_init to action terms so non-controlled joints track init pose
        if hasattr(self.scene, "robot") and hasattr(self.scene.robot, "init_state"):
            q_init_dict = getattr(self.scene.robot.init_state, "joint_pos", None)
            if isinstance(q_init_dict, dict):
                for term_name in ("leg_joint_pos", "arm_joint_pos"):
                    term = getattr(self.actions, term_name, None)
                    if term is not None and hasattr(term, "q_init"):
                        term.q_init = q_init_dict
                    if term is not None and hasattr(term, "set_q_init"):
                        term.set_q_init(q_init_dict)

        if hasattr(self.scene, "height_scanner"):
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, "contact_forces"):
            self.scene.contact_forces.update_period = self.sim.dt

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # joint init randomization toggle (training path)
        if getattr(self, "enable_randomization", True) and hasattr(self, "events") and self.events is not None:
            if getattr(self, "enable_joint_init_randomization", True):
                self.events.reset_robot_joints = EventTerm(
                    func=mdp.reset_joints_by_offset,
                    mode="reset",
                    params={
                        "position_range": tuple(self.joint_init_position_range),
                        "velocity_range": tuple(self.joint_init_velocity_range),
                    },
                )
            else:
                self.events.reset_robot_joints = EventTerm(
                    func=mdp.reset_joints_by_offset,
                    mode="reset",
                    params={
                        "position_range": (0.0, 0.0),
                        "velocity_range": (0.0, 0.0),
                    },
                )

        # training: toggle randomization
        if not self.enable_randomization and hasattr(self, "events") and self.events is not None:
            # disable randomization terms
            self.events.physics_material = None
            if hasattr(self.events, "add_base_mass"):
                self.events.add_base_mass = None
            if hasattr(self.events, "randomize_rigid_body_com"):
                self.events.randomize_rigid_body_com = None
            if hasattr(self.events, "randomize_rigid_body_mass"):
                self.events.randomize_rigid_body_mass = None
            if hasattr(self.events, "randomize_armature_scale"):
                self.events.randomize_armature_scale = None
            if hasattr(self.events, "randomize_armature_friction"):
                self.events.randomize_armature_friction = None
            if hasattr(self.events, "randomize_damping"):
                self.events.randomize_damping = None
            if hasattr(self.events, "base_external_force_torque"):
                self.events.base_external_force_torque = None
            if hasattr(self.events, "push_robot"):
                self.events.push_robot = None
            if hasattr(self.events, "apply_external_force"):
                self.events.apply_external_force = None

            # deterministic reset
            self.events.reset_base = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.0, 0.0),
                        "y": (-0.0, 0.0),
                        "z": (-0.0, 0.0),
                        "roll": (-0.0, 0.0),
                        "pitch": (-0.0, 0.0),
                        "yaw": (-0.0, 0.0),
                    },
                    "velocity_range": {
                        "x": (-0.0, 0.0),
                        "y": (-0.0, 0.0),
                        "z": (-0.0, 0.0),
                        "roll": (-0.0, 0.0),
                        "pitch": (-0.0, 0.0),
                        "yaw": (-0.0, 0.0),
                    },
                },
            )
            self.events.reset_robot_joints = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.0, 0.0),
                    "velocity_range": (-0.0, 0.0),
                },
            )


@configclass
class TocabiArmMotionEnvCfg_PLAY(TocabiArmMotionEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # play: contact processing 켜서 센서/접촉 확인
        self.sim.disable_contact_processing = False

        # play: 가볍게
        self.scene.num_envs = 3
        self.scene.env_spacing = 2.5

        # terrain: 고정 (curriculum/레벨 끄기)
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # GRF sensors (tocabi foot prim path가 right_foot/left_foot인 경우)
        self.scene.rf_GRF = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/R_Foot_Link",
            debug_vis=False,
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/RightFootContactSensor"),
            max_contact_data_count_per_env=5,
        )

        self.scene.lf_GRF = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/L_Foot_Link",
            debug_vis=False,
            filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
            visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/LeftFootContactSensor"),
            max_contact_data_count_per_env=5,
        )

        # play: 관측 노이즈 끄기
        self.observations.leg_actor.enable_corruption = False
        self.observations.arm_actor.enable_corruption = False

        # play: 랜덤화/외란 끄기
        self.enable_scheduled_push = False
        self.events.physics_material = None
        if hasattr(self.events, "add_base_mass"):
            self.events.add_base_mass = None
        if hasattr(self.events, "randomize_rigid_body_com"):
            self.events.randomize_rigid_body_com = None
        if hasattr(self.events, "randomize_rigid_body_mass"):
            self.events.randomize_rigid_body_mass = None
        if hasattr(self.events, "randomize_armature_scale"):
            self.events.randomize_armature_scale = None
        if hasattr(self.events, "randomize_armature_friction"):
            self.events.randomize_armature_friction = None
        if hasattr(self.events, "randomize_damping"):
            self.events.randomize_damping = None
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "apply_external_force"):
            self.events.apply_external_force = None

        # play debug: use randomized reset to check penetration/robustness
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

        if getattr(self, "enable_joint_init_randomization", True):
            self.events.reset_robot_joints = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": tuple(self.joint_init_position_range),
                    "velocity_range": tuple(self.joint_init_velocity_range),
                },
            )
        else:
            self.events.reset_robot_joints = EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                },
            )

        # play: 사실상 무한 에피소드 + 커맨드 고정(정지)
        self.episode_length_s = int(1e7)
        self.commands.base_velocity.resampling_time_range = (int(1e7), int(1e7))
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.step_command.step_period_range = (40.0, 40.0)

"""
TODO:

1) rotor_inertia (=joint armature) is not set yet (v)
2) In env.scene["robot"].data, body_names and joint_names are not in the correct order.
3) Camera View (v)
4) Velocity Arrow change / World-frame vel command (v)
5) Foot step arrow visualization (v)
6) Plot contact forces (v)
7) Termination reward (v)
8) Training speed too slow
9) URDF color update (v) 
10) Video recording speed to real-time (v)
11) Joint Jacobian transfer
12) Keyboard control of the velocity command (v)
13) Option to disable logging (v)
14) soft joint torque / vel limit is weird? (v)
15) Rest of the config setting match with IsaacGym (v)
16) Video recording / Screenshot / Animation for play script (v)
17) Make full joint urdf / arm only urdf
18) Find the difference between lab vanilla and IsaacGyn vanilla #! Because of normalization?
19) Rendering speed with --cpu is too slow
20) Save the code / Load the cfg files when running play script
21) Non-noisy critic observation (v)
22) With camera, I cannot start the training? (v)

"""
