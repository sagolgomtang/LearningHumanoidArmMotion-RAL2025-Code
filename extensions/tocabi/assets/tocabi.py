from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from extensions import ISAACLAB_BRL_ROOT_DIR

from extensions.tocabi.assets.tocabi_parameters import effort_limit  # confirmed
from extensions.tocabi.assets.tocabi_parameters import velocity_limit, stiffness, damping, armature, friction  # keep


# ====== 너가 정한 "파라미터 attribute 이름" (유지) ======
LEG_PATTERNS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
ARM_PATTERNS = ["shoulder1", "shoulder2", "shoulder3", "elbow"]
ARM_LOCKED_PATTERNS = ["armlink", "forearm", "wrist1", "wrist2"]
TRUNK_PATTERNS = ["waist1", "waist2", "upperbody", "neck", "head"]

# ====== URDF joint name(Available strings) 기준 regex ======
LEG_JOINT_EXPR = [
    r".*_HipYaw_Joint",
    r".*_HipRoll_Joint",
    r".*_HipPitch_Joint",
    r".*_Knee_Joint",
    r".*_AnklePitch_Joint",
    r".*_AnkleRoll_Joint",
]
ARM_JOINT_EXPR = [
    r".*_Shoulder1_Joint",
    r".*_Shoulder2_Joint",
    r".*_Shoulder3_Joint",
    r".*_Elbow_Joint",
]
ARM_LOCKED_JOINT_EXPR = [
    r".*_Armlink_Joint",
    r".*_Forearm_Joint",
    r".*_Wrist1_Joint",
    r".*_Wrist2_Joint",
]
TRUNK_JOINT_EXPR = [
    r"Waist1_Joint",
    r"Waist2_Joint",
    r"Upperbody_Joint",
    r"Neck_Joint",
    r"Head_Joint",
]


TOCABI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        link_density=1.0e-3,
        asset_path=f"{ISAACLAB_BRL_ROOT_DIR}/resources/tocabi/urdf/dyros_tocabi.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.93),
        joint_pos={
            r".*_HipYaw_Joint": 0.0,
            r".*_HipRoll_Joint": 0.0,
            r".*_HipPitch_Joint": -0.24,
            r".*_Knee_Joint": 0.6,
            r".*_AnklePitch_Joint": -0.36,
            r".*_AnkleRoll_Joint": 0.0,
            r"Waist1_Joint": 0.0,
            r"Waist2_Joint": 0.0,
            r"Upperbody_Joint": 0.0,
            r"L_Shoulder1_Joint": 0.3,
            r"L_Shoulder2_Joint": 0.3,
            r"L_Shoulder3_Joint": 1.5,
            r"L_Armlink_Joint": -1.27,
            r"L_Elbow_Joint": -1.0,
            r"L_Forearm_Joint": 0.0,
            r"L_Wrist1_Joint": -1.0,
            r"L_Wrist2_Joint": 0.0,
            r"Neck_Joint": 0.0,
            r"Head_Joint": 0.0,
            r"R_Shoulder1_Joint": -0.3,
            r"R_Shoulder2_Joint": -0.3,
            r"R_Shoulder3_Joint": -1.5,
            r"R_Armlink_Joint": 1.27,
            r"R_Elbow_Joint": 1.0,
            r"R_Forearm_Joint": 0.0,
            r"R_Wrist1_Joint": 1.0,
            r"R_Wrist2_Joint": 0.0,
        },
        joint_vel={r".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    soft_joint_vel_limit_factor=0.9,
    soft_joint_torque_limit_factor=0.8,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=LEG_JOINT_EXPR,

            effort_limit={expr: getattr(effort_limit, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},

            # unconfirmed: 유지 (삭제/주석처리/값변경 안 함) + key만 URDF regex로 맞춤
            velocity_limit={expr: getattr(velocity_limit, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},
            stiffness={expr: getattr(stiffness, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},
            damping={expr: getattr(damping, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},
            armature={expr: getattr(armature, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},
            friction={expr: getattr(friction, p) for expr, p in zip(LEG_JOINT_EXPR, LEG_PATTERNS)},

            apply_humanoid_jacobian=True,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=ARM_JOINT_EXPR,

            effort_limit={expr: getattr(effort_limit, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},

            # unconfirmed: 유지 (삭제/주석처리/값변경 안 함) + key만 URDF regex로 맞춤
            velocity_limit={expr: getattr(velocity_limit, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},
            stiffness={expr: getattr(stiffness, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},
            damping={expr: getattr(damping, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},
            armature={expr: getattr(armature, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},
            friction={expr: getattr(friction, p) for expr, p in zip(ARM_JOINT_EXPR, ARM_PATTERNS)},
            min_delay=0,
            max_delay=2,
        ),
        "arm_locked": IdealPDActuatorCfg(
            joint_names_expr=ARM_LOCKED_JOINT_EXPR,

            effort_limit={expr: getattr(effort_limit, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},

            velocity_limit={expr: getattr(velocity_limit, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},
            stiffness={expr: getattr(stiffness, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},
            damping={expr: getattr(damping, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},
            armature={expr: getattr(armature, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},
            friction={expr: getattr(friction, p) for expr, p in zip(ARM_LOCKED_JOINT_EXPR, ARM_LOCKED_PATTERNS)},
        ),
        "trunk_locked": IdealPDActuatorCfg(
            joint_names_expr=TRUNK_JOINT_EXPR,

            effort_limit={expr: getattr(effort_limit, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
            velocity_limit={expr: getattr(velocity_limit, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
            stiffness={expr: getattr(stiffness, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
            damping={expr: getattr(damping, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
            armature={expr: getattr(armature, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
            friction={expr: getattr(friction, p) for expr, p in zip(TRUNK_JOINT_EXPR, TRUNK_PATTERNS)},
        ),
    },
)
