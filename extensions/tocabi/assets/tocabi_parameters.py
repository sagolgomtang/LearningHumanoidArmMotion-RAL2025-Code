"""
This file contains the humanoid parameters for the robot.
    - effort_limit: The maximum torque that can be applied to the joint.
    - velocity_limit: The maximum velocity that can be applied to the joint.
    - stiffness: The stiffness of the joint PD controller.
    - damping: The damping of the joint PD controller.
    - rotor_inertias: The inertia of the rotor.
    - gear_ratio: The gear ratio of the joint.
    - armature: The armature of the joint.
    - friction: The friction of the joint.
"""

from dataclasses import dataclass

# Patterns you said you use:
# LEG_PATTERNS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
# ARM_PATTERNS = ["shoulder1", "shoulder2", "shoulder3", "armlink", "elbow", "forearm", "wrist1", "wrist2"]


@dataclass(frozen=True)
class effort_limit:
    # legs
    hip_yaw      = 333.0
    hip_roll     = 232.0
    hip_pitch    = 263.0
    knee         = 289.0
    ankle_pitch  = 222.0
    ankle_roll   = 166.0

    # arms (rename shoulder_* -> shoulder1/2/3)
    shoulder1    = 64.0
    shoulder2    = 64.0
    shoulder3    = 64.0
    armlink      = 64.0
    elbow        = 23.0
    forearm      = 23.0
    wrist1       = 10.0
    wrist2       = 10.0

    # trunk/head
    waist1       = 303.0
    waist2       = 303.0
    upperbody    = 303.0
    neck         = 10.0
    head         = 10.0


@dataclass(frozen=True)
class velocity_limit:
    # legs
    hip_yaw      = 48.0
    hip_roll     = 48.0
    hip_pitch    = 40.0
    knee         = 20.0
    ankle_pitch  = 24.0
    ankle_roll   = 24.0

    # arms
    shoulder1    = 50.0
    shoulder2    = 50.0
    shoulder3    = 50.0
    armlink      = 50.0
    elbow        = 50.0
    forearm      = 50.0
    wrist1       = 50.0
    wrist2       = 50.0

    # trunk/head
    waist1       = 50.0
    waist2       = 50.0
    upperbody    = 50.0
    neck         = 50.0
    head         = 50.0


from dataclasses import dataclass

@dataclass(frozen=True)
class stiffness:
    # legs (K/9)
    hip_yaw      = 0.0
    hip_roll     = 0.0
    hip_pitch    = 0.0
    knee         = 0.0
    ankle_pitch  = 0.0
    ankle_roll   = 0.0

    # waist / upper body (K/9)
    waist1       = 0.0
    waist2       = 0.0
    upperbody    = 0.0

    # neck / head (K/9)
    neck         = 0.0
    head         = 0.0

    # arms (K/9)
    shoulder1    = 0.0
    shoulder2    = 0.0
    shoulder3    = 0.0
    armlink      = 0.0
    elbow        = 0.0
    forearm      = 0.0
    wrist1       = 0.0
    wrist2       = 0.0


@dataclass(frozen=True)
class damping:
    # legs (D/3)
    hip_yaw      = 1.0
    hip_roll     = 1.0
    hip_pitch    = 1.0
    knee         = 1.0
    ankle_pitch  = 1.0
    ankle_roll   = 1.0

    # waist / upper body (D/3)
    waist1       = 1.0
    waist2       = 1.0
    upperbody    = 1.0

    # neck / head (D/3)
    neck         = 1.0
    head         = 1.0

    # arms (D/3)
    shoulder1    = 1.0
    shoulder2    = 1.0
    shoulder3    = 1.0
    armlink      = 1.0
    elbow        = 1.0
    forearm      = 1.0
    wrist1       = 1.0
    wrist2       = 1.0

@dataclass(frozen=True)
class armature:
    hip_yaw      = 0.614
    hip_roll     = 0.862
    hip_pitch    = 1.09
    knee         = 1.09
    ankle_pitch  = 1.09
    ankle_roll   = 0.360

    shoulder1    = 0.18
    shoulder2    = 0.18
    shoulder3    = 0.18
    armlink      = 0.18
    elbow        = 0.0032
    forearm      = 0.0032
    wrist1       = 0.0032
    wrist2       = 0.0032

    waist1       = 0.078
    waist2       = 0.078
    upperbody    = 0.078
    neck         = 0.0032
    head         = 0.0032


@dataclass(frozen=True)
class friction:
    hip_yaw      = 0.3
    hip_roll     = 0.3
    hip_pitch    = 0.3
    knee         = 0.3
    ankle_pitch  = 0.3
    ankle_roll   = 0.3

    shoulder1    = 0.0
    shoulder2    = 0.0
    shoulder3    = 0.0
    armlink      = 0.0
    elbow        = 0.0
    forearm      = 0.0
    wrist1       = 0.0
    wrist2       = 0.0

    waist1       = 0.0
    waist2       = 0.0
    upperbody    = 0.0
    neck         = 0.0
    head         = 0.0
