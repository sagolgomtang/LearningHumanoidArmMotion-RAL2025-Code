# extensions/tocabi/utils/constants.py
from __future__ import annotations

# Pinocchio order로 쓰겠다는 의미였지만,
# 지금은 "MJCF에 정의된 actuated joint 순서"를 그대로 Pin order로 가정해서 넣습니다.
# (virtual_joint/free joint는 제외)

PIN_JOINT_NAMES = [
    # --- Left leg (6) ---
    "L_HipYaw_Joint",
    "L_HipRoll_Joint",
    "L_HipPitch_Joint",
    "L_Knee_Joint",
    "L_AnklePitch_Joint",
    "L_AnkleRoll_Joint",

    # --- Right leg (6) ---
    "R_HipYaw_Joint",
    "R_HipRoll_Joint",
    "R_HipPitch_Joint",
    "R_Knee_Joint",
    "R_AnklePitch_Joint",
    "R_AnkleRoll_Joint",

    # --- Waist (3) ---
    "Waist1_Joint",
    "Waist2_Joint",
    "Upperbody_Joint",

    # --- Left arm (8) ---
    "L_Shoulder1_Joint",
    "L_Shoulder2_Joint",
    "L_Shoulder3_Joint",
    "L_Armlink_Joint",
    "L_Elbow_Joint",
    "L_Forearm_Joint",
    "L_Wrist1_Joint",
    "L_Wrist2_Joint",

    # --- Neck/Head (2) ---
    "Neck_Joint",
    "Head_Joint",

    # --- Right arm (8) ---
    "R_Shoulder1_Joint",
    "R_Shoulder2_Joint",
    "R_Shoulder3_Joint",
    "R_Armlink_Joint",
    "R_Elbow_Joint",
    "R_Forearm_Joint",
    "R_Wrist1_Joint",
    "R_Wrist2_Joint",
]
