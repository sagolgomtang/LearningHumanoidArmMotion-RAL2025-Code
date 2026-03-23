"""
Generates casadi functions for forward kinematics and jacobian computation using Pinocchio library.
https://github.com/stack-of-tasks/pinocchio
"""

import sys
import numpy as np

sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as ca
from casadi import Function, SX

from extensions import ISAACLAB_BRL_ROOT_DIR
from extensions.humanoid.dynamics import PINOCCHIO_CASADI_FUNCTIONS_DIR

np.set_printoptions(threshold=1000, linewidth=1000, precision=6)


# -------------------------
# helpers
# -------------------------
def list_frames(model):
    return [f.name for f in model.frames]


def get_frame_id(model, frame_name: str) -> int:
    """Get frame id or raise with helpful message."""
    try:
        return model.getFrameId(frame_name)
    except Exception as e:
        frames = list_frames(model)
        msg = (
            f"Frame '{frame_name}' not found.\n"
            f"Available frames (first 50): {frames[:50]}\n"
            f"Total frames: {len(frames)}"
        )
        raise RuntimeError(msg) from e


def find_first_existing_frame(model, candidates):
    """Return the first candidate frame name that exists in model.frames."""
    frames = set(list_frames(model))
    for name in candidates:
        if name in frames:
            return name
    return None


def test_forward_kinematics_from_urdf(urdf_path, frame_name=None, base_candidates=None):
    """
    Loads a URDF into a Pinocchio model, runs forward kinematics for a test joint config,
    and returns the final pose (position + rotation) of a given frame.
    """

    if base_candidates is None:
        # tocabi / common URDF naming candidates
        base_candidates = ["Pelvis_Link", "pelvis", "base", "base_link", "root"]

    print("==== Numeric Version ====")
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()

    print("model name:", model.name)
    print("nq:", model.nq, "nv:", model.nv)
    print("==== joints ====")
    for j in range(model.njoints):
        print(j, model.names[j], model.joints[j].nq, model.joints[j].nv)
    print("==== frames (first 50) ====")
    for j, fr in enumerate(model.frames[:50]):
        print(j, fr.name)

    # -------- test q --------
    # 기본은 neutral로(안전). 필요하면 아래 하드코딩 q로 테스트 가능.
    q_num = pin.neutral(model)
    q_dot_num = np.zeros(model.nv)
    q_ddot_num = np.zeros(model.nv)

    # (선택) 네가 넣은 하드코딩 테스트 벡터를 쓰고 싶으면 아래 주석 해제.
    # 단, 길이가 맞아야 한다.
    # q_num = np.array([...])
    # q_dot_num = np.array([...])
    # q_ddot_num = np.array([...])

    assert q_num.shape[0] == model.nq, f"q_num length mismatch: got {q_num.shape[0]}, expected {model.nq}"
    assert q_dot_num.shape[0] == model.nv, f"q_dot_num length mismatch: got {q_dot_num.shape[0]}, expected {model.nv}"
    assert q_ddot_num.shape[0] == model.nv, f"q_ddot_num length mismatch: got {q_ddot_num.shape[0]}, expected {model.nv}"

    pin.forwardKinematics(model, data, q_num, q_dot_num, q_ddot_num)
    pin.updateFramePlacements(model, data)

    if frame_name is not None:
        frame_id = get_frame_id(model, frame_name)
        oMf = data.oMf[frame_id]
        print(f"[FK] Frame {frame_name}\n  pos: {oMf.translation}\n  rot:\n{oMf.rotation}")

        J = pin.computeFrameJacobian(model, data, q_num, frame_id, pin.LOCAL_WORLD_ALIGNED)
        print(f"[J] shape: {J.shape}")

        v = pin.getFrameVelocity(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED)
        print(f"[V] lin: {v.linear}, ang: {v.angular}")

    base_name = find_first_existing_frame(model, base_candidates)
    if base_name is None:
        print("[WARN] No base frame candidate found. Skipping base pose print.")
        return

    base_id = get_frame_id(model, base_name)
    base_oMf = data.oMf[base_id]
    print(f"[BASE] using '{base_name}'\n  pos: {base_oMf.translation}\n  rot:\n{base_oMf.rotation}")

    # Dynamics
    M = pin.crba(model, data, q_num)
    C = pin.computeCoriolisMatrix(model, data, q_num, q_dot_num)
    G = pin.computeGeneralizedGravity(model, data, q_num)
    print(f"[M] {M.shape}, [C] {C.shape}, [G] {G.shape}")


def save_forward_kinematics(urdf_name, urdf_path, frame_names, base_candidates=None):
    """
    Saves the forward kinematics/dynamics casadi functions for multiple frames.
    Output files go to: PINOCCHIO_CASADI_FUNCTIONS_DIR
    """

    if base_candidates is None:
        base_candidates = ["Pelvis_Link", "pelvis", "base", "base_link", "root"]

    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    frames_set = set([f.name for f in cmodel.frames])
    for name in frame_names:
        if name not in frames_set:
            raise RuntimeError(f"Requested frame '{name}' not found. (Check URDF frame names)")

    base_name = find_first_existing_frame(cmodel, base_candidates)
    if base_name is None:
        raise RuntimeError(
            f"No base frame candidate found among {base_candidates}. "
            f"Check URDF frames and update base_candidates."
        )

    print("model name:", model.name)
    print("saving to:", PINOCCHIO_CASADI_FUNCTIONS_DIR)
    print("base frame:", base_name)

    q_sym = SX.sym("q_sym", model.nq)
    q_dot_sym = SX.sym("q_dot_sym", model.nv)
    q_ddot_sym = SX.sym("q_ddot_sym", model.nv)

    # ---- base pose ----
    cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym, q_ddot_sym)
    cpin.updateFramePlacements(cmodel, cdata)

    base_id = get_frame_id(cmodel, base_name)
    base_oMf = cdata.oMf[base_id]
    base_pos = base_oMf.translation
    base_rot = base_oMf.rotation

    Function(f"base_pos_{urdf_name}", [q_sym], [base_pos]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_pos_{urdf_name}.casadi"
    )
    Function(f"base_rot_{urdf_name}", [q_sym], [base_rot]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/base_rot_{urdf_name}.casadi"
    )

    # ---- dynamics ----
    M = cpin.crba(cmodel, cdata, q_sym)
    C = cpin.computeCoriolisMatrix(cmodel, cdata, q_sym, q_dot_sym) @ q_dot_sym
    G = cpin.computeGeneralizedGravity(cmodel, cdata, q_sym)

    Function(f"M_{urdf_name}", [q_sym], [M]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{urdf_name}.casadi"
    )
    Function(f"C_{urdf_name}", [q_sym, q_dot_sym], [C]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/C_{urdf_name}.casadi"
    )
    Function(f"G_{urdf_name}", [q_sym], [G]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/G_{urdf_name}.casadi"
    )

    # ---- centroidal / CoM ----
    CMM = cpin.computeCentroidalMap(cmodel, cdata, q_sym)
    dCMM = cpin.computeCentroidalMapTimeVariation(cmodel, cdata, q_sym, q_dot_sym)
    CM = cpin.computeCentroidalMomentum(cmodel, cdata, q_sym, q_dot_sym)
    dCM = cpin.computeCentroidalMomentumTimeVariation(cmodel, cdata, q_sym, q_dot_sym, q_ddot_sym)
    CoM = cpin.centerOfMass(cmodel, cdata, q_sym)

    Function(f"CMM_{urdf_name}", [q_sym], [CMM]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CMM_{urdf_name}.casadi"
    )
    Function(f"dCMM_{urdf_name}", [q_sym, q_dot_sym], [dCMM]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCMM_{urdf_name}.casadi"
    )
    Function(f"CM_{urdf_name}", [q_sym, q_dot_sym], [CM.linear, CM.angular]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CM_{urdf_name}.casadi"
    )
    Function(f"dCM_{urdf_name}", [q_sym, q_dot_sym, q_ddot_sym], [dCM.linear, dCM.angular]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/dCM_{urdf_name}.casadi"
    )
    Function(f"CoM_{urdf_name}", [q_sym], [CoM]).expand().save(
        f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/CoM_{urdf_name}.casadi"
    )

    # ---- per-frame FK / Jacobian ----
    for frame_name in frame_names:
        cpin.forwardKinematics(cmodel, cdata, q_sym, q_dot_sym)
        cpin.updateFramePlacements(cmodel, cdata)

        frame_id = get_frame_id(cmodel, frame_name)
        coMf = cdata.oMf[frame_id]
        cpos = coMf.translation
        crot = coMf.rotation

        # FK (이름을 frame별로 유니크하게)
        Function(f"fk_{frame_name}_{urdf_name}", [q_sym], [cpos, crot]).expand().save(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}_{urdf_name}.casadi"
        )

        # Jacobian
        jac = cpin.computeFrameJacobian(cmodel, cdata, q_sym, frame_id, pin.LOCAL_WORLD_ALIGNED)
        Function(f"jacobian_{frame_name}_{urdf_name}", [q_sym], [jac]).expand().save(
            f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_{frame_name}_{urdf_name}.casadi"
        )


def load_forward_kinematics(urdf_name, urdf_path, frame_names):
    print("\n==== Load Casadi Functions ====")
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    q_num = pin.neutral(model)
    q_dot_num = np.zeros(model.nv)

    M_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/M_{urdf_name}.casadi")
    C_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/C_{urdf_name}.casadi")
    G_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/G_{urdf_name}.casadi")

    M = M_fn(q_num)
    C = C_fn(q_num, q_dot_num)
    G = G_fn(q_num)

    print(f"[M] {np.array(M).shape}")
    print(f"[C] {np.array(C).shape}")
    print(f"[G] {np.array(G).shape}")

    for frame_name in frame_names:
        fk_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/fk_{frame_name}_{urdf_name}.casadi")
        pos, rot = fk_fn(q_num)
        print(f"[FK] {frame_name} pos: {np.array(pos).reshape(-1)}")

        jac_fn = Function.load(f"{PINOCCHIO_CASADI_FUNCTIONS_DIR}/jacobian_{frame_name}_{urdf_name}.casadi")
        J = jac_fn(q_num)
        print(f"[J] {frame_name} shape: {np.array(J).shape}")


if __name__ == "__main__":
    urdf_name = "dyros_tocabi"
    urdf_path = f"{ISAACLAB_BRL_ROOT_DIR}/resources/tocabi/urdf/{urdf_name}.urdf"

    frame_to_query = "L_Foot_Link"
    test_forward_kinematics_from_urdf(urdf_path, frame_to_query)

    frames_to_query = ["R_Foot_Link", "L_Foot_Link"]
    save_forward_kinematics(urdf_name, urdf_path, frames_to_query)

    load_forward_kinematics(urdf_name, urdf_path, frames_to_query)
