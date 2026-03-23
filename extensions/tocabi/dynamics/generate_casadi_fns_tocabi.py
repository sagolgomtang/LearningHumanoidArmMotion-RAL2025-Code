"""
Generate CasADi functions (FK / Jacobian / M,C,G / CoM / Centroidal terms) from TOCABI URDF using Pinocchio.

Output directory: PINOCCHIO_CASADI_FUNCTIONS_DIR (e.g., extensions/tocabi/dynamics/casadi_fns)

IMPORTANT:
- Save ONLY using Function.serialize() (latin1 bytes) to avoid CasADi Function.load() stream issues.
- Later, load ONLY using Function.deserialize() from latin1 bytes.
"""

import os
import sys
from pathlib import Path

import numpy as np

# ✅ 1) CasADi 먼저 import해서 conda casadi로 고정
import casadi as ca
from casadi import Function, SX

print("[gen] casadi:", ca.__version__, ca.__file__)

# ✅ 2) Pinocchio는 그 다음 (append 유지)
sys.path.append("/opt/openrobots/lib/python3.10/site-packages")
import pinocchio as pin
from pinocchio import casadi as cpin

print("[gen] pinocchio:", pin.__version__, pin.__file__)

from extensions import ISAACLAB_BRL_ROOT_DIR
from extensions.tocabi.dynamics import PINOCCHIO_CASADI_FUNCTIONS_DIR


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_serialized(fn: Function, out_path: str):
    """
    Save CasADi Function in serialize() format as latin1 BYTES.
    This avoids Function.load() binary stream incompatibility.
    """
    s = fn.serialize()
    Path(out_path).write_bytes(s.encode("latin1"))


def print_model_info(model: pin.Model):
    print(f"[Pinocchio] model name: {model.name}")
    print(f"  nq={model.nq}, nv={model.nv}, njoints={model.njoints}, nframes={model.nframes}")
    print("\n[Joints]")
    for j in range(model.njoints):
        print(f"  {j:03d}  {model.names[j]:30s}  nq={model.joints[j].nq}  nv={model.joints[j].nv}")
    print("\n[Frames] (first 50)")
    for i in range(min(model.nframes, 50)):
        print(f"  {i:03d}  {model.frames[i].name}")
    if model.nframes > 50:
        print("  ...")


def assert_frame_exists(model: pin.Model, frame_name: str):
    try:
        _ = model.getFrameId(frame_name)
    except Exception as e:
        raise RuntimeError(f"Frame not found: '{frame_name}'. Check URDF frame names.") from e


def save_tocabi_casadi_functions(urdf_name: str, urdf_path: str, frame_names: list[str]):
    ensure_dir(PINOCCHIO_CASADI_FUNCTIONS_DIR)

    # ---- Build numeric model (sanity check) ----
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    print_model_info(model)

    for fn in frame_names:
        assert_frame_exists(model, fn)

    # ---- Build CasADi model ----
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    q = SX.sym("q", model.nq)          # generalized coordinates
    qdot = SX.sym("qdot", model.nv)    # generalized velocities
    qddot = SX.sym("qddot", model.nv)  # generalized accelerations

    outdir = Path(PINOCCHIO_CASADI_FUNCTIONS_DIR)

    # ---- Dynamics terms ----
    M = cpin.crba(cmodel, cdata, q)
    C = cpin.computeCoriolisMatrix(cmodel, cdata, q, qdot) @ qdot
    G = cpin.computeGeneralizedGravity(cmodel, cdata, q)

    M_fn = Function(f"M_{urdf_name}", [q], [M]).expand()
    C_fn = Function(f"C_{urdf_name}", [q, qdot], [C]).expand()
    G_fn = Function(f"G_{urdf_name}", [q], [G]).expand()

    save_serialized(M_fn, str(outdir / f"M_{urdf_name}.casadi"))
    save_serialized(C_fn, str(outdir / f"C_{urdf_name}.casadi"))
    save_serialized(G_fn, str(outdir / f"G_{urdf_name}.casadi"))

    # ---- CoM / Centroidal terms ----
    CMM = cpin.computeCentroidalMap(cmodel, cdata, q)
    dCMM = cpin.computeCentroidalMapTimeVariation(cmodel, cdata, q, qdot)

    CM = cpin.computeCentroidalMomentum(cmodel, cdata, q, qdot)   # spatial momentum
    dCM = cpin.computeCentroidalMomentumTimeVariation(cmodel, cdata, q, qdot, qddot)
    CoM = cpin.centerOfMass(cmodel, cdata, q)

    CMM_fn = Function(f"CMM_{urdf_name}", [q], [CMM]).expand()
    dCMM_fn = Function(f"dCMM_{urdf_name}", [q, qdot], [dCMM]).expand()
    CM_fn = Function(f"CM_{urdf_name}", [q, qdot], [CM.linear, CM.angular]).expand()
    dCM_fn = Function(f"dCM_{urdf_name}", [q, qdot, qddot], [dCM.linear, dCM.angular]).expand()
    CoM_fn = Function(f"CoM_{urdf_name}", [q], [CoM]).expand()

    save_serialized(CMM_fn, str(outdir / f"CMM_{urdf_name}.casadi"))
    save_serialized(dCMM_fn, str(outdir / f"dCMM_{urdf_name}.casadi"))
    save_serialized(CM_fn, str(outdir / f"CM_{urdf_name}.casadi"))
    save_serialized(dCM_fn, str(outdir / f"dCM_{urdf_name}.casadi"))
    save_serialized(CoM_fn, str(outdir / f"CoM_{urdf_name}.casadi"))

    # ---- Base frame pose (Pelvis_Link) ----
    base_frame = "Pelvis_Link"
    assert_frame_exists(model, base_frame)

    cpin.forwardKinematics(cmodel, cdata, q, qdot, qddot)
    cpin.updateFramePlacements(cmodel, cdata)
    base_id = cmodel.getFrameId(base_frame)
    base_oMf = cdata.oMf[base_id]
    base_pos = base_oMf.translation
    base_rot = base_oMf.rotation

    base_pos_fn = Function(f"base_pos_{urdf_name}", [q], [base_pos]).expand()
    base_rot_fn = Function(f"base_rot_{urdf_name}", [q], [base_rot]).expand()

    save_serialized(base_pos_fn, str(outdir / f"base_pos_{urdf_name}.casadi"))
    save_serialized(base_rot_fn, str(outdir / f"base_rot_{urdf_name}.casadi"))

    # ---- FK + Jacobian for selected frames ----
    for frame_name in frame_names:
        cpin.forwardKinematics(cmodel, cdata, q, qdot)
        cpin.updateFramePlacements(cmodel, cdata)

        frame_id = cmodel.getFrameId(frame_name)
        oMf = cdata.oMf[frame_id]
        pos = oMf.translation
        rot = oMf.rotation

        fk_fn = Function(f"fk_{frame_name}_{urdf_name}", [q], [pos, rot]).expand()
        save_serialized(fk_fn, str(outdir / f"fk_{frame_name}_{urdf_name}.casadi"))

        J = cpin.computeFrameJacobian(cmodel, cdata, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
        jac_fn = Function(f"jacobian_{frame_name}_{urdf_name}", [q], [J]).expand()
        save_serialized(jac_fn, str(outdir / f"jacobian_{frame_name}_{urdf_name}.casadi"))

    print("\n[Done] Saved CasADi functions to:")
    print(f"  {PINOCCHIO_CASADI_FUNCTIONS_DIR}")
    print("[Done] Note: saved as serialize() latin1 BYTES (load with Function.deserialize).")


if __name__ == "__main__":
    # ✅ suffix는 env에서 강제로 dyros_tocabi로 맞출 거라서 여기서도 동일하게 생성
    urdf_name = "dyros_tocabi"
    urdf_path = f"{ISAACLAB_BRL_ROOT_DIR}/resources/tocabi/urdf/{urdf_name}.urdf"

    frames_to_query = ["R_Foot_Link", "L_Foot_Link"]
    save_tocabi_casadi_functions(urdf_name, urdf_path, frames_to_query)
