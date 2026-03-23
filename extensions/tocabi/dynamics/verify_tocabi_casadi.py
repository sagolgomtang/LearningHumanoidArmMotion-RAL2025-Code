#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
excution example:
python extensions/tocabi/dynamics/verify_tocabi_casadi.py \
  --urdf /home/user/rl_ws/src/LearningHumanoidArmMotion-RAL2025-Code/resources/tocabi/URDF/dyros_tocabi.urdf \
  --casadi_dir /home/user/rl_ws/src/LearningHumanoidArmMotion-RAL2025-Code/extensions/tocabi/dynamics/casadi_fns \
  --urdf_name dyros_tocabi \
  --base_frame Pelvis_Link
"""

import argparse
import os
from pathlib import Path

import numpy as np
import casadi as ca
from casadi import Function

import pinocchio as pin
from pinocchio import casadi as cpin  # noqa: F401  (kept for compatibility, not used here)


def _as_col_dm(x: np.ndarray) -> ca.DM:
    """Convert numpy vector to CasADi DM column vector."""
    x = np.asarray(x).reshape(-1)
    return ca.DM(x).reshape((x.shape[0], 1))


def _normalize_quat_in_q(q: np.ndarray) -> np.ndarray:
    """
    Pinocchio free-flyer configuration order is:
      [x, y, z, qx, qy, qz, qw, joints...]
    Normalize the quaternion part to avoid NaNs in CasADi functions.
    """
    q = q.copy()
    quat = q[3:7].copy()
    n = np.linalg.norm(quat)
    if not np.isfinite(n) or n < 1e-12:
        q[3:7] = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        q[3:7] = quat / n
    return q


def _np_allclose(a, b, atol, rtol):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, np.nan
    diff = np.abs(a - b)
    if np.any(~np.isfinite(diff)):
        return False, np.nan
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    return ok, float(np.max(diff))


def discover_frames_from_dir(casadi_dir: Path, urdf_name: str):
    """
    Discover frames that have both fk and jacobian casadi files.
    Expected:
      fk_<Frame>_<urdf_name>.casadi
      jacobian_<Frame>_<urdf_name>.casadi
    """
    fk = {}
    jac = {}

    for p in casadi_dir.glob(f"fk_*_{urdf_name}.casadi"):
        # fk_<Frame>_<urdf>.casadi  -> frame is middle part
        stem = p.stem  # fk_L_Foot_Link_dyros_tocabi
        prefix = f"fk_"
        suffix = f"_{urdf_name}"
        frame = stem[len(prefix):-len(suffix)]
        fk[frame] = p

    for p in casadi_dir.glob(f"jacobian_*_{urdf_name}.casadi"):
        stem = p.stem  # jacobian_L_Foot_Link_dyros_tocabi
        prefix = "jacobian_"
        suffix = f"_{urdf_name}"
        frame = stem[len(prefix):-len(suffix)]
        jac[frame] = p

    frames = sorted(set(fk.keys()).intersection(set(jac.keys())))
    return frames


def _load_serialized_function(path: Path) -> Function:
    with open(path, "rb") as f:
        data = f.read()
    return Function.deserialize(data.decode("latin1"))


def load_required_functions(casadi_dir: Path, urdf_name: str):
    def _must(path: Path, label: str):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} casadi file: {path}")
        return path

    base_pos_p = _must(casadi_dir / f"base_pos_{urdf_name}.casadi", "base_pos")
    base_rot_p = _must(casadi_dir / f"base_rot_{urdf_name}.casadi", "base_rot")

    M_p = _must(casadi_dir / f"M_{urdf_name}.casadi", "M")
    C_p = _must(casadi_dir / f"C_{urdf_name}.casadi", "C")
    G_p = _must(casadi_dir / f"G_{urdf_name}.casadi", "G")

    base_pos_fn = _load_serialized_function(base_pos_p)
    base_rot_fn = _load_serialized_function(base_rot_p)

    M_fn = _load_serialized_function(M_p)
    C_fn = _load_serialized_function(C_p)
    G_fn = _load_serialized_function(G_p)

    return base_pos_fn, base_rot_fn, M_fn, C_fn, G_fn


def load_fk_jacobian(casadi_dir: Path, urdf_name: str, frame: str):
    fk_p = casadi_dir / f"fk_{frame}_{urdf_name}.casadi"
    jac_p = casadi_dir / f"jacobian_{frame}_{urdf_name}.casadi"
    if not fk_p.exists():
        raise FileNotFoundError(f"Missing FK casadi file: {fk_p}")
    if not jac_p.exists():
        raise FileNotFoundError(f"Missing Jacobian casadi file: {jac_p}")
    return _load_serialized_function(fk_p), _load_serialized_function(jac_p)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True, help="Path to dyros_tocabi.urdf")
    parser.add_argument("--casadi_dir", type=str, required=True, help="Directory containing *.casadi")
    parser.add_argument("--urdf_name", type=str, default="dyros_tocabi", help="URDF name used in filenames")
    parser.add_argument("--base_frame", type=str, default="Pelvis_Link", help="Base frame name (used for numeric check)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pos_atol", type=float, default=1e-6)
    parser.add_argument("--pos_rtol", type=float, default=1e-6)
    parser.add_argument("--rot_atol", type=float, default=1e-9)
    parser.add_argument("--rot_rtol", type=float, default=1e-9)
    parser.add_argument("--jac_atol", type=float, default=1e-5)
    parser.add_argument("--jac_rtol", type=float, default=1e-5)
    parser.add_argument("--dyn_atol", type=float, default=1e-5)
    parser.add_argument("--dyn_rtol", type=float, default=1e-5)
    parser.add_argument("--frames", nargs="*", default=None, help="Optional: explicit frame list to verify")
    args = parser.parse_args()

    URDF_PATH = Path(args.urdf).resolve()
    CASADI_DIR = Path(args.casadi_dir).resolve()
    urdf_name = args.urdf_name

    print_header("PATH CHECK")
    print(f"URDF_PATH     : {URDF_PATH}")
    print(f"CASADI_FNS_DIR: {CASADI_DIR}")

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    if not CASADI_DIR.exists():
        raise FileNotFoundError(f"CasADi dir not found: {CASADI_DIR}")

    # Build numeric Pinocchio model
    model = pin.buildModelFromUrdf(str(URDF_PATH), pin.JointModelFreeFlyer())
    data = model.createData()

    print(f"\n[Pinocchio] model name: {model.name}")
    print(f"  nq={model.nq}, nv={model.nv}, njoints={model.njoints}, nframes={model.nframes}")

    # Load required casadi fns
    base_pos_fn, base_rot_fn, M_fn, C_fn, G_fn = load_required_functions(CASADI_DIR, urdf_name)

    # Determine frames to verify
    if args.frames is not None and len(args.frames) > 0:
        frames = args.frames
    else:
        frames = discover_frames_from_dir(CASADI_DIR, urdf_name)

    if len(frames) == 0:
        print("\n[Warn] No frames discovered for FK/Jacobian verification.")
        print("       (Expected files: fk_<Frame>_<urdf>.casadi and jacobian_<Frame>_<urdf>.casadi)")
    else:
        print("\n[Frames to verify]")
        for fr in frames:
            print(f"  - {fr}")

    # Create a safe random test state
    rng = np.random.default_rng(args.seed)

    q = pin.neutral(model).copy()
    # randomize joint part only (after free-flyer 7 dof)
    if model.nq > 7:
        q[7:] = rng.uniform(low=-0.2, high=0.2, size=(model.nq - 7,))
    # also add small base translation so pos isn't trivially 0
    q[0:3] = rng.uniform(low=-0.1, high=0.1, size=(3,))
    q = _normalize_quat_in_q(q)

    v = rng.standard_normal(model.nv) * 0.1

    # Make casadi inputs (column vectors)
    q_ca = _as_col_dm(q)
    v_ca = _as_col_dm(v)

    # Run numeric FK / dynamics
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    # Base frame (Pelvis_Link by default)
    print_header("VERIFY: Pelvis (base_pos/base_rot)")
    base_frame = args.base_frame
    base_id = model.getFrameId(base_frame)
    oMf_base = data.oMf[base_id]
    pos_num = np.asarray(oMf_base.translation).reshape(3,)
    rot_num = np.asarray(oMf_base.rotation).reshape(3, 3)

    pos_ca = np.array(base_pos_fn(q_ca)).reshape(3,)
    rot_ca = np.array(base_rot_fn(q_ca)).reshape(3, 3)

    ok_pos, err_pos = _np_allclose(pos_num, pos_ca, args.pos_atol, args.pos_rtol)
    ok_rot, err_rot = _np_allclose(rot_num, rot_ca, args.rot_atol, args.rot_rtol)

    print(f"[Pelvis pos] ok={ok_pos}  max|err|={err_pos:.3e}  shape={pos_num.shape}")
    if not ok_pos:
        print(f"  -> atol={args.pos_atol}, rtol={args.pos_rtol}")
        print(f"  pos_num={pos_num}")
        print(f"  pos_ca ={pos_ca}")
    print(f"[Pelvis rot] ok={ok_rot}  max|err|={err_rot:.3e}  shape={rot_num.shape}")

    # FK + Jacobian checks
    print_header("VERIFY: FK + Jacobian for frames existing in your casadi_dir")
    fk_all_ok = True
    for fr in frames:
        fk_fn, jac_fn = load_fk_jacobian(CASADI_DIR, urdf_name, fr)

        fid = model.getFrameId(fr)
        oMf = data.oMf[fid]
        pos_num = np.asarray(oMf.translation).reshape(3,)
        rot_num = np.asarray(oMf.rotation).reshape(3, 3)

        # Numeric Jacobian (LOCAL_WORLD_ALIGNED) 6 x nv
        J_num = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_num = np.asarray(J_num)

        pos_ca, rot_ca = fk_fn(q_ca)
        pos_ca = np.array(pos_ca).reshape(3,)
        rot_ca = np.array(rot_ca).reshape(3, 3)
        J_ca = np.array(jac_fn(q_ca))

        ok_p, err_p = _np_allclose(pos_num, pos_ca, args.pos_atol, args.pos_rtol)
        ok_r, err_r = _np_allclose(rot_num, rot_ca, args.rot_atol, args.rot_rtol)
        ok_j, err_j = _np_allclose(J_num, J_ca, args.jac_atol, args.jac_rtol)

        print(f"[{fr} pos] ok={ok_p}  max|err|={err_p:.3e}  shape={pos_num.shape}")
        if not ok_p:
            print(f"  -> atol={args.pos_atol}, rtol={args.pos_rtol}")
        print(f"[{fr} rot] ok={ok_r}  max|err|={err_r:.3e}  shape={rot_num.shape}")
        print(f"[{fr} J] ok={ok_j}  max|err|={err_j:.3e}  shape={J_num.shape}")
        if not ok_j:
            print(f"  -> atol={args.jac_atol}, rtol={args.jac_rtol}")

        fk_all_ok = fk_all_ok and ok_p and ok_r and ok_j

    # Dynamics checks
    print_header("VERIFY: Dynamics (M, C, G)")
    M_num = pin.crba(model, data, q)
    M_num = np.asarray(M_num)

    # C@v numeric (vector)
    Cmat = pin.computeCoriolisMatrix(model, data, q, v)
    Cvec_num = np.asarray(Cmat @ v).reshape(-1)

    G_num = np.asarray(pin.computeGeneralizedGravity(model, data, q)).reshape(-1)

    M_ca = np.array(M_fn(q_ca))
    Cvec_ca = np.array(C_fn(q_ca, v_ca)).reshape(-1)
    G_ca = np.array(G_fn(q_ca)).reshape(-1)

    ok_M, err_M = _np_allclose(M_num, M_ca, args.dyn_atol, args.dyn_rtol)
    ok_C, err_C = _np_allclose(Cvec_num, Cvec_ca, args.dyn_atol, args.dyn_rtol)
    ok_G, err_G = _np_allclose(G_num, G_ca, args.dyn_atol, args.dyn_rtol)

    print(f"[M] ok={ok_M}  max|err|={err_M:.3e}  shape={M_num.shape}")
    print(f"[C@v] ok={ok_C}  max|err|={err_C:.3e}  shape={Cvec_num.shape}")
    if not ok_C:
        print(f"  -> atol={args.dyn_atol}, rtol={args.dyn_rtol}")
    print(f"[G] ok={ok_G}  max|err|={err_G:.3e}  shape={G_num.shape}")
    if not ok_G:
        print(f"  -> atol={args.dyn_atol}, rtol={args.dyn_rtol}")

    all_ok = ok_pos and ok_rot and fk_all_ok and ok_M and ok_C and ok_G

    print_header("RESULT")
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")

        # Extra debug if NaNs appear
        any_nan = (
            (not np.isfinite(pos_ca).all())
            or (not np.isfinite(Cvec_ca).all())
            or (not np.isfinite(G_ca).all())
        )
        if any_nan:
            print("\n[Debug] NaN detected in CasADi outputs. Dumping input sanity:")
            print("  q finite:", np.isfinite(q).all(), "v finite:", np.isfinite(v).all())
            print("  quat (qx,qy,qz,qw):", q[3:7], "norm:", float(np.linalg.norm(q[3:7])))
            print("  base_pos_fn(q):", np.array(base_pos_fn(q_ca)).reshape(-1))


if __name__ == "__main__":
    main()
