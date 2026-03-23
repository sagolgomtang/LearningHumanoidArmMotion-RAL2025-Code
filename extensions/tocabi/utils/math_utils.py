from __future__ import annotations

import torch
from typing import Tuple


# -------------------------
# Basic utils (Torch only)
# -------------------------

@torch.jit.script
def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    # (D,) -> (1, D)
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


@torch.jit.script
def quat_wxyz_to_xyzw(q_wxyz: torch.Tensor) -> torch.Tensor:
    # q: (N,4)  wxyz -> xyzw
    return torch.stack((q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]), dim=1)

@torch.jit.script
def quat_xyzw_to_wxyz(q_xyzw: torch.Tensor) -> torch.Tensor:
    # q: (N,4)  xyzw -> wxyz
    return torch.stack((q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]), dim=1)


@torch.jit.script
def skew_symmetric(p: torch.Tensor) -> torch.Tensor:
    """
    p: (N,3) or (3,)
    return: (N,3,3)
    """
    p = _ensure_2d(p)
    # TorchScript: assert 대신 runtime check는 간단히
    if p.size(1) != 3:
        raise RuntimeError("skew_symmetric: last dim must be 3")

    zero = torch.zeros_like(p[:, 0])
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]

    row0 = torch.stack((zero, -pz,  py), dim=1)
    row1 = torch.stack(( pz, zero, -px), dim=1)
    row2 = torch.stack((-py,  px, zero), dim=1)
    return torch.stack((row0, row1, row2), dim=1)


@torch.jit.script
def adjoint_matrix_twist(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    twist order: [v, w]
    Ad = [[R, skew(p)R],
          [0,    R    ]]
    R: (N,3,3) or (3,3)
    p: (N,3) or (3,)
    return: (N,6,6)
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    p = _ensure_2d(p)

    if R.size(1) != 3 or R.size(2) != 3:
        raise RuntimeError("adjoint_matrix_twist: R must be (N,3,3)")
    if p.size(1) != 3:
        raise RuntimeError("adjoint_matrix_twist: p must be (N,3)")

    skew_p = skew_symmetric(p)  # (N,3,3)
    upper = torch.cat((R, torch.matmul(skew_p, R)), dim=2)                 # (N,3,6)
    lower = torch.cat((torch.zeros_like(R), R), dim=2)                     # (N,3,6)
    return torch.cat((upper, lower), dim=1)                                # (N,6,6)


@torch.jit.script
def adjoint_matrix_wrench(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    wrench order: [F, tau]
    Ad* = [[R,    0    ],
           [skew(p)R, R]]
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    p = _ensure_2d(p)

    if R.size(1) != 3 or R.size(2) != 3:
        raise RuntimeError("adjoint_matrix_wrench: R must be (N,3,3)")
    if p.size(1) != 3:
        raise RuntimeError("adjoint_matrix_wrench: p must be (N,3)")

    skew_p = skew_symmetric(p)
    upper = torch.cat((R, torch.zeros_like(R)), dim=2)
    lower = torch.cat((torch.matmul(skew_p, R), R), dim=2)
    return torch.cat((upper, lower), dim=1)


# -------------------------
# Permutation helpers
# -------------------------

@torch.jit.script
def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    """
    perm: (K,) long, meaning: y = x[perm]
    return inv: (K,) so that x = y[inv]
    """
    if perm.dim() != 1:
        raise RuntimeError("invert_permutation: perm must be 1D")
    K = perm.numel()
    inv = torch.empty((K,), device=perm.device, dtype=torch.long)
    # TorchScript for-loop OK
    for i in range(K):
        inv[perm[i]] = i
    return inv


@torch.jit.script
def reorder_last_dim(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    x: (..., K)
    perm: (K,)
    return: (..., K) = x[..., perm]
    """
    if perm.dim() != 1:
        raise RuntimeError("reorder_last_dim: perm must be 1D")
    return x.index_select(-1, perm)


# -------------------------
# Generalized conversions (tocabi-friendly)
# -------------------------

@torch.jit.script
def convert_gen_coord_from_isaaclab_to_pin(gen_coord: torch.Tensor, joint_perm_isaac_to_pin: torch.Tensor) -> torch.Tensor:
    """
    gen_coord: (N, 7+nj) = [pos(3), quat_wxyz(4), joints_isaac(nj)]
    out:       (N, 7+nj) = [pos(3), quat_xyzw(4), joints_pin(nj)]
    """
    gen_coord = _ensure_2d(gen_coord)
    if gen_coord.size(1) < 7:
        raise RuntimeError("convert_gen_coord_from_isaaclab_to_pin: need 7+nj")

    pos = gen_coord[:, 0:3]
    quat = quat_wxyz_to_xyzw(gen_coord[:, 3:7])
    joints = gen_coord[:, 7:]

    # perm length must match joint dim
    if joints.size(1) != joint_perm_isaac_to_pin.numel():
        raise RuntimeError("convert_gen_coord_from_isaaclab_to_pin: joint_perm size mismatch")

    joints_pin = reorder_last_dim(joints, joint_perm_isaac_to_pin)
    return torch.cat((pos, quat, joints_pin), dim=1)


@torch.jit.script
def convert_gen_coord_from_pin_to_isaaclab(gen_coord: torch.Tensor, joint_perm_pin_to_isaac: torch.Tensor) -> torch.Tensor:
    """
    gen_coord: (N, 7+nj) = [pos(3), quat_xyzw(4), joints_pin(nj)]
    out:       (N, 7+nj) = [pos(3), quat_wxyz(4), joints_isaac(nj)]
    """
    gen_coord = _ensure_2d(gen_coord)
    if gen_coord.size(1) < 7:
        raise RuntimeError("convert_gen_coord_from_pin_to_isaaclab: need 7+nj")

    pos = gen_coord[:, 0:3]
    quat = quat_xyzw_to_wxyz(gen_coord[:, 3:7])
    joints = gen_coord[:, 7:]

    if joints.size(1) != joint_perm_pin_to_isaac.numel():
        raise RuntimeError("convert_gen_coord_from_pin_to_isaaclab: joint_perm size mismatch")

    joints_isaac = reorder_last_dim(joints, joint_perm_pin_to_isaac)
    return torch.cat((pos, quat, joints_isaac), dim=1)


@torch.jit.script
def convert_gen_vel_from_isaaclab_to_pin(gen_vel: torch.Tensor, joint_perm_isaac_to_pin: torch.Tensor) -> torch.Tensor:
    """
    gen_vel: (N, 6+nj) = [root_lin(3), root_ang(3), joints_isaac(nj)]
    out:     (N, 6+nj) = [root_lin(3), root_ang(3), joints_pin(nj)]
    """
    gen_vel = _ensure_2d(gen_vel)
    if gen_vel.size(1) < 6:
        raise RuntimeError("convert_gen_vel_from_isaaclab_to_pin: need 6+nj")

    base = gen_vel[:, 0:6]
    joints = gen_vel[:, 6:]
    if joints.size(1) != joint_perm_isaac_to_pin.numel():
        raise RuntimeError("convert_gen_vel_from_isaaclab_to_pin: joint_perm size mismatch")

    joints_pin = reorder_last_dim(joints, joint_perm_isaac_to_pin)
    return torch.cat((base, joints_pin), dim=1)


@torch.jit.script
def convert_gen_force_from_pin_to_isaaclab(gen_force: torch.Tensor, joint_perm_pin_to_isaac: torch.Tensor) -> torch.Tensor:
    """
    gen_force: (N, 6+nj) = [F(3), tau_base(3), tau_joints_pin(nj)]  (Pin)
    out:       (N, 6+nj) = [F(3), tau_base(3), tau_joints_isaac(nj)]
    """
    gen_force = _ensure_2d(gen_force)
    if gen_force.size(1) < 6:
        raise RuntimeError("convert_gen_force_from_pin_to_isaaclab: need 6+nj")

    base = gen_force[:, 0:6]
    joints = gen_force[:, 6:]
    if joints.size(1) != joint_perm_pin_to_isaac.numel():
        raise RuntimeError("convert_gen_force_from_pin_to_isaaclab: joint_perm size mismatch")

    joints_isaac = reorder_last_dim(joints, joint_perm_pin_to_isaac)
    return torch.cat((base, joints_isaac), dim=1)


# -------------------------
# Jacobian / Mass matrix conversions
# -------------------------

@torch.jit.script
def convert_jacobian_from_pin_to_isaaclab(jacobian: torch.Tensor, genvel_perm_pin_to_isaac: torch.Tensor) -> torch.Tensor:
    """
    jacobian: (N, 6, 6+nj) in Pin generalized velocity order
    out:      (N, 6, 6+nj) in Isaac generalized velocity order
    -> reorder last dim
    """
    if jacobian.dim() != 3 or jacobian.size(1) != 6:
        raise RuntimeError("convert_jacobian_from_pin_to_isaaclab: jacobian must be (N,6,K)")

    K = jacobian.size(2)
    if K != genvel_perm_pin_to_isaac.numel():
        raise RuntimeError("convert_jacobian_from_pin_to_isaaclab: perm size mismatch")

    return jacobian.index_select(2, genvel_perm_pin_to_isaac)


@torch.jit.script
def convert_mass_matrix_from_pin_to_isaaclab(mass_matrix: torch.Tensor, genvel_perm_pin_to_isaac: torch.Tensor) -> torch.Tensor:
    """
    mass_matrix: (N, K, K) in Pin generalized velocity order
    out:         (N, K, K) in Isaac generalized velocity order
    -> reorder both dims
    """
    if mass_matrix.dim() != 3:
        raise RuntimeError("convert_mass_matrix_from_pin_to_isaaclab: mass_matrix must be (N,K,K)")

    K = mass_matrix.size(1)
    if mass_matrix.size(2) != K:
        raise RuntimeError("convert_mass_matrix_from_pin_to_isaaclab: mass_matrix must be square")
    if K != genvel_perm_pin_to_isaac.numel():
        raise RuntimeError("convert_mass_matrix_from_pin_to_isaaclab: perm size mismatch")

    mm = mass_matrix.index_select(1, genvel_perm_pin_to_isaac)
    mm = mm.index_select(2, genvel_perm_pin_to_isaac)
    return mm


# -------------------------
# Joint-only order conversion
# -------------------------

@torch.jit.script
def convert_joint_order_from_isaaclab_to_pin(joint_order: torch.Tensor, joint_perm_isaac_to_pin: torch.Tensor) -> torch.Tensor:
    """
    joint_order: (N, nj) or (nj,)
    out:         (N, nj)
    """
    joint_order = _ensure_2d(joint_order)
    if joint_order.size(1) != joint_perm_isaac_to_pin.numel():
        raise RuntimeError("convert_joint_order_from_isaaclab_to_pin: perm size mismatch")

    return reorder_last_dim(joint_order, joint_perm_isaac_to_pin)
