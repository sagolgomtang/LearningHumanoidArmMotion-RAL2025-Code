#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import torch


def _tensor_l2_norm(state_dict: dict) -> float:
    total = 0.0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += float(torch.sum(v.float() ** 2).item())
    return total ** 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect leg/arm checkpoint weight norms.")
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt")
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        print(f"[ERR] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu")

    leg_sd = ckpt.get("leg_model_state_dict", None)
    arm_sd = ckpt.get("arm_model_state_dict", None)

    if leg_sd is None:
        print("[WARN] leg_model_state_dict not found")
    else:
        leg_norm = _tensor_l2_norm(leg_sd)
        print(f"[LEG] l2_norm={leg_norm:.6f} params={len(leg_sd)}")

    if arm_sd is None:
        print("[WARN] arm_model_state_dict not found")
    else:
        arm_norm = _tensor_l2_norm(arm_sd)
        print(f"[ARM] l2_norm={arm_norm:.6f} params={len(arm_sd)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
