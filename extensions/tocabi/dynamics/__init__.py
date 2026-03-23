# extensions/tocabi/dynamics/__init__.py
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
PINOCCHIO_CASADI_FUNCTIONS_DIR = str(_THIS_DIR / "casadi_fns")
