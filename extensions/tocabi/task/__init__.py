# extensions/tocabi/task/__init__.py
import gymnasium as gym

def _register_envs():
    # 여기서 "패키지(task) 자체"를 다시 import 하지 않게
    # 필요한 클래스만 모듈에서 직접 뽑아옵니다.
    from .tocabi_arm_motion import TocabiArmMotionEnv
    from .tocabi_arm_motion_task_cfg import TocabiArmMotionEnvCfg, TocabiArmMotionEnvCfg_PLAY
    from .tocabi_arm_motion_cfg import TocabiArmMotionPPORunnerCfg

    gym.register(
        id="tocabi_arm_motion",
        entry_point=TocabiArmMotionEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": TocabiArmMotionEnvCfg,
            "rsl_rl_cfg_entry_point": TocabiArmMotionPPORunnerCfg,
        },
    )

    gym.register(
        id="tocabi_arm_motion_play",
        entry_point=TocabiArmMotionEnv,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": TocabiArmMotionEnvCfg_PLAY,
            "rsl_rl_cfg_entry_point": TocabiArmMotionPPORunnerCfg,
        },
    )

_register_envs()
