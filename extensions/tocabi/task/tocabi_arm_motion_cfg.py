# Copyright (c) 2022-2024, The ISAACLAB Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlModularOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
    
@configclass
class TocabiArmMotionPPORunnerCfg(RslRlModularOnPolicyRunnerCfg):
    # master switch for arm policy update (set False to disable arm)
    enable_arm = False
    # actor observation history switch (env cfg keeps history length)
    enable_actor_history = True
    # adaptive leg lipschitz regularization (runner-side weight tuning)
    leg_lipschitz_adaptive = True
    leg_lipschitz_fixed_weight = 0.005
    leg_lipschitz_target_ratio = 0.05
    leg_lipschitz_adapt_factor = 1.05
    leg_lipschitz_deadband = 0.7
    leg_lipschitz_min_weight = 0.005
    leg_lipschitz_max_weight = 0.01
    # adaptive arm lipschitz regularization (runner-side weight tuning)
    arm_lipschitz_adaptive = True
    arm_lipschitz_fixed_weight = 0.001
    arm_lipschitz_target_ratio = 0.05
    arm_lipschitz_adapt_factor = 1.05
    arm_lipschitz_deadband = 0.7
    arm_lipschitz_min_weight = 0.001
    arm_lipschitz_max_weight = 0.01
    seed = 42 # -1
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "Tocabi_Arm_Motion"
    logger = "wandb"
    enable_logging = True
    wandb_project = "Tocabi_Arm_Motion"
    store_code_state = True
    leg_update_enabled = True
    leg_frozen_onnx_path = None
    leg_frozen_pt_path = None
    arm_update_enabled = enable_arm
    arm_std_bounds = None
    leg_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        normalize_obs=True,
    )
    arm_policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[512, 256, 256],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        normalize_obs=True,
        custom_initialization=True
    )
    _leg_algo_kwargs = dict(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12143,  # 0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.00041,  # 1.e-5,
        schedule="adaptive",
        gamma=0.9751,  # 0.99,
        lam=0.95,
        desired_kl=0.01,
        mirror_loss_weight=0.1,
        lipschitz_loss_weight=0.1,
        max_grad_norm=1.0,
    )
    try:
        leg_algorithm = RslRlPpoAlgorithmCfg(**_leg_algo_kwargs)
    except TypeError:
        _leg_algo_kwargs.pop("lipschitz_loss_weight", None)
        leg_algorithm = RslRlPpoAlgorithmCfg(**_leg_algo_kwargs)
        leg_algorithm.lipschitz_loss_weight = 0.1

    _arm_algo_kwargs = dict(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate= 1.e-5,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        mirror_loss_weight=1.0,
        lipschitz_loss_weight=0.1,
    )
    try:
        arm_algorithm = RslRlPpoAlgorithmCfg(**_arm_algo_kwargs)
    except TypeError:
        _arm_algo_kwargs.pop("lipschitz_loss_weight", None)
        arm_algorithm = RslRlPpoAlgorithmCfg(**_arm_algo_kwargs)
        arm_algorithm.lipschitz_loss_weight = 0.1
