# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg
from ..rl import RslRlPpoActorCriticCNNCfg, RslRlPpoActorCriticCrossMHACfg, RslRlPpoActorCriticCrossMhaRnnCfg
from ..mdp.symmetry.h1 import compute_symmetric_states

@configclass
class BasePPORunnerS1Cfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    obs_groups = {
        # "policy": ["policy", "policy_map"],
        # "critic": ["critic", "critic_map"],
        "policy": ["policy"],
        "critic": ["critic"],
    }
    save_interval = 100
    experiment_name = ""  # same as task name
    # resume = False
    policy = RslRlPpoActorCriticCrossMhaRnnCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_cnn_cfg={
            "output_channels": [16, 61],
            "kernel_size": 5,
            "stride": 1,
            "dilation": 1,
            "padding": "zeros",
            "norm": "layer",
            "activation": "elu",
            "max_pool": False,
            "global_pool": "none",
            "flatten": False,
        },
        actor_mha_cfg={
            "num_heads": 16,
            "dropout": 0.0,
            "bias": True,
            "add_bias_kv": False,
            "add_zero_attn": False,
            "kdim": None,
            "vdim": None,
            "batch_first": True,
        },

    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005, # stage2: 0.002
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # symmetry_cfg=RslRlSymmetryCfg(
        #     use_data_augmentation = True,
        #     use_mirror_loss=False,
        #     data_augmentation_func=compute_symmetric_states,
        #     mirror_loss_coeff = 0.0
        # )
    )

@configclass
class BasePPORunnerS2Cfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    obs_groups = {
        "policy": ["policy", "policy_map"],
        "critic": ["critic", "critic_map"],
    }
    save_interval = 100
    experiment_name = ""  # same as task name
    # resume = False
    policy = RslRlPpoActorCriticCrossMhaRnnCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_cnn_cfg={
            "output_channels": [16, 61],
            "kernel_size": 5,
            "stride": 1,
            "dilation": 1,
            "padding": "zeros",
            "norm": "layer",
            "activation": "elu",
            "max_pool": False,
            "global_pool": "none",
            "flatten": False,
        },
        actor_mha_cfg={
            "num_heads": 16,
            "dropout": 0.0,
            "bias": True,
            "add_bias_kv": False,
            "add_zero_attn": False,
            "kdim": None,
            "vdim": None,
            "batch_first": True,
        },
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002, # stage2: 0.002
        num_learning_epochs=5,
        num_mini_batches=6,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation = True,
            use_mirror_loss=False,
            data_augmentation_func=compute_symmetric_states,
            mirror_loss_coeff = 0.0
        )
    )