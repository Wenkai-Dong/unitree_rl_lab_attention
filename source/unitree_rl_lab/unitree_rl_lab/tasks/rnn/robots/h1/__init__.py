import gymnasium as gym


# actor 与 critic 网络共享
gym.register(
    id="Unitree-H1-Rnn-S1-V0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rnn_env_cfg_s1:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rnn_env_cfg_s1:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.rnn.agents.rsl_rl_ppo_cfg:BasePPORunnerS1Cfg",
    },
)

gym.register(
    id="Unitree-H1-Rnn-S2-V0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rnn_env_cfg_s2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rnn_env_cfg_s2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.rnn.agents.rsl_rl_ppo_cfg:BasePPORunnerS2Cfg",
    },
)