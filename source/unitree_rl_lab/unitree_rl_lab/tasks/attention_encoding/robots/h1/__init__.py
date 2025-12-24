import gymnasium as gym

gym.register(
    id="Unitree-H1-Attention-Encoding-S1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg_s1:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.attention_env_cfg_s1:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.attention_encoding.agents.rsl_rl_ppo_cfg:BasePPORunnerS1Cfg",
    },
)

gym.register(
    id="Unitree-H1-Attention-Encoding-S2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_env_cfg_s2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.attention_env_cfg_s2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.attention_encoding.agents.rsl_rl_ppo_cfg:BasePPORunnerS2Cfg",
    },
)