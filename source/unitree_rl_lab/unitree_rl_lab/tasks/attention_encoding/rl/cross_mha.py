from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

@configclass
class RslRlPpoActorCriticCrossMHACfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticCrossMHA"
    """The policy class name. Default is ActorCriticCrossMHA."""

    actor_cnn_cfg: dict[str, dict] | dict | None = None

    critic_cnn_cfg: dict[str, dict] | dict | None = None