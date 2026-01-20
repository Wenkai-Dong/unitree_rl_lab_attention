from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class RslRlPpoActorCriticCrossMhaRnnCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks with recurrent layers."""

    class_name: str = "ActorCriticRecurrent"
    """The policy class name. Default is ActorCriticCrossMhaRnn."""

    actor_cnn_cfg: dict[str, dict] | dict | None = None

    actor_mha_cfg: dict[str, dict] | dict | None = None

    actor_rnn_cfg: dict[str, dict] | dict | None = None

    critic_rnn_cfg: dict[str, dict] | dict | None = None
