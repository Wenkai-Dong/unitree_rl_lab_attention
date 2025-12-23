# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from unitree_rl_lab.tasks.attention_encoding.terrains.attention_terrains_cfg import (
    MeshConcentricBeamsTerrainCfg,
    mesh_concentric_beams_terrain
)

ATTENTION_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10,
    num_rows=10,
    num_cols=9,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        # "Rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     noise_range=(0.01, 0.08),
        #     noise_step=(0.005),
        #     downsampled_scale=0.2,
        #     border_width=1,
        # ),
        "Rough":terrain_gen.HfRandomUniformTerrainCfg(
            noise_range=(0.02, 0.15),
            noise_step=0.02,
            border_width=1.
        ),
        "Stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.35,
            inverted=True,
            platform_width=2,
            border_width=1,
        ),
        "StairsInverted": terrain_gen.HfPyramidStairsTerrainCfg(
            step_height_range=(0.01, 0.3),
            step_width=0.35,
            inverted=False,
            platform_width=2,
            border_width=1,
        ),
        "Gaps": terrain_gen.MeshGapTerrainCfg(
            gap_width_range=(0.1,1.1),
            platform_width=2.,
        ),
        "GridStones": terrain_gen.HfSteppingStonesTerrainCfg(
            stone_height_max=0.2,
            stone_width_range=(0.3,0.8),
            stone_distance_range=(0.05,0.4),
            holes_depth=-10,
            platform_width=1.5,
            border_width=1,
        ),
        # "Pallets": terrain_gen.MeshRailsTerrainCfg(
        #     rail_thickness_range=(0.5,1.),
        #     rail_height_range=(0.1,1.),
        #     platform_width=2.,
        # ),
        "Pallets": MeshConcentricBeamsTerrainCfg(
            proportion=1.0,
            # 1. 必填：告诉它用哪个函数 (对应第二步注册的名字)
            function=mesh_concentric_beams_terrain,
            # 2. 基础参数
            platform_width=1.5,
            beam_thickness=3.0,
            # 3. 课程难度参数 (根据你的需求调整)
            step_height_range=(0.0, 0.2),  # 难度越高，下沉越深
            beam_width_range=(0.6, 0.1),  # 难度越高，路越窄
            gap_width_range=(0.1, 0.8),  # 难度越高，缝隙越大
        ),
        "Pits": terrain_gen.MeshPitTerrainCfg(
            pit_depth_range=(0.1,0.6),
            double_pit=True,
            platform_width=2.,
        ),
        "PitsInverted": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(0.1,0.7),
            double_box=True,
            platform_width=2.,
        ),

        "Beams": terrain_gen.MeshStarTerrainCfg(
            num_bars=4,
            bar_width_range=(0.05,1.),
            bar_height_range=(5.,5.),
            platform_width=2.,
        ),
    },
)
"""All terrains configuration."""
