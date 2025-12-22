from dataclasses import dataclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass
import numpy as np
import trimesh
from shapely.geometry import box


@configclass
@dataclass
class MeshConcentricBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    回字形横梁地形配置
    """
    platform_width: float = 2.0  # 中心出生点平台的宽度
    beam_thickness: float = 3.0  # 地形厚度（Z轴）

    # --- 课程参数 ---
    # 难度 0 -> 1 变化时：
    step_height_range: tuple[float, float] = (0.0, 0.3)  # 高度噪声幅度
    beam_width_range: tuple[float, float] = (1.0, 0.2)  # 横梁宽度：从宽变窄
    gap_width_range: tuple[float, float] = (0.1, 0.4)  # 缝隙宽度：从窄变宽


def mesh_concentric_beams_terrain(
        difficulty: float, cfg: MeshConcentricBeamsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    生成同心回字形横梁地形。
    修复点：
    1. 自动根据当前宽度和间距计算能放下多少圈。
    2. 保证中心平台始终存在。
    3. 自动填充外围安全地面。
    """
    # 1. 解析课程难度参数
    gap = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    width = cfg.beam_width_range[0] + difficulty * (cfg.beam_width_range[1] - cfg.beam_width_range[0])
    noise_mag = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])

    # 2. 基础常量计算
    # 地形半长（正方形的一半）
    terrain_half_size = min(cfg.size[0], cfg.size[1]) / 2
    # 安全边距：最外圈保留 1.0 米平地，防止机器人从最后一关掉出地图
    safety_margin = 1.0

    # 3. 核心修复：显式计算能放下多少个回字形 (num_rings)
    # 可用半径 = 总半径 - 安全边距 - 中心平台半径
    available_radius = terrain_half_size - safety_margin - (cfg.platform_width / 2)
    # 单个回字形占用的径向距离 = 缝隙 + 梁宽
    unit_thickness = gap + width

    # 向下取整，算出最多能放几圈
    if unit_thickness > 0.001 and available_radius > 0:
        num_rings = int(available_radius / unit_thickness)
    else:
        num_rings = 0

    # 初始化网格列表
    meshes_list = list()

    # ==========================================
    # 4. 生成中心平台 (Spawn Point) - 保证始终存在
    # ==========================================
    # 为了防止 Z-fighting 或方便行走，中心平台高度设为平地高度
    # 假设平地高度为 -half_thickness (表面在 0)
    # 或者为了与横梁对齐，我们把所有物体基准设为 Z=0 表面

    center_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * cfg.beam_thickness)
    center_mesh = trimesh.creation.box(
        (cfg.platform_width, cfg.platform_width, cfg.beam_thickness),
        trimesh.transformations.translation_matrix(center_pos)
    )
    meshes_list.append(center_mesh)

    # ==========================================
    # 5. 循环生成回字形横梁
    # ==========================================
    # 初始内边长 = 中心平台 + 两侧第一个缝隙
    current_inner_side = cfg.platform_width + 2 * gap

    for i in range(num_rings):
        # 当前环的外边长 = 内边长 + 两侧梁宽
        current_outer_side = current_inner_side + 2 * width

        # 使用 Shapely 制作回字形截面 (大正方形 减去 小正方形)
        outer_rect = box(-current_outer_side / 2, -current_outer_side / 2, current_outer_side / 2,
                         current_outer_side / 2)
        inner_rect = box(-current_inner_side / 2, -current_inner_side / 2, current_inner_side / 2,
                         current_inner_side / 2)
        beam_2d = outer_rect.difference(inner_rect)

        # 计算随机高度 (相对于 Z=0 平面的偏移)
        z_noise = np.random.uniform(-noise_mag, noise_mag)

        # 挤出并平移
        # 注意：box生成是以中心为基准的，extrude是从0向上挤出的
        # 我们希望 Mesh 的顶面位于 z_noise
        beam_mesh = trimesh.creation.extrude_polygon(beam_2d, height=cfg.beam_thickness)

        # 平移到地形中心 + 高度调整
        # extrude 出来的物体底面在 Z=0，顶面在 Z=thickness
        # 我们要让顶面变成 z_noise，所以整体向下移 thickness，再加 z_noise
        beam_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], z_noise - cfg.beam_thickness)
        beam_mesh.apply_transform(trimesh.transformations.translation_matrix(beam_pos))

        meshes_list.append(beam_mesh)

        # 更新下一圈的内边长 = 当前外边长 + 两侧缝隙
        current_inner_side = current_outer_side + 2 * gap

    # ==========================================
    # 6. 生成外围安全地面 (Safety Border)
    # ==========================================
    # 我们用一个巨大的地形块矩形，减去中间已经被占用的区域（最后一圈横梁 + 最后的缝隙）
    # 中间被挖空的区域边长：
    hole_side = current_inner_side  # 注意：循环结束后 current_inner_side 已经加上了最后的 gap

    total_rect = box(-cfg.size[0] / 2, -cfg.size[1] / 2, cfg.size[0] / 2, cfg.size[1] / 2)
    # 限制挖空区域不要超过地形总大小 (虽然逻辑上不应该发生)
    hole_side = min(hole_side, min(cfg.size[0], cfg.size[1]))
    hole_rect = box(-hole_side / 2, -hole_side / 2, hole_side / 2, hole_side / 2)

    border_2d = total_rect.difference(hole_rect)

    if not border_2d.is_empty:
        # 地面高度设为 0 (标准平地)
        border_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.beam_thickness)  # 顶面在 0
        border_mesh = trimesh.creation.extrude_polygon(border_2d, height=cfg.beam_thickness)
        border_mesh.apply_transform(trimesh.transformations.translation_matrix(border_pos))
        meshes_list.append(border_mesh)

    # ==========================================
    # 7. 设置地形原点
    # ==========================================
    # 机器人出生在中心平台上方 0.1m 处
    origin = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0 + 0.1])

    return meshes_list, origin