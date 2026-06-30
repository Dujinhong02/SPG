# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import math
# from pathlib import Path
# import argparse
# import open3d as o3d

# def is_valid_mesh(m: o3d.geometry.TriangleMesh) -> bool:
#     return (m is not None) and (len(m.vertices) > 0) and (len(m.triangles) > 0)

# def is_valid_pcd(p: o3d.geometry.PointCloud) -> bool:
#     return (p is not None) and (len(p.points) > 0)

# def load_geometry(ply_path: Path):
#     """
#     先按TriangleMesh读取；若无三角面，改按PointCloud读取。
#     对网格尝试三角化；对点云估计法向。
#     返回 (geom, 'mesh'|'pcd') 或 (None, None) 表示无效。
#     """
#     # 尝试按网格读取
#     try:
#         m = o3d.io.read_triangle_mesh(str(ply_path), print_progress=False)
#     except Exception:
#         m = None

#     if m is not None and len(m.vertices) > 0:
#         # 有顶点但可能没有三角面
#         if len(m.triangles) == 0:
#             # 有些PLY包含多边形但未三角化；尝试三角化
#             try:
#                 m = m.triangulate()
#             except Exception:
#                 pass
#         if is_valid_mesh(m):
#             m.compute_vertex_normals()
#             return m, "mesh"
#         # 到这里说明按网格仍无效，回退为点云
#     # 按点云读取
#     try:
#         p = o3d.io.read_point_cloud(str(ply_path), print_progress=False)
#         if is_valid_pcd(p):
#             # 估计法向（便于着色/光照；若只是点也可省略）
#             p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
#             return p, "pcd"
#     except Exception:
#         pass
#     return None, None

# def aabb_xy_size(geom):
#     if isinstance(geom, o3d.geometry.TriangleMesh):
#         aabb = geom.get_axis_aligned_bounding_box()
#     else:
#         aabb = geom.get_axis_aligned_bounding_box()
#     ext = aabb.get_extent()
#     return float(ext[0]), float(ext[1])

# def center_and_drop(geom):
#     aabb = geom.get_axis_aligned_bounding_box()
#     geom.translate(-aabb.get_center())
#     aabb = geom.get_axis_aligned_bounding_box()
#     geom.translate((0, 0, -aabb.min_bound[2]))

# def make_ground_grid(size_x, size_y, step):
#     lines, pts = [], []
#     x = -size_x/2
#     while x <= size_x/2 + 1e-9:
#         pts += [[x, -size_y/2, 0.0], [x, size_y/2, 0.0]]
#         lines.append([len(pts)-2, len(pts)-1])
#         x += step
#     y = -size_y/2
#     while y <= size_y/2 + 1e-9:
#         pts += [[-size_x/2, y, 0.0], [size_x/2, y, 0.0]]
#         lines.append([len(pts)-2, len(pts)-1])
#         y += step
#     grid = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(pts),
#         lines=o3d.utility.Vector2iVector(lines),
#     )
#     grid.paint_uniform_color([0.6, 0.6, 0.6])
#     return grid

# def parse_ranges(ranges_str: str, total: int):
#     """
#     ranges_str: 例如 "0-100,101-300" 或 "5,7-9"
#     返回按出现顺序去重后的索引列表（越界自动丢弃）。
#     """
#     if not ranges_str:
#         return []

#     parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
#     idx_list = []
#     for p in parts:
#         if "-" in p:
#             a, b = p.split("-", 1)
#             a, b = int(a), int(b)
#             if b < a:
#                 a, b = b, a
#             idx_list.extend(range(a, b + 1))  # 右端包含
#         else:
#             idx_list.append(int(p))

#     # 去重但保留顺序 + clip
#     seen = set()
#     out = []
#     for i in idx_list:
#         if 0 <= i < total and i not in seen:
#             out.append(i)
#             seen.add(i)
#     return out

# def sort_by_parent_folder(ply_paths):
#     # 使用父文件夹的名称进行排序，假设父文件夹名是数字字符串
#     return sorted(ply_paths, key=lambda x: int(x.parent.name))



# def main():
#     ap = argparse.ArgumentParser()
#     # ap.add_argument("--root", type=str, default='shot_fpfh/descriptors/dataset/3dnet_norm', help=".../dataset/mesh/model1")
#     ap.add_argument("--root", type=str, default='shot_fpfh/descriptors/dataset/model1_norm', help=".../dataset/mesh/model1")
#     ap.add_argument("--cols", type=int, default=10)
#     ap.add_argument("--margin_ratio", type=float, default=0.2)
#     ap.add_argument("--skip_pcd", action="store_true", help="仅显示网格，忽略点云")
#     ap.add_argument("--skip_mesh", action="store_true", help="仅显示点云，忽略网格")
#     ap.add_argument("--n", type=int, default=-1,help="按顺序仅取前 n 个；<=0 表示不启用")
#     # ap.add_argument("--ranges", type=str, default="920-1000",help="按索引范围选择，格式如 '0-100,101-300'，两端包含；优先级高于 --n")
#     ap.add_argument("--ranges", type=str, default="",help="按索引范围选择，格式如 '0-100,101-300'，两端包含；优先级高于 --n")

#     args = ap.parse_args()

#     root = Path(args.root).expanduser().resolve()
#     # 收集所有nontextured.ply
#     ply_paths = sorted([d/"nontextured_norm.ply" for d in root.iterdir()
#                         if d.is_dir() and (d/"nontextured_norm.ply").exists()])
#     ply_paths = sort_by_parent_folder(ply_paths)
#     if not ply_paths:
#         print(f"[ERROR] {root} 下未找到任何 nontextured.ply")
#         return

#     # -------- 按范围/数量选择子集（读取前裁剪） --------
#     if args.ranges.strip():
#         indices = parse_ranges(args.ranges, len(ply_paths))
#         ply_paths = [ply_paths[i] for i in indices]
#         print(f"[INFO] 使用 ranges='{args.ranges}' 选择 {len(ply_paths)} 个")
#     elif args.n is not None and args.n > 0:
#         ply_paths = ply_paths[:args.n]
#         print(f"[INFO] 使用 n={args.n} 选择前 {len(ply_paths)} 个")
#     # ----------------------------------------------


#     geoms, types = [], []
#     for ply in ply_paths:
#         g, t = load_geometry(ply)
#         if g is None:
#             print(f"[WARN] 无效或不支持的PLY：{ply}")
#             continue
#         if (t == "pcd" and args.skip_pcd) or (t == "mesh" and args.skip_mesh):
#             continue
#         center_and_drop(g)
#         geoms.append(g)
#         types.append(t)

#     if not geoms:
#         print("[ERROR] 没有可视化的几何（可能都为空网格或被过滤为点云/网格）。")
#         return

#     # 统计单元格尺寸
#     max_dx = max(aabb_xy_size(g)[0] for g in geoms)
#     max_dy = max(aabb_xy_size(g)[1] for g in geoms)
#     cell_x = max_dx * (1.0 + args.margin_ratio)
#     cell_y = max_dy * (1.0 + args.margin_ratio)

#     n = len(geoms)
#     cols = max(1, args.cols)
#     rows = (n + cols - 1) // cols
#     origin_x = - (cols - 1) * cell_x / 2.0
#     origin_y = - (rows - 1) * cell_y / 2.0

#     arranged = []
#     for i, g in enumerate(geoms):
#         r, c = divmod(i, cols)
#         tx = origin_x + c * cell_x
#         ty = origin_y + r * cell_y
#         arranged.append(g.translate((tx, ty, 0.0), relative=True))

#     # 地网与坐标轴
#     ground = make_ground_grid(size_x=max(cols*cell_x, cell_x*4),
#                               size_y=max(rows*cell_y, cell_y*4),
#                               step=max(min(cell_x, cell_y)/10.0, 0.05))
#     arranged.append(ground)
#     arranged.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

#     # 关键：Open3D 支持 mesh 与 point cloud 混合；不会再尝试把空网格绑定到SimpleShader
#     print(f"[INFO] 有效几何共 {n} 个（网格={types.count('mesh')}，点云={types.count('pcd')}）")
#     o3d.visualization.draw_geometries(arranged)

# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import math
# import copy
# from pathlib import Path
# import argparse
# import open3d as o3d
# import numpy as np
# import random

# # --- 基础加载与处理函数 ---

# def is_valid_mesh(m: o3d.geometry.TriangleMesh) -> bool:
#     return (m is not None) and (len(m.vertices) > 0) and (len(m.triangles) > 0)

# def is_valid_pcd(p: o3d.geometry.PointCloud) -> bool:
#     return (p is not None) and (len(p.points) > 0)

# def load_geometry(ply_path: Path):
#     # 尝试按 Mesh 读取
#     try:
#         m = o3d.io.read_triangle_mesh(str(ply_path), print_progress=False)
#     except Exception:
#         m = None

#     if m is not None and len(m.vertices) > 0:
#         if len(m.triangles) == 0:
#             try: m = m.triangulate()
#             except: pass
#         if is_valid_mesh(m):
#             m.compute_vertex_normals()
#             return m, "mesh"
    
#     # 回退按 PointCloud 读取
#     try:
#         p = o3d.io.read_point_cloud(str(ply_path), print_progress=False)
#         if is_valid_pcd(p):
#             p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
#             return p, "pcd"
#     except Exception:
#         pass
#     return None, None

# def aabb_xy_size(geom):
#     aabb = geom.get_axis_aligned_bounding_box()
#     ext = aabb.get_extent()
#     return float(ext[0]), float(ext[1])

# def center_and_drop(geom):
#     aabb = geom.get_axis_aligned_bounding_box()
#     geom.translate(-aabb.get_center())
#     aabb = geom.get_axis_aligned_bounding_box()
#     geom.translate((0, 0, -aabb.min_bound[2]))

# def sample_indices_from_segments(total, seed=0):
#     """
#     按指定分段规则采样索引：
#     0-403      -> 随机100个
#     404-577    -> 随机80个
#     578-785    -> 随机100个
#     785-1063   -> 随机80个
#     1073-1154  -> 全部
#     注意：
#       - 785 会重复，最后会去重
#       - 1064-1072 不包含
#     """
#     random.seed(seed)

#     segments = [
#         (0, 403, 100, "random"),
#         (404, 577, 80, "random"),
#         (578, 785, 100, "random"),
#         (785, 1063, 80, "random"),
#         (1073, 1154, None, "all"),
#     ]

#     selected = []

#     for start, end, k, mode in segments:
#         # 裁剪到合法范围
#         start = max(0, start)
#         end = min(total - 1, end)
#         if start > end:
#             continue

#         candidates = list(range(start, end + 1))

#         if mode == "all":
#             picked = candidates
#         else:
#             kk = min(k, len(candidates))
#             picked = random.sample(candidates, kk)

#         selected.extend(picked)

#     # 去重 + 排序，避免 785 重复
#     selected = sorted(set(selected))
#     return selected

# def parse_ranges(ranges_str: str, total: int):
#     if not ranges_str: return []
#     parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
#     idx_list = []
#     for p in parts:
#         if "-" in p:
#             a, b = p.split("-", 1)
#             try:
#                 a, b = int(a), int(b)
#                 if b < a: a, b = b, a
#                 idx_list.extend(range(a, b + 1))
#             except ValueError: pass
#         else:
#             try: idx_list.append(int(p))
#             except ValueError: pass
#     seen = set()
#     out = []
#     for i in idx_list:
#         if 0 <= i < total and i not in seen:
#             out.append(i)
#             seen.add(i)
#     return out

# def sort_by_parent_folder(ply_paths):
#     try:
#         return sorted(ply_paths, key=lambda x: int(x.parent.name))
#     except ValueError:
#         return sorted(ply_paths)

# # --- 核心修改：合并与交互 ---

# def merge_to_single_pointcloud(geoms):
#     """
#     将所有几何体（Mesh或PointCloud）合并为一个巨大的PointCloud。
#     VisualizerWithVertexSelection 只能接受一个 geometry。
#     """
#     combined_pcd = o3d.geometry.PointCloud()
    
#     for g in geoms:
#         # 如果是Mesh，转换为PointCloud（取顶点）
#         if isinstance(g, o3d.geometry.TriangleMesh):
#             temp_pcd = o3d.geometry.PointCloud()
#             temp_pcd.points = g.vertices
#             # 保留颜色（如果有）
#             if g.has_vertex_colors():
#                 temp_pcd.colors = g.vertex_colors
#             # 保留法向（如果有）
#             if g.has_vertex_normals():
#                 temp_pcd.normals = g.vertex_normals
#             combined_pcd += temp_pcd
#         elif isinstance(g, o3d.geometry.PointCloud):
#             combined_pcd += g
            
#     return combined_pcd

# def run_interactive_picker(merged_pcd, ply_paths, layout_info):
#     origin_x, origin_y, cell_x, cell_y, cols, total_items = layout_info

#     print("\n" + "="*60)
#     print(" [交互模式说明] ")
#     print(" 1. 因工具限制，所有物体已合并为单一视图，且去除了网格线。")
#     print(" 2. 按住 [Shift] + [鼠标左键] 点击物体。")
#     print(" 3. 终端将输出该物体对应的原始路径。")
#     print("="*60 + "\n")

#     vis = o3d.visualization.VisualizerWithVertexSelection()
#     vis.create_window(window_name="Open3D Picker", width=1280, height=800)
    
#     vis.add_geometry(merged_pcd)

#     last_pick_count = 0
    
#     while True:
#         if not vis.poll_events():
#             break
#         vis.update_renderer()

#         picked_points = vis.get_picked_points()
        
#         # 关键检查：确保 picked_points 列表非空
#         if len(picked_points) > 0:
            
#             # --- 仅处理最新点击（列表的最后一个元素） ---
#             latest_point = picked_points[-1] 
#             coord = latest_point.coord
            
#             # --- 逆向计算 ---
#             c = int(round((coord[0] - origin_x) / cell_x))
#             r = int(round((coord[1] - origin_y) / cell_y))
#             idx = r * cols + c
            
#             if 0 <= idx < len(ply_paths):
#                 folder_name = ply_paths[idx].parent.name
#                 print(f"\n[DETECTED] Index: {idx} | ID: {folder_name}")
#                 print(f"  Path: \033[92m{ply_paths[idx]}\033[0m") 
#             else:
#                 print("\n[INFO] Clicked empty space.")
                
#             # *** 核心修复：清空选点历史 ***
#             # 必须在处理完当前点击后立即清空，为下一次点击做准备。
#             vis.clear_picked_points()
            
#             # 由于清空了，我们将 last_pick_count 设为 0
#             last_pick_count = 0 
            
#         else:
#              # 如果列表为空，则记录当前的空状态
#              last_pick_count = 0

#     vis.destroy_window()

# def main():
#     ap = argparse.ArgumentParser()
#     # ap.add_argument("--root", type=str, default='shot_fpfh/descriptors/dataset/model1_norm')
#     ap.add_argument("--root", type=str, default='/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/database_2k')
#     ap.add_argument("--cols", type=int, default=35)
#     # ap.add_argument("--margin_ratio", type=float, default=0.5)
#     ap.add_argument("--margin_x_ratio", type=float, default=0.2, help="列间距比例")
#     ap.add_argument("--margin_y_ratio", type=float, default=0.0, help="行间距比例")
#     ap.add_argument("--skip_pcd", action="store_true", help="仅显示网格")
#     ap.add_argument("--skip_mesh", action="store_true", help="仅显示点云")
#     ap.add_argument("--n", type=int, default=-1)
#     ap.add_argument("--ranges", type=str, default="0-1154")
#     ap.add_argument("--range_step", type=int, default=1,
#                 help="对 --ranges 选中的索引按步长采样，例如 10 表示每隔10个显示1个")
#     ap.add_argument("--seed", type=int, default=0, help="随机采样种子")
    
#     args = ap.parse_args()
#     root = Path(args.root).expanduser().resolve()
    
#     all_ply_paths = sorted([d/"nontextured.ply" for d in root.iterdir()
#                         if d.is_dir() and (d/"nontextured.ply").exists()])
#     all_ply_paths = sort_by_parent_folder(all_ply_paths)
    
#     if not all_ply_paths:
#         print(f"[ERROR] {root} 下未找到任何 nontextured.ply")
#         return

#     # 筛选逻辑
#     # target_ply_paths = []
#     # if args.ranges.strip():
#     #     raw_indices = parse_ranges(args.ranges, len(all_ply_paths))
#     #     step = max(1, args.range_step)
#     #     indices = raw_indices[::step]

#     #     target_ply_paths = [all_ply_paths[i] for i in indices]
#     #     print(f"[INFO] Range 筛选: 原始 {len(raw_indices)} 个 -> 每隔 {step} 个取1个 -> 实际 {len(target_ply_paths)} 个")
#     # elif args.n is not None and args.n > 0:
#     #     target_ply_paths = all_ply_paths[:args.n]
#     #     print(f"[INFO] Top-N 筛选: {len(target_ply_paths)} 个")
#     # else:
#     #     target_ply_paths = all_ply_paths

#     # 筛选逻辑
#     target_ply_paths = []

#     indices = sample_indices_from_segments(len(all_ply_paths), seed=args.seed)

#     target_ply_paths = [all_ply_paths[i] for i in indices]

#     print(f"[INFO] 分段随机采样后共 {len(target_ply_paths)} 个")
#     print(f"[INFO] 索引范围示例: {indices[:20]} ...")

#     geoms = []
#     valid_paths = [] 
    
#     for ply in target_ply_paths:
#         g, t = load_geometry(ply)
#         if g is None:
#             print(f"[WARN] Skip invalid: {ply.parent.name}")
#             continue
#         if (t == "pcd" and args.skip_pcd) or (t == "mesh" and args.skip_mesh):
#             continue
            
#         center_and_drop(g)
#         geoms.append(g)
#         valid_paths.append(ply) 

#     if not geoms:
#         print("[ERROR] 没有几何体可显示。")
#         return

#     # 计算 Grid 参数
#     max_dx = max(aabb_xy_size(g)[0] for g in geoms)
#     max_dy = max(aabb_xy_size(g)[1] for g in geoms)
#     # cell_x = max_dx * (1.0 + args.margin_ratio)
#     # cell_y = max_dy * (1.0 + args.margin_ratio)
#     # cell_x = max_dx * (1.0 + args.margin_x_ratio)
#     cell_x = 0.2
#     cell_y = 0.3

#     n = len(geoms)
#     cols = max(1, args.cols)
#     rows = (n + cols - 1) // cols
    
#     origin_x = - (cols - 1) * cell_x / 2.0
#     origin_y = - (rows - 1) * cell_y / 2.0

#     # 1. 先进行平移排布
#     print("[INFO] 正在排布几何体...")
#     for i, g in enumerate(geoms):
#         r, c = divmod(i, cols)
#         tx = origin_x + c * cell_x
#         ty = origin_y + r * cell_y
#         g.translate((tx, ty, 0.0), relative=True)

#     # 2. 关键修复：合并为一个大的点云
#     # 注意：我们不能把背景网格(Grid Lineset)加进去，因为LineSet不能合并进PointCloud，
#     # 且VisualizerWithVertexSelection只能显示一个Geometry。
#     print("[INFO] 正在合并几何体以支持交互选点...")
#     merged_geom = merge_to_single_pointcloud(geoms)
    
#     layout_info = (origin_x, origin_y, cell_x, cell_y, cols, n)

#     print(f"[INFO] 启动可视化。有效物体: {n}")
    
#     run_interactive_picker(merged_geom, valid_paths, layout_info)

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import copy
from pathlib import Path
import argparse
import open3d as o3d
import numpy as np
import random

# --- 基础加载与处理函数 ---

def is_valid_mesh(m: o3d.geometry.TriangleMesh) -> bool:
    return (m is not None) and (len(m.vertices) > 0) and (len(m.triangles) > 0)

def is_valid_pcd(p: o3d.geometry.PointCloud) -> bool:
    return (p is not None) and (len(p.points) > 0)

def load_geometry(ply_path: Path):
    # 尝试按 Mesh 读取
    try:
        m = o3d.io.read_triangle_mesh(str(ply_path), print_progress=False)
    except Exception:
        m = None

    if m is not None and len(m.vertices) > 0:
        if len(m.triangles) == 0:
            try: m = m.triangulate()
            except: pass
        if is_valid_mesh(m):
            m.compute_vertex_normals()
            return m, "mesh"
    
    # 回退按 PointCloud 读取
    try:
        p = o3d.io.read_point_cloud(str(ply_path), print_progress=False)
        if is_valid_pcd(p):
            p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            return p, "pcd"
    except Exception:
        pass
    return None, None

def aabb_xy_size(geom):
    aabb = geom.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    return float(ext[0]), float(ext[1])

def center_and_drop(geom):
    aabb = geom.get_axis_aligned_bounding_box()
    geom.translate(-aabb.get_center())
    aabb = geom.get_axis_aligned_bounding_box()
    geom.translate((0, 0, -aabb.min_bound[2]))

def sample_indices_from_segments(total, seed=0):
    """
    按指定分段规则采样索引：
    0-403      -> 随机100个
    404-577    -> 随机80个
    578-785    -> 随机100个
    785-1063   -> 随机80个
    1073-1154  -> 全部
    注意：
      - 785 会重复，最后会去重
      - 1064-1072 不包含
    """
    random.seed(seed)

    segments = [
        (0, 403, 100, "random"),
        (404, 577, 80, "random"),
        (578, 785, 100, "random"),
        (785, 1063, 80, "random"),
        (1073, 1154, None, "all"),
    ]

    selected = []

    for start, end, k, mode in segments:
        # 裁剪到合法范围
        start = max(0, start)
        end = min(total - 1, end)
        if start > end:
            continue

        candidates = list(range(start, end + 1))

        if mode == "all":
            picked = candidates
        else:
            kk = min(k, len(candidates))
            picked = random.sample(candidates, kk)

        selected.extend(picked)

    # 去重 + 排序，避免 785 重复
    selected = sorted(set(selected))
    return selected

def parse_ranges(ranges_str: str, total: int):
    if not ranges_str: return []
    parts = [p.strip() for p in ranges_str.split(",") if p.strip()]
    idx_list = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                a, b = int(a), int(b)
                if b < a: a, b = b, a
                idx_list.extend(range(a, b + 1))
            except ValueError: pass
        else:
            try: idx_list.append(int(p))
            except ValueError: pass
    seen = set()
    out = []
    for i in idx_list:
        if 0 <= i < total and i not in seen:
            out.append(i)
            seen.add(i)
    return out

def sort_by_parent_folder(ply_paths):
    try:
        return sorted(ply_paths, key=lambda x: int(x.parent.name))
    except ValueError:
        return sorted(ply_paths)

# --- 核心修改：合并与交互 ---

def merge_to_single_pointcloud(geoms):
    """
    将所有几何体（Mesh或PointCloud）合并为一个巨大的PointCloud。
    VisualizerWithVertexSelection 只能接受一个 geometry。
    """
    combined_pcd = o3d.geometry.PointCloud()
    
    for g in geoms:
        # 如果是Mesh，转换为PointCloud（取顶点）
        if isinstance(g, o3d.geometry.TriangleMesh):
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = g.vertices
            # 保留颜色（如果有）
            if g.has_vertex_colors():
                temp_pcd.colors = g.vertex_colors
            # 保留法向（如果有）
            if g.has_vertex_normals():
                temp_pcd.normals = g.vertex_normals
            combined_pcd += temp_pcd
        elif isinstance(g, o3d.geometry.PointCloud):
            combined_pcd += g
            
    return combined_pcd

def run_interactive_picker(merged_pcd, ply_paths, layout_info):
    origin_x, origin_y, cell_x, cell_y, cols, total_items = layout_info

    print("\n" + "="*60)
    print(" [交互模式说明] ")
    print(" 1. 因工具限制，所有物体已合并为单一视图，且去除了网格线。")
    print(" 2. 按住 [Shift] + [鼠标左键] 点击物体。")
    print(" 3. 终端将输出该物体对应的原始路径。")
    print("="*60 + "\n")

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name="Open3D Picker", width=1280, height=800)
    
    vis.add_geometry(merged_pcd)

    last_pick_count = 0
    
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

        picked_points = vis.get_picked_points()
        
        # 关键检查：确保 picked_points 列表非空
        if len(picked_points) > 0:
            
            # --- 仅处理最新点击（列表的最后一个元素） ---
            latest_point = picked_points[-1] 
            coord = latest_point.coord
            
            # --- 逆向计算 ---
            c = int(round((coord[0] - origin_x) / cell_x))
            r = int(round((coord[1] - origin_y) / cell_y))
            idx = r * cols + c
            
            if 0 <= idx < len(ply_paths):
                folder_name = ply_paths[idx].parent.name
                print(f"\n[DETECTED] Index: {idx} | ID: {folder_name}")
                print(f"  Path: \033[92m{ply_paths[idx]}\033[0m") 
            else:
                print("\n[INFO] Clicked empty space.")
                
            # *** 核心修复：清空选点历史 ***
            # 必须在处理完当前点击后立即清空，为下一次点击做准备。
            vis.clear_picked_points()
            
            # 由于清空了，我们将 last_pick_count 设为 0
            last_pick_count = 0 
            
        else:
             # 如果列表为空，则记录当前的空状态
             last_pick_count = 0

    vis.destroy_window()

# def main():
#     TARGET_IDS = [
#         336, 321, 350, 549, 306,
#         573, 377, 363, 364, 291,
#         391, 349, 525, 335, 378,
#         276, 675, 392, 292, 260,
#         666, 320, 307, 1063, 322,
#         985, 337, 277, 686, 1045,
#         305, 656, 634, 955, 501,
#         1051, 665, 1009, 1069, 351,
#     ]
    
#     ap = argparse.ArgumentParser()
#     # ap.add_argument("--root", type=str, default='shot_fpfh/descriptors/dataset/model1_norm')
#     ap.add_argument("--root", type=str, default='/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/database_2k')
#     ap.add_argument("--cols", type=int, default=5)
#     # ap.add_argument("--margin_ratio", type=float, default=0.5)
#     ap.add_argument("--margin_x_ratio", type=float, default=0.2, help="列间距比例")
#     ap.add_argument("--margin_y_ratio", type=float, default=0.0, help="行间距比例")
#     ap.add_argument("--skip_pcd", action="store_true", help="仅显示网格")
#     ap.add_argument("--skip_mesh", action="store_true", help="仅显示点云")
#     ap.add_argument("--n", type=int, default=-1)
#     ap.add_argument("--ranges", type=str, default="0-1154")
#     ap.add_argument("--range_step", type=int, default=1,
#                 help="对 --ranges 选中的索引按步长采样，例如 10 表示每隔10个显示1个")
#     ap.add_argument("--seed", type=int, default=0, help="随机采样种子")
    
#     args = ap.parse_args()
#     root = Path(args.root).expanduser().resolve()
    
#     all_ply_paths = sorted([d/"nontextured.ply" for d in root.iterdir()
#                         if d.is_dir() and (d/"nontextured.ply").exists()])
#     all_ply_paths = sort_by_parent_folder(all_ply_paths)
    
#     if not all_ply_paths:
#         print(f"[ERROR] {root} 下未找到任何 nontextured.ply")
#         return

#     # 筛选逻辑
#     # target_ply_paths = []
#     # if args.ranges.strip():
#     #     raw_indices = parse_ranges(args.ranges, len(all_ply_paths))
#     #     step = max(1, args.range_step)
#     #     indices = raw_indices[::step]

#     #     target_ply_paths = [all_ply_paths[i] for i in indices]
#     #     print(f"[INFO] Range 筛选: 原始 {len(raw_indices)} 个 -> 每隔 {step} 个取1个 -> 实际 {len(target_ply_paths)} 个")
#     # elif args.n is not None and args.n > 0:
#     #     target_ply_paths = all_ply_paths[:args.n]
#     #     print(f"[INFO] Top-N 筛选: {len(target_ply_paths)} 个")
#     # else:
#     #     target_ply_paths = all_ply_paths

#     # 筛选逻辑
#     # target_ply_paths = []

#     # indices = sample_indices_from_segments(len(all_ply_paths), seed=args.seed)

#     # target_ply_paths = [all_ply_paths[i] for i in indices]

#     # print(f"[INFO] 分段随机采样后共 {len(target_ply_paths)} 个")
#     # print(f"[INFO] 索引范围示例: {indices[:20]} ...")

#     # 只显示指定 ID 对应的 nontextured.ply，顺序按 TARGET_IDS 保持
#     id_to_path = {}
#     for p in all_ply_paths:
#         try:
#             obj_id = int(p.parent.name)
#             id_to_path[obj_id] = p
#         except ValueError:
#             continue

#     target_ply_paths = [id_to_path[obj_id] for obj_id in TARGET_IDS if obj_id in id_to_path]
#     missing_ids = [obj_id for obj_id in TARGET_IDS if obj_id not in id_to_path]

#     print(f"[INFO] 指定 ID 共 {len(TARGET_IDS)} 个，成功找到 {len(target_ply_paths)} 个")
#     if missing_ids:
#         print(f"[WARN] 以下 ID 未找到对应的 nontextured.ply: {missing_ids}")
    
#     geoms = []
#     valid_paths = [] 
    
#     for ply in target_ply_paths:
#         g, t = load_geometry(ply)
#         if g is None:
#             print(f"[WARN] Skip invalid: {ply.parent.name}")
#             continue
#         if (t == "pcd" and args.skip_pcd) or (t == "mesh" and args.skip_mesh):
#             continue
            
#         center_and_drop(g)
#         geoms.append(g)
#         valid_paths.append(ply) 

#     if not geoms:
#         print("[ERROR] 没有几何体可显示。")
#         return

#     # 计算 Grid 参数
#     max_dx = max(aabb_xy_size(g)[0] for g in geoms)
#     max_dy = max(aabb_xy_size(g)[1] for g in geoms)
#     # cell_x = max_dx * (1.0 + args.margin_ratio)
#     # cell_y = max_dy * (1.0 + args.margin_ratio)
#     # cell_x = max_dx * (1.0 + args.margin_x_ratio)
#     cell_x = 0.2
#     cell_y = 0.3

#     n = len(geoms)
#     cols = max(1, args.cols)
#     rows = (n + cols - 1) // cols
    
#     origin_x = - (cols - 1) * cell_x / 2.0
#     origin_y = - (rows - 1) * cell_y / 2.0

#     # 1. 先进行平移排布
#     print("[INFO] 正在排布几何体...")
#     for i, g in enumerate(geoms):
#         r, c = divmod(i, cols)
#         tx = origin_x + c * cell_x
#         ty = origin_y + r * cell_y
#         g.translate((tx, ty, 0.0), relative=True)

#     # 2. 关键修复：合并为一个大的点云
#     # 注意：我们不能把背景网格(Grid Lineset)加进去，因为LineSet不能合并进PointCloud，
#     # 且VisualizerWithVertexSelection只能显示一个Geometry。
#     print("[INFO] 正在合并几何体以支持交互选点...")
#     merged_geom = merge_to_single_pointcloud(geoms)
    
#     layout_info = (origin_x, origin_y, cell_x, cell_y, cols, n)

#     print(f"[INFO] 启动可视化。有效物体: {n}")
    
#     run_interactive_picker(merged_geom, valid_paths, layout_info)

# if __name__ == "__main__":
#     main()


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--root", type=str, default='shot_fpfh/descriptors/dataset/model1_norm')
    ap.add_argument("--root", type=str, default='/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/database_2k')
    ap.add_argument("--cols", type=int, default=35)
    # ap.add_argument("--margin_ratio", type=float, default=0.5)
    ap.add_argument("--margin_x_ratio", type=float, default=0.2, help="列间距比例")
    ap.add_argument("--margin_y_ratio", type=float, default=0.0, help="行间距比例")
    ap.add_argument("--skip_pcd", action="store_true", help="仅显示网格")
    ap.add_argument("--skip_mesh", action="store_true", help="仅显示点云")
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--ranges", type=str, default="0-1154")
    ap.add_argument("--range_step", type=int, default=1,
                help="对 --ranges 选中的索引按步长采样，例如 10 表示每隔10个显示1个")
    ap.add_argument("--seed", type=int, default=0, help="随机采样种子")
    
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    
    all_ply_paths = sorted([d/"nontextured.ply" for d in root.iterdir()
                        if d.is_dir() and (d/"nontextured.ply").exists()])
    all_ply_paths = sort_by_parent_folder(all_ply_paths)
    
    if not all_ply_paths:
        print(f"[ERROR] {root} 下未找到任何 nontextured.ply")
        return

    # 筛选逻辑
    # target_ply_paths = []
    # if args.ranges.strip():
    #     raw_indices = parse_ranges(args.ranges, len(all_ply_paths))
    #     step = max(1, args.range_step)
    #     indices = raw_indices[::step]

    #     target_ply_paths = [all_ply_paths[i] for i in indices]
    #     print(f"[INFO] Range 筛选: 原始 {len(raw_indices)} 个 -> 每隔 {step} 个取1个 -> 实际 {len(target_ply_paths)} 个")
    # elif args.n is not None and args.n > 0:
    #     target_ply_paths = all_ply_paths[:args.n]
    #     print(f"[INFO] Top-N 筛选: {len(target_ply_paths)} 个")
    # else:
    #     target_ply_paths = all_ply_paths

    # 筛选逻辑
    target_ply_paths = []

    indices = sample_indices_from_segments(len(all_ply_paths), seed=args.seed)

    target_ply_paths = [all_ply_paths[i] for i in indices]

    print(f"[INFO] 分段随机采样后共 {len(target_ply_paths)} 个")
    print(f"[INFO] 索引范围示例: {indices[:20]} ...")

    geoms = []
    valid_paths = [] 
    
    for ply in target_ply_paths:
        g, t = load_geometry(ply)
        if g is None:
            print(f"[WARN] Skip invalid: {ply.parent.name}")
            continue
        if (t == "pcd" and args.skip_pcd) or (t == "mesh" and args.skip_mesh):
            continue
            
        center_and_drop(g)
        geoms.append(g)
        valid_paths.append(ply) 

    if not geoms:
        print("[ERROR] 没有几何体可显示。")
        return

    # 计算 Grid 参数
    max_dx = max(aabb_xy_size(g)[0] for g in geoms)
    max_dy = max(aabb_xy_size(g)[1] for g in geoms)
    # cell_x = max_dx * (1.0 + args.margin_ratio)
    # cell_y = max_dy * (1.0 + args.margin_ratio)
    # cell_x = max_dx * (1.0 + args.margin_x_ratio)
    cell_x = 0.2
    cell_y = 0.3

    n = len(geoms)
    cols = max(1, args.cols)
    rows = (n + cols - 1) // cols
    
    origin_x = - (cols - 1) * cell_x / 2.0
    origin_y = - (rows - 1) * cell_y / 2.0

    # 1. 先进行平移排布
    print("[INFO] 正在排布几何体...")
    for i, g in enumerate(geoms):
        r, c = divmod(i, cols)
        tx = origin_x + c * cell_x
        ty = origin_y + r * cell_y
        g.translate((tx, ty, 0.0), relative=True)

    # 2. 关键修复：合并为一个大的点云
    # 注意：我们不能把背景网格(Grid Lineset)加进去，因为LineSet不能合并进PointCloud，
    # 且VisualizerWithVertexSelection只能显示一个Geometry。
    print("[INFO] 正在合并几何体以支持交互选点...")
    merged_geom = merge_to_single_pointcloud(geoms)
    
    layout_info = (origin_x, origin_y, cell_x, cell_y, cols, n)

    print(f"[INFO] 启动可视化。有效物体: {n}")
    
    run_interactive_picker(merged_geom, valid_paths, layout_info)

if __name__ == "__main__":
    main()
