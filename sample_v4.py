# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 保守版本，全为稳定抓取 + 保障：深入会碰撞则不深入
# 并增加：候选为空/有效抓取为空的ID打印 + 原因统计 + JSON报告
# """

# import os
# import json
# import math
# import argparse
# from dataclasses import dataclass, asdict
# from typing import List, Tuple, Optional, Callable, Dict, Any

# import numpy as np
# import open3d as o3d
# from tqdm import tqdm

# # ---------------- Hand parameters ----------------
# HAND_PARAMS_REAL = {
#     "finger_width":        0.015,
#     "hand_outer_diameter": 0.167,
#     "hand_depth":          0.0475,
#     "hand_height":         0.02,
#     "palm_thickness":      0.004,
# }

# # ============================================================
# # 0) Basic utils
# # ============================================================
# def normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=float)
#     n = np.linalg.norm(v)
#     if n < 1e-12:
#         return v
#     return v / n

# def unit(v: np.ndarray) -> np.ndarray:
#     return normalize(v)

# def orthonormal_basis_from(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Return two unit vectors (e1,e2) that span the plane orthogonal to v."""
#     v = unit(v)
#     if abs(v[0]) < 0.9:
#         ref = np.array([1.0, 0.0, 0.0], dtype=float)
#     else:
#         ref = np.array([0.0, 1.0, 0.0], dtype=float)
#     e1 = np.cross(v, ref)
#     e1 = unit(e1)
#     e2 = np.cross(v, e1)
#     e2 = unit(e2)
#     return e1, e2

# def fibonacci_sphere(n: int) -> np.ndarray:
#     """Nearly-uniform directions on a sphere. Returns (n,3). Deterministic."""
#     n = int(n)
#     if n <= 0:
#         return np.zeros((0, 3), dtype=float)
#     ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
#     dirs = []
#     for i in range(n):
#         z = 1.0 - 2.0 * (i + 0.5) / n
#         r = math.sqrt(max(0.0, 1.0 - z * z))
#         theta = ga * i
#         x = r * math.cos(theta)
#         y = r * math.sin(theta)
#         dirs.append([x, y, z])
#     return np.asarray(dirs, dtype=float)

# def aabb_center(pts: np.ndarray) -> np.ndarray:
#     mn = pts.min(axis=0)
#     mx = pts.max(axis=0)
#     return 0.5 * (mn + mx)

# def safe_insert_distance(r_eff: float, hand_params: dict, frac: float, safety: float = 0.002) -> float:
#     """
#     计算“沿 approach 深入”的插入量，并做 hand_depth 上限保护（保守限幅）。
#     - r_eff: 有效半径（圆柱/球= r；椭圆柱=短轴半径；圆台=该 z 截面半径；椭球=最短半轴）
#     - safety: 预留裕量，避免掌心贴太近（m）
#     """
#     r_eff = float(r_eff)
#     frac = float(frac)
#     if frac <= 0.0 or r_eff <= 1e-12:
#         return 0.0
#     depth = float(hand_params["hand_depth"])
#     desired = frac * r_eff
#     insert_max = max(0.0, depth - r_eff - float(safety))
#     return float(min(desired, insert_max))

# def obj_id_from_folder(folder: str) -> str:
#     return os.path.basename(os.path.normpath(folder))

# # ============================================================
# # 1) Core geometry helpers
# # ============================================================
# def world_to_local(P: np.ndarray, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
#     """R: 3x3 with columns (approach, binormal, axis) in world."""
#     return (R.T @ (P - pos).T).T

# def make_frame(approach: np.ndarray, binormal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Return (approach, binormal, axis) as an orthonormal right-handed frame."""
#     a = unit(approach)
#     b = unit(binormal)
#     ax = np.cross(a, b)
#     if np.linalg.norm(ax) < 1e-8:
#         return a, b, np.array([0.0, 0.0, 0.0], dtype=float)
#     ax = unit(ax)
#     b = unit(b - np.dot(b, a) * a)
#     ax = unit(np.cross(a, b))
#     return a, b, ax

# def find_palm_center_direct(
#     p_sample: np.ndarray,
#     approach: np.ndarray,
#     hand_params: dict,
# ) -> np.ndarray:
#     """
#     解析式：已知 p_sample 为“指尖中心/夹爪闭合中线在指尖处的参考点”，
#     手掌中心 = p_sample 沿 -approach 平移 x_tip，其中 x_tip = palm_t/2 + depth
#     """
#     approach = unit(approach)
#     fw      = float(hand_params["finger_width"])
#     depth   = float(hand_params["hand_depth"])
#     palm_t  = float(hand_params.get("palm_thickness", fw))
#     x_tip   = palm_t / 2.0 + depth
#     return p_sample - x_tip * approach

# def collide_finger_palm(P_local: np.ndarray, width: float, hand_params: dict, eps_contact: float = 5e-4) -> bool:
#     """
#     基础碰撞检测（闭合到 width 后）：
#     - 允许点落在指内侧表面附近（|y|-width/2 <= eps_contact）
#     - 禁止点进入指体厚度 fw 或掌心体积
#     """
#     fw      = float(hand_params["finger_width"])
#     depth   = float(hand_params["hand_depth"])
#     h       = float(hand_params["hand_height"])
#     palm_t  = float(hand_params.get("palm_thickness", fw))

#     x = P_local[:, 0]
#     y = P_local[:, 1]
#     z = P_local[:, 2]

#     x_front = palm_t / 2.0
#     x_tip   = palm_t / 2.0 + depth
#     yL      = width / 2.0

#     # 指体（内侧表面 y=±yL 允许少量接触容差，指体厚度向外扩 fw）
#     mask_finger = (
#         (x >= x_front) & (x <= x_tip) &
#         (np.abs(z) <= h / 2.0) &
#         (
#             ((y >= ( yL + eps_contact)) & (y <= ( yL + fw))) |
#             ((y <= (-yL - eps_contact)) & (y >= (-yL - fw)))
#         )
#     )

#     # 掌心体积（x 在 [-palm_t/2, +palm_t/2]）
#     mask_palm = (
#         (x >= -palm_t / 2.0) & (x <= palm_t / 2.0) &
#         (np.abs(z) <= h / 2.0) &
#         (np.abs(y) <= (yL + fw))
#     )

#     return bool(np.any(mask_finger | mask_palm))

# def require_some_insertion(P_local: np.ndarray, width: float, hand_params: dict, min_insert: float) -> bool:
#     """
#     可选：要求物体在手指通道内至少进入 min_insert 深度。
#     min_insert=0 时等价于不启用该约束。
#     """
#     if min_insert <= 0.0:
#         return True

#     fw      = float(hand_params["finger_width"])
#     depth   = float(hand_params["hand_depth"])
#     h       = float(hand_params["hand_height"])
#     palm_t  = float(hand_params.get("palm_thickness", fw))

#     x = P_local[:, 0]
#     y = P_local[:, 1]
#     z = P_local[:, 2]

#     x_front = palm_t / 2.0
#     x_tip   = palm_t / 2.0 + depth
#     yL      = width / 2.0

#     mask_channel = (
#         (x >= x_front) & (x <= x_tip) &
#         (np.abs(z) <= h / 2.0) &
#         (np.abs(y) <= yL)
#     )
#     if not np.any(mask_channel):
#         return False

#     x_cloud_front = float(np.max(x[mask_channel]))
#     return (x_cloud_front - x_front) >= float(min_insert)

# # ============================================================
# # 2) Primitive helper math (analytic widths)
# # ============================================================
# def frustum_radius_at_z(r_bottom: float, r_top: float, H: float, z: float) -> float:
#     """z in [-H/2, +H/2], bottom at -H/2."""
#     if H < 1e-9:
#         return float(min(r_bottom, r_top))
#     t = (z + H / 2.0) / H
#     return float(r_bottom + t * (r_top - r_bottom))

# def regular_ngon_ray_max_t(n: int, Rv: float, theta: float) -> float:
#     """
#     Regular n-gon centered at origin, vertex radius Rv.
#     Max t s.t. t*[cosθ,sinθ] inside polygon (apothem half-space).
#     """
#     n = int(n)
#     if n < 3:
#         return 0.0
#     apothem = Rv * math.cos(math.pi / n)
#     u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
#     best_t = float("inf")
#     for i in range(n):
#         phi = 2.0 * math.pi * (i + 0.5) / n
#         ni = np.array([math.cos(phi), math.sin(phi)], dtype=float)
#         denom = float(np.dot(ni, u))
#         if denom <= 1e-9:
#             continue
#         t = apothem / denom
#         if t < best_t:
#             best_t = t
#     if not np.isfinite(best_t):
#         best_t = Rv
#     return float(best_t)

# def ellipse_radius_along_dir(rx: float, ry: float, ux: float, uy: float) -> float:
#     """Ellipse (x/rx)^2+(y/ry)^2=1, return boundary radius along direction u."""
#     denom = math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2) + 1e-12
#     return 1.0 / denom

# def ellipsoid_radius_along_dir(rx: float, ry: float, rz: float, u: np.ndarray) -> float:
#     """Ellipsoid (x/rx)^2+(y/ry)^2+(z/rz)^2=1, boundary radius along direction u."""
#     u = unit(u)
#     denom = math.sqrt((u[0] / rx) ** 2 + (u[1] / ry) ** 2 + (u[2] / rz) ** 2) + 1e-12
#     return 1.0 / denom

# def make_width_fn(ptype: str, dims: list, delta_close: float, hand_params: dict) -> Optional[Callable]:
#     """
#     对规则体优先使用解析 width（避免点云稀疏导致 width 偏小）。
#     兼容 top-bottom pinch（binormal≈z）时，width 用高度 H。
#     """
#     fw   = float(hand_params["finger_width"])
#     Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

#     def _ok(w: float) -> Optional[float]:
#         w = float(w)
#         return w if (w > 0.0 and w <= Wmax) else None

#     if ptype == "cylinder":
#         r, H = float(dims[0]), float(dims[1])
#         def _w(p_sample, approach, binormal, axis):
#             b = np.abs(np.asarray(binormal, dtype=float))
#             if b[2] >= max(b[0], b[1]):  # pinch top-bottom
#                 return _ok(H + float(delta_close))
#             return _ok(2.0 * r + float(delta_close))
#         return _w

#     if ptype == "ellip_cyl":
#         rx, ry, H = float(dims[0]), float(dims[1]), float(dims[2])
#         def _w(p_sample, approach, binormal, axis):
#             b = np.abs(np.asarray(binormal, dtype=float))
#             if b[2] >= max(b[0], b[1]):  # pinch top-bottom
#                 return _ok(H + float(delta_close))
#             bxy = np.array([binormal[0], binormal[1]], dtype=float)
#             nb = np.linalg.norm(bxy)
#             if nb < 1e-9:
#                 return None
#             bxy /= nb
#             rad = ellipse_radius_along_dir(rx, ry, float(bxy[0]), float(bxy[1]))
#             return _ok(2.0 * rad + float(delta_close))
#         return _w

#     if ptype == "frustum":
#         rb, rt, H = float(dims[0]), float(dims[1]), float(dims[2])
#         def _w(p_sample, approach, binormal, axis):
#             b = np.abs(np.asarray(binormal, dtype=float))
#             if b[2] >= max(b[0], b[1]):  # pinch top-bottom
#                 return _ok(H + float(delta_close))
#             z = float(p_sample[2])
#             rad = frustum_radius_at_z(rb, rt, H, z)
#             return _ok(2.0 * rad + float(delta_close))
#         return _w

#     if ptype == "sphere":
#         r = float(dims[0])
#         def _w(p_sample, approach, binormal, axis):
#             return _ok(2.0 * r + float(delta_close))
#         return _w

#     if ptype == "ellipsoid":
#         rx, ry, rz = float(dims[0]), float(dims[1]), float(dims[2])
#         def _w(p_sample, approach, binormal, axis):
#             rad = ellipsoid_radius_along_dir(rx, ry, rz, binormal)
#             return _ok(2.0 * rad + float(delta_close))
#         return _w

#     if ptype == "poly_prism":
#         n, Rv, H = int(round(float(dims[0]))), float(dims[1]), float(dims[2])
#         def _w(p_sample, approach, binormal, axis):
#             b = np.asarray(binormal, dtype=float)
#             if abs(float(b[2])) >= max(abs(float(b[0])), abs(float(b[1]))):
#                 return _ok(float(H) + float(delta_close))
#             th = math.atan2(float(b[1]), float(b[0]))
#             r_plus  = regular_ngon_ray_max_t(n, Rv, th)
#             r_minus = regular_ngon_ray_max_t(n, Rv, th + math.pi)
#             rad = max(r_plus, r_minus)  # odd n needs both sides
#             return _ok(2.0 * rad + float(delta_close))
#         return _w

#     if ptype == "box":
#         w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
#         def _w(p_sample, approach, binormal, axis):
#             v = np.abs(np.asarray(binormal, dtype=float))
#             k = int(np.argmax(v))
#             size = [w, h, d][k]
#             return _ok(float(size) + float(delta_close))
#         return _w

#     return None

# # ============================================================
# # 3) Candidate generators per primitive
# # 统一返回 5 元组: (p_base, p_ins, approach, binormal, axis)
# # ============================================================
# def z_levels_for_side_grasp(H: float, hand_h: float, step: float, margin: float = 0.002) -> np.ndarray:
#     """
#     在 step 网格上用整数索引采样，确保：
#     - z 间距严格为 step
#     - 若中心 0 在可行范围内，会自动包含
#     - 仍保留 margin 与 hand_h 的端面避让逻辑
#     """
#     if H <= 0:
#         return np.array([0.0], dtype=float)

#     step = float(step)
#     if step <= 0:
#         return np.array([0.0], dtype=float)

#     zmin = -H / 2.0 + hand_h / 2.0 + margin
#     zmax =  H / 2.0 - hand_h / 2.0 - margin
#     if zmin > zmax:
#         return np.array([0.0], dtype=float)

#     eps = 1e-12
#     k_min = int(math.ceil((zmin - eps) / step))
#     k_max = int(math.floor((zmax + eps) / step))

#     if k_min > k_max:
#         return np.array([0.0], dtype=float)

#     ks = np.arange(k_min, k_max + 1, dtype=np.int64)
#     zs = ks.astype(float) * step
#     zs = zs[(zs >= zmin - 1e-9) & (zs <= zmax + 1e-9)]

#     if zs.size == 0:
#         return np.array([0.0], dtype=float)

#     return zs

# def gen_side_family_angles(angle_step_deg: float) -> np.ndarray:
#     step = max(1e-6, float(angle_step_deg))
#     degs = np.arange(0.0, 360.0, step, dtype=float)
#     return np.deg2rad(degs)

# def gen_candidates_cylinder(r: float, H: float, hand_params: dict,
#                             angle_step_deg: float, height_step: float,
#                             include_topdown: bool, include_topbottom: bool,
#                             topdown_insert_depths: List[float],
#                             approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     cands = []
#     r = float(r); H = float(H)

#     hand_h = float(hand_params["hand_height"])
#     thetas = gen_side_family_angles(angle_step_deg)
#     zs = z_levels_for_side_grasp(H, hand_h, height_step)

#     # Side family
#     for z in zs:
#         for th in thetas:
#             u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#             approach = -u
#             binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)
#             approach, binormal, axis = make_frame(approach, binormal)
#             if np.linalg.norm(axis) < 1e-8:
#                 continue

#             p_base = np.array([0.0, 0.0, float(z)], dtype=float)
#             ins = safe_insert_distance(r_eff=r, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
#             p_ins = p_base + ins * approach
#             cands.append((p_base, p_ins, approach, binormal, axis))

#     # Top-down family
#     if include_topdown:
#         for side in (+1.0, -1.0):
#             face_z = side * (H / 2.0)
#             for d in topdown_insert_depths:
#                 z_s = face_z - side * float(d)
#                 for th in thetas:
#                     approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
#                     binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#                     approach, binormal, axis = make_frame(approach, binormal)
#                     if np.linalg.norm(axis) < 1e-8:
#                         continue
#                     p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
#                     cands.append((p_base, p_base, approach, binormal, axis))

#     # Top-bottom pinch (binormal=z)
#     if include_topbottom:
#         fw = float(hand_params["finger_width"])
#         Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
#         if H < Wmax + 0.02:
#             thetas2 = gen_side_family_angles(15.0)
#             for th in thetas2:
#                 u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#                 approach = -u
#                 binormal = np.array([0.0, 0.0, 1.0], dtype=float)
#                 approach, binormal, axis = make_frame(approach, binormal)
#                 if np.linalg.norm(axis) < 1e-8:
#                     continue
#                 p_base = np.array([0.0, 0.0, 0.0], dtype=float)
#                 cands.append((p_base, p_base, approach, binormal, axis))

#     return cands

# def gen_candidates_ellip_cyl(rx: float, ry: float, H: float, hand_params: dict,
#                             angle_step_deg: float, height_step: float,
#                             include_topdown: bool, include_topbottom: bool,
#                             topdown_insert_depths: List[float],
#                             approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     cands = []
#     rx = float(rx); ry = float(ry); H = float(H)

#     hand_h = float(hand_params["hand_height"])
#     zs = z_levels_for_side_grasp(H, hand_h, height_step)

#     circle_tol = 1e-6
#     if abs(rx - ry) <= circle_tol:
#         thetas_side = gen_side_family_angles(angle_step_deg)
#         thetas_top  = gen_side_family_angles(angle_step_deg)
#     else:
#         thetas_side = np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=float)
#         thetas_top  = thetas_side.copy()

#     # Side family
#     for z in zs:
#         for th in thetas_side:
#             u2 = np.array([math.cos(th), math.sin(th)], dtype=float)
#             approach = -np.array([u2[0], u2[1], 0.0], dtype=float)
#             binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)

#             approach, binormal, axis = make_frame(approach, binormal)
#             if np.linalg.norm(axis) < 1e-8:
#                 continue

#             p_base = np.array([0.0, 0.0, float(z)], dtype=float)
#             r_ins = min(rx, ry)
#             ins = safe_insert_distance(r_eff=r_ins, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
#             p_ins = p_base + ins * approach
#             cands.append((p_base, p_ins, approach, binormal, axis))

#     # Top-down family
#     if include_topdown:
#         for side in (+1.0, -1.0):
#             face_z = side * (H / 2.0)
#             for d_ins in topdown_insert_depths:
#                 z_s = face_z - side * float(d_ins)
#                 for th in thetas_top:
#                     approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
#                     binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)

#                     approach, binormal, axis = make_frame(approach, binormal)
#                     if np.linalg.norm(axis) < 1e-8:
#                         continue

#                     p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
#                     cands.append((p_base, p_base, approach, binormal, axis))

#     # Top-bottom pinch (binormal=z)
#     if include_topbottom:
#         fw = float(hand_params["finger_width"])
#         Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
#         if H < Wmax + 0.02:
#             thetas2 = gen_side_family_angles(15.0)
#             for th in thetas2:
#                 u2 = np.array([math.cos(th), math.sin(th)], dtype=float)
#                 approach = -np.array([u2[0], u2[1], 0.0], dtype=float)
#                 binormal = np.array([0.0, 0.0, 1.0], dtype=float)
#                 approach, binormal, axis = make_frame(approach, binormal)
#                 if np.linalg.norm(axis) < 1e-8:
#                     continue
#                 p_base = np.array([0.0, 0.0, 0.0], dtype=float)
#                 cands.append((p_base, p_base, approach, binormal, axis))

#     return cands

# def gen_candidates_frustum(r_bottom: float, r_top: float, H: float, hand_params: dict,
#                            angle_step_deg: float, height_step: float,
#                            include_topdown: bool, include_topbottom: bool,
#                            topdown_insert_depths: List[float],
#                            approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     cands = []
#     rb = float(r_bottom); rt = float(r_top); H = float(H)

#     hand_h = float(hand_params["hand_height"])
#     thetas = gen_side_family_angles(angle_step_deg)
#     zs = z_levels_for_side_grasp(H, hand_h, height_step)

#     # Side family: each z cross-section is circle
#     for z in zs:
#         for th in thetas:
#             u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#             approach = -u
#             binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)
#             approach, binormal, axis = make_frame(approach, binormal)
#             if np.linalg.norm(axis) < 1e-8:
#                 continue

#             p_base = np.array([0.0, 0.0, float(z)], dtype=float)
#             rad = frustum_radius_at_z(rb, rt, H, float(z))
#             ins = safe_insert_distance(r_eff=rad, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
#             p_ins = p_base + ins * approach
#             cands.append((p_base, p_ins, approach, binormal, axis))

#     if include_topdown:
#         for side in (+1.0, -1.0):
#             face_z = side * (H / 2.0)
#             for d in topdown_insert_depths:
#                 z_s = face_z - side * float(d)
#                 for th in thetas:
#                     approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
#                     binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#                     approach, binormal, axis = make_frame(approach, binormal)
#                     if np.linalg.norm(axis) < 1e-8:
#                         continue
#                     p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
#                     cands.append((p_base, p_base, approach, binormal, axis))

#     if include_topbottom:
#         fw = float(hand_params["finger_width"])
#         Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
#         if H < Wmax + 0.02:
#             thetas2 = gen_side_family_angles(15.0)
#             for th in thetas2:
#                 u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#                 approach = -u
#                 binormal = np.array([0.0, 0.0, 1.0], dtype=float)
#                 approach, binormal, axis = make_frame(approach, binormal)
#                 if np.linalg.norm(axis) < 1e-8:
#                     continue
#                 p_base = np.array([0.0, 0.0, 0.0], dtype=float)
#                 cands.append((p_base, p_base, approach, binormal, axis))

#     return cands

# def gen_candidates_sphere(r: float, n_dirs: int, n_rot: int,
#                          approach_insert_frac: float, hand_params: dict) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     cands = []
#     dirs = fibonacci_sphere(int(n_dirs))
#     center = np.zeros(3, dtype=float)
#     r = float(r)

#     # 深入量（对球统一按 r）
#     ins = safe_insert_distance(r_eff=r, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

#     for d in dirs:
#         d = unit(d)
#         approach = -d
#         e1, e2 = orthonormal_basis_from(approach)
#         for k in range(int(n_rot)):
#             psi = 2.0 * math.pi * k / max(1, int(n_rot))
#             binormal = math.cos(psi) * e1 + math.sin(psi) * e2
#             a, b, c = make_frame(approach, binormal)
#             if np.linalg.norm(c) < 1e-8:
#                 continue
#             p_base = center
#             p_ins = p_base + ins * a
#             cands.append((p_base, p_ins, a, b, c))
#     return cands

# def gen_candidates_ellipsoid(rx: float, ry: float, rz: float,
#                             angle_step_deg: float = 20.0,
#                             use_extremes_only: bool = False,
#                             add_binormal_sign: bool = True,
#                             approach_insert_frac: float = 1/3,
#                             hand_params: Optional[dict] = None
#                             ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     椭球抓取：
#     - p_base 固定中心
#     - binormal 取主轴方向（±x/±y/±z 或仅最长+最短）
#     - approach 在 binormal 正交平面内旋转采样
#     - p_ins = p_base + insert * approach
#     """
#     if hand_params is None:
#         hand_params = HAND_PARAMS_REAL

#     cands = []
#     center = np.zeros(3, dtype=float)
#     rx = float(rx); ry = float(ry); rz = float(rz)

#     axes = [
#         np.array([1.0, 0.0, 0.0], dtype=float),
#         np.array([0.0, 1.0, 0.0], dtype=float),
#         np.array([0.0, 0.0, 1.0], dtype=float),
#     ]
#     radii = [rx, ry, rz]

#     if use_extremes_only:
#         i_max = int(np.argmax(radii))
#         i_min = int(np.argmin(radii))
#         idxs = [i_max] if i_max == i_min else [i_max, i_min]
#         base_binormals = [axes[i] for i in idxs]
#     else:
#         base_binormals = axes

#     step = max(1e-6, float(angle_step_deg))
#     psis = np.deg2rad(np.arange(0.0, 360.0, step, dtype=float))

#     r_ins = min(rx, ry, rz)
#     ins = safe_insert_distance(r_eff=r_ins, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

#     signs = (+1.0, -1.0) if add_binormal_sign else (+1.0,)
#     for b0 in base_binormals:
#         for s in signs:
#             binormal = float(s) * b0
#             e1, e2 = orthonormal_basis_from(binormal)
#             for psi in psis:
#                 approach = math.cos(float(psi)) * e1 + math.sin(float(psi)) * e2
#                 a, b, c = make_frame(approach, binormal)
#                 if np.linalg.norm(c) < 1e-8:
#                     continue
#                 p_base = center
#                 p_ins = p_base + ins * a
#                 cands.append((p_base, p_ins, a, b, c))

#     return cands

# def gen_candidates_poly_prism(
#     n: int,
#     Rv: float,
#     H: float,
#     hand_params: dict,
#     height_step: float = 0.01,
#     edge_margin: float = 0.01,
#     yaw_face0: float = 0.0,
#     include_topbottom: bool = True,
#     topbottom_angle_step_deg: float = 15.0,
#     approach_insert_frac: float = 1/3,
# ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     正 n 棱柱候选（按你原逻辑保留）：
#     - Side family: 每个 z 采 n 个（每个侧面一个）
#       binormal: 侧面外法向（⊥侧面）
#       axis:     +z
#       approach: (binormal × z) 切向
#       p_base:   (0,0,z)
#     """
#     cands = []
#     n = int(n)
#     Rv = float(Rv); H = float(H)

#     hand_h = float(hand_params["hand_height"])
#     zs = z_levels_for_side_grasp(H, hand_h, step=float(height_step), margin=float(edge_margin))
#     z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

#     # 深入：棱柱这里 approach 是切向，不一定“深入更稳”，但仍按你的需求生成 p_ins（若限幅为 0 就等于不深入）
#     ins = safe_insert_distance(r_eff=Rv, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

#     for z in zs:
#         p_base0 = np.array([0.0, 0.0, float(z)], dtype=float)
#         for i in range(n):
#             phi = float(yaw_face0) + 2.0 * math.pi * float(i) / float(n)
#             binormal = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=float)

#             approach = np.cross(binormal, z_axis)  # tangent
#             a, b, c = make_frame(approach, binormal)
#             if np.linalg.norm(c) < 1e-8:
#                 continue

#             if np.dot(c, z_axis) < 0.0:
#                 b = -b
#                 c = -c

#             p_base = p_base0.copy()
#             p_ins = p_base + ins * a
#             cands.append((p_base, p_ins, a, b, c))

#     if include_topbottom:
#         fw = float(hand_params["finger_width"])
#         Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
#         if H < Wmax + 0.02:
#             thetas = gen_side_family_angles(topbottom_angle_step_deg)
#             for th in thetas:
#                 approach = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
#                 binormal = z_axis.copy()
#                 a, b, c = make_frame(approach, binormal)
#                 if np.linalg.norm(c) < 1e-8:
#                     continue
#                 p_base = np.array([0.0, 0.0, 0.0], dtype=float)
#                 cands.append((p_base, p_base, a, b, c))

#     return cands

# def axis_slide_offsets(half_len: float, edge_margin: float, step: float) -> List[float]:
#     """
#     在 [-half_len+edge_margin, +half_len-edge_margin] 内，
#     以 step 为网格返回对称 offsets（不含 0，避免重复基准 grasp）。
#     """
#     half_len = float(half_len)
#     edge_margin = float(edge_margin)
#     step = float(step)

#     lim = half_len - edge_margin
#     if lim <= 1e-9 or step <= 1e-12:
#         return []

#     eps = 1e-12
#     kmax = int(math.floor((lim + eps) / step))
#     if kmax <= 0:
#         return []

#     offs = []
#     for k in range(1, kmax + 1):
#         offs.append(-k * step)
#         offs.append(+k * step)
#     return offs

# def gen_candidates_box(w: float, h: float, d: float,
#                        surface_insert: float = 0.01,
#                        add_roll90: bool = True,
#                        enable_axis_slide: bool = True,
#                        axis_step: float = 0.01,
#                        edge_margin: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     """
#     基准：6 面 * 2 roll = 12
#     扩展：每个 pose 沿 axis 正反向平移（步长 axis_step），直到距边 >= edge_margin
#     返回 (p_base, p_ins, approach, binormal, axis)，这里 p_ins=p_base（box 不做额外深入）
#     """
#     cands = []

#     axes = {
#         "x": np.array([1.0, 0.0, 0.0], dtype=float),
#         "y": np.array([0.0, 1.0, 0.0], dtype=float),
#         "z": np.array([0.0, 0.0, 1.0], dtype=float),
#     }
#     size = {"x": float(w), "y": float(h), "z": float(d)}
#     half = {"x": float(w) / 2.0, "y": float(h) / 2.0, "z": float(d) / 2.0}
#     keys = ["x", "y", "z"]

#     def append_with_axis_slide(p_sample: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
#         p_base = p_sample
#         cands.append((p_base, p_base, a, b, c))
#         if not enable_axis_slide:
#             return

#         kk = int(np.argmax(np.abs(c)))
#         kkey = keys[kk]

#         if abs(float(p_sample[kk])) > 1e-9:
#             return

#         offs = axis_slide_offsets(half[kkey], edge_margin=edge_margin, step=axis_step)
#         for off in offs:
#             p2 = p_sample + float(off) * c
#             cands.append((p2, p2, a, b, c))

#     for akey in ["x", "y", "z"]:
#         for side in (+1.0, -1.0):
#             outward = side * axes[akey]
#             approach0 = -outward  # into object
#             p_face = outward * half[akey]

#             ins = min(float(surface_insert), float(size[akey]))
#             p_sample = p_face + ins * approach0

#             tkeys = [k for k in ["x", "y", "z"] if k != akey]
#             t1 = tkeys[0]

#             binormal0 = axes[t1]
#             a, b, c = make_frame(approach0, binormal0)
#             if np.linalg.norm(c) < 1e-8:
#                 continue
#             append_with_axis_slide(p_sample, a, b, c)

#             if add_roll90:
#                 b2 = c
#                 a2, b2, c2 = make_frame(a, b2)
#                 if np.linalg.norm(c2) < 1e-8:
#                     continue
#                 append_with_axis_slide(p_sample, a2, b2, c2)

#     return cands

# # ============================================================
# # 4) Poly prism centering helpers
# # ============================================================
# def _cross2d(o, a, b) -> float:
#     return float((a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]))

# def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
#     """Andrew monotone chain. 返回按逆时针顺序的凸包顶点 (M,2)。"""
#     pts = np.asarray(points_xy, dtype=float)
#     if pts.shape[0] < 3:
#         return pts

#     pts = np.unique(pts, axis=0)
#     if pts.shape[0] < 3:
#         return pts
#     pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

#     lower = []
#     for p in pts:
#         while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
#             lower.pop()
#         lower.append(p)
#     upper = []
#     for p in pts[::-1]:
#         while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
#             upper.pop()
#         upper.append(p)

#     hull = np.vstack((lower[:-1], upper[:-1]))
#     return hull

# def polygon_area_centroid(poly: np.ndarray) -> Tuple[float, np.ndarray]:
#     """
#     poly: (M,2) convex polygon vertices in CCW order.
#     返回 (area, centroid_xy)
#     """
#     P = np.asarray(poly, dtype=float)
#     if P.shape[0] < 3:
#         return 0.0, np.mean(P, axis=0) if P.shape[0] > 0 else np.zeros(2)

#     x = P[:, 0]; y = P[:, 1]
#     x2 = np.roll(x, -1); y2 = np.roll(y, -1)
#     cross = x * y2 - x2 * y
#     A = 0.5 * np.sum(cross)

#     if abs(A) < 1e-12:
#         return 0.0, np.mean(P, axis=0)

#     cx = (1.0 / (6.0 * A)) * np.sum((x + x2) * cross)
#     cy = (1.0 / (6.0 * A)) * np.sum((y + y2) * cross)
#     return float(abs(A)), np.array([cx, cy], dtype=float)

# def center_poly_prism(pts: np.ndarray) -> np.ndarray:
#     """棱柱：XY 用截面凸包面积质心；Z 用上下端面中点。"""
#     c = np.zeros(3, dtype=float)
#     hull = convex_hull_2d(pts[:, 0:2])
#     _, centroid_xy = polygon_area_centroid(hull)
#     c[0:2] = centroid_xy
#     c[2] = 0.5 * (float(pts[:, 2].min()) + float(pts[:, 2].max()))
#     return c

# def estimate_prism_yaw_face0(pts_c: np.ndarray, n: int) -> float:
#     xy = pts_c[:, 0:2]
#     hull = convex_hull_2d(xy)
#     if hull.shape[0] < 3:
#         r = np.linalg.norm(xy, axis=1)
#         idx = int(np.argmax(r))
#         theta_v = math.atan2(float(xy[idx, 1]), float(xy[idx, 0]))
#         return theta_v + math.pi / float(n)

#     r = np.linalg.norm(hull, axis=1)
#     idx = int(np.argmax(r))
#     theta_v = math.atan2(float(hull[idx, 1]), float(hull[idx, 0]))
#     return theta_v + math.pi / float(n)

# def estimate_poly_prism_dims_from_cloud(pts_c: np.ndarray) -> Tuple[float, float]:
#     """返回 (Rv, H)"""
#     mn = pts_c.min(axis=0)
#     mx = pts_c.max(axis=0)
#     H = float(mx[2] - mn[2])
#     r = np.linalg.norm(pts_c[:, 0:2], axis=1)
#     Rv = float(np.quantile(r, 0.995))
#     return Rv, H

# # ============================================================
# # 5) Validation + selection + save
# # ============================================================
# def uniform_subsample_by_z(G: np.ndarray, max_keep: int, z_bin_size: float) -> np.ndarray:
#     if G.shape[0] <= max_keep:
#         return G
#     if z_bin_size <= 1e-12:
#         z_bin_size = 1e-3

#     z = G[:, 2]
#     zb = np.round(z / z_bin_size).astype(np.int64)

#     order = np.argsort(zb, kind="mergesort")
#     Gs = G[order]
#     zbs = zb[order]

#     groups = []
#     start = 0
#     n = Gs.shape[0]
#     while start < n:
#         end = start + 1
#         while end < n and zbs[end] == zbs[start]:
#             end += 1
#         groups.append((start, end))
#         start = end

#     nz = len(groups)
#     if nz <= 0:
#         idx = np.linspace(0, Gs.shape[0] - 1, max_keep, dtype=int)
#         return Gs[idx]

#     per = int(math.ceil(max_keep / nz))
#     chosen = []
#     for (s, e) in groups:
#         m = e - s
#         if m <= per:
#             chosen.extend(range(s, e))
#         else:
#             ii = np.linspace(0, m - 1, per, dtype=int)
#             chosen.extend((s + ii).tolist())

#     chosen = np.array(chosen, dtype=int)
#     if chosen.size > max_keep:
#         chosen = chosen[:max_keep]
#     return Gs[chosen]

# def estimate_width_from_cloud(P_local: np.ndarray, hand_params: dict, delta_close=0.015) -> Optional[float]:
#     fw = float(hand_params["finger_width"])
#     depth = float(hand_params["hand_depth"])
#     h = float(hand_params["hand_height"])
#     palm_t = float(hand_params.get("palm_thickness", fw))
#     Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

#     x = P_local[:, 0]
#     y = P_local[:, 1]
#     z = P_local[:, 2]

#     x_front = palm_t / 2.0
#     x_tip = palm_t / 2.0 + depth
#     mask_region = (x >= x_front) & (x <= x_tip) & (np.abs(z) <= h / 2.0)
#     if not np.any(mask_region):
#         return None

#     y_r = y[mask_region]
#     y_abs = np.abs(y_r)
#     y_max_abs = float(np.max(y_abs))
#     width = 2.0 * y_max_abs + float(delta_close)

#     if width <= 0.0 or width > Wmax:
#         return None
#     return float(width)

# @dataclass
# class ValidateStats:
#     total_candidates: int = 0
#     axis_bad: int = 0
#     width_none: int = 0
#     width_out_of_range: int = 0
#     min_insert_fail: int = 0
#     collision_fail: int = 0
#     fallback_used: int = 0
#     kept: int = 0

# def validate_candidates_on_cloud(
#     pts: np.ndarray,
#     candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
#     hand_params: dict,
#     delta_close: float,
#     max_keep: int,
#     min_insert: float,
#     z_bin_size: float,
#     width_fn: Optional[Callable] = None,
# ) -> Tuple[np.ndarray, ValidateStats]:
#     """
#     核心保障：
#     - 每个候选有 p_ins（深入）和 p_base（不深入）
#     - 先试 p_ins，若碰撞则回退到 p_base
#     - 最终仍需通过 min_insert（可选）和碰撞检测
#     """
#     stats = ValidateStats(total_candidates=len(candidates))
#     grasps = []

#     fw   = float(hand_params["finger_width"])
#     Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

#     for cand in candidates:
#         if not (isinstance(cand, (list, tuple)) and len(cand) == 5):
#             continue
#         p_base, p_ins, approach, binormal, axis = cand

#         approach = unit(approach)
#         binormal = unit(binormal)
#         axis = unit(axis)
#         if np.linalg.norm(axis) < 1e-8:
#             stats.axis_bad += 1
#             continue

#         R = np.column_stack([approach, binormal, axis])

#         def eval_at(p_try: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
#             pos = find_palm_center_direct(p_sample=p_try, approach=approach, hand_params=hand_params)
#             P_local = world_to_local(pts, pos, R)

#             width = None
#             if width_fn is not None:
#                 width = width_fn(p_try, approach, binormal, axis)
#             if width is None:
#                 width = estimate_width_from_cloud(P_local, hand_params, delta_close=delta_close)

#             return width, pos, P_local

#         # 1) 先试深入
#         p_try = p_ins
#         width, pos, P_local = eval_at(p_try)
#         if width is None:
#             stats.width_none += 1
#             continue
#         if not (0.0 < float(width) <= Wmax):
#             stats.width_out_of_range += 1
#             continue

#         # 2) 深入会碰撞则回退
#         if (np.linalg.norm(np.asarray(p_ins) - np.asarray(p_base)) > 1e-12) and collide_finger_palm(P_local, float(width), hand_params, eps_contact=5e-4):
#             stats.fallback_used += 1
#             p_try = p_base
#             width2, pos2, P_local2 = eval_at(p_try)
#             if width2 is None:
#                 stats.width_none += 1
#                 continue
#             if not (0.0 < float(width2) <= Wmax):
#                 stats.width_out_of_range += 1
#                 continue
#             width, pos, P_local = float(width2), pos2, P_local2

#         # 3) 最终约束
#         if not require_some_insertion(P_local, float(width), hand_params, min_insert=min_insert):
#             stats.min_insert_fail += 1
#             continue

#         if collide_finger_palm(P_local, float(width), hand_params, eps_contact=5e-4):
#             stats.collision_fail += 1
#             continue

#         grasp = np.zeros(13, dtype=float)
#         grasp[0:3] = pos
#         grasp[3:6] = axis
#         grasp[6:9] = approach
#         grasp[9:12] = binormal
#         grasp[12] = float(width)
#         grasps.append(grasp)

#     if len(grasps) == 0:
#         return np.zeros((0, 13), dtype=float), stats

#     G = np.vstack(grasps)
#     stats.kept = int(G.shape[0])

#     if max_keep is not None and max_keep > 0 and G.shape[0] > max_keep:
#         G = uniform_subsample_by_z(G, max_keep=max_keep, z_bin_size=z_bin_size)

#     return G, stats

# def summarize_stats(stats: ValidateStats) -> str:
#     d = asdict(stats)
#     # 主因：从失败项里挑最大
#     fail_keys = ["axis_bad", "width_none", "width_out_of_range", "min_insert_fail", "collision_fail"]
#     main_key = max(fail_keys, key=lambda k: d.get(k, 0))
#     return (f"main={main_key}:{d.get(main_key,0)} | "
#             f"total={d['total_candidates']}, kept={d['kept']}, "
#             f"axis_bad={d['axis_bad']}, width_none={d['width_none']}, width_out={d['width_out_of_range']}, "
#             f"min_insert_fail={d['min_insert_fail']}, collision_fail={d['collision_fail']}, fallback_used={d['fallback_used']}")

# # ============================================================
# # 6) Per-folder pipeline
# # ============================================================
# def generate_for_one_folder(
#     folder: str,
#     hand_params: dict,
#     delta_close: float,
#     angle_step_deg: float,
#     height_step: float,
#     max_grasps: int,
#     include_topdown: bool,
#     include_topbottom: bool,
#     topdown_insert_depths: List[float],
#     sphere_dirs: int,
#     sphere_rot: int,
#     min_insert: float,
#     approach_insert_frac: float,
#     verbose_empty: bool,
# ) -> Tuple[bool, int, Dict[str, Any]]:
#     """
#     返回:
#       succ: 是否为有效文件夹（存在meta+ply且能读点云）
#       k: 保存的抓取数
#       report: 用于汇总/输出原因
#     """
#     rep: Dict[str, Any] = {"id": obj_id_from_folder(folder), "folder": folder}

#     meta_path = os.path.join(folder, "meta.json")
#     pcd_path = os.path.join(folder, "nontextured.ply")
#     if (not os.path.exists(meta_path)) or (not os.path.exists(pcd_path)):
#         rep["status"] = "skip_missing_files"
#         rep["missing_meta"] = (not os.path.exists(meta_path))
#         rep["missing_ply"] = (not os.path.exists(pcd_path))
#         return False, 0, rep

#     try:
#         with open(meta_path, "r") as f:
#             meta = json.load(f)
#     except Exception as e:
#         rep["status"] = "bad_meta_json"
#         rep["error"] = str(e)
#         return False, 0, rep

#     ptype = meta.get("type", "")
#     dims = meta.get("dims", [])
#     rep["ptype"] = ptype
#     rep["dims"] = dims

#     try:
#         pcd = o3d.io.read_point_cloud(pcd_path)
#         pts = np.asarray(pcd.points, dtype=float)
#     except Exception as e:
#         rep["status"] = "bad_pointcloud"
#         rep["error"] = str(e)
#         return False, 0, rep

#     if pts.shape[0] < 50:
#         rep["status"] = "too_few_points"
#         rep["num_points"] = int(pts.shape[0])
#         return False, 0, rep

#     # 居中（大多数类型）
#     center = aabb_center(pts)
#     pts_c = pts - center
#     rep["center_used"] = center.tolist()

#     candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
#     try:
#         if ptype == "cylinder":
#             r, H = float(dims[0]), float(dims[1])
#             candidates = gen_candidates_cylinder(
#                 r, H, hand_params, angle_step_deg, height_step,
#                 include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
#             )
#         elif ptype == "ellip_cyl":
#             rx, ry, H = float(dims[0]), float(dims[1]), float(dims[2])
#             candidates = gen_candidates_ellip_cyl(
#                 rx, ry, H, hand_params, angle_step_deg, height_step,
#                 include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
#             )
#         elif ptype == "frustum":
#             rb, rt, H = float(dims[0]), float(dims[1]), float(dims[2])
#             candidates = gen_candidates_frustum(
#                 rb, rt, H, hand_params, angle_step_deg, height_step,
#                 include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
#             )
#         elif ptype == "sphere":
#             r = float(dims[0])
#             candidates = gen_candidates_sphere(
#                 r=r, n_dirs=sphere_dirs, n_rot=sphere_rot,
#                 approach_insert_frac=approach_insert_frac,
#                 hand_params=hand_params
#             )
#         elif ptype == "ellipsoid":
#             rx, ry, rz = float(dims[0]), float(dims[1]), float(dims[2])
#             if abs(rx - ry) < 1e-6 and abs(ry - rz) < 1e-6:
#                 candidates = gen_candidates_sphere(
#                     r=rx, n_dirs=sphere_dirs, n_rot=sphere_rot,
#                     approach_insert_frac=approach_insert_frac,
#                     hand_params=hand_params
#                 )
#             else:
#                 candidates = gen_candidates_ellipsoid(
#                     rx, ry, rz,
#                     angle_step_deg=angle_step_deg,
#                     use_extremes_only=False,
#                     add_binormal_sign=True,
#                     approach_insert_frac=approach_insert_frac,
#                     hand_params=hand_params
#                 )
#         elif ptype == "poly_prism":
#             n = int(round(float(dims[0])))

#             # poly_prism: 用更稳中心
#             center = center_poly_prism(pts)
#             pts_c = pts - center
#             rep["center_used"] = center.tolist()
#             rep["center_mode"] = "poly_prism_area_centroid"

#             Rv_est, H_est = estimate_poly_prism_dims_from_cloud(pts_c)
#             yaw0 = estimate_prism_yaw_face0(pts_c, n)

#             candidates = gen_candidates_poly_prism(
#                 n=n,
#                 Rv=Rv_est,
#                 H=H_est,
#                 hand_params=hand_params,
#                 height_step=height_step,
#                 edge_margin=0.01,
#                 yaw_face0=yaw0,
#                 include_topbottom=include_topbottom,
#                 topbottom_angle_step_deg=15.0,
#                 approach_insert_frac=approach_insert_frac,
#             )
#             dims = [float(n), float(Rv_est), float(H_est)]
#             rep["dims_reestimated"] = dims
#         elif ptype == "box":
#             mn = pts_c.min(axis=0)
#             mx = pts_c.max(axis=0)
#             ext = mx - mn
#             w, h, d = float(ext[0]), float(ext[1]), float(ext[2])
#             dims = [w, h, d]
#             rep["dims_reestimated"] = dims

#             candidates = gen_candidates_box(
#                 w=w, h=h, d=d,
#                 surface_insert=0.015,
#                 add_roll90=True
#             )
#         else:
#             rep["status"] = "unknown_primitive"
#             return False, 0, rep

#     except Exception as e:
#         rep["status"] = "candidate_generation_exception"
#         rep["error"] = str(e)
#         if verbose_empty:
#             print(f"[CAND_EXC] id={rep['id']} ptype={ptype} error={e}")
#         return False, 0, rep

#     rep["num_candidates"] = int(len(candidates))
#     if len(candidates) == 0:
#         rep["status"] = "empty_candidates"
#         if verbose_empty:
#             print(f"[EMPTY_CAND] id={rep['id']} ptype={ptype} dims={rep.get('dims')} reason=generator_returned_empty")
#         # 仍创建空文件（与原逻辑一致）
#         out_txt = os.path.join(folder, "pose_best.txt")
#         out_npy = os.path.join(folder, "pose_best.npy")

#         # 统一创建空 npy（shape=(0,13)），避免下游读不到
#         np.save(out_npy, np.zeros((0, 13), dtype=np.float32))

#         # 可选：保留 txt 兼容旧链路（不想要就删掉这两行）
#         open(out_txt, "w").close()

#         return True, 0, rep

#     width_fn = make_width_fn(ptype, dims, delta_close=delta_close, hand_params=hand_params)
#     grasps_c, stats = validate_candidates_on_cloud(
#         pts=pts_c,
#         candidates=candidates,
#         hand_params=hand_params,
#         delta_close=delta_close,
#         max_keep=max_grasps,
#         min_insert=min_insert,
#         z_bin_size=max(1e-6, float(height_step)),
#         width_fn=width_fn,
#     )
#     rep["validate_stats"] = asdict(stats)

#     out_txt = os.path.join(folder, "pose_best.txt")
#     out_npy = os.path.join(folder, "pose_best.npy")

#     if grasps_c.shape[0] == 0:
#         rep["status"] = "empty_valid_grasps"
#         if verbose_empty:
#             print(f"[EMPTY_VALID] id={rep['id']} ptype={ptype} dims={rep.get('dims')} | {summarize_stats(stats)}")

#         np.save(out_npy, np.zeros((0, 13), dtype=np.float32))
#         # 可选：保留 txt 兼容旧链路
#         open(out_txt, "w").close()

#         return True, 0, rep

#     # 平移回原始点云坐标系
#     grasps = grasps_c.copy()
#     grasps[:, 0:3] += center

#     # 保存 npy（推荐 float32：更快/更省空间；若你要完全不损精度可改成 float64）
#     np.save(out_npy, grasps.astype(np.float32, copy=False))

#     # 可选：保留 txt 兼容旧链路（不需要就注释掉）
#     np.savetxt(out_txt, grasps, fmt="%.6f", delimiter=", ")

#     rep["status"] = "ok"
#     return True, int(grasps.shape[0]), rep


# # ============================================================
# # 7) Main
# # ============================================================
# def main():
#     parser = argparse.ArgumentParser("Generate structured grasps per primitive using meta.json, and save pose_best.txt")
#     parser.add_argument("--base_path", type=str, default='shot_fpfh/descriptors/dataset/database_2k',
#                         help="Dataset root")
#     parser.add_argument("--delta_close", type=float, default=0.025,
#                         help="Total clearance (m). width = 2*radius + delta_close")
#     parser.add_argument("--angle_step_deg", type=float, default=20.0,
#                         help="Angle step for ring sampling on side/topdown families.")
#     parser.add_argument("--height_step", type=float, default=0.01,
#                         help="Height step (m) for side-ring sampling.")
#     parser.add_argument("--max_grasps", type=int, default=1024,
#                         help="Max grasps saved per object (z-binned uniform subsample if exceeded).")
#     parser.add_argument("--include_topdown", action="store_true",
#                         help="Enable top/bottom approaching family.")
#     parser.add_argument("--include_topbottom", action="store_true",
#                         help="Enable top-bottom pinch family (binormal=z).")
#     parser.add_argument("--topdown_depths", type=str, default="0.01,0.02,0.03",
#                         help="Comma-separated insertion depths (m) for topdown family.")
#     parser.add_argument("--sphere_dirs", type=int, default=48,
#                         help="Number of fibonacci directions for sphere/ellipsoid.")
#     parser.add_argument("--sphere_rot", type=int, default=3,
#                         help="Number of binormal rotations per direction for sphere/ellipsoid.")
#     parser.add_argument("--min_insert", type=float, default=0.0,
#                         help="Optional: minimum insertion depth inside finger channel (m). Default 0 disables.")
#     parser.add_argument("--approach_insert_frac", type=float, default=1.0/3.0,
#                         help="Try move p_sample along +approach by (frac * radius) for stability. Will fallback if collides.")
#     parser.add_argument("--verbose_empty", action="store_true",
#                         help="Print per-id reasons for empty candidates/empty valid grasps.")
#     parser.add_argument("--report_path", type=str, default="",
#                         help="If set, save a JSON report to this path (e.g. empty_report.json).")
#     args = parser.parse_args()

#     base_path = args.base_path
#     if not os.path.isdir(base_path):
#         raise FileNotFoundError(base_path)

#     topdown_depths: List[float] = []
#     for s in args.topdown_depths.split(","):
#         s = s.strip()
#         if not s:
#             continue
#         try:
#             topdown_depths.append(float(s))
#         except Exception:
#             pass
#     if len(topdown_depths) == 0:
#         topdown_depths = [0.01]

#     folders = []
#     for name in os.listdir(base_path):
#         p = os.path.join(base_path, name)
#         if not os.path.isdir(p):
#             continue
#         if os.path.exists(os.path.join(p, "nontextured.ply")) and os.path.exists(os.path.join(p, "meta.json")):
#             folders.append(p)
#     folders.sort()

#     total = 0
#     ok = 0
#     kept = 0

#     reports: List[Dict[str, Any]] = []
#     empty_cand_ids = []
#     empty_valid_ids = []

#     for folder in tqdm(folders, desc="Generating grasps"):
#         total += 1
#         succ, k, rep = generate_for_one_folder(
#             folder=folder,
#             hand_params=HAND_PARAMS_REAL,
#             delta_close=args.delta_close,
#             angle_step_deg=args.angle_step_deg,
#             height_step=args.height_step,
#             max_grasps=args.max_grasps,
#             include_topdown=args.include_topdown,
#             include_topbottom=args.include_topbottom,
#             topdown_insert_depths=topdown_depths,
#             sphere_dirs=args.sphere_dirs,
#             sphere_rot=args.sphere_rot,
#             min_insert=args.min_insert,
#             approach_insert_frac=args.approach_insert_frac,
#             verbose_empty=args.verbose_empty,
#         )
#         if succ:
#             ok += 1
#             kept += k

#         reports.append(rep)
#         if rep.get("status") == "empty_candidates":
#             empty_cand_ids.append(rep.get("id"))
#         if rep.get("status") == "empty_valid_grasps":
#             empty_valid_ids.append(rep.get("id"))

#     print(f"\n[Done] processed={total}, valid_folders={ok}, total_saved_grasps={kept}")
#     print(f"[Empty] empty_candidates={len(empty_cand_ids)}, empty_valid_grasps={len(empty_valid_ids)}")

#     if len(empty_cand_ids) > 0:
#         print("  - empty_candidates ids (first 50):", empty_cand_ids[:50])
#     if len(empty_valid_ids) > 0:
#         print("  - empty_valid_grasps ids (first 50):", empty_valid_ids[:50])

#     if args.report_path:
#         try:
#             with open(args.report_path, "w") as f:
#                 json.dump(reports, f, ensure_ascii=False, indent=2)
#             print(f"[Report] saved to: {args.report_path}")
#         except Exception as e:
#             print(f"[Report] failed to save: {args.report_path} | error={e}")

# if __name__ == "__main__":
#     os.environ["OMP_NUM_THREADS"] = "1"
#     os.environ["MKL_NUM_THREADS"] = "1"
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计版本：不再生成 pose_best.txt / pose_best.npy
只统计每个物体最终有效 grasp 数量，并输出：
- 平均 grasp 数
- 中位数 / 标准差 / 最小 / 最大 / 分位数
- 精确数量分布
- 分箱分布
- 按 primitive type 的统计
- JSON 报告
"""

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Callable, Dict, Any
from collections import Counter, defaultdict

import numpy as np
import open3d as o3d
from tqdm import tqdm

# ---------------- Hand parameters ----------------
HAND_PARAMS_REAL = {
    "finger_width":        0.015,
    "hand_outer_diameter": 0.167,
    "hand_depth":          0.0475,
    "hand_height":         0.02,
    "palm_thickness":      0.004,
}

# ============================================================
# 0) Basic utils
# ============================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def unit(v: np.ndarray) -> np.ndarray:
    return normalize(v)

def orthonormal_basis_from(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors (e1,e2) that span the plane orthogonal to v."""
    v = unit(v)
    if abs(v[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    e1 = np.cross(v, ref)
    e1 = unit(e1)
    e2 = np.cross(v, e1)
    e2 = unit(e2)
    return e1, e2

def fibonacci_sphere(n: int) -> np.ndarray:
    """Nearly-uniform directions on a sphere. Returns (n,3). Deterministic."""
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
    dirs = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = ga * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        dirs.append([x, y, z])
    return np.asarray(dirs, dtype=float)

def aabb_center(pts: np.ndarray) -> np.ndarray:
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return 0.5 * (mn + mx)

def safe_insert_distance(r_eff: float, hand_params: dict, frac: float, safety: float = 0.002) -> float:
    """
    计算“沿 approach 深入”的插入量，并做 hand_depth 上限保护（保守限幅）。
    - r_eff: 有效半径（圆柱/球= r；椭圆柱=短轴半径；圆台=该 z 截面半径；椭球=最短半轴）
    - safety: 预留裕量，避免掌心贴太近（m）
    """
    r_eff = float(r_eff)
    frac = float(frac)
    if frac <= 0.0 or r_eff <= 1e-12:
        return 0.0
    depth = float(hand_params["hand_depth"])
    desired = frac * r_eff
    insert_max = max(0.0, depth - r_eff - float(safety))
    return float(min(desired, insert_max))

def obj_id_from_folder(folder: str) -> str:
    return os.path.basename(os.path.normpath(folder))

# ============================================================
# 1) Core geometry helpers
# ============================================================
def world_to_local(P: np.ndarray, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """R: 3x3 with columns (approach, binormal, axis) in world."""
    return (R.T @ (P - pos).T).T

def make_frame(approach: np.ndarray, binormal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (approach, binormal, axis) as an orthonormal right-handed frame."""
    a = unit(approach)
    b = unit(binormal)
    ax = np.cross(a, b)
    if np.linalg.norm(ax) < 1e-8:
        return a, b, np.array([0.0, 0.0, 0.0], dtype=float)
    ax = unit(ax)
    b = unit(b - np.dot(b, a) * a)
    ax = unit(np.cross(a, b))
    return a, b, ax

def find_palm_center_direct(
    p_sample: np.ndarray,
    approach: np.ndarray,
    hand_params: dict,
) -> np.ndarray:
    """
    解析式：已知 p_sample 为“指尖中心/夹爪闭合中线在指尖处的参考点”，
    手掌中心 = p_sample 沿 -approach 平移 x_tip，其中 x_tip = palm_t/2 + depth
    """
    approach = unit(approach)
    fw      = float(hand_params["finger_width"])
    depth   = float(hand_params["hand_depth"])
    palm_t  = float(hand_params.get("palm_thickness", fw))
    x_tip   = palm_t / 2.0 + depth
    return p_sample - x_tip * approach

def collide_finger_palm(P_local: np.ndarray, width: float, hand_params: dict, eps_contact: float = 5e-4) -> bool:
    """
    基础碰撞检测（闭合到 width 后）：
    - 允许点落在指内侧表面附近（|y|-width/2 <= eps_contact）
    - 禁止点进入指体厚度 fw 或掌心体积
    """
    fw      = float(hand_params["finger_width"])
    depth   = float(hand_params["hand_depth"])
    h       = float(hand_params["hand_height"])
    palm_t  = float(hand_params.get("palm_thickness", fw))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_front = palm_t / 2.0
    x_tip   = palm_t / 2.0 + depth
    yL      = width / 2.0

    mask_finger = (
        (x >= x_front) & (x <= x_tip) &
        (np.abs(z) <= h / 2.0) &
        (
            ((y >= ( yL + eps_contact)) & (y <= ( yL + fw))) |
            ((y <= (-yL - eps_contact)) & (y >= (-yL - fw)))
        )
    )

    mask_palm = (
        (x >= -palm_t / 2.0) & (x <= palm_t / 2.0) &
        (np.abs(z) <= h / 2.0) &
        (np.abs(y) <= (yL + fw))
    )

    return bool(np.any(mask_finger | mask_palm))

def require_some_insertion(P_local: np.ndarray, width: float, hand_params: dict, min_insert: float) -> bool:
    """
    可选：要求物体在手指通道内至少进入 min_insert 深度。
    min_insert=0 时等价于不启用该约束。
    """
    if min_insert <= 0.0:
        return True

    fw      = float(hand_params["finger_width"])
    depth   = float(hand_params["hand_depth"])
    h       = float(hand_params["hand_height"])
    palm_t  = float(hand_params.get("palm_thickness", fw))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_front = palm_t / 2.0
    x_tip   = palm_t / 2.0 + depth
    yL      = width / 2.0

    mask_channel = (
        (x >= x_front) & (x <= x_tip) &
        (np.abs(z) <= h / 2.0) &
        (np.abs(y) <= yL)
    )
    if not np.any(mask_channel):
        return False

    x_cloud_front = float(np.max(x[mask_channel]))
    return (x_cloud_front - x_front) >= float(min_insert)

# ============================================================
# 2) Primitive helper math (analytic widths)
# ============================================================
def frustum_radius_at_z(r_bottom: float, r_top: float, H: float, z: float) -> float:
    """z in [-H/2, +H/2], bottom at -H/2."""
    if H < 1e-9:
        return float(min(r_bottom, r_top))
    t = (z + H / 2.0) / H
    return float(r_bottom + t * (r_top - r_bottom))

def regular_ngon_ray_max_t(n: int, Rv: float, theta: float) -> float:
    """
    Regular n-gon centered at origin, vertex radius Rv.
    Max t s.t. t*[cosθ,sinθ] inside polygon (apothem half-space).
    """
    n = int(n)
    if n < 3:
        return 0.0
    apothem = Rv * math.cos(math.pi / n)
    u = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    best_t = float("inf")
    for i in range(n):
        phi = 2.0 * math.pi * (i + 0.5) / n
        ni = np.array([math.cos(phi), math.sin(phi)], dtype=float)
        denom = float(np.dot(ni, u))
        if denom <= 1e-9:
            continue
        t = apothem / denom
        if t < best_t:
            best_t = t
    if not np.isfinite(best_t):
        best_t = Rv
    return float(best_t)

def ellipse_radius_along_dir(rx: float, ry: float, ux: float, uy: float) -> float:
    """Ellipse (x/rx)^2+(y/ry)^2=1, return boundary radius along direction u."""
    denom = math.sqrt((ux / rx) ** 2 + (uy / ry) ** 2) + 1e-12
    return 1.0 / denom

def ellipsoid_radius_along_dir(rx: float, ry: float, rz: float, u: np.ndarray) -> float:
    """Ellipsoid (x/rx)^2+(y/ry)^2+(z/rz)^2=1, boundary radius along direction u."""
    u = unit(u)
    denom = math.sqrt((u[0] / rx) ** 2 + (u[1] / ry) ** 2 + (u[2] / rz) ** 2) + 1e-12
    return 1.0 / denom

def make_width_fn(ptype: str, dims: list, delta_close: float, hand_params: dict) -> Optional[Callable]:
    """
    对规则体优先使用解析 width（避免点云稀疏导致 width 偏小）。
    兼容 top-bottom pinch（binormal≈z）时，width 用高度 H。
    """
    fw   = float(hand_params["finger_width"])
    Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

    def _ok(w: float) -> Optional[float]:
        w = float(w)
        return w if (w > 0.0 and w <= Wmax) else None

    if ptype == "cylinder":
        r, H = float(dims[0]), float(dims[1])
        def _w(p_sample, approach, binormal, axis):
            b = np.abs(np.asarray(binormal, dtype=float))
            if b[2] >= max(b[0], b[1]):  # pinch top-bottom
                return _ok(H + float(delta_close))
            return _ok(2.0 * r + float(delta_close))
        return _w

    if ptype == "ellip_cyl":
        rx, ry, H = float(dims[0]), float(dims[1]), float(dims[2])
        def _w(p_sample, approach, binormal, axis):
            b = np.abs(np.asarray(binormal, dtype=float))
            if b[2] >= max(b[0], b[1]):  # pinch top-bottom
                return _ok(H + float(delta_close))
            bxy = np.array([binormal[0], binormal[1]], dtype=float)
            nb = np.linalg.norm(bxy)
            if nb < 1e-9:
                return None
            bxy /= nb
            rad = ellipse_radius_along_dir(rx, ry, float(bxy[0]), float(bxy[1]))
            return _ok(2.0 * rad + float(delta_close))
        return _w

    if ptype == "frustum":
        rb, rt, H = float(dims[0]), float(dims[1]), float(dims[2])
        def _w(p_sample, approach, binormal, axis):
            b = np.abs(np.asarray(binormal, dtype=float))
            if b[2] >= max(b[0], b[1]):  # pinch top-bottom
                return _ok(H + float(delta_close))
            z = float(p_sample[2])
            rad = frustum_radius_at_z(rb, rt, H, z)
            return _ok(2.0 * rad + float(delta_close))
        return _w

    if ptype == "sphere":
        r = float(dims[0])
        def _w(p_sample, approach, binormal, axis):
            return _ok(2.0 * r + float(delta_close))
        return _w

    if ptype == "ellipsoid":
        rx, ry, rz = float(dims[0]), float(dims[1]), float(dims[2])
        def _w(p_sample, approach, binormal, axis):
            rad = ellipsoid_radius_along_dir(rx, ry, rz, binormal)
            return _ok(2.0 * rad + float(delta_close))
        return _w

    if ptype == "poly_prism":
        n, Rv, H = int(round(float(dims[0]))), float(dims[1]), float(dims[2])
        def _w(p_sample, approach, binormal, axis):
            b = np.asarray(binormal, dtype=float)
            if abs(float(b[2])) >= max(abs(float(b[0])), abs(float(b[1]))):
                return _ok(float(H) + float(delta_close))
            th = math.atan2(float(b[1]), float(b[0]))
            r_plus  = regular_ngon_ray_max_t(n, Rv, th)
            r_minus = regular_ngon_ray_max_t(n, Rv, th + math.pi)
            rad = max(r_plus, r_minus)
            return _ok(2.0 * rad + float(delta_close))
        return _w

    if ptype == "box":
        w, h, d = float(dims[0]), float(dims[1]), float(dims[2])
        def _w(p_sample, approach, binormal, axis):
            v = np.abs(np.asarray(binormal, dtype=float))
            k = int(np.argmax(v))
            size = [w, h, d][k]
            return _ok(float(size) + float(delta_close))
        return _w

    return None

# ============================================================
# 3) Candidate generators per primitive
# ============================================================
def z_levels_for_side_grasp(H: float, hand_h: float, step: float, margin: float = 0.002) -> np.ndarray:
    if H <= 0:
        return np.array([0.0], dtype=float)

    step = float(step)
    if step <= 0:
        return np.array([0.0], dtype=float)

    zmin = -H / 2.0 + hand_h / 2.0 + margin
    zmax =  H / 2.0 - hand_h / 2.0 - margin
    if zmin > zmax:
        return np.array([0.0], dtype=float)

    eps = 1e-12
    k_min = int(math.ceil((zmin - eps) / step))
    k_max = int(math.floor((zmax + eps) / step))

    if k_min > k_max:
        return np.array([0.0], dtype=float)

    ks = np.arange(k_min, k_max + 1, dtype=np.int64)
    zs = ks.astype(float) * step
    zs = zs[(zs >= zmin - 1e-9) & (zs <= zmax + 1e-9)]

    if zs.size == 0:
        return np.array([0.0], dtype=float)

    return zs

def gen_side_family_angles(angle_step_deg: float) -> np.ndarray:
    step = max(1e-6, float(angle_step_deg))
    degs = np.arange(0.0, 360.0, step, dtype=float)
    return np.deg2rad(degs)

def gen_candidates_cylinder(r: float, H: float, hand_params: dict,
                            angle_step_deg: float, height_step: float,
                            include_topdown: bool, include_topbottom: bool,
                            topdown_insert_depths: List[float],
                            approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []
    r = float(r); H = float(H)

    hand_h = float(hand_params["hand_height"])
    thetas = gen_side_family_angles(angle_step_deg)
    zs = z_levels_for_side_grasp(H, hand_h, height_step)

    for z in zs:
        for th in thetas:
            u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
            approach = -u
            binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)
            approach, binormal, axis = make_frame(approach, binormal)
            if np.linalg.norm(axis) < 1e-8:
                continue

            p_base = np.array([0.0, 0.0, float(z)], dtype=float)
            ins = safe_insert_distance(r_eff=r, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
            p_ins = p_base + ins * approach
            cands.append((p_base, p_ins, approach, binormal, axis))

    if include_topdown:
        for side in (+1.0, -1.0):
            face_z = side * (H / 2.0)
            for d in topdown_insert_depths:
                z_s = face_z - side * float(d)
                for th in thetas:
                    approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
                    binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
                    approach, binormal, axis = make_frame(approach, binormal)
                    if np.linalg.norm(axis) < 1e-8:
                        continue
                    p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
                    cands.append((p_base, p_base, approach, binormal, axis))

    if include_topbottom:
        fw = float(hand_params["finger_width"])
        Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
        if H < Wmax + 0.02:
            thetas2 = gen_side_family_angles(15.0)
            for th in thetas2:
                u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
                approach = -u
                binormal = np.array([0.0, 0.0, 1.0], dtype=float)
                approach, binormal, axis = make_frame(approach, binormal)
                if np.linalg.norm(axis) < 1e-8:
                    continue
                p_base = np.array([0.0, 0.0, 0.0], dtype=float)
                cands.append((p_base, p_base, approach, binormal, axis))

    return cands

def gen_candidates_ellip_cyl(rx: float, ry: float, H: float, hand_params: dict,
                            angle_step_deg: float, height_step: float,
                            include_topdown: bool, include_topbottom: bool,
                            topdown_insert_depths: List[float],
                            approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []
    rx = float(rx); ry = float(ry); H = float(H)

    hand_h = float(hand_params["hand_height"])
    zs = z_levels_for_side_grasp(H, hand_h, height_step)

    circle_tol = 1e-6
    if abs(rx - ry) <= circle_tol:
        thetas_side = gen_side_family_angles(angle_step_deg)
        thetas_top  = gen_side_family_angles(angle_step_deg)
    else:
        thetas_side = np.array([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], dtype=float)
        thetas_top  = thetas_side.copy()

    for z in zs:
        for th in thetas_side:
            u2 = np.array([math.cos(th), math.sin(th)], dtype=float)
            approach = -np.array([u2[0], u2[1], 0.0], dtype=float)
            binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)

            approach, binormal, axis = make_frame(approach, binormal)
            if np.linalg.norm(axis) < 1e-8:
                continue

            p_base = np.array([0.0, 0.0, float(z)], dtype=float)
            r_ins = min(rx, ry)
            ins = safe_insert_distance(r_eff=r_ins, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
            p_ins = p_base + ins * approach
            cands.append((p_base, p_ins, approach, binormal, axis))

    if include_topdown:
        for side in (+1.0, -1.0):
            face_z = side * (H / 2.0)
            for d_ins in topdown_insert_depths:
                z_s = face_z - side * float(d_ins)
                for th in thetas_top:
                    approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
                    binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)

                    approach, binormal, axis = make_frame(approach, binormal)
                    if np.linalg.norm(axis) < 1e-8:
                        continue

                    p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
                    cands.append((p_base, p_base, approach, binormal, axis))

    if include_topbottom:
        fw = float(hand_params["finger_width"])
        Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
        if H < Wmax + 0.02:
            thetas2 = gen_side_family_angles(15.0)
            for th in thetas2:
                u2 = np.array([math.cos(th), math.sin(th)], dtype=float)
                approach = -np.array([u2[0], u2[1], 0.0], dtype=float)
                binormal = np.array([0.0, 0.0, 1.0], dtype=float)
                approach, binormal, axis = make_frame(approach, binormal)
                if np.linalg.norm(axis) < 1e-8:
                    continue
                p_base = np.array([0.0, 0.0, 0.0], dtype=float)
                cands.append((p_base, p_base, approach, binormal, axis))

    return cands

def gen_candidates_frustum(r_bottom: float, r_top: float, H: float, hand_params: dict,
                           angle_step_deg: float, height_step: float,
                           include_topdown: bool, include_topbottom: bool,
                           topdown_insert_depths: List[float],
                           approach_insert_frac: float) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []
    rb = float(r_bottom); rt = float(r_top); H = float(H)

    hand_h = float(hand_params["hand_height"])
    thetas = gen_side_family_angles(angle_step_deg)
    zs = z_levels_for_side_grasp(H, hand_h, height_step)

    for z in zs:
        for th in thetas:
            u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
            approach = -u
            binormal = np.array([math.cos(th + math.pi / 2.0), math.sin(th + math.pi / 2.0), 0.0], dtype=float)
            approach, binormal, axis = make_frame(approach, binormal)
            if np.linalg.norm(axis) < 1e-8:
                continue

            p_base = np.array([0.0, 0.0, float(z)], dtype=float)
            rad = frustum_radius_at_z(rb, rt, H, float(z))
            ins = safe_insert_distance(r_eff=rad, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)
            p_ins = p_base + ins * approach
            cands.append((p_base, p_ins, approach, binormal, axis))

    if include_topdown:
        for side in (+1.0, -1.0):
            face_z = side * (H / 2.0)
            for d in topdown_insert_depths:
                z_s = face_z - side * float(d)
                for th in thetas:
                    approach = -side * np.array([0.0, 0.0, 1.0], dtype=float)
                    binormal = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
                    approach, binormal, axis = make_frame(approach, binormal)
                    if np.linalg.norm(axis) < 1e-8:
                        continue
                    p_base = np.array([0.0, 0.0, float(z_s)], dtype=float)
                    cands.append((p_base, p_base, approach, binormal, axis))

    if include_topbottom:
        fw = float(hand_params["finger_width"])
        Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
        if H < Wmax + 0.02:
            thetas2 = gen_side_family_angles(15.0)
            for th in thetas2:
                u = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
                approach = -u
                binormal = np.array([0.0, 0.0, 1.0], dtype=float)
                approach, binormal, axis = make_frame(approach, binormal)
                if np.linalg.norm(axis) < 1e-8:
                    continue
                p_base = np.array([0.0, 0.0, 0.0], dtype=float)
                cands.append((p_base, p_base, approach, binormal, axis))

    return cands

def gen_candidates_sphere(r: float, n_dirs: int, n_rot: int,
                         approach_insert_frac: float, hand_params: dict) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []
    dirs = fibonacci_sphere(int(n_dirs))
    center = np.zeros(3, dtype=float)
    r = float(r)

    ins = safe_insert_distance(r_eff=r, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

    for d in dirs:
        d = unit(d)
        approach = -d
        e1, e2 = orthonormal_basis_from(approach)
        for k in range(int(n_rot)):
            psi = 2.0 * math.pi * k / max(1, int(n_rot))
            binormal = math.cos(psi) * e1 + math.sin(psi) * e2
            a, b, c = make_frame(approach, binormal)
            if np.linalg.norm(c) < 1e-8:
                continue
            p_base = center
            p_ins = p_base + ins * a
            cands.append((p_base, p_ins, a, b, c))
    return cands

def gen_candidates_ellipsoid(rx: float, ry: float, rz: float,
                            angle_step_deg: float = 20.0,
                            use_extremes_only: bool = False,
                            add_binormal_sign: bool = True,
                            approach_insert_frac: float = 1/3,
                            hand_params: Optional[dict] = None
                            ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if hand_params is None:
        hand_params = HAND_PARAMS_REAL

    cands = []
    center = np.zeros(3, dtype=float)
    rx = float(rx); ry = float(ry); rz = float(rz)

    axes = [
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    ]
    radii = [rx, ry, rz]

    if use_extremes_only:
        i_max = int(np.argmax(radii))
        i_min = int(np.argmin(radii))
        idxs = [i_max] if i_max == i_min else [i_max, i_min]
        base_binormals = [axes[i] for i in idxs]
    else:
        base_binormals = axes

    step = max(1e-6, float(angle_step_deg))
    psis = np.deg2rad(np.arange(0.0, 360.0, step, dtype=float))

    r_ins = min(rx, ry, rz)
    ins = safe_insert_distance(r_eff=r_ins, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

    signs = (+1.0, -1.0) if add_binormal_sign else (+1.0,)
    for b0 in base_binormals:
        for s in signs:
            binormal = float(s) * b0
            e1, e2 = orthonormal_basis_from(binormal)
            for psi in psis:
                approach = math.cos(float(psi)) * e1 + math.sin(float(psi)) * e2
                a, b, c = make_frame(approach, binormal)
                if np.linalg.norm(c) < 1e-8:
                    continue
                p_base = center
                p_ins = p_base + ins * a
                cands.append((p_base, p_ins, a, b, c))

    return cands

def gen_candidates_poly_prism(
    n: int,
    Rv: float,
    H: float,
    hand_params: dict,
    height_step: float = 0.01,
    edge_margin: float = 0.01,
    yaw_face0: float = 0.0,
    include_topbottom: bool = True,
    topbottom_angle_step_deg: float = 15.0,
    approach_insert_frac: float = 1/3,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []
    n = int(n)
    Rv = float(Rv); H = float(H)

    hand_h = float(hand_params["hand_height"])
    zs = z_levels_for_side_grasp(H, hand_h, step=float(height_step), margin=float(edge_margin))
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    ins = safe_insert_distance(r_eff=Rv, hand_params=hand_params, frac=approach_insert_frac, safety=0.002)

    for z in zs:
        p_base0 = np.array([0.0, 0.0, float(z)], dtype=float)
        for i in range(n):
            phi = float(yaw_face0) + 2.0 * math.pi * float(i) / float(n)
            binormal = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=float)

            approach = np.cross(binormal, z_axis)
            a, b, c = make_frame(approach, binormal)
            if np.linalg.norm(c) < 1e-8:
                continue

            if np.dot(c, z_axis) < 0.0:
                b = -b
                c = -c

            p_base = p_base0.copy()
            p_ins = p_base + ins * a
            cands.append((p_base, p_ins, a, b, c))

    if include_topbottom:
        fw = float(hand_params["finger_width"])
        Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)
        if H < Wmax + 0.02:
            thetas = gen_side_family_angles(topbottom_angle_step_deg)
            for th in thetas:
                approach = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
                binormal = z_axis.copy()
                a, b, c = make_frame(approach, binormal)
                if np.linalg.norm(c) < 1e-8:
                    continue
                p_base = np.array([0.0, 0.0, 0.0], dtype=float)
                cands.append((p_base, p_base, a, b, c))

    return cands

def axis_slide_offsets(half_len: float, edge_margin: float, step: float) -> List[float]:
    half_len = float(half_len)
    edge_margin = float(edge_margin)
    step = float(step)

    lim = half_len - edge_margin
    if lim <= 1e-9 or step <= 1e-12:
        return []

    eps = 1e-12
    kmax = int(math.floor((lim + eps) / step))
    if kmax <= 0:
        return []

    offs = []
    for k in range(1, kmax + 1):
        offs.append(-k * step)
        offs.append(+k * step)
    return offs

def gen_candidates_box(w: float, h: float, d: float,
                       surface_insert: float = 0.01,
                       add_roll90: bool = True,
                       enable_axis_slide: bool = True,
                       axis_step: float = 0.01,
                       edge_margin: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    cands = []

    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=float),
        "y": np.array([0.0, 1.0, 0.0], dtype=float),
        "z": np.array([0.0, 0.0, 1.0], dtype=float),
    }
    size = {"x": float(w), "y": float(h), "z": float(d)}
    half = {"x": float(w) / 2.0, "y": float(h) / 2.0, "z": float(d) / 2.0}
    keys = ["x", "y", "z"]

    def append_with_axis_slide(p_sample: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        p_base = p_sample
        cands.append((p_base, p_base, a, b, c))
        if not enable_axis_slide:
            return

        kk = int(np.argmax(np.abs(c)))
        kkey = keys[kk]

        if abs(float(p_sample[kk])) > 1e-9:
            return

        offs = axis_slide_offsets(half[kkey], edge_margin=edge_margin, step=axis_step)
        for off in offs:
            p2 = p_sample + float(off) * c
            cands.append((p2, p2, a, b, c))

    for akey in ["x", "y", "z"]:
        for side in (+1.0, -1.0):
            outward = side * axes[akey]
            approach0 = -outward
            p_face = outward * half[akey]

            ins = min(float(surface_insert), float(size[akey]))
            p_sample = p_face + ins * approach0

            tkeys = [k for k in ["x", "y", "z"] if k != akey]
            t1 = tkeys[0]

            binormal0 = axes[t1]
            a, b, c = make_frame(approach0, binormal0)
            if np.linalg.norm(c) < 1e-8:
                continue
            append_with_axis_slide(p_sample, a, b, c)

            if add_roll90:
                b2 = c
                a2, b2, c2 = make_frame(a, b2)
                if np.linalg.norm(c2) < 1e-8:
                    continue
                append_with_axis_slide(p_sample, a2, b2, c2)

    return cands

# ============================================================
# 4) Poly prism centering helpers
# ============================================================
def _cross2d(o, a, b) -> float:
    return float((a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]))

def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.shape[0] < 3:
        return pts

    pts = np.unique(pts, axis=0)
    if pts.shape[0] < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    return hull

def polygon_area_centroid(poly: np.ndarray) -> Tuple[float, np.ndarray]:
    P = np.asarray(poly, dtype=float)
    if P.shape[0] < 3:
        return 0.0, np.mean(P, axis=0) if P.shape[0] > 0 else np.zeros(2)

    x = P[:, 0]; y = P[:, 1]
    x2 = np.roll(x, -1); y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    A = 0.5 * np.sum(cross)

    if abs(A) < 1e-12:
        return 0.0, np.mean(P, axis=0)

    cx = (1.0 / (6.0 * A)) * np.sum((x + x2) * cross)
    cy = (1.0 / (6.0 * A)) * np.sum((y + y2) * cross)
    return float(abs(A)), np.array([cx, cy], dtype=float)

def center_poly_prism(pts: np.ndarray) -> np.ndarray:
    c = np.zeros(3, dtype=float)
    hull = convex_hull_2d(pts[:, 0:2])
    _, centroid_xy = polygon_area_centroid(hull)
    c[0:2] = centroid_xy
    c[2] = 0.5 * (float(pts[:, 2].min()) + float(pts[:, 2].max()))
    return c

def estimate_prism_yaw_face0(pts_c: np.ndarray, n: int) -> float:
    xy = pts_c[:, 0:2]
    hull = convex_hull_2d(xy)
    if hull.shape[0] < 3:
        r = np.linalg.norm(xy, axis=1)
        idx = int(np.argmax(r))
        theta_v = math.atan2(float(xy[idx, 1]), float(xy[idx, 0]))
        return theta_v + math.pi / float(n)

    r = np.linalg.norm(hull, axis=1)
    idx = int(np.argmax(r))
    theta_v = math.atan2(float(hull[idx, 1]), float(hull[idx, 0]))
    return theta_v + math.pi / float(n)

def estimate_poly_prism_dims_from_cloud(pts_c: np.ndarray) -> Tuple[float, float]:
    mn = pts_c.min(axis=0)
    mx = pts_c.max(axis=0)
    H = float(mx[2] - mn[2])
    r = np.linalg.norm(pts_c[:, 0:2], axis=1)
    Rv = float(np.quantile(r, 0.995))
    return Rv, H

# ============================================================
# 5) Validation + selection + stats
# ============================================================
def uniform_subsample_by_z(G: np.ndarray, max_keep: int, z_bin_size: float) -> np.ndarray:
    if G.shape[0] <= max_keep:
        return G
    if z_bin_size <= 1e-12:
        z_bin_size = 1e-3

    z = G[:, 2]
    zb = np.round(z / z_bin_size).astype(np.int64)

    order = np.argsort(zb, kind="mergesort")
    Gs = G[order]
    zbs = zb[order]

    groups = []
    start = 0
    n = Gs.shape[0]
    while start < n:
        end = start + 1
        while end < n and zbs[end] == zbs[start]:
            end += 1
        groups.append((start, end))
        start = end

    nz = len(groups)
    if nz <= 0:
        idx = np.linspace(0, Gs.shape[0] - 1, max_keep, dtype=int)
        return Gs[idx]

    per = int(math.ceil(max_keep / nz))
    chosen = []
    for (s, e) in groups:
        m = e - s
        if m <= per:
            chosen.extend(range(s, e))
        else:
            ii = np.linspace(0, m - 1, per, dtype=int)
            chosen.extend((s + ii).tolist())

    chosen = np.array(chosen, dtype=int)
    if chosen.size > max_keep:
        chosen = chosen[:max_keep]
    return Gs[chosen]

def estimate_width_from_cloud(P_local: np.ndarray, hand_params: dict, delta_close=0.015) -> Optional[float]:
    fw = float(hand_params["finger_width"])
    depth = float(hand_params["hand_depth"])
    h = float(hand_params["hand_height"])
    palm_t = float(hand_params.get("palm_thickness", fw))
    Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_front = palm_t / 2.0
    x_tip = palm_t / 2.0 + depth
    mask_region = (x >= x_front) & (x <= x_tip) & (np.abs(z) <= h / 2.0)
    if not np.any(mask_region):
        return None

    y_r = y[mask_region]
    y_abs = np.abs(y_r)
    y_max_abs = float(np.max(y_abs))
    width = 2.0 * y_max_abs + float(delta_close)

    if width <= 0.0 or width > Wmax:
        return None
    return float(width)

@dataclass
class ValidateStats:
    total_candidates: int = 0
    axis_bad: int = 0
    width_none: int = 0
    width_out_of_range: int = 0
    min_insert_fail: int = 0
    collision_fail: int = 0
    fallback_used: int = 0
    kept: int = 0

def validate_candidates_on_cloud(
    pts: np.ndarray,
    candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hand_params: dict,
    delta_close: float,
    max_keep: int,
    min_insert: float,
    z_bin_size: float,
    width_fn: Optional[Callable] = None,
) -> Tuple[np.ndarray, ValidateStats]:
    stats = ValidateStats(total_candidates=len(candidates))
    grasps = []

    fw   = float(hand_params["finger_width"])
    Wmax = float(hand_params["hand_outer_diameter"] - 2.0 * fw)

    for cand in candidates:
        if not (isinstance(cand, (list, tuple)) and len(cand) == 5):
            continue
        p_base, p_ins, approach, binormal, axis = cand

        approach = unit(approach)
        binormal = unit(binormal)
        axis = unit(axis)
        if np.linalg.norm(axis) < 1e-8:
            stats.axis_bad += 1
            continue

        R = np.column_stack([approach, binormal, axis])

        def eval_at(p_try: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
            pos = find_palm_center_direct(p_sample=p_try, approach=approach, hand_params=hand_params)
            P_local = world_to_local(pts, pos, R)

            width = None
            if width_fn is not None:
                width = width_fn(p_try, approach, binormal, axis)
            if width is None:
                width = estimate_width_from_cloud(P_local, hand_params, delta_close=delta_close)

            return width, pos, P_local

        p_try = p_ins
        width, pos, P_local = eval_at(p_try)
        if width is None:
            stats.width_none += 1
            continue
        if not (0.0 < float(width) <= Wmax):
            stats.width_out_of_range += 1
            continue

        if (np.linalg.norm(np.asarray(p_ins) - np.asarray(p_base)) > 1e-12) and collide_finger_palm(P_local, float(width), hand_params, eps_contact=5e-4):
            stats.fallback_used += 1
            p_try = p_base
            width2, pos2, P_local2 = eval_at(p_try)
            if width2 is None:
                stats.width_none += 1
                continue
            if not (0.0 < float(width2) <= Wmax):
                stats.width_out_of_range += 1
                continue
            width, pos, P_local = float(width2), pos2, P_local2

        if not require_some_insertion(P_local, float(width), hand_params, min_insert=min_insert):
            stats.min_insert_fail += 1
            continue

        if collide_finger_palm(P_local, float(width), hand_params, eps_contact=5e-4):
            stats.collision_fail += 1
            continue

        grasp = np.zeros(13, dtype=float)
        grasp[0:3] = pos
        grasp[3:6] = axis
        grasp[6:9] = approach
        grasp[9:12] = binormal
        grasp[12] = float(width)
        grasps.append(grasp)

    if len(grasps) == 0:
        return np.zeros((0, 13), dtype=float), stats

    G = np.vstack(grasps)
    stats.kept = int(G.shape[0])

    if max_keep is not None and max_keep > 0 and G.shape[0] > max_keep:
        G = uniform_subsample_by_z(G, max_keep=max_keep, z_bin_size=z_bin_size)

    return G, stats

def summarize_stats(stats: ValidateStats) -> str:
    d = asdict(stats)
    fail_keys = ["axis_bad", "width_none", "width_out_of_range", "min_insert_fail", "collision_fail"]
    main_key = max(fail_keys, key=lambda k: d.get(k, 0))
    return (f"main={main_key}:{d.get(main_key,0)} | "
            f"total={d['total_candidates']}, kept={d['kept']}, "
            f"axis_bad={d['axis_bad']}, width_none={d['width_none']}, width_out={d['width_out_of_range']}, "
            f"min_insert_fail={d['min_insert_fail']}, collision_fail={d['collision_fail']}, fallback_used={d['fallback_used']}")

def build_count_summary(counts: List[int]) -> Dict[str, Any]:
    if len(counts) == 0:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p10": 0.0,
            "p90": 0.0,
            "exact_distribution": {},
            "range_distribution": {}
        }

    arr = np.asarray(counts, dtype=np.int32)

    exact_counter = Counter(arr.tolist())
    exact_distribution = {str(k): int(exact_counter[k]) for k in sorted(exact_counter.keys())}

    bins = [
        ("0",       lambda x: x == 0),
        ("1",       lambda x: x == 1),
        ("2-5",     lambda x: (x >= 2)   & (x <= 5)),
        ("6-10",    lambda x: (x >= 6)   & (x <= 10)),
        ("11-20",   lambda x: (x >= 11)  & (x <= 20)),
        ("21-50",   lambda x: (x >= 21)  & (x <= 50)),
        ("51-100",  lambda x: (x >= 51)  & (x <= 100)),
        ("101-200", lambda x: (x >= 101) & (x <= 200)),
        ("201-500", lambda x: (x >= 201) & (x <= 500)),
        ("501+",    lambda x: x >= 501),
    ]
    range_distribution = {}
    for name, fn in bins:
        range_distribution[name] = int(np.sum(fn(arr)))

    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "exact_distribution": exact_distribution,
        "range_distribution": range_distribution,
    }

# ============================================================
# 6) Per-folder pipeline
# ============================================================
def generate_for_one_folder(
    folder: str,
    hand_params: dict,
    delta_close: float,
    angle_step_deg: float,
    height_step: float,
    max_grasps: int,
    include_topdown: bool,
    include_topbottom: bool,
    topdown_insert_depths: List[float],
    sphere_dirs: int,
    sphere_rot: int,
    min_insert: float,
    approach_insert_frac: float,
    verbose_empty: bool,
) -> Tuple[bool, int, Dict[str, Any]]:
    """
    返回:
      succ: 是否为有效文件夹（存在meta+ply且能读点云）
      k: 最终有效 grasp 数
      report: 用于汇总/输出原因
    """
    rep: Dict[str, Any] = {"id": obj_id_from_folder(folder), "folder": folder}

    meta_path = os.path.join(folder, "meta.json")
    pcd_path = os.path.join(folder, "nontextured.ply")
    if (not os.path.exists(meta_path)) or (not os.path.exists(pcd_path)):
        rep["status"] = "skip_missing_files"
        rep["missing_meta"] = (not os.path.exists(meta_path))
        rep["missing_ply"] = (not os.path.exists(pcd_path))
        rep["num_grasps"] = 0
        return False, 0, rep

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as e:
        rep["status"] = "bad_meta_json"
        rep["error"] = str(e)
        rep["num_grasps"] = 0
        return False, 0, rep

    ptype = meta.get("type", "")
    dims = meta.get("dims", [])
    rep["ptype"] = ptype
    rep["dims"] = dims

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points, dtype=float)
    except Exception as e:
        rep["status"] = "bad_pointcloud"
        rep["error"] = str(e)
        rep["num_grasps"] = 0
        return False, 0, rep

    if pts.shape[0] < 50:
        rep["status"] = "too_few_points"
        rep["num_points"] = int(pts.shape[0])
        rep["num_grasps"] = 0
        return False, 0, rep

    center = aabb_center(pts)
    pts_c = pts - center
    rep["center_used"] = center.tolist()

    candidates: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        if ptype == "cylinder":
            r, H = float(dims[0]), float(dims[1])
            candidates = gen_candidates_cylinder(
                r, H, hand_params, angle_step_deg, height_step,
                include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
            )
        elif ptype == "ellip_cyl":
            rx, ry, H = float(dims[0]), float(dims[1]), float(dims[2])
            candidates = gen_candidates_ellip_cyl(
                rx, ry, H, hand_params, angle_step_deg, height_step,
                include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
            )
        elif ptype == "frustum":
            rb, rt, H = float(dims[0]), float(dims[1]), float(dims[2])
            candidates = gen_candidates_frustum(
                rb, rt, H, hand_params, angle_step_deg, height_step,
                include_topdown, include_topbottom, topdown_insert_depths, approach_insert_frac
            )
        elif ptype == "sphere":
            r = float(dims[0])
            candidates = gen_candidates_sphere(
                r=r, n_dirs=sphere_dirs, n_rot=sphere_rot,
                approach_insert_frac=approach_insert_frac,
                hand_params=hand_params
            )
        elif ptype == "ellipsoid":
            rx, ry, rz = float(dims[0]), float(dims[1]), float(dims[2])
            if abs(rx - ry) < 1e-6 and abs(ry - rz) < 1e-6:
                candidates = gen_candidates_sphere(
                    r=rx, n_dirs=sphere_dirs, n_rot=sphere_rot,
                    approach_insert_frac=approach_insert_frac,
                    hand_params=hand_params
                )
            else:
                candidates = gen_candidates_ellipsoid(
                    rx, ry, rz,
                    angle_step_deg=angle_step_deg,
                    use_extremes_only=False,
                    add_binormal_sign=True,
                    approach_insert_frac=approach_insert_frac,
                    hand_params=hand_params
                )
        elif ptype == "poly_prism":
            n = int(round(float(dims[0])))

            center = center_poly_prism(pts)
            pts_c = pts - center
            rep["center_used"] = center.tolist()
            rep["center_mode"] = "poly_prism_area_centroid"

            Rv_est, H_est = estimate_poly_prism_dims_from_cloud(pts_c)
            yaw0 = estimate_prism_yaw_face0(pts_c, n)

            candidates = gen_candidates_poly_prism(
                n=n,
                Rv=Rv_est,
                H=H_est,
                hand_params=hand_params,
                height_step=height_step,
                edge_margin=0.01,
                yaw_face0=yaw0,
                include_topbottom=include_topbottom,
                topbottom_angle_step_deg=15.0,
                approach_insert_frac=approach_insert_frac,
            )
            dims = [float(n), float(Rv_est), float(H_est)]
            rep["dims_reestimated"] = dims
        elif ptype == "box":
            mn = pts_c.min(axis=0)
            mx = pts_c.max(axis=0)
            ext = mx - mn
            w, h, d = float(ext[0]), float(ext[1]), float(ext[2])
            dims = [w, h, d]
            rep["dims_reestimated"] = dims

            candidates = gen_candidates_box(
                w=w, h=h, d=d,
                surface_insert=0.015,
                add_roll90=True
            )
        else:
            rep["status"] = "unknown_primitive"
            rep["num_grasps"] = 0
            return False, 0, rep

    except Exception as e:
        rep["status"] = "candidate_generation_exception"
        rep["error"] = str(e)
        rep["num_grasps"] = 0
        if verbose_empty:
            print(f"[CAND_EXC] id={rep['id']} ptype={ptype} error={e}")
        return False, 0, rep

    rep["num_candidates"] = int(len(candidates))
    if len(candidates) == 0:
        rep["status"] = "empty_candidates"
        rep["num_grasps"] = 0
        if verbose_empty:
            print(f"[EMPTY_CAND] id={rep['id']} ptype={ptype} dims={rep.get('dims')} reason=generator_returned_empty")
        return True, 0, rep

    width_fn = make_width_fn(ptype, dims, delta_close=delta_close, hand_params=hand_params)
    grasps_c, stats = validate_candidates_on_cloud(
        pts=pts_c,
        candidates=candidates,
        hand_params=hand_params,
        delta_close=delta_close,
        max_keep=max_grasps,
        min_insert=min_insert,
        z_bin_size=max(1e-6, float(height_step)),
        width_fn=width_fn,
    )
    rep["validate_stats"] = asdict(stats)

    if grasps_c.shape[0] == 0:
        rep["status"] = "empty_valid_grasps"
        rep["num_grasps"] = 0
        if verbose_empty:
            print(f"[EMPTY_VALID] id={rep['id']} ptype={ptype} dims={rep.get('dims')} | {summarize_stats(stats)}")
        return True, 0, rep

    num_grasps = int(grasps_c.shape[0])
    rep["num_grasps"] = num_grasps
    rep["status"] = "ok"
    return True, num_grasps, rep

# ============================================================
# 7) Main
# ============================================================
def main():
    parser = argparse.ArgumentParser("Count structured grasps per primitive using meta.json (no file saving)")
    parser.add_argument("--base_path", type=str, default='shot_fpfh/descriptors/dataset/database_2k',
                        help="Dataset root")
    parser.add_argument("--delta_close", type=float, default=0.025,
                        help="Total clearance (m). width = 2*radius + delta_close")
    parser.add_argument("--angle_step_deg", type=float, default=20.0,
                        help="Angle step for ring sampling on side/topdown families.")
    parser.add_argument("--height_step", type=float, default=0.01,
                        help="Height step (m) for side-ring sampling.")
    parser.add_argument("--max_grasps", type=int, default=1024,
                        help="Max grasps counted per object (z-binned uniform subsample if exceeded).")
    parser.add_argument("--include_topdown", action="store_true",
                        help="Enable top/bottom approaching family.")
    parser.add_argument("--include_topbottom", action="store_true",
                        help="Enable top-bottom pinch family (binormal=z).")
    parser.add_argument("--topdown_depths", type=str, default="0.01,0.02,0.03",
                        help="Comma-separated insertion depths (m) for topdown family.")
    parser.add_argument("--sphere_dirs", type=int, default=48,
                        help="Number of fibonacci directions for sphere/ellipsoid.")
    parser.add_argument("--sphere_rot", type=int, default=3,
                        help="Number of binormal rotations per direction for sphere/ellipsoid.")
    parser.add_argument("--min_insert", type=float, default=0.0,
                        help="Optional: minimum insertion depth inside finger channel (m). Default 0 disables.")
    parser.add_argument("--approach_insert_frac", type=float, default=1.0/3.0,
                        help="Try move p_sample along +approach by (frac * radius) for stability. Will fallback if collides.")
    parser.add_argument("--verbose_empty", action="store_true",
                        help="Print per-id reasons for empty candidates/empty valid grasps.")
    parser.add_argument("--report_path", type=str, default="",
                        help="If set, save a JSON report to this path (e.g. grasp_count_report.json).")
    args = parser.parse_args()

    base_path = args.base_path
    if not os.path.isdir(base_path):
        raise FileNotFoundError(base_path)

    topdown_depths: List[float] = []
    for s in args.topdown_depths.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            topdown_depths.append(float(s))
        except Exception:
            pass
    if len(topdown_depths) == 0:
        topdown_depths = [0.01]

    folders = []
    for name in os.listdir(base_path):
        p = os.path.join(base_path, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "nontextured.ply")) and os.path.exists(os.path.join(p, "meta.json")):
            folders.append(p)
    folders.sort()

    total = 0
    ok = 0
    kept = 0

    reports: List[Dict[str, Any]] = []
    empty_cand_ids = []
    empty_valid_ids = []

    counts_all_valid = []       # 所有有效文件夹（包含0）
    counts_nonzero = []         # 仅 grasp > 0
    counts_by_type = defaultdict(list)

    for folder in tqdm(folders, desc="Counting grasps"):
        total += 1
        succ, k, rep = generate_for_one_folder(
            folder=folder,
            hand_params=HAND_PARAMS_REAL,
            delta_close=args.delta_close,
            angle_step_deg=args.angle_step_deg,
            height_step=args.height_step,
            max_grasps=args.max_grasps,
            include_topdown=args.include_topdown,
            include_topbottom=args.include_topbottom,
            topdown_insert_depths=topdown_depths,
            sphere_dirs=args.sphere_dirs,
            sphere_rot=args.sphere_rot,
            min_insert=args.min_insert,
            approach_insert_frac=args.approach_insert_frac,
            verbose_empty=args.verbose_empty,
        )

        if succ:
            ok += 1
            kept += k

            counts_all_valid.append(int(k))
            if k > 0:
                counts_nonzero.append(int(k))

            ptype = rep.get("ptype", "unknown")
            counts_by_type[ptype].append(int(k))

        reports.append(rep)
        if rep.get("status") == "empty_candidates":
            empty_cand_ids.append(rep.get("id"))
        if rep.get("status") == "empty_valid_grasps":
            empty_valid_ids.append(rep.get("id"))

    print(f"\n[Done] processed={total}, valid_folders={ok}, total_grasps={kept}")
    print(f"[Empty] empty_candidates={len(empty_cand_ids)}, empty_valid_grasps={len(empty_valid_ids)}")

    if len(empty_cand_ids) > 0:
        print("  - empty_candidates ids (first 50):", empty_cand_ids[:50])
    if len(empty_valid_ids) > 0:
        print("  - empty_valid_grasps ids (first 50):", empty_valid_ids[:50])

    summary_all = build_count_summary(counts_all_valid)
    summary_nonzero = build_count_summary(counts_nonzero)

    print("\n[Grasp Count Summary | valid folders, including zeros]")
    print(json.dumps(summary_all, ensure_ascii=False, indent=2))

    print("\n[Grasp Count Summary | non-zero only]")
    print(json.dumps(summary_nonzero, ensure_ascii=False, indent=2))

    print("\n[Range Distribution | valid folders, including zeros]")
    for k, v in summary_all["range_distribution"].items():
        print(f"  {k:>8s}: {v}")

    print("\n[Per Primitive Type]")
    for ptype in sorted(counts_by_type.keys()):
        s = build_count_summary(counts_by_type[ptype])
        print(
            f"  {ptype:12s} | n={s['n']:4d} | mean={s['mean']:.2f} | "
            f"median={s['median']:.2f} | min={s['min']:4d} | max={s['max']:4d}"
        )

    if args.report_path:
        summary_pack = {
            "overall_including_zero": summary_all,
            "overall_nonzero_only": summary_nonzero,
            "per_type": {k: build_count_summary(v) for k, v in sorted(counts_by_type.items())},
            "empty_candidates_ids": empty_cand_ids,
            "empty_valid_grasps_ids": empty_valid_ids,
            "reports": reports,
        }
        try:
            with open(args.report_path, "w") as f:
                json.dump(summary_pack, f, ensure_ascii=False, indent=2)
            print(f"[Report] saved to: {args.report_path}")
        except Exception as e:
            print(f"[Report] failed to save: {args.report_path} | error={e}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    main()