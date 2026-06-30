import os
import json
import math
import numpy as np
import open3d as o3d
import multiprocessing
import time
from tqdm import tqdm
from typing import Tuple, List

# ==============================================================================
# 1. 核心算法复用 (保持不变)
# ==============================================================================

def project_to_plane(points: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = n / (np.linalg.norm(n) + 1e-12)
    return points - np.outer(points @ n, n)

def voxel_downsample_2d_on_plane(points_proj: np.ndarray, n: np.ndarray, voxel=0.005) -> np.ndarray:
    n = n / (np.linalg.norm(n) + 1e-12)
    tmp = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, tmp); u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u);  v /= (np.linalg.norm(v) + 1e-12)
    uv = np.stack([points_proj @ u, points_proj @ v], axis=1)
    keys = np.floor(uv / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points_proj[idx]

def pca_in_plane(points_proj: np.ndarray, n: np.ndarray):
    P = points_proj - points_proj.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    e1 = Vt[0]
    e1 = e1 - e1.dot(n) * n; e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1);    e2 /= (np.linalg.norm(e2) + 1e-12)
    return e1, e2

def compute_sobb_extents(points: np.ndarray, table_normal=np.array([0.,0.,1.]), voxel=0.005) -> np.ndarray:
    if points.shape[0] < 5: return np.zeros(3)
    n = (table_normal / (np.linalg.norm(table_normal) + 1e-12))
    P_proj = project_to_plane(points, n)
    P_proj_ds = voxel_downsample_2d_on_plane(P_proj, n, voxel=voxel)
    if P_proj_ds.shape[0] < 3: P_proj_ds = P_proj
    e1, e2 = pca_in_plane(P_proj_ds, n)
    R = np.stack([e1, e2, n], axis=1)
    Q = (points @ R)
    ext = Q.max(axis=0) - Q.min(axis=0)
    return ext

def compute_fpfh_descriptor_o3d(pcd, radius, max_nn=100):
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return np.array(fpfh.data).T

def top2_feature_pairs_from_fpfh(fpfh: np.ndarray, use_ratio_bin=True) -> np.ndarray:
    h = np.asarray(fpfh, dtype=float)
    if h.shape[0] == 0: return np.zeros((0, 2), dtype=np.int64)
    denom = h.sum(axis=1, keepdims=True) + 1e-12
    h_norm = h / denom
    idx_sorted = np.argsort(h_norm, axis=1)
    top1 = idx_sorted[:, -1]
    top2 = idx_sorted[:, -2]
    if not use_ratio_bin: return np.stack([top1, top2], axis=1).astype(np.int64)
    v1 = h_norm[np.arange(h_norm.shape[0]), top1]
    v2 = h_norm[np.arange(h_norm.shape[0]), top2]
    r = v1 / (v1 + v2 + 1e-12)
    r_bin = np.zeros_like(r, dtype=np.int64)
    r_bin[(r >= 0.55) & (r < 0.70)] = 1
    r_bin[(r >= 0.70) & (r < 0.85)] = 2
    r_bin[r >= 0.85] = 3
    top2_enc = top2 + 33 * r_bin
    pairs = np.stack([top1, top2_enc], axis=1).astype(np.int64)
    return np.sort(pairs, axis=1)

# ==============================================================================
# 2. 几何生成器 (新增 Ellipsoid, 移除 Cone)
# ==============================================================================

def create_superquadric_mesh(scale, eps1, eps2, resolution=50):
    a, b, c = scale
    if min(a,b,c) < 1e-4: return None
    eta = np.linspace(-math.pi/2, math.pi/2, resolution)
    omega = np.linspace(-math.pi, math.pi, resolution)
    eta, omega = np.meshgrid(eta, omega)
    eta = eta.flatten()
    omega = omega.flatten()
    def sgn_pow(val, p): return np.sign(val) * (np.abs(val) ** p)
    x = a * sgn_pow(np.cos(eta), eps1) * sgn_pow(np.cos(omega), eps2)
    y = b * sgn_pow(np.cos(eta), eps1) * sgn_pow(np.sin(omega), eps2)
    z = c * sgn_pow(np.sin(eta), eps1)
    vertices = np.stack([x, y, z], axis=1)
    if np.isnan(vertices).any() or np.isinf(vertices).any(): return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    try: mesh, _ = pcd.compute_convex_hull()
    except: return None
    mesh.compute_vertex_normals()
    return mesh

def create_ellipsoid_mesh(rx, ry, rz, resolution=30):
    """
    [新增] 椭球体：通过非均匀缩放球体实现
    覆盖：鹅卵石、土豆、芒果、扁球
    """
    if min(rx, ry, rz) < 1e-4: return None
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    
    # 手动修改顶点实现非均匀缩放
    verts = np.asarray(mesh.vertices)
    verts[:, 0] *= rx
    verts[:, 1] *= ry
    verts[:, 2] *= rz
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    return mesh

def create_capsule_mesh(radius, height, resolution=30):
    if radius < 1e-4: return None
    cyl_height = max(0, height - 2 * radius)
    if cyl_height <= 0: return o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cyl_height, resolution=resolution, split=4)
    top = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    bot = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    top.translate([0, 0, cyl_height/2])
    bot.translate([0, 0, -cyl_height/2])
    combined = cyl + top + bot
    combined.compute_vertex_normals()
    return combined

def create_elliptical_cylinder(r_x, r_y, height, resolution=40):
    if min(r_x, r_y, height) < 1e-4: return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=height, resolution=resolution, split=4)
    verts = np.asarray(cyl.vertices)
    verts[:, 0] *= r_x
    verts[:, 1] *= r_y
    cyl.vertices = o3d.utility.Vector3dVector(verts)
    cyl.compute_vertex_normals()
    return cyl

def create_frustum_mesh(r_bottom, r_top, height, resolution=30):
    if min(r_bottom, r_top, height) < 1e-4:
        return None

    mesh = o3d.geometry.TriangleMesh()
    vertices = []

    # 顶面环
    z_top =  height / 2.0
    z_bot = -height / 2.0
    for i in range(resolution):
        theta = 2.0 * math.pi * i / resolution
        vertices.append([
            r_top * math.cos(theta),
            r_top * math.sin(theta),
            z_top
        ])

    # 底面环
    for i in range(resolution):
        theta = 2.0 * math.pi * i / resolution
        vertices.append([
            r_bottom * math.cos(theta),
            r_bottom * math.sin(theta),
            z_bot
        ])

    # 顶 / 底中心索引（注意顺序）
    top_c_idx = len(vertices)
    vertices.append([0.0, 0.0, z_top])
    bot_c_idx = len(vertices)
    vertices.append([0.0, 0.0, z_bot])

    triangles = []

    for i in range(resolution):
        j = (i + 1) % resolution

        top_i = i
        top_j = j
        bot_i = i + resolution
        bot_j = j + resolution

        # 侧面：两个三角形组成一个四边形
        triangles.append([top_i, bot_i, top_j])
        triangles.append([top_j, bot_i, bot_j])

        # 顶面扇形
        triangles.append([top_c_idx, top_i, top_j])

        # 底面扇形（反向，保证法向一致）
        triangles.append([bot_c_idx, bot_j, bot_i])

    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def create_polygonal_prism_mesh(n_sides: int, radius: float, height: float):
    """
    规则 n 边棱柱：
      - 横截面是正 n 边形（在 XY 平面）
      - 高度沿 Z 方向

    参数:
      n_sides : 边数 (>=3)
      radius  : 顶点到中心的半径
      height  : 总高度
    """
    n_sides = int(round(n_sides))
    if n_sides < 3:
        return None
    if min(radius, height) < 1e-4:
        return None

    verts = []

    # 顶面环
    z_top =  height / 2.0
    z_bot = -height / 2.0
    for i in range(n_sides):
        theta = 2.0 * math.pi * i / n_sides
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append([x, y, z_top])

    # 底面环
    for i in range(n_sides):
        theta = 2.0 * math.pi * i / n_sides
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append([x, y, z_bot])

    # 顶面、底面中心点
    top_center_idx = len(verts)
    verts.append([0.0, 0.0, z_top])
    bot_center_idx = len(verts)
    verts.append([0.0, 0.0, z_bot])

    triangles = []

    # 侧面 + 顶面 + 底面
    for i in range(n_sides):
        j = (i + 1) % n_sides

        # 侧面：两个三角形组成一个四边形
        top_i = i
        top_j = j
        bot_i = n_sides + i
        bot_j = n_sides + j

        triangles.append([top_i, bot_i, top_j])
        triangles.append([top_j, bot_i, bot_j])

        # 顶面扇形
        triangles.append([top_center_idx, top_i, top_j])

        # 底面扇形（注意朝向反一下）
        triangles.append([bot_center_idx, bot_j, bot_i])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def create_primitive_mesh_v2(prim_type, params):
    mesh = None
    try:
        # ------- 特殊分支：多边棱柱，需要整数边数 -------
        if prim_type == "poly_prism":
            n_sides = int(round(params[0]))
            radius  = max(1e-4, float(params[1]))
            height  = max(1e-4, float(params[2]))
            mesh = create_polygonal_prism_mesh(n_sides, radius, height)

        else:
            # 其余类型仍然用原来的 safe_params 逻辑
            safe_params = [max(1e-4, float(p)) for p in params]

            if prim_type == "box":
                mesh = o3d.geometry.TriangleMesh.create_box(
                    width=safe_params[0], height=safe_params[1], depth=safe_params[2]
                )
            elif prim_type == "cylinder":
                mesh = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=safe_params[0], height=safe_params[1],
                    resolution=30, split=4
                )
            elif prim_type == "sphere":
                mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=safe_params[0], resolution=30
                )
            elif prim_type == "ellipsoid":
                mesh = create_ellipsoid_mesh(
                    safe_params[0], safe_params[1], safe_params[2]
                )
            elif prim_type == "torus":
                mesh = o3d.geometry.TriangleMesh.create_torus(
                    torus_radius=safe_params[0], tube_radius=safe_params[1],
                    radial_resolution=30, tubular_resolution=20
                )
            elif prim_type == "capsule":
                mesh = create_capsule_mesh(safe_params[0], safe_params[1])
            elif prim_type == "ellip_cyl":
                mesh = create_elliptical_cylinder(
                    safe_params[0], safe_params[1], safe_params[2]
                )
            elif prim_type == "frustum":
                mesh = create_frustum_mesh(
                    safe_params[0], safe_params[1], safe_params[2]
                )
            elif prim_type == "rounded_box":
                mesh = create_superquadric_mesh(
                    [safe_params[0]/2, safe_params[1]/2, safe_params[2]/2],
                    0.2, 0.2
                )
            elif prim_type == "pillow":
                mesh = create_superquadric_mesh(
                    [safe_params[0]/2, safe_params[1]/2, safe_params[2]/2],
                    1.0, 0.2
                )

        if mesh is not None:
            mesh.translate(-mesh.get_center())
            mesh.compute_vertex_normals()
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
    except Exception:
        return None
    return mesh

# ==============================================================================
# 3. 任务执行核心
# ==============================================================================

def run_task_internal(args):
    (target_folder, prim_type, params, 
     highres_voxel, down_voxel, normal_radius, fpfh_radius, gripper_limit, min_gen_size) = args
    
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    try:
        mesh = create_primitive_mesh_v2(prim_type, params)
        if mesh is None or not mesh.has_triangles(): return 0

        area = mesh.get_surface_area()
        if area < 1e-6: return 0
        sample_density = (highres_voxel / 2.0) ** 2
        n_samples = int(area / sample_density)
        n_samples = max(2000, min(n_samples, 200000))
        pcd_high = mesh.sample_points_uniformly(number_of_points=n_samples)
        
        pcd_down = pcd_high.voxel_down_sample(voxel_size=down_voxel)
        
        # [修改] 允许非常薄的物体 (只要有 10 个点就能算 FPFH)
        if len(pcd_down.points) < 10: return 0 

        pts_down = np.asarray(pcd_down.points)
        sobb_extents = compute_sobb_extents(pts_down, voxel=down_voxel)
        
        # 至少有一个维度 <= gripper_limit（14 cm）
        if np.min(sobb_extents) > gripper_limit:
            return 0 
        
        # 所有维度至少为 min_gen_size（这里设置为 1 cm）
        # 防止生成过薄/过小的物体
        if np.min(sobb_extents) < min_gen_size:
            return 0

        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
        pcd_down.orient_normals_consistent_tangent_plane(50)
        fpfh = compute_fpfh_descriptor_o3d(pcd_down, radius=fpfh_radius)
        pairs = top2_feature_pairs_from_fpfh(fpfh, use_ratio_bin=True)
        
        os.makedirs(target_folder, exist_ok=True)
        o3d.io.write_triangle_mesh(os.path.join(target_folder, "model.obj"), mesh, write_ascii=True, print_progress=False)
        o3d.io.write_point_cloud(os.path.join(target_folder, "highres.ply"), pcd_high, write_ascii=True, print_progress=False)
        o3d.io.write_point_cloud(os.path.join(target_folder, "highres.pcd"), pcd_high, write_ascii=True, print_progress=False)
        o3d.io.write_point_cloud(os.path.join(target_folder, "nontextured.ply"), pcd_down, write_ascii=True, print_progress=False)
        
        np.savez(
            os.path.join(target_folder, "cache.npz"),
            points=pts_down,
            normals=np.asarray(pcd_down.normals),
            fpfh=fpfh,
            pairs=pairs,
            sobb=sobb_extents,
            scale=1.0, 
            center=np.zeros(3)
        )
        
        meta = {
            "id": os.path.basename(target_folder),
            "type": prim_type,
            "dims": list(params),
            "valid_grasp": True,
            "params": {"nr": normal_radius, "fr": fpfh_radius, "vx": down_voxel}
        }
        with open(os.path.join(target_folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
            
        return 1 
    except Exception:
        return 0 

def batch_process_wrapper(queue, batch_tasks):
    success_count = 0
    for task in batch_tasks:
        try:
            res = run_task_internal(task)
            success_count += res
        except Exception:
            pass 
    queue.put(success_count)

def main():
    dst_root = "/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/database_2k" 
    
    # 算法参数
    highres_voxel = 0.001
    down_voxel    = 0.005
    normal_radius = 0.01
    fpfh_radius   = 0.02
    
    # 抓取与尺寸限制
    gripper_limit = 0.12  # 抓取宽度（14 cm）
    min_gen_size  = 0.01  # 所有维度下限 1 cm
    
    tasks = []
    idx = 0
    
    def add(ptype, pparams):
        nonlocal idx
        tasks.append((
            os.path.join(dst_root, f"{idx:05d}"), 
            ptype, pparams,
            highres_voxel, down_voxel, normal_radius, fpfh_radius, gripper_limit, min_gen_size
        ))
        idx += 1

    print(f"[Info] Configuring database (Cyl/Box/Frustum/EllipCyl/Sphere/Ellipsoid/PolyPrism)...")

    # ================= 1. Cylinder (标准圆柱) =================
    # 半径: 0.5 cm ~ 4.0 cm（直径 1 cm ~ 8 cm）
    radii = np.linspace(0.005, 0.04, 25) 
    aspects = np.concatenate([
        np.linspace(0.1, 0.9, 5),    # 盘
        np.linspace(1.0, 3.0, 8),    # 罐
        np.linspace(3.5, 12.0, 8),   # 杆
        [15.0, 20.0]                 # 极长
    ])
    for r in radii:
        if r > 0.04:
            continue
        for ar in aspects:
            h = 2 * r * ar
            # 所有尺寸 >= 1 cm
            if 2 * r < min_gen_size:   # 直径 < 1 cm
                continue
            if h < min_gen_size:
                continue
            if h > 0.40:
                continue 
            add("cylinder", (r, h))

    # ================= 2. Elliptical Cylinder (椭圆柱) =================
    # 控制数量：目标 ~ 200 个左右
    # 长轴半径 (Major Radius): 1.5 cm ~ 4 cm，采样点从 15 个减到 8 个
    major_radii = np.linspace(0.015, 0.04, 8)
    
    # 扁平率 (短轴/长轴)，从 6 个减到 4 个，仍保留“很扁 / 一般扁 / 接近圆”的代表
    flattening = [0.2, 0.4, 0.6, 0.8]
    
    # 高度：用 6 个点覆盖 3 cm ~ 20 cm 区间
    heights = np.linspace(0.03, 0.20, 6)
    
    for r_maj in major_radii:
        if r_maj > 0.04:
            continue
        for ratio in flattening:
            r_min = r_maj * ratio
            
            # 所有尺寸 >= 1 cm（两个直径都至少 1 cm）
            if min(2 * r_min, 2 * r_maj) < min_gen_size:
                continue
            
            for h in heights:
                if h < min_gen_size:
                    continue
                # 只要有一个方向 <= 14 cm 就可以抓取
                if min(2*r_min, 2*r_maj, h) > gripper_limit:
                    continue
                
                add("ellip_cyl", (r_maj, r_min, h))

    # ================= 3. Box (方块/板/条) =================
    base_sizes = np.linspace(0.015, 0.135, 20) 
    ratios = [
        (1.0, 1.0),   # Cube
        (1.0, 0.6), (1.0, 0.3),     # Brick
        (1.0, 0.1), (1.0, 0.05),    # Plate
        (0.5, 0.5), (0.3, 0.3),     # Bar
        (3.0, 0.1), (0.5, 0.1)      # Strip
    ]
    import random
    random.seed(42)
    for b in base_sizes:
        for (ry, rz) in ratios:
            for _ in range(2): 
                w = b
                h = b * ry * random.uniform(0.95, 1.05)
                d = b * rz * random.uniform(0.95, 1.05)
                # 至少有一个维度 <= 14 cm，所有维度 >= 1 cm
                if min(w, h, d) > gripper_limit:
                    continue
                if min(w, h, d) < min_gen_size:
                    continue
                add("box", (w, h, d))

    # ================= 4. Frustum (圆台/杯子) =================
    # 底部半径：1 cm ~ 4 cm
    base_radii = np.linspace(0.01, 0.04, 10)
    # 锥度比
    taper_ratios = [0.5, 0.65, 0.8, 1.2, 1.5, 2.0]
    heights = np.linspace(0.05, 0.18, 6)
    
    for r_b in base_radii:
        if r_b > 0.04:
            continue
        for ratio in taper_ratios:
            r_t = r_b * ratio
            # 顶/底半径都不能超过 4 cm
            if max(r_b, r_t) > 0.04:
                continue
            for h in heights:
                # 所有尺寸 >= 1 cm
                if min(2*r_b, 2*r_t, h) < min_gen_size:
                    continue
                # 至少一个维度 <= 14 cm
                if min(2*r_b, 2*r_t, h) > gripper_limit:
                    continue
                add("frustum", (r_b, r_t, h))

    # ================= 5 & 6. Ellipsoid + Sphere（总量约束 + 均匀覆盖） =================
    # 先收集所有合法候选，再统一随机下采样到最多 100 个，避免尺寸集中在某一段

    ellip_candidates = []
    # 半轴：1 cm ~ 4 cm
    axes = np.linspace(0.01, 0.04, 10)
    shape_factors = [
        (0.9, 0.8), (0.7, 0.5), (0.5, 0.3), (1.0, 0.3)
    ]
    for a in axes:
        for (fy, fz) in shape_factors:
            for _ in range(2):
                rx = a
                ry = a * fy * random.uniform(0.95, 1.05)
                rz = a * fz * random.uniform(0.95, 1.05)

                # 半轴不能超过 4 cm，所有尺度 >= 1 cm
                if max(rx, ry, rz) > 0.04:
                    continue
                if min(2*rx, 2*ry, 2*rz) < min_gen_size:  # 直径
                    continue
                # 至少一个维度 <= 14 cm
                if min(2*rx, 2*ry, 2*rz) > gripper_limit:
                    continue

                ellip_candidates.append(("ellipsoid", (rx, ry, rz)))

    sphere_candidates = []
    for r in np.linspace(0.005, 0.04, 15):
        # 半径 <= 4 cm，直径在 [1 cm, 14 cm] 范围
        if 2 * r < min_gen_size:   # 直径 < 1 cm
            continue
        if 2 * r > gripper_limit:  # 直径 > 14 cm（这里理论上不会，因为 r<=0.04）
            continue
        sphere_candidates.append(("sphere", (r,)))

    # 合并候选并做总量限制
    ellip_sphere_candidates = ellip_candidates + sphere_candidates
    MAX_ELLIP_SPHERE = 100

    if len(ellip_sphere_candidates) > MAX_ELLIP_SPHERE:
        random.shuffle(ellip_sphere_candidates)  # 保证从整个参数网格中均匀抽样
        ellip_sphere_candidates = ellip_sphere_candidates[:MAX_ELLIP_SPHERE]

    for ptype, pparams in ellip_sphere_candidates:
        add(ptype, pparams)


    print(f"[Info] Total Tasks Prepared: {len(tasks)}")
    print("[Info] Running with BATCHED PROCESS ISOLATION...")
    
    # 批处理执行
    BATCH_SIZE = 50
    batches = [tasks[i:i + BATCH_SIZE] for i in range(0, len(tasks), BATCH_SIZE)]
    
    valid_count = 0
    pbar = tqdm(total=len(tasks))
    
    for batch in batches:
        try:
            ctx = multiprocessing.get_context('spawn')
            queue = ctx.Queue()
            p = ctx.Process(target=batch_process_wrapper, args=(queue, batch))
            p.start()
            p.join()
            if p.exitcode == 0 and not queue.empty():
                valid_count += queue.get()
        except Exception:
            pass
        pbar.update(len(batch))
        
    pbar.close()
    print(f"\n[Success] Generated {valid_count} primitives in '{dst_root}'.")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    main()
