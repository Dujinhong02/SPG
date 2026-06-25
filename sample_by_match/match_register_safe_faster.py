"""
总流程(包括点云按模块分割——各块点云(包括完整)进行匹配、配对、gg映射——全部gg汇总(已转换到p的坐标系))
1、filter faster and update
Author: djh
"""
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 若你用到 torch+cuda 并希望严格确定性（必须在 import torch 前设置）
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import ctypes
import os

# # 你的 conda 环境前缀
# CONDA_PREFIX = "/home/ubuntu/miniconda3/envs/pmatch"

# # pip 安装的 nvidia-nvjitlink-cu12 对应的库路径
# lib_nvjit = os.path.join(
#     CONDA_PREFIX,
#     "lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12",
# )

# # 先把正确版本的 nvJitLink 以 RTLD_GLOBAL 方式加载进来
# ctypes.CDLL(lib_nvjit, mode=ctypes.RTLD_GLOBAL)

import copy
import os
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional
from sklearn.neighbors import KDTree
import time
from termcolor import cprint
from itertools import permutations
import sys
sys.path.insert(0,'/root/catkin_ws/src/more_than_grasp/sample_by_match/PartField')
# from cluster import cluster
# from inference_and_clustering import cluster
import os
import time
import pickle
import numpy as np
import open3d as o3d
from itertools import permutations
import random
import torch
from functools import lru_cache

def set_global_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 牺牲速度换确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@lru_cache(maxsize=8192)
def load_pose_best_array(folder_or_path: str) -> np.ndarray:
    """
    返回 (N,13) float32/float64 数组
    - 优先读取 pose_best.npy
    - 不存在则回退 pose_best.txt
    folder_or_path: 可以传物体文件夹，也可以直接传 pose_best.npy / pose_best.txt
    """
    p = folder_or_path

    if os.path.isdir(p):
        npy_path = os.path.join(p, "pose_best.npy")
        txt_path = os.path.join(p, "pose_best.txt")
    else:
        # 传进来是具体文件
        if p.endswith(".npy"):
            npy_path, txt_path = p, p[:-4] + ".txt"
        elif p.endswith(".txt"):
            txt_path, npy_path = p, p[:-4] + ".npy"
        else:
            # 传进来是不带后缀的 base
            npy_path, txt_path = p + ".npy", p + ".txt"

    if os.path.isfile(npy_path):
        arr = np.load(npy_path)
    elif os.path.isfile(txt_path):
        # 你的保存分隔符是 ", "，loadtxt 用 "," 也能正常解析（会忽略空格）
        arr = np.loadtxt(txt_path, delimiter=",")
        # 兼容单行
        if arr.ndim == 1:
            arr = arr[None, :]
        # 可选：读到后顺便落盘 npy，加速后续
        try:
            np.save(npy_path, arr.astype(np.float32, copy=False))
        except Exception:
            pass
    else:
        return np.zeros((0, 13), dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 13:
        raise ValueError(f"pose_best expects (N,13), got {arr.shape}")
    return arr

def load_grasps_components(folder_or_path: str):
    G = load_pose_best_array(folder_or_path)
    pos      = G[:, 0:3]
    axis     = G[:, 3:6]
    approach = G[:, 6:9]
    binormal = G[:, 9:12]
    width    = G[:, 12]
    return pos, axis, approach, binormal, width

def rot_angle_deg(R1, R2):
    R = R1.T @ R2
    cosv = (np.trace(R) - 1.0) / 2.0
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def grasp_pose(g):
    pos = np.asarray(g[0:3], float)
    axis = np.asarray(g[3:6], float)
    approach = np.asarray(g[6:9], float)
    binormal = np.asarray(g[9:12], float)
    width = float(g[12])
    R = build_orthonormal_frame(approach, binormal, axis)  # 你已有
    return pos, R, width

def grasp_distance(g1, g2, w_pos=1.0, w_rot=1.0, w_w=0.3,
                   pos_scale=0.02, rot_scale_deg=20.0, width_scale=0.02):
    """
    返回一个“越大越不同”的距离（归一化后加权）。
    pos_scale/rot_scale/width_scale 是“差异显著”的尺度。
    """
    p1, R1, w1 = grasp_pose(g1)
    p2, R2, w2 = grasp_pose(g2)

    dp = np.linalg.norm(p1 - p2) / (pos_scale + 1e-12)
    dr = rot_angle_deg(R1, R2) / (rot_scale_deg + 1e-12)
    dw = abs(w1 - w2) / (width_scale + 1e-12)
    return w_pos * dp + w_rot * dr + w_w * dw

def is_similar(g1, g2, pos_thr=0.01, rot_thr_deg=10.0, width_thr=0.01):
    p1, R1, w1 = grasp_pose(g1)
    p2, R2, w2 = grasp_pose(g2)
    if np.linalg.norm(p1 - p2) < pos_thr and rot_angle_deg(R1, R2) < rot_thr_deg and abs(w1 - w2) < width_thr:
        return True
    return False

def select_diverse_grasps(sorted_grasps, target_k=10,
                          pos_thr=0.01, rot_thr_deg=12.0, width_thr=0.01,
                          relax_steps=(1.0, 1.5, 2.0, 3.0),
                          prekeep_max=60,
                          width_for_similarity=None):
    """
    sorted_grasps: 已按偏好从好到差排序后的 grasp (N,13)
    width_for_similarity: (N,) 与 sorted_grasps 对齐；用于相似度/距离计算的 width（建议传“增宽前”的 orig width）
    返回: (K,13)
    """
    sorted_grasps = np.asarray(sorted_grasps, dtype=float)
    if sorted_grasps.shape[0] <= target_k:
        return sorted_grasps

    cand = sorted_grasps[:min(prekeep_max, sorted_grasps.shape[0])]
    if width_for_similarity is not None:
        wsim = np.asarray(width_for_similarity, dtype=float)[:cand.shape[0]]
    else:
        wsim = cand[:, 12].copy()

    for factor in relax_steps:
        keep = [0]  # 存索引
        remaining = list(range(1, cand.shape[0]))

        while len(keep) < target_k and remaining:
            best_j = None
            best_score = -1.0

            for j in remaining:
                g = cand[j]

                # 去重：和任意已选太像就跳过
                too_sim = False
                for kk in keep:
                    if is_similar_w(g, cand[kk],
                                    w1=wsim[j], w2=wsim[kk],
                                    pos_thr=pos_thr*factor,
                                    rot_thr_deg=rot_thr_deg*factor,
                                    width_thr=width_thr*factor):
                        too_sim = True
                        break
                if too_sim:
                    continue

                # max-min：离已选集合“最远的最近距离”最大
                dmin = min(grasp_distance_w(g, cand[kk], w1=wsim[j], w2=wsim[kk]) for kk in keep)
                if dmin > best_score:
                    best_score = dmin
                    best_j = j

            if best_j is None:
                break

            keep.append(best_j)
            remaining.remove(best_j)

        if len(keep) >= target_k:
            return cand[np.asarray(keep[:target_k], dtype=int)]

    # 兜底：放宽仍不足，按偏好补满
    return cand[:target_k]

# def plane_pcd(side_len=0.4, voxel_size=0.002, z0=0.0) -> o3d.geometry.PointCloud:
#     # 每边体素数
#     n = int(round(side_len / voxel_size))
#     if not np.isclose(n * voxel_size, side_len):
#         raise ValueError(f"side_len 必须是 voxel_size 的整数倍：{n} * {voxel_size} = {n*voxel_size} ≠ {side_len}")

#     half = side_len / 2.0
#     edges = np.linspace(-half, half, n + 1)          # n+1 条边界
#     centers = (edges[:-1] + edges[1:]) / 2.0         # n 个体素中心

#     X, Y = np.meshgrid(centers, centers, indexing="xy")
#     Z = np.full_like(X, float(z0))

#     points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     return pcd

@lru_cache(maxsize=8)
def _cached_plane_points(side_len: float, voxel_size: float, z0: float) -> np.ndarray:
    # 每边体素数
    n = int(round(side_len / voxel_size))
    if not np.isclose(n * voxel_size, side_len):
        raise ValueError(f"side_len 必须是 voxel_size 的整数倍：{n} * {voxel_size} = {n*voxel_size} ≠ {side_len}")

    half = side_len / 2.0
    edges = np.linspace(-half, half, n + 1)          # n+1 条边界
    centers = (edges[:-1] + edges[1:]) / 2.0         # n 个体素中心

    X, Y = np.meshgrid(centers, centers, indexing="xy")
    Z = np.full_like(X, float(z0))

    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

def plane_pcd(side_len=0.4, voxel_size=0.002, z0=0.0) -> o3d.geometry.PointCloud:
    points = _cached_plane_points(float(side_len), float(voxel_size), float(z0))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.copy())
    return pcd

def transform_grasps(pos, axis, approach, binormal, width, T):
    """
    返回 (N,13): [pos, axis, approach, binormal, width]
    预分配输出，避免 hstack 的大内存分配。
    """
    if T.shape != (4, 4):
        raise ValueError("T 必须是 4x4 齐次变换矩阵")

    pos = np.asarray(pos, dtype=float)
    axis = np.asarray(axis, dtype=float)
    approach = np.asarray(approach, dtype=float)
    binormal = np.asarray(binormal, dtype=float)
    width = np.asarray(width, dtype=float).reshape(-1)

    R = T[:3, :3]
    t = T[:3, 3]

    N = pos.shape[0]
    out = np.empty((N, 13), dtype=float)

    out[:, 0:3] = (R @ pos.T).T + t
    out[:, 3:6] = (R @ axis.T).T
    out[:, 6:9] = (R @ approach.T).T
    out[:, 9:12] = (R @ binormal.T).T
    out[:, 12] = width

    return out

def _unit3(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)

def grasp_pose_w(g, width_override=None):
    pos = np.asarray(g[0:3], float)
    axis = np.asarray(g[3:6], float)
    approach = np.asarray(g[6:9], float)
    binormal = np.asarray(g[9:12], float)
    width = float(width_override) if width_override is not None else float(g[12])
    R = build_orthonormal_frame(approach, binormal, axis)  # 你原有函数
    return pos, R, width

def grasp_distance_w(g1, g2, w1=None, w2=None,
                     w_pos=1.0, w_rot=1.0, w_w=0.3,
                     pos_scale=0.02, rot_scale_deg=20.0, width_scale=0.02):
    p1, R1, ww1 = grasp_pose_w(g1, w1)
    p2, R2, ww2 = grasp_pose_w(g2, w2)
    dp = np.linalg.norm(p1 - p2) / (pos_scale + 1e-12)
    dr = rot_angle_deg(R1, R2) / (rot_scale_deg + 1e-12)  # 你原有函数
    dw = abs(ww1 - ww2) / (width_scale + 1e-12)
    return w_pos * dp + w_rot * dr + w_w * dw

def is_similar_w(g1, g2, w1=None, w2=None, pos_thr=0.01, rot_thr_deg=10.0, width_thr=0.01):
    p1, R1, ww1 = grasp_pose_w(g1, w1)
    p2, R2, ww2 = grasp_pose_w(g2, w2)
    if np.linalg.norm(p1 - p2) < pos_thr and rot_angle_deg(R1, R2) < rot_thr_deg and abs(ww1 - ww2) < width_thr:
        return True
    return False

from functools import lru_cache

@lru_cache(maxsize=4096)
def _cached_npz_points(npz_path: str, voxel: float):
    with np.load(npz_path) as data:
        pts = np.asarray(data["points"], dtype=np.float64)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if voxel and voxel > 0:
        pc = pc.voxel_down_sample(float(voxel))
    pc.translate(-pc.get_center())

    # 缓存“处理后的点”，避免缓存可变的 Open3D 对象
    return np.asarray(pc.points, dtype=np.float64)

def load_pcd_from_cache_npz(npz_path, voxel=0.005):
    pts = _cached_npz_points(npz_path, float(voxel))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def build_index_arrays(index_data):
    ids = []
    paths = []
    Tsq = []
    c_shape = []
    c_ratio = []
    c_scale = []
    c_ems = []

    for e in index_data:
        ems = e.get("ems", None)
        if ems is None or ems.get("T_sq", None) is None:
            continue
        ids.append(int(e["id"]))
        paths.append(e["path"].replace('/home/djh/code/shot-fpfh', '/home/ubuntu/task/more_than_grasp/fpfh'))
        Tsq.append(np.asarray(ems["T_sq"], dtype=np.float64))

        sh = np.asarray(ems.get("ems_shape", ems.get("shape")), dtype=np.float64).reshape(2)
        sc = np.asarray(ems.get("ems_scale", ems.get("scale")), dtype=np.float64).reshape(3)
        ra = np.asarray(ems.get("ems_ratio", ems.get("ratio")), dtype=np.float64).reshape(3)

        c_shape.append(sh)
        c_scale.append(sc)
        c_ratio.append(ra)
        c_ems.append(ems)

    c_shape = np.stack(c_shape, axis=0)
    c_scale = np.stack(c_scale, axis=0)
    c_ratio = np.stack(c_ratio, axis=0)
    c_ratio_log = np.log(c_ratio + 1e-12)

    # 预计算：候选的几何均值（abs_scale_distance 用）
    c_gm = np.exp(np.mean(np.log(c_scale + 1e-12), axis=1))

    return {
        "ids": np.asarray(ids, dtype=np.int32),
        "paths": paths,
        "Tsq": Tsq,           # list[4x4]
        "c_shape": c_shape,   # (M,2)
        "c_scale": c_scale,   # (M,3)
        "c_ratio_log": c_ratio_log,  # (M,3)
        "c_gm": c_gm,         # (M,)
        "c_ems": c_ems,       # list[dict]
    }

def ems_match_fast(query_pcd_norm, idxA, top_k_shape, top_k_scale, top_n,
                   w_ratio, w_eps, lambda_scale, lambda_abs,
                   hard_scale_tol=None, hard_abs_tol=None, visualize=False):
    t0 = time.time()
    qems, sq, info = fit_ems_from_pcd(query_pcd_norm, visualize=visualize, arc_length_viz=0.004, point_size=8.0)
    if qems is None:
        return [], None
    t1 = time.time()
    # print(f"Tems:{t1-t0}")
    q_vars = gen_theta_variants_query(qems)

    # 先堆叠成数组，避免后面 list 索引
    q_ratio_logs = np.stack([np.log(np.asarray(v["ratio"], np.float64)+1e-12) for v in q_vars], axis=0)  # (Q,3)
    q_shapes     = np.stack([np.asarray(v["shape"], np.float64) for v in q_vars], axis=0)                        # (Q,2)
    q_scales     = np.stack([np.asarray(v["scale"], np.float64) for v in q_vars], axis=0)                        # (Q,3)
    q_Ts         = np.stack([np.asarray(v["T_sq"], np.float64) for v in q_vars], axis=0)                         # (Q,4,4)

    c_ratio_log = idxA["c_ratio_log"]      # (M,3)
    c_shape     = idxA["c_shape"]          # (M,2)

    # ---- StageA：对 Q 维做广播，直接取 min ----
    # dr: (Q,M), de: (Q,M)
    dr = np.linalg.norm(c_ratio_log[None, :, :] - q_ratio_logs[:, None, :], axis=2)
    de = np.linalg.norm(c_shape[None, :, :]     - q_shapes[:, None, :],     axis=2)
    ds = w_ratio * dr + w_eps * de              # (Q,M)

    best_q_idx = np.argmin(ds, axis=0).astype(np.int32)  # (M,)
    best_dshape = ds[best_q_idx, np.arange(ds.shape[1])]
    best_dratio = dr[best_q_idx, np.arange(dr.shape[1])]
    best_deps   = de[best_q_idx, np.arange(de.shape[1])]

    kA = min(int(top_k_shape), best_dshape.size)
    idx_topA = np.argpartition(best_dshape, kA-1)[:kA]
    idx_topA = idx_topA[np.argsort(best_dshape[idx_topA], kind="mergesort")]

    # ---- StageB：向量化计算 dscale/dabs/score ----
    qi = best_q_idx[idx_topA]                # (kA,)
    qs = q_scales[qi]                        # (kA,3)
    cs = idxA["c_scale"][idx_topA]           # (kA,3)

    Sq = np.exp(np.mean(np.log(qs + 1e-12), axis=1))     # (kA,)
    dscale = np.abs(np.log((Sq + 1e-12) / (idxA["c_gm"][idx_topA] + 1e-12)))  # (kA,)

    rel = np.abs(qs - cs) / (np.maximum(qs, cs) + 1e-12)
    dabs = np.mean(rel, axis=1)                           # (kA,)

    # 硬阈值 mask
    mask = np.ones_like(dscale, dtype=bool)
    if hard_scale_tol is not None:
        mask &= dscale <= np.log(1.0 + float(hard_scale_tol))
    if hard_abs_tol is not None:
        mask &= dabs <= float(hard_abs_tol)

    if not np.any(mask):
        return [], qems

    idx_keep = idx_topA[mask]
    qi_keep  = qi[mask]
    score = (best_dshape[idx_keep]
             + lambda_scale * dscale[mask]
             + lambda_abs   * dabs[mask])

    order = np.argsort(score, kind="mergesort")
    take = order[:min(int(top_k_scale), order.size)]
    idx_final = idx_keep[take]
    qi_final  = qi_keep[take]

    # ---- 只对最终 TopK 组装 dict ----
    out = []
    for n_i, j in enumerate(idx_final):
        qii = int(qi_final[n_i])
        out.append({
            "id": int(idxA["ids"][j]),
            "path": idxA["paths"][j],
            "dshape": float(best_dshape[j]),
            "dratio": float(best_dratio[j]),
            "dε": float(best_deps[j]),
            "dscale": float(dscale[mask][take][n_i]),
            "dabs": float(dabs[mask][take][n_i]),
            "score": float(score[take][n_i]),
            "ems": idxA["c_ems"][j],
            "Tq_used": q_Ts[qii],
            "Tc_used": np.asarray(idxA["Tsq"][j], dtype=np.float64),
            "q_used": q_vars[qii],
            "ems_used": {"ems_shape": idxA["c_shape"][j], "ems_scale": idxA["c_scale"][j], "T_sq": idxA["Tsq"][j]},
        })
    return out, qems

def _signed_perm_mats_det_pos():
    mats = []
    base = np.eye(3)
    for perm in permutations([0,1,2], 3):
        P = base[:, perm]  # columns permuted
        # sign flips: diag(sx,sy,sz)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    S = np.diag([sx, sy, sz])
                    M = P @ S
                    if np.linalg.det(M) > 0:   # keep right-handed
                        mats.append(M)
    return mats  # usually 24


def _nn_trimmed_score(src_pts, kdt: KDTree, trim=0.9):
    d, _ = kdt.query(src_pts, k=1)
    d = d.reshape(-1)
    if d.size == 0:
        return np.inf
    d = np.sort(d)
    d = d[:max(10, int(trim * d.size))]
    return float(np.mean(d*d))  # MSE

def align_candidate_to_query_by_Tsq(
    query_pcd_norm, cand_pcd_norm,
    Tq, Tc,
    do_icp=False, voxel=0.005, icp_max_iter=30,
    resolve_sq_axis=False,
):
    Tq = np.asarray(Tq, float)
    Tc = np.asarray(Tc, float)
    Tc_inv = np.linalg.inv(Tc)

    # 1) 预建 query KDTree（一次）
    q_pts = np.asarray(query_pcd_norm.points, dtype=float)
    kdt = KDTree(q_pts)

    c_pts = np.asarray(cand_pcd_norm.points, dtype=float)

    # 2) 枚举 M，先用“快速误差”选最优 M（不跑 ICP）
    Ms = [np.eye(3)]
    if resolve_sq_axis:
        for m in _signed_perm_mats_det_pos():
            if not np.allclose(m, np.eye(3)):
                Ms.append(m)

    best = None
    for M3 in Ms:
        M = np.eye(4)
        M[:3, :3] = M3
        T_ems = Tq @ M @ Tc_inv

        # 直接用 numpy 变换点（比 deepcopy+transform 快很多）
        R = T_ems[:3, :3]
        t = T_ems[:3, 3]
        c_tf = (R @ c_pts.T).T + t

        score = _nn_trimmed_score(c_tf, kdt, trim=0.6)  # 越小越好
        
        if (best is None) or (score < best["score"]):
            best = {"score": score, "T_ems": T_ems, "M": M}

    # 3) 用最优 T_ems 生成 aligned 点云
    cand_aligned = copy.deepcopy(cand_pcd_norm)
    cand_aligned.transform(best["T_ems"])

    # 4) 可选：只对“最优 M”跑一次 ICP 微调
    T_icp = np.eye(4)
    fitness = rmse = None
    if do_icp:
        # 建议先用 point-to-point（更快，不需要法向）
        T_icp, fitness, rmse = _icp_refine_T_point2point(
            cand_aligned, query_pcd_norm, voxel=voxel, max_iter=icp_max_iter
        )
        cand_aligned.transform(T_icp)

    T_total = T_icp @ best["T_ems"]
    dbg = {"pre_score": best["score"], "fitness": fitness, "rmse": rmse, "M": best["M"]}
    return cand_aligned, T_total, best["T_ems"], T_icp, dbg

def _icp_refine_T_point2point(source_pcd, target_pcd, voxel=0.005, max_iter=30):
    max_corr = float(3.0 * voxel)
    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter))
    )
    return reg.transformation, reg.fitness, reg.inlier_rmse


def register_by_superquadric(query_pcd_norm, top_entries,Tq=None,
                                                     n=5, voxel=0.005,
                                                     center_mode="keep",
                                                     do_icp=False, icp_max_iter=30,
                                                     visualize = False,
                                                     hand_params = None):
    all_grasp_poses = []
    q_vis = copy.deepcopy(query_pcd_norm)
    q_vis.paint_uniform_color([1.0, 0.2, 0.2])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    if Tq is None:
        qems, qsq, qinfo = fit_ems_from_pcd(query_pcd_norm, visualize=visualize, center_mode=center_mode)
        if qems is None:
            print("[Error] Query EMS fitting failed.")
            return
        Tq = qems["T_sq"]

    for rank, e in enumerate(top_entries[:n]):
        cand = load_pcd_from_cache_npz(e["path"], voxel=voxel)
        cand.paint_uniform_color([0.2, 0.6, 1.0])

        # 关键：优先使用 ems_match 命中的 variant 变换
        Tq_entry = np.asarray(e.get("Tq_used", Tq), dtype=float)
        Tc_entry = np.asarray(e.get("Tc_used", e["ems"]["T_sq"]), dtype=float)

        cand_aligned, T_total, T_ems, T_icp, dbg = align_candidate_to_query_by_Tsq(
            query_pcd_norm=query_pcd_norm,
            cand_pcd_norm=cand,
            Tq=Tq_entry,
            Tc=Tc_entry,
            do_icp=do_icp,
            voxel=voxel,
            icp_max_iter=icp_max_iter,

            resolve_sq_axis=False,
        )


        # print(f"[Rank{rank}] id={e.get('id','?')},score={e.get('score')} dshape={e.get('dshape')} dscale={e.get('dscale')} dabs={e.get('dabs')}")

        if visualize:
            o3d.visualization.draw_geometries(
                # [q_vis, cand_aligned, frame],
                [q_vis, cand_aligned],
                window_name=f"Aligned Rank{rank} id={e.get('id','?')}"
            )

        base_path = os.path.dirname(e["path"])
        # grasp_path = base_path + "/grasp_pose.txt"
        # grasp_path = base_path + "/pose_best.txt"
        grasp_path = os.path.join(base_path, "pose_best.txt")
        if not os.path.isfile(grasp_path):
            print(f"[WARN] grasp file missing, skip: {grasp_path}")
            continue
        pos, axis, approach, binormal, width = load_grasps_components(grasp_path)
        # T_inv = np.linalg.inv(T_total)
        grasps2p = transform_grasps(pos, axis, approach, binormal, width, T_total)

        if visualize:
            geo2 = [q_vis]
            pos2      = grasps2p[:, 0:3]
            axis2     = grasps2p[:, 3:6]
            approach2 = grasps2p[:, 6:9]
            binormal2 = grasps2p[:, 9:12]
            width2    = grasps2p[:, 12]
            # for i in range(pos2.shape[0]):
            #     mesh2 = create_gripper_mesh(pos2[i], approach2[i], binormal2[i], axis2[i],width2[i],hand_params)
            #     geo2.extend(mesh2)
            # o3d.visualization.draw_geometries(geo2)
        
        all_grasp_poses.append(grasps2p)

    return all_grasp_poses

def _Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def _ratio_from_scale(scale3):
    sc = np.asarray(scale3, dtype=float).reshape(3)
    sc = np.abs(sc) + 1e-12
    gm = gmean3(sc)
    return sc / (gm + 1e-12)

def _extract_theta_query(qems):
    # query: shape/scale/T_sq/ratio
    eps = np.asarray(qems["shape"], dtype=float).reshape(2)
    sc  = np.asarray(qems["scale"], dtype=float).reshape(3)
    T   = np.asarray(qems["T_sq"], dtype=float).reshape(4,4)
    return eps, sc, T

def _extract_theta_cand(ems):
    # cand: ems_shape/ems_scale/T_sq (或 fallback)
    eps = np.asarray(ems.get("ems_shape", ems.get("shape")), dtype=float).reshape(2)
    sc  = np.asarray(ems.get("ems_scale", ems.get("scale")), dtype=float).reshape(3)
    T   = np.asarray(ems.get("T_sq"), dtype=float).reshape(4,4)
    return eps, sc, T

def gen_theta_variants_query(qems):
    """
    输出的每个 variant 仍用 query schema:
      {"shape","scale","ratio","T_sq"}
    """
    eps, sc, T = _extract_theta_query(qems)
    eps1, eps2 = float(eps[0]), float(eps[1])

    vars_ = []
    # v0: identity
    vars_.append({
        "shape": np.array([eps1, eps2], float),
        "scale": np.abs(sc).astype(float),
        "ratio": _ratio_from_scale(sc),
        "T_sq": np.array(T, float),
    })

    if (eps2 < 0.015) or ((2-eps2) < 0.015):
        # if abs(sc[0] - sc[1]) < 0.001:
        if abs(sc[0] - sc[1]) / (max(abs(sc[0]), abs(sc[1])) + 1e-12) < 0.05:
            # duality: eps2' = 2 - eps2
            eps_dual = np.array([eps1, 2.0 - eps2], dtype=float)

            # (ax,ay)/sqrt(2), az unchanged
            sc_dual = np.array(sc, float)
            sc_dual[0] = sc_dual[0] / np.sqrt(2.0)
            sc_dual[1] = sc_dual[1] / np.sqrt(2.0)

            # R' = R * Rz(pi/4), t unchanged
            T_dual = np.array(T, float)
            T_dual[:3, :3] = T_dual[:3, :3] @ _Rz(np.pi / 4.0)

            vars_.append({
                "shape": eps_dual,
                "scale": np.abs(sc_dual),
                "ratio": _ratio_from_scale(sc_dual),
                "T_sq": T_dual,
            })

    return vars_

def gen_theta_variants_cand(ems):
    """
    输出的每个 variant 使用 cand schema:
      {"ems_shape","ems_scale","ems_ratio","T_sq"}
    """
    eps, sc, T = _extract_theta_cand(ems)
    eps1, eps2 = float(eps[0]), float(eps[1])

    vars_ = []
    # v0: identity
    vars_.append({
        "ems_shape": np.array([eps1, eps2], float),
        "ems_scale": np.abs(sc).astype(float),
        "ems_ratio": _ratio_from_scale(sc),
        "T_sq": np.array(T, float),
    })

    return vars_

def gmean3(x):
    x = np.asarray(x, dtype=float)
    return float((x[0] * x[1] * x[2]) ** (1.0 / 3.0))

def _import_ems():
    import sys
    sys.path.insert(
        0,
        "/root/catkin_ws/src/more_than_grasp/sample_by_match/EMS-superquadric_fitting/Python/src"
    )
    try:
        from EMS.EMS_recovery import EMS_recovery
        return EMS_recovery
    except Exception:
        import EMS.EMS_recovery as ems_mod
        return ems_mod.EMS_recovery

def visualize_ems_open3d(points_xyz, sq, arc_length=0.005, point_size=6.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.paint_uniform_color([0.1, 0.1, 0.9])

    mesh = superquadric_to_mesh_o3d(sq, arc_length=arc_length)
    mesh.paint_uniform_color([0.9, 0.1, 0.1])

    vis = o3d.visualization.Visualizer()
    vis.create_window("EMS Superquadric (query)", 900, 700)
    # vis.add_geometry(pcd)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def fit_ems_from_pcd(query_pcd_norm: o3d.geometry.PointCloud,
                     visualize=False,
                     arc_length_viz=0.005,
                     point_size=6.0,
                     center_mode="keep"):  
    sq, info = ems_fit_autotune(
        query_pcd_norm,
        outlier_ratios=(0.001),
        adaptive_upper=(True, False),
        preprocess_voxel=0.0,
        center_mode=center_mode,       
        arc_length_eval=arc_length_viz
    )
    if sq is None:
        print(f'INFO:{info}')
        return None, None, info

    shape, scale, R, t = _safe_get_sq_params(sq)
    scale = np.abs(scale)

    gm = (scale[0]*scale[1]*scale[2]) ** (1/3)
    ratio = scale / (gm + 1e-12)

    T_sq = np.eye(4)
    T_sq[:3, :3] = R
    T_sq[:3, 3] = t

    qems = {
        "shape": shape,
        "scale": scale,
        "ratio": ratio,
        "T_sq": T_sq,        
        "fit_info": info
    }

    if visualize:
        pts = np.asarray(query_pcd_norm.points, dtype=float)
        visualize_ems_open3d(pts, sq, arc_length=arc_length_viz, point_size=point_size)

    return qems, sq, info

def _safe_get_sq_params(sq):
    # 兼容不同 EMS 版本：属性名可能是 shape/scale/translation，也可能是 _shape/_scale/_translation
    shape = np.array(getattr(sq, "shape", getattr(sq, "_shape", None)), dtype=float).reshape(-1)
    scale = np.array(getattr(sq, "scale", getattr(sq, "_scale", None)), dtype=float).reshape(-1)
    t = np.array(getattr(sq, "translation", getattr(sq, "_translation", [0,0,0])), dtype=float).reshape(-1)

    if shape.size != 2 or scale.size != 3 or t.size != 3:
        raise ValueError("Bad superquadric params (shape/scale/translation).")

    # rotation：优先 Rotation 对象
    R = None
    r_obj = getattr(sq, "_r", None)
    if r_obj is not None:
        try:
            R = r_obj.as_matrix()
        except Exception:
            R = None

    # fallback：euler
    if R is None:
        euler = getattr(sq, "euler", None)
        if euler is not None:
            euler = np.array(euler, dtype=float).reshape(-1)
            if euler.size == 3:
                # ZYX yaw-pitch-roll（与你原代码一致）
                alpha, beta, gamma = euler
                ca, sa = np.cos(alpha), np.sin(alpha)
                cb, sb = np.cos(beta),  np.sin(beta)
                cg, sg = np.cos(gamma), np.sin(gamma)
                Rz = np.array([[ca, -sa, 0],[sa, ca, 0],[0,0,1]])
                Ry = np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
                Rx = np.array([[1,0,0],[0,cg,-sg],[0,sg,cg]])
                R = Rz @ Ry @ Rx
            else:
                R = np.eye(3)
        else:
            R = np.eye(3)

    return shape, scale, R, t


def superquadric_to_mesh_o3d(sq, arc_length=0.005):
    """
    arc_length: 以“米”为单位的近似边长控制。越小越密。
    """
    (eps1, eps2), (a1, a2, a3), R, t = _safe_get_sq_params(sq)

    a1, a2, a3 = np.abs([a1, a2, a3]) + 1e-12
    L = float(max(a1, a2, a3))

    # 按“真实长度”设置采样密度（下限/上限防爆）
    arc = max(float(arc_length), 1e-4)
    n_eta   = int(np.clip(np.ceil(np.pi * L / arc),  24, 220))
    n_omega = int(np.clip(np.ceil(2*np.pi * L / arc), 48, 420))

    eta   = np.linspace(-np.pi/2.0, np.pi/2.0, n_eta)
    omega = np.linspace(-np.pi, np.pi, n_omega, endpoint=False)
    eta, omega = np.meshgrid(eta, omega)

    cos_eta, sin_eta = np.cos(eta), np.sin(eta)
    cos_omg, sin_omg = np.cos(omega), np.sin(omega)

    def f(x, e):
        return np.sign(x) * (np.abs(x) ** e)

    x = a1 * f(cos_eta, eps1) * f(cos_omg, eps2)
    y = a2 * f(cos_eta, eps1) * f(sin_omg, eps2)
    z = a3 * f(sin_eta, eps1)

    V = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)  # (N,3)
    V = (R @ V.T).T + t.reshape(1,3)

    triangles = []
    for i in range(n_omega):
        i2 = (i + 1) % n_omega
        for j in range(n_eta - 1):
            idx0 = i  * n_eta + j
            idx1 = idx0 + 1
            idx2 = i2 * n_eta + j
            idx3 = idx2 + 1
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(V.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh

def _pcd_preprocess_for_ems(pcd: o3d.geometry.PointCloud,
                            voxel=0.0,
                            remove_outlier=True,
                            nb_neighbors=20,
                            std_ratio=2.0,
                            center_mode="keep"):
    """
    center_mode:
      - "keep": 不平移（保留绝对尺度/位置）
      - "center": 平移到中心（更稳，适合你已在外面 translate(-center) 的情况）
    """
    pc = o3d.geometry.PointCloud(pcd)

    if voxel and voxel > 0:
        pc = pc.voxel_down_sample(float(voxel))

    if remove_outlier and len(pc.points) >= nb_neighbors:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))

    pts = np.asarray(pc.points, dtype=float)
    if pts.shape[0] < 50:
        return pc, np.zeros(3)

    shift = np.zeros(3)
    if center_mode == "center":
        shift = pts.mean(axis=0)
        pc.translate(-shift)

    return pc, shift

def _sq_implicit_score(points_xyz: np.ndarray, sq):
    (eps1, eps2), (a1,a2,a3), R, t = _safe_get_sq_params(sq)
    a1, a2, a3 = np.abs([a1,a2,a3]) + 1e-12

    eps1 = float(np.clip(eps1, 1e-3, 2.0))
    eps2 = float(np.clip(eps2, 1e-3, 2.0))

    P = points_xyz.astype(float)
    # world -> sq local
    X = (R.T @ (P - t.reshape(1,3)).T).T
    x = np.abs(X[:,0]) / a1
    y = np.abs(X[:,1]) / a2
    z = np.abs(X[:,2]) / a3

    # 常用隐式形式（用于“相对比较”足够稳定）
    # g = ((x^(2/eps2)+y^(2/eps2))^(eps2/eps1) + z^(2/eps1))^(eps1/2)
    term_xy = (x ** (2.0/eps2) + y ** (2.0/eps2)) ** (eps2/eps1)
    term_z  = z ** (2.0/eps1)
    g = (term_xy + term_z) ** (eps1/2.0)

    err = np.abs(g - 1.0)
    med = float(np.median(err))
    p90 = float(np.quantile(err, 0.90))
    return med + 0.7 * p90


# def load_global_index(index_path):
#     with open(index_path, "rb") as f:
#         return pickle.load(f)

class NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 将 numpy._core.* 映射到 numpy.core.*
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_global_index(index_path):
    with open(index_path, "rb") as f:
        try:
            return NumpyCoreCompatUnpickler(f).load()
        except TypeError:
            # 兼容少数旧pickle（例如需要encoding参数的情况）
            f.seek(0)
            return NumpyCoreCompatUnpickler(f).load()

# def ems_fit_autotune(pcd: o3d.geometry.PointCloud,
#                      outlier_ratios=(0.01),
#                      adaptive_upper=(True, False),
#                      preprocess_voxel=0.0,
#                      center_mode="center",
#                      arc_length_eval=0.005):
#     """
#     返回:
#       best_sq, best_info(dict)
#     """
#     t0 = time.time()

#     EMS_recovery = _import_ems()
#     pc2, shift = _pcd_preprocess_for_ems(
#         pcd,
#         voxel=preprocess_voxel,
#         remove_outlier=True,
#         nb_neighbors=20,
#         std_ratio=2.0,
#         center_mode=center_mode
#     )
#     pts = np.asarray(pc2.points, dtype=float)
#     if pts.shape[0] < 30:
#         return None, {"reason": "too_few_points"}

#     best = {"score": np.inf}
#     best_sq = None
    
#     r = outlier_ratios
#     # for r in outlier_ratios:
#     for au in adaptive_upper:
#         try:
#             t1 = time.time()
#             sq, _ = EMS_recovery(
#                 pts,
#                 OutlierRatio=float(r),
#                 AdaptiveUpperBound=bool(au),
#                     MaxIterationEM=12,
#                     MaxOptiIterations=2,
#                     MaxiSwitch=1,
#             )
#             print(f't1:{time.time()-t1}')
#         except Exception:
#             continue

#         t2 = time.time()
#         score = _sq_implicit_score(pts, sq)
#         med = p90 = None


#         if score < best["score"]:
#             best = {
#                 "score": score,
#                 "median": med,
#                 "p90": p90,
#                 "OutlierRatio": float(r),
#                 "AdaptiveUpperBound": bool(au),
#                 "shift": shift.tolist(),
#                 "n_points": int(pts.shape[0]),
#             }
#             best_sq = sq
#         print(f't2:{time.time()-t2}')
#     print(f'total:{time.time()-t0}')
#     return best_sq, best

def ems_fit_autotune(pcd: o3d.geometry.PointCloud,
                     outlier_ratios=(0.01,),
                     adaptive_upper=(True, False),
                     preprocess_voxel=0.0,
                     center_mode="center",
                     arc_length_eval=0.005,
                     profile=False):
    t_all0 = time.perf_counter()

    EMS_recovery = _import_ems()

    t0 = time.perf_counter()
    pc2, shift = _pcd_preprocess_for_ems(
        pcd,
        voxel=preprocess_voxel,
        remove_outlier=True,
        nb_neighbors=20,
        std_ratio=2.0,
        center_mode=center_mode
    )
    pts = np.asarray(pc2.points, dtype=float)
    t1 = time.perf_counter()

    if pts.shape[0] < 30:
        return None, {"reason": "too_few_points"}

    best = {"score": np.inf}
    best_sq = None

    r = outlier_ratios
    t_recovery = 0.0
    t_score = 0.0

    for au in adaptive_upper:
        try:
            ta = time.perf_counter()
            sq, _ = EMS_recovery(
                pts,
                OutlierRatio=float(r),
                AdaptiveUpperBound=bool(au),
                MaxIterationEM=12,
                MaxOptiIterations=2,
                MaxiSwitch=1,
            )
            tb = time.perf_counter()
            t_recovery += (tb - ta)
        except Exception:
            continue

        tc = time.perf_counter()
        score = _sq_implicit_score(pts, sq)
        td = time.perf_counter()
        t_score += (td - tc)

        if score < best["score"]:
            best = {
                "score": score,
                "OutlierRatio": float(r),
                "AdaptiveUpperBound": bool(au),
                "shift": shift.tolist(),
                "n_points": int(pts.shape[0]),
            }
            best_sq = sq

    if profile:
        t_all1 = time.perf_counter()
        print("\n[ems_fit_autotune profile]")
        print(f"total        : {t_all1 - t_all0:.6f}s")
        print(f"preprocess   : {t1 - t0:.6f}s")
        print(f"recovery sum : {t_recovery:.6f}s")
        print(f"score sum    : {t_score:.6f}s")
        print(f"n_points     : {pts.shape[0]}")
        print("")

    return best_sq, best

def shape_distance(query_ems, cand_ems, w_ratio=1.0, w_eps=0.5):
    rq = np.asarray(query_ems["ratio"], dtype=float)

    rc = np.asarray(
        cand_ems.get("ems_ratio", cand_ems.get("ratio")),
        dtype=float
    )
    d_ratio = float(np.linalg.norm(np.log(rq + 1e-12) - np.log(rc + 1e-12)))

    eq = np.asarray(query_ems["shape"], dtype=float)
    ec = np.asarray(
        cand_ems.get("ems_shape", cand_ems.get("shape")),
        dtype=float
    )
    d_ε = float(np.linalg.norm(eq - ec))
    return w_ratio * d_ratio + w_eps * d_ε, d_ratio, d_ε

def abs_scale_distance(query_scale, cand_scale):
    qs = np.asarray(query_scale, dtype=float)
    cs = np.asarray(cand_scale, dtype=float)
    Sq = gmean3(qs)
    Sc = gmean3(cs)

    return float(abs(np.log((Sq + 1e-12) / (Sc + 1e-12))))

def abs_axes_distance(query_scale, cand_scale):
    qs = np.asarray(query_scale, dtype=float)
    cs = np.asarray(cand_scale, dtype=float)

    rel = np.abs(qs - cs) / (np.maximum(qs, cs) + 1e-12)
    # rel = np.abs(qs - cs) / (qs + cs + 1e-12)
    d = float(np.mean(rel))

    return d
    
def ems_match(query_pcd_norm,
              index_data,
              top_k_shape,
              top_k_scale,
              top_n,
              w_ratio,
              w_eps,
              lambda_scale,
              lambda_abs,
              hard_scale_tol=None,   
              hard_abs_tol=None,
              visualize = None,
              return_ranked = True):
    t0 = time.time()    
    qems, sq, info = fit_ems_from_pcd(query_pcd_norm, visualize=visualize, arc_length_viz=0.004, point_size=8.0)
    t1 = time.time()
    # print(f'EMS Time:{t1-t0}')
    # print(qems)
    if qems is None:
        print("[Error] Query EMS fitting failed.", info)
        return [] ,None
    # print(f'qshape:{qems["shape"]}, qscale:{qems["scale"]}')

    q_vars = gen_theta_variants_query(qems)
    # print(q_vars)
    # -------- 阶段A：形状距离 Top200 --------
    scored = []
    for e in index_data:
        ems = e.get("ems", None)
        if ems is None:
            continue
        # 没有 T_sq 的 entry 直接跳过（否则对齐无法闭环）
        if ems.get("T_sq", None) is None:
            continue

        c_vars = gen_theta_variants_cand(ems)
        
        best = None
        for qv in q_vars:
            for cv in c_vars:
                # 用 variant 的 shape/ratio 计算形状相似
                dshape, dratio, dε = shape_distance(
                    {"shape": qv["shape"], "ratio": qv["ratio"]},
                    cv,
                    w_ratio=w_ratio, w_eps=w_eps
                )
                if (best is None) or (dshape < best["dshape"]):
                    best = {
                        "dshape": float(dshape),
                        "dratio": float(dratio),
                        "dε": float(dε),
                        "q_used": qv,    
                        "c_used": cv,    
                    }

        if best is None:
            continue

        # 注意：把 best 的 q_used/c_used 一起保存
        scored.append((best["dshape"], best["dratio"], best["dε"], e, best["q_used"], best["c_used"]))

    scored.sort(key=lambda x: x[0]) 
    stageA = scored[:min(top_k_shape, len(scored))]
    # print("qems keys:", qems.keys())
    # if stageA:
    #     print("entry keys example:", stageA[0][3].keys())

    # -------- 阶段B：绝对尺度筛选/重排 Top50 --------
    refined = []
    for dshape, dratio, dε, e, q_used, c_used in stageA:
        ems = e["ems"]

        # 关键：尺度项使用同一对 variant 的 scale（严格 S_step）
        qs_used = np.asarray(q_used["scale"], dtype=float).reshape(3)
        cs_used = np.asarray(c_used["ems_scale"], dtype=float).reshape(3)

        dscale = abs_scale_distance(qs_used, cs_used)
        dabs   = abs_axes_distance(qs_used, cs_used)


        # 可选硬阈值（你担心尺度难平衡时，可以先不启用）
        if hard_scale_tol is not None:
            if dscale > np.log(1.0 + float(hard_scale_tol)):
                continue
        if hard_abs_tol is not None:
            if dabs > float(hard_abs_tol):
                continue

        score = dshape + lambda_scale * dscale + lambda_abs * dabs
        refined.append({
            "id": e["id"],
            "path": e["path"],
            "dshape": float(dshape),
            "dratio": float(dratio),
            "dε": float(dε),
            "dscale": float(dscale),
            "dabs": float(dabs),
            "score": float(score),

            # 原始 ems 仍保留（便于 debug/打印）
            "ems": ems,

            # 关键：把命中的等价变换后的 T_sq 带走（理论闭环）
            "Tq_used": np.asarray(q_used["T_sq"], dtype=float),
            "Tc_used": np.asarray(c_used["T_sq"], dtype=float),

            # 可选：保留命中的参数（排查时非常有用）
            "q_used": q_used,
            "ems_used": c_used,
        })


    if not refined:
        refined = []
        for dshape, dratio, dε, e, q_used, c_used in stageA:
            qs_used = np.asarray(q_used["scale"], dtype=float).reshape(3)
            cs_used = np.asarray(c_used["ems_scale"], dtype=float).reshape(3)

            dscale = abs_scale_distance(qs_used, cs_used)
            dabs   = abs_axes_distance(qs_used, cs_used)
            score = dshape + lambda_scale * dscale + lambda_abs * dabs

            refined.append({
                "id": e["id"],
                "path": e["path"],
                "dshape": float(dshape),
                "dratio": float(dratio),
                "dε": float(dε),
                "dscale": float(dscale),
                "dabs": float(dabs),
                "score": float(score),
                "ems": e["ems"],

                "Tq_used": np.asarray(q_used["T_sq"], dtype=float),
                "Tc_used": np.asarray(c_used["T_sq"], dtype=float),
                "q_used": q_used,
                "ems_used": c_used,
            })

    refined.sort(key=lambda x: x["score"])
    stageB = refined[:min(top_k_scale, len(refined))]

    t2 = time.time()
    # print(f'Match Time {t2-t1}')
    if return_ranked:
        # 只打印前 top_n 个，避免刷屏
        show = stageB[:min(top_n, len(stageB))]
        for i, t in enumerate(show):
            id = np.asarray(t["id"], dtype=int)
            csh = np.asarray(t["ems_used"]["ems_shape"], dtype=float).reshape(2)
            cs = np.asarray(t["ems_used"]["ems_scale"], dtype=float).reshape(3)
            dshape = np.asarray(t["dshape"], dtype=float)
            dscale = np.asarray(t["dscale"], dtype=float)
            dabs = np.asarray(t["dabs"], dtype=float)
            score = np.asarray(t["score"], dtype=float)

            # print(f"{id}, c = {csh}{cs}, score = {score} = {dshape:.4f} + {lambda_scale} x {dscale:.4f} + {lambda_abs} x {dabs:.4f}")
        return stageB, qems
    
    # 最终 TopN
    top = stageB[:min(top_n, len(stageB))]
    for i, t in enumerate(top):
        qs = np.asarray(t["q_used"]["scale"], dtype=float).reshape(3)
        cs = np.asarray(t["ems_used"]["ems_scale"], dtype=float).reshape(3)
        print(f"[Top{i}] q_scale={qs}, c_scale={cs}")
    return top, qems

# --------------------------grasp pose-----------------------

def create_gripper_mesh(
    pos: np.ndarray,
    approach: np.ndarray,
    binormal: np.ndarray,
    axis: np.ndarray,
    width: float,
    hand_params: dict,
    color: list = [0.9, 0.2, 0.2]
):

    """
    构造一个平行夹爪的完整网格：
    - pos     : 掌心中心（world）
    - approach: 接近方向（+X）
    - binormal: 两指连线方向（+Y）
    - axis    : 手爪高度方向（+Z）
    - width   : 当前抓取的张口宽度（两指内侧间距）

    满足：掌心中心 = pos；掌心前只有两指；掌心后无其他几何。
    """

    # -------- 1) 单位化方向，并构造 R --------
    def unit(v):
        n = np.linalg.norm(v) + 1e-12
        return v / n

    a = unit(approach)
    b = unit(binormal)
    c = unit(axis)

    # 列向量 = 局部轴在 world 中的方向
    R = np.column_stack((a, b, c))

    # -------- 2) 手爪尺寸 --------
    fw   = float(hand_params["finger_width"])
    od   = float(hand_params["hand_outer_diameter"])
    depth = float(hand_params["hand_depth"])
    # 手指高度
    finger_h = float(hand_params["hand_height"])
    # 掌心高度，没有给时退化为 finger_h，保证兼容性
    palm_t   = float(hand_params.get("palm_thickness", fw))
    palm_h   = float(hand_params.get("palm_height", finger_h))

    # 当前 grasp 的开口
    aperture = float(width)
    max_aperture = od - 2.0 * fw
    if aperture > max_aperture + 1e-6:
        print(f"[WARN] width {aperture:.4f} > max_aperture {max_aperture:.4f}")

    # -------- 3) 在局部坐标系下建模 --------
    # 约定局部坐标:
    #   原点 O 在掌心中心；
    #   掌心厚度沿 X ∈ [-palm_t/2, +palm_t/2]
    #   手指从 X = +palm_t/2 往 +X 方向伸出 depth 长度
    geoms_local = []

    # 3.1 掌心块
    palm_size = np.array([
        palm_t,                  # X: 厚度
        aperture + 2.0 * fw,     # Y: 总宽 (两指 + 两侧厚度)
        palm_h                        # Z: 高度
    ], dtype=float)

    palm = o3d.geometry.TriangleMesh.create_box(
        width=palm_size[0],
        height=palm_size[1],
        depth=palm_size[2],
    )
    # create_box 默认 box 在 [0, sx]×[0, sy]×[0, sz]，我们要让中心在 (0,0,0)
    palm.translate(-palm_size / 2.0)
    palm.compute_vertex_normals()
    geoms_local.append(palm)

    # 3.2 两根手指
    finger_size = np.array([
        depth,   # X: 长度 (从掌心前端往前伸)
        fw,      # Y: 厚度 (指自身宽度)
        finger_h        # Z: 高度
    ], dtype=float)

    # 内侧间距为 aperture，指心在 Y = ±(aperture/2 + fw/2)
    y_off = aperture / 2.0 + fw / 2.0

    # 指在 X 方向的位置:
    #   掌心前端面在 X = +palm_t/2
    #   手指从该面往前 depth，中心在 X = palm_t/2 + depth/2
    x_center = palm_t / 2.0 + depth / 2.0

    def make_finger(center_local):
        box = o3d.geometry.TriangleMesh.create_box(
            width=finger_size[0],
            height=finger_size[1],
            depth=finger_size[2],
        )
        box.translate(center_local - finger_size / 2.0)
        box.compute_vertex_normals()
        return box

    center1 = np.array([x_center,  y_off, 0.0], dtype=float)
    center2 = np.array([x_center, -y_off, 0.0], dtype=float)

    finger1 = make_finger(center1)
    finger2 = make_finger(center2)
    geoms_local.extend([finger1, finger2])

    for geom in geoms_local:
        geom.paint_uniform_color(color)
    # -------- 4) 变换到 world 坐标: 原点 pos, 旋转 R --------
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = np.asarray(pos, dtype=float)

    geoms_world = []
    for g in geoms_local:
        g.transform(T)        
        geoms_world.append(g) 
    return geoms_world


# ------------------------ 基础工具 ------------------------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)

def estimate_cloud_resolution(pts_world: np.ndarray, tree: KDTree, n_samples: int = 5000) -> float:
    n = pts_world.shape[0]
    if n < 10:
        return 0.001
    m = min(n_samples, n)
    # 按顺序等间距抽样，完全确定性
    idx = np.linspace(0, n - 1, m, dtype=int)
    sub = pts_world[idx]
    d, _ = tree.query(sub, k=2)
    return float(np.median(d[:, 1]))

def build_orthonormal_frame(approach: np.ndarray, binormal: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    构造正交且右手的 R = [a b c]，列向量为 local轴在world中的方向：
      a = +X (approach), b = +Y (binormal), c = +Z (axis)
    会对 binormal/axis 做正交化，避免输入不严格正交导致的误差累积。
    """
    a = unit(approach)

    b = np.asarray(binormal, dtype=float).reshape(3)
    b = b - np.dot(b, a) * a
    if np.linalg.norm(b) < 1e-8:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        b = tmp - np.dot(tmp, a) * a
    b = unit(b)

    c = np.asarray(axis, dtype=float).reshape(3)
    c = c - np.dot(c, a) * a - np.dot(c, b) * b
    if np.linalg.norm(c) < 1e-8:
        c = np.cross(a, b)
    c = unit(c)

    R = np.column_stack((a, b, c))
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R


def world_to_local(pts_world: np.ndarray, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    world -> local: P_local = (P_world - pos) @ R
    其中 R 的列向量是 local轴在world中的方向，因此投影到local用 @R。
    """
    return (pts_world - pos.reshape(1, 3)) @ R


def ensure_normals(pcd: o3d.geometry.PointCloud,
                  radius: float = 0.01,
                  max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    若点云无 normals，则估计 normals（兜底）。
    """
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        pcd.normalize_normals()
    return pcd

def build_pmatch_visualization_geometries(
    pcd_base: o3d.geometry.Geometry,
    filtered_grasps: np.ndarray,
    hand_params_visual: dict,
):
    """
    为 pmatch_for_draw 构建可视化几何体，便于复用到显示与录制。
    """
    geometries = [copy.deepcopy(pcd_base)]

    grasps = np.asarray(filtered_grasps, dtype=float)
    if grasps.size == 0:
        return geometries

    pos      = grasps[:, 0:3]
    axis     = grasps[:, 3:6]
    approach = grasps[:, 6:9]
    binormal = grasps[:, 9:12]
    width    = grasps[:, 12]

    for i in range(pos.shape[0]):
        geometries.extend(
            create_gripper_mesh(
                pos[i], approach[i], binormal[i], axis[i],
                width[i], hand_params_visual
            )
        )
    return geometries

def _create_visualizer_window(window_name, width, height, visible):
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(
            window_name=window_name,
            width=int(width),
            height=int(height),
            visible=bool(visible),
        )
    except TypeError:
        vis.create_window(
            window_name=window_name,
            width=int(width),
            height=int(height),
        )
    return vis

def resolve_orbit_video_output_path(output_target, prefix="pmatch_orbit"):
    output_target = os.path.abspath(os.path.expanduser(str(output_target)))
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    root, ext = os.path.splitext(output_target)

    if ext.lower() in video_suffixes:
        parent = os.path.dirname(output_target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(output_target):
            return output_target

        counter = 1
        while True:
            candidate = f"{root}_{counter:03d}{ext}"
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    os.makedirs(output_target, exist_ok=True)
    counter = 1
    while True:
        candidate = os.path.join(output_target, f"{prefix}_{counter:03d}.mp4")
        if not os.path.exists(candidate):
            return candidate
        counter += 1

# def record_orbit_video_around_world_z(
#     geometries,
#     orbit_center,
#     output_path,
#     window_name="Orbit Video Recorder",
#     width=1280,
#     height=960,
#     fps=30.0,
#     duration_sec=6.0,
#     visible=True,
#     revolution_sec=6.0,
# ):
#     """
#     围绕世界 z 轴旋转几何体并录制窗口视频。
#     duration_sec <= 0 时按 revolution_sec 持续录制，直到窗口被手动关闭。
#     """
#     try:
#         import cv2
#     except ImportError as exc:
#         raise ImportError("record_orbit_video_around_world_z 需要安装 cv2 / opencv-python。") from exc

#     output_path = os.path.abspath(os.path.expanduser(str(output_path)))
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     frame_period = 1.0 / max(float(fps), 1e-6)
#     orbit_center = np.asarray(orbit_center, dtype=float).reshape(3)

#     endless_recording = duration_sec is None or float(duration_sec) <= 0.0
#     if endless_recording:
#         frame_count = None
#         delta_angle = 2.0 * np.pi / max(float(fps) * float(revolution_sec), 1.0)
#     else:
#         frame_count = max(int(round(float(fps) * float(duration_sec))), 1)
#         delta_angle = 2.0 * np.pi / frame_count

#     cos_a = float(np.cos(delta_angle))
#     sin_a = float(np.sin(delta_angle))
#     rot_z_step = np.array(
#         [
#             [cos_a, -sin_a, 0.0],
#             [sin_a,  cos_a, 0.0],
#             [0.0,    0.0,   1.0],
#         ],
#         dtype=float,
#     )

#     vis = _create_visualizer_window(window_name, width, height, visible)
#     try:
#         rotating_geometries = [copy.deepcopy(geometry) for geometry in geometries]
#         for geometry in rotating_geometries:
#             vis.add_geometry(geometry)

#         vis.poll_events()
#         vis.update_renderer()

#         writer = cv2.VideoWriter(
#             output_path,
#             cv2.VideoWriter_fourcc(*"mp4v"),
#             float(fps),
#             (int(width), int(height)),
#         )
#         if not writer.isOpened():
#             raise RuntimeError(f"Failed to open video writer for: {output_path}")

#         try:
#             next_frame_time = time.perf_counter()
#             frame_index = 0
#             while True:
#                 now = time.perf_counter()
#                 if now < next_frame_time:
#                     time.sleep(next_frame_time - now)
#                 next_frame_time += frame_period

#                 alive = vis.poll_events()
#                 if alive is False:
#                     break
#                 vis.update_renderer()

#                 frame_rgb = np.asarray(
#                     vis.capture_screen_float_buffer(do_render=True),
#                     dtype=float,
#                 )
#                 frame_uint8 = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
#                 writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

#                 frame_index += 1
#                 if (not endless_recording) and frame_index >= frame_count:
#                     break

#                 for geometry in rotating_geometries:
#                     geometry.rotate(rot_z_step, center=orbit_center)
#                     vis.update_geometry(geometry)
#         finally:
#             writer.release()
#     finally:
#         vis.destroy_window()

#     return output_path

def record_orbit_video_around_world_z(
    geometries,
    orbit_center,
    output_path,
    window_name="Orbit Video Recorder",
    width=1280,
    height=960,
    fps=30.0,
    duration_sec=6.0,
    visible=True,
    revolution_sec=6.0,   # 新增：保持旋转速度的关键
):
    import cv2
    import numpy as np
    import time

    frame_period = 1.0 / max(float(fps), 1e-6)
    orbit_center = np.asarray(orbit_center, dtype=float).reshape(3)

    frame_count = max(int(round(float(fps) * float(duration_sec))), 1)

    # 关键改这里：每帧转角由 revolution_sec 决定，而不是 duration_sec
    delta_angle = 2.0 * np.pi / max(float(fps) * float(revolution_sec), 1.0)

    cos_a = float(np.cos(delta_angle))
    sin_a = float(np.sin(delta_angle))
    rot_z_step = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=float)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=int(width), height=int(height), visible=visible)

    try:
        rotating_geometries = [copy.deepcopy(g) for g in geometries]
        for g in rotating_geometries:
            vis.add_geometry(g)

        vis.poll_events()
        vis.update_renderer()

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )

        next_frame_time = time.perf_counter()
        for _ in range(frame_count):
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_period

            vis.poll_events()
            vis.update_renderer()

            frame_rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True), dtype=float)
            frame_uint8 = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

            for g in rotating_geometries:
                g.rotate(rot_z_step, center=orbit_center)
                vis.update_geometry(g)

        writer.release()
    finally:
        vis.destroy_window()


def record_pmatch_grasp_orbit_video(
    pcd_base,
    filtered_grasps,
    output_path,
    hand_params_visual,
    orbit_center=None,
    width=1280,
    height=960,
    fps=30.0,
    duration_sec=6.0,
    visible=True,
):
    geometries = build_pmatch_visualization_geometries(
        pcd_base,
        filtered_grasps,
        hand_params_visual,
    )
    resolved_output_path = resolve_orbit_video_output_path(
        output_path,
        prefix="pmatch_grasps_single",
    )
    if orbit_center is None:
        orbit_center = np.asarray(pcd_base.get_center(), dtype=float)

    return record_orbit_video_around_world_z(
        geometries,
        orbit_center,
        resolved_output_path,
        window_name=f"pmatch grasps ({len(filtered_grasps)})",
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
        visible=visible,
    )

# ------------------------ Layer-1：无穿透碰撞（实体体积判定） ------------------------
def crop_indices_near_gripper(tree: KDTree,
                              pos: np.ndarray,
                              width: float,
                              fw: float,
                              depth: float,
                              h: float,
                              palm_t: float,
                              extra: float) -> np.ndarray:
    """
    用包围球裁剪夹爪附近点，加速。
    """
    x_max = palm_t / 2.0 + depth
    y_max = width / 2.0 + fw
    z_max = h / 2.0
    r = float(np.sqrt(x_max * x_max + y_max * y_max + z_max * z_max) + extra)
    idx = tree.query_radius(np.asarray(pos, dtype=float).reshape(1, 3), r=r)[0]
    return idx


def collision_free_gripper_volume(P_local: np.ndarray,
                                  width: float,
                                  hand_params: dict,
                                  penetration_slack: float = 0.0) -> bool:
    """
    判定夹爪实体(palm + 两指)与点云是否“无穿透”。
    返回 True 表示无穿透；False 表示存在点落入实体内部（穿透）。

    penetration_slack > 0 时会“收缩实体”，允许点贴近表面而不算穿透。
    """
    fw     = float(hand_params["finger_width"])
    depth  = float(hand_params["hand_depth"])
    finger_h     = float(hand_params["hand_height"])
    palm_t = float(hand_params.get("palm_thickness", fw))
    palm_h     = float(hand_params["palm_height"])

    s = float(max(0.0, penetration_slack))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    # palm 实体体积（收缩 s）
    in_palm = (
        (x >= -palm_t / 2.0 + s) & (x <= +palm_t / 2.0 - s) &
        (np.abs(y) <= (width / 2.0 + fw - s)) &
        (np.abs(z) <= (palm_h / 2.0 - s))
    )

    # fingers 实体体积（收缩 s）
    x_f_min = palm_t / 2.0 + s
    x_f_max = palm_t / 2.0 + depth - s

    in_f1 = (
        (x >= x_f_min) & (x <= x_f_max) &
        (y >= (width / 2.0 + s)) & (y <= (width / 2.0 + fw - s)) &
        (np.abs(z) <= (finger_h / 2.0 - s))
    )
    in_f2 = (
        (x >= x_f_min) & (x <= x_f_max) &
        (y >= (-(width / 2.0 + fw) + s)) & (y <= (-(width / 2.0) - s)) &
        (np.abs(z) <= (finger_h / 2.0 - s))
    )

    penetrates = bool(np.any(in_palm | in_f1 | in_f2))
    return (not penetrates)


# ------------------------ Layer-2：夹缝内必须有点 ------------------------
def has_points_between_fingers(P_local: np.ndarray,
                               width: float,
                               hand_params: dict,
                               min_points: int = 5,
                               margin_y: float = 0.005,
                               x_clearance_front: float = 0.0,
                               x_clearance_back: float = 0.0) -> Tuple[bool, np.ndarray]:
    """
    夹缝空间：
      x ∈ [palm_t/2 + front, palm_t/2 + depth - back]
      |y| <= width/2 - margin_y
      |z| <= h/2
    返回 (has_enough, mask_between)。
    """
    depth  = float(hand_params["hand_depth"])
    finger_h      = float(hand_params["hand_height"])
    palm_t = float(hand_params.get("palm_thickness", float(hand_params["finger_width"])))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    mask = (
        (x >= x_min) & (x <= x_max) &
        (np.abs(y) <= y_lim) &
        (np.abs(z) <= finger_h / 2.0)
    )
    return (np.count_nonzero(mask) >= int(min_points)), mask


# ------------------------ Layer-3：最大d + 法线投影角度 ------------------------
def best_span_and_normal_projection_check(P_local: np.ndarray,
                                         idx_full: np.ndarray,
                                         normals_world: np.ndarray,
                                         approach_world: np.ndarray,
                                         binormal_world: np.ndarray,
                                         width: float,
                                         hand_params: dict,
                                         span_thresh: float = 0.01,
                                         angle_thresh_deg: float = 40.0,
                                         xz_res: float = 0.005,
                                         min_pts_line: int = 5,
                                         y_percentile: float = 0.0,
                                         margin_y: float = 0.0,
                                         x_clearance_front: float = 0.0,
                                         x_clearance_back: float = 0.0) -> Tuple[bool, float, Optional[Tuple[int, int]]]:

    fw     = float(hand_params["finger_width"])
    depth  = float(hand_params["hand_depth"])
    finger_h      = float(hand_params["hand_height"])
    palm_t = float(hand_params.get("palm_thickness", fw))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    mask = (
        (x >= x_min) & (x <= x_max) &
        (np.abs(z) <= finger_h / 2.0) &
        (np.abs(y) <= y_lim)
    )
    if not np.any(mask):
        return False, 0.0, None

    xm = x[mask]
    ym = y[mask]
    zm = z[mask]
    ids_full = idx_full[mask]     # 对应到原点云索引
    y_local = ym                 # 端点侧向判别用

    # 分桶 key
    ix = np.floor((xm - x_min) / float(xz_res)).astype(np.int32)
    iz = np.floor((zm + finger_h / 2.0) / float(xz_res)).astype(np.int32)
    key = ix * 100000 + iz

    order = np.argsort(key)
    key_s = key[order]
    y_s   = ym[order]
    yl_s  = y_local[order]
    id_s  = ids_full[order]

    dmax = 0.0
    best = None  # (idx_lo, idx_hi, ylo_local, yhi_local)

    start = 0
    n = key_s.shape[0]
    while start < n:
        end = start + 1
        while end < n and key_s[end] == key_s[start]:
            end += 1

        cnt = end - start
        if cnt >= int(min_pts_line):
            ys  = y_s[start:end]
            yls = yl_s[start:end]
            ids = id_s[start:end]

            if y_percentile > 0.0:
                lo = np.percentile(ys, y_percentile)
                hi = np.percentile(ys, 100.0 - y_percentile)
                i_lo = int(np.argmin(np.abs(ys - lo)))
                i_hi = int(np.argmin(np.abs(ys - hi)))
            else:
                i_lo = int(np.argmin(ys))
                i_hi = int(np.argmax(ys))

            span = float(ys[i_hi] - ys[i_lo])
            if span > dmax:
                dmax = span
                best = (int(ids[i_lo]), int(ids[i_hi]), float(yls[i_lo]), float(yls[i_hi]))

        start = end

    if best is None or dmax < float(span_thresh):
        return False, float(dmax), None

    a = unit(approach_world)
    b = unit(binormal_world)

    idx1, idx2, y1_local, y2_local = best

    def endpoint_pass(idxp: int, yloc: float) -> Tuple[bool, float]:
        nvec = normals_world[idxp]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        # 对应侧张开方向：y>0 用 +b；y<0 用 -b
        b_side = (1.0 if yloc >= 0.0 else -1.0) * b

        # 法线朝向不稳定时，先翻转到与 b_side 同半球，保证“对应侧”语义
        if np.dot(nvec, b_side) < 0.0:
            nvec = -nvec

        # 直接用原始法线与对应侧张开方向计算夹角（不做投影）
        cosv = float(np.clip(np.dot(nvec, b_side), -1.0, 1.0))
        ang  = float(np.degrees(np.arccos(cosv)))
        # if ang <= float(angle_thresh_deg):
        #     print(f'ANG: {ang}')
        return (ang <= float(angle_thresh_deg)), ang


    ok1, ang1 = endpoint_pass(idx1, y1_local)
    ok2, ang2 = endpoint_pass(idx2, y2_local)

    pass_norm = (ok1 or ok2)
    return pass_norm, float(dmax), (idx1, idx2)


# ------------------------ filting ------------------------

def filter_grasps(
    all_grasps_world: np.ndarray,
    p_pcd: o3d.geometry.PointCloud,
    hand_params: dict,

    # Layer-1
    eps_scale: float = 1.0,                 # res 的倍数，用于自适应
    collision_slack_scale: float = 0.5,     # 穿透判定的“收缩实体”裕量 = collision_slack_scale * res
    use_radius_crop: bool = True,

    # Layer-2
    min_points_between: int = 5,
    margin_y_between: float = 0.005,
    x_clearance_front: float = 0.0,
    x_clearance_back: float = 0.0,

    # Layer-2.5：通过 1&2 后尝试增大宽度再碰撞检测
    widen_after_layer2: float = 0.015,   # 1.5cm
    enable_widen_after_layer2: bool = True,

    # Layer-3
    span_thresh: float = 0.01,
    angle_thresh_deg: float = 40.0,
    xz_res: float = 0.005,
    min_pts_line: int = 5,
    y_percentile: float = 0.0,
    margin_y_span: float = 0.0,

    # 可选：保底回退
    min_keep: int = 0,
    return_meta:bool = False,

    width_orig_external: Optional[np.ndarray] = None,
    return_width_orig: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """
    输入：
      all_grasps_world: (G,13) = [pos(3), axis(3), approach(3), binormal(3), width(1)]
      p_pcd:            Open3D 点云（建议已包含 normals）
      hand_params:      夹爪参数

    输出：
      filtered_grasps, kept_indices
    """

    if all_grasps_world is None or len(all_grasps_world) == 0:
        return np.empty((0, 13), dtype=float), []

    grasps_mod = np.asarray(all_grasps_world, dtype=float).copy()
    if width_orig_external is None:
        width_orig_all = grasps_mod[:, 12].copy()
    else:
        width_orig_all = np.asarray(width_orig_external, dtype=float).reshape(-1)
        assert width_orig_all.shape[0] == grasps_mod.shape[0], "width_orig_external 必须与 all_grasps_world 行数一致"
        
    p_pcd = ensure_normals(p_pcd, radius=0.01, max_nn=30)
    pts_world = np.asarray(p_pcd.points)
    normals_world = np.asarray(p_pcd.normals)

    if pts_world.ndim != 2 or pts_world.shape[1] != 3 or pts_world.shape[0] < 10:
        return np.empty((0, all_grasps_world.shape[1])), []

    tree = KDTree(pts_world)
    res = estimate_cloud_resolution(pts_world, tree)
    eps = float(eps_scale * res)
    penetration_slack = float(max(0.0, collision_slack_scale) * res)

    fw     = float(hand_params["finger_width"])
    depth  = float(hand_params["hand_depth"])
    finger_h      = float(hand_params["hand_height"])
    palm_t = float(hand_params.get("palm_thickness", fw))
    palm_h      = float(hand_params["palm_height"])
    H = max(finger_h,palm_h)

    keep1, keep2, keep3 = [], [], []

    for i, g in enumerate(grasps_mod):
        pos      = np.asarray(g[0:3], dtype=float)
        axis     = np.asarray(g[3:6], dtype=float)
        approach = np.asarray(g[6:9], dtype=float)
        binormal = np.asarray(g[9:12], dtype=float)
        width    = float(g[12])

        # 构造正交框架
        R = build_orthonormal_frame(approach, binormal, axis)

        # 近邻裁剪
        if use_radius_crop:
            idx_near = crop_indices_near_gripper(tree, pos, width, fw, depth, H, palm_t, extra=eps)
            if idx_near.size == 0:
                continue
            P = pts_world[idx_near]
        else:
            idx_near = np.arange(pts_world.shape[0], dtype=int)
            P = pts_world

        P_local = world_to_local(P, pos, R)

        # -------- Layer-1：无穿透（实体体积）--------
        if not collision_free_gripper_volume(P_local, width, hand_params, penetration_slack=penetration_slack):
            continue
        keep1.append(i)

        # -------- Layer-2：夹缝内有足够点 --------
        has_between, mask_between = has_points_between_fingers(
            P_local, width, hand_params,
            min_points=min_points_between,
            margin_y=margin_y_between,
            x_clearance_front=x_clearance_front,
            x_clearance_back=x_clearance_back
        )
        if not has_between:
            continue
        keep2.append(i)

        # -------- Layer-2.5：尝试两次增大 width（每次 + widen_after_layer2）再碰撞 --------
        orig_width = float(width)
        final_width = orig_width

        if enable_widen_after_layer2 and widen_after_layer2 > 1e-9:
            fw = float(hand_params["finger_width"])
            od = hand_params.get("hand_outer_diameter", None)
            max_aperture = (float(od) - 2.0 * fw) if od is not None and np.isfinite(float(od)) else np.inf

            # 两次尝试：w1 = +Δ，w2 = +2Δ（都要 clip 到 max_aperture）
            w1 = min(orig_width + float(widen_after_layer2), max_aperture)
            w2 = min(orig_width + 2.0 * float(widen_after_layer2), max_aperture)

            # 去掉无效/重复的尝试
            trials = []
            if w1 > orig_width + 1e-6:
                trials.append(w1)
            if w2 > (trials[-1] if trials else orig_width) + 1e-6:
                trials.append(w2)

            if trials:
                # 关键优化：只用“最大宽度”做一次邻域裁剪，后面两次 check 复用同一批点
                w_max = max(trials)

                if use_radius_crop:
                    idx_try = crop_indices_near_gripper(tree, pos, w_max, fw, depth, H, palm_t, extra=eps)
                    if idx_try.size > 0:
                        P_try = pts_world[idx_try]
                        P_local_try = world_to_local(P_try, pos, R)

                        for w_try in trials:
                            ok = collision_free_gripper_volume(
                                P_local_try, w_try, hand_params,
                                penetration_slack=penetration_slack
                            )
                            if ok:
                                final_width = w_try
                            else:
                                break  # 更大只会更容易碰撞，直接停
                else:
                    # 不裁剪时直接复用全局 P_local
                    for w_try in trials:
                        ok = collision_free_gripper_volume(
                            P_local, w_try, hand_params,
                            penetration_slack=penetration_slack
                        )
                        if ok:
                            final_width = w_try
                        else:
                            break

        # 写回最终宽度，后续 Layer-3 用它
        grasps_mod[i, 12] = final_width
        width = final_width

        # -------- Layer-3：最大d + 法线投影角度 --------
        # 注意：Layer-3 同样应该用“最终 width”对应的邻域点，避免裁剪半径偏小
        if use_radius_crop:
            idx_near3 = crop_indices_near_gripper(tree, pos, width, fw, depth, H, palm_t, extra=eps)
            if idx_near3.size == 0:
                continue
            P3 = pts_world[idx_near3]
            P_local3 = world_to_local(P3, pos, R)
            idx_full3 = np.asarray(idx_near3, dtype=int)
        else:
            P_local3 = P_local
            idx_full3 = np.asarray(idx_near, dtype=int)

        pass3, dmax, ends = best_span_and_normal_projection_check(
            P_local=P_local3,
            idx_full=idx_full3,
            normals_world=normals_world,
            approach_world=approach,
            binormal_world=binormal,
            width=width,
            hand_params=hand_params,
            span_thresh=span_thresh,
            angle_thresh_deg=angle_thresh_deg,
            xz_res=xz_res,
            min_pts_line=min_pts_line,
            y_percentile=y_percentile,
            margin_y=margin_y_span,
            x_clearance_front=x_clearance_front,
            x_clearance_back=x_clearance_back
        )
        if not pass3:
            # 如果你希望：增宽通过碰撞，但 layer3 不过时回退原宽度再试一次 layer3，也能加一个 fallback
            continue

        keep3.append(i)


    cprint(f'after1:{len(keep1)},after2:{len(keep2)},after3:{len(keep3)}')

    keep3_ids = np.asarray(keep3, dtype=int)
    keep3_grasps = grasps_mod[keep3_ids] if keep3_ids.size > 0 else np.empty((0, 13), dtype=float)

    if len(keep3) >= min_keep:
        cprint("Label_3 pass",'green')
        final_ids = keep3
        stage_tag = 3
    elif len(keep2) >= 30:
        cprint("Label_2 pass",'green')
        final_ids = keep2
        stage_tag = 2
    else:
        cprint("no pass,remain",'red')
        final_ids = []
        stage_tag = 0

    if len(final_ids) == 0:
        if return_meta:
            if return_width_orig:
                keep3_w0 = width_orig_all[keep3_ids] if (keep3_ids is not None and len(keep3_ids) > 0) else np.empty((0,), float)
                return (np.empty((0, all_grasps_world.shape[1])), [], stage_tag,
                        keep3_grasps, keep3_ids.tolist(),
                        np.empty((0,), float), keep3_w0)
            return np.empty((0, all_grasps_world.shape[1])), [], stage_tag, keep3_grasps, keep3_ids.tolist()
        return np.empty((0, all_grasps_world.shape[1])), []


    final_ids = np.asarray(final_ids, dtype=int)
    final_grasps = grasps_mod[final_ids]

    if return_meta:
        if return_width_orig:
            final_w0 = width_orig_all[final_ids]
            keep3_w0 = width_orig_all[keep3_ids] if (keep3_ids is not None and len(keep3_ids) > 0) else np.empty((0,), float)
            return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist(), final_w0, keep3_w0
        return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist()

# --------------filter fast--------------------------------
def has_points_between_fingers_count(P_local: np.ndarray,
                                     width: float,
                                     hand_params: dict,
                                     min_points: int = 5,
                                     margin_y: float = 0.005,
                                     x_clearance_front: float = 0.0,
                                     x_clearance_back: float = 0.0) -> bool:
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", float(hand_params["finger_width"])))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    mask = (
        (x >= x_min) & (x <= x_max) &
        (np.abs(y) <= y_lim) &
        (np.abs(z) <= finger_h / 2.0)
    )
    return (np.count_nonzero(mask) >= int(min_points))


def best_span_and_normal_projection_check_fast(P_local: np.ndarray,
                                               idx_full: np.ndarray,
                                               normals_world: np.ndarray,
                                               approach_world: np.ndarray,
                                               binormal_world: np.ndarray,
                                               width: float,
                                               hand_params: dict,
                                               span_thresh: float = 0.01,
                                               angle_thresh_deg: float = 40.0,
                                               xz_res: float = 0.005,
                                               min_pts_line: int = 5,
                                               y_percentile: float = 0.0,
                                               margin_y: float = 0.0,
                                               x_clearance_front: float = 0.0,
                                               x_clearance_back: float = 0.0):
    fw       = float(hand_params["finger_width"])
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", fw))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    mask = (
        (x >= x_min) & (x <= x_max) &
        (np.abs(z) <= finger_h / 2.0) &
        (np.abs(y) <= y_lim)
    )
    if not np.any(mask):
        return False, 0.0, None

    xm = x[mask]
    ym = y[mask]
    zm = z[mask]
    ids_full = idx_full[mask]
    y_local = ym

    # 分桶 key（与原逻辑一致）
    ix = np.floor((xm - x_min) / float(xz_res)).astype(np.int32)
    iz = np.floor((zm + finger_h / 2.0) / float(xz_res)).astype(np.int32)
    key = (ix.astype(np.int64) * 100000) + iz.astype(np.int64)

    order = np.argsort(key)
    key_s = key[order]
    y_s   = ym[order]
    yl_s  = y_local[order]
    id_s  = ids_full[order]

    n = key_s.shape[0]
    if n == 0:
        return False, 0.0, None

    # 用边界切分替代 while 聚类
    cuts = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate(([0], cuts))
    ends   = np.concatenate((cuts, [n]))

    dmax = 0.0
    best = None  # (idx_lo, idx_hi, ylo_local, yhi_local)

    for s, e in zip(starts, ends):
        cnt = e - s
        if cnt < int(min_pts_line):
            continue

        ys  = y_s[s:e]
        yls = yl_s[s:e]
        ids = id_s[s:e]

        if y_percentile > 0.0:
            lo = np.percentile(ys, y_percentile)
            hi = np.percentile(ys, 100.0 - y_percentile)
            i_lo = int(np.argmin(np.abs(ys - lo)))
            i_hi = int(np.argmin(np.abs(ys - hi)))
        else:
            i_lo = int(np.argmin(ys))
            i_hi = int(np.argmax(ys))

        span = float(ys[i_hi] - ys[i_lo])
        if span > dmax:
            dmax = span
            best = (int(ids[i_lo]), int(ids[i_hi]), float(yls[i_lo]), float(yls[i_hi]))

    if best is None or dmax < float(span_thresh):
        return False, float(dmax), None

    a = unit(approach_world)
    b = unit(binormal_world)

    idx1, idx2, y1_local, y2_local = best

    def endpoint_pass(idxp: int, yloc: float):
        nvec = normals_world[idxp]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        b_side = (1.0 if yloc >= 0.0 else -1.0) * b
        if np.dot(nvec, b_side) < 0.0:
            nvec = -nvec

        cosv = float(np.clip(np.dot(nvec, b_side), -1.0, 1.0))
        ang  = float(np.degrees(np.arccos(cosv)))
        return (ang <= float(angle_thresh_deg)), ang

    ok1, _ = endpoint_pass(idx1, y1_local)
    ok2, _ = endpoint_pass(idx2, y2_local)

    pass_norm = (ok1 or ok2)
    return pass_norm, float(dmax), (idx1, idx2)

def best_span_and_normal_projection_check_nosort(
    P_local: np.ndarray,
    idx_full: np.ndarray,
    normals_world: np.ndarray,
    approach_world: np.ndarray,
    binormal_world: np.ndarray,
    width: float,
    hand_params: dict,
    span_thresh: float = 0.01,
    angle_thresh_deg: float = 40.0,
    xz_res: float = 0.005,
    min_pts_line: int = 5,
    y_percentile: float = 0.0,   # 注意：>0 会回退到原实现
    margin_y: float = 0.0,
    x_clearance_front: float = 0.0,
    x_clearance_back: float = 0.0
):
    # 为了不影响行为：当 y_percentile>0 时，直接回退到你已有的排序实现
    if y_percentile > 0.0:
        return best_span_and_normal_projection_check_fast(
            P_local, idx_full, normals_world,
            approach_world, binormal_world,
            width, hand_params,
            span_thresh=span_thresh,
            angle_thresh_deg=angle_thresh_deg,
            xz_res=xz_res,
            min_pts_line=min_pts_line,
            y_percentile=y_percentile,
            margin_y=margin_y,
            x_clearance_front=x_clearance_front,
            x_clearance_back=x_clearance_back
        )

    fw       = float(hand_params["finger_width"])
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", fw))

    x = P_local[:, 0]
    y = P_local[:, 1]
    z = P_local[:, 2]

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    mask = (
        (x >= x_min) & (x <= x_max) &
        (np.abs(z) <= finger_h / 2.0) &
        (np.abs(y) <= y_lim)
    )
    if not np.any(mask):
        return False, 0.0, None

    xm = x[mask]
    ym = y[mask]
    zm = z[mask]
    ids_full = idx_full[mask]
    y_local = ym

    # 构造离散网格维度（很小，典型几十到几百个 bin）
    ix_max = int(np.floor((x_max - x_min) / float(xz_res))) + 1
    iz_max = int(np.floor((finger_h) / float(xz_res))) + 1
    if ix_max <= 0 or iz_max <= 0:
        return False, 0.0, None
    n_bins = ix_max * iz_max

    ix = np.floor((xm - x_min) / float(xz_res)).astype(np.int32)
    iz = np.floor((zm + finger_h / 2.0) / float(xz_res)).astype(np.int32)

    # 处理边界数值误差
    valid = (ix >= 0) & (ix < ix_max) & (iz >= 0) & (iz < iz_max)
    if not np.any(valid):
        return False, 0.0, None

    ix = ix[valid]
    iz = iz[valid]
    ym = ym[valid]
    y_local = y_local[valid]
    ids_full = ids_full[valid]

    bin_id = (ix.astype(np.int64) * iz_max) + iz.astype(np.int64)

    counts = np.bincount(bin_id, minlength=n_bins)
    # min/max y per bin
    min_y = np.full((n_bins,), np.inf, dtype=np.float64)
    max_y = np.full((n_bins,), -np.inf, dtype=np.float64)
    np.minimum.at(min_y, bin_id, ym)
    np.maximum.at(max_y, bin_id, ym)

    valid_bins = (counts >= int(min_pts_line)) & np.isfinite(min_y) & np.isfinite(max_y)
    if not np.any(valid_bins):
        return False, 0.0, None

    spans = (max_y - min_y)
    spans[~valid_bins] = 0.0

    dmax = float(np.max(spans))
    if dmax < float(span_thresh):
        return False, dmax, None

    bin_best = int(np.argmax(spans))

    # 只在“最佳 bin”上取端点（成本很低）
    sel = (bin_id == bin_best)
    ys  = ym[sel]
    yls = y_local[sel]
    ids = ids_full[sel]

    i_lo = int(np.argmin(ys))
    i_hi = int(np.argmax(ys))

    idx1 = int(ids[i_lo]); y1_local = float(yls[i_lo])
    idx2 = int(ids[i_hi]); y2_local = float(yls[i_hi])

    b = unit(binormal_world)

    def endpoint_pass(idxp: int, yloc: float):
        nvec = normals_world[idxp]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)
        b_side = (1.0 if yloc >= 0.0 else -1.0) * b
        if np.dot(nvec, b_side) < 0.0:
            nvec = -nvec
        cosv = float(np.clip(np.dot(nvec, b_side), -1.0, 1.0))
        ang  = float(np.degrees(np.arccos(cosv)))
        return (ang <= float(angle_thresh_deg)), ang

    ok1, _ = endpoint_pass(idx1, y1_local)
    ok2, _ = endpoint_pass(idx2, y2_local)

    return (ok1 or ok2), dmax, (idx1, idx2)

from dataclasses import dataclass
@dataclass
class FilterContext:
    pts_world: np.ndarray
    normals_world: np.ndarray
    tree: KDTree
    res: float


def build_filter_context(
    p_pcd: o3d.geometry.PointCloud,
    normal_radius: float = 0.01,
    normal_max_nn: int = 30,
) -> FilterContext:
    """
    为 filter_grasps_fast 预构建上下文，供多次调用复用。
    不改变结果，只减少重复开销。
    """
    if (not p_pcd.has_normals()) or (len(p_pcd.normals) != len(p_pcd.points)):
        p_pcd = ensure_normals(p_pcd, radius=normal_radius, max_nn=normal_max_nn)

    pts_world = np.asarray(p_pcd.points)
    normals_world = np.asarray(p_pcd.normals)

    if pts_world.ndim != 2 or pts_world.shape[1] != 3 or pts_world.shape[0] < 10:
        raise ValueError("build_filter_context: 点云无效或点数过少")

    tree = KDTree(pts_world)
    res = estimate_cloud_resolution(pts_world, tree)

    return FilterContext(
        pts_world=pts_world,
        normals_world=normals_world,
        tree=tree,
        res=float(res),
    )

def world_to_local_xyz(P_world: np.ndarray, R: np.ndarray, posR: np.ndarray):
    """
    等价于:
        P_local = (P_world - pos) @ R
    但不显式构造 (N,3) 的 P_local，只返回三列 x/y/z。

    其中 posR = pos @ R
    """
    x = P_world @ R[:, 0] - posR[0]
    y = P_world @ R[:, 1] - posR[1]
    z = P_world @ R[:, 2] - posR[2]
    return x, y, z

def collision_free_gripper_volume_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    width: float,
    hand_params: dict,
    penetration_slack: float = 0.0,
    abs_y: Optional[np.ndarray] = None,
    abs_z: Optional[np.ndarray] = None,
) -> bool:
    """
    与 collision_free_gripper_volume(P_local, ...) 逻辑一致，
    输入改成局部坐标三列 x/y/z。
    """
    fw       = float(hand_params["finger_width"])
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", fw))
    palm_h   = float(hand_params["palm_height"])

    s = float(max(0.0, penetration_slack))

    if abs_y is None:
        abs_y = np.abs(y)
    if abs_z is None:
        abs_z = np.abs(z)

    # palm
    in_palm = (
        (x >= -palm_t / 2.0 + s) & (x <= +palm_t / 2.0 - s) &
        (abs_y <= (width / 2.0 + fw - s)) &
        (abs_z <= (palm_h / 2.0 - s))
    )

    # fingers
    x_f_min = palm_t / 2.0 + s
    x_f_max = palm_t / 2.0 + depth - s

    in_f1 = (
        (x >= x_f_min) & (x <= x_f_max) &
        (y >= (width / 2.0 + s)) & (y <= (width / 2.0 + fw - s)) &
        (abs_z <= (finger_h / 2.0 - s))
    )
    in_f2 = (
        (x >= x_f_min) & (x <= x_f_max) &
        (y >= (-(width / 2.0 + fw) + s)) & (y <= (-(width / 2.0) - s)) &
        (abs_z <= (finger_h / 2.0 - s))
    )

    penetrates = bool(np.any(in_palm | in_f1 | in_f2))
    return (not penetrates)

def has_points_between_fingers_count_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    width: float,
    hand_params: dict,
    min_points: int = 5,
    margin_y: float = 0.005,
    x_clearance_front: float = 0.0,
    x_clearance_back: float = 0.0,
    abs_y: Optional[np.ndarray] = None,
    abs_z: Optional[np.ndarray] = None,
) -> bool:
    """
    与 has_points_between_fingers_count(P_local, ...) 逻辑一致，
    输入改成局部坐标三列 x/y/z。
    """
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", float(hand_params["finger_width"])))

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    if abs_y is None:
        abs_y = np.abs(y)
    if abs_z is None:
        abs_z = np.abs(z)

    mask = (
        (x >= x_min) & (x <= x_max) &
        (abs_y <= y_lim) &
        (abs_z <= finger_h / 2.0)
    )
    return (np.count_nonzero(mask) >= int(min_points))

def best_span_and_normal_projection_check_nosort_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    idx_full: np.ndarray,
    normals_world: np.ndarray,
    approach_world: np.ndarray,
    binormal_world: np.ndarray,
    width: float,
    hand_params: dict,
    span_thresh: float = 0.01,
    angle_thresh_deg: float = 40.0,
    xz_res: float = 0.005,
    min_pts_line: int = 5,
    y_percentile: float = 0.0,
    margin_y: float = 0.0,
    x_clearance_front: float = 0.0,
    x_clearance_back: float = 0.0,
    abs_y: Optional[np.ndarray] = None,
    abs_z: Optional[np.ndarray] = None,
):
    """
    与 best_span_and_normal_projection_check_nosort(...) 等价，
    输入改成局部坐标三列 x/y/z。
    """

    fw       = float(hand_params["finger_width"])
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", fw))

    x_min = palm_t / 2.0 + float(x_clearance_front)
    x_max = palm_t / 2.0 + depth - float(x_clearance_back)
    y_lim = max(0.0, width / 2.0 - float(margin_y))

    if abs_y is None:
        abs_y = np.abs(y)
    if abs_z is None:
        abs_z = np.abs(z)

    mask = (
        (x >= x_min) & (x <= x_max) &
        (abs_z <= finger_h / 2.0) &
        (abs_y <= y_lim)
    )
    if not np.any(mask):
        return False, 0.0, None

    xm = x[mask]
    ym = y[mask]
    zm = z[mask]
    ids_full = idx_full[mask]
    y_local = ym

    # y_percentile > 0 时，保持原排序逻辑
    if y_percentile > 0.0:
        ix = np.floor((xm - x_min) / float(xz_res)).astype(np.int32)
        iz = np.floor((zm + finger_h / 2.0) / float(xz_res)).astype(np.int32)
        key = (ix.astype(np.int64) * 100000) + iz.astype(np.int64)

        order = np.argsort(key)
        key_s = key[order]
        y_s   = ym[order]
        yl_s  = y_local[order]
        id_s  = ids_full[order]

        n = key_s.shape[0]
        if n == 0:
            return False, 0.0, None

        cuts = np.flatnonzero(np.diff(key_s)) + 1
        starts = np.concatenate(([0], cuts))
        ends   = np.concatenate((cuts, [n]))

        dmax = 0.0
        best = None

        for s, e in zip(starts, ends):
            cnt = e - s
            if cnt < int(min_pts_line):
                continue

            ys  = y_s[s:e]
            yls = yl_s[s:e]
            ids = id_s[s:e]

            lo = np.percentile(ys, y_percentile)
            hi = np.percentile(ys, 100.0 - y_percentile)
            i_lo = int(np.argmin(np.abs(ys - lo)))
            i_hi = int(np.argmin(np.abs(ys - hi)))

            span = float(ys[i_hi] - ys[i_lo])
            if span > dmax:
                dmax = span
                best = (int(ids[i_lo]), int(ids[i_hi]), float(yls[i_lo]), float(yls[i_hi]))

        if best is None or dmax < float(span_thresh):
            return False, float(dmax), None

        idx1, idx2, y1_local, y2_local = best

    else:
        ix_max = int(np.floor((x_max - x_min) / float(xz_res))) + 1
        iz_max = int(np.floor((finger_h) / float(xz_res))) + 1
        if ix_max <= 0 or iz_max <= 0:
            return False, 0.0, None

        ix = np.floor((xm - x_min) / float(xz_res)).astype(np.int32)
        iz = np.floor((zm + finger_h / 2.0) / float(xz_res)).astype(np.int32)

        valid = (ix >= 0) & (ix < ix_max) & (iz >= 0) & (iz < iz_max)
        if not np.any(valid):
            return False, 0.0, None

        ix = ix[valid]
        iz = iz[valid]
        ym2 = ym[valid]
        y_local2 = y_local[valid]
        ids_full2 = ids_full[valid]

        bin_id = (ix.astype(np.int64) * iz_max) + iz.astype(np.int64)
        n_bins = ix_max * iz_max

        counts = np.bincount(bin_id, minlength=n_bins)

        min_y = np.full((n_bins,), np.inf, dtype=np.float64)
        max_y = np.full((n_bins,), -np.inf, dtype=np.float64)
        np.minimum.at(min_y, bin_id, ym2)
        np.maximum.at(max_y, bin_id, ym2)

        valid_bins = (counts >= int(min_pts_line)) & np.isfinite(min_y) & np.isfinite(max_y)
        if not np.any(valid_bins):
            return False, 0.0, None

        spans = (max_y - min_y)
        spans[~valid_bins] = 0.0

        dmax = float(np.max(spans))
        if dmax < float(span_thresh):
            return False, dmax, None

        bin_best = int(np.argmax(spans))
        sel = (bin_id == bin_best)

        ys  = ym2[sel]
        yls = y_local2[sel]
        ids = ids_full2[sel]

        i_lo = int(np.argmin(ys))
        i_hi = int(np.argmax(ys))

        idx1 = int(ids[i_lo]); y1_local = float(yls[i_lo])
        idx2 = int(ids[i_hi]); y2_local = float(yls[i_hi])

    b = unit(binormal_world)

    def endpoint_pass(idxp: int, yloc: float):
        nvec = normals_world[idxp]
        nvec = nvec / (np.linalg.norm(nvec) + 1e-12)

        b_side = (1.0 if yloc >= 0.0 else -1.0) * b
        if np.dot(nvec, b_side) < 0.0:
            nvec = -nvec

        cosv = float(np.clip(np.dot(nvec, b_side), -1.0, 1.0))
        ang  = float(np.degrees(np.arccos(cosv)))
        return (ang <= float(angle_thresh_deg)), ang

    ok1, _ = endpoint_pass(idx1, y1_local)
    ok2, _ = endpoint_pass(idx2, y2_local)

    return (ok1 or ok2), float(dmax), (idx1, idx2)

def filter_grasps_fast(
    all_grasps_world: np.ndarray,
    p_pcd: o3d.geometry.PointCloud,
    hand_params: dict,

    eps_scale: float = 1.0,
    collision_slack_scale: float = 0.5,
    use_radius_crop: bool = True,

    min_points_between: int = 5,
    margin_y_between: float = 0.005,
    x_clearance_front: float = 0.0,
    x_clearance_back: float = 0.0,

    widen_after_layer2: float = 0.015,
    enable_widen_after_layer2: bool = True,

    span_thresh: float = 0.01,
    angle_thresh_deg: float = 40.0,
    xz_res: float = 0.005,
    min_pts_line: int = 5,
    y_percentile: float = 0.0,
    margin_y_span: float = 0.0,

    min_keep: int = 0,
    return_meta: bool = False,

    width_orig_external: Optional[np.ndarray] = None,
    return_width_orig: bool = False,

    return_keep2: bool = False,
    enable_layer3_angle_check: bool = True,

    profile: bool = False,

    filter_ctx: Optional[FilterContext] = None,
):
    t_func0 = time.perf_counter()

    if all_grasps_world is None or len(all_grasps_world) == 0:
        if return_meta:
            stage_tag = 0
            empty_g = np.empty((0, 13), dtype=float)
            empty_ids = []
            if return_width_orig:
                empty_w = np.empty((0,), dtype=float)
                if return_keep2:
                    return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w, empty_g, empty_ids, empty_w
                return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w
            else:
                if return_keep2:
                    return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_g, empty_ids
                return empty_g, empty_ids, stage_tag, empty_g, empty_ids
        return np.empty((0, 13), dtype=float), []

    grasps_mod = np.asarray(all_grasps_world, dtype=float).copy()
    N = grasps_mod.shape[0]

    if width_orig_external is None:
        width_orig_all = grasps_mod[:, 12].copy()
    else:
        width_orig_all = np.asarray(width_orig_external, dtype=float).reshape(-1)
        assert width_orig_all.shape[0] == grasps_mod.shape[0], "width_orig_external 必须与 all_grasps_world 行数一致"

    # ---------- context ----------
    t_ctx0 = time.perf_counter()
    if filter_ctx is None:
        filter_ctx = build_filter_context(p_pcd, normal_radius=0.01, normal_max_nn=30)
    pts_world = filter_ctx.pts_world
    normals_world = filter_ctx.normals_world
    tree = filter_ctx.tree
    res = float(filter_ctx.res)
    t_ctx1 = time.perf_counter()
    # ----------------------------

    if pts_world.ndim != 2 or pts_world.shape[1] != 3 or pts_world.shape[0] < 10:
        if return_meta:
            stage_tag = 0
            empty_g = np.empty((0, grasps_mod.shape[1]), dtype=float)
            empty_ids = []
            if return_width_orig:
                empty_w = np.empty((0,), dtype=float)
                if return_keep2:
                    return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w, empty_g, empty_ids, empty_w
                return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w
            else:
                if return_keep2:
                    return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_g, empty_ids
                return empty_g, empty_ids, stage_tag, empty_g, empty_ids
        return np.empty((0, grasps_mod.shape[1]), dtype=float), []

    eps = float(eps_scale * res)
    penetration_slack = float(max(0.0, collision_slack_scale) * res)

    fw       = float(hand_params["finger_width"])
    depth    = float(hand_params["hand_depth"])
    finger_h = float(hand_params["hand_height"])
    palm_t   = float(hand_params.get("palm_thickness", fw))
    palm_h   = float(hand_params["palm_height"])
    H = max(finger_h, palm_h)

    keep1, keep2, keep3 = [], [], []

    od = hand_params.get("hand_outer_diameter", None)
    max_aperture = (float(od) - 2.0 * fw) if od is not None and np.isfinite(float(od)) else np.inf

    # ---------------- grasp 切分 ----------------
    pos_all      = grasps_mod[:, 0:3]
    axis_all     = grasps_mod[:, 3:6]
    approach_all = grasps_mod[:, 6:9]
    binormal_all = grasps_mod[:, 9:12]
    width0_all   = grasps_mod[:, 12].copy()

    # ---------------- 预计算 frame ----------------
    t_pre0 = time.perf_counter()
    R_all = np.empty((N, 3, 3), dtype=float)
    for i in range(N):
        R_all[i] = build_orthonormal_frame(
            approach_all[i],
            binormal_all[i],
            axis_all[i]
        )
    posR_all = np.einsum("ni,nij->nj", pos_all, R_all)
    t_pre1 = time.perf_counter()

    # ---------------- 预计算增宽 ----------------
    if enable_widen_after_layer2 and widen_after_layer2 > 1e-9:
        w1_all = np.minimum(width0_all + float(widen_after_layer2), max_aperture)
        w2_all = np.minimum(width0_all + 2.0 * float(widen_after_layer2), max_aperture)

        has_w1 = w1_all > (width0_all + 1e-6)
        has_w2 = w2_all > (np.maximum(w1_all, width0_all) + 1e-6)

        w_crop_all = width0_all.copy()
        w_crop_all = np.where(has_w1, np.maximum(w_crop_all, w1_all), w_crop_all)
        w_crop_all = np.where(has_w2, np.maximum(w_crop_all, w2_all), w_crop_all)
    else:
        w1_all = width0_all.copy()
        w2_all = width0_all.copy()
        has_w1 = np.zeros((N,), dtype=bool)
        has_w2 = np.zeros((N,), dtype=bool)
        w_crop_all = width0_all.copy()

    # ---------------- 批量半径查询 ----------------
    t_pre2 = time.perf_counter()
    if use_radius_crop:
        x_max = palm_t / 2.0 + depth
        z_max = H / 2.0
        y_max_all = w_crop_all / 2.0 + fw
        r_all = np.sqrt(x_max * x_max + y_max_all * y_max_all + z_max * z_max) + eps
        idx_near_all = tree.query_radius(pos_all, r=r_all)
        full_idx = None
    else:
        idx_near_all = None
        full_idx = np.arange(pts_world.shape[0], dtype=int)
    t_pre3 = time.perf_counter()

    # ---------------- profile ----------------
    t_ctx = (t_ctx1 - t_ctx0)
    t_pre = (t_pre1 - t_pre0) + (t_pre3 - t_pre2)

    t_frame = 0.0
    t_crop = 0.0
    t_transform = 0.0
    t_l1 = 0.0
    t_l2 = 0.0
    t_l25 = 0.0
    t_l3 = 0.0

    n_total = 0
    n_crop_nonempty = 0
    n_pass_l1 = 0
    n_pass_l2 = 0
    n_pass_l3 = 0
    # ----------------------------------------

    for i in range(N):
        n_total += 1

        t0 = time.perf_counter()

        pos      = pos_all[i]
        axis     = axis_all[i]
        approach = approach_all[i]
        binormal = binormal_all[i]
        width0   = float(width0_all[i])
        R        = R_all[i]
        posR     = posR_all[i]

        t1 = time.perf_counter()
        t_frame += (t1 - t0)

        # crop
        t2 = time.perf_counter()
        if use_radius_crop:
            idx_near = np.asarray(idx_near_all[i], dtype=int)
            if idx_near.size == 0:
                t3 = time.perf_counter()
                t_crop += (t3 - t2)
                continue
        else:
            idx_near = full_idx
        t3 = time.perf_counter()
        t_crop += (t3 - t2)
        n_crop_nonempty += 1

        idx_full = idx_near

        # local xyz
        t4 = time.perf_counter()
        P = pts_world[idx_near]
        x, y, z = world_to_local_xyz(P, R, posR)
        abs_y = np.abs(y)
        abs_z = np.abs(z)
        t5 = time.perf_counter()
        t_transform += (t5 - t4)

        # L1
        t6 = time.perf_counter()
        ok_l1 = collision_free_gripper_volume_xyz(
            x, y, z,
            width0,
            hand_params,
            penetration_slack=penetration_slack,
            abs_y=abs_y,
            abs_z=abs_z,
        )
        t7 = time.perf_counter()
        t_l1 += (t7 - t6)

        if not ok_l1:
            continue
        keep1.append(i)
        n_pass_l1 += 1

        # L2
        t8 = time.perf_counter()
        ok_l2 = has_points_between_fingers_count_xyz(
            x, y, z,
            width0,
            hand_params,
            min_points=min_points_between,
            margin_y=margin_y_between,
            x_clearance_front=x_clearance_front,
            x_clearance_back=x_clearance_back,
            abs_y=abs_y,
            abs_z=abs_z,
        )
        t9 = time.perf_counter()
        t_l2 += (t9 - t8)

        if not ok_l2:
            continue
        keep2.append(i)
        n_pass_l2 += 1

        # L2.5
        t10 = time.perf_counter()
        final_width = width0

        if has_w1[i]:
            w_try = float(w1_all[i])
            ok = collision_free_gripper_volume_xyz(
                x, y, z,
                w_try,
                hand_params,
                penetration_slack=penetration_slack,
                abs_y=abs_y,
                abs_z=abs_z,
            )
            if ok:
                final_width = w_try

                if has_w2[i]:
                    w_try2 = float(w2_all[i])
                    ok2 = collision_free_gripper_volume_xyz(
                        x, y, z,
                        w_try2,
                        hand_params,
                        penetration_slack=penetration_slack,
                        abs_y=abs_y,
                        abs_z=abs_z,
                    )
                    if ok2:
                        final_width = w_try2

        grasps_mod[i, 12] = final_width
        t11 = time.perf_counter()
        t_l25 += (t11 - t10)

        # L3
        if enable_layer3_angle_check:
            t12 = time.perf_counter()
            pass3, dmax, ends = best_span_and_normal_projection_check_nosort_xyz(
                x=x,
                y=y,
                z=z,
                idx_full=idx_full,
                normals_world=normals_world,
                approach_world=approach,
                binormal_world=binormal,
                width=final_width,
                hand_params=hand_params,
                span_thresh=span_thresh,
                angle_thresh_deg=angle_thresh_deg,
                xz_res=xz_res,
                min_pts_line=min_pts_line,
                y_percentile=y_percentile,
                margin_y=margin_y_span,
                x_clearance_front=x_clearance_front,
                x_clearance_back=x_clearance_back,
                abs_y=abs_y,
                abs_z=abs_z,
            )
            t13 = time.perf_counter()
            t_l3 += (t13 - t12)

            if not pass3:
                continue
        else:
            pass3 = True

        keep3.append(i)
        n_pass_l3 += 1

    if profile:
        t_func1 = time.perf_counter()
        print("\n[filter_grasps_fast profile]")
        print(f"total function time : {t_func1 - t_func0:.6f}s")
        print(f"total grasps        : {n_total}")
        print(f"nonempty crop       : {n_crop_nonempty}")
        print(f"pass L1             : {n_pass_l1}")
        print(f"pass L2             : {n_pass_l2}")
        print(f"pass L3             : {n_pass_l3}")
        print(f"context build       : {t_ctx:.6f}s")
        print(f"precompute          : {t_pre:.6f}s")
        print(f"frame build         : {t_frame:.6f}s")
        print(f"crop/query          : {t_crop:.6f}s")
        print(f"transform           : {t_transform:.6f}s")
        print(f"layer1              : {t_l1:.6f}s")
        print(f"layer2              : {t_l2:.6f}s")
        print(f"layer2.5            : {t_l25:.6f}s")
        print(f"layer3              : {t_l3:.6f}s")
        if n_total > 0:
            print(f"avg / grasp         : {(t_func1 - t_func0) / n_total:.6f}s")
        print("")

    cprint(f'after1:{len(keep1)},after2:{len(keep2)},after3:{len(keep3)}')

    keep3_ids = np.asarray(keep3, dtype=int)
    keep3_grasps = grasps_mod[keep3_ids] if keep3_ids.size > 0 else np.empty((0, 13), dtype=float)

    keep2_ids = np.asarray(keep2, dtype=int)
    keep2_grasps = grasps_mod[keep2_ids] if keep2_ids.size > 0 else np.empty((0, 13), dtype=float)

    if len(keep3) >= min_keep:
        cprint("Label_3 pass", 'green')
        final_ids = keep3
        stage_tag = 3 if enable_layer3_angle_check else 2
    elif len(keep2) >= 50:
        cprint("Label_2 pass", 'green')
        final_ids = keep2
        stage_tag = 2
    else:
        cprint("no pass,remain", 'red')
        final_ids = []
        stage_tag = 0

    if len(final_ids) == 0:
        if return_meta:
            if return_width_orig:
                keep3_w0 = width_orig_all[keep3_ids] if keep3_ids.size > 0 else np.empty((0,), float)
                keep2_w0 = width_orig_all[keep2_ids] if keep2_ids.size > 0 else np.empty((0,), float)
                if return_keep2:
                    return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
                            keep3_grasps, keep3_ids.tolist(),
                            np.empty((0,), float), keep3_w0,
                            keep2_grasps, keep2_ids.tolist(), keep2_w0)
                return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
                        keep3_grasps, keep3_ids.tolist(),
                        np.empty((0,), float), keep3_w0)

            if return_keep2:
                return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
                        keep3_grasps, keep3_ids.tolist(),
                        keep2_grasps, keep2_ids.tolist())
            return np.empty((0, grasps_mod.shape[1])), [], stage_tag, keep3_grasps, keep3_ids.tolist()

        return np.empty((0, grasps_mod.shape[1])), []

    final_ids = np.asarray(final_ids, dtype=int)
    final_grasps = grasps_mod[final_ids]

    if return_meta:
        if return_width_orig:
            final_w0 = width_orig_all[final_ids]
            keep3_w0 = width_orig_all[keep3_ids] if keep3_ids.size > 0 else np.empty((0,), float)
            keep2_w0 = width_orig_all[keep2_ids] if keep2_ids.size > 0 else np.empty((0,), float)

            if return_keep2:
                return (final_grasps, final_ids.tolist(), stage_tag,
                        keep3_grasps, keep3_ids.tolist(),
                        final_w0, keep3_w0,
                        keep2_grasps, keep2_ids.tolist(), keep2_w0)

            return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist(), final_w0, keep3_w0

        if return_keep2:
            return (final_grasps, final_ids.tolist(), stage_tag,
                    keep3_grasps, keep3_ids.tolist(),
                    keep2_grasps, keep2_ids.tolist())
        return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist()

    return final_grasps, final_ids.tolist()

# def filter_grasps_fast(
#     all_grasps_world: np.ndarray,
#     p_pcd: o3d.geometry.PointCloud,
#     hand_params: dict,

#     eps_scale: float = 1.0,
#     collision_slack_scale: float = 0.5,
#     use_radius_crop: bool = True,

#     min_points_between: int = 5,
#     margin_y_between: float = 0.005,
#     x_clearance_front: float = 0.0,
#     x_clearance_back: float = 0.0,

#     widen_after_layer2: float = 0.015,
#     enable_widen_after_layer2: bool = True,

#     span_thresh: float = 0.01,
#     angle_thresh_deg: float = 40.0,
#     xz_res: float = 0.005,
#     min_pts_line: int = 5,
#     y_percentile: float = 0.0,
#     margin_y_span: float = 0.0,

#     min_keep: int = 0,
#     return_meta: bool = False,

#     width_orig_external: Optional[np.ndarray] = None,
#     return_width_orig: bool = False,

#     return_keep2: bool = False,
#     enable_layer3_angle_check: bool = True,

#     profile: bool = False,
# ):
#     t_func0 = time.perf_counter()

#     if all_grasps_world is None or len(all_grasps_world) == 0:
#         if return_meta:
#             stage_tag = 0
#             empty_g = np.empty((0, 13), dtype=float)
#             empty_ids = []
#             if return_width_orig:
#                 empty_w = np.empty((0,), dtype=float)
#                 if return_keep2:
#                     return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w, empty_g, empty_ids, empty_w
#                 return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w
#             else:
#                 if return_keep2:
#                     return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_g, empty_ids
#                 return empty_g, empty_ids, stage_tag, empty_g, empty_ids
#         return np.empty((0, 13), dtype=float), []

#     grasps_mod = np.asarray(all_grasps_world, dtype=float).copy()
#     N = grasps_mod.shape[0]

#     if width_orig_external is None:
#         width_orig_all = grasps_mod[:, 12].copy()
#     else:
#         width_orig_all = np.asarray(width_orig_external, dtype=float).reshape(-1)
#         assert width_orig_all.shape[0] == grasps_mod.shape[0], "width_orig_external 必须与 all_grasps_world 行数一致"

#     if (not p_pcd.has_normals()) or (len(p_pcd.normals) != len(p_pcd.points)):
#         p_pcd = ensure_normals(p_pcd, radius=0.01, max_nn=30)

#     pts_world = np.asarray(p_pcd.points)
#     normals_world = np.asarray(p_pcd.normals)

#     if pts_world.ndim != 2 or pts_world.shape[1] != 3 or pts_world.shape[0] < 10:
#         if return_meta:
#             stage_tag = 0
#             empty_g = np.empty((0, grasps_mod.shape[1]), dtype=float)
#             empty_ids = []
#             if return_width_orig:
#                 empty_w = np.empty((0,), dtype=float)
#                 if return_keep2:
#                     return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w, empty_g, empty_ids, empty_w
#                 return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_w, empty_w
#             else:
#                 if return_keep2:
#                     return empty_g, empty_ids, stage_tag, empty_g, empty_ids, empty_g, empty_ids
#                 return empty_g, empty_ids, stage_tag, empty_g, empty_ids
#         return np.empty((0, grasps_mod.shape[1]), dtype=float), []

#     tree = KDTree(pts_world)
#     res = estimate_cloud_resolution(pts_world, tree)
#     eps = float(eps_scale * res)
#     penetration_slack = float(max(0.0, collision_slack_scale) * res)

#     fw       = float(hand_params["finger_width"])
#     depth    = float(hand_params["hand_depth"])
#     finger_h = float(hand_params["hand_height"])
#     palm_t   = float(hand_params.get("palm_thickness", fw))
#     palm_h   = float(hand_params["palm_height"])
#     H = max(finger_h, palm_h)

#     keep1, keep2, keep3 = [], [], []

#     od = hand_params.get("hand_outer_diameter", None)
#     max_aperture = (float(od) - 2.0 * fw) if od is not None and np.isfinite(float(od)) else np.inf

#     # ---------------- grasp 字段预切分 ----------------
#     pos_all      = grasps_mod[:, 0:3]
#     axis_all     = grasps_mod[:, 3:6]
#     approach_all = grasps_mod[:, 6:9]
#     binormal_all = grasps_mod[:, 9:12]
#     width0_all   = grasps_mod[:, 12].copy()

#     # ---------------- 预计算局部坐标系 ----------------
#     t_pre0 = time.perf_counter()
#     R_all = np.empty((N, 3, 3), dtype=float)
#     for i in range(N):
#         R_all[i] = build_orthonormal_frame(
#             approach_all[i],
#             binormal_all[i],
#             axis_all[i]
#         )
#     posR_all = np.einsum("ni,nij->nj", pos_all, R_all)
#     t_pre1 = time.perf_counter()

#     # ---------------- 预计算增宽候选 ----------------
#     if enable_widen_after_layer2 and widen_after_layer2 > 1e-9:
#         w1_all = np.minimum(width0_all + float(widen_after_layer2), max_aperture)
#         w2_all = np.minimum(width0_all + 2.0 * float(widen_after_layer2), max_aperture)

#         has_w1 = w1_all > (width0_all + 1e-6)
#         has_w2 = w2_all > (np.maximum(w1_all, width0_all) + 1e-6)

#         w_crop_all = width0_all.copy()
#         w_crop_all = np.where(has_w1, np.maximum(w_crop_all, w1_all), w_crop_all)
#         w_crop_all = np.where(has_w2, np.maximum(w_crop_all, w2_all), w_crop_all)
#     else:
#         w1_all = width0_all.copy()
#         w2_all = width0_all.copy()
#         has_w1 = np.zeros((N,), dtype=bool)
#         has_w2 = np.zeros((N,), dtype=bool)
#         w_crop_all = width0_all.copy()

#     # ---------------- 批量 query_radius ----------------
#     t_pre2 = time.perf_counter()
#     if use_radius_crop:
#         x_max = palm_t / 2.0 + depth
#         z_max = H / 2.0
#         y_max_all = w_crop_all / 2.0 + fw
#         r_all = np.sqrt(x_max * x_max + y_max_all * y_max_all + z_max * z_max) + eps
#         idx_near_all = tree.query_radius(pos_all, r=r_all)
#         full_idx = None
#     else:
#         idx_near_all = None
#         full_idx = np.arange(pts_world.shape[0], dtype=int)
#     t_pre3 = time.perf_counter()

#     # ---------------- profile ----------------
#     t_pre = (t_pre1 - t_pre0) + (t_pre3 - t_pre2)
#     t_frame = 0.0
#     t_crop = 0.0
#     t_transform = 0.0
#     t_l1 = 0.0
#     t_l2 = 0.0
#     t_l25 = 0.0
#     t_l3 = 0.0

#     n_total = 0
#     n_crop_nonempty = 0
#     n_pass_l1 = 0
#     n_pass_l2 = 0
#     n_pass_l3 = 0
#     # ----------------------------------------

#     for i in range(N):
#         n_total += 1

#         # 0) 取 grasp + 取预计算 frame
#         t0 = time.perf_counter()

#         pos      = pos_all[i]
#         axis     = axis_all[i]
#         approach = approach_all[i]
#         binormal = binormal_all[i]
#         width0   = float(width0_all[i])
#         R        = R_all[i]
#         posR     = posR_all[i]

#         t1 = time.perf_counter()
#         t_frame += (t1 - t0)

#         # 1) 邻域裁剪
#         t2 = time.perf_counter()
#         if use_radius_crop:
#             idx_near = np.asarray(idx_near_all[i], dtype=int)
#             if idx_near.size == 0:
#                 t3 = time.perf_counter()
#                 t_crop += (t3 - t2)
#                 continue
#         else:
#             idx_near = full_idx
#         t3 = time.perf_counter()
#         t_crop += (t3 - t2)
#         n_crop_nonempty += 1

#         idx_full = idx_near

#         # 2) 单次局部坐标变换，但只算 x/y/z
#         t4 = time.perf_counter()
#         P = pts_world[idx_near]
#         x, y, z = world_to_local_xyz(P, R, posR)
#         t5 = time.perf_counter()
#         t_transform += (t5 - t4)

#         # 3) Layer-1
#         t6 = time.perf_counter()
#         ok_l1 = collision_free_gripper_volume_xyz(
#             x, y, z,
#             width0,
#             hand_params,
#             penetration_slack=penetration_slack
#         )
#         t7 = time.perf_counter()
#         t_l1 += (t7 - t6)

#         if not ok_l1:
#             continue
#         keep1.append(i)
#         n_pass_l1 += 1

#         # 4) Layer-2
#         t8 = time.perf_counter()
#         ok_l2 = has_points_between_fingers_count_xyz(
#             x, y, z,
#             width0,
#             hand_params,
#             min_points=min_points_between,
#             margin_y=margin_y_between,
#             x_clearance_front=x_clearance_front,
#             x_clearance_back=x_clearance_back
#         )
#         t9 = time.perf_counter()
#         t_l2 += (t9 - t8)

#         if not ok_l2:
#             continue
#         keep2.append(i)
#         n_pass_l2 += 1

#         # 5) Layer-2.5
#         t10 = time.perf_counter()
#         final_width = width0

#         if has_w1[i]:
#             w_try = float(w1_all[i])
#             ok = collision_free_gripper_volume_xyz(
#                 x, y, z,
#                 w_try,
#                 hand_params,
#                 penetration_slack=penetration_slack
#             )
#             if ok:
#                 final_width = w_try

#                 if has_w2[i]:
#                     w_try2 = float(w2_all[i])
#                     ok2 = collision_free_gripper_volume_xyz(
#                         x, y, z,
#                         w_try2,
#                         hand_params,
#                         penetration_slack=penetration_slack
#                     )
#                     if ok2:
#                         final_width = w_try2

#         grasps_mod[i, 12] = final_width
#         t11 = time.perf_counter()
#         t_l25 += (t11 - t10)

#         # 6) Layer-3
#         if enable_layer3_angle_check:
#             t12 = time.perf_counter()
#             pass3, dmax, ends = best_span_and_normal_projection_check_nosort_xyz(
#                 x=x,
#                 y=y,
#                 z=z,
#                 idx_full=idx_full,
#                 normals_world=normals_world,
#                 approach_world=approach,
#                 binormal_world=binormal,
#                 width=final_width,
#                 hand_params=hand_params,
#                 span_thresh=span_thresh,
#                 angle_thresh_deg=angle_thresh_deg,
#                 xz_res=xz_res,
#                 min_pts_line=min_pts_line,
#                 y_percentile=y_percentile,
#                 margin_y=margin_y_span,
#                 x_clearance_front=x_clearance_front,
#                 x_clearance_back=x_clearance_back
#             )
#             t13 = time.perf_counter()
#             t_l3 += (t13 - t12)

#             if not pass3:
#                 continue
#         else:
#             pass3 = True

#         keep3.append(i)
#         n_pass_l3 += 1

#     if profile:
#         t_func1 = time.perf_counter()
#         print("\n[filter_grasps_fast profile]")
#         print(f"total function time : {t_func1 - t_func0:.6f}s")
#         print(f"total grasps        : {n_total}")
#         print(f"nonempty crop       : {n_crop_nonempty}")
#         print(f"pass L1             : {n_pass_l1}")
#         print(f"pass L2             : {n_pass_l2}")
#         print(f"pass L3             : {n_pass_l3}")
#         print(f"precompute          : {t_pre:.6f}s")
#         print(f"frame build         : {t_frame:.6f}s")
#         print(f"crop/query          : {t_crop:.6f}s")
#         print(f"transform           : {t_transform:.6f}s")
#         print(f"layer1              : {t_l1:.6f}s")
#         print(f"layer2              : {t_l2:.6f}s")
#         print(f"layer2.5            : {t_l25:.6f}s")
#         print(f"layer3              : {t_l3:.6f}s")
#         if n_total > 0:
#             print(f"avg / grasp         : {(t_func1 - t_func0) / n_total:.6f}s")
#         print("")

#     cprint(f'after1:{len(keep1)},after2:{len(keep2)},after3:{len(keep3)}')

#     keep3_ids = np.asarray(keep3, dtype=int)
#     keep3_grasps = grasps_mod[keep3_ids] if keep3_ids.size > 0 else np.empty((0, 13), dtype=float)

#     keep2_ids = np.asarray(keep2, dtype=int)
#     keep2_grasps = grasps_mod[keep2_ids] if keep2_ids.size > 0 else np.empty((0, 13), dtype=float)

#     if len(keep3) >= min_keep:
#         cprint("Label_3 pass", 'green')
#         final_ids = keep3
#         stage_tag = 3 if enable_layer3_angle_check else 2
#     elif len(keep2) >= 50:
#         cprint("Label_2 pass", 'green')
#         final_ids = keep2
#         stage_tag = 2
#     else:
#         cprint("no pass,remain", 'red')
#         final_ids = []
#         stage_tag = 0

#     if len(final_ids) == 0:
#         if return_meta:
#             if return_width_orig:
#                 keep3_w0 = width_orig_all[keep3_ids] if keep3_ids.size > 0 else np.empty((0,), float)
#                 keep2_w0 = width_orig_all[keep2_ids] if keep2_ids.size > 0 else np.empty((0,), float)
#                 if return_keep2:
#                     return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
#                             keep3_grasps, keep3_ids.tolist(),
#                             np.empty((0,), float), keep3_w0,
#                             keep2_grasps, keep2_ids.tolist(), keep2_w0)
#                 return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
#                         keep3_grasps, keep3_ids.tolist(),
#                         np.empty((0,), float), keep3_w0)

#             if return_keep2:
#                 return (np.empty((0, grasps_mod.shape[1])), [], stage_tag,
#                         keep3_grasps, keep3_ids.tolist(),
#                         keep2_grasps, keep2_ids.tolist())
#             return np.empty((0, grasps_mod.shape[1])), [], stage_tag, keep3_grasps, keep3_ids.tolist()

#         return np.empty((0, grasps_mod.shape[1])), []

#     final_ids = np.asarray(final_ids, dtype=int)
#     final_grasps = grasps_mod[final_ids]

#     if return_meta:
#         if return_width_orig:
#             final_w0 = width_orig_all[final_ids]
#             keep3_w0 = width_orig_all[keep3_ids] if keep3_ids.size > 0 else np.empty((0,), float)
#             keep2_w0 = width_orig_all[keep2_ids] if keep2_ids.size > 0 else np.empty((0,), float)

#             if return_keep2:
#                 return (final_grasps, final_ids.tolist(), stage_tag,
#                         keep3_grasps, keep3_ids.tolist(),
#                         final_w0, keep3_w0,
#                         keep2_grasps, keep2_ids.tolist(), keep2_w0)

#             return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist(), final_w0, keep3_w0

#         if return_keep2:
#             return (final_grasps, final_ids.tolist(), stage_tag,
#                     keep3_grasps, keep3_ids.tolist(),
#                     keep2_grasps, keep2_ids.tolist())
#         return final_grasps, final_ids.tolist(), stage_tag, keep3_grasps, keep3_ids.tolist()

#     return final_grasps, final_ids.tolist()



def prefilter_remove_upward_approach(
    grasps: np.ndarray,
    up_world: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    angle_thresh_deg: float = 60.0,
    keep_nan: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    删除 approach 指向 up_world 附近的 grasp。
    grasps: (G,13) [pos, axis, approach, binormal, width]
    返回: (grasps_kept, kept_indices_in_original)
    """
    G = grasps.shape[0]
    if G == 0:
        return grasps, np.empty((0,), dtype=int)

    up = np.asarray(up_world, dtype=float).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)

    a = grasps[:, 6:9].astype(float)                         # approach
    an = np.linalg.norm(a, axis=1) + 1e-12
    a_unit = a / an[:, None]

    cosv = (a_unit @ up)                                     # (G,)
    thr = float(np.cos(np.deg2rad(angle_thresh_deg)))

    # “向上”的定义：cosv > thr
    # 删除向上的 => keep = ~(cosv > thr)
    if keep_nan:
        keep = ~(cosv > thr)
    else:
        keep = (~(cosv > thr)) & np.isfinite(cosv)

    kept_idx = np.nonzero(keep)[0].astype(int)
    return grasps[kept_idx], kept_idx

def select_diverse_grasps_fps(
    grasps: np.ndarray,
    K: int = 100,
    pos_scale: float = 0.02,          # 2cm：位置差达到 2cm 算“显著不同”
    ang_scale_deg: float = 15.0,      # 15°：姿态差达到 15° 算“显著不同”
    width_scale: float = 0.02,        # 2cm：开口差达到 2cm 算“显著不同”
    seed: str = "centroid",           # "centroid" / "random" / "first"
    random_seed: int = 0,
    binormal_flip_equiv: bool = False # True: 认为 b 和 -b 等价（可选）
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 grasps (N,13) 中选出 K 个差异尽量大的 grasp（贪心 FPS / k-center 近似）。
    返回: (selected_grasps, selected_indices_in_input)
    """
    N = grasps.shape[0]
    if N == 0:
        return grasps, np.empty((0,), dtype=int)
    if N <= K:
        return grasps, np.arange(N, dtype=int)

    pos = grasps[:, 0:3].astype(np.float64)
    a = grasps[:, 6:9].astype(np.float64)
    b = grasps[:, 9:12].astype(np.float64)
    w = grasps[:, 12].astype(np.float64)

    # 单位化方向
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b /= (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)

    # 选初始点
    if seed == "centroid":
        c = pos.mean(axis=0)
        seed_idx = int(np.argmin(np.linalg.norm(pos - c[None, :], axis=1)))
    elif seed == "random":
        rng = np.random.RandomState(random_seed)
        seed_idx = int(rng.randint(0, N))
    else:
        seed_idx = 0

    selected = np.empty((K,), dtype=int)
    selected[0] = seed_idx

    # min_dist[i]：点 i 到“已选集合”的最小距离（越大越该被选）
    min_dist = np.full((N,), np.inf, dtype=np.float64)

    # 把选中点标记掉
    chosen_mask = np.zeros((N,), dtype=bool)

    def update_min_dist(j: int):
        # 位置距离
        dp = np.linalg.norm(pos - pos[j], axis=1) / max(pos_scale, 1e-12)

        # 姿态距离：max(angle(a), angle(b))
        ca = np.clip(a @ a[j], -1.0, 1.0)
        ang_a = np.degrees(np.arccos(ca)) / max(ang_scale_deg, 1e-12)

        cb = np.clip(b @ b[j], -1.0, 1.0)
        if binormal_flip_equiv:
            cb = np.abs(cb)  # b 与 -b 视为等价
        ang_b = np.degrees(np.arccos(cb)) / max(ang_scale_deg, 1e-12)

        dang = np.maximum(ang_a, ang_b)

        # 宽度距离
        dw = np.abs(w - w[j]) / max(width_scale, 1e-12)

        d = dp * dp + dang * dang + dw * dw
        # 更新到已选集合的最小距离
        np.minimum(min_dist, d, out=min_dist)

        # 已选点本身不再被选
        min_dist[chosen_mask] = -np.inf

    # 初始化一次
    chosen_mask[seed_idx] = True
    update_min_dist(seed_idx)

    # 逐步选最远点
    for t in range(1, K):
        j = int(np.argmax(min_dist))
        selected[t] = j
        chosen_mask[j] = True
        update_min_dist(j)

    return grasps[selected], selected


# -------------------- spliting --------------------
# def split_pcd_by_color(pcd_path,visualize,is_cluster):
#     pcds = []

#     pcd_ori = o3d.io.read_point_cloud(pcd_path)
#     # pcd_ori.scale(0.001, center=pcd_ori.get_center())
#     pts_ori = np.asarray(pcd_ori.points)
#     # pcd_ori = pcd_ori.voxel_down_sample(voxel_size=voxel_norm)
#     if visualize:
#         o3d.visualization.draw_geometries([pcd_ori])
    
#     pcds.append(pcd_ori)
#     t0 = time.time()
#     if is_cluster:
#         pcd_colored, points_cluster, labels = cluster(pcd_path,0.67)
#         # pcd_colored, points_cluster, labels = cluster(pcd_path,0.6)
#         t1 = time.time()
#         cprint(f'T0:{t1-t0}','red')
#         labels = np.asarray(labels, dtype=np.int64)
#         unique_labs = np.unique(labels)
#         print(f"找到 {len(unique_labs)} 个簇")

#         if len(unique_labs) > 1:
#             for lab in unique_labs:
#                 mask = (labels == lab)
#                 pts_i = pts_ori[mask]
#                 if pts_i.shape[0] == 0:
#                     continue

#                 sub = o3d.geometry.PointCloud()
#                 sub.points = o3d.utility.Vector3dVector(pts_i)

#                 cols_all = np.asarray(pcd_colored.colors)
#                 cols_i = cols_all[mask] if cols_all.shape[0] == pts_ori.shape[0] \
#                         else np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (pts_i.shape[0], 1))
#                 sub.colors = o3d.utility.Vector3dVector(cols_i)
#                 pcds.append(sub)
#     #     else:
#     #         pcds.append(pcd_ori)
#     # else:
#     #     pcds.append(pcd_ori)

#     return pcd_ori,pcds

# def split_obj_by_color(obj_path,pcd_path='/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/obj.pcd'):
#     pcds = []

#     pcd_ori = obj_to_pcd(obj_path,pcd_path)
#     # pcd_ori = pcd_ori.voxel_down_sample(voxel_size=voxel_norm)
#     o3d.visualization.draw_geometries([pcd_ori])
#     pts_ori = np.asarray(pcd_ori.points)
#     pcds.append(pcd_ori)
#     t0 = time.time()
#     pcd_colored, points_cluster, labels = cluster(pcd_path)
#     t1 = time.time()
#     cprint(f'T0:{t1-t0}','red')
#     labels = np.asarray(labels, dtype=np.int64)
#     unique_labs = np.unique(labels)
#     print(f"找到 {len(unique_labs)} 个簇")

#     if len(unique_labs) > 1:
#         for lab in unique_labs:
#             mask = (labels == lab)
#             pts_i = pts_ori[mask]
#             if pts_i.shape[0] == 0:
#                 continue

#             sub = o3d.geometry.PointCloud()
#             sub.points = o3d.utility.Vector3dVector(pts_i)

#             cols_all = np.asarray(pcd_colored.colors)
#             cols_i = cols_all[mask] if cols_all.shape[0] == pts_ori.shape[0] \
#                     else np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (pts_i.shape[0], 1))
#             sub.colors = o3d.utility.Vector3dVector(cols_i)
#             pcds.append(sub)
#     # else:
#     #     pcds.append(pcd_ori)

#     return pcd_ori,pcds
#---------------------normal--------------------
def transform_grasps_from_norm(grasps_n: np.ndarray,
                               center: np.ndarray,) -> np.ndarray:
    """
    grasps_n: (M,13) in normalized P frame
    return  : (M,13) in original P frame
    """
    out = np.asarray(grasps_n, dtype=float).copy()
    out[:, 0:3] = out[:, 0:3]  + center
    # a = out[:, 6:9]
    # a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    # out[:, 0:3] -=  a * 0.005
    # out[:,-1] = out[:,-1] + 0.01
    return out

def convert_all_grasps_pos(all_grasps_world: np.ndarray,
                           hand_params: dict,
                           src: str = "palm",
                           dst: str = "tip") -> np.ndarray:
    """
    批量转换 grasp 的 pos 表示：
      grasp 格式: [pos(3), axis(3), approach(3), binormal(3), width(1)] -> (N,13)

    src/dst:
      - "palm" -> "tip": pos 向 +approach 平移 x_tip
      - "tip"  -> "palm": pos 向 -approach 平移 x_tip
    """
    g = np.asarray(all_grasps_world, dtype=float).copy()
    if g.size == 0:
        return g
    assert g.ndim == 2 and g.shape[1] >= 13, f"expect (N,13), got {g.shape}"

    a = g[:, 6:9]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    palm_t = float(hand_params.get("palm_thickness", hand_params["finger_width"]))
    depth  = float(hand_params["hand_depth"])
    x_tip  = palm_t / 2.0 + depth

    if src == "palm" and dst == "tip":
        g[:, 0:3] += a * x_tip
    elif src == "tip" and dst == "palm":
        g[:, 0:3] -= a * x_tip
    else:
        raise ValueError(f"Unsupported conversion: {src} -> {dst}")

    return g

def obj_to_pcd(obj_path, pcd_path='/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset', voxel_size=0.002):
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"找不到输入文件: {obj_path}")

    # 1. 读取 OBJ 网格
    print(f"[INFO] 读取 OBJ 网格: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if mesh.is_empty():
        raise RuntimeError("读取到的网格为空，请检查 OBJ 文件是否正确。")

    # 2. 网格顶点转为点云
    print("[INFO] 在网格曲面上采样点云")

    # 先保证网格有法线（Poisson 采样需要）
    mesh.compute_vertex_normals()

    # 根据表面积和期望点间距估算采样点数
    target_spacing = 0.002  # 期望的点间距 ≈ 你的体素大小
    area = mesh.get_surface_area()
    n_points = max(int(area / (target_spacing ** 2)), 1000)  # 至少给个下限，避免太少

    print(f"[INFO] 网格表面积: {area:.6f}, 采样点数: {n_points}")

    # Poisson 磁盘采样（点分布更均匀），也可以用 sample_points_uniformly
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=n_points,
        init_factor=5
    )

    # 给点云估计法线，方便后续可视化/处理
    pcd.estimate_normals()
    print(f"[INFO] 采样后点数量: {len(pcd.points)}")

    # 3. 体素降采样
    if voxel_size is not None and voxel_size > 0:
        print(f"[INFO] 进行体素降采样, voxel_size = {voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[INFO] 降采样后点数量: {len(pcd.points)}")

    # 4. 输出路径处理
    if pcd_path is None:
        base, _ = os.path.splitext(obj_path)
        pcd_path = base + f"_voxel{voxel_size:.4f}.pcd"

    # 5. 保存为 PCD
    print(f"[INFO] 保存点云到: {pcd_path}")
    ok = o3d.io.write_point_cloud(pcd_path, pcd)
    if not ok:
        raise RuntimeError("保存 PCD 失败，请检查路径/权限。")
    print("[INFO] 完成。")

    return pcd

def inv_T(T: np.ndarray) -> np.ndarray:
    """Rigid transform inverse: [R t; 0 1]^{-1} = [R^T, -R^T t; 0 1]"""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Rt = R.T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti

def _unit3(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)

def _grasp_bucket_key(g, pos_bin=0.01, dir_bin=0.2, width_bin=0.02):
    """
    粗分桶 key：
    - pos_bin   : 位置量化步长（建议略大于 pos_thr）
    - dir_bin   : 方向向量分桶（[-1,1] 量化）
    - width_bin : 宽度量化
    """
    g = np.asarray(g, dtype=float).reshape(-1)
    p = g[0:3]
    a = _unit3(g[6:9])      # approach
    b = _unit3(g[9:12])     # binormal
    w = float(g[12])

    # 位置量化
    kp = tuple(np.floor(p / float(pos_bin)).astype(np.int32).tolist())

    # 方向量化：把 [-1,1] 平移到 [0,2] 再量化
    def qdir(x):
        return int(np.floor((float(x) + 1.0) / float(dir_bin)))
    ka = (qdir(a[0]), qdir(a[1]), qdir(a[2]))
    kb = (qdir(b[0]), qdir(b[1]), qdir(b[2]))

    kw = int(np.floor(w / float(width_bin)))
    return (kp, ka, kb, kw)

def merge_unique_grasps(pool: np.ndarray,
                                  pool_w0: np.ndarray,
                                  new_grasps: np.ndarray,
                                  new_w0: np.ndarray,
                                  pos_thr=0.005,
                                  rot_thr_deg=5.0,
                                  width_thr=0.01,
                                  max_total=None,
                                  pos_bin=0.01,
                                  dir_bin=0.2,
                                  width_bin=0.02,
                                  search_neighbor_buckets=False,
                                  bucket_state=None):
    """
    合并去重：返回 (pool_new, pool_w0_new, bucket_state)
    - 去重判断仍用 is_similar（用 grasp 里最终 width）
    - pool_w0/new_w0 用来保存“orig width”，与 pool/new_grasps 一一对应
    """
    if bucket_state is None:
        bucket_state = {"buckets": {}}
    buckets = bucket_state.get("buckets", {})
    bucket_state["buckets"] = buckets

    if new_grasps is None or np.asarray(new_grasps).size == 0:
        if pool is None:
            pool = np.empty((0, 13), dtype=float)
        if pool_w0 is None:
            pool_w0 = np.empty((0,), dtype=float)
        return np.asarray(pool, float), np.asarray(pool_w0, float), bucket_state

    new_grasps = np.asarray(new_grasps, dtype=float)
    if new_grasps.ndim == 1:
        new_grasps = new_grasps[None, :]
    new_w0 = np.asarray(new_w0, dtype=float).reshape(-1)
    assert new_w0.shape[0] == new_grasps.shape[0], "new_w0 必须与 new_grasps 行数一致"

    if pool is None or pool.size == 0:
        out = new_grasps.copy()
        out_w0 = new_w0.copy()
        buckets.clear()
        for i in range(out.shape[0]):
            k = _grasp_bucket_key(out[i], pos_bin=pos_bin, dir_bin=dir_bin, width_bin=width_bin)
            buckets.setdefault(k, []).append(i)

        if max_total is not None and out.shape[0] > int(max_total):
            out = out[:int(max_total)]
            out_w0 = out_w0[:int(max_total)]
            buckets.clear()
            for i in range(out.shape[0]):
                k = _grasp_bucket_key(out[i], pos_bin=pos_bin, dir_bin=dir_bin, width_bin=width_bin)
                buckets.setdefault(k, []).append(i)
        return out, out_w0, bucket_state

    pool = np.asarray(pool, dtype=float)
    pool_w0 = np.asarray(pool_w0, dtype=float).reshape(-1)
    assert pool_w0.shape[0] == pool.shape[0], "pool_w0 必须与 pool 行数一致"

    base_n = pool.shape[0]

    # 若 buckets 为空：为 pool 建桶一次
    if len(buckets) == 0:
        for i in range(base_n):
            k = _grasp_bucket_key(pool[i], pos_bin=pos_bin, dir_bin=dir_bin, width_bin=width_bin)
            buckets.setdefault(k, []).append(i)

    def neighbor_keys(k):
        if not search_neighbor_buckets:
            yield k
            return
        (kp, ka, kb, kw) = k
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    kp2 = (kp[0] + dx, kp[1] + dy, kp[2] + dz)
                    yield (kp2, ka, kb, kw)

    add_list = []
    add_w0_list = []

    for g, w0 in zip(new_grasps, new_w0):
        if max_total is not None and (base_n + len(add_list)) >= int(max_total):
            break

        k0 = _grasp_bucket_key(g, pos_bin=pos_bin, dir_bin=dir_bin, width_bin=width_bin)

        cand_indices = []
        for kk in neighbor_keys(k0):
            cand_indices.extend(buckets.get(kk, []))

        dup = False
        for idx in cand_indices:
            if idx < base_n:
                g_ref = pool[idx]
            else:
                g_ref = add_list[idx - base_n]
            if is_similar(g, g_ref, pos_thr=pos_thr, rot_thr_deg=rot_thr_deg, width_thr=width_thr):  # 你原有函数
                dup = True
                break

        if not dup:
            add_list.append(g.copy())
            add_w0_list.append(float(w0))
            new_idx = base_n + len(add_list) - 1
            buckets.setdefault(k0, []).append(new_idx)

    if len(add_list) == 0:
        return pool, pool_w0, bucket_state

    out = np.vstack([pool, np.vstack(add_list)])
    out_w0 = np.concatenate([pool_w0, np.asarray(add_w0_list, dtype=float)], axis=0)
    return out, out_w0, bucket_state

def pmatch(path):
    set_global_seed(0)
    HAND_PARAMS_REAL = {
        "finger_width":        0.015,
        "hand_outer_diameter": 0.167,
        "hand_depth":          0.0475,
        "hand_height":         0.02,
        "palm_thickness":      0.004,
        "palm_height":         0.0334,
    }
    HAND_PARAMS_VISUAL = {
        "finger_width":        0.002,
        "hand_outer_diameter": 0.141,
        "hand_depth":          0.0475,
        "hand_height":         0.002,
        "palm_thickness":      0.002,
        "palm_height":         0.002,
    }
    normal_radius = 0.01
    voxel_norm = 0.005

    # pcd split
    t0 = time.time()
    pcd_p_ori = o3d.io.read_point_cloud(path)
    # pcd_p_ori, _ =  pcd_p_ori.remove_radius_outlier(8,0.006)
    ct = pcd_p_ori.get_center()
    plane = plane_pcd()
    plane.translate([ct[0], ct[1], 0.105], relative=True)
    t1 = time.time()
    cprint(f'Time for partfield:{t1-t0}', 'green')

    pcd_p_ori.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
    plane.normals = o3d.utility.Vector3dVector(plane_n)

    pcd_with_plane = pcd_p_ori + plane
    # o3d.visualization.draw_geometries([pcd_with_plane])

    index_path = os.path.join(
        "/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k",
        "global_index_ems.pkl"
    )
    index_data = load_global_index(index_path)
    index_arrays = build_index_arrays(index_data)

    # ------- (A) 先为每个 part 计算一次 ranked 列表（不要在回溯里重复跑 ems_match）-------
    BATCH = 10
    MAX_CANDS = 80
    parts = []

    pcd_p_o3d_norm = pcd_p_ori.voxel_down_sample(voxel_size=voxel_norm)
    center = pcd_p_o3d_norm.get_center()
    pcd_p_o3d_norm.translate(-center)

    ranked, qems = ems_match_fast(
        query_pcd_norm=pcd_p_o3d_norm,
        idxA=index_arrays,
        top_k_shape=200,
        top_k_scale=MAX_CANDS,
        top_n=min(BATCH, MAX_CANDS),
        w_ratio=1.0, w_eps=1.2,
        lambda_scale=1.6, lambda_abs=2.0,
        hard_scale_tol=None,
        hard_abs_tol=None,
        visualize=False
    )


    parts.append({
        "pcd_norm": pcd_p_o3d_norm,
        "center": center,
        "ranked": ranked,
        "Tq": qems["T_sq"],
    })

    # ------- (B) 回溯轮次：累计 Stage2 通过的 grasp（keep2），注入下一轮一起 filter -------
    stage2_pool = np.empty((0, 13), dtype=float)      # 存最终 width 的 grasp（可能已增宽）
    stage2_w0_pool = np.empty((0,), dtype=float)      # 与 stage2_pool 对齐：存 orig width（增宽前）
    stage2_bucket_state = {"buckets": {}}             # 桶化索引（跨轮复用）
    round_id = 0

    while True:
        start = round_id * BATCH
        end   = start + BATCH

        any_added = False
        round_grasps_world = []

        # -------- (1) 生成本轮新 grasp 候选 --------
        for part in parts:
            entries = part["ranked"][start:end]
            if entries is None or len(entries) == 0:
                continue

            any_added = True

            grasps_list = register_by_superquadric(
                query_pcd_norm=part["pcd_norm"],
                top_entries=entries,
                Tq=part["Tq"],
                n=len(entries),
                voxel=voxel_norm,
                center_mode="keep",
                do_icp=False,
                visualize=False,
                hand_params=HAND_PARAMS_REAL,
            )

            grasps_list = [g for g in grasps_list if isinstance(g, np.ndarray) and g.size > 0]
            if len(grasps_list) == 0:
                continue

            grasps = np.vstack(grasps_list)
            grasps = transform_grasps_from_norm(grasps, part["center"])
            round_grasps_world.append(grasps)

        # ranked 用尽：返回 stage2_pool（有多少算多少）
        if not any_added:
            # print("[Backtrack] ranked exhausted.")
            if stage2_pool.shape[0] > 0:
                filtered_grasps = stage2_pool
                filtered_w0 = stage2_w0_pool if stage2_w0_pool.shape[0] == stage2_pool.shape[0] else stage2_pool[:, 12].copy()
                # print(f"[Backtrack] return stage2_pool={stage2_pool.shape[0]}")
            else:
                filtered_grasps = np.empty((0, 13), dtype=float)
                filtered_w0 = np.empty((0,), dtype=float)
                # print("[Backtrack] return empty")
            break

        if len(round_grasps_world) == 0:
            round_id += 1
            continue

        all_grasps_stack = np.vstack(round_grasps_world)

        # -------- (2) 把历史 stage2_pool 注入本轮过滤候选 --------
        if stage2_pool.shape[0] > 0:
            all_for_filter = np.vstack([stage2_pool, all_grasps_stack])
            w0_external = np.concatenate([
                stage2_w0_pool,
                all_grasps_stack[:, 12].copy()
            ], axis=0)
        else:
            all_for_filter = all_grasps_stack
            w0_external = all_grasps_stack[:, 12].copy()

        # -------- (3) 过滤：返回 Stage2/Stage3 结论 + keep2（用于累计）--------
        t0 = time.time()
        grasps0 = all_for_filter
        grasps1, idx_map = prefilter_remove_upward_approach(
            grasps0,
            up_world=np.array([0, 0, 1.0]),
            angle_thresh_deg=80.0
        )
        if w0_external is not None:
            width1 = np.asarray(w0_external).reshape(-1)[idx_map]
        else:
            width1 = None

        # 注意：这里开启 return_keep2=True，解包多 3 个返回值（keep2_grasps/ids/w0）
        (final_grasps, kept_ids, stage_tag,
         keep3_grasps, keep3_ids,
         final_w0, keep3_w0,
         keep2_grasps, keep2_ids, keep2_w0) = filter_grasps_fast(
            all_grasps_world=grasps1,
            p_pcd=pcd_with_plane,
            hand_params=HAND_PARAMS_REAL,

            eps_scale=2,
            collision_slack_scale=0.0,
            min_points_between=50,
            margin_y_between=0.005,
            widen_after_layer2=0.01,
            enable_widen_after_layer2=True,
            span_thresh=0.006,
            angle_thresh_deg=20.0,
            xz_res=0.005,
            min_pts_line=3,
            y_percentile=3,
            min_keep=30,

            return_meta=True,
            width_orig_external=width1,
            return_width_orig=True,
            return_keep2=True
        )
        t1 = time.time()
        print(f'filter time: {t1-t0}')

        # -------- (4) 累计“本轮单个 Stage2 通过”的 grasp（keep2）--------
        if keep2_grasps is not None and keep2_grasps.shape[0] > 0:
            stage2_pool, stage2_w0_pool, stage2_bucket_state = merge_unique_grasps(
                stage2_pool, stage2_w0_pool,
                keep2_grasps, keep2_w0,
                pos_thr=0.005, rot_thr_deg=5.0, width_thr=0.01,
                max_total=None,
                pos_bin=0.01, dir_bin=0.2, width_bin=0.02,
                search_neighbor_buckets=False,
                bucket_state=stage2_bucket_state
            )

        # -------- (5) 终止条件：Stage3 达标立即返回；Stage2 达标立即返回（按你要求）--------
        if stage_tag == 3:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            break

        if stage_tag == 2:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            break

        round_id += 1
        continue

    if filtered_grasps.shape[0] > 80:
        filtered_grasps, sel_idx = select_diverse_grasps_fps(
            filtered_grasps,
            K=80,
            pos_scale=0.02,
            ang_scale_deg=15.0,
            width_scale=0.02,
            seed="centroid",
            random_seed=0,
            binormal_flip_equiv=False
        )

    geo = [pcd_with_plane]
    # filtered_grasps[:,2] += 0.02
    pos      = filtered_grasps[:, 0:3]
    axis     = filtered_grasps[:, 3:6]
    approach = filtered_grasps[:, 6:9]
    binormal = filtered_grasps[:, 9:12]
    width    = filtered_grasps[:, 12]
    for i in range(pos.shape[0]):
        mesh = create_gripper_mesh(
            pos[i], approach[i], binormal[i], axis[i],
            width[i], HAND_PARAMS_VISUAL
        )
        geo.extend(mesh)
    # o3d.visualization.draw_geometries(geo)

    output_grasp = convert_all_grasps_pos(
        filtered_grasps, HAND_PARAMS_REAL, src="palm", dst="tip"
    )

    # output_grasp = np.tile(output_grasp, (12, 1))

    # print(f'Final grasp:{len(output_grasp)}')

    return output_grasp

# def pmatch_for_draw(path):
#     set_global_seed(0)
#     HAND_PARAMS_REAL = {
#         "finger_width":        0.015,
#         "hand_outer_diameter": 0.167,
#         "hand_depth":          0.0475,
#         "hand_height":         0.02,
#         "palm_thickness":      0.004,
#         "palm_height":         0.0334,
#     }
#     HAND_PARAMS_VISUAL = {
#         "finger_width":        0.002,
#         "hand_outer_diameter": 0.141,
#         "hand_depth":          0.0475,
#         "hand_height":         0.002,
#         "palm_thickness":      0.002,
#         "palm_height":         0.002,
#     }
#     normal_radius = 0.01
#     voxel_norm = 0.005

#     # pcd split
#     t0 = time.time()
#     pcd_p_ori = o3d.io.read_point_cloud(path)
#     # o3d.visualization.draw_geometries([pcd_p_ori])

    
#     # pcd_p_ori, _ =  pcd_p_ori.remove_radius_outlier(8,0.006)
#     ct = pcd_p_ori.get_center()
#     plane = plane_pcd()
#     plane.translate([ct[0], ct[1], 0.105], relative=True)
#     t1 = time.time()
#     # cprint(f'Time for partfield:{t1-t0}', 'green')

#     pcd_p_ori.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
#     )
#     plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
#     plane.normals = o3d.utility.Vector3dVector(plane_n)

#     pcd_with_plane = pcd_p_ori + plane
#     # o3d.visualization.draw_geometries([pcd_with_plane])

#     filter_ctx = build_filter_context(
#         pcd_with_plane,
#         normal_radius=normal_radius,
#         normal_max_nn=30,
#     )
    
#     index_path = os.path.join(
#         "/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k",
#         "global_index_ems.pkl"
#     )
#     index_data = load_global_index(index_path)
#     index_arrays = build_index_arrays(index_data)

#     # ------- (A) 先为每个 part 计算一次 ranked 列表（不要在回溯里重复跑 ems_match）-------
#     BATCH = 5
#     MAX_CANDS = 25
#     parts = []

#     pcd_p_o3d_norm = pcd_p_ori.voxel_down_sample(voxel_size=voxel_norm)
#     center = pcd_p_o3d_norm.get_center()
#     pcd_p_o3d_norm.translate(-center)

#     t0 = time.time()
#     ranked, qems = ems_match_fast(
#         query_pcd_norm=pcd_p_o3d_norm,
#         idxA=index_arrays,
#         top_k_shape=200,
#         top_k_scale=MAX_CANDS,
#         top_n=min(BATCH, MAX_CANDS),
#         w_ratio=1.0, w_eps=1.2,
#         lambda_scale=1.6, lambda_abs=2.0,
#         hard_scale_tol=None,
#         hard_abs_tol=None,
#         visualize=False
#     )
#     print(f'Tems:{time.time()-t0}')

#     parts.append({
#         "pcd_norm": pcd_p_o3d_norm,
#         "center": center,
#         "ranked": ranked,
#         "Tq": qems["T_sq"],
#     })

#     # ------- (B) 回溯轮次：累计 Stage2 通过的 grasp（keep2），注入下一轮一起 filter -------
#     stage2_pool = np.empty((0, 13), dtype=float)      # 存最终 width 的 grasp（可能已增宽）
#     stage2_w0_pool = np.empty((0,), dtype=float)      # 与 stage2_pool 对齐：存 orig width（增宽前）
#     stage2_bucket_state = {"buckets": {}}             # 桶化索引（跨轮复用）
#     round_id = 0

#     while True:
#         start = round_id * BATCH
#         end   = start + BATCH

#         any_added = False
#         round_grasps_world = []

#         # -------- (1) 生成本轮新 grasp 候选 --------
#         for part in parts:
#             entries = part["ranked"][start:end]
#             if entries is None or len(entries) == 0:
#                 continue

#             any_added = True

#             t1 = time.time()
#             grasps_list = register_by_superquadric(
#                 query_pcd_norm=part["pcd_norm"],
#                 top_entries=entries,
#                 Tq=part["Tq"],
#                 n=len(entries),
#                 voxel=voxel_norm,
#                 center_mode="keep",
#                 do_icp=False,
#                 visualize=False,
#                 hand_params=HAND_PARAMS_REAL,
#             )
#             # print(f'Tregister:{time.time()-t1}')
#             grasps_list = [g for g in grasps_list if isinstance(g, np.ndarray) and g.size > 0]
#             if len(grasps_list) == 0:
#                 continue

#             grasps = np.vstack(grasps_list)
#             grasps = transform_grasps_from_norm(grasps, part["center"])
#             round_grasps_world.append(grasps)

#         # ranked 用尽：返回 stage2_pool（有多少算多少）
#         if not any_added:
#             # print("[Backtrack] ranked exhausted.")
#             if stage2_pool.shape[0] > 0:
#                 filtered_grasps = stage2_pool
#                 filtered_w0 = stage2_w0_pool if stage2_w0_pool.shape[0] == stage2_pool.shape[0] else stage2_pool[:, 12].copy()
#                 # print(f"[Backtrack] return stage2_pool={stage2_pool.shape[0]}")
#             else:
#                 filtered_grasps = np.empty((0, 13), dtype=float)
#                 filtered_w0 = np.empty((0,), dtype=float)
#                 # print("[Backtrack] return empty")
#             break

#         if len(round_grasps_world) == 0:
#             round_id += 1
#             continue

#         all_grasps_stack = np.vstack(round_grasps_world)

#         # -------- (2) 把历史 stage2_pool 注入本轮过滤候选 --------
#         if stage2_pool.shape[0] > 0:
#             all_for_filter = np.vstack([stage2_pool, all_grasps_stack])
#             w0_external = np.concatenate([
#                 stage2_w0_pool,
#                 all_grasps_stack[:, 12].copy()
#             ], axis=0)
#         else:
#             all_for_filter = all_grasps_stack
#             w0_external = all_grasps_stack[:, 12].copy()

#         # -------- (3) 过滤：返回 Stage2/Stage3 结论 + keep2（用于累计）--------
#         t0 = time.time()
#         grasps0 = all_for_filter
#         grasps1, idx_map = prefilter_remove_upward_approach(
#             grasps0,
#             up_world=np.array([0, 0, 1.0]),
#             angle_thresh_deg=80.0
#         )
#         if w0_external is not None:
#             width1 = np.asarray(w0_external).reshape(-1)[idx_map]
#         else:
#             width1 = None



#         # 注意：这里开启 return_keep2=True，解包多 3 个返回值（keep2_grasps/ids/w0）
#         (final_grasps, kept_ids, stage_tag,
#          keep3_grasps, keep3_ids,
#          final_w0, keep3_w0,
#          keep2_grasps, keep2_ids, keep2_w0) = filter_grasps_fast(
#             all_grasps_world=grasps1,
#             p_pcd=pcd_with_plane,
#             hand_params=HAND_PARAMS_REAL,

#             eps_scale=2,
#             collision_slack_scale=0.0,
#             min_points_between=50,
#             margin_y_between=0.005,
#             widen_after_layer2=0.01,
#             enable_widen_after_layer2=True,
#             span_thresh=0.006,
#             angle_thresh_deg=20.0,
#             xz_res=0.005,
#             min_pts_line=3,
#             y_percentile=0,
#             min_keep=20,

#             return_meta=True,
#             width_orig_external=width1,
#             return_width_orig=True,
#             return_keep2=True,

#             enable_layer3_angle_check=True,
#             profile=False,

#             filter_ctx=filter_ctx
            
#         )
#         t1 = time.time()
#         print(f'filter time: {t1-t0}')

#         # -------- (4) 累计“本轮单个 Stage2 通过”的 grasp（keep2）--------
#         if keep2_grasps is not None and keep2_grasps.shape[0] > 0:
#             stage2_pool, stage2_w0_pool, stage2_bucket_state = merge_unique_grasps(
#                 stage2_pool, stage2_w0_pool,
#                 keep2_grasps, keep2_w0,
#                 pos_thr=0.005, rot_thr_deg=5.0, width_thr=0.01,
#                 max_total=None,
#                 pos_bin=0.01, dir_bin=0.2, width_bin=0.02,
#                 search_neighbor_buckets=False,
#                 bucket_state=stage2_bucket_state
#             )

#         # -------- (5) 终止条件：Stage3 达标立即返回；Stage2 达标立即返回（按你要求）--------
#         if stage_tag == 3:
#             filtered_grasps = final_grasps
#             filtered_w0 = final_w0
#             break

#         if stage_tag == 2:
#             filtered_grasps = final_grasps
#             filtered_w0 = final_w0
#             break

#         round_id += 1
#         continue

#     if filtered_grasps.shape[0] > 40:
#         filtered_grasps, sel_idx = select_diverse_grasps_fps(
#             filtered_grasps,
#             K=40,
#             pos_scale=0.02,
#             ang_scale_deg=15.0,
#             width_scale=0.02,
#             seed="centroid",
#             random_seed=0,
#             binormal_flip_equiv=False
#         )

#     geo = [pcd_with_plane]
#     # filtered_grasps[:,2] += 0.02
#     pos      = filtered_grasps[:, 0:3]
#     axis     = filtered_grasps[:, 3:6]
#     approach = filtered_grasps[:, 6:9]
#     binormal = filtered_grasps[:, 9:12]
#     width    = filtered_grasps[:, 12]
#     for i in range(pos.shape[0]):
#         mesh = create_gripper_mesh(
#             pos[i], approach[i], binormal[i], axis[i],
#             width[i], HAND_PARAMS_VISUAL
#         )
#         geo.extend(mesh)
#     # o3d.visualization.draw_geometries(geo)

#     output_grasp = convert_all_grasps_pos(
#         filtered_grasps, HAND_PARAMS_REAL, src="palm", dst="tip"
#     )

#     # output_grasp = np.tile(output_grasp, (12, 1))

#     print(f'Final grasp:{len(output_grasp)}')

#     return output_grasp, filtered_grasps


# def pmatch_for_draw(path):
#     set_global_seed(0)

#     HAND_PARAMS_REAL = {
#         "finger_width":        0.015,
#         "hand_outer_diameter": 0.167,
#         "hand_depth":          0.0475,
#         "hand_height":         0.02,
#         "palm_thickness":      0.004,
#         "palm_height":         0.0334,
#     }
#     HAND_PARAMS_VISUAL = {
#         "finger_width":        0.002,
#         "hand_outer_diameter": 0.141,
#         "hand_depth":          0.0475,
#         "hand_height":         0.002,
#         "palm_thickness":      0.002,
#         "palm_height":         0.002,
#     }

#     normal_radius = 0.01
#     voxel_norm = 0.005

#     # 这些阈值建议和 filter_grasps_fast 里保持一致
#     MIN_KEEP_STAGE3 = 20
#     MIN_KEEP_STAGE2 = 50

#     # --------------------------------------------------
#     # 1) 读点云 / 构平面 / normals
#     # --------------------------------------------------
#     t_all0 = time.time()

#     pcd_p_ori = o3d.io.read_point_cloud(path)

#     ct = pcd_p_ori.get_center()
#     plane = plane_pcd()
#     plane.translate([ct[0], ct[1], 0.105], relative=True)

#     pcd_p_ori.estimate_normals(
#         search_param=o3d.geometry.KDTreeSearchParamHybrid(
#             radius=normal_radius, max_nn=30
#         )
#     )
#     plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
#     plane.normals = o3d.utility.Vector3dVector(plane_n)

#     pcd_with_plane = pcd_p_ori + plane

#     # 只构建一次 filter context
#     filter_ctx = build_filter_context(
#         pcd_with_plane,
#         normal_radius=normal_radius,
#         normal_max_nn=30,
#     )

#     # --------------------------------------------------
#     # 2) 读数据库索引
#     # --------------------------------------------------
#     index_path = os.path.join(
#         "/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k",
#         "global_index_ems.pkl"
#     )
#     index_data = load_global_index(index_path)
#     index_arrays = build_index_arrays(index_data)

#     # --------------------------------------------------
#     # 3) query 归一化 + EMS match（只做一次）
#     # --------------------------------------------------
#     BATCH = 5
#     MAX_CANDS = 40

#     pcd_p_o3d_norm = pcd_p_ori.voxel_down_sample(voxel_size=voxel_norm)
#     center = pcd_p_o3d_norm.get_center()
#     pcd_p_o3d_norm.translate(-center)

#     t_ems0 = time.time()
#     ranked, qems = ems_match_fast(
#         query_pcd_norm=pcd_p_o3d_norm,
#         idxA=index_arrays,
#         top_k_shape=200,
#         top_k_scale=MAX_CANDS,
#         top_n=min(BATCH, MAX_CANDS),
#         w_ratio=1.0,
#         w_eps=1.2,
#         lambda_scale=1.6,
#         lambda_abs=2.0,
#         hard_scale_tol=None,
#         hard_abs_tol=None,
#         visualize=False
#     )
#     # print(f"Tems:{time.time() - t_ems0}")

#     if qems is None or ranked is None or len(ranked) == 0:
#         print("[WARN] EMS match failed or empty ranked.")
#         return np.empty((0, 13), dtype=float), np.empty((0, 13), dtype=float)

#     print(*(item["path"] for item in ranked), sep="\n")
#     parts = [{
#         "pcd_norm": pcd_p_o3d_norm,
#         "center": center,
#         "ranked": ranked,
#         "Tq": qems["T_sq"],
#     }]

#     # --------------------------------------------------
#     # 4) 增量池：Stage2 / Stage3 分开累计
#     #    注意：以后每一轮只过滤“新 grasps”
#     # --------------------------------------------------
#     stage2_pool = np.empty((0, 13), dtype=float)
#     stage2_w0_pool = np.empty((0,), dtype=float)
#     stage2_bucket_state = {"buckets": {}}

#     stage3_pool = np.empty((0, 13), dtype=float)
#     stage3_w0_pool = np.empty((0,), dtype=float)
#     stage3_bucket_state = {"buckets": {}}

#     round_id = 0
#     filtered_grasps = np.empty((0, 13), dtype=float)
#     filtered_w0 = np.empty((0,), dtype=float)

#     while True:
#         start = round_id * BATCH
#         end = start + BATCH

#         any_added = False
#         round_grasps_world = []

#         # --------------------------------------------------
#         # (A) 本轮只生成新 grasp
#         # --------------------------------------------------
#         for part in parts:
#             entries = part["ranked"][start:end]
#             if entries is None or len(entries) == 0:
#                 continue

#             any_added = True

#             t_reg0 = time.time()
#             grasps_list = register_by_superquadric(
#                 query_pcd_norm=part["pcd_norm"],
#                 top_entries=entries,
#                 Tq=part["Tq"],
#                 n=len(entries),
#                 voxel=voxel_norm,
#                 center_mode="keep",
#                 do_icp=False,
#                 visualize=True,
#                 hand_params=HAND_PARAMS_REAL,
#             )
#             # print(f"Tregister:{time.time() - t_reg0}")

#             grasps_list = [g for g in grasps_list if isinstance(g, np.ndarray) and g.size > 0]
#             if len(grasps_list) == 0:
#                 continue

#             grasps = np.vstack(grasps_list)
#             grasps = transform_grasps_from_norm(grasps, part["center"])
#             round_grasps_world.append(grasps)

#         # ranked 用尽：不再继续
#         if not any_added:
#             if stage3_pool.shape[0] >= MIN_KEEP_STAGE3:
#                 filtered_grasps = stage3_pool
#                 filtered_w0 = stage3_w0_pool
#             elif stage2_pool.shape[0] > 0:
#                 filtered_grasps = stage2_pool
#                 filtered_w0 = stage2_w0_pool
#             elif stage3_pool.shape[0] > 0:
#                 filtered_grasps = stage3_pool
#                 filtered_w0 = stage3_w0_pool
#             else:
#                 filtered_grasps = np.empty((0, 13), dtype=float)
#                 filtered_w0 = np.empty((0,), dtype=float)
#             break

#         if len(round_grasps_world) == 0:
#             round_id += 1
#             continue

#         all_grasps_stack = np.vstack(round_grasps_world)

#         # --------------------------------------------------
#         # (B) 只对“新 grasp”做 upward prefilter
#         # --------------------------------------------------
#         t_filter0 = time.time()

#         grasps1, idx_map = prefilter_remove_upward_approach(
#             all_grasps_stack,
#             up_world=np.array([0.0, 0.0, 1.0]),
#             angle_thresh_deg=80.0
#         )

#         if grasps1.shape[0] == 0:
#             print(f'filter time: {time.time() - t_filter0}')
#             round_id += 1
#             continue

#         width1 = all_grasps_stack[:, 12].copy()[idx_map]

#         # --------------------------------------------------
#         # (C) 只过滤“新 grasp”
#         # --------------------------------------------------
#         (
#             final_grasps_cur, kept_ids_cur, stage_tag_cur,
#             keep3_grasps_cur, keep3_ids_cur,
#             final_w0_cur, keep3_w0_cur,
#             keep2_grasps_cur, keep2_ids_cur, keep2_w0_cur
#         ) = filter_grasps_fast(
#             all_grasps_world=grasps1,
#             p_pcd=pcd_with_plane,
#             hand_params=HAND_PARAMS_REAL,

#             eps_scale=2,
#             collision_slack_scale=0.0,
#             min_points_between=50,
#             margin_y_between=0.005,
#             widen_after_layer2=0.01,
#             enable_widen_after_layer2=True,
#             span_thresh=0.006,
#             angle_thresh_deg=20.0,
#             xz_res=0.005,
#             min_pts_line=3,
#             y_percentile=0,
#             min_keep=MIN_KEEP_STAGE3,

#             return_meta=True,
#             width_orig_external=width1,
#             return_width_orig=True,
#             return_keep2=True,

#             enable_layer3_angle_check=True,
#             profile=False,
#             filter_ctx=filter_ctx
#         )

#         print(f'filter time: {time.time() - t_filter0}')

#         # --------------------------------------------------
#         # (D) 增量合并到 Stage2 pool
#         # --------------------------------------------------
#         if keep2_grasps_cur is not None and keep2_grasps_cur.shape[0] > 0:
#             stage2_pool, stage2_w0_pool, stage2_bucket_state = merge_unique_grasps(
#                 stage2_pool, stage2_w0_pool,
#                 keep2_grasps_cur, keep2_w0_cur,
#                 pos_thr=0.005,
#                 rot_thr_deg=5.0,
#                 width_thr=0.01,
#                 max_total=None,
#                 pos_bin=0.01,
#                 dir_bin=0.2,
#                 width_bin=0.02,
#                 search_neighbor_buckets=False,
#                 bucket_state=stage2_bucket_state
#             )

#         # --------------------------------------------------
#         # (E) 增量合并到 Stage3 pool
#         # --------------------------------------------------
#         if keep3_grasps_cur is not None and keep3_grasps_cur.shape[0] > 0:
#             stage3_pool, stage3_w0_pool, stage3_bucket_state = merge_unique_grasps(
#                 stage3_pool, stage3_w0_pool,
#                 keep3_grasps_cur, keep3_w0_cur,
#                 pos_thr=0.005,
#                 rot_thr_deg=5.0,
#                 width_thr=0.01,
#                 max_total=None,
#                 pos_bin=0.01,
#                 dir_bin=0.2,
#                 width_bin=0.02,
#                 search_neighbor_buckets=False,
#                 bucket_state=stage3_bucket_state
#             )

#         # --------------------------------------------------
#         # (F) 用“累计池”决定是否停止
#         # --------------------------------------------------
#         if stage3_pool.shape[0] >= MIN_KEEP_STAGE3:
#             filtered_grasps = stage3_pool
#             filtered_w0 = stage3_w0_pool
#             # print(f"[accum] Stage3 pool reached {stage3_pool.shape[0]}")
#             break

#         if stage2_pool.shape[0] >= MIN_KEEP_STAGE2:
#             filtered_grasps = stage2_pool
#             filtered_w0 = stage2_w0_pool
#             # print(f"[accum] Stage2 pool reached {stage2_pool.shape[0]}")
#             break

#         round_id += 1

#     # --------------------------------------------------
#     # 5) 多样性裁剪
#     # --------------------------------------------------
#     if filtered_grasps.shape[0] > 40:
#         filtered_grasps, sel_idx = select_diverse_grasps_fps(
#             filtered_grasps,
#             K=40,
#             pos_scale=0.02,
#             ang_scale_deg=15.0,
#             width_scale=0.02,
#             seed="centroid",
#             random_seed=0,
#             binormal_flip_equiv=False
#         )
#         if filtered_w0.shape[0] == sel_idx.shape[0] or filtered_w0.shape[0] == len(sel_idx):
#             filtered_w0 = filtered_w0[sel_idx]

#     # --------------------------------------------------
#     # 6) 可视化 grasp mesh（如需要）
#     # --------------------------------------------------
#     geo = [pcd_with_plane]
#     pos      = filtered_grasps[:, 0:3]
#     axis     = filtered_grasps[:, 3:6]
#     approach = filtered_grasps[:, 6:9]
#     binormal = filtered_grasps[:, 9:12]
#     width    = filtered_grasps[:, 12]

#     for i in range(pos.shape[0]):
#         mesh = create_gripper_mesh(
#             pos[i], approach[i], binormal[i], axis[i],
#             width[i], HAND_PARAMS_VISUAL
#         )
#         geo.extend(mesh)
#     # o3d.visualization.draw_geometries(geo)

#     # --------------------------------------------------
#     # 7) palm -> tip
#     # --------------------------------------------------
#     output_grasp = convert_all_grasps_pos(
#         filtered_grasps,
#         HAND_PARAMS_REAL,
#         src="palm",
#         dst="tip"
#     )

#     print(f'Final grasp:{len(output_grasp)}')
#     # print(f'Total pmatch_for_draw time: {time.time() - t_all0}')

#     return output_grasp, filtered_grasps

@lru_cache(maxsize=4)
def get_cached_index_arrays(index_path: str):
    index_path = os.path.abspath(index_path)
    index_data = load_global_index(index_path)
    return build_index_arrays(index_data)

def pmatch_for_draw(path,
                    show_visualization: bool = False,
                    record_video_path: Optional[str] = None,
                    video_width: int = 1280,
                    video_height: int = 960,
                    video_fps: float = 30.0,
                    video_duration_sec: float = 12.0,
                    video_visible: bool = True):
    set_global_seed(0)

    HAND_PARAMS_REAL = {
        "finger_width":        0.015,
        "hand_outer_diameter": 0.167,
        "hand_depth":          0.0475,
        "hand_height":         0.02,
        "palm_thickness":      0.004,
        "palm_height":         0.0334,
    }
    HAND_PARAMS_VISUAL = {
        "finger_width":        0.002,
        "hand_outer_diameter": 0.141,
        "hand_depth":          0.0475,
        "hand_height":         0.002,
        "palm_thickness":      0.002,
        "palm_height":         0.002,
    }

    normal_radius = 0.01
    voxel_norm = 0.005

    # 这些阈值建议和 filter_grasps_fast 里保持一致
    MIN_KEEP_STAGE3 = 20
    MIN_KEEP_STAGE2 = 50

    # --------------------------------------------------
    # 1) 读点云 / 构平面 / normals
    # --------------------------------------------------
    t_all0 = time.time()

    pcd_p_ori = o3d.io.read_point_cloud(path)

    ct = pcd_p_ori.get_center()
    plane = o3d.io.read_point_cloud('/root/catkin_ws/src/more_than_grasp/refine/data/plane_dir/cloud_object3.ply')
    # plane = plane_pcd()
    # plane.translate([ct[0], ct[1], 0.105], relative=True)

    pcd_p_ori.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=30
        )
    )
    # plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
    # plane.normals = o3d.utility.Vector3dVector(plane_n)

    pcd_with_plane = pcd_p_ori + plane

    # 只构建一次 filter context
    filter_ctx = build_filter_context(
        pcd_with_plane,
        normal_radius=normal_radius,
        normal_max_nn=30,
    )

    # --------------------------------------------------
    # 2) 读数据库索引
    # --------------------------------------------------
    index_path = os.path.join(
        "/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k",
        "global_index_ems.pkl"
    )
    index_arrays = get_cached_index_arrays(index_path)

    # --------------------------------------------------
    # 3) query 归一化 + EMS match（只做一次）
    # --------------------------------------------------
    BATCH = 5
    MAX_CANDS = 40

    pcd_p_o3d_norm = pcd_p_ori.voxel_down_sample(voxel_size=voxel_norm)
    center = pcd_p_o3d_norm.get_center()
    pcd_p_o3d_norm.translate(-center)

    t_ems0 = time.time()
    ranked, qems = ems_match_fast(
        query_pcd_norm=pcd_p_o3d_norm,
        idxA=index_arrays,
        top_k_shape=200,
        top_k_scale=MAX_CANDS,
        top_n=min(BATCH, MAX_CANDS),
        w_ratio=1.0,
        w_eps=1.2,
        lambda_scale=1.6,
        lambda_abs=2.0,
        hard_scale_tol=None,
        hard_abs_tol=None,
        visualize=True
    )
    # print(f"Tems:{time.time() - t_ems0}")

    if qems is None or ranked is None or len(ranked) == 0:
        print("[WARN] EMS match failed or empty ranked.")
        return np.empty((0, 13), dtype=float), np.empty((0, 13), dtype=float)

    # print(*(item["path"] for item in ranked), sep="\n")
    parts = [{
        "pcd_norm": pcd_p_o3d_norm,
        "center": center,
        "ranked": ranked,
        "Tq": qems["T_sq"],
    }]

    # --------------------------------------------------
    # 4) 增量池：Stage2 / Stage3 分开累计
    #    注意：以后每一轮只过滤“新 grasps”
    # --------------------------------------------------
    stage2_pool = np.empty((0, 13), dtype=float)
    stage2_w0_pool = np.empty((0,), dtype=float)
    stage2_bucket_state = {"buckets": {}}

    stage3_pool = np.empty((0, 13), dtype=float)
    stage3_w0_pool = np.empty((0,), dtype=float)
    stage3_bucket_state = {"buckets": {}}

    round_id = 0
    filtered_grasps = np.empty((0, 13), dtype=float)
    filtered_w0 = np.empty((0,), dtype=float)

    while True:
        start = round_id * BATCH
        end = start + BATCH

        any_added = False
        round_grasps_world = []

        # --------------------------------------------------
        # (A) 本轮只生成新 grasp
        # --------------------------------------------------
        for part in parts:
            entries = part["ranked"][start:end]
            if entries is None or len(entries) == 0:
                continue

            any_added = True

            t_reg0 = time.time()
            grasps_list = register_by_superquadric(
                query_pcd_norm=part["pcd_norm"],
                top_entries=entries,
                Tq=part["Tq"],
                n=len(entries),
                voxel=voxel_norm,
                center_mode="keep",
                do_icp=False,
                visualize=True,
                hand_params=HAND_PARAMS_REAL,
            )
            # print(f"Tregister:{time.time() - t_reg0}")

            grasps_list = [g for g in grasps_list if isinstance(g, np.ndarray) and g.size > 0]
            if len(grasps_list) == 0:
                continue

            grasps = np.vstack(grasps_list)
            grasps = transform_grasps_from_norm(grasps, part["center"])
            round_grasps_world.append(grasps)

        # ranked 用尽：不再继续
        if not any_added:
            if stage3_pool.shape[0] >= MIN_KEEP_STAGE3:
                filtered_grasps = stage3_pool
                filtered_w0 = stage3_w0_pool
            elif stage2_pool.shape[0] > 0:
                filtered_grasps = stage2_pool
                filtered_w0 = stage2_w0_pool
            elif stage3_pool.shape[0] > 0:
                filtered_grasps = stage3_pool
                filtered_w0 = stage3_w0_pool
            else:
                filtered_grasps = np.empty((0, 13), dtype=float)
                filtered_w0 = np.empty((0,), dtype=float)
            break

        if len(round_grasps_world) == 0:
            round_id += 1
            continue

        all_grasps_stack = np.vstack(round_grasps_world)

        # --------------------------------------------------
        # (B) 只对“新 grasp”做 upward prefilter
        # --------------------------------------------------
        t_filter0 = time.time()

        grasps1, idx_map = prefilter_remove_upward_approach(
            all_grasps_stack,
            up_world=np.array([0.0, 0.0, 1.0]),
            angle_thresh_deg=80.0
        )

        if grasps1.shape[0] == 0:
            print(f'filter time: {time.time() - t_filter0}')
            round_id += 1
            continue

        width1 = all_grasps_stack[:, 12].copy()[idx_map]

        # --------------------------------------------------
        # (C) 只过滤“新 grasp”
        # --------------------------------------------------
        (
            final_grasps_cur, kept_ids_cur, stage_tag_cur,
            keep3_grasps_cur, keep3_ids_cur,
            final_w0_cur, keep3_w0_cur,
            keep2_grasps_cur, keep2_ids_cur, keep2_w0_cur
        ) = filter_grasps_fast(
            all_grasps_world=grasps1,
            p_pcd=pcd_with_plane,
            hand_params=HAND_PARAMS_REAL,

            eps_scale=2,
            collision_slack_scale=0.0,
            min_points_between=50,
            margin_y_between=0.005,
            widen_after_layer2=0.01,
            enable_widen_after_layer2=True,
            span_thresh=0.006,
            angle_thresh_deg=20.0,
            xz_res=0.005,
            min_pts_line=3,
            y_percentile=0,
            min_keep=MIN_KEEP_STAGE3,

            return_meta=True,
            width_orig_external=width1,
            return_width_orig=True,
            return_keep2=True,

            enable_layer3_angle_check=True,
            profile=False,
            filter_ctx=filter_ctx
        )

        print(f'filter time: {time.time() - t_filter0}')

        # --------------------------------------------------
        # (D) 增量合并到 Stage2 pool
        # --------------------------------------------------
        if keep2_grasps_cur is not None and keep2_grasps_cur.shape[0] > 0:
            stage2_pool, stage2_w0_pool, stage2_bucket_state = merge_unique_grasps(
                stage2_pool, stage2_w0_pool,
                keep2_grasps_cur, keep2_w0_cur,
                pos_thr=0.005,
                rot_thr_deg=5.0,
                width_thr=0.01,
                max_total=None,
                pos_bin=0.01,
                dir_bin=0.2,
                width_bin=0.02,
                search_neighbor_buckets=False,
                bucket_state=stage2_bucket_state
            )

        # --------------------------------------------------
        # (E) 增量合并到 Stage3 pool
        # --------------------------------------------------
        if keep3_grasps_cur is not None and keep3_grasps_cur.shape[0] > 0:
            stage3_pool, stage3_w0_pool, stage3_bucket_state = merge_unique_grasps(
                stage3_pool, stage3_w0_pool,
                keep3_grasps_cur, keep3_w0_cur,
                pos_thr=0.005,
                rot_thr_deg=5.0,
                width_thr=0.01,
                max_total=None,
                pos_bin=0.01,
                dir_bin=0.2,
                width_bin=0.02,
                search_neighbor_buckets=False,
                bucket_state=stage3_bucket_state
            )

        # --------------------------------------------------
        # (F) 用“累计池”决定是否停止
        # --------------------------------------------------
        if stage3_pool.shape[0] >= MIN_KEEP_STAGE3:
            filtered_grasps = stage3_pool
            filtered_w0 = stage3_w0_pool
            # print(f"[accum] Stage3 pool reached {stage3_pool.shape[0]}")
            break

        if stage2_pool.shape[0] >= MIN_KEEP_STAGE2:
            filtered_grasps = stage2_pool
            filtered_w0 = stage2_w0_pool
            # print(f"[accum] Stage2 pool reached {stage2_pool.shape[0]}")
            break

        round_id += 1

    # --------------------------------------------------
    # 5) 多样性裁剪
    # --------------------------------------------------
    if filtered_grasps.shape[0] > 40:
        filtered_grasps, sel_idx = select_diverse_grasps_fps(
            filtered_grasps,
            K=40,
            pos_scale=0.02,
            ang_scale_deg=15.0,
            width_scale=0.02,
            seed="centroid",
            random_seed=0,
            binormal_flip_equiv=False
        )
        if filtered_w0.shape[0] == sel_idx.shape[0] or filtered_w0.shape[0] == len(sel_idx):
            filtered_w0 = filtered_w0[sel_idx]

    # --------------------------------------------------
    # 6) 可视化 grasp mesh（如需要）
    # --------------------------------------------------
    geo = [pcd_with_plane]
    pos      = filtered_grasps[:, 0:3]
    axis     = filtered_grasps[:, 3:6]
    approach = filtered_grasps[:, 6:9]
    binormal = filtered_grasps[:, 9:12]
    width    = filtered_grasps[:, 12]

    for i in range(pos.shape[0]):
        mesh = create_gripper_mesh(
            pos[i], approach[i], binormal[i], axis[i],
            width[i], HAND_PARAMS_VISUAL
        )
        geo.extend(mesh)
    o3d.visualization.draw_geometries(geo)

    # geo = build_pmatch_visualization_geometries(
    #     pcd_with_plane,
    #     filtered_grasps,
    #     HAND_PARAMS_VISUAL,
    # )

    # if record_video_path:
    #     saved_video_path = record_pmatch_grasp_orbit_video(
    #         pcd_with_plane,
    #         filtered_grasps,
    #         record_video_path,
    #         HAND_PARAMS_VISUAL,
    #         orbit_center=pcd_p_ori.get_center(),
    #         width=video_width,
    #         height=video_height,
    #         fps=video_fps,
    #         duration_sec=video_duration_sec,
    #         visible=video_visible,
    #     )
    #     print(f"[INFO] Saved pmatch orbit video to {saved_video_path}")

    # if show_visualization:
    #     o3d.visualization.draw_geometries(
    #         geo,
    #         window_name=f"pmatch grasps ({len(filtered_grasps)})",
    #     )

    # --------------------------------------------------
    # 7) palm -> tip
    # --------------------------------------------------
    output_grasp = convert_all_grasps_pos(
        filtered_grasps,
        HAND_PARAMS_REAL,
        src="palm",
        dst="tip"
    )

    print(f'Final grasp:{len(output_grasp)}')
    # print(f'Total pmatch_for_draw time: {time.time() - t_all0}')

    return output_grasp, filtered_grasps

def pmatch_without_callibration_with_other(path,global_pcd):
    set_global_seed(0)
    HAND_PARAMS_REAL = {
        "finger_width":        0.015,
        "hand_outer_diameter": 0.167,
        "hand_depth":          0.0475,
        "hand_height":         0.02,
        "palm_thickness":      0.004,
        "palm_height":         0.0334,
    }
    HAND_PARAMS_VISUAL = {
        "finger_width":        0.002,
        "hand_outer_diameter": 0.141,
        "hand_depth":          0.0475,
        "hand_height":         0.002,
        "palm_thickness":      0.002,
        "palm_height":         0.002,
    }
    normal_radius = 0.01
    voxel_norm = 0.005

    # pcd split
    t0 = time.time()
    pcd_p_ori = o3d.io.read_point_cloud(path)
    # pcd_p_ori, _ =  pcd_p_ori.remove_radius_outlier(8,0.006)
    ct = pcd_p_ori.get_center()
    plane = plane_pcd(side_len=0.4)
    plane.translate([ct[0], ct[1], -0.005], relative=True)
    t1 = time.time()
    # cprint(f'Time for partfield:{t1-t0}', 'green')

    pcd_p_ori.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
    plane.normals = o3d.utility.Vector3dVector(plane_n)

    # pcd_with_plane = pcd_p_ori + plane
    global_pcd = global_pcd + plane
    # o3d.visualization.draw_geometries([pcd_with_plane])

    index_path = os.path.join(
        "/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k",
        "global_index_ems.pkl"
    )
    index_data = load_global_index(index_path)
    index_arrays = build_index_arrays(index_data)

    # ------- (A) 先为每个 part 计算一次 ranked 列表（不要在回溯里重复跑 ems_match）-------
    BATCH = 10
    MAX_CANDS = 80
    parts = []

    pcd_p_o3d_norm = pcd_p_ori.voxel_down_sample(voxel_size=voxel_norm)
    if len(pcd_p_o3d_norm.points) < 50:
        return []
    center = pcd_p_o3d_norm.get_center()
    pcd_p_o3d_norm.translate(-center)

    ranked, qems = ems_match_fast(
        query_pcd_norm=pcd_p_o3d_norm,
        idxA=index_arrays,
        top_k_shape=200,
        top_k_scale=MAX_CANDS,
        top_n=min(BATCH, MAX_CANDS),
        w_ratio=1.0, w_eps=1.2,
        lambda_scale=1.6, lambda_abs=2.0,
        hard_scale_tol=None,
        hard_abs_tol=None,
        visualize=False
    )

    if qems is None:
        return []
    parts.append({
        "pcd_norm": pcd_p_o3d_norm,
        "center": center,
        "ranked": ranked,
        "Tq": qems["T_sq"],
    })

    # ------- (B) 回溯轮次：累计 Stage2 通过的 grasp（keep2），注入下一轮一起 filter -------
    stage2_pool = np.empty((0, 13), dtype=float)      # 存最终 width 的 grasp（可能已增宽）
    stage2_w0_pool = np.empty((0,), dtype=float)      # 与 stage2_pool 对齐：存 orig width（增宽前）
    stage2_bucket_state = {"buckets": {}}             # 桶化索引（跨轮复用）
    round_id = 0

    while True:
        start = round_id * BATCH
        end   = start + BATCH

        any_added = False
        round_grasps_world = []

        # -------- (1) 生成本轮新 grasp 候选 --------
        for part in parts:
            entries = part["ranked"][start:end]
            if entries is None or len(entries) == 0:
                continue

            any_added = True

            grasps_list = register_by_superquadric(
                query_pcd_norm=part["pcd_norm"],
                top_entries=entries,
                Tq=part["Tq"],
                n=len(entries),
                voxel=voxel_norm,
                center_mode="keep",
                do_icp=False,
                visualize=False,
                hand_params=HAND_PARAMS_REAL,
            )

            grasps_list = [g for g in grasps_list if isinstance(g, np.ndarray) and g.size > 0]
            if len(grasps_list) == 0:
                continue

            grasps = np.vstack(grasps_list)
            grasps = transform_grasps_from_norm(grasps, part["center"])
            round_grasps_world.append(grasps)

        # ranked 用尽：返回 stage2_pool（有多少算多少）
        if not any_added:
            print("[Backtrack] ranked exhausted.")
            if stage2_pool.shape[0] > 0:
                filtered_grasps = stage2_pool
                filtered_w0 = stage2_w0_pool if stage2_w0_pool.shape[0] == stage2_pool.shape[0] else stage2_pool[:, 12].copy()
                print(f"[Backtrack] return stage2_pool={stage2_pool.shape[0]}")
            else:
                filtered_grasps = np.empty((0, 13), dtype=float)
                filtered_w0 = np.empty((0,), dtype=float)
                print("[Backtrack] return empty")
            break

        if len(round_grasps_world) == 0:
            round_id += 1
            continue

        all_grasps_stack = np.vstack(round_grasps_world)

        # -------- (2) 把历史 stage2_pool 注入本轮过滤候选 --------
        if stage2_pool.shape[0] > 0:
            all_for_filter = np.vstack([stage2_pool, all_grasps_stack])
            w0_external = np.concatenate([
                stage2_w0_pool,
                all_grasps_stack[:, 12].copy()
            ], axis=0)
        else:
            all_for_filter = all_grasps_stack
            w0_external = all_grasps_stack[:, 12].copy()

        # -------- (3) 过滤：返回 Stage2/Stage3 结论 + keep2（用于累计）--------
        t0 = time.time()
        grasps0 = all_for_filter
        grasps1, idx_map = prefilter_remove_upward_approach(
            grasps0,
            up_world=np.array([0, 0, 1.0]),
            angle_thresh_deg=80.0
        )
        if w0_external is not None:
            width1 = np.asarray(w0_external).reshape(-1)[idx_map]
        else:
            width1 = None

        (final_grasps, kept_ids, stage_tag,
         keep3_grasps, keep3_ids,
         final_w0, keep3_w0,
         keep2_grasps, keep2_ids, keep2_w0) = filter_grasps_fast(
            all_grasps_world=grasps1,
            p_pcd=global_pcd,
            hand_params=HAND_PARAMS_REAL,

            eps_scale=2,
            collision_slack_scale=0.0,
            min_points_between=50,
            margin_y_between=0.005,
            widen_after_layer2=0.01,
            enable_widen_after_layer2=True,
            span_thresh=0.006,
            angle_thresh_deg=20.0,
            xz_res=0.005,
            min_pts_line=3,
            y_percentile=3,
            min_keep=30,

            return_meta=True,
            width_orig_external=width1,
            return_width_orig=True,
            return_keep2=True
        )
        t1 = time.time()
        print(f'filter time: {t1-t0}')

        # -------- (4) 累计“本轮单个 Stage2 通过”的 grasp（keep2）--------
        if keep2_grasps is not None and keep2_grasps.shape[0] > 0:
            stage2_pool, stage2_w0_pool, stage2_bucket_state = merge_unique_grasps(
                stage2_pool, stage2_w0_pool,
                keep2_grasps, keep2_w0,
                pos_thr=0.005, rot_thr_deg=5.0, width_thr=0.01,
                max_total=None,
                pos_bin=0.01, dir_bin=0.2, width_bin=0.02,
                search_neighbor_buckets=False,
                bucket_state=stage2_bucket_state
            )

        # -------- (5) 终止条件：Stage3 达标立即返回；Stage2 达标立即返回（按你要求）--------
        if stage_tag == 3:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            break

        if stage_tag == 2:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            break

        round_id += 1
        continue

    if filtered_grasps.shape[0] > 80:
        filtered_grasps, sel_idx = select_diverse_grasps_fps(
            filtered_grasps,
            K=80,
            pos_scale=0.02,
            ang_scale_deg=15.0,
            width_scale=0.02,
            seed="centroid",
            random_seed=0,
            binormal_flip_equiv=False
        )

    geo = [global_pcd]
    # filtered_grasps[:,2] += 0.02
    pos      = filtered_grasps[:, 0:3]
    axis     = filtered_grasps[:, 3:6]
    approach = filtered_grasps[:, 6:9]
    binormal = filtered_grasps[:, 9:12]
    width    = filtered_grasps[:, 12]
    for i in range(pos.shape[0]):
        mesh = create_gripper_mesh(
            pos[i], approach[i], binormal[i], axis[i],
            width[i], HAND_PARAMS_VISUAL
        )
        geo.extend(mesh)
    # o3d.visualization.draw_geometries(geo)

    output_grasp = convert_all_grasps_pos(
        filtered_grasps, HAND_PARAMS_REAL, src="palm", dst="tip"
    )

    # output_grasp = np.tile(output_grasp, (12, 1))

    print(f'Final grasp:{len(output_grasp)}')

    return output_grasp

# -------------------- Main --------------------
if __name__ == "__main__":
    # ply_p = '/root/catkin_ws/src/detection/scripts/FoundationStereo-master/output_ros/sam_objects/cloud_object20.ply'
    # ply_p = 'shot_fpfh/descriptors/dataset/mesh/single/008_single.ply'
    # ply_p = '/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/model1/061/nontextured.ply'
    ply_p = '/root/catkin_ws/src/more_than_grasp/refine/data/obj_ply_2/cloud_object1.ply'
    # ply_p = '/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/realsense_data/000.ply'
    # ply_p = '/home/ubuntu/task/more_than_grasp/datasets/data1/obj_sv_ply/ply_obj_00030.ply'
    # ply_p = '/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/photoneo/cloud_object3.ply'
    # obj_p = '/home/djh/code/shot-fpfh/shot_fpfh/descriptors/dataset/3dnet/037.obj'
    grasp_poses, _ = pmatch_for_draw(ply_p)
    t1 = time.time()
    grasp_poses, _ = pmatch_for_draw(ply_p)
    # grasp_pose = main(obj_p)
    t2 = time.time()
    print(t2-t1)
 
#  00064 coco 



