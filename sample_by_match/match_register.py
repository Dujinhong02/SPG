"""
总流程(包括点云按模块分割——各块点云(包括完整)进行匹配、配对、gg映射——全部gg汇总(已转换到p的坐标系))
1、提速+迭代策略更新(保存通过Stage3的候选)
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

def plane_pcd(side_len=0.4, voxel_size=0.002, z0=0.0) -> o3d.geometry.PointCloud:
    # 每边体素数
    n = int(round(side_len / voxel_size))
    if not np.isclose(n * voxel_size, side_len):
        raise ValueError(f"side_len 必须是 voxel_size 的整数倍：{n} * {voxel_size} = {n*voxel_size} ≠ {side_len}")

    half = side_len / 2.0
    edges = np.linspace(-half, half, n + 1)          # n+1 条边界
    centers = (edges[:-1] + edges[1:]) / 2.0         # n 个体素中心

    X, Y = np.meshgrid(centers, centers, indexing="xy")
    Z = np.full_like(X, float(z0))

    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
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
        paths.append(e["path"])
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
    print(f"Tems:{t1-t0}")
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
                [q_vis, cand_aligned, frame],
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
    vis.add_geometry(pcd)
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
        outlier_ratios=(0.10),
        adaptive_upper=(True, False),
        preprocess_voxel=0.0,
        center_mode=center_mode,       
        arc_length_eval=arc_length_viz
    )

    if sq is None:
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


def load_global_index(index_path):
    with open(index_path, "rb") as f:
        return pickle.load(f)

def ems_fit_autotune(pcd: o3d.geometry.PointCloud,
                     outlier_ratios=(0.05, 0.10, 0.20, 0.30),
                     adaptive_upper=(True, False),
                     preprocess_voxel=0.0,
                     center_mode="center",
                     arc_length_eval=0.005):
    """
    返回:
      best_sq, best_info(dict)
    """
    EMS_recovery = _import_ems()
    pc2, shift = _pcd_preprocess_for_ems(
        pcd,
        voxel=preprocess_voxel,
        remove_outlier=True,
        nb_neighbors=20,
        std_ratio=2.0,
        center_mode=center_mode
    )
    pts = np.asarray(pc2.points, dtype=float)
    if pts.shape[0] < 30:
        return None, {"reason": "too_few_points"}

    best = {"score": np.inf}
    best_sq = None
    
    r = outlier_ratios
    # for r in outlier_ratios:
    for au in adaptive_upper:
        try:
            sq, _ = EMS_recovery(
                pts,
                OutlierRatio=float(r),
                AdaptiveUpperBound=bool(au),
                    MaxIterationEM=12,
                    MaxOptiIterations=2,
                    MaxiSwitch=1,
            )
        except Exception:
            continue

        score = _sq_implicit_score(pts, sq)
        med = p90 = None

        if score < best["score"]:
            best = {
                "score": score,
                "median": med,
                "p90": p90,
                "OutlierRatio": float(r),
                "AdaptiveUpperBound": bool(au),
                "shift": shift.tolist(),
                "n_points": int(pts.shape[0]),
            }
            best_sq = sq

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
    print(f'EMS Time:{t1-t0}')
    # print(qems)
    if qems is None:
        print("[Error] Query EMS fitting failed.", info)
        return [] ,None
    print(f'qshape:{qems["shape"]}, qscale:{qems["scale"]}')

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
    print(f'Match Time {t2-t1}')
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

            print(f"{id}, c = {csh}{cs}, score = {score} = {dshape:.4f} + {lambda_scale} x {dscale:.4f} + {lambda_abs} x {dabs:.4f}")
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
            "hand_height":         0.004,   
            "palm_thickness":      0.004,
            "palm_height":         0.004,
        }
    normal_radius = 0.01
    voxel_norm = 0.005
    # pcd split
    t0 = time.time()
    # pcd_p_ori, pcds = split_obj_by_color(path)
    # pcd_p_ori, pcds = split_pcd_by_color(path,visualize=False,is_cluster=False)
    pcd_p_ori = o3d.io.read_point_cloud(path)
    ct = pcd_p_ori.get_center()
    pts = np.asarray(pcd_p_ori.points)
    z_min = min(pts[:,2])
    # plane = plane_pcd()
    plane = o3d.io.read_point_cloud('/root/catkin_ws/src/detection/scripts/FoundationStereo-master/output_ros/sam_objects/cloud_object26.ply')
    # plane.translate([ct[0], ct[1], -0.0005], relative=True)
    # plane.translate([ct[0], ct[1], z_min], relative=True)
    t1 = time.time()
    cprint(f'Time for partfield:{t1-t0}','green')

    pcd_p_ori.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    plane_n = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (len(plane.points), 1))
    plane.normals = o3d.utility.Vector3dVector(plane_n)

    pcd_with_plane = pcd_p_ori + plane
    # pcd_with_plane = pcd_p_ori 
    
    index_path = os.path.join("/root/catkin_ws/src/more_than_grasp/sample_by_match/dataset/database_1k", "global_index_ems.pkl")
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

    # ------- (B) 回溯轮次：每轮取“后续 5 个”，对齐→生成 grasps→筛选；每轮清空旧 grasp -------
    stage3_pool = np.empty((0, 13), dtype=float)      # 存最终 width 的 grasp
    stage3_w0_pool = np.empty((0,), dtype=float)      # 与 stage3_pool 对齐：存 orig width（增宽前）
    stage3_bucket_state = {"buckets": {}}             # 桶化索引（跨轮复用）
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
            grasps = transform_grasps_from_norm(grasps, part["center"])  # 你的原函数
            round_grasps_world.append(grasps)

        # ranked 用尽：返回 stage3_pool（有多少算多少）
        if not any_added:
            print("[Backtrack] ranked exhausted.")
            if stage3_pool.shape[0] > 0:
                filtered_grasps = stage3_pool
                filtered_w0 = stage3_w0_pool if stage3_w0_pool.shape[0] == stage3_pool.shape[0] else stage3_pool[:, 12].copy()
                print(f"[Backtrack] return stage3_pool={stage3_pool.shape[0]}")
            else:
                filtered_grasps = np.empty((0, 13), dtype=float)
                filtered_w0 = np.empty((0,), dtype=float)
                print("[Backtrack] return empty")
            break

        if len(round_grasps_world) == 0:
            print(f"[Backtrack] round={round_id} produced no grasps, try next batch...")
            round_id += 1
            continue

        all_grasps_stack = np.vstack(round_grasps_world)

        # -------- (2) 把历史 stage3_pool 注入本轮过滤候选（几何判断用最终 width）--------
        if stage3_pool.shape[0] > 0:
            all_for_filter = np.vstack([stage3_pool, all_grasps_stack])

            # 关键：为 filter_grasps 提供 orig width 外部数组（pool 部分用 stage3_w0_pool，新候选部分用当前 width）
            w0_external = np.concatenate([
                stage3_w0_pool,
                all_grasps_stack[:, 12].copy()
            ], axis=0)
        else:
            all_for_filter = all_grasps_stack
            w0_external = all_grasps_stack[:, 12].copy()

        # -------- (3) 过滤：返回本轮 Stage2/Stage3 结论 + keep3_grasps + (final_w0/keep3_w0) --------
        final_grasps, kept_ids, stage_tag, keep3_grasps, keep3_ids, final_w0, keep3_w0 = filter_grasps(
            all_grasps_world=all_for_filter,
            p_pcd=pcd_with_plane,
            hand_params=HAND_PARAMS_REAL,

            eps_scale=2,
            collision_slack_scale=0.0,
            min_points_between=10,
            margin_y_between=0.005,
            widen_after_layer2=0.01,
            enable_widen_after_layer2=True,
            span_thresh=0.006,
            angle_thresh_deg=20.0,
            xz_res=0.005,
            min_pts_line=3,
            y_percentile=3,
            min_keep=15,

            return_meta=True,
            width_orig_external=w0_external,
            return_width_orig=True
        )

        # -------- (4) 无论本轮是否达标，都累计“本轮单个 Stage3 通过”的 grasp（同步 orig width）--------
        if keep3_grasps is not None and keep3_grasps.shape[0] > 0:
            stage3_pool, stage3_w0_pool, stage3_bucket_state = merge_unique_grasps(
                stage3_pool, stage3_w0_pool,
                keep3_grasps, keep3_w0,
                pos_thr=0.005, rot_thr_deg=5.0, width_thr=0.01,
                max_total=None,
                pos_bin=0.01, dir_bin=0.2, width_bin=0.02,
                search_neighbor_buckets=False,
                bucket_state=stage3_bucket_state
            )

        # -------- (5) 终止条件：Stage3 先达标 or Stage2 先达标 --------
        if stage_tag == 3:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            print(f"[Backtrack] round={round_id} Stage3 pass -> return {filtered_grasps.shape[0]} (pool={stage3_pool.shape[0]}) ranks[{start}:{end})")
            break

        if stage_tag == 2:
            filtered_grasps = final_grasps
            filtered_w0 = final_w0
            print(f"[Backtrack] round={round_id} Stage2 pass -> return {filtered_grasps.shape[0]} (pool={stage3_pool.shape[0]}) ranks[{start}:{end})")
            break

        print(f"[Backtrack] round={round_id} Stage2&3 fail, keep stage3_pool={stage3_pool.shape[0]}, continue next batch...")
        round_id += 1
        continue


    if filtered_grasps.shape[0] > 15:
        down = np.array([0.0, 0.0, -1.0], dtype=float)
        # down = np.array([0.0, 1.0, -3.0], dtype=float)

        approaches = filtered_grasps[:, 6:9]
        a_unit = approaches / (np.linalg.norm(approaches, axis=1, keepdims=True) + 1e-12)
        dots = (a_unit @ down.reshape(3, 1)).reshape(-1)

        idx_sorted = np.argsort(-dots)
        filtered_sorted = filtered_grasps[idx_sorted]
        filtered_w0_sorted = filtered_w0[idx_sorted] if filtered_w0 is not None and filtered_w0.shape[0] == filtered_grasps.shape[0] else filtered_sorted[:, 12].copy()

        if filtered_sorted.shape[0] >= 30:
            kcap = min(30, filtered_sorted.shape[0])
            filtered_sorted = filtered_sorted[:kcap]
            filtered_w0_sorted = filtered_w0_sorted[:kcap]

            filtered_grasps = select_diverse_grasps(
                filtered_sorted,
                target_k=15,
                pos_thr=0.005,
                rot_thr_deg=5.0,
                width_thr=0.01,
                prekeep_max=60,
                width_for_similarity=filtered_w0_sorted   
            )
            print(f'Select diverse grasps:{filtered_grasps.shape[0]}')
        else:
            filtered_grasps = filtered_sorted[:15]
            print(f'Select down grasp:{filtered_grasps.shape[0]}')

    else:
        print("Only grasps:", filtered_grasps.shape[0])

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

    output_grasp = convert_all_grasps_pos(
        filtered_grasps, HAND_PARAMS_REAL, src="palm", dst="tip"
    )

    return output_grasp

# -------------------- Main --------------------
if __name__ == "__main__":

    ply_p = '/root/catkin_ws/src/detection/scripts/FoundationStereo-master/output_ros/sam_objects/cloud_object20.ply'

    # t1 = time.time()
    grasp_pose = pmatch(ply_p)
    # grasp_pose = main(obj_p)
    # t2 = time.time()
    # print(t2-t1)
 
