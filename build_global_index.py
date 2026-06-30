# import os
# import glob
# import json
# import pickle
# import argparse
# import numpy as np
# from tqdm import tqdm
# from collections import Counter

# def build_index(dataset_root, index_filename="global_index.pkl"):
#     # 1. 确定搜索路径
#     root_abs = os.path.abspath(dataset_root)
#     # 搜索 pattern: database_flat/*/cache.npz
#     search_pattern = os.path.join(root_abs, "*", "cache.npz")
#     files = sorted(glob.glob(search_pattern))
    
#     if not files:
#         print(f"[Error] No cache.npz files found in {root_abs}")
#         return

#     print(f"[Info] Found {len(files)} objects. Building optimized index...")
    
#     index_data = []
    
#     for fpath in tqdm(files):
#         try:
#             # 获取文件夹名称作为 ID (例如 "000")
#             obj_id = os.path.basename(os.path.dirname(fpath))
            
#             # 读取 npz
#             # 我们只需要轻量级数据：pairs, sobb, scale
#             with np.load(fpath) as data:
#                 # [优化] 预先将 pairs 转为 Counter 直方图
#                 # 这样匹配时就不需要实时计算 Counter 了，极速！
#                 pairs_array = data["pairs"]
#                 # 注意：numpy array 转 tuple 才能作为字典 key
#                 pairs_counter = Counter(map(tuple, pairs_array.tolist()))
                
#                 entry = {
#                     "id": obj_id,
#                     "path": fpath,          # 完整路径，方便后续读取重型数据
#                     "counts": pairs_counter,# 预计算的特征直方图 (用于 QS)
#                     "sobb": data["sobb"],   # 尺寸 (用于 SS)
#                     "scale": float(data["scale"]) # 缩放比例
#                 }
                
#             # 读取参数校验用的 meta.json
#             meta_path = os.path.join(os.path.dirname(fpath), "meta.json")
#             if os.path.exists(meta_path):
#                 with open(meta_path, 'r') as mf:
#                     meta = json.load(mf)
#                     entry["params"] = meta.get("params", {})
            
#             index_data.append(entry)
                
#         except Exception as e:
#             print(f"[Warn] Failed to process {fpath}: {e}")
            
#     # 2. 保存到 dataset_root 根目录下
#     out_path = os.path.join(root_abs, index_filename)
    
#     print(f"[Info] Saving index to: {out_path}")
#     with open(out_path, "wb") as f:
#         pickle.dump(index_data, f)
    
#     # 打印统计信息
#     size_mb = os.path.getsize(out_path) / 1024 / 1024
#     print(f"[Success] Index built! Size: {size_mb:.2f} MB")
#     print(f"          Contains {len(index_data)} entries.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Build fast in-memory index for the database")
#     # 默认路径设为您的 database_flat
#     parser.add_argument("--root", default="shot_fpfh/descriptors/dataset/database_1k", 
#                         help="Path to the database root folder")
#     parser.add_argument("--name", default="global_index.pkl", 
#                         help="Name of the index file")
    
#     args = parser.parse_args()
    
#     build_index(args.root, args.name)


# ems version
import os
import glob
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
import importlib

# def _import_ems():
#     for mod_name, fn_name in [
#         ("EMS.EMS_recovery", "EMS_recovery"),
#         ("EMS", "EMS_recovery"),
#     ]:
#         try:
#             m = importlib.import_module(mod_name)
#             fn = getattr(m, fn_name, None)
#             if callable(fn):
#                 return fn
#         except Exception:
#             pass
#     raise ImportError("Cannot import EMS_recovery from EMS.*")

def _import_ems():
    try:
        from EMS.EMS_recovery import EMS_recovery
        return EMS_recovery
    except Exception:
        import EMS.EMS_recovery as ems_mod
        return ems_mod.EMS_recovery
    
def gmean3(x):
    x = np.asarray(x, dtype=float)
    return float((x[0] * x[1] * x[2]) ** (1.0 / 3.0))

def fit_ems_from_points(points_xyz: np.ndarray,
                        max_points=6000,
                        center_mode="center",   # 新增：和你在线一致
                        seed=0):
    EMS_recovery = _import_ems()

    pts = np.asarray(points_xyz, dtype=float)
    if pts.shape[0] < 30:
        return None

    # 1) 先中心化（和你在线 translate(-center) 对齐）
    shift = np.zeros(3, dtype=float)
    if center_mode == "center":
        shift = pts.mean(axis=0)
        pts = pts - shift

    # 2) 可选：随机下采样加速（离线也可做）
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], max_points, replace=False)
        pts_fit = pts[idx]
    else:
        pts_fit = pts

    # 3) EMS 拟合（离线只跑一次即可，不做 autotune）
    try:
        sq, _ = EMS_recovery(pts_fit)
    except Exception as e:
        print(f"[Warn] EMS_recovery failed: {e}")
        return None

    shape = np.array(getattr(sq, "shape", getattr(sq, "_shape", None)), dtype=float).reshape(-1)
    scale = np.array(getattr(sq, "scale", getattr(sq, "_scale", None)), dtype=float).reshape(-1)
    t = np.array(getattr(sq, "translation", getattr(sq, "_translation", [0,0,0])), dtype=float).reshape(-1)

    if shape.size != 2 or scale.size != 3 or t.size != 3:
        return None

    scale = np.abs(scale) + 1e-12
    ratio = scale / (gmean3(scale) + 1e-12)
    # ratio_sorted = np.sort(scale / (gmean3(scale) + 1e-12))[::-1]

    # 4) 旋转：优先 _r Rotation
    R = None
    r_obj = getattr(sq, "_r", None)
    if r_obj is not None:
        try:
            R = r_obj.as_matrix()
            quat = r_obj.as_quat()  # (x,y,z,w)
        except Exception:
            R = None
            quat = None
    else:
        quat = None

    if R is None:
        # 没有旋转就退化为单位阵（至少能跑）
        R = np.eye(3, dtype=float)

    # 5) 关键：构造并保存 T_sq（SQ->PC），注意 PC 是“中心化后的点云坐标”
    T_sq = np.eye(4, dtype=float)
    T_sq[:3, :3] = R
    T_sq[:3,  3] = t

    return {
        "ems_shape": shape.tolist(),                  # [eps1, eps2]
        "ems_scale": scale.tolist(),                  # [a,b,c] 绝对长度
        "ems_ratio": ratio.tolist(),    # 无尺度轴比
        "ems_quat": quat.tolist() if quat is not None else None,
        "ems_t": t.tolist(),                          # (可留着 debug)
        "T_sq": T_sq.tolist(),                        
        "center_shift": shift.tolist(),               
        "center_mode": center_mode,
    }



# =============== 原 build_index 主体（增加 ems 字段） ===============
def build_index(dataset_root, index_filename="global_index.pkl", do_ems=True):
    root_abs = os.path.abspath(dataset_root)
    search_pattern = os.path.join(root_abs, "*", "cache.npz")
    files = sorted(glob.glob(search_pattern))

    if not files:
        print(f"[Error] No cache.npz files found in {root_abs}")
        return

    print(f"[Info] Found {len(files)} objects. Building index... (EMS={do_ems})")

    index_data = []

    for fpath in tqdm(files):
        try:
            obj_id = os.path.basename(os.path.dirname(fpath))

            with np.load(fpath) as data:
                pairs_array = data["pairs"]
                pairs_counter = Counter(map(tuple, pairs_array.tolist()))

                entry = {
                    "id": obj_id,
                    "path": fpath,
                    "counts": pairs_counter,
                    "sobb": data["sobb"].astype(float),
                    "scale": float(data.get("scale", 1.0)),
                }

                # [新增] EMS 参数
                if do_ems:
                    pts = data["points"].astype(float)
                    ems = fit_ems_from_points(pts, center_mode="center", max_points=6000, seed=0)
                    entry["ems"] = ems

            meta_path = os.path.join(os.path.dirname(fpath), "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                    entry["params"] = meta.get("params", {})

            index_data.append(entry)

        except Exception as e:
            print(f"[Warn] Failed to process {fpath}: {e}")

    out_path = os.path.join(root_abs, index_filename)
    print(f"[Info] Saving index to: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(index_data, f)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    ok_ems = sum(1 for x in index_data if x.get("ems") is not None)
    print(f"[Success] Index built! Size: {size_mb:.2f} MB")
    print(f"          Contains {len(index_data)} entries. EMS_ok={ok_ems}/{len(index_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build global index with optional EMS params")
    parser.add_argument("--root", default="shot_fpfh/descriptors/dataset/database_2k")
    parser.add_argument("--name", default="global_index_ems.pkl")
    parser.add_argument("--no_ems", action="store_true", help="Disable EMS fitting")
    args = parser.parse_args()

    build_index(args.root, args.name, do_ems=(not args.no_ems))