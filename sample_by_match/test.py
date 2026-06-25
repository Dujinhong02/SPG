import numpy as np
import open3d as o3d

def get_mask_bottom_origin_z_axis(local_points, radius, height, angle_deg=30.0):
    """
    输入: 已经变换到局部坐标系的点云 (N, 3)
    
    坐标系严格定义:
      - Origin (0,0,0): 底面中心 (Bottom Center)
      - Z axis (+Z): 顶面 -> 底面 (Top -> Bottom)
        -> 推导: 底面是0，顶面是-Height。圆柱体位于 Z轴的 [-Height, 0] 区间。
      - Y axis: 平行于切线
      - X axis: 右手系生成的切面法线 (控制宽度)
    """
    x = local_points[:, 0]
    y = local_points[:, 1]
    z = local_points[:, 2]
    
    # --- 1. 高度约束 (Z轴) ---
    # 关键修正点: 有效区间是 [-Height, 0]
    mask_height = (z >= -height) & (z <= 0)
    
    # --- 2. 切面宽度约束 (X轴) ---
    # X轴是切面的法线，控制保留宽度
    cut_half_width = radius * np.sin(np.deg2rad(angle_deg))
    mask_cut = np.abs(x) <= cut_half_width
    
    # --- 3. 圆柱半径约束 (XY平面) ---
    # 截面是垂直于 Z轴 的 XY 平面
    mask_radius = (x**2 + y**2) <= radius**2
    
    # --- 4. 综合 ---
    final_mask = mask_height & mask_cut & mask_radius
    
    return final_mask, cut_half_width

def visualize_correction(local_points, radius, height, angle_deg=30.0):
    # 1. 计算 Mask
    mask, cut_w = get_mask_bottom_origin_z_axis(local_points, radius, height, angle_deg)
    
    print(f"输入点数: {len(local_points)}")
    print(f"内部点数: {np.sum(mask)}")
    
    # 2. 颜色区分
    pcd_in = o3d.geometry.PointCloud()
    pcd_in.points = o3d.utility.Vector3dVector(local_points[mask])
    pcd_in.paint_uniform_color([1, 0, 0]) # 红色 (内部)
    
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(local_points[~mask])
    pcd_out.paint_uniform_color([0.9, 0.9, 0.9]) # 灰色 (外部)
    
    # 3. 坐标轴 (原点 = 底面中心)
    # 红色=X, 绿色=Y, 蓝色=Z
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=radius, origin=[0,0,0])
    
    # 4. 绘制蓝色线框 (辅助验证)
    # 这里的关键是 Z 的坐标范围是 [-height, 0]
    
    # 定义8个顶点
    # Z=0 (底面), Z=-height (顶面)
    # X= +/- cut_w
    # Y= +/- radius (为了画框方便，画个外接矩形)
    
    box_corners = [
        [-cut_w, -radius, 0],       [-cut_w, radius, 0],       # Bottom (-X)
        [cut_w, -radius, 0],        [cut_w, radius, 0],        # Bottom (+X)
        [-cut_w, -radius, -height], [-cut_w, radius, -height], # Top (-X)
        [cut_w, -radius, -height],  [cut_w, radius, -height]   # Top (+X)
    ]
    
    # 连接线
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0], # 底面框 (Z=0)
        [4, 5], [5, 7], [7, 6], [6, 4], # 顶面框 (Z=-height)
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接棱
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(box_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0, 0, 1]) # 蓝色
    
    print("\n可视化验证 (Bottom Origin + Z Axis Corrected):")
    print("1. [坐标原点]: 位于底面中心。")
    print("2. [蓝色轴 Z]: 指向 Top->Bottom 方向。")
    print("   -> 注意: 红色点云应该位于蓝色轴的**反方向** (负半轴)。")
    print("   -> 也就是 Z轴 像尾巴一样从红色点云的屁股后面伸出来。")
    print("3. [蓝色线框]: 完美包裹住红色点云，且 Z 范围是 [0 到 -H]。")
    
    o3d.visualization.draw_geometries(
        [pcd_in, pcd_out, frame, line_set],
        window_name="Corrected: Bottom Origin, Z-Axis",
        width=1024, height=768
    )

# ==========================================
#   测试脚本
# ==========================================
if __name__ == "__main__":
    # 生成测试点云
    # 注意: 为了验证负轴，我们需要把点云生成在 Z 的负半轴区域
    N = 50000
    pts_x = np.random.uniform(-60, 60, N)
    pts_y = np.random.uniform(-60, 60, N)
    # Z轴生成在 [-100, 20] 之间，覆盖圆柱体 [-80, 0]
    pts_z = np.random.uniform(-100, 20, N)
    
    local_pts = np.column_stack((pts_x, pts_y, pts_z))
    
    # 几何参数
    R = 40.0
    H = 80.0
    Angle = 30.0
    
    visualize_correction(local_pts, R, H, Angle)