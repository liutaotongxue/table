"""
直接从NPZ深度文件计算桌面平面方程
自动运行，无需交互输入
"""

import numpy as np
import os
from pathlib import Path
import open3d as o3d

def load_npz_depth_data(npz_path):
    """加载NPZ文件中的深度数据"""
    try:
        data = np.load(npz_path)
        print(f"NPZ文件键值: {list(data.keys())}")
        
        # 尝试不同的可能键名
        depth_mm = None
        if 'depth' in data:
            depth_mm = data['depth'].astype(np.float32)
        elif 'depth_mm' in data:
            depth_mm = data['depth_mm'].astype(np.float32)
        elif 'arr_0' in data:
            depth_mm = data['arr_0'].astype(np.float32)
        else:
            # 使用第一个数组
            keys = list(data.keys())
            if keys:
                depth_mm = data[keys[0]].astype(np.float32)
                print(f"使用键: {keys[0]}")
        
        if depth_mm is not None:
            print(f"深度图尺寸: {depth_mm.shape}")
            print(f"深度范围: {np.min(depth_mm[depth_mm > 0]):.1f} - {np.max(depth_mm):.1f} mm")
            print(f"有效点数: {np.sum(depth_mm > 0)}")
        
        return depth_mm
    except Exception as e:
        print(f"加载NPZ文件失败: {e}")
        return None

def depth_to_point_cloud(depth_mm, fx=214.15, fy=212.80, cx=162.52, cy=127.85):
    """将深度图转换为3D点云"""
    h, w = depth_mm.shape
    
    # 创建像素坐标网格
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    # 有效深度掩码 - 假设桌面在合理范围内
    valid_mask = (depth_mm > 200) & (depth_mm < 1500)  # 20cm到1.5m
    
    # 转换为相机坐标系
    z = depth_mm[valid_mask] / 1000.0  # 转换为米
    x = (xx[valid_mask] - cx) * z / fx
    y = (yy[valid_mask] - cy) * z / fy
    
    points_3d = np.column_stack([x, y, z])
    
    print(f"生成3D点数: {len(points_3d)}")
    
    return points_3d

def fit_plane_ransac(points_3d, distance_threshold=0.02, num_iterations=1000):
    """使用RANSAC拟合平面"""
    if len(points_3d) < 100:
        print("点数太少，无法拟合平面")
        return None, 0
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # RANSAC平面拟合
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations
    )
    
    if len(inliers) < 50:
        print("内点太少，平面拟合质量差")
        return None, 0
    
    # 平面参数 [a, b, c, d]: ax + by + cz + d = 0
    a, b, c, d = plane_model
    
    # 确保法向量指向一致（通常c应该是负的，表示z轴向上）
    if c > 0:
        a, b, c, d = -a, -b, -c, -d
    
    print(f"平面拟合成功！内点数: {len(inliers)} / {len(points_3d)} ({100*len(inliers)/len(points_3d):.1f}%)")
    
    return np.array([a, b, c, d]), len(inliers)

def analyze_plane(plane_model):
    """分析平面特性"""
    if plane_model is None:
        return None
    
    a, b, c, d = plane_model
    
    # 归一化法向量
    norm = np.sqrt(a**2 + b**2 + c**2)
    nx, ny, nz = a/norm, b/norm, c/norm
    
    # 计算与z轴的夹角
    angle_with_z = np.arccos(abs(nz)) * 180 / np.pi
    
    # 原点到平面的距离
    distance = abs(d) / norm
    
    print(f"\n桌面平面方程:")
    print(f"  {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(f"\n平面参数:")
    print(f"  a = {a:.6f}")
    print(f"  b = {b:.6f}")
    print(f"  c = {c:.6f}")
    print(f"  d = {d:.6f}")
    print(f"\n平面分析:")
    print(f"  归一化法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
    print(f"  与垂直方向夹角: {angle_with_z:.1f}° {'(接近水平)' if angle_with_z < 15 else '(倾斜)'}")
    print(f"  原点到平面距离: {distance:.3f}m ({distance*1000:.1f}mm)")
    
    return {
        'plane': plane_model,
        'normal': [nx, ny, nz],
        'angle': angle_with_z,
        'distance': distance
    }

def process_npz_file(npz_path):
    """处理单个NPZ文件"""
    print(f"\n{'='*60}")
    print(f"从NPZ文件计算桌面平面方程")
    print(f"文件: {npz_path}")
    print(f"{'='*60}")
    
    # 1. 加载深度数据
    depth_mm = load_npz_depth_data(npz_path)
    if depth_mm is None:
        return None
    
    # 2. 转换为3D点云
    points_3d = depth_to_point_cloud(depth_mm)
    
    # 3. RANSAC拟合平面
    plane_model, inlier_count = fit_plane_ransac(points_3d)
    
    if plane_model is None:
        print("平面拟合失败")
        return None
    
    # 4. 分析平面
    result = analyze_plane(plane_model)
    
    return result

def batch_process_multiple_files():
    """批量处理多个文件找出最佳平面"""
    # 修改这里为您的NPZ文件所在目录
    depth_dir = Path("C:/Users/14101/Desktop/table/pre_depth")  # 直接指定完整路径
    
    print(f"\n批量处理NPZ文件...")
    
    # 尝试几个不同的文件索引
    test_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    results = []
    
    for idx in test_indices:
        npz_path = depth_dir / f"{idx}.npz"
        if npz_path.exists():
            print(f"\n处理文件 {idx}.npz...")
            try:
                result = process_npz_file(str(npz_path))
                if result and result['angle'] < 30:  # 只保留相对水平的平面
                    results.append({
                        'index': idx,
                        'plane': result['plane'],
                        'angle': result['angle'],
                        'distance': result['distance']
                    })
            except Exception as e:
                print(f"处理文件 {idx}.npz 时出错: {e}")
                continue
    
    if results:
        # 找出最接近水平的平面
        best = min(results, key=lambda x: x['angle'])
        print(f"\n{'='*60}")
        print(f"最佳桌面平面 (文件 {best['index']}.npz):")
        plane = best['plane']
        print(f"  方程: {plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0")
        print(f"  与垂直夹角: {best['angle']:.1f}°")
        print(f"  距离原点: {best['distance']*1000:.1f}mm")
        print(f"{'='*60}")
        
        return best['plane']
    else:
        print("没有找到合适的桌面平面")
        return None

def main():
    """主函数"""
    print("NPZ深度文件 -> 桌面平面方程 计算器")
    print("="*60)
    
    # 设置文件路径
    # 修改这里为您的NPZ文件所在目录
    depth_dir = Path("C:/Users/14101/Desktop/table/pre_depth")  # 直接指定完整路径
    
    # 先尝试单个文件
    test_file = depth_dir / "50.npz"  # 修改这里为您要测试的文件名
    
    if test_file.exists():
        print("方式1: 处理单个文件...")
        result = process_npz_file(str(test_file))
        
        if result:
            print(f"\n成功从单个文件获得平面方程！")
        else:
            print(f"\n单个文件处理失败，尝试批量处理...")
            batch_process_multiple_files()
    else:
        print(f"文件 {test_file} 不存在，尝试批量处理...")
        batch_process_multiple_files()

if __name__ == "__main__":
    main()