"""
直接从NPZ深度文件计算桌面平面方程
不需要相机硬件，不需要YOLO检测
"""

import numpy as np
import os
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_npz_depth_data(npz_path):
    """加载NPZ文件中的深度数据"""
    try:
        data = np.load(npz_path)
        print(f"\nNPZ文件键值: {list(data.keys())}")
        
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
    
    # 有效深度掩码
    valid_mask = (depth_mm > 100) & (depth_mm < 2000)  # 10cm到2m
    
    # 转换为相机坐标系
    z = depth_mm[valid_mask] / 1000.0  # 转换为米
    x = (xx[valid_mask] - cx) * z / fx
    y = (yy[valid_mask] - cy) * z / fy
    
    points_3d = np.column_stack([x, y, z])
    
    print(f"生成3D点数: {len(points_3d)}")
    
    return points_3d, valid_mask

def fit_plane_ransac(points_3d, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """使用RANSAC拟合平面"""
    if len(points_3d) < 100:
        print("点数太少，无法拟合平面")
        return None, None
    
    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # RANSAC平面拟合
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    if len(inliers) < 50:
        print("内点太少，平面拟合质量差")
        return None, None
    
    # 平面参数 [a, b, c, d]: ax + by + cz + d = 0
    a, b, c, d = plane_model
    
    # 确保法向量指向一致（通常c应该是负的，表示z轴向上）
    if c > 0:
        a, b, c, d = -a, -b, -c, -d
    
    print(f"\n✅ 平面拟合成功！")
    print(f"  内点数: {len(inliers)} / {len(points_3d)} ({100*len(inliers)/len(points_3d):.1f}%)")
    
    return np.array([a, b, c, d]), inliers

def analyze_plane(plane_model):
    """分析平面特性"""
    if plane_model is None:
        return
    
    a, b, c, d = plane_model
    
    # 归一化法向量
    norm = np.sqrt(a**2 + b**2 + c**2)
    nx, ny, nz = a/norm, b/norm, c/norm
    
    # 计算与z轴的夹角
    angle_with_z = np.arccos(abs(nz)) * 180 / np.pi
    
    # 原点到平面的距离
    distance = abs(d) / norm
    
    print(f"\n平面分析：")
    print(f"  平面方程: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(f"  归一化法向量: ({nx:.3f}, {ny:.3f}, {nz:.3f})")
    print(f"  与垂直方向夹角: {angle_with_z:.1f}°")
    print(f"  原点到平面距离: {distance:.3f}m ({distance*1000:.1f}mm)")
    
    if angle_with_z < 10:
        print(f"  判断: 这是一个接近水平的平面（桌面）")
    elif angle_with_z > 80:
        print(f"  判断: 这是一个接近垂直的平面（可能是墙面）")
    else:
        print(f"  判断: 这是一个倾斜的平面")
    
    return {
        'plane': plane_model,
        'normal': [nx, ny, nz],
        'angle': angle_with_z,
        'distance': distance
    }

def visualize_point_cloud_and_plane(points_3d, plane_model, inliers):
    """可视化点云和拟合的平面"""
    fig = plt.figure(figsize=(12, 5))
    
    # 3D点云视图
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 下采样显示
    sample_indices = np.random.choice(len(points_3d), min(5000, len(points_3d)), replace=False)
    sampled_points = points_3d[sample_indices]
    
    # 判断哪些是内点
    is_inlier = np.zeros(len(points_3d), dtype=bool)
    is_inlier[inliers] = True
    sampled_is_inlier = is_inlier[sample_indices]
    
    # 绘制外点（蓝色）和内点（红色）
    ax1.scatter(sampled_points[~sampled_is_inlier, 0], 
               sampled_points[~sampled_is_inlier, 1], 
               sampled_points[~sampled_is_inlier, 2], 
               c='blue', s=1, alpha=0.5, label='外点')
    ax1.scatter(sampled_points[sampled_is_inlier, 0], 
               sampled_points[sampled_is_inlier, 1], 
               sampled_points[sampled_is_inlier, 2], 
               c='red', s=1, alpha=0.8, label='内点（平面）')
    
    # 绘制拟合的平面
    if plane_model is not None:
        a, b, c, d = plane_model
        
        # 在内点范围内创建平面网格
        inlier_points = points_3d[inliers]
        x_range = [inlier_points[:, 0].min(), inlier_points[:, 0].max()]
        y_range = [inlier_points[:, 1].min(), inlier_points[:, 1].max()]
        
        xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], 20),
                            np.linspace(y_range[0], y_range[1], 20))
        
        # 从平面方程计算z: ax + by + cz + d = 0 => z = -(ax + by + d) / c
        if abs(c) > 0.01:
            zz = -(a * xx + b * yy + d) / c
            ax1.plot_surface(xx, yy, zz, alpha=0.3, color='green')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D点云和拟合平面')
    ax1.legend()
    
    # 深度图视图
    ax2 = fig.add_subplot(122)
    depth_display = points_3d[:, 2].reshape(-1)
    depth_img = np.zeros((240, 320))  # 假设标准深度图尺寸
    valid_indices = np.where(is_inlier)[0]
    
    # 这里需要知道原始的像素坐标，简化处理
    ax2.hist(points_3d[inliers, 2], bins=50, alpha=0.7, color='red', label='平面深度分布')
    ax2.set_xlabel('深度 (m)')
    ax2.set_ylabel('点数')
    ax2.set_title('平面点深度分布')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def process_single_npz(npz_path, visualize=True):
    """处理单个NPZ文件"""
    print(f"\n{'='*60}")
    print(f"处理文件: {npz_path}")
    print(f"{'='*60}")
    
    # 1. 加载深度数据
    depth_mm = load_npz_depth_data(npz_path)
    if depth_mm is None:
        return None
    
    # 2. 转换为3D点云
    points_3d, valid_mask = depth_to_point_cloud(depth_mm)
    
    # 3. RANSAC拟合平面
    plane_model, inliers = fit_plane_ransac(points_3d)
    
    if plane_model is None:
        return None
    
    # 4. 分析平面
    result = analyze_plane(plane_model)
    
    # 5. 可视化（可选）
    if visualize and result is not None:
        visualize_point_cloud_and_plane(points_3d, plane_model, inliers)
    
    return result

def batch_process_npz_files(directory, max_files=10):
    """批量处理NPZ文件"""
    npz_files = list(Path(directory).glob("*.npz"))
    print(f"找到 {len(npz_files)} 个NPZ文件")
    
    results = []
    
    for i, npz_file in enumerate(npz_files[:max_files]):
        print(f"\n[{i+1}/{min(len(npz_files), max_files)}] 处理: {npz_file.name}")
        result = process_single_npz(str(npz_file), visualize=False)
        
        if result is not None:
            results.append({
                'file': npz_file.name,
                'plane': result['plane'],
                'angle': result['angle'],
                'distance': result['distance']
            })
    
    # 找出最佳平面（最接近水平）
    if results:
        best = max(results, key=lambda x: x['angle'])
        print(f"\n{'='*60}")
        print(f"最佳桌面平面（最接近水平）:")
        print(f"  文件: {best['file']}")
        print(f"  方程: {best['plane'][0]:.6f}x + {best['plane'][1]:.6f}y + {best['plane'][2]:.6f}z + {best['plane'][3]:.6f} = 0")
        print(f"  与垂直夹角: {best['angle']:.1f}°")
        print(f"{'='*60}")
        
        return best['plane']
    
    return None

def main():
    """主函数"""
    print("NPZ深度文件 -> 桌面平面方程 转换工具")
    print("="*60)
    
    # 设置路径
    base_path = Path("C:/Users/14101/Desktop/table")
    depth_dir = base_path / "pre_depth"
    
    print("\n选择处理模式:")
    print("1. 处理单个NPZ文件（带可视化）")
    print("2. 批量处理多个文件（找出最佳平面）")
    print("3. 处理指定文件")
    
    choice = input("\n请选择 (1/2/3，默认1): ").strip() or "1"
    
    if choice == "1":
        # 处理单个文件
        file_idx = input("输入文件索引 (0-193，默认50): ").strip() or "50"
        npz_path = depth_dir / f"{file_idx}.npz"
        
        if npz_path.exists():
            result = process_single_npz(str(npz_path), visualize=True)
            
            if result:
                # 保存结果
                save = input("\n是否保存这个平面方程? (y/n): ")
                if save.lower() == 'y':
                    import json
                    output_file = "detected_plane.json"
                    with open(output_file, 'w') as f:
                        json.dump({
                            'plane_model': result['plane'].tolist(),
                            'normal': result['normal'],
                            'angle_with_vertical': result['angle'],
                            'distance_from_origin': result['distance'],
                            'source_file': str(npz_path)
                        }, f, indent=4)
                    print(f"平面方程已保存到: {output_file}")
        else:
            print(f"文件不存在: {npz_path}")
    
    elif choice == "2":
        # 批量处理
        max_files = int(input("处理多少个文件? (默认10): ").strip() or "10")
        best_plane = batch_process_npz_files(str(depth_dir), max_files)
    
    elif choice == "3":
        # 处理指定文件
        filename = input("输入NPZ文件名: ").strip()
        npz_path = depth_dir / filename
        
        if npz_path.exists():
            process_single_npz(str(npz_path), visualize=True)
        else:
            print(f"文件不存在: {npz_path}")

if __name__ == "__main__":
    main()