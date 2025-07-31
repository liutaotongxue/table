"""
桌面相机视角的平面检测系统
专门优化用于检测相机下方的水平桌面
坐标系：X-左右，Y-上下，Z-前后（深度）
"""

import numpy as np
import os
from pathlib import Path
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============= 配置区域 =============
CONFIG = {
    # NPZ文件设置
    "NPZ_DIRECTORY": "C:/Users/14101/Desktop/table/pre_depth",
    "TEST_FILE": "0.npz",
    "BATCH_FILES": ["10.npz", "20.npz", "30.npz", "40.npz", "50.npz", 
                   "60.npz", "70.npz", "80.npz", "90.npz", "100.npz"],
    
    # 深度过滤范围（毫米）
    "MIN_DEPTH_MM": 100,      # 降低最小深度，捕获近处桌面
    "MAX_DEPTH_MM": 800,      # 降低最大深度，排除远处物体
    "TABLE_DEPTH_MM": 400,    # 预期桌面深度范围
    
    # 相机内参
    "CAMERA_FX": 214.15,
    "CAMERA_FY": 212.80,
    "CAMERA_CX": 162.52,
    "CAMERA_CY": 127.85,
    
    # RANSAC参数
    "RANSAC_DISTANCE_THRESHOLD": 0.01,  # 更严格的阈值
    "RANSAC_ITERATIONS": 2000,
    
    # 桌面检测参数
    "HEIGHT_TOLERANCE": 0.03,    # 3cm容差
    "MIN_TABLE_POINTS": 500,     # 最少点数
    "HORIZONTAL_THRESHOLD": 0.8,  # 水平判定阈值
    "IMAGE_LOWER_RATIO": 0.6,    # 图像下部比例（用于寻找桌面）
}

class DesktopTableDetector:
    """桌面相机视角的桌面检测器"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.debug_mode = True  # 调试模式
        
    def load_npz_depth_data(self, npz_path):
        """加载NPZ文件中的深度数据"""
        try:
            data = np.load(npz_path)
            print(f"\n加载文件: {npz_path}")
            print(f"NPZ文件键值: {list(data.keys())}")
            
            # 尝试不同的可能键名
            depth_mm = None
            for key in ['depth', 'depth_mm', 'arr_0']:
                if key in data:
                    depth_mm = data[key].astype(np.float32)
                    print(f"使用键: {key}")
                    break
            
            if depth_mm is None and len(data.keys()) > 0:
                key = list(data.keys())[0]
                depth_mm = data[key].astype(np.float32)
                print(f"使用第一个键: {key}")
            
            if depth_mm is not None:
                print(f"深度图尺寸: {depth_mm.shape}")
                valid_depth = depth_mm[depth_mm > 0]
                if len(valid_depth) > 0:
                    print(f"深度范围: {np.min(valid_depth):.1f} - {np.max(valid_depth):.1f} mm")
                    print(f"有效点数: {len(valid_depth)} / {depth_mm.size} ({100*len(valid_depth)/depth_mm.size:.1f}%)")
            
            return depth_mm
            
        except Exception as e:
            print(f"加载NPZ文件失败: {e}")
            return None
    
    def analyze_depth_distribution(self, depth_mm):
        """分析深度分布，识别不同的深度层"""
        h, w = depth_mm.shape
        
        # 分析不同区域的深度分布
        upper_region = depth_mm[:int(h*0.4), :]
        middle_region = depth_mm[int(h*0.4):int(h*0.7), :]
        lower_region = depth_mm[int(h*0.7):, :]
        
        print("\n深度分布分析:")
        
        for region_name, region_data in [("上部", upper_region), 
                                         ("中部", middle_region), 
                                         ("下部", lower_region)]:
            valid_data = region_data[region_data > 0]
            if len(valid_data) > 0:
                print(f"\n{region_name}区域:")
                print(f"  平均深度: {np.mean(valid_data):.1f} mm")
                print(f"  中位深度: {np.median(valid_data):.1f} mm")
                print(f"  深度范围: {np.min(valid_data):.1f} - {np.max(valid_data):.1f} mm")
    
    def visualize_depth_regions(self, depth_mm):
        """可视化深度图的不同区域，专门用于桌面检测"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        h, w = depth_mm.shape
        
        # 1. 原始深度图，标记区域
        ax = axes[0, 0]
        im = ax.imshow(depth_mm, cmap='viridis')
        
        # 标记不同区域
        ax.axhline(y=h*0.4, color='red', linestyle='--', linewidth=2, label='上部边界')
        ax.axhline(y=h*0.7, color='yellow', linestyle='--', linewidth=2, label='下部边界')
        ax.set_title('深度图区域划分')
        ax.legend()
        plt.colorbar(im, ax=ax, label='深度 (mm)')
        
        # 2. 近距离区域（可能的桌面）
        ax = axes[0, 1]
        near_mask = (depth_mm > self.config["MIN_DEPTH_MM"]) & \
                   (depth_mm < self.config["TABLE_DEPTH_MM"])
        near_depth = np.where(near_mask, depth_mm, 0)
        im = ax.imshow(near_depth, cmap='hot')
        ax.set_title(f'近距离区域 ({self.config["MIN_DEPTH_MM"]}-{self.config["TABLE_DEPTH_MM"]}mm)')
        plt.colorbar(im, ax=ax, label='深度 (mm)')
        
        # 3. 图像下部的近距离区域
        ax = axes[0, 2]
        lower_near_mask = near_mask & (np.arange(h)[:, None] > h * self.config["IMAGE_LOWER_RATIO"])
        lower_near_depth = np.where(lower_near_mask, depth_mm, 0)
        im = ax.imshow(lower_near_depth, cmap='plasma')
        ax.set_title('下部近距离区域（桌面候选）')
        plt.colorbar(im, ax=ax, label='深度 (mm)')
        
        # 4. 深度直方图对比
        ax = axes[1, 0]
        upper_depth = depth_mm[:int(h*0.4), :].flatten()
        lower_depth = depth_mm[int(h*0.7):, :].flatten()
        
        upper_valid = upper_depth[upper_depth > 0]
        lower_valid = lower_depth[lower_depth > 0]
        
        if len(upper_valid) > 0:
            ax.hist(upper_valid, bins=50, alpha=0.5, label='上部（天花板/远处）', color='blue')
        if len(lower_valid) > 0:
            ax.hist(lower_valid, bins=50, alpha=0.5, label='下部（桌面候选）', color='red')
        
        ax.set_xlabel('深度 (mm)')
        ax.set_ylabel('像素数')
        ax.set_title('上下部深度分布对比')
        ax.legend()
        ax.set_xlim([0, 1000])
        
        # 5. 深度梯度（找平坦区域）
        ax = axes[1, 1]
        # 计算梯度
        grad_y = np.gradient(depth_mm, axis=0)
        grad_x = np.gradient(depth_mm, axis=1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 只在有效深度区域计算梯度
        valid_mask = depth_mm > 0
        grad_mag[~valid_mask] = np.inf
        
        # 找平坦区域（梯度小）
        flat_threshold = np.percentile(grad_mag[valid_mask], 20)
        flat_regions = grad_mag < flat_threshold
        
        ax.imshow(flat_regions, cmap='gray')
        ax.set_title('平坦区域（低梯度=白色）')
        
        # 6. 综合桌面候选区域
        ax = axes[1, 2]
        # 结合多个条件
        table_candidate = lower_near_mask & flat_regions
        ax.imshow(table_candidate, cmap='RdYlGn')
        ax.set_title('综合桌面候选区域')
        
        plt.tight_layout()
        plt.show()
    
    def depth_to_point_cloud_with_pixel_info(self, depth_mm):
        """将深度图转换为3D点云，保留像素坐标信息"""
        h, w = depth_mm.shape
        
        # 创建像素坐标网格
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # 有效深度掩码
        valid_mask = (depth_mm > self.config["MIN_DEPTH_MM"]) & \
                    (depth_mm < self.config["MAX_DEPTH_MM"])
        
        # 转换为相机坐标系
        z = depth_mm[valid_mask] / 1000.0  # 转换为米
        x = (xx[valid_mask] - self.config["CAMERA_CX"]) * z / self.config["CAMERA_FX"]
        # Y轴：图像Y向下，3D Y向上
        y = -(yy[valid_mask] - self.config["CAMERA_CY"]) * z / self.config["CAMERA_FY"]
        
        points_3d = np.column_stack([x, y, z])
        pixel_coords = np.column_stack([xx[valid_mask], yy[valid_mask]])
        
        print(f"\n生成3D点云:")
        print(f"  总点数: {len(points_3d)}")
        print(f"  X范围: [{np.min(points_3d[:, 0]):.3f}, {np.max(points_3d[:, 0]):.3f}] m")
        print(f"  Y范围: [{np.min(points_3d[:, 1]):.3f}, {np.max(points_3d[:, 1]):.3f}] m")
        print(f"  Z范围: [{np.min(points_3d[:, 2]):.3f}, {np.max(points_3d[:, 2]):.3f}] m")
        
        return points_3d, pixel_coords
    
    def find_table_in_lower_region(self, points_3d, pixel_coords):
        """在图像下部区域寻找桌面"""
        h_threshold = 240 * self.config["IMAGE_LOWER_RATIO"]  # 假设图像高度240
        
        # 只保留图像下部的点
        lower_mask = pixel_coords[:, 1] > h_threshold
        lower_points = points_3d[lower_mask]
        
        print(f"\n下部区域分析:")
        print(f"  下部区域点数: {len(lower_points)} / {len(points_3d)} ({100*len(lower_points)/len(points_3d):.1f}%)")
        
        if len(lower_points) < self.config["MIN_TABLE_POINTS"]:
            print("  下部区域点数不足")
            return None, None
        
        # 进一步过滤：只保留近距离的点
        z_values = lower_points[:, 2]
        near_mask = z_values < self.config["TABLE_DEPTH_MM"] / 1000.0
        near_lower_points = lower_points[near_mask]
        
        print(f"  近距离下部点数: {len(near_lower_points)}")
        
        if len(near_lower_points) < self.config["MIN_TABLE_POINTS"]:
            return None, None
        
        # 使用RANSAC拟合平面
        return self.fit_plane_with_constraints(near_lower_points)
    
    def fit_plane_with_constraints(self, points_3d):
        """带约束的平面拟合，优先寻找水平面"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        best_plane = None
        best_score = -np.inf
        
        # 多次尝试，选择最水平的平面
        for i in range(5):
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.config["RANSAC_DISTANCE_THRESHOLD"],
                ransac_n=3,
                num_iterations=self.config["RANSAC_ITERATIONS"]
            )
            
            if len(inliers) < 100:
                continue
            
            # 评估平面质量
            a, b, c, d = plane_model
            norm = np.sqrt(a**2 + b**2 + c**2)
            normal = np.array([a, b, c]) / norm
            
            # 评分：Y分量越大越好（越水平），内点越多越好
            horizontality = abs(normal[1])
            inlier_ratio = len(inliers) / len(points_3d)
            
            # 额外惩罚Z分量过大的平面（避免选到垂直面）
            z_penalty = max(0, abs(normal[2]) - 0.3) * 2
            
            score = horizontality * inlier_ratio * len(inliers) / 1000 - z_penalty
            
            if self.debug_mode:
                print(f"\n  尝试 {i+1}:")
                print(f"    法向量: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
                print(f"    水平度: {horizontality:.3f}")
                print(f"    内点比: {inlier_ratio:.3f}")
                print(f"    Z惩罚: {z_penalty:.3f}")
                print(f"    得分: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_plane = plane_model
                best_inliers = inliers
        
        if best_plane is not None:
            return best_plane, points_3d[best_inliers]
        
        return None, None
    
    def validate_table_plane(self, plane_model, points_3d):
        """验证检测到的平面是否为合理的桌面"""
        a, b, c, d = plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        normal = np.array([a, b, c]) / norm
        
        # 检查1：法向量应该主要指向上方
        if abs(normal[1]) < 0.7:
            print("  验证失败：平面不够水平")
            return False
        
        # 检查2：平面高度应该合理（相机下方）
        # 对于y=0的点，平面方程：ax + by + cz + d = 0
        # 当x=0, z=0.3（30cm）时，y = -d/b - c*0.3/b
        estimated_height = -d/b - c*0.3/b if abs(b) > 0.1 else 0
        
        print(f"\n  平面验证:")
        print(f"    估计高度: {estimated_height:.3f} m")
        print(f"    法向量Y分量: {normal[1]:.3f}")
        
        # 桌面应该在相机下方（Y<0）但不能太远
        if estimated_height > 0.1 or estimated_height < -0.5:
            print(f"    验证失败：高度不合理 ({estimated_height:.3f}m)")
            return False
        
        # 检查3：计算平面上的点分布
        distances = np.abs(a * points_3d[:, 0] + b * points_3d[:, 1] + 
                          c * points_3d[:, 2] + d) / norm
        on_plane_mask = distances < 0.02
        on_plane_points = points_3d[on_plane_mask]
        
        if len(on_plane_points) < self.config["MIN_TABLE_POINTS"]:
            print(f"    验证失败：平面点数不足 ({len(on_plane_points)})")
            return False
        
        # 计算平面范围
        x_range = np.max(on_plane_points[:, 0]) - np.min(on_plane_points[:, 0])
        z_range = np.max(on_plane_points[:, 2]) - np.min(on_plane_points[:, 2])
        
        print(f"    平面尺寸: {x_range:.2f} x {z_range:.2f} m")
        
        # 桌面尺寸应该合理（不能太小或太大）
        if x_range < 0.2 or z_range < 0.1 or x_range > 3.0 or z_range > 2.0:
            print(f"    验证失败：尺寸不合理")
            return False
        
        print("    ✓ 验证通过！")
        return True
    
    def visualize_3d_results(self, points_3d, pixel_coords, table_plane=None):
        """可视化3D结果，特别标注桌面区域"""
        fig = plt.figure(figsize=(18, 6))
        
        # 采样点云
        if len(points_3d) > 10000:
            sample_idx = np.random.choice(len(points_3d), 10000, replace=False)
            sampled_points = points_3d[sample_idx]
            sampled_pixels = pixel_coords[sample_idx]
        else:
            sampled_points = points_3d
            sampled_pixels = pixel_coords
        
        # 根据像素位置着色（下部区域用不同颜色）
        h_threshold = 240 * self.config["IMAGE_LOWER_RATIO"]
        colors = np.where(sampled_pixels[:, 1] > h_threshold, 'red', 'blue')
        
        # 子图1: 3D点云
        ax1 = fig.add_subplot(131, projection='3d')
        scatter = ax1.scatter(sampled_points[:, 0], 
                            sampled_points[:, 1], 
                            sampled_points[:, 2], 
                            c=colors, s=1, alpha=0.6)
        
        # 如果找到桌面，绘制平面
        if table_plane is not None:
            a, b, c, d = table_plane
            xlim = ax1.get_xlim()
            zlim = ax1.get_zlim()
            xx, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                               np.linspace(zlim[0], zlim[1], 10))
            yy = (-a * xx - c * zz - d) / b
            ax1.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
        
        ax1.set_xlabel('X (左右) [m]')
        ax1.set_ylabel('Y (上下) [m]')
        ax1.set_zlabel('Z (前后) [m]')
        ax1.set_title('3D点云（红=下部，蓝=上部）')
        
        # 子图2: 俯视图
        ax2 = fig.add_subplot(132)
        scatter2 = ax2.scatter(sampled_points[:, 0], 
                             sampled_points[:, 2], 
                             c=sampled_points[:, 1], 
                             cmap='viridis', s=1, alpha=0.6)
        ax2.set_xlabel('X (左右) [m]')
        ax2.set_ylabel('Z (前后/深度) [m]')
        ax2.set_title('俯视图')
        ax2.axis('equal')
        plt.colorbar(scatter2, ax=ax2, label='高度 Y [m]')
        
        # 子图3: 侧视图
        ax3 = fig.add_subplot(133)
        scatter3 = ax3.scatter(sampled_points[:, 2], 
                             sampled_points[:, 1], 
                             c=colors, s=1, alpha=0.6)
        
        # 标记预期的桌面高度范围
        ax3.axhspan(-0.3, -0.05, alpha=0.2, color='green', label='预期桌面高度')
        
        ax3.set_xlabel('Z (前后/深度) [m]')
        ax3.set_ylabel('Y (上下/高度) [m]')
        ax3.set_title('侧视图（红=下部，蓝=上部）')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_plane(self, plane_model):
        """详细分析平面参数"""
        if plane_model is None:
            return None
        
        a, b, c, d = plane_model
        
        # 归一化法向量
        norm = np.sqrt(a**2 + b**2 + c**2)
        nx, ny, nz = a/norm, b/norm, c/norm
        
        # 计算与各轴的夹角
        angle_with_x = np.arccos(np.clip(abs(nx), 0, 1)) * 180 / np.pi
        angle_with_y = np.arccos(np.clip(abs(ny), 0, 1)) * 180 / np.pi
        angle_with_z = np.arccos(np.clip(abs(nz), 0, 1)) * 180 / np.pi
        
        # 计算与期望法向量(0,1,0)的偏差
        expected_normal = np.array([0, 1, 0])
        actual_normal = np.array([nx, ny, nz])
        dot_product = np.clip(np.dot(actual_normal, expected_normal), -1, 1)
        deviation_angle = np.arccos(abs(dot_product)) * 180 / np.pi
        
        # 原点到平面的距离
        distance = abs(d) / norm
        
        # 估计桌面高度（当x=0, z=0.3时的y值）
        table_height = (-d - a*0 - c*0.3) / b if abs(b) > 0.1 else np.inf
        
        print(f"\n{'='*60}")
        print(f"平面详细分析:")
        print(f"{'='*60}")
        print(f"平面方程: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        print(f"\n归一化法向量: ({nx:.4f}, {ny:.4f}, {nz:.4f})")
        print(f"\n与坐标轴的夹角:")
        print(f"  与X轴: {angle_with_x:.1f}°")
        print(f"  与Y轴: {angle_with_y:.1f}° {'✓ (接近平行)' if angle_with_y < 20 else ''}")
        print(f"  与Z轴: {angle_with_z:.1f}°")
        print(f"\n与期望法向量(0,1,0)的偏差: {deviation_angle:.1f}°")
        print(f"\n平面位置:")
        print(f"  原点到平面距离: {distance:.3f}m")
        print(f"  估计桌面高度(z=0.3m处): {table_height:.3f}m")
        
        print(f"\n判定结果:")
        if deviation_angle < 20 and -0.5 < table_height < -0.05:
            print(f"  ✓ 这是一个合理的水平桌面")
        elif deviation_angle < 30:
            print(f"  ~ 这可能是轻微倾斜的桌面")
        else:
            print(f"  ✗ 这不太可能是桌面")
        
        print(f"{'='*60}")
        
        return {
            'plane': plane_model,
            'normal': [nx, ny, nz],
            'deviation_angle': deviation_angle,
            'table_height': table_height,
            'is_valid_table': deviation_angle < 30 and -0.5 < table_height < -0.05
        }
    
    def process_single_file(self, npz_path):
        """处理单个NPZ文件的完整流程"""
        print(f"\n{'#'*80}")
        print(f"桌面检测 - 优化版")
        print(f"文件: {npz_path}")
        print(f"{'#'*80}")
        
        # 1. 加载深度数据
        depth_mm = self.load_npz_depth_data(npz_path)
        if depth_mm is None:
            return None
        
        # 2. 分析深度分布
        self.analyze_depth_distribution(depth_mm)
        
        # 3. 可视化深度区域
        print("\n可视化深度数据...")
        self.visualize_depth_regions(depth_mm)
        
        # 4. 转换为3D点云（保留像素信息）
        points_3d, pixel_coords = self.depth_to_point_cloud_with_pixel_info(depth_mm)
        
        # 5. 在下部区域寻找桌面
        print("\n在图像下部区域寻找桌面...")
        table_plane, table_points = self.find_table_in_lower_region(points_3d, pixel_coords)
        
        # 6. 验证桌面
        if table_plane is not None:
            if self.validate_table_plane(table_plane, points_3d):
                # 7. 详细分析
                result = self.analyze_plane(table_plane)
                
                # 8. 可视化结果
                print("\n可视化3D结果...")
                self.visualize_3d_results(points_3d, pixel_coords, table_plane)
                
                return result
            else:
                print("\n桌面验证失败，尝试其他方法...")
        
        # 备选方案：全局搜索
        print("\n尝试全局搜索水平面...")
        return self.global_horizontal_plane_search(points_3d)
    
    def global_horizontal_plane_search(self, points_3d):
        """全局搜索水平面作为备选方案"""
        # 按Y值分层
        y_values = points_3d[:, 1]
        y_bins = np.linspace(np.min(y_values), np.max(y_values), 20)
        
        best_plane = None
        best_score = -np.inf
        
        for i in range(len(y_bins)-1):
            # 提取该层的点
            layer_mask = (y_values >= y_bins[i]) & (y_values < y_bins[i+1])
            layer_points = points_3d[layer_mask]
            
            if len(layer_points) < self.config["MIN_TABLE_POINTS"]:
                continue
            
            # 拟合平面
            plane, inlier_points = self.fit_plane_with_constraints(layer_points)
            
            if plane is not None:
                # 评估
                a, b, c, d = plane
                norm = np.sqrt(a**2 + b**2 + c**2)
                normal = np.array([a, b, c]) / norm
                
                # 只考虑Y值为负（相机下方）的平面
                layer_height = np.mean(y_values[layer_mask])
                if layer_height > 0:
                    continue
                
                score = abs(normal[1]) * len(inlier_points) / 1000
                
                if score > best_score:
                    best_score = score
                    best_plane = plane
        
        if best_plane is not None:
            return self.analyze_plane(best_plane)
        
        print("\n未能找到合适的水平桌面")
        return None
    
    def batch_process(self):
        """批量处理多个文件"""
        depth_dir = Path(self.config["NPZ_DIRECTORY"])
        
        print(f"\n{'#'*80}")
        print(f"批量处理模式")
        print(f"目录: {depth_dir}")
        print(f"{'#'*80}")
        
        results = []
        
        for filename in self.config["BATCH_FILES"]:
            npz_path = depth_dir / filename
            if npz_path.exists():
                try:
                    result = self.process_single_file(str(npz_path))
                    if result and result['is_valid_table']:
                        results.append({
                            'filename': filename,
                            'plane': result['plane'],
                            'normal': result['normal'],
                            'deviation_angle': result['deviation_angle'],
                            'table_height': result['table_height']
                        })
                except Exception as e:
                    print(f"\n处理文件 {filename} 时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\n文件不存在: {npz_path}")
        
        # 总结结果
        if results:
            print(f"\n{'='*80}")
            print(f"批量处理总结:")
            print(f"{'='*80}")
            print(f"成功检测到 {len(results)} 个桌面")
            
            # 找出最佳结果
            best = min(results, key=lambda x: x['deviation_angle'])
            print(f"\n最佳结果:")
            print(f"  文件: {best['filename']}")
            print(f"  法向量: ({best['normal'][0]:.3f}, {best['normal'][1]:.3f}, {best['normal'][2]:.3f})")
            print(f"  偏差角度: {best['deviation_angle']:.1f}°")
            print(f"  桌面高度: {best['table_height']:.3f}m")
            
            # 计算平均平面
            avg_normal = np.mean([r['normal'] for r in results], axis=0)
            avg_normal = avg_normal / np.linalg.norm(avg_normal)
            print(f"\n平均法向量: ({avg_normal[0]:.3f}, {avg_normal[1]:.3f}, {avg_normal[2]:.3f})")
            
            return best['plane']
        else:
            print("\n批量处理未找到合适的桌面")
            return None

def main():
    """主函数"""
    # 设置matplotlib支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    
    detector = DesktopTableDetector(CONFIG)
    
    # 处理单个文件
    test_file = Path(CONFIG["NPZ_DIRECTORY"]) / CONFIG["TEST_FILE"]
    
    if test_file.exists():
        print("处理单个测试文件...")
        result = detector.process_single_file(str(test_file))
        
        if not result or not result['is_valid_table']:
            print("\n单个文件未找到合理的桌面，尝试批量处理...")
            detector.batch_process()
    else:
        print(f"测试文件 {test_file} 不存在，直接进行批量处理...")
        detector.batch_process()

if __name__ == "__main__":
    main()