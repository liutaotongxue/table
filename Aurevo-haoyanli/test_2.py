"""
修正版 - 增强版桌面检测器
修复了维度不匹配的问题
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D

# ============= 配置区域 =============
CONFIG = {
    # NPZ文件设置
    "NPZ_DIRECTORY": "C:/Users/14101/Desktop/table/pre_depth",
    "TEST_FILE": "190.npz",
    "BATCH_FILES": ["10.npz", "20.npz", "30.npz", "40.npz", "50.npz", 
                   "60.npz", "70.npz", "80.npz", "90.npz", "100.npz"],
    
    # 深度过滤范围（毫米）
    "MIN_DEPTH_MM": 100,
    "MAX_DEPTH_MM": 700,
    "TABLE_DEPTH_MM": 400,
    
    # 相机内参
    "CAMERA_FX": 214.15,
    "CAMERA_FY": 212.80,
    "CAMERA_CX": 162.52,
    "CAMERA_CY": 127.85,
    
    # RANSAC参数
    "RANSAC_DISTANCE_THRESHOLD": 0.005,
    "RANSAC_ITERATIONS": 2000,
    
    # 桌面检测参数
    "HEIGHT_TOLERANCE": 0.03,
    "MIN_TABLE_POINTS": 500,
    "HORIZONTAL_THRESHOLD": 0.8,
    "IMAGE_LOWER_RATIO": 0.6,
    
    # 增强检测参数
    "GRADIENT_THRESHOLD": 30,
    "MIN_OBJECT_SIZE": 1000,
    "DEPTH_CONSISTENCY": 20,
    "DOWNSAMPLE_FACTOR": 4,
}

class EnhancedTableDetector:
    """增强版桌面检测器 - 包含干扰物体去除"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.debug_mode = True
    
    def load_npz_depth_data(self, npz_path):
        """加载NPZ文件中的深度数据"""
        try:
            data = np.load(npz_path)
            print(f"\n加载文件: {npz_path}")
            print(f"NPZ文件键值: {list(data.keys())}")
            
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
    
    def depth_to_point_cloud_with_mask(self, depth_mm, mask=None):
        """将深度图转换为3D点云，可选择性地应用mask"""
        h, w = depth_mm.shape
        
        # 创建像素坐标网格
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # 有效深度掩码
        valid_mask = (depth_mm > self.config["MIN_DEPTH_MM"]) & \
                    (depth_mm < self.config["MAX_DEPTH_MM"])
        
        # 如果提供了额外的mask，与有效深度mask结合
        if mask is not None:
            valid_mask = valid_mask & mask
        
        # 转换为相机坐标系
        z = depth_mm[valid_mask] / 1000.0  # 转换为米
        x = (xx[valid_mask] - self.config["CAMERA_CX"]) * z / self.config["CAMERA_FX"]
        y = -(yy[valid_mask] - self.config["CAMERA_CY"]) * z / self.config["CAMERA_FY"]
        
        points_3d = np.column_stack([x, y, z])
        pixel_coords = np.column_stack([xx[valid_mask], yy[valid_mask]])
        
        return points_3d, pixel_coords, valid_mask
    
    def detect_table_region(self, depth_mm):
        """检测桌面区域"""
        h, w = depth_mm.shape
        
        # 1. 计算局部深度变化
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # 局部平均深度
        local_mean = cv2.filter2D(depth_mm.astype(np.float32), -1, kernel)
        
        # 局部深度差异
        local_diff = np.abs(depth_mm - local_mean)
        
        # 平坦区域：局部差异小
        flat_mask = local_diff < self.config['DEPTH_CONSISTENCY']
        
        # 2. 基于位置的过滤
        y_coords = np.arange(h)[:, None]
        position_weight = y_coords / h  # 0到1，下部接近1
        
        # 3. 基于深度范围的过滤
        depth_range_mask = (depth_mm > self.config['MIN_DEPTH_MM']) & \
                          (depth_mm < self.config['TABLE_DEPTH_MM'])
        
        # 4. 综合判断
        table_score = flat_mask.astype(float) * position_weight * depth_range_mask
        
        # 使用阈值分割
        table_candidate_mask = table_score > 0.3
        
        # 5. 形态学操作清理
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        table_candidate_mask = cv2.morphologyEx(
            table_candidate_mask.astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel_morph
        )
        table_candidate_mask = cv2.morphologyEx(
            table_candidate_mask, 
            cv2.MORPH_CLOSE, 
            kernel_morph
        )
        
        # 6. 找最大连通区域
        labeled, num_features = ndimage.label(table_candidate_mask)
        
        if num_features > 0:
            # 计算每个区域的大小
            region_sizes = []
            for i in range(1, num_features + 1):
                region_size = np.sum(labeled == i)
                region_sizes.append((i, region_size))
            
            # 选择最大的区域作为桌面
            region_sizes.sort(key=lambda x: x[1], reverse=True)
            largest_region_label = region_sizes[0][0]
            
            final_table_mask = (labeled == largest_region_label)
        else:
            final_table_mask = table_candidate_mask
        
        return final_table_mask.astype(bool)
    
    def fit_plane_with_constraints(self, points_3d):
        """带约束的平面拟合"""
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
            
            # 评分：Y分量越大越好（越水平）
            horizontality = abs(normal[1])
            inlier_ratio = len(inliers) / len(points_3d)
            
            # 额外惩罚Z分量过大的平面
            z_penalty = max(0, abs(normal[2]) - 0.3) * 2
            
            score = horizontality * inlier_ratio * len(inliers) / 1000 - z_penalty
            
            if self.debug_mode:
                print(f"\n  尝试 {i+1}:")
                print(f"    法向量: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")
                print(f"    水平度: {horizontality:.3f}")
                print(f"    内点比: {inlier_ratio:.3f}")
                print(f"    得分: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_plane = plane_model
                best_inliers = inliers
        
        if best_plane is not None:
            return best_plane, points_3d[best_inliers]
        
        return None, None
    
    def analyze_plane(self, plane_model):
        """分析平面参数"""
        if plane_model is None:
            return None
        
        a, b, c, d = plane_model
        
        # 归一化法向量
        norm = np.sqrt(a**2 + b**2 + c**2)
        nx, ny, nz = a/norm, b/norm, c/norm
        
        # 计算与期望法向量(0,1,0)的偏差
        expected_normal = np.array([0, 0.9999, 0.0131])
        actual_normal = np.array([nx, ny, nz])
        dot_product = np.clip(np.dot(actual_normal, expected_normal), -1, 1)
        deviation_angle = np.arccos(abs(dot_product)) * 180 / np.pi
        
        # 估计桌面高度
        table_height = (-d - a*0 - c*0.3) / b if abs(b) > 0.1 else np.inf
        
        print(f"\n{'='*60}")
        print(f"平面分析结果:")
        print(f"{'='*60}")
        print(f"平面方程: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        print(f"归一化法向量: ({nx:.4f}, {ny:.4f}, {nz:.4f})")
        print(f"与期望法向量(0,0.9999,0.0131)的偏差: {deviation_angle:.1f}°")
        print(f"估计桌面高度: {table_height:.3f}m")
        
        print(f"\n判定:")
        if deviation_angle < 20 and -0.5 < table_height < -0.05:
            print(f"  ✓ 这是一个合理的水平桌面")
        else:
            print(f"  ✗ 不太可能是桌面")
        
        print(f"{'='*60}")
        
        return {
            'plane': plane_model,
            'normal': [nx, ny, nz],
            'deviation_angle': deviation_angle,
            'table_height': table_height,
            'is_valid_table': deviation_angle < 30 and -0.5 < table_height < -0.05
        }
    
    def visualize_results(self, depth_mm, table_mask, all_points, table_points, plane_model=None):
        """可视化结果"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. 原始深度图
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(depth_mm, cmap='viridis')
        ax1.set_title('原始深度图')
        plt.colorbar(im1, ax=ax1)
        
        # 2. 桌面检测结果
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(table_mask, cmap='Greens')
        ax2.set_title('检测到的桌面区域')
        
        # 3. 过滤后的深度图
        ax3 = plt.subplot(2, 3, 3)
        filtered_depth = depth_mm.copy()
        filtered_depth[~table_mask] = 0
        im3 = ax3.imshow(filtered_depth, cmap='viridis')
        ax3.set_title('桌面区域深度')
        plt.colorbar(im3, ax=ax3)
        
        # 4. 3D点云
        ax4 = plt.subplot(2, 3, 4, projection='3d')
        
        # 采样显示
        def sample_points(points, max_size=5000):
            if len(points) > max_size:
                idx = np.random.choice(len(points), max_size, replace=False)
                return points[idx]
            return points
        
        # 显示所有点和桌面点
        all_sampled = sample_points(all_points, 3000)
        table_sampled = sample_points(table_points, 2000)
        
        ax4.scatter(all_sampled[:, 0], all_sampled[:, 1], all_sampled[:, 2], 
                   c='gray', s=1, alpha=0.3, label='所有点')
        ax4.scatter(table_sampled[:, 0], table_sampled[:, 1], table_sampled[:, 2], 
                   c='red', s=2, alpha=0.8, label='桌面点')
        
        # 绘制平面
        if plane_model is not None:
            a, b, c, d = plane_model
            xlim = ax4.get_xlim()
            zlim = ax4.get_zlim()
            xx, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                               np.linspace(zlim[0], zlim[1], 10))
            yy = (-a * xx - c * zz - d) / b
            ax4.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
        
        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_zlabel('Z [m]')
        ax4.set_title('3D点云（红色=桌面）')
        ax4.legend()
        
        # 5. 侧视图
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(all_sampled[:, 2], all_sampled[:, 1], 
                   c='gray', s=1, alpha=0.3, label='所有点')
        ax5.scatter(table_sampled[:, 2], table_sampled[:, 1], 
                   c='red', s=2, alpha=0.8, label='桌面点')
        ax5.set_xlabel('Z (深度) [m]')
        ax5.set_ylabel('Y (高度) [m]')
        ax5.set_title('侧视图')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. 俯视图
        ax6 = plt.subplot(2, 3, 6)
        ax6.scatter(all_sampled[:, 0], all_sampled[:, 2], 
                   c='gray', s=1, alpha=0.3, label='所有点')
        ax6.scatter(table_sampled[:, 0], table_sampled[:, 2], 
                   c='red', s=2, alpha=0.8, label='桌面点')
        ax6.set_xlabel('X [m]')
        ax6.set_ylabel('Z (深度) [m]')
        ax6.set_title('俯视图')
        ax6.axis('equal')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
    
    def process_file(self, npz_path):
        """处理单个文件的完整流程"""
        print(f"\n{'#'*80}")
        print(f"增强版桌面检测 - 干扰物体去除")
        print(f"文件: {npz_path}")
        print(f"{'#'*80}")
        
        # 1. 加载深度数据
        depth_mm = self.load_npz_depth_data(npz_path)
        if depth_mm is None:
            return None
        
        # 2. 检测桌面区域
        print("\n检测桌面区域...")
        table_mask = self.detect_table_region(depth_mm)
        print(f"桌面区域像素: {np.sum(table_mask)} ({100*np.sum(table_mask)/table_mask.size:.1f}%)")
        
        # 3. 生成所有点的点云（用于对比）
        print("\n生成3D点云...")
        all_points, _, _ = self.depth_to_point_cloud_with_mask(depth_mm)
        print(f"所有点数: {len(all_points)}")
        
        # 4. 生成桌面点的点云
        table_points, _, _ = self.depth_to_point_cloud_with_mask(depth_mm, table_mask)
        print(f"桌面点数: {len(table_points)}")
        
        if len(table_points) < self.config['MIN_TABLE_POINTS']:
            print("桌面点数不足")
            return None
        
        # 5. 拟合平面
        print("\n拟合桌面平面...")
        plane_model, inlier_points = self.fit_plane_with_constraints(table_points)
        
        if plane_model is None:
            print("平面拟合失败")
            return None
        
        # 6. 分析结果
        result = self.analyze_plane(plane_model)
        
        # 7. 可视化
        self.visualize_results(depth_mm, table_mask, all_points, table_points, plane_model)
        
        return result

def main():
    """主函数"""
    # 设置matplotlib
    import matplotlib
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建检测器
    detector = EnhancedTableDetector(CONFIG)
    
    # 处理文件
    test_file = Path(CONFIG["NPZ_DIRECTORY"]) / CONFIG["TEST_FILE"]
    
    if test_file.exists():
        print("开始处理...")
        result = detector.process_file(str(test_file))
        
        if result and result['is_valid_table']:
            print("\n" + "="*60)
            print("✅ 成功检测到桌面！")
            print("="*60)
            print(f"法向量: ({result['normal'][0]:.4f}, {result['normal'][1]:.4f}, {result['normal'][2]:.4f})")
            print(f"偏差角度: {result['deviation_angle']:.1f}°")
            print(f"桌面高度: {result['table_height']:.3f}m")
            
            print("\n与原始结果对比:")
            print("原始法向量: (-0.0115, 0.9824, 0.1867)")
            print("原始偏差: 10.8°")
            
            if result['deviation_angle'] < 10.8:
                improvement = 10.8 - result['deviation_angle']
                print(f"\n 改进效果: 偏差减少了 {improvement:.1f}°")
                print(f" 精度提升: {improvement/10.8*100:.1f}%")
            else:
                print(f"\n⚠️ 偏差没有改善，可能需要调整参数")
        else:
            print("\n 检测失败")
    else:
        print(f"文件不存在: {test_file}")

if __name__ == "__main__":
    main()
