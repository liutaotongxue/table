"""
专门针对桌面检测的精确解决方案
重点：正确识别桌面而不是其他物体
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
import time

# ============= 配置 =============
CONFIG = {
    # 文件设置
    "NPZ_DIRECTORY": "C:/Users/14101/Desktop/table/pre_depth",
    "TEST_FILE": "4.npz",
    
    # 相机内参
    "CAMERA_FX": 214.15,
    "CAMERA_FY": 212.80,
    "CAMERA_CX": 162.52,
    "CAMERA_CY": 127.85,
    
    # RANSAC参数
    "RANSAC_DISTANCE_THRESHOLD": 0.008,
    "RANSAC_ITERATIONS": 2000,
}

class AccurateTableDetector:
    """精确的桌面检测器"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
    def load_npz_depth_data(self, npz_path):
        """加载深度数据"""
        try:
            data = np.load(npz_path)
            print(f"\n加载文件: {npz_path}")
            
            depth_mm = data[list(data.keys())[0]].astype(np.float32)
            
            if depth_mm is not None:
                valid_depth = depth_mm[depth_mm > 0]
                print(f"深度图尺寸: {depth_mm.shape}")
                print(f"深度范围: {np.min(valid_depth):.1f} - {np.max(valid_depth):.1f} mm")
            
            return depth_mm
            
        except Exception as e:
            print(f"加载失败: {e}")
            return None
    
    def analyze_depth_distribution(self, depth_mm):
        """详细分析深度分布，找到不同的物体层"""
        h, w = depth_mm.shape
        
        print("\n分析深度分布...")
        
        # 1. 按行分析深度分布
        row_depths = []
        for y in range(0, h, 10):  # 每10行采样
            row = depth_mm[y, :]
            valid = row[row > 0]
            if len(valid) > 0:
                row_depths.append({
                    'y': y,
                    'mean': np.mean(valid),
                    'median': np.median(valid),
                    'std': np.std(valid),
                    'min': np.min(valid),
                    'max': np.max(valid)
                })
        
        # 2. 可视化深度分布
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始深度图
        ax = axes[0, 0]
        im = ax.imshow(depth_mm, cmap='viridis')
        ax.set_title('原始深度图')
        ax.axhline(y=h*0.6, color='red', linestyle='--', label='60%位置')
        ax.axhline(y=h*0.8, color='yellow', linestyle='--', label='80%位置')
        plt.colorbar(im, ax=ax)
        
        # 深度按行分布
        ax = axes[0, 1]
        y_positions = [d['y'] for d in row_depths]
        mean_depths = [d['mean'] for d in row_depths]
        ax.plot(mean_depths, y_positions, 'b-', label='平均深度')
        ax.invert_yaxis()
        ax.set_xlabel('深度 (mm)')
        ax.set_ylabel('图像行位置')
        ax.set_title('深度垂直分布')
        ax.grid(True)
        
        # 深度直方图（分区域）
        ax = axes[0, 2]
        # 上部区域
        upper = depth_mm[:int(h*0.4), :]
        upper_valid = upper[upper > 0]
        # 中部区域
        middle = depth_mm[int(h*0.4):int(h*0.7), :]
        middle_valid = middle[middle > 0]
        # 下部区域
        lower = depth_mm[int(h*0.7):, :]
        lower_valid = lower[lower > 0]
        
        bins = np.linspace(0, 700, 70)
        ax.hist(upper_valid, bins=bins, alpha=0.3, label='上部', color='blue')
        ax.hist(middle_valid, bins=bins, alpha=0.3, label='中部', color='green')
        ax.hist(lower_valid, bins=bins, alpha=0.3, label='下部', color='red')
        ax.set_xlabel('深度 (mm)')
        ax.set_ylabel('像素数')
        ax.set_title('分区域深度直方图')
        ax.legend()
        
        # 3. 找到可能的桌面深度
        # 桌面应该在下部区域，且深度相对集中
        if len(lower_valid) > 0:
            # 使用核密度估计找峰值
            hist, bins = np.histogram(lower_valid, bins=50)
            # 找到最高峰
            peak_idx = np.argmax(hist)
            table_depth_estimate = (bins[peak_idx] + bins[peak_idx + 1]) / 2
            
            print(f"\n下部区域深度分析:")
            print(f"  平均深度: {np.mean(lower_valid):.1f} mm")
            print(f"  中位深度: {np.median(lower_valid):.1f} mm")
            print(f"  峰值深度: {table_depth_estimate:.1f} mm")
            
            # 可视化下部区域
            ax = axes[1, 0]
            lower_display = np.where(depth_mm > 0, depth_mm, np.nan)
            lower_display[:int(h*0.7), :] = np.nan
            im = ax.imshow(lower_display, cmap='hot')
            ax.set_title('下部区域深度')
            plt.colorbar(im, ax=ax)
            
            # 深度梯度
            ax = axes[1, 1]
            grad_y = np.gradient(depth_mm, axis=0)
            grad_mag = np.abs(grad_y)
            grad_display = np.where(depth_mm > 0, grad_mag, 0)
            im = ax.imshow(grad_display, cmap='hot')
            ax.set_title('垂直深度梯度')
            plt.colorbar(im, ax=ax)
            
            # 深度层分割
            ax = axes[1, 2]
            # 基于深度值分层
            depth_layers = np.zeros_like(depth_mm)
            depth_ranges = [(0, 200), (200, 300), (300, 400), (400, 500), (500, 700)]
            for i, (d_min, d_max) in enumerate(depth_ranges):
                mask = (depth_mm >= d_min) & (depth_mm < d_max)
                depth_layers[mask] = i + 1
            
            im = ax.imshow(depth_layers, cmap='tab10')
            ax.set_title('深度分层')
            plt.colorbar(im, ax=ax, ticks=range(len(depth_ranges)+1))
            
            plt.tight_layout()
            plt.show()
            
            return table_depth_estimate
        else:
            plt.tight_layout()
            plt.show()
            return None
    
    def detect_table_by_position_and_depth(self, depth_mm, estimated_table_depth):
        """基于位置和深度检测桌面"""
        h, w = depth_mm.shape
        
        print(f"\n基于位置和深度检测桌面...")
        print(f"预估桌面深度: {estimated_table_depth:.1f} mm")
        
        # 1. 创建多个候选mask
        masks = []
        
        # 方法1: 严格的下部区域 + 深度范围
        mask1 = np.zeros_like(depth_mm, dtype=bool)
        y_start = int(h * 0.65)  # 只看下部35%
        depth_tolerance = 50  # mm
        
        for y in range(y_start, h):
            for x in range(w):
                if depth_mm[y, x] > 0:
                    if abs(depth_mm[y, x] - estimated_table_depth) < depth_tolerance:
                        mask1[y, x] = True
        
        # 方法2: 基于深度连续性（从底部开始）
        mask2 = np.zeros_like(depth_mm, dtype=bool)
        
        # 从底部开始，向上扩展
        for x in range(w):
            # 找到该列底部的有效深度
            for y in range(h-1, h//2, -1):
                if depth_mm[y, x] > 0:
                    seed_depth = depth_mm[y, x]
                    # 向上扩展相似深度
                    for y2 in range(y, -1, -1):
                        if depth_mm[y2, x] > 0:
                            if abs(depth_mm[y2, x] - seed_depth) < 30:
                                mask2[y2, x] = True
                            else:
                                break
                    break
        
        # 方法3: 水平条带检测
        mask3 = np.zeros_like(depth_mm, dtype=bool)
        
        # 在下部区域找水平条带
        for y in range(int(h*0.6), h):
            row = depth_mm[y, :]
            valid = row > 0
            if np.sum(valid) > w * 0.5:  # 至少半行有数据
                row_mean = np.mean(row[valid])
                if abs(row_mean - estimated_table_depth) < 40:
                    # 这一行可能是桌面
                    for x in range(w):
                        if row[x] > 0 and abs(row[x] - row_mean) < 20:
                            mask3[y, x] = True
        
        # 组合masks（投票）
        combined = mask1.astype(int) + mask2.astype(int) + mask3.astype(int)
        final_mask = combined >= 2  # 至少2个方法同意
        
        # 形态学清理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        # 保留最大连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
        
        if num_labels > 1:
            # 选择最大且在下部的区域
            best_label = 0
            best_score = -1
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                y_center = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2
                
                # 评分：面积大且位置靠下
                score = area * (y_center / h)
                
                if score > best_score and area > 1000:
                    best_score = score
                    best_label = i
            
            if best_label > 0:
                final_mask = (labels == best_label)
            else:
                final_mask = np.zeros_like(labels, dtype=bool)
        else:
            final_mask = np.zeros_like(final_mask, dtype=bool)
        
        # 可视化检测过程
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(mask1, cmap='Reds')
        axes[0, 0].set_title('方法1: 位置+深度范围')
        
        axes[0, 1].imshow(mask2, cmap='Greens')
        axes[0, 1].set_title('方法2: 底部扩展')
        
        axes[0, 2].imshow(mask3, cmap='Blues')
        axes[0, 2].set_title('方法3: 水平条带')
        
        axes[1, 0].imshow(combined, cmap='hot')
        axes[1, 0].set_title('投票结果')
        
        axes[1, 1].imshow(final_mask, cmap='Greens')
        axes[1, 1].set_title(f'最终桌面区域 ({np.sum(final_mask)} 像素)')
        
        # 验证结果
        if np.sum(final_mask) > 0:
            table_depths = depth_mm[final_mask]
            axes[1, 2].hist(table_depths, bins=50, color='green')
            axes[1, 2].axvline(x=np.mean(table_depths), color='red', 
                              linestyle='--', label=f'均值: {np.mean(table_depths):.0f}mm')
            axes[1, 2].set_xlabel('深度 (mm)')
            axes[1, 2].set_ylabel('像素数')
            axes[1, 2].set_title('检测区域深度分布')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
        return final_mask
    
    def depth_to_points(self, depth_mm, mask=None):
        """深度转点云"""
        h, w = depth_mm.shape
        
        fx, fy = self.config['CAMERA_FX'], self.config['CAMERA_FY']
        cx, cy = self.config['CAMERA_CX'], self.config['CAMERA_CY']
        
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        valid = depth_mm > 0
        
        if mask is not None:
            valid = valid & mask
        
        z = depth_mm[valid] / 1000.0
        x = (u[valid] - cx) * z / fx
        y = -(v[valid] - cy) * z / fy
        
        return np.column_stack([x, y, z])
    
    def fit_plane_carefully(self, points):
        """仔细的平面拟合"""
        if len(points) < 500:
            print(f"点数不足: {len(points)}")
            return None
        
        print(f"\n拟合平面 (点数: {len(points)})...")
        
        # 先进行简单的高度过滤
        y_values = points[:, 1]
        y_median = np.median(y_values)
        y_mad = np.median(np.abs(y_values - y_median))
        
        # 使用MAD（中位绝对偏差）进行稳健过滤
        inlier_mask = np.abs(y_values - y_median) < 3 * y_mad
        filtered_points = points[inlier_mask]
        
        print(f"高度过滤后: {len(points)} -> {len(filtered_points)}")
        
        if len(filtered_points) < 100:
            filtered_points = points  # 如果过滤太多，使用原始点
        
        # RANSAC平面拟合
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        best_plane = None
        best_score = -np.inf
        
        # 尝试不同的参数
        for threshold in [0.005, 0.008, 0.01, 0.015]:
            try:
                plane_model, inliers = pcd.segment_plane(
                    distance_threshold=threshold,
                    ransac_n=3,
                    num_iterations=2000
                )
                
                if len(inliers) < 100:
                    continue
                
                a, b, c, d = plane_model
                normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
                
                # 检查水平度
                horizontality = abs(normal[1])
                if horizontality < 0.8:
                    continue
                
                # 评分
                inlier_ratio = len(inliers) / len(filtered_points)
                score = horizontality * inlier_ratio * len(inliers)
                
                print(f"  阈值={threshold}: 水平度={horizontality:.3f}, "
                      f"内点={len(inliers)}, 得分={score:.0f}")
                
                if score > best_score:
                    best_score = score
                    best_plane = plane_model
                    
            except Exception as e:
                print(f"  拟合失败 (threshold={threshold}): {e}")
        
        return best_plane
    
    def process(self, npz_path):
        """主处理流程"""
        print(f"\n{'#'*80}")
        print("精确桌面检测")
        print(f"{'#'*80}")
        
        # 1. 加载数据
        depth_mm = self.load_npz_depth_data(npz_path)
        if depth_mm is None:
            return None
        
        # 2. 分析深度分布
        estimated_table_depth = self.analyze_depth_distribution(depth_mm)
        
        if estimated_table_depth is None:
            print("无法估计桌面深度")
            return None
        
        # 3. 检测桌面
        table_mask = self.detect_table_by_position_and_depth(depth_mm, estimated_table_depth)
        
        print(f"\n检测到桌面像素: {np.sum(table_mask)}")
        
        # 4. 生成点云
        all_points = self.depth_to_points(depth_mm)
        table_points = self.depth_to_points(depth_mm, table_mask)
        
        print(f"桌面点数: {len(table_points)}")
        
        # 5. 拟合平面
        plane_model = self.fit_plane_carefully(table_points)
        
        if plane_model is not None:
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            
            if normal[1] < 0:
                normal = -normal
                a, b, c, d = -a, -b, -c, -d
            
            deviation_angle = np.arccos(np.clip(abs(normal[1]), 0, 1)) * 180 / np.pi
            
            print(f"\n{'='*60}")
            print("检测结果:")
            print(f"{'='*60}")
            print(f"法向量: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
            print(f"偏差角度: {deviation_angle:.2f}°")
            print(f"{'='*60}")
            
            # 最终可视化
            self.visualize_final_result(depth_mm, table_mask, all_points, 
                                      table_points, plane_model)
            
            return {
                'normal': normal.tolist(),
                'deviation_angle': deviation_angle,
                'plane': plane_model
            }
        else:
            print("\n平面拟合失败")
            return None
    
    def visualize_final_result(self, depth_mm, table_mask, all_points, 
                              table_points, plane_model):
        """最终结果可视化"""
        fig = plt.figure(figsize=(16, 12))
        
        # 采样
        def sample(pts, n=5000):
            if len(pts) > n:
                idx = np.random.choice(len(pts), n, replace=False)
                return pts[idx]
            return pts
        
        all_sample = sample(all_points, 5000)
        table_sample = sample(table_points, 3000)
        
        # 3D视图
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        ax.scatter(all_sample[:, 0], all_sample[:, 1], all_sample[:, 2], 
                  c='gray', s=0.5, alpha=0.3)
        ax.scatter(table_sample[:, 0], table_sample[:, 1], table_sample[:, 2], 
                  c='red', s=2, alpha=0.8)
        
        # 绘制平面
        if plane_model is not None:
            a, b, c, d = plane_model
            xlim = ax.get_xlim()
            zlim = ax.get_zlim()
            xx, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                               np.linspace(zlim[0], zlim[1], 10))
            yy = (-a * xx - c * zz - d) / b
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D点云')
        ax.view_init(elev=20, azim=-60)
        
        # 原始深度图
        ax = fig.add_subplot(2, 3, 2)
        im = ax.imshow(depth_mm, cmap='viridis')
        ax.set_title('原始深度图')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 检测结果
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(table_mask, cmap='Greens')
        ax.set_title(f'桌面区域 ({np.sum(table_mask)} 像素)')
        
        # 侧视图
        ax = fig.add_subplot(2, 3, 4)
        ax.scatter(table_sample[:, 2], table_sample[:, 1], 
                  c='red', s=1, alpha=0.8, label='桌面')
        ax.scatter(all_sample[:, 2], all_sample[:, 1], 
                  c='gray', s=0.5, alpha=0.3, label='其他')
        ax.set_xlabel('Z [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('侧视图')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 俯视图
        ax = fig.add_subplot(2, 3, 5)
        ax.scatter(table_sample[:, 0], table_sample[:, 2], 
                  c='red', s=1, alpha=0.8)
        ax.scatter(all_sample[:, 0], all_sample[:, 2], 
                  c='gray', s=0.5, alpha=0.3)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title('俯视图')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # 统计信息
        ax = fig.add_subplot(2, 3, 6)
        ax.axis('off')
        
        info = "检测统计:\n" + "="*30 + "\n"
        info += f"桌面像素: {np.sum(table_mask):,}\n"
        info += f"桌面点数: {len(table_points):,}\n"
        
        if plane_model is not None:
            a, b, c, d = plane_model
            normal = np.array([a, b, c]) / np.sqrt(a**2 + b**2 + c**2)
            if normal[1] < 0:
                normal = -normal
            
            deviation = np.arccos(abs(normal[1])) * 180 / np.pi
            
            info += f"\n平面参数:\n" + "="*30 + "\n"
            info += f"法向量: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n"
            info += f"偏差角: {deviation:.1f}°\n"
            
            # 桌面尺寸
            if len(table_sample) > 0:
                x_range = table_sample[:, 0].max() - table_sample[:, 0].min()
                z_range = table_sample[:, 2].max() - table_sample[:, 2].min()
                info += f"\n桌面尺寸:\n"
                info += f"宽: {x_range:.2f} m\n"
                info += f"深: {z_range:.2f} m"
        
        ax.text(0.1, 0.9, info, fontsize=12, va='top', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    detector = AccurateTableDetector(CONFIG)
    
    test_file = Path(CONFIG["NPZ_DIRECTORY"]) / CONFIG["TEST_FILE"]
    
    if test_file.exists():
        result = detector.process(str(test_file))
        
        if result:
            print(f"\n✅ 检测成功！")
            print(f"偏差角度: {result['deviation_angle']:.2f}°")

if __name__ == "__main__":
    main()