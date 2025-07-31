"""
完整的视频序列桌面检测器
包含所有必要的类和配置定义
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import pandas as pd

# ============= 完整配置 =============
CONFIG = {
    # NPZ文件设置
    "NPZ_DIRECTORY": "C:/Users/14101/Desktop/table/pre_depth",
    "TEST_FILE": "0.npz",
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
    
    # 批处理参数
    "BATCH_DIRECTORY": "C:/Users/14101/Desktop/table/pre_depth",
    "MIN_CONSENSUS_FRAMES": 10,  # 最少需要多少帧达成共识
    "CONSENSUS_THRESHOLD": 0.7,  # 70%的帧同意才认为是桌面
    "OUTLIER_THRESHOLD": 0.1,   # 偏差超过10%认为是异常帧
    "SAVE_RESULTS": True,        # 是否保存结果
    "RESULTS_DIR": "C:/Users/14101/Desktop/table/results",
}

class EnhancedTableDetector:
    """基础的增强版桌面检测器"""
    
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

class VideoSequenceTableDetector(EnhancedTableDetector):
    """视频序列桌面检测器"""
    
    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.frame_results = []  # 存储每帧的结果
        self.consensus_plane = None  # 共识平面
        
    def process_video_sequence(self, directory_path=None):
        """处理整个视频序列"""
        if directory_path is None:
            directory_path = self.config["BATCH_DIRECTORY"]
        
        # 获取所有npz文件
        npz_files = sorted(Path(directory_path).glob("*.npz"))
        print(f"\n找到 {len(npz_files)} 个NPZ文件")
        
        if len(npz_files) == 0:
            print("未找到NPZ文件！")
            return None
        
        # 1. 处理每一帧
        print("\n" + "="*80)
        print("第一阶段：逐帧处理")
        print("="*80)
        
        for idx, npz_file in enumerate(npz_files):
            print(f"\n处理帧 {idx+1}/{len(npz_files)}: {npz_file.name}")
            
            try:
                result = self.process_single_frame(str(npz_file), frame_idx=idx)
                if result is not None:
                    self.frame_results.append({
                        'frame_idx': idx,
                        'filename': npz_file.name,
                        'result': result,
                        'timestamp': datetime.now()
                    })
            except Exception as e:
                print(f"  处理失败: {e}")
                continue
        
        # 2. 分析所有帧的结果
        print("\n" + "="*80)
        print("第二阶段：综合分析")
        print("="*80)
        
        if len(self.frame_results) < self.config["MIN_CONSENSUS_FRAMES"]:
            print(f"有效帧数不足 ({len(self.frame_results)} < {self.config['MIN_CONSENSUS_FRAMES']})")
            return None
        
        # 3. 计算共识平面
        self.consensus_plane = self.compute_consensus_plane()
        
        # 4. 识别和过滤异常帧
        self.identify_outlier_frames()
        
        # 5. 生成综合报告
        report = self.generate_comprehensive_report()
        
        # 6. 可视化结果
        self.visualize_sequence_results()
        
        # 7. 保存结果
        if self.config["SAVE_RESULTS"]:
            self.save_results(report)
        
        return report
    
    def process_single_frame(self, npz_path, frame_idx):
        """处理单帧，返回简化的结果"""
        # 加载深度数据
        depth_mm = self.load_npz_depth_data(npz_path)
        if depth_mm is None:
            return None
        
        # 检测桌面区域
        table_mask = self.detect_table_region(depth_mm)
        table_pixel_count = np.sum(table_mask)
        
        if table_pixel_count < self.config['MIN_TABLE_POINTS']:
            print(f"  桌面点数不足: {table_pixel_count}")
            return None
        
        # 生成点云
        table_points, _, _ = self.depth_to_point_cloud_with_mask(depth_mm, table_mask)
        
        if len(table_points) < self.config['MIN_TABLE_POINTS']:
            return None
        
        # 拟合平面（简化输出）
        self.debug_mode = False  # 暂时关闭详细输出
        plane_model, inlier_points = self.fit_plane_with_constraints(table_points)
        self.debug_mode = True
        
        if plane_model is None:
            return None
        
        # 分析平面
        a, b, c, d = plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        nx, ny, nz = a/norm, b/norm, c/norm
        
        # 确保法向量向上
        if ny < 0:
            nx, ny, nz, d = -nx, -ny, -nz, -d
            plane_model = [-a, -b, -c, -d]
        
        deviation_angle = np.arccos(np.clip(abs(ny), 0, 1)) * 180 / np.pi
        
        # 返回结果
        return {
            'plane_model': plane_model,
            'normal': [nx, ny, nz],
            'deviation_angle': deviation_angle,
            'table_points': table_points,
            'table_mask': table_mask,
            'depth_mm': depth_mm,
            'inlier_points': inlier_points,
            'table_pixel_count': table_pixel_count
        }
    
    def compute_consensus_plane(self):
        """计算多帧的共识平面"""
        print("\n计算共识平面...")
        
        # 收集所有有效的平面参数
        all_normals = []
        all_planes = []
        all_deviations = []
        
        for frame_data in self.frame_results:
            result = frame_data['result']
            all_normals.append(result['normal'])
            all_planes.append(result['plane_model'])
            all_deviations.append(result['deviation_angle'])
        
        all_normals = np.array(all_normals)
        all_planes = np.array(all_planes)
        all_deviations = np.array(all_deviations)
        
        # 方法1：使用中位数（对异常值鲁棒）
        median_normal = np.median(all_normals, axis=0)
        median_normal = median_normal / np.linalg.norm(median_normal)
        
        # 方法2：去除异常值后的平均
        # 使用MAD（中位绝对偏差）识别异常值
        mad = np.median(np.abs(all_deviations - np.median(all_deviations)))
        threshold = np.median(all_deviations) + 3 * mad
        
        inlier_mask = all_deviations < threshold
        inlier_normals = all_normals[inlier_mask]
        
        if len(inlier_normals) > 0:
            mean_normal = np.mean(inlier_normals, axis=0)
            mean_normal = mean_normal / np.linalg.norm(mean_normal)
        else:
            mean_normal = median_normal
        
        # 计算共识平面的偏差
        consensus_deviation = np.arccos(np.clip(abs(mean_normal[1]), 0, 1)) * 180 / np.pi
        
        print(f"  有效帧数: {len(self.frame_results)}")
        print(f"  中位数法向量: ({median_normal[0]:.4f}, {median_normal[1]:.4f}, {median_normal[2]:.4f})")
        print(f"  去异常值平均法向量: ({mean_normal[0]:.4f}, {mean_normal[1]:.4f}, {mean_normal[2]:.4f})")
        print(f"  共识偏差角: {consensus_deviation:.2f}°")
        print(f"  偏差角标准差: {np.std(all_deviations):.2f}°")
        
        return {
            'median_normal': median_normal.tolist(),
            'mean_normal': mean_normal.tolist(),
            'consensus_deviation': consensus_deviation,
            'all_deviations': all_deviations.tolist(),
            'deviation_std': float(np.std(all_deviations)),
            'inlier_count': int(np.sum(inlier_mask)),
            'total_count': len(all_deviations)
        }
    
    def identify_outlier_frames(self):
        """识别异常帧"""
        print("\n识别异常帧...")
        
        if self.consensus_plane is None:
            return
        
        consensus_normal = np.array(self.consensus_plane['mean_normal'])
        outliers = []
        
        for frame_data in self.frame_results:
            frame_normal = np.array(frame_data['result']['normal'])
            
            # 计算与共识法向量的夹角
            dot_product = np.clip(np.dot(frame_normal, consensus_normal), -1, 1)
            angle_diff = np.arccos(dot_product) * 180 / np.pi
            
            # 标记异常
            if angle_diff > self.config["OUTLIER_THRESHOLD"] * 180:
                outliers.append({
                    'frame': frame_data['filename'],
                    'angle_diff': angle_diff,
                    'deviation': frame_data['result']['deviation_angle']
                })
        
        if outliers:
            print(f"  发现 {len(outliers)} 个异常帧:")
            for outlier in outliers[:5]:  # 只显示前5个
                print(f"    {outlier['frame']}: 偏差 {outlier['angle_diff']:.1f}°")
        else:
            print("  未发现明显异常帧")
        
        self.outlier_frames = outliers
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        print("\n生成综合报告...")
        
        # 统计信息
        all_deviations = [f['result']['deviation_angle'] for f in self.frame_results]
        all_pixel_counts = [f['result']['table_pixel_count'] for f in self.frame_results]
        
        report = {
            'summary': {
                'total_frames': len(self.frame_results),
                'consensus_deviation': self.consensus_plane['consensus_deviation'],
                'mean_deviation': np.mean(all_deviations),
                'std_deviation': np.std(all_deviations),
                'min_deviation': np.min(all_deviations),
                'max_deviation': np.max(all_deviations),
                'median_deviation': np.median(all_deviations),
                'mean_table_pixels': np.mean(all_pixel_counts),
                'outlier_count': len(self.outlier_frames) if hasattr(self, 'outlier_frames') else 0
            },
            'consensus_plane': self.consensus_plane,
            'frame_results': self.frame_results,
            'outliers': self.outlier_frames if hasattr(self, 'outlier_frames') else []
        }
        
        # 打印摘要
        print(f"\n{'='*60}")
        print("检测结果摘要")
        print(f"{'='*60}")
        print(f"总帧数: {report['summary']['total_frames']}")
        print(f"共识偏差角: {report['summary']['consensus_deviation']:.2f}°")
        print(f"平均偏差角: {report['summary']['mean_deviation']:.2f}° ± {report['summary']['std_deviation']:.2f}°")
        print(f"偏差范围: [{report['summary']['min_deviation']:.2f}°, {report['summary']['max_deviation']:.2f}°]")
        print(f"异常帧数: {report['summary']['outlier_count']}")
        print(f"平均桌面像素: {report['summary']['mean_table_pixels']:.0f}")
        print(f"{'='*60}")
        
        return report
    
    def visualize_sequence_results(self):
        """可视化序列结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 偏差角度分布
        ax = axes[0, 0]
        deviations = [f['result']['deviation_angle'] for f in self.frame_results]
        ax.hist(deviations, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=self.consensus_plane['consensus_deviation'], 
                  color='red', linestyle='--', label='共识偏差')
        ax.set_xlabel('偏差角度 (°)')
        ax.set_ylabel('帧数')
        ax.set_title('偏差角度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 偏差角度时序图
        ax = axes[0, 1]
        frame_indices = [f['frame_idx'] for f in self.frame_results]
        ax.plot(frame_indices, deviations, 'b-', marker='o', markersize=4)
        ax.axhline(y=self.consensus_plane['consensus_deviation'], 
                  color='red', linestyle='--', label='共识偏差')
        ax.set_xlabel('帧索引')
        ax.set_ylabel('偏差角度 (°)')
        ax.set_title('偏差角度时序变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 法向量分布（3D）
        ax = axes[0, 2]
        ax = plt.subplot(2, 3, 3, projection='3d')
        normals = np.array([f['result']['normal'] for f in self.frame_results])
        ax.scatter(normals[:, 0], normals[:, 1], normals[:, 2], 
                  c='blue', s=20, alpha=0.6, label='各帧法向量')
        
        # 添加共识法向量
        consensus_n = self.consensus_plane['mean_normal']
        ax.scatter([consensus_n[0]], [consensus_n[1]], [consensus_n[2]], 
                  c='red', s=100, marker='*', label='共识法向量')
        
        # 添加理想法向量
        ax.scatter([0], [0.9999], [0.0131], c='green', s=100, marker='^', label='理想法向量')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('法向量分布')
        ax.legend()
        
        # 4. 桌面像素数变化
        ax = axes[1, 0]
        pixel_counts = [f['result']['table_pixel_count'] for f in self.frame_results]
        ax.plot(frame_indices, pixel_counts, 'g-', marker='s', markersize=4)
        ax.set_xlabel('帧索引')
        ax.set_ylabel('桌面像素数')
        ax.set_title('检测到的桌面大小变化')
        ax.grid(True, alpha=0.3)
        
        # 5. 最佳和最差帧对比
        ax = axes[1, 1]
        ax.axis('off')
        
        # 找到最佳和最差帧
        best_idx = np.argmin(deviations)
        worst_idx = np.argmax(deviations)
        
        info_text = "帧质量分析\n" + "="*30 + "\n"
        info_text += f"最佳帧: {self.frame_results[best_idx]['filename']}\n"
        info_text += f"  偏差: {deviations[best_idx]:.2f}°\n"
        info_text += f"  像素: {pixel_counts[best_idx]}\n\n"
        info_text += f"最差帧: {self.frame_results[worst_idx]['filename']}\n"
        info_text += f"  偏差: {deviations[worst_idx]:.2f}°\n"
        info_text += f"  像素: {pixel_counts[worst_idx]}\n\n"
        info_text += f"改进: {deviations[worst_idx] - deviations[best_idx]:.2f}°"
        
        ax.text(0.1, 0.5, info_text, fontsize=12, va='center', 
               transform=ax.transAxes, family='monospace')
        
        # 6. 稳定性评估
        ax = axes[1, 2]
        
        # 计算滑动窗口标准差
        window_size = min(10, len(deviations) // 3)
        if window_size > 2:
            rolling_std = pd.Series(deviations).rolling(window=window_size).std()
            ax.plot(frame_indices, rolling_std, 'r-', linewidth=2)
            ax.set_xlabel('帧索引')
            ax.set_ylabel('局部标准差 (°)')
            ax.set_title(f'稳定性分析 (窗口={window_size})')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, '帧数不足\n无法分析稳定性', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, report):
        """保存结果到文件"""
        results_dir = Path(self.config["RESULTS_DIR"])
        results_dir.mkdir(exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON报告
        json_file = results_dir / f"table_detection_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            # 将numpy数组转换为列表
            json_report = self._convert_to_json_serializable(report)
            json.dump(json_report, f, indent=4)
        
        print(f"\n结果已保存到: {json_file}")
        
        # 保存CSV摘要
        csv_file = results_dir / f"table_detection_summary_{timestamp}.csv"
        summary_df = pd.DataFrame([report['summary']])
        summary_df.to_csv(csv_file, index=False)
        
        print(f"摘要已保存到: {csv_file}")
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def compute_refined_table_points(self, consensus_threshold=0.7):
        """基于多帧共识计算精炼的桌面点云"""
        print("\n计算精炼的桌面点云...")
        
        # 收集所有帧的桌面mask
        h, w = self.frame_results[0]['result']['depth_mm'].shape
        consensus_mask = np.zeros((h, w), dtype=float)
        
        # 累积每个像素被识别为桌面的次数
        for frame_data in self.frame_results:
            table_mask = frame_data['result']['table_mask']
            consensus_mask += table_mask.astype(float)
        
        # 归一化到[0, 1]
        consensus_mask /= len(self.frame_results)
        
        # 只保留超过阈值的像素
        refined_mask = consensus_mask >= consensus_threshold
        
        print(f"  共识阈值: {consensus_threshold}")
        print(f"  精炼后桌面像素: {np.sum(refined_mask)} "
              f"({100*np.sum(refined_mask)/refined_mask.size:.1f}%)")
        
        # 可视化共识mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(consensus_mask, cmap='hot')
        axes[0].set_title('像素共识度')
        axes[0].set_xlabel('像素')
        axes[0].set_ylabel('像素')
        
        axes[1].imshow(refined_mask, cmap='Greens')
        axes[1].set_title(f'精炼桌面区域 (阈值={consensus_threshold})')
        
        # 显示一个示例帧的深度图作为参考
        example_depth = self.frame_results[0]['result']['depth_mm']
        axes[2].imshow(example_depth, cmap='viridis')
        axes[2].set_title('参考深度图')
        
        plt.tight_layout()
        plt.show()
        
        return refined_mask, consensus_mask

def main():
    """主函数 - 批处理视频序列"""
    # 设置matplotlib
    import matplotlib
    matplotlib.use('TkAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建视频序列检测器
    detector = VideoSequenceTableDetector(CONFIG)
    
    print("="*80)
    print("视频序列桌面检测系统")
    print("="*80)
    
    # 处理整个视频序列
    report = detector.process_video_sequence()
    
    if report:
        # 计算精炼的桌面区域
        refined_mask, consensus_mask = detector.compute_refined_table_points(
            consensus_threshold=0.7
        )
        
        # 与基础方法对比
        print("\n" + "="*60)
        print("与基础方法对比")
        print("="*60)
        print("基础方法: 单帧检测，偏差10.8°")
        print(f"视频序列方法: {report['summary']['total_frames']}帧综合，"
              f"共识偏差{report['summary']['consensus_deviation']:.2f}°")
        
        if report['summary']['consensus_deviation'] < 10.8:
            improvement = 10.8 - report['summary']['consensus_deviation']
            print(f"\n✨ 改进效果显著！")
            print(f"✨ 偏差减少: {improvement:.2f}°")
            print(f"✨ 精度提升: {improvement/10.8*100:.1f}%")
        
        print(f"\n算法鲁棒性评估:")
        print(f"  偏差标准差: {report['summary']['std_deviation']:.2f}°")
        print(f"  异常帧比例: {report['summary']['outlier_count']/report['summary']['total_frames']*100:.1f}%")
        
        # 询问是否查看详细帧结果
        if input("\n是否查看各帧详细结果？(y/n): ").lower() == 'y':
            for i, frame_data in enumerate(report['frame_results'][:10]):  # 只显示前10帧
                print(f"\n帧 {i+1}: {frame_data['filename']}")
                print(f"  偏差: {frame_data['result']['deviation_angle']:.2f}°")
                print(f"  桌面像素: {frame_data['result']['table_pixel_count']}")

if __name__ == "__main__":
    main()