"""
简化版视频序列桌面检测器
只保留核心功能，输出综合评估结果
"""

import numpy as np
import cv2
from pathlib import Path
import open3d as o3d

class SimpleTableDetector:
    """简化的桌面检测器"""
    
    def __init__(self):
        # 相机内参（与原版相同）
        self.fx = 214.15
        self.fy = 212.80
        self.cx = 162.52
        self.cy = 127.85
        
        # 检测参数（与原版相同）
        self.min_depth = 100  # mm
        self.max_depth = 700  # mm
        self.table_depth = 400  # mm
        self.ransac_threshold = 0.005  # 5mm
        self.ransac_iterations = 2000
        self.min_table_points = 500
        self.height_tolerance = 0.005
        self.horizontal_threshold = 0.95
        self.image_lower_ratio = 0.6
        self.depth_consistency = 20
        self.min_consensus_frames = 10
        self.outlier_threshold = 10.0  # 度
        
    def load_depth(self, npz_path):
        """加载深度数据"""
        data = np.load(npz_path)
        # 尝试常见的键
        for key in ['depth', 'depth_mm', 'arr_0']:
            if key in data:
                return data[key].astype(np.float32)
        # 使用第一个键
        return data[list(data.keys())[0]].astype(np.float32)
    
    def depth_to_points(self, depth_mm):
        """深度图转3D点云"""
        h, w = depth_mm.shape
        
        # 创建像素坐标
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        
        # 有效深度mask
        mask = (depth_mm > self.min_depth) & (depth_mm < self.max_depth)
        
        # 转换到3D
        z = depth_mm[mask] / 1000.0  # 转为米
        x = (xx[mask] - self.cx) * z / self.fx
        y = -(yy[mask] - self.cy) * z / self.fy
        
        return np.column_stack([x, y, z])
    
    def detect_table_simple(self, depth_mm):
        """简单的桌面检测"""
        h, w = depth_mm.shape
        
        # 使用原版的参数：只处理图像下60%部分
        lower_part = int(h * self.image_lower_ratio)
        depth_lower = depth_mm[lower_part:, :]
        
        # 找到最常见的深度值（可能是桌面）
        valid_depths = depth_lower[(depth_lower > self.min_depth) & 
                                  (depth_lower < self.table_depth)]
        
        if len(valid_depths) < self.min_table_points:
            return None
        
        # 使用直方图找主要深度
        hist, bins = np.histogram(valid_depths, bins=50)
        main_depth_idx = np.argmax(hist)
        main_depth = (bins[main_depth_idx] + bins[main_depth_idx + 1]) / 2
        
        # 创建桌面mask（使用原版的depth_consistency参数）
        table_mask = np.abs(depth_mm - main_depth) < self.depth_consistency
        
        # 只保留下部分
        table_mask[:lower_part, :] = False
        
        return table_mask
    
    def fit_plane(self, points):
        """使用RANSAC拟合平面"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # RANSAC平面拟合（使用原版参数）
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.ransac_threshold,
            ransac_n=3,
            num_iterations=self.ransac_iterations
        )
        
        return plane_model, inliers
    
    def analyze_plane(self, plane_model):
        """分析平面参数"""
        a, b, c, d = plane_model
        
        # 归一化法向量
        norm = np.sqrt(a**2 + b**2 + c**2)
        nx, ny, nz = a/norm, b/norm, c/norm
        
        # 确保法向量向上
        if ny < 0:
            nx, ny, nz = -nx, -ny, -nz
        
        # 计算偏差角（与理想桌面法向量的夹角）
        ideal_normal = np.array([0, 0.9999, 0.0131])
        dot_product = nx * ideal_normal[0] + ny * ideal_normal[1] + nz * ideal_normal[2]
        deviation_angle = np.arccos(np.clip(abs(dot_product), 0, 1)) * 180 / np.pi
        
        return {
            'normal': [nx, ny, nz],
            'deviation_angle': deviation_angle
        }
    
    def process_frame(self, npz_path):
        """处理单帧"""
        try:
            # 加载深度
            depth_mm = self.load_depth(npz_path)
            
            # 检测桌面
            table_mask = self.detect_table_simple(depth_mm)
            if table_mask is None:
                return None
            
            # 获取桌面点云
            all_points = self.depth_to_points(depth_mm)
            
            # 使用mask筛选桌面点
            h, w = depth_mm.shape
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            valid_mask = (depth_mm > self.min_depth) & (depth_mm < self.max_depth)
            
            # 获取桌面区域的3D点
            table_indices = table_mask[valid_mask]
            if np.sum(table_indices) < self.min_table_points:
                return None
                
            table_points = all_points[table_indices]
            
            # 拟合平面
            plane_model, inliers = self.fit_plane(table_points)
            
            # 分析结果
            result = self.analyze_plane(plane_model)
            result['point_count'] = len(table_points)
            result['inlier_ratio'] = len(inliers) / len(table_points)
            
            # 添加内点的质心，用于后续计算d值
            inlier_points = table_points[inliers]
            result['inlier_centroid'] = np.mean(inlier_points, axis=0)
            
            return result
            
        except Exception as e:
            print(f"处理失败: {e}")
            return None


def process_video_sequence(directory_path, min_consensus_frames=10):
    """处理视频序列并输出综合结果"""
    
    detector = SimpleTableDetector()
    
    # 获取所有npz文件
    npz_files = sorted(Path(directory_path).glob("*.npz"))
    print(f"\n找到 {len(npz_files)} 个文件")
    
    # 存储所有结果
    results = []
    
    # 处理每一帧
    print("\n处理中...")
    for i, npz_file in enumerate(npz_files):
        result = detector.process_frame(str(npz_file))
        if result:
            results.append({
                'file': npz_file.name,
                'deviation': result['deviation_angle'],
                'normal': result['normal'],
                'points': result['point_count'],
                'centroid': result['inlier_centroid']  # 保存质心
            })
            print(f"  {npz_file.name}: 偏差 {result['deviation_angle']:.1f}°")
    
    # 检查是否有足够的有效帧
    if len(results) < min_consensus_frames:
        print(f"\n有效帧数不足 ({len(results)} < {min_consensus_frames})")
        return None
    
    # ========== 第一步：计算初步共识 ==========
    all_normals = np.array([r['normal'] for r in results])
    consensus_normal = np.median(all_normals, axis=0)
    consensus_normal = consensus_normal / np.linalg.norm(consensus_normal)
    
    # ========== 第二步：异常帧检测 ==========
    outliers = []
    outlier_threshold = detector.outlier_threshold  # 10度
    
    for i, result in enumerate(results):
        frame_normal = np.array(result['normal'])
        
        # 计算与共识法向量的夹角
        dot_product = np.clip(np.dot(frame_normal, consensus_normal), -1, 1)
        angle_diff = np.arccos(dot_product) * 180 / np.pi
        
        # 判断是否为异常帧
        if angle_diff > outlier_threshold:
            outliers.append({
                'file': result['file'],
                'angle_diff': angle_diff,
                'deviation': result['deviation']
            })
    
    # ========== 第三步：根据异常帧比例决定处理策略 ==========
    outlier_ratio = len(outliers) / len(results)
    
    print("\n" + "="*50)
    
    if outlier_ratio > 0.25:  # 异常帧超过25%
        print("⚠️ 视频素材质量评估")
        print("="*50)
        print(f"\n异常帧比例: {outlier_ratio*100:.1f}% (>{25}%)")
        print("结论: 视频素材质量较差，建议重新采集数据")
        print(f"\n异常帧列表 ({len(outliers)} 个):")
        for outlier in outliers[:10]:  # 最多显示10个
            print(f"  - {outlier['file']}: 偏离共识 {outlier['angle_diff']:.1f}°")
        if len(outliers) > 10:
            print(f"  - ... 还有 {len(outliers)-10} 个异常帧")
        
        print("\n建议:")
        print("  1. 检查相机是否稳定")
        print("  2. 确保场景光照充足且均匀")
        print("  3. 避免快速移动或抖动")
        print("  4. 检查深度相机是否正常工作")
        print("="*50)
        
        return {
            'status': 'poor_quality',
            'outlier_ratio': outlier_ratio,
            'outlier_count': len(outliers),
            'total_frames': len(results)
        }
    
    # ========== 第四步：排除异常帧后重新分析 ==========
    # 过滤掉异常帧
    normal_results = [r for r in results if r['file'] not in [o['file'] for o in outliers]]
    
    print("综合评估结果（已排除异常帧）")
    print("="*50)
    
    # 重新计算所有统计（只使用正常帧）
    normal_deviations = [r['deviation'] for r in normal_results]
    normal_normals = np.array([r['normal'] for r in normal_results])
    
    # 重新计算共识法向量（只基于正常帧）
    refined_consensus_normal = np.median(normal_normals, axis=0)
    refined_consensus_normal = refined_consensus_normal / np.linalg.norm(refined_consensus_normal)
    
    # 计算平面方程的d值
    # 收集所有正常帧的质心
    normal_centroids = np.array([r['centroid'] for r in normal_results])
    # 计算所有质心的平均值，作为共识平面上的一个点
    consensus_point = np.mean(normal_centroids, axis=0)
    # 计算d值：ax + by + cz + d = 0 => d = -(ax + by + cz)
    d_value = -(refined_consensus_normal[0] * consensus_point[0] + 
                refined_consensus_normal[1] * consensus_point[1] + 
                refined_consensus_normal[2] * consensus_point[2])
    
    # 共识偏差角
    ideal_normal = np.array([0, 0.9999, 0.0131])
    consensus_dot_product = np.dot(refined_consensus_normal, ideal_normal)
    consensus_deviation = np.arccos(np.clip(abs(consensus_dot_product), 0, 1)) * 180 / np.pi
    
    # 基本统计
    print(f"\n1. 数据质量:")
    print(f"   - 总帧数: {len(npz_files)}")
    print(f"   - 有效帧数: {len(results)}")
    print(f"   - 正常帧数: {len(normal_results)} (排除了 {len(outliers)} 个异常帧)")
    print(f"   - 异常帧比例: {outlier_ratio*100:.1f}% (<25%, 质量合格)")
    
    print(f"\n2. 统计结果（基于 {len(normal_results)} 个正常帧）:")
    print(f"   - 平均偏差: {np.mean(normal_deviations):.2f}°")
    print(f"   - 偏差范围: [{np.min(normal_deviations):.2f}°, {np.max(normal_deviations):.2f}°]")
    print(f"   - 标准差: {np.std(normal_deviations):.2f}°")
    
    print(f"\n3. 共识平面:")
    print(f"   - 共识法向量: ({refined_consensus_normal[0]:.4f}, {refined_consensus_normal[1]:.4f}, {refined_consensus_normal[2]:.4f})")
    print(f"   - 共识偏差角: {consensus_deviation:.2f}°")
    
    # 稳定性评估（基于正常帧）
    print(f"\n4. 稳定性评估:")
    std_dev = np.std(normal_deviations)
    if std_dev < 1.0:
        print(f"   - 结果稳定性: 极佳 (标准差 < 1°)")
    elif std_dev < 2.0:
        print(f"   - 结果稳定性: 优秀 (标准差 < 2°)")
    elif std_dev < 5.0:
        print(f"   - 结果稳定性: 良好 (标准差 < 5°)")
    else:
        print(f"   - 结果稳定性: 一般 (标准差 >= 5°)")
    
    # 找出最佳和最差帧（只在正常帧中）
    best_idx = np.argmin(normal_deviations)
    worst_idx = np.argmax(normal_deviations)
    
    print(f"\n5. 关键帧分析:")
    print(f"   - 最佳帧: {normal_results[best_idx]['file']} (偏差 {normal_deviations[best_idx]:.2f}°)")
    print(f"   - 最差帧: {normal_results[worst_idx]['file']} (偏差 {normal_deviations[worst_idx]:.2f}°)")
    
    if outliers:
        print(f"\n6. 已排除的异常帧:")
        for i, outlier in enumerate(outliers[:3]):
            print(f"   - {outlier['file']}: 偏离共识 {outlier['angle_diff']:.1f}°")
        if len(outliers) > 3:
            print(f"   - ... 还有 {len(outliers)-3} 个异常帧")
    
    # 最终评估
    print(f"\n7. 最终评估:")
    print(f"   ✓ 数据质量: 合格 (异常率 {outlier_ratio*100:.1f}%)")
    print(f"   ✓ 检测成功率: {len(normal_results)/len(npz_files)*100:.1f}%")
    print(f"   ✓ 推荐使用共识偏差角: {consensus_deviation:.2f}°")
    
    # 与单帧方法对比
    if consensus_deviation < 10.8:  # 假设单帧方法偏差是10.8°
        improvement = 10.8 - consensus_deviation
        print(f"   ✓ 相比单帧方法改进: {improvement:.2f}° ({improvement/10.8*100:.1f}%)")
    
    # 最终结果
    print(f"\n8. 最终结果:")
    print(f"   ► 共识法向量: ({refined_consensus_normal[0]:.6f}, {refined_consensus_normal[1]:.6f}, {refined_consensus_normal[2]:.6f})")
    print(f"   ► 共识偏差角: {consensus_deviation:.4f}°")
    print(f"   ► 共识平面质心: ({consensus_point[0]:.6f}, {consensus_point[1]:.6f}, {consensus_point[2]:.6f}) m")
    print(f"   ► 共识平面方程: {refined_consensus_normal[0]:.6f}x + {refined_consensus_normal[1]:.6f}y + {refined_consensus_normal[2]:.6f}z + {d_value:.6f} = 0")
    print(f"   ► 平面到原点距离: {abs(d_value):.6f} m")
    
    print("="*50)
    
    return {
        'status': 'success',
        'consensus_deviation': consensus_deviation,
        'consensus_normal': refined_consensus_normal.tolist(),
        'consensus_point': consensus_point.tolist(),
        'plane_d': d_value,
        'plane_equation': {
            'a': refined_consensus_normal[0],
            'b': refined_consensus_normal[1],
            'c': refined_consensus_normal[2],
            'd': d_value
        },
        'mean_deviation': np.mean(normal_deviations),
        'std_deviation': np.std(normal_deviations),
        'min_deviation': np.min(normal_deviations),
        'max_deviation': np.max(normal_deviations),
        'success_rate': len(normal_results)/len(npz_files),
        'outlier_count': len(outliers),
        'outlier_ratio': outlier_ratio,
        'normal_frame_count': len(normal_results),
        'total_frame_count': len(results)
    }


if __name__ == "__main__":
    # 设置你的NPZ文件目录
    directory = "C:/Users/14101/Desktop/table/pre_depth"  # 修改为你的路径
    
    # 如果目录不存在，使用当前目录
    if not Path(directory).exists():
        directory = "."
        print(f"使用当前目录: {Path(directory).absolute()}")
    
    # 处理并获取结果
    final_result = process_video_sequence(directory)