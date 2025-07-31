"""
简单的平面方程读取器
直接读取已保存的平面方程或从深度数据拟合平面
"""

import numpy as np
import json
import os
from pathlib import Path

def read_saved_plane():
    """读取已保存的平面方程"""
    plane_file = "Aurevo-haoyanli/parameters/table_plane_calibration.json"
    
    try:
        with open(plane_file, 'r') as f:
            data = json.load(f)
            plane = data['plane_model']
            print("\n已保存的桌面平面方程：")
            print(f"  {plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0")
            print(f"\n具体参数：")
            print(f"  a = {plane[0]:.6f}")
            print(f"  b = {plane[1]:.6f}")
            print(f"  c = {plane[2]:.6f}")
            print(f"  d = {plane[3]:.6f}")
            
            # 计算法向量的角度
            import math
            # 法向量 (a, b, c)
            norm = math.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
            nx, ny, nz = plane[0]/norm, plane[1]/norm, plane[2]/norm
            
            # 与z轴的夹角（通常桌面接近水平）
            angle_with_z = math.acos(abs(nz)) * 180 / math.pi
            print(f"\n平面分析：")
            print(f"  法向量：({nx:.3f}, {ny:.3f}, {nz:.3f})")
            print(f"  与垂直方向夹角：{angle_with_z:.1f}°")
            
            # 计算原点到平面的距离
            distance = abs(plane[3]) / norm
            print(f"  原点到平面距离：{distance:.3f}m")
            
            return plane
            
    except FileNotFoundError:
        print(f"错误：找不到平面标定文件 {plane_file}")
        return None
    except Exception as e:
        print(f"错误：读取平面数据失败 - {e}")
        return None

def simple_plane_from_depth(depth_file_path):
    """从深度数据直接拟合平面（不使用YOLO检测）"""
    try:
        # 加载深度数据
        depth_data = np.load(depth_file_path)
        
        # 获取深度数组
        if 'depth' in depth_data:
            depth_mm = depth_data['depth'].astype(np.float32)
        elif 'depth_mm' in depth_data:
            depth_mm = depth_data['depth_mm'].astype(np.float32)
        else:
            keys = list(depth_data.keys())
            if keys:
                depth_mm = depth_data[keys[0]].astype(np.float32)
            else:
                return None
        
        # 简单的深度过滤（假设桌面在一定深度范围内）
        valid_mask = (depth_mm > 100) & (depth_mm < 1000)  # 10cm到1m
        
        if np.sum(valid_mask) < 100:
            return None
        
        # 获取有效点的坐标
        y_coords, x_coords = np.where(valid_mask)
        z_values = depth_mm[valid_mask]
        
        # 使用最小二乘法拟合平面
        # 平面方程：z = ax + by + c
        # 转换为：ax + by - z + c = 0
        A = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])
        coeffs, _, _, _ = np.linalg.lstsq(A, z_values, rcond=None)
        
        # 转换为标准形式 ax + by + cz + d = 0
        a, b, c = coeffs[0], coeffs[1], -1.0
        d = coeffs[2]
        
        # 归一化
        norm = np.sqrt(a**2 + b**2 + c**2)
        plane = [a/norm, b/norm, c/norm, d/norm]
        
        return np.array(plane)
        
    except Exception as e:
        print(f"处理深度数据时出错：{e}")
        return None

def scan_all_depth_files():
    """扫描所有深度文件并尝试拟合平面"""
    base_path = Path("C:/Users/14101/Desktop/table")
    depth_dir = base_path / "pre_depth"
    
    print("\n扫描所有深度文件...")
    
    valid_planes = []
    
    for i in range(194):
        depth_path = depth_dir / f"{i}.npz"
        if depth_path.exists():
            plane = simple_plane_from_depth(str(depth_path))
            if plane is not None:
                valid_planes.append({
                    'index': i,
                    'plane': plane,
                    'vertical_angle': np.arccos(abs(plane[2])) * 180 / np.pi
                })
    
    if valid_planes:
        print(f"\n成功从 {len(valid_planes)} 个深度文件中拟合出平面")
        
        # 找出最接近水平的平面（与垂直方向夹角最大）
        best = max(valid_planes, key=lambda x: x['vertical_angle'])
        print(f"\n最佳平面（来自文件 {best['index']}.npz）：")
        plane = best['plane']
        print(f"  方程：{plane[0]:.6f}x + {plane[1]:.6f}y + {plane[2]:.6f}z + {plane[3]:.6f} = 0")
        print(f"  与垂直方向夹角：{best['vertical_angle']:.1f}°")

def main():
    print("桌面平面方程获取工具")
    print("=" * 50)
    
    # 1. 读取已保存的平面方程
    saved_plane = read_saved_plane()
    
    # 2. 询问是否扫描深度文件
    if saved_plane is not None:
        scan = input("\n是否扫描深度文件以查找其他可能的平面？(y/n): ")
        if scan.lower() == 'y':
            scan_all_depth_files()
    else:
        print("\n未找到已保存的平面方程，开始扫描深度文件...")
        scan_all_depth_files()

if __name__ == "__main__":
    main()