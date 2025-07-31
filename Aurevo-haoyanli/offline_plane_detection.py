"""
离线桌面平面检测脚本
从预存的深度和图像数据中检测桌面并计算平面方程
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config import ParamatersSetting as PS
from calibration_and_correction import (
    load_calibration_data, 
    load_depth_correction_model,
    depth_bytes_to_mm_array,
    apply_depth_correction,
    apply_temporal_filter,
    apply_spatial_filter
)
from table_detection_or import (
    initialize_yolo_model,
    detect_table_with_yolo,
    process_table_mask,
    fit_table_plane_with_ransac
)
from frame_process_and_visualize_or import visualize_results

def process_saved_data(image_path, depth_path):
    """处理保存的图像和深度数据"""
    
    print(f"\n处理数据：")
    print(f"  图像：{image_path}")
    print(f"  深度：{depth_path}")
    
    # 1. 读取图像
    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        print(f"错误：无法读取图像 {image_path}")
        return None
    
    # 2. 读取深度数据
    depth_data = np.load(depth_path)
    
    # 检查深度数据格式
    if 'depth' in depth_data:
        depth_mm = depth_data['depth'].astype(np.float32)
    elif 'depth_mm' in depth_data:
        depth_mm = depth_data['depth_mm'].astype(np.float32)
    else:
        # 尝试获取第一个数组
        keys = list(depth_data.keys())
        if keys:
            depth_mm = depth_data[keys[0]].astype(np.float32)
            print(f"  使用深度数据键：{keys[0]}")
        else:
            print("错误：深度文件中没有找到有效数据")
            return None
    
    print(f"  深度图尺寸：{depth_mm.shape}")
    print(f"  深度范围：{np.min(depth_mm):.1f} - {np.max(depth_mm):.1f} mm")
    
    # 3. 检测桌面
    table_mask_rgb, table_box = detect_table_with_yolo(rgb_image)
    
    if table_mask_rgb is None:
        print("  未检测到桌面")
        return None
    
    # 4. 处理掩码
    table_mask_processed = process_table_mask(table_mask_rgb)
    
    if table_mask_processed is None:
        print("  桌面掩码处理失败")
        return None
    
    # 5. 调整掩码尺寸以匹配深度图
    if table_mask_processed.shape != depth_mm.shape:
        table_mask_depth = cv2.resize(
            table_mask_processed.astype(np.uint8),
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        table_mask_depth = table_mask_processed
    
    # 6. 拟合平面
    plane_model, plane_score = fit_table_plane_with_ransac(depth_mm, table_mask_depth)
    
    if plane_model is None:
        print("  平面拟合失败")
        return None
    
    print(f"\n✅ 成功拟合平面！")
    print(f"  平面方程：{plane_model[0]:.4f}x + {plane_model[1]:.4f}y + {plane_model[2]:.4f}z + {plane_model[3]:.4f} = 0")
    print(f"  内点数量：{plane_score}")
    
    # 7. 可视化结果
    results = {
        "depth_mm": depth_mm,
        "rgb_undistorted": rgb_image,
        "table_mask_rgb": table_mask_processed,
        "table_mask_depth": table_mask_depth,
        "plane_model": plane_model,
        "plane_score": plane_score,
        "table_box": table_box,
        "left_eye_center": None,
        "right_eye_center": None,
    }
    
    vis_image = visualize_results(results, plane_is_locked_display_flag=True, locked_plane_details_for_viz=plane_model)
    
    return {
        "plane_model": plane_model,
        "plane_score": plane_score,
        "visualization": vis_image
    }

def main():
    """主函数"""
    
    print("离线桌面平面检测程序")
    print("=" * 50)
    
    # 初始化
    print("\n初始化系统...")
    
    # 1. 加载标定数据
    if not load_calibration_data(use_16bit_mode=True):
        print("标定数据加载失败")
        return
    
    # 2. 初始化YOLO
    if not initialize_yolo_model():
        print("YOLO模型初始化失败")
        return
    
    print("系统初始化完成！")
    
    # 设置数据路径
    base_path = Path("C:/Users/14101/Desktop/table")
    pic_dir = base_path / "pic"
    depth_dir = base_path / "pre_depth"
    
    # 处理特定文件或批量处理
    process_single = True  # 设置为False以处理所有文件
    
    if process_single:
        # 处理单个文件示例
        image_idx = 50  # 您可以修改这个索引
        image_path = pic_dir / f"{image_idx}.png"
        depth_path = depth_dir / f"{image_idx}.npz"
        
        if image_path.exists() and depth_path.exists():
            result = process_saved_data(str(image_path), str(depth_path))
            
            if result:
                # 显示结果
                cv2.imshow("Plane Detection Result", result["visualization"])
                print("\n按任意键关闭窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 保存平面方程
                save_result = input("\n是否保存这个平面方程？(y/n): ")
                if save_result.lower() == 'y':
                    from calibration_and_correction import save_plane_calibration
                    save_plane_calibration(result["plane_model"])
                    print("平面方程已保存！")
        else:
            print(f"文件不存在：{image_path} 或 {depth_path}")
    
    else:
        # 批量处理所有文件
        all_planes = []
        
        for i in range(194):  # 0-193
            image_path = pic_dir / f"{i}.png"
            depth_path = depth_dir / f"{i}.npz"
            
            if image_path.exists() and depth_path.exists():
                result = process_saved_data(str(image_path), str(depth_path))
                if result:
                    all_planes.append({
                        "index": i,
                        "plane": result["plane_model"],
                        "score": result["plane_score"]
                    })
        
        # 找出最佳平面
        if all_planes:
            best_plane = max(all_planes, key=lambda x: x["score"])
            print(f"\n最佳平面（来自图像 {best_plane['index']}）：")
            plane = best_plane["plane"]
            print(f"  方程：{plane[0]:.4f}x + {plane[1]:.4f}y + {plane[2]:.4f}z + {plane[3]:.4f} = 0")
            print(f"  内点数：{best_plane['score']}")

if __name__ == "__main__":
    main()