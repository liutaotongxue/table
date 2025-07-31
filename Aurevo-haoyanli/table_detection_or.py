import cv2
import os
from ultralytics import YOLO
from config import ParamatersSetting as PS
import torch
import numpy as np
import open3d as o3d

def draw_table_plane_overlay(
    rgb_image, plane_model, camera_params_rgb
):  # <--- MODIFIED: params specific
    """
    在RGB图像上绘制桌面平面的可视化覆盖 (当平面被锁定时)
    """
    if plane_model is None or rgb_image is None or camera_params_rgb is None:
        return rgb_image

    try:
        # 获取RGB相机参数 (如果需要用于投影，但这里主要显示信息)
        # fx_rgb, fy_rgb, cx_rgb, cy_rgb = camera_params_rgb
        height, width = rgb_image.shape[:2]
        overlay = rgb_image.copy()
        a, b, c, d = plane_model

        # 在图像底部绘制平面方程 (避免与顶部信息冲突)
        text_y_start = height - 70  # Adjust if needed

        cv2.putText(
            overlay,
            "Locked Plane (ax+by+cz+d=0):",
            (10, text_y_start),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Smaller font
            (255, 150, 0),  # Different color for distinction
            1,
        )
        cv2.putText(
            overlay,
            f"a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}",
            (10, text_y_start + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,  # Smaller font
            (255, 150, 0),
            1,
        )

        # 简单的指示器 (例如，在图像中心附近)
        # center_x, center_y = width // 2, height // 2
        # cv2.circle(overlay, (center_x, center_y), 5, (255, 150, 0), 1)
        # cv2.putText(
        #     overlay,
        #     "Ref",
        #     (center_x + 10, center_y + 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.4,
        #     (255, 150, 0),
        #     1,
        # )
        return overlay

    except Exception as e:
        print(f"绘制桌面平面覆盖失败: {e}")
        return rgb_image

def initialize_yolo_model():
    """初始化YOLO模型"""

    if not os.path.exists(PS.YOLO_MODEL_NAME):
        print(f"错误: YOLO模型文件不存在: {PS.YOLO_MODEL_NAME}")
        return False

    try:
        # 强制使用CPU以避免CUDA兼容性问题
        PS.yolo_device = "cpu"
        PS.yolo_model = YOLO(PS.YOLO_MODEL_NAME)
        PS.yolo_model.to(PS.yolo_device)
        print(f"YOLO模型加载成功，使用设备: {PS.yolo_device}")
        return True
    except Exception as e:
        print(f"错误: YOLO模型加载失败: {e}")
        return False

def detect_table_with_yolo(rgb_image):
    """
    使用YOLO检测桌面
    """

    if PS.yolo_model is None or rgb_image is None:
        return None, None

    try:
        # YOLO推理
        results = PS.yolo_model(
            rgb_image,
            conf=PS.YOLO_CONF_THRESHOLD,
            iou=PS.YOLO_IOU_THRESHOLD,
            classes=PS.TABLE_CLASS_ID,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None, None

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return None, None

        # 获取最大的桌面掩码
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()

        if len(masks) == 0:
            return None, None

        # 选择面积最大的掩码
        areas = [np.sum(mask) for mask in masks]
        max_idx = np.argmax(areas)

        table_mask = masks[max_idx]
        table_box = boxes[max_idx]

        # 调整掩码尺寸到RGB图像大小
        if table_mask.shape != rgb_image.shape[:2]:
            table_mask = cv2.resize(
                table_mask.astype(np.uint8),
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        return table_mask, table_box

    except Exception as e:
        print(f"错误: YOLO桌面检测失败: {e}")
        return None, None


def process_table_mask(mask):
    """
    对桌面掩码进行后处理
    """
    if mask is None:
        return None

    try:
        # 转换为uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # 形态学操作
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PS.MASK_OPEN_KERNEL_SIZE, PS.MASK_OPEN_KERNEL_SIZE)
        )
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (PS.MASK_CLOSE_KERNEL_SIZE, PS.MASK_CLOSE_KERNEL_SIZE)
        )

        # 开运算去除噪声
        mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_kernel)
        # 闭运算填补空洞
        mask_processed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, close_kernel)

        # 面积滤波
        contours, _ = cv2.findContours(
            mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return None  # 保留最大面积的轮廓
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))  # Ensure max_idx is an int
        max_area = areas[max_idx]

        image_area = mask_processed.shape[0] * mask_processed.shape[1]
        if max_area < image_area * PS.MASK_MIN_AREA_PERCENT:
            return None

        # 创建最终掩码
        final_mask = np.zeros_like(mask_processed)
        selected_contour = contours[max_idx]
        cv2.fillPoly(final_mask, [selected_contour], (255,))

        return final_mask.astype(bool)

    except Exception as e:
        print(f"错误: 掩码后处理失败: {e}")
        return mask

# ==============================================================================
# === 平面拟合函数 ===
# ==============================================================================
def fit_table_plane_with_ransac(depth_mm, table_mask_depth):
    """
    使用RANSAC拟合桌面平面
    """
    if depth_mm is None or table_mask_depth is None:
        return None, 0

    try:
        # 获取有效深度点
        valid_mask = (
            (depth_mm > PS.MIN_DEPTH_MM) & (depth_mm < PS.MAX_DEPTH_MM) & table_mask_depth
        )
        if np.sum(valid_mask) < PS.MIN_POINTS_FOR_PLANE_FIT:
            return None, 0  # <--- 失败时返回 0 分

        # 转换为3D点云
        valid_pixels = np.where(valid_mask)
        y_coords, x_coords = valid_pixels
        z_coords = depth_mm[valid_pixels] / 1000.0  # 转换为米        # 转换为相机坐标系
        if PS.fx_ir is None or PS.fy_ir is None or PS.cx_ir is None or PS.cy_ir is None:
            print("错误: IR相机内参未初始化")
            return None, 0

        x_cam = (x_coords - PS.cx_ir) * z_coords / PS.fx_ir
        y_cam = (y_coords - PS.cy_ir) * z_coords / PS.fy_ir

        points_3d = np.column_stack([x_cam, y_cam, z_coords])

        if len(points_3d) < PS.MIN_POINTS_FOR_PLANE_FIT:
            return None, 0

        # 创建Open3D点云
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_3d)

        # RANSAC平面拟合
        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=PS.PLANE_FIT_DISTANCE_THRESHOLD_M,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inliers) < PS.MIN_INLIERS_FOR_PLANE_FIT:
            return None, 0

        # 提取平面参数 [a, b, c, d]: ax + by + cz + d = 0
        a, b, c, d = (
            plane_model  # 确保法向量指向相机 (c < 0 if plane is in front and normal points to camera)
        )
        # Or ensure d is consistent if plane equation is ax+by+cz=d
        if (
            c > 0
        ):  # This convention depends on how d is defined. If ax+by+cz+d=0 and origin is camera, d is related to distance.
            # If normal (a,b,c) points towards camera from plane, c should be negative if z is positive outwards.
            # Let's assume standard ax+by+cz+d=0, with (a,b,c) as normal vector.
            # If we want normal to generally point "towards" camera from typical table below, its z component would be positive.
            # However, Open3D's convention might vary. The key is consistency.
            # The original code had: if c > 0: a,b,c,d = -a,-b,-c,-d. This flips the normal and d.
            # This ensures that 'c' has a particular sign, perhaps for consistency in interpreting 'd'.
            a, b, c, d = -a, -b, -c, -d

        # 返回平面模型和内点数量（作为分数）
        return np.array([a, b, c, d], dtype=np.float64), len(
            inliers
        )  # <--- 修改返回值, ensure numpy array

    except Exception as e:
        print(f"错误: 平面拟合失败: {e}")
        return None, 0

