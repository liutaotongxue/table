import os

import numpy as np
from config import ParamatersSetting as PS
from calibration_and_correction import depth_bytes_to_mm_array,apply_depth_correction,apply_temporal_filter,apply_spatial_filter
import cv2
from table_detection_or import detect_table_with_yolo,process_table_mask,fit_table_plane_with_ransac,draw_table_plane_overlay
from eye_recognition import detect_eyes_with_mediapipe


def process_frame(depth_bytes, rgb_image, cam_config):
    """
    处理单帧数据
    """

    # 解析相机深度模式
    cam_deep_mode = cam_config[1]

    # 1. 深度数据处理
    depth_mm_pre = depth_bytes_to_mm_array(depth_bytes, cam_deep_mode)
    if depth_mm_pre is None:
        # Create a dummy black image for depth if processing failed, to allow RGB processing
        depth_mm_final = np.zeros((PS.DEPTH_H, PS.DEPTH_W), dtype=np.float32)
    else:
        depth_mm_corrected = apply_depth_correction(depth_mm_pre)
        depth_mm_temporal = apply_temporal_filter(depth_mm_corrected)
        depth_mm_final = apply_spatial_filter(depth_mm_temporal)
        if (
            depth_mm_final is None
        ):  # Should not happen if spatial filter returns input on error
            depth_mm_final = (
                depth_mm_temporal
                if depth_mm_temporal is not None
                else depth_mm_corrected
            )

    # 2. RGB图像去畸变
    rgb_undistorted = None
    if rgb_image is not None:
        if PS.mapx_rgb is not None and PS.mapy_rgb is not None:
            try:
                # Ensure rgb_image is contiguous, sometimes helps remap
                if not rgb_image.flags["C_CONTIGUOUS"]:
                    rgb_image = np.ascontiguousarray(rgb_image)
                rgb_undistorted = cv2.remap(
                    rgb_image, PS.mapx_rgb, PS.mapy_rgb, cv2.INTER_LINEAR
                )
            except Exception as e:
                print(f"RGB去畸变失败: {e}")
                rgb_undistorted = rgb_image.copy()  # Use original if remap fails
        else:
            rgb_undistorted = rgb_image.copy()

    # If rgb_undistorted is still None (e.g. rgb_image was None), create a placeholder
    if rgb_undistorted is None:
        # Create a black RGB image of expected output size if RGB processing failed
        rgb_undistorted = np.zeros(
            (
                PS.calib_rgb_h if PS.calib_rgb_h > 0 else PS.RGB_H_OUTPUT,
                PS.calib_rgb_w if PS.calib_rgb_w > 0 else PS.RGB_W_OUTPUT,
                3,
            ),
            dtype=np.uint8,
        )

    # 3. YOLO桌面检测
    table_mask_rgb, table_box = detect_table_with_yolo(rgb_undistorted)

    # 4. 桌面掩码处理和转换
    table_mask_processed = None
    table_mask_depth = None
    if table_mask_rgb is not None:
        table_mask_processed = process_table_mask(table_mask_rgb)

        if table_mask_processed is not None:
            # 将RGB掩码转换为深度图尺寸
            table_mask_depth = cv2.resize(
                table_mask_processed.astype(np.uint8),
                (PS.DEPTH_W, PS.DEPTH_H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    # 5. 平面拟合 (This is for real-time plane detection if not locked)
    current_frame_plane_model = None
    current_frame_plane_score = 0
    # inlier_points = None # Not currently used or returned by fit_table_plane_with_ransac
    if table_mask_depth is not None and depth_mm_final is not None:
        current_frame_plane_model, current_frame_plane_score = (
            fit_table_plane_with_ransac(depth_mm_final, table_mask_depth)
        )
    
    # 6. 眼部检测
    left_eye_center, right_eye_center = detect_eyes_with_mediapipe(rgb_undistorted)

    # 7. 计算眼部到桌面距离 (This part is tricky, depends on whether using locked or current plane)
    # The actual distance calculation against LOCKED_PLANE_MODEL or current_frame_plane_model
    # will be handled in the main loop after deciding which plane to use.
    # Here, we just gather the raw detection results.
    # The `ema_eye_to_table_dist_mm` will be updated in the main loop.

    return {
        "depth_mm": depth_mm_final,
        "rgb_undistorted": rgb_undistorted,
        "table_mask_rgb": table_mask_processed,
        "table_mask_depth": table_mask_depth,
        "plane_model": current_frame_plane_model,  # Plane detected in this frame
        "plane_score": current_frame_plane_score,  # Score for this frame's plane
        # "inlier_points": inlier_points, # Not used
        "left_eye_center": left_eye_center,
        "right_eye_center": right_eye_center,
        # "eye_to_table_dist": None, # Will be calculated in main
        # "ema_eye_to_table_dist": ema_eye_to_table_dist_mm, # Will be updated in main
        "table_box": table_box,
    }


# ==============================================================================
# === 可视化函数 ===
# ==============================================================================
def visualize_results(
    results, plane_is_locked_display_flag=False, locked_plane_details_for_viz=None
):
    """
    可视化检测结果
    plane_is_locked_display_flag: Controls general "locked" appearance (color, text)
    locked_plane_details_for_viz: Specific plane model to draw details for (equation)
    """
    if results is None:
        return None

    try:
        rgb_vis = (
            results["rgb_undistorted"].copy()
            if results["rgb_undistorted"] is not None
            else np.zeros((PS.RGB_H_OUTPUT, PS.RGB_W_OUTPUT, 3), dtype=np.uint8)  # Fallback
        )

        # 确定掩码和边框颜色
        # If plane_is_locked_display_flag is true, means we are using a fixed plane (calibrated/loaded)
        overlay_color = (
            [255, 0, 0] if plane_is_locked_display_flag else [0, 255, 0]
        )  # Locked:Blue, Real-time:Green

        # 绘制桌面检测结果 (mask and box from current frame's YOLO)
        if results["table_mask_rgb"] is not None:
            mask_overlay = np.zeros_like(rgb_vis)
            mask_overlay[results["table_mask_rgb"]] = (
                overlay_color  # Color indicates plane source
            )
            rgb_vis = cv2.addWeighted(rgb_vis, 0.7, mask_overlay, 0.3, 0)

            if results["table_box"] is not None:
                x1, y1, x2, y2 = results["table_box"][:4].astype(int)
                cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), overlay_color, 2)
                cv2.putText(
                    rgb_vis,
                    "Table",  # This is current YOLO detection
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    overlay_color,
                    2,
                )

        # If a specific locked plane's details are provided, draw them
        if (
            locked_plane_details_for_viz is not None and PS.fx_rgb is not None
        ):  # Check fx_rgb to ensure calib loaded
            camera_params_rgb_tuple = (PS.fx_rgb, PS.fy_rgb, PS.cx_rgb, PS.cy_rgb)
            rgb_vis = draw_table_plane_overlay(
                rgb_vis, locked_plane_details_for_viz, camera_params_rgb_tuple
            )

        # 绘制眼部检测结果
        if results["left_eye_center"] is not None:
            cv2.circle(
                rgb_vis, tuple(results["left_eye_center"]), 5, (0, 0, 255), -1
            )  # Red for eyes
            cv2.putText(
                rgb_vis,
                "L",
                tuple(results["left_eye_center"] + np.array([10, 0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        if results["right_eye_center"] is not None:
            cv2.circle(rgb_vis, tuple(results["right_eye_center"]), 5, (0, 0, 255), -1)
            cv2.putText(
                rgb_vis,
                "R",
                tuple(results["right_eye_center"] + np.array([10, 0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # 显示距离信息 (These are passed in results by main loop after calculation)
        info_y = 30
        if (
            results.get("eye_to_table_dist_final") is not None
        ):  # Key name changed for clarity
            dist_text = f"Eye-Table: {results['eye_to_table_dist_final']:.1f} mm"
            cv2.putText(
                rgb_vis,
                dist_text,
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # White
                2,
            )
            info_y += 25

        if results.get("ema_eye_to_table_dist_final") is not None:  # Key name changed
            ema_text = f"EMA: {results['ema_eye_to_table_dist_final']:.1f} mm"
            cv2.putText(
                rgb_vis,
                ema_text,
                (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # Yellow
                2,
            )
            info_y += 25

        # 显示深度模式
        mode_text = f"Depth: {'16b' if PS.DEPTH_MODE == 0 else '8b'}"  # Shorter text
        cv2.putText(
            rgb_vis,
            mode_text,
            (10, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),  # Cyan
            2,
        )

        # 显示锁定状态 (controlled by plane_is_locked_display_flag)
        if plane_is_locked_display_flag:  # If true, implies we are using a fixed plane
            lock_text = "LOCKED"
            text_color = (255, 100, 100)  # Light Blue for locked
        else:
            lock_text = "REAL-TIME"
            text_color = (100, 255, 100)  # Light Green for real-time

        # Position for LOCK_TEXT / REAL-TIME (top-right)
        text_size, _ = cv2.getTextSize(lock_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            rgb_vis,
            lock_text,
            (rgb_vis.shape[1] - text_size[0] - 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )

        return rgb_vis

    except Exception as e:
        print(f"错误: 可视化失败: {e}")
        return results.get(
            "rgb_undistorted", np.zeros((PS.RGB_H_OUTPUT, PS.RGB_W_OUTPUT, 3), dtype=np.uint8)
        )


if __name__ == '__main__':

    from commucation_and_decode import frame_config_decode,frame_payload_decode
    # 读取二进制文件

    with open(f'raw_frame_data/pic_1753154592.2118583', 'rb') as f:  # 'rb' 表示二进制读取模式
        raw_frame_data = f.read()  # 读取全部内容到字节对象（bytes）

    cam_config_tuple = frame_config_decode(raw_frame_data[16:28])

    depth_bytes, _, _, rgb_image_raw = frame_payload_decode(
        raw_frame_data[28:], cam_config_tuple
    )
    from calibration_and_correction import load_calibration_data

    results = process_frame(depth_bytes, rgb_image_raw, cam_config_tuple)
    # rgb_image_raw=cv2.imread('rgb_image_raw.png')
    # cv2.imshow('image',rgb_image_raw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # from a import masks
    # print(masks.shape)