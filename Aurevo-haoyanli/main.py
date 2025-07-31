import cv2
import argparse
from config import ParamatersSetting as PS
from calibration_and_correction import load_calibration_data,load_depth_correction_model,load_plane_calibration,save_plane_calibration
from commucation_and_decode import *
from table_detection_or import initialize_yolo_model
from eye_recognition import initialize_mediapipe,get_eye_depth_from_neighbor
from frame_process_and_visualize_or import *
from caculate_distance import *
import eye_recognition as er
import time


def main():
    # 引用全局变量，以便在函数内修改它们

    parser = argparse.ArgumentParser(description="统一的眼部到桌面检测程序")
    parser.add_argument("--host", default=PS.DEFAULT_HOST, help="相机IP地址")
    parser.add_argument("--port", type=int, default=PS.DEFAULT_PORT, help="相机端口")
    parser.add_argument(
        "--depth-mode",
        type=int,
        choices=[0, 1],
        default=None,  # Default to config file or 0
        help="深度模式: 0=16-bit, 1=8-bit (覆盖配置文件)",
    )
    parser.add_argument("--save-video", action="store_true", help="保存输出视频")
    parser.add_argument(
        "--output-video", default="unified_detection_output.avi", help="输出视频文件名"
    )
    args = parser.parse_args()

    # --- 初始化 ---
    print("初始化系统组件...")

    if args.depth_mode is not None:  # CLI argument overrides config
        PS.DEPTH_MODE = args.depth_mode

    use_16bit = PS.DEPTH_MODE == 0
    print(f"使用深度模式: {'16-bit' if use_16bit else '8-bit'}")

    if not load_calibration_data(use_16bit):
        print("标定数据加载失败，退出。")
        return
    load_depth_correction_model(use_16bit)  # Optional, continues if not found
    if not initialize_yolo_model():
        print("YOLO模型初始化失败，退出。")
        return
    if not initialize_mediapipe():
        print("MediaPipe初始化可能存在问题，但程序将继续。")
        # Allow continuation even if mediapipe has issues, as per its init function

    # --- Attempt to load saved table plane calibration ---
    loaded_calibrated_plane = load_plane_calibration(PS.TABLE_PLANE_CALIBRATION_FILE)
    if loaded_calibrated_plane is not None:
        PS.LOCKED_PLANE_MODEL = loaded_calibrated_plane
        PS.PLANE_IS_LOCKED = True
        PS.IN_CALIBRATION_MODE = False  # Skip automatic calibration
        print(f"成功加载已保存的桌面平面标定: {PS.TABLE_PLANE_CALIBRATION_FILE}")
        pm = PS.LOCKED_PLANE_MODEL
        print(
            f"  锁定的平面方程: a={pm[0]:.4f}, b={pm[1]:.4f}, c={pm[2]:.4f}, d={pm[3]:.4f}"
        )
    else:
        print("未找到或加载已保存的桌面平面。将进行自动校准。")
        # IN_CALIBRATION_MODE is already True by default

    print(f"配置相机: {args.host}:{args.port}")
    config_bytes = frame_config_encode()  # Uses global DEPTH_MODE
    if config_bytes is None or not post_encode_config(
        config_bytes, args.host, args.port
    ):
        print("相机配置失败，退出程序")
        return
    time.sleep(1)  # Allow camera to apply settings

    video_writer = None
    if args.save_video:
        # Use calib_rgb_w, calib_rgb_h which are loaded from calibration file
        fourcc = cv2.VideoWriter.fourcc(*"XVID")
        video_writer = cv2.VideoWriter(
            args.output_video,
            fourcc,
            10.0,  # Adjusted FPS for video
            (
                PS.calib_rgb_w if PS.calib_rgb_w > 0 else PS.RGB_W_OUTPUT,
                PS.calib_rgb_h if PS.calib_rgb_h > 0 else PS.RGB_H_OUTPUT,
            ),
        )

    if PS.IN_CALIBRATION_MODE:  # Only print this if we are actually calibrating
        print("\n=========================================================")
        print(f"开始自动校准阶段，将持续 {PS.CALIBRATION_FRAMES_TOTAL} 帧...")
        print("请确保桌面稳定且清晰可见。")
        print("=========================================================\n")

    print("按 'q' 退出, 's' 切换深度模式 (将重置校准), 'u' 解锁并重新校准。")

    # --- 主处理循环 ---
    frame_display_count = 0  # For FPS calculation
    fps_time_start = time.time()
    try:
        while True:
            # 1. 获取帧数据
            raw_frame_data = get_frame_from_http(args.host, args.port)
            if raw_frame_data is None or len(raw_frame_data) < 28:
                time.sleep(0.01)  # Avoid busy loop if no data
                continue

            cam_config_tuple = frame_config_decode(raw_frame_data[16:28])
            if cam_config_tuple is None:
                continue

            # Ensure DEPTH_MODE matches camera's actual mode if there's a mismatch
            # This is important if camera could be configured externally or startup race condition
            if cam_config_tuple[1] != PS.DEPTH_MODE:
                print(
                    f"警告: 相机回报深度模式 ({cam_config_tuple[1]}) 与程序设定 ({PS.DEPTH_MODE}) 不符。请检查配置或重启。"
                )
                # Could attempt to reconfigure or adapt, for now, just warn.
                # Or, trust the camera's report:
                # DEPTH_MODE = cam_config_tuple[1]
                # print(f"  已根据相机回报自动调整程序深度模式为: {DEPTH_MODE}")
                # use_16bit = DEPTH_MODE == 0
                # load_calibration_data(use_16bit)
                # load_depth_correction_model(use_16bit)

            depth_bytes, _, _, rgb_image_raw = frame_payload_decode(
                raw_frame_data[28:], cam_config_tuple
            )
            # print('rgb_image_raw',rgb_image_raw)
            if depth_bytes is None and rgb_image_raw is None:
                continue

            # 2. 核心处理 (YOLO, Mediapipe, depth processing, real-time plane fit)
            results = process_frame(depth_bytes, rgb_image_raw, cam_config_tuple)
            if results is None:  # Should ideally return a minimal dict even on failure
                cv2.waitKey(1)
                continue

            # 3. 状态管理和逻辑处理
            # 3.1 自动校准模式
            if PS.IN_CALIBRATION_MODE:
                PS.calibration_frame_count += 1
                current_plane_model_in_frame = results.get("plane_model")
                current_plane_score_in_frame = results.get("plane_score", 0)

                if (
                    current_plane_model_in_frame is not None
                    and current_plane_score_in_frame > PS.best_plane_score
                ):
                    PS.best_plane_score = current_plane_score_in_frame
                    PS.best_plane_model = current_plane_model_in_frame
                    print(
                        f"校准中 [帧 {PS.calibration_frame_count}/{PS.CALIBRATION_FRAMES_TOTAL}]: "
                        f"更优平面，分数(内点数): {PS.best_plane_score}"
                    )

                if PS.calibration_frame_count >= PS.CALIBRATION_FRAMES_TOTAL:
                    PS.IN_CALIBRATION_MODE = False
                    PS.calibration_frame_count = 0  # Reset for next potential calibration
                    if PS.best_plane_model is not None:
                        PS.LOCKED_PLANE_MODEL = PS.best_plane_model
                        PS.PLANE_IS_LOCKED = True
                        save_plane_calibration(
                            PS.LOCKED_PLANE_MODEL, PS.TABLE_PLANE_CALIBRATION_FILE
                        )  # <--- SAVE HERE
                        print("\n=================== 校准完成 ===================")
                        print(
                            f"已自动锁定最优桌面平面！数据已保存到 {PS.TABLE_PLANE_CALIBRATION_FILE}"
                        )
                        pm = PS.LOCKED_PLANE_MODEL
                        print(
                            f"  锁定的平面方程: a={pm[0]:.4f}, b={pm[1]:.4f}, c={pm[2]:.4f}, d={pm[3]:.4f}"
                        )
                        print("==============================================\n")
                    else:
                        print(
                            "\n警告: 校准结束，但未能找到任何可用桌面。请检查场景或按 'u' 重新校准。\n"
                        )
                        # PLANE_IS_LOCKED remains False, LOCKED_PLANE_MODEL remains None

            # 3.2 根据锁定状态，决定使用哪个平面模型进行距离计算
            plane_to_use_for_distance_calc = None
            if PS.PLANE_IS_LOCKED and PS.LOCKED_PLANE_MODEL is not None:
                plane_to_use_for_distance_calc = PS.LOCKED_PLANE_MODEL
            elif (
                not PS.IN_CALIBRATION_MODE and results.get("plane_model") is not None
            ):  # Use real-time if not locked and not calibrating
                plane_to_use_for_distance_calc = results["plane_model"]

            # 4. 计算最终眼部到选定平面的距离
            final_eye_to_table_dist_mm = None
            if plane_to_use_for_distance_calc is not None and (
                results.get("left_eye_center") is not None
                or results.get("right_eye_center") is not None
            ):
                eye_distances_mm = []
                for eye_center_key in ["left_eye_center", "right_eye_center"]:
                    eye_center_coords = results.get(eye_center_key)
                    if eye_center_coords is not None:
                        # Get eye depth using processed depth map (`results["depth_mm"]`)
                        eye_depth_val_mm = get_eye_depth_from_neighbor(
                            results["depth_mm"], eye_center_coords
                        )
                        if eye_depth_val_mm is not None:
                            # Convert eye RGB coord to depth map coord for depth_to_3d_point
                            # (This scaling is also done inside get_eye_depth_from_neighbor, could be optimized)
                            if PS.calib_rgb_w > 0:  # ensure not division by zero
                                eye_x_depth_coord = int(
                                    eye_center_coords[0] * (PS.DEPTH_W / PS.calib_rgb_w)
                                )
                                eye_y_depth_coord = int(
                                    eye_center_coords[1] * (PS.DEPTH_H / PS.calib_rgb_h)
                                )

                                eye_3d_coords_m = depth_to_3d_point(
                                    eye_depth_val_mm,
                                    eye_x_depth_coord,
                                    eye_y_depth_coord,
                                )
                                if eye_3d_coords_m is not None:
                                    dist_mm = calculate_point_to_plane_distance(
                                        eye_3d_coords_m, plane_to_use_for_distance_calc
                                    )
                                    if dist_mm is not None:
                                        eye_distances_mm.append(dist_mm)

                if eye_distances_mm:
                    final_eye_to_table_dist_mm = np.mean(eye_distances_mm)
                    # EMA smoothing for the final calculated distance
                    if PS.ema_eye_to_table_dist_mm is None:  # Initialize EMA
                        PS.ema_eye_to_table_dist_mm = final_eye_to_table_dist_mm
                    else:
                        PS.ema_eye_to_table_dist_mm = (
                            PS.EMA_ALPHA_EYE_TABLE_DIST * final_eye_to_table_dist_mm
                            + (1 - PS.EMA_ALPHA_EYE_TABLE_DIST) * PS.ema_eye_to_table_dist_mm
                        )

            # Add final distances to results for visualization
            results["eye_to_table_dist_final"] = final_eye_to_table_dist_mm
            results["ema_eye_to_table_dist_final"] = PS.ema_eye_to_table_dist_mm

            # 5. 可视化
            # Determine visual state for "locked" appearance (mask color, "LOCKED" text)
            # True if using a fixed plane (loaded or calibrated successfully) AND not currently in calibration mode.
            display_as_locked = PS.PLANE_IS_LOCKED and not PS.IN_CALIBRATION_MODE

            # Determine if we should draw the detailed plane equation overlay
            # Only for the final, confirmed LOCKED_PLANE_MODEL
            plane_model_for_overlay = None
            if display_as_locked and PS.LOCKED_PLANE_MODEL is not None:
                plane_model_for_overlay = PS.LOCKED_PLANE_MODEL

            vis_image = visualize_results(
                results,
                plane_is_locked_display_flag=display_as_locked,
                locked_plane_details_for_viz=plane_model_for_overlay,
            )

            if vis_image is not None:
                if PS.IN_CALIBRATION_MODE:  # Display calibration progress bar
                    # Progress bar should be based on `calibration_frame_count`
                    progress_percent = (
                        PS.calibration_frame_count / PS.CALIBRATION_FRAMES_TOTAL
                    )
                    bar_width_pixels = int(
                        vis_image.shape[1] * 0.4
                    )  # 40% of image width
                    filled_width = int(bar_width_pixels * progress_percent)

                    calib_text_pos_y = vis_image.shape[0] - 40
                    bar_pos_y = vis_image.shape[0] - 25

                    cv2.putText(
                        vis_image,
                        f"CALIBRATING DESKTOP... ({PS.calibration_frame_count}/{PS.CALIBRATION_FRAMES_TOTAL})",
                        (10, calib_text_pos_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    # Draw progress bar background
                    cv2.rectangle(
                        vis_image,
                        (10, bar_pos_y - 12),
                        (10 + bar_width_pixels, bar_pos_y + 2),
                        (80, 80, 80),
                        -1,
                    )
                    # Draw progress bar fill
                    cv2.rectangle(
                        vis_image,
                        (10, bar_pos_y - 12),
                        (10 + filled_width, bar_pos_y + 2),
                        (0, 255, 255),
                        -1,
                    )

                cv2.imshow("Unified Eye-Table Detection", vis_image)
                if video_writer is not None:
                    video_writer.write(vis_image)

            # 6. 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("程序退出 (q键按下)。")
                break
            elif key == ord("u"):  # 解锁并重新校准
                print("\n>>> 手动触发重新校准 <<<\n")
                PS.PLANE_IS_LOCKED = False
                PS.LOCKED_PLANE_MODEL = None
                PS.IN_CALIBRATION_MODE = True
                PS.calibration_frame_count = 0  # Reset counter
                PS.best_plane_score = -1  # Reset best score for new calibration
                PS.best_plane_model = None  # Reset best model
                PS.ema_eye_to_table_dist_mm = None  # Reset EMA
                print(f"开始自动校准阶段，将持续 {PS.CALIBRATION_FRAMES_TOTAL} 帧...")
            elif key == ord("s"):  # 切换深度模式
                new_depth_mode = 1 - PS.DEPTH_MODE
                use_16bit_new = new_depth_mode == 0
                print(f"\n请求切换到深度模式: {'16-bit' if use_16bit_new else '8-bit'}")

                # Update global DEPTH_MODE before reconfiguring camera
                PS.DEPTH_MODE = new_depth_mode

                # Reload calibration and correction model for the new mode
                if not load_calibration_data(use_16bit_new):
                    print(
                        f"错误: 切换到 {'16-bit' if use_16bit_new else '8-bit'} 模式的标定数据加载失败。保持当前模式。"
                    )
                    PS.DEPTH_MODE = 1 - new_depth_mode  # Revert
                    continue  # Skip reconfiguration if essential data fails to load
                load_depth_correction_model(use_16bit_new)  # This is optional

                # Reconfigure camera
                config_bytes_new = (
                    frame_config_encode()
                )  # Uses updated global DEPTH_MODE
                if config_bytes_new is not None and post_encode_config(
                    config_bytes_new, args.host, args.port
                ):
                    print(
                        f"相机已配置为 {'16-bit' if use_16bit_new else '8-bit'} 深度模式。"
                    )
                    time.sleep(1)  # 等待相机配置生效
                else:
                    print("错误: 切换相机深度模式失败。保持当前模式。")
                    PS.DEPTH_MODE = 1 - new_depth_mode  # Revert
                    load_calibration_data(PS.DEPTH_MODE == 0)  # Reload old calib
                    load_depth_correction_model(
                        PS.DEPTH_MODE == 0
                    )  # Reload old correction
                    continue

                # 切换模式后，重置校准状态
                print(
                    "深度模式已切换。建议重新校准桌面（如果之前已校准）。按 'u' 进行校准。"
                )
                PS.PLANE_IS_LOCKED = False
                PS.LOCKED_PLANE_MODEL = None
                PS.IN_CALIBRATION_MODE = True  # Enter calibration mode
                PS.calibration_frame_count = 0
                PS.best_plane_score = -1
                PS.best_plane_model = None
                PS.ema_eye_to_table_dist_mm = None  # Reset EMA

            # 7. 打印FPS
            frame_display_count += 1
            if frame_display_count >= 30:  # Print roughly every 30 frames
                current_time = time.time()
                fps = frame_display_count / (current_time - fps_time_start)
                status_text = f"FPS: {fps:.1f} | "
                if PS.IN_CALIBRATION_MODE:
                    status_text += f"校准中... ({PS.calibration_frame_count}/{PS.CALIBRATION_FRAMES_TOTAL})"
                elif PS.ema_eye_to_table_dist_mm is not None:
                    status_text += f"眼-桌距离: {PS.ema_eye_to_table_dist_mm:.1f} mm "
                    status_text += f"{'(LOCKED)' if PS.PLANE_IS_LOCKED and PS.LOCKED_PLANE_MODEL is not None else '(实时平面)'}"
                else:
                    status_text += "等待检测..."
                print(status_text)
                fps_time_start = current_time
                frame_display_count = 0

    except KeyboardInterrupt:
        print("\n用户中断程序")
    finally:
        if video_writer is not None:
            print("释放视频写入器...")
            video_writer.release()
        print("关闭窗口...")
        cv2.destroyAllWindows()
        er.initialize_mediapipe()
        face_mesh_model=er.face_mesh_model
        if face_mesh_model is not None and hasattr(face_mesh_model, "close"):
            print("关闭MediaPipe模型...")
            face_mesh_model.close()  # type: ignore
        print("程序结束。")


if __name__ == "__main__":
    main()