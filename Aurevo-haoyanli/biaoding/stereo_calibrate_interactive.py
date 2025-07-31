# --- START OF FILE stereo_calibrate_interactive.py ---

import requests
import struct
import numpy as np
import cv2
import time
import os
import argparse
import sys
import glob # Import glob for file searching

# --- Configuration ---
DEFAULT_HOST = '192.168.233.1'
DEFAULT_PORT = 80

# --- Helper Functions (Paste the FULL definitions from the previous valid script here) ---
def frame_config_decode(frame_config):
    try:
        if len(frame_config) != 12: return None
        return struct.unpack("<BBBBBBBBi", frame_config)
    except struct.error: return None

def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
    params = [trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time]
    if not all(isinstance(p, int) for p in params): raise ValueError("Config params must be integers.")
    return struct.pack("<BBBBBBBBi", *params)

def frame_payload_decode(frame_data: bytes, with_config: tuple):
    # This function needs the full implementation from the previous script that correctly
    # decodes and returns ir_frame (numpy array) and rgb_frame (numpy array BGR).
    # Ensure it handles potential errors gracefully.
    if with_config is None: return (None, None, None, None)
    try:
        if len(frame_data) < 8: return (None, None, None, None)
        deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
        frame_payload = frame_data[8:]
        total_payload_len = len(frame_payload)
        current_offset = 0
        # Depth (simplified extraction - not used but parsed)
        expected_depth_size = 0
        if 0 <= with_config[1] <= 1: expected_depth_size = (320*240*2) >> with_config[1]
        if current_offset + expected_depth_size > deep_data_size: expected_depth_size = max(0, deep_data_size - current_offset)
        # deepth_img = frame_payload[current_offset : current_offset + expected_depth_size] if expected_depth_size > 0 and expected_depth_size <= total_payload_len - current_offset else None
        current_offset += expected_depth_size # Advance offset even if not storing
        # IR (simplified extraction)
        expected_ir_size = 0
        ir_bytes = None # Initialize ir_bytes
        if 0 <= with_config[3] <= 1: expected_ir_size = (320*240*2) >> with_config[3]
        if current_offset + expected_ir_size > deep_data_size: expected_ir_size = max(0, deep_data_size - current_offset)
        if expected_ir_size > 0 and expected_ir_size <= total_payload_len - current_offset:
            ir_bytes = frame_payload[current_offset : current_offset + expected_ir_size]
        current_offset += expected_ir_size
        # Status (skipped)
        # status_img = None
        current_offset = deep_data_size # Go to end of deep data section
        # RGB
        rgb_payload_offset = deep_data_size
        actual_rgb_available = total_payload_len - rgb_payload_offset
        rgb_img_bytes = None
        if actual_rgb_available >= 0 and rgb_data_size >= 0:
             if actual_rgb_available != rgb_data_size and rgb_data_size > 0: pass
             use_size = min(actual_rgb_available, rgb_data_size) if rgb_data_size > 0 else actual_rgb_available
             rgb_img_bytes = frame_payload[rgb_payload_offset : rgb_payload_offset + use_size] if use_size > 0 else None
        # Decode RGB (JPG) -> Returns BGR Numpy Array or None
        rgb_final = None
        if (rgb_img_bytes is not None) and (with_config[6] == 1):
            try:
                jpeg = cv2.imdecode(np.frombuffer(rgb_img_bytes, 'uint8'), cv2.IMREAD_COLOR)
                if jpeg is not None: rgb_final = jpeg
            except Exception: pass
        # Convert IR bytes to numpy array
        ir_final = None
        if ir_bytes is not None and 0 <= with_config[3] <= 1:
             try:
                 ir_dtype = np.uint16 if with_config[3] == 0 else np.uint8
                 expected_bytes = 320 * 240 * (2 if ir_dtype == np.uint16 else 1)
                 if len(ir_bytes) == expected_bytes:
                     ir_final = np.frombuffer(ir_bytes, dtype=ir_dtype).reshape((240, 320))
             except Exception: pass
        return (None, ir_final, None, rgb_final)
    except Exception: return (None, None, None, None)

def post_encode_config(config, host=DEFAULT_HOST, port=DEFAULT_PORT):
    try:
        r = requests.post(f'http://{host}:{port}/set_cfg', data=config, timeout=5)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"错误：发送配置失败: {e}", file=sys.stderr); return False

def get_frame_from_http(host=DEFAULT_HOST, port=DEFAULT_PORT):
    try:
        r = requests.get(f'http://{host}:{port}/getdeep', timeout=2)
        r.raise_for_status()
        if len(r.content) < 16: return None
        return r.content
    except requests.exceptions.RequestException: return None
# --- End of Helper Functions ---


# --- Stereo Calibration Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Interactive Stereo Calibration for MaixSense RGB and IR Cameras")
    # Shared parameters
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help="Camera IP address.")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help="Camera port.")
    parser.add_argument('--cols', type=int, required=True, help="Number of inner corners horizontally.")
    parser.add_argument('--rows', type=int, required=True, help="Number of inner corners vertically.")
    parser.add_argument('--size', type=float, required=True, help="Size of a chessboard square in mm.")
    parser.add_argument('--num_pairs', type=int, default=25, help="Total number of valid image pairs required.")
    parser.add_argument('--outdir', type=str, default='stereo_calibration_output', help="Directory to save captured images and results.")
    # IR specific parameter
    parser.add_argument('--ir_bit', type=int, default=1, choices=[0, 1], help="IR sensor bit depth (0: 16-bit, 1: 8-bit). Default is 8-bit.")
    # Optional flags for calibration process
    parser.add_argument('--no_initial_calib', action='store_true', help="Skip initial single camera calibration guess.")
    parser.add_argument('--fix_intrinsic', action='store_true', help="Fix intrinsic parameters during stereo calibration (use initial guess).")

    args = parser.parse_args()

    HOST = args.host
    PORT = args.port
    CHECKERBOARD_SIZE = (args.cols, args.rows)
    SQUARE_SIZE = args.size # in mm
    IR_BIT_MODE = args.ir_bit # 0 for 16bit, 1 for 8bit

    # --- 1. Initialization ---
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        print(f"已创建输出目录: {args.outdir}")
    else:
        print(f"输出目录已存在: {args.outdir}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5) # Criteria for stereo calibration

    # Prepare object points (common for both cameras)
    object_points_3D = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    object_points_3D[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points_3D *= SQUARE_SIZE

    # Lists to store points from valid PAIRS
    object_points_list = [] # 3D points for valid pairs
    image_points_rgb = []   # 2D points from RGB image for valid pairs
    image_points_ir = []    # 2D points from IR image for valid pairs

    # ===== RESUME LOGIC START =====
    captured_pair_count = 0
    existing_files = glob.glob(os.path.join(args.outdir, 'pair_*_rgb.png'))
    if existing_files:
        max_num = 0
        for f in existing_files:
            try:
                num = int(os.path.basename(f).split('_')[1])
                # Also check if corresponding IR exists
                ir_f = os.path.join(args.outdir, f"pair_{num:02d}_ir.png")
                if os.path.exists(ir_f):
                    max_num = max(max_num, num)
                else:
                    print(f"警告: 找到 RGB 图像 {f} 但缺少对应的 IR 图像，可能需要手动清理。")
            except (IndexError, ValueError):
                print(f"警告: 无法从文件名解析配对编号: {f}")
        if max_num > 0:
            captured_pair_count = max_num
            print(f"检测到已存在的图像对，将从第 {captured_pair_count + 1} 对开始捕获。")
        else:
            print("未检测到有效的已存在图像对，将从头开始捕获。")
    else:
        print("输出目录为空，将从头开始捕获。")
    # ===== RESUME LOGIC END =====

    # --- 2. Configure Camera for Simultaneous Output ---
    print("正在配置相机以同时输出 RGB 和 IR ...")
    config = frame_config_encode(rgb_mode=1, ir_mode=IR_BIT_MODE, deep_mode=255, rgb_res=0)
    if not post_encode_config(config, host=HOST, port=PORT):
        print("配置相机失败。正在退出。", file=sys.stderr)
        sys.exit(1)
    time.sleep(0.5) # Give camera time

    # --- 3. Capture Image Pairs ---
    print("\n--- 开始捕获图像对 ---")
    print(f"目标: {args.num_pairs} 对有效图像 | 当前已有: {captured_pair_count}")
    print("操作说明:")
    print(" - 将棋盘格放置在两个摄像头的共同视野中。")
    print(" - 按 'c' 键尝试捕获当前帧对。")
    print(" - (找到角点后) 按 'c' 或 Enter 接受并保存, 按 'd' 删除并重试当前对。")
    print(" - 按 'q' 键结束捕获并开始标定。")
    print("-" * 30)

    cv2.namedWindow('RGB Feed')
    cv2.namedWindow('IR Feed')
    rgb_frame_shape_wh = None
    ir_frame_shape_wh = (320, 240) # Assume typical IR size

    while captured_pair_count < args.num_pairs:
        current_pair_number = captured_pair_count + 1 # The number we are trying to capture

        # Display current status (before getting frame)
        status_text = f"Needed: {args.num_pairs} | Current: {captured_pair_count} | Target: {current_pair_number}"
        # Placeholder frames while waiting
        display_rgb_status = np.zeros((480, 640, 3), dtype=np.uint8)
        display_ir_status = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(display_rgb_status, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(display_ir_status, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow('RGB Feed', display_rgb_status)
        cv2.imshow('IR Feed', display_ir_status)

        raw_frame = get_frame_from_http(host=HOST, port=PORT)
        if raw_frame is None:
            key = cv2.waitKey(50) & 0xFF # Wait a bit longer if no frame
            if key == ord('q'): break
            continue

        if len(raw_frame) < 28: continue
        config_tuple = frame_config_decode(raw_frame[16:16+12])
        if config_tuple is None: continue

        _, ir_frame, _, rgb_frame = frame_payload_decode(raw_frame[16+12:], config_tuple)
        valid_rgb = rgb_frame is not None
        valid_ir = ir_frame is not None

        # Prepare display frames
        display_rgb = np.zeros((480, 640, 3), dtype=np.uint8) if not valid_rgb else rgb_frame.copy()
        display_ir_bgr = np.zeros((240, 320, 3), dtype=np.uint8)

        if valid_ir:
             if ir_frame_shape_wh is None: ir_frame_shape_wh = ir_frame.shape[::-1][1:]
             if ir_frame.dtype == np.uint16:
                 max_val = np.percentile(ir_frame, 99.5) if np.max(ir_frame) > 0 else 1
                 display_ir_norm = np.clip((ir_frame / max_val) * 255.0, 0, 255).astype(np.uint8)
             else: display_ir_norm = ir_frame
             display_ir_bgr = cv2.cvtColor(display_ir_norm, cv2.COLOR_GRAY2BGR)
        if valid_rgb and rgb_frame_shape_wh is None:
            rgb_frame_shape_wh = (rgb_frame.shape[1], rgb_frame.shape[0])

        # Display progress
        cv2.putText(display_rgb, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_ir_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('RGB Feed', display_rgb)
        cv2.imshow('IR Feed', display_ir_bgr)
        key = cv2.waitKey(1) & 0xFF # Check for key press immediately

        if key == ord('q'):
            print("\n用户退出捕获。")
            break
        elif key == ord('c'): # Try to capture
            if not valid_rgb or not valid_ir:
                print(f"  尝试捕获第 {current_pair_number} 对失败：RGB 或 IR 图像无效。")
                time.sleep(0.5) # Pause briefly on failure
                continue

            print(f"尝试检测第 {current_pair_number} 对图像的角点...")

            # Convert to grayscale
            gray_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
            if ir_frame.dtype == np.uint16:
                 max_val_ir = np.max(ir_frame) if np.max(ir_frame) > 0 else 1
                 gray_ir = np.clip((ir_frame / max_val_ir) * 255.0, 0, 255).astype(np.uint8)
            else:
                 gray_ir = ir_frame

            # Find corners in BOTH images
            ret_rgb, corners_rgb = cv2.findChessboardCorners(gray_rgb, CHECKERBOARD_SIZE, None)
            ret_ir, corners_ir = cv2.findChessboardCorners(gray_ir, CHECKERBOARD_SIZE, None)

            if ret_rgb and ret_ir:
                print("  在 RGB 和 IR 中均找到角点!")

                # Refine corners
                corners_subpix_rgb = cv2.cornerSubPix(gray_rgb, corners_rgb, (11, 11), (-1, -1), criteria)
                corners_subpix_ir = cv2.cornerSubPix(gray_ir, corners_ir, (11, 11), (-1, -1), criteria)

                # Draw corners for feedback
                display_rgb_drawn = display_rgb.copy()
                display_ir_drawn = display_ir_bgr.copy()
                cv2.drawChessboardCorners(display_rgb_drawn, CHECKERBOARD_SIZE, corners_subpix_rgb, ret_rgb)
                cv2.drawChessboardCorners(display_ir_drawn, CHECKERBOARD_SIZE, corners_subpix_ir, ret_ir)

                # Add confirmation text
                text_confirm = "Accept: [c] or [Enter] | Delete: [d]"
                cv2.putText(display_rgb_drawn, text_confirm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_ir_drawn, text_confirm, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.imshow('RGB Feed', display_rgb_drawn)
                cv2.imshow('IR Feed', display_ir_drawn)

                # ===== INTERACTIVE DELETION LOGIC START =====
                while True:
                    key_confirm = cv2.waitKey(0) & 0xFF # Wait indefinitely for decision
                    if key_confirm == ord('d'):
                        print(f"  用户删除第 {current_pair_number} 对图像。请重新放置棋盘格并按 'c'。")
                        # No need to delete files as they haven't been saved yet
                        # No need to decrement count as it wasn't incremented
                        # Break inner loop to retry capture for the same number
                        break # Exit the waitKey loop, outer loop continues
                    elif key_confirm == ord('c') or key_confirm == 13: # 13 is Enter key
                        print(f"  用户接受第 {current_pair_number} 对图像。")
                        # Add points to lists
                        object_points_list.append(object_points_3D)
                        image_points_rgb.append(corners_subpix_rgb)
                        image_points_ir.append(corners_subpix_ir)

                        # Save the PAIR of grayscale images
                        rgb_filename = os.path.join(args.outdir, f"pair_{current_pair_number:02d}_rgb.png")
                        ir_filename = os.path.join(args.outdir, f"pair_{current_pair_number:02d}_ir.png")
                        save_ok_rgb = cv2.imwrite(rgb_filename, gray_rgb)
                        save_ok_ir = cv2.imwrite(ir_filename, gray_ir)

                        if save_ok_rgb and save_ok_ir:
                             print(f"  已保存图像对: {os.path.basename(rgb_filename)}, {os.path.basename(ir_filename)}")
                             captured_pair_count += 1 # Increment count ONLY after saving successfully
                        else:
                             print(f"  警告：保存图像对 {current_pair_number} 失败！此对将不被使用。", file=sys.stderr)
                             # Remove points if saving failed? Or just don't increment count?
                             # Let's not increment count and the user will have to capture one more
                             object_points_list.pop()
                             image_points_rgb.pop()
                             image_points_ir.pop()

                        break # Exit the waitKey loop, outer loop proceeds to next number
                    elif key_confirm == ord('q'):
                         print("\n用户退出捕获。")
                         captured_pair_count = args.num_pairs + 1 # Force outer loop to exit
                         break # Exit the waitKey loop
                    else:
                        print("  无效按键。请按 'c'/'Enter' 接受或 'd' 删除。")
                # ===== INTERACTIVE DELETION LOGIC END =====

            else: # Corners not found in both
                if not ret_rgb: print("  未在 RGB 图像中找到角点。")
                if not ret_ir: print("  未在 IR 图像中找到角点。")
                print("  此图像对无效，请调整棋盘格后重试。")
                time.sleep(0.5) # Pause briefly

    cv2.destroyAllWindows()

    # --- 4. Perform Calibration ---
    # (Calibration logic remains the same as the previous "complete code" version
    #  that includes saving rvecs/tvecs)
    if captured_pair_count < args.num_pairs:
        print(f"\n标定取消或未完成。仅有 {len(object_points_list)} 对有效图像 (目标 {args.num_pairs})。") # Use list length as final count
        sys.exit(0)

    # Use the length of the collected points list as the definitive number of pairs
    final_pair_count = len(object_points_list)
    print(f"\n使用 {final_pair_count} 对有效图像开始标定...")

    if final_pair_count < 10: # Need a minimum number
        print(f"错误：有效图像对数量 ({final_pair_count}) 太少，无法进行可靠的标定。", file=sys.stderr)
        sys.exit(1)


    # --- 4a. Optional: Initial Single Camera Calibration ---
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None
    rvecs1, tvecs1 = None, None # Initialize variables to store extrinsics for camera 1 (RGB)

    if not args.no_initial_calib:
        print("正在进行初始单目相机标定估算...")
        try:
            if rgb_frame_shape_wh is None: # Try to determine from collected points if needed
                 if image_points_rgb: # Get shape from the first detected corners data structure (less reliable)
                     # Need to load an image again, or have saved the shape earlier
                     # Let's try reloading the first saved image
                     first_rgb_path = os.path.join(args.outdir, 'pair_01_rgb.png') # Assumes pair 01 exists and is valid
                     if os.path.exists(first_rgb_path):
                         first_rgb_img = cv2.imread(first_rgb_path, cv2.IMREAD_GRAYSCALE)
                         if first_rgb_img is not None: rgb_frame_shape_wh = first_rgb_img.shape[::-1]
                 if rgb_frame_shape_wh is None: raise ValueError("无法确定RGB图像尺寸")

            if ir_frame_shape_wh is None: # Try to determine from files
                 first_ir_path = os.path.join(args.outdir, 'pair_01_ir.png')
                 if os.path.exists(first_ir_path):
                     first_ir_img = cv2.imread(first_ir_path, cv2.IMREAD_GRAYSCALE)
                     if first_ir_img is not None: ir_frame_shape_wh = first_ir_img.shape[::-1]
                 if ir_frame_shape_wh is None: ir_frame_shape_wh=(320,240) # Fallback to default

            print(f"  使用 RGB 尺寸: {rgb_frame_shape_wh}")
            ret1, cameraMatrix1, distCoeffs1, rvecs1_list, tvecs1_list = cv2.calibrateCamera(
                object_points_list, image_points_rgb, rgb_frame_shape_wh, None, None
            )
            if ret1 and rvecs1_list is not None and tvecs1_list is not None:
                 rvecs1 = np.array(rvecs1_list); tvecs1 = np.array(tvecs1_list)
            else: rvecs1, tvecs1 = None, None

            print(f"  使用 IR 尺寸: {ir_frame_shape_wh}")
            ret2, cameraMatrix2, distCoeffs2, _, _ = cv2.calibrateCamera(
                object_points_list, image_points_ir, ir_frame_shape_wh, None, None
            )
            if ret1 and ret2:
                print("  初始单目标定完成。")
                if rvecs1 is None or tvecs1 is None: print("  警告：RGB 单目标定成功但未能获取外参(rvecs/tvecs)。")
            else:
                print("  初始单目标定失败，将不使用初始猜测。")
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None; rvecs1, tvecs1 = None, None
        except Exception as e_calib:
            print(f"  初始单目标定过程中发生错误: {e_calib}", file=sys.stderr)
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None; rvecs1, tvecs1 = None, None

    # --- 4b. Stereo Calibration ---
    print("\n正在进行立体标定...")
    if rgb_frame_shape_wh is None: print("错误：无法执行立体标定，RGB图像尺寸未知。", file=sys.stderr); sys.exit(1)

    flags = 0
    if args.fix_intrinsic:
        if cameraMatrix1 is not None and distCoeffs1 is not None and cameraMatrix2 is not None and distCoeffs2 is not None:
             print("  将固定内参进行立体标定。"); flags |= cv2.CALIB_FIX_INTRINSIC
        else: print("  警告：请求固定内参，但初始估算失败或跳过。将不固定内参。", file=sys.stderr)
    if cameraMatrix1 is not None and distCoeffs1 is not None: flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    if cameraMatrix2 is not None and distCoeffs2 is not None: flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    try:
        retval, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            object_points_list, image_points_rgb, image_points_ir,
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
            rgb_frame_shape_wh, criteria=criteria_stereo, flags=flags
        )
    except Exception as e_stereo:
        print(f"\n错误：在调用 cv2.stereoCalibrate 时出错: {e_stereo}", file=sys.stderr); sys.exit(1)

    # --- 5. Results ---
    if retval > 0:
        print("\n立体标定成功!")
        print(f"  RMS 重投影误差: {retval} 像素")
        if retval > 1.0: print("  警告：立体标定重投影误差较高 (> 1.0)。结果可能不准确。")

        print("\nRGB 相机内参矩阵 (M1):"); print(M1)
        print("\nRGB 畸变系数 (d1):"); print(d1)
        print("\nIR 相机内参矩阵 (M2):"); print(M2)
        print("\nIR 畸变系数 (d2):"); print(d2)
        print("\n旋转矩阵 (R - 从 IR 到 RGB):"); print(R)
        print("\n平移向量 (T - 从 IR 到 RGB, 单位: mm):"); print(T)

        output_file = os.path.join(args.outdir, "stereo_calibration_data.npz")
        try:
            save_dict = {
                'mtx1': M1, 'dist1': d1, 'mtx2': M2, 'dist2': d2,
                'R': R, 'T': T, 'E': E, 'F': F, 'error': retval,
                'rgb_size': rgb_frame_shape_wh, 'ir_size': ir_frame_shape_wh
            }
            if rvecs1 is not None and isinstance(rvecs1, np.ndarray) and \
               tvecs1 is not None and isinstance(tvecs1, np.ndarray):
                save_dict['rvecs'] = rvecs1 # Save RGB extrinsics
                save_dict['tvecs'] = tvecs1
                print("  将保存初始单目标定外参 (rvecs, tvecs for RGB)。")
            else: print("  警告：未计算或无法保存初始单目标定外参。")
            np.savez(output_file, **save_dict)
            print(f"\n立体标定结果已保存到: {output_file}")
        except Exception as save_err: print(f"\n错误：保存立体标定数据失败: {save_err}", file=sys.stderr)
    else: print("\n立体标定失败。")

    print("\n--- 立体标定流程结束 ---")

if __name__ == "__main__":
    main()

# --- END OF FILE stereo_calibrate_interactive.py ---