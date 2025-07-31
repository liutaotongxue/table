import numpy as np
import cv2
import os
import argparse
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description="Re-calibrate Stereo Camera using a subset of images")
    # Input arguments
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing the original captured image pairs.")
    parser.add_argument('--cols', type=int, required=True, help="Number of inner corners horizontally.")
    parser.add_argument('--rows', type=int, required=True, help="Number of inner corners vertically.")
    parser.add_argument('--size', type=float, required=True, help="Size of a chessboard square in mm.")
    parser.add_argument('--exclude', type=int, nargs='+', default=[], help="List of pair numbers (integers) to exclude from re-calibration.")
    parser.add_argument('--ir_bit', type=int, default=1, choices=[0, 1], help="IR sensor bit depth used during capture (0: 16-bit, 1: 8-bit). Default is 8-bit.")
    parser.add_argument('--out_file', type=str, default='stereo_recalibrated_data.npz', help="Output file name for the re-calibration results.")
    # Optional flags similar to original calibration
    parser.add_argument('--no_initial_calib', action='store_true', help="Skip initial single camera calibration guess.")
    parser.add_argument('--fix_intrinsic', action='store_true', help="Fix intrinsic parameters during stereo calibration (use initial guess).")

    args = parser.parse_args()

    # --- 1. Initialization and Parameter Setup ---
    CHECKERBOARD_SIZE = (args.cols, args.rows)
    SQUARE_SIZE = args.size
    EXCLUDE_LIST = set(args.exclude) # Use a set for efficient lookup

    if not os.path.isdir(args.img_dir):
        print(f"错误：图像目录未找到: {args.img_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"将从目录 '{args.img_dir}' 加载图像")
    print(f"将排除以下图像对编号: {sorted(list(EXCLUDE_LIST))}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    object_points_3D = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    object_points_3D[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points_3D *= SQUARE_SIZE

    # Lists for storing points from SELECTED pairs
    object_points_list = []
    image_points_rgb = []
    image_points_ir = []
    loaded_pair_count = 0
    rgb_shape_wh = None # Will be determined from loaded images
    ir_shape_wh = (320, 240) # Assume typical IR size

    # --- 2. Load Images and Detect Corners (Filtering) ---
    print("\n正在加载图像并检测角点...")
    rgb_images = sorted(glob.glob(os.path.join(args.img_dir, 'pair_*_rgb.png')))

    for rgb_path in rgb_images:
        try:
            pair_num = int(os.path.basename(rgb_path).split('_')[1])
        except (IndexError, ValueError):
             print(f"警告：无法从文件名解析配对编号: {rgb_path}", file=sys.stderr)
             continue

        # Skip if this pair number is in the exclude list
        if pair_num in EXCLUDE_LIST:
            # print(f"  跳过配对 {pair_num} (在排除列表中)")
            continue

        ir_path = os.path.join(args.img_dir, f"pair_{pair_num:02d}_ir.png")
        if not os.path.exists(ir_path):
            print(f"警告：找不到配对 {pair_num} 的对应 IR 图像: {ir_path}", file=sys.stderr)
            continue

        img_rgb_gray = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        # Load IR based on bit depth assumption (or check metadata if available)
        # We assume the saved file is grayscale 8-bit as saved by the previous script
        img_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if img_rgb_gray is None or img_ir_gray is None:
            print(f"警告：无法加载图像对 {pair_num}", file=sys.stderr)
            continue

        # Store image shapes
        if rgb_shape_wh is None: rgb_shape_wh = img_rgb_gray.shape[::-1]
        # ir_shape_wh is assumed fixed, but could check img_ir_gray.shape[::-1]

        # Find corners in both
        ret_rgb, corners_rgb = cv2.findChessboardCorners(img_rgb_gray, CHECKERBOARD_SIZE, None)
        ret_ir, corners_ir = cv2.findChessboardCorners(img_ir_gray, CHECKERBOARD_SIZE, None)

        if ret_rgb and ret_ir:
            loaded_pair_count += 1
            # print(f"  成功处理配对 {pair_num}")
            object_points_list.append(object_points_3D)
            corners_subpix_rgb = cv2.cornerSubPix(img_rgb_gray, corners_rgb, (11, 11), (-1, -1), criteria)
            corners_subpix_ir = cv2.cornerSubPix(img_ir_gray, corners_ir, (11, 11), (-1, -1), criteria)
            image_points_rgb.append(corners_subpix_rgb)
            image_points_ir.append(corners_subpix_ir)
        # else:
            # print(f"  配对 {pair_num} 未在两个图像中都找到角点，已跳过。")


    print(f"\n成功加载并处理了 {loaded_pair_count} 对有效图像（已排除指定项）。")

    if loaded_pair_count < 10: # Need a minimum number for good calibration
        print(f"错误：剩余的有效图像对数量 ({loaded_pair_count}) 太少，无法进行可靠的标定。", file=sys.stderr)
        sys.exit(1)
    if rgb_shape_wh is None:
         print("错误：未能从加载的图像中确定 RGB 图像尺寸。", file=sys.stderr)
         sys.exit(1)

    # --- 3. Perform Re-Calibration ---

    # --- 3a. Optional: Initial Single Camera Calibration ---
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None
    if not args.no_initial_calib:
        print("正在进行初始单目相机标定估算 (使用筛选后的数据)...")
        try:
            print(f"  使用 RGB 尺寸: {rgb_shape_wh}")
            ret1, cameraMatrix1, distCoeffs1, _, _ = cv2.calibrateCamera(object_points_list, image_points_rgb, rgb_shape_wh, None, None)
            print(f"  使用 IR 尺寸: {ir_shape_wh}")
            ret2, cameraMatrix2, distCoeffs2, _, _ = cv2.calibrateCamera(object_points_list, image_points_ir, ir_shape_wh, None, None)
            if ret1 and ret2:
                print("  初始单目标定完成。")
            else:
                print("  初始单目标定失败，将不使用初始猜测。")
                cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None
        except Exception as e_calib:
            print(f"  初始单目标定过程中发生错误: {e_calib}", file=sys.stderr)
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2 = None, None, None, None

    # --- 3b. Stereo Calibration ---
    print("\n正在进行立体标定 (使用筛选后的数据)...")

    flags = 0
    if args.fix_intrinsic:
        if cameraMatrix1 is not None and distCoeffs1 is not None and cameraMatrix2 is not None and distCoeffs2 is not None:
             print("  将固定内参进行立体标定。")
             flags |= cv2.CALIB_FIX_INTRINSIC
        else:
             print("  警告：请求固定内参，但初始估算失败或跳过。将不固定内参。", file=sys.stderr)

    if cameraMatrix1 is not None and distCoeffs1 is not None: flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    if cameraMatrix2 is not None and distCoeffs2 is not None: flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    try:
        retval, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            object_points_list, image_points_rgb, image_points_ir,
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
            rgb_shape_wh, criteria=criteria_stereo, flags=flags
        )
    except Exception as e_stereo:
        print(f"\n错误：在调用 cv2.stereoCalibrate 时出错: {e_stereo}", file=sys.stderr)
        sys.exit(1)

    # --- 4. Results ---
    if retval > 0:
        print("\n重新标定成功!")
        print(f"  新的 RMS 重投影误差: {retval} 像素")
        # Compare with previous error if available?

        print("\n新 RGB 相机内参矩阵 (M1):"); print(M1)
        print("\n新 RGB 畸变系数 (d1):"); print(d1)
        print("\n新 IR 相机内参矩阵 (M2):"); print(M2)
        print("\n新 IR 畸变系数 (d2):"); print(d2)
        print("\n新 旋转矩阵 (R - 从 IR 到 RGB):"); print(R)
        print("\n新 平移向量 (T - 从 IR 到 RGB, 单位: mm):"); print(T)

        # Save results
        try:
            np.savez(args.out_file, mtx1=M1, dist1=d1, mtx2=M2, dist2=d2, R=R, T=T, E=E, F=F, error=retval, rgb_size=rgb_shape_wh, ir_size=ir_shape_wh)
            print(f"\n重新标定结果已保存到: {args.out_file}")
        except Exception as save_err:
             print(f"\n错误：保存重新标定数据失败: {save_err}", file=sys.stderr)
    else:
        print("\n重新标定失败。")

    print("\n--- 重新标定流程结束 ---")

if __name__ == "__main__":
    main()