import numpy as np
import cv2
import os
import argparse
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description="Analyze Stereo Calibration Reprojection Errors per Image Pair")
    parser.add_argument('--calib_file', type=str, required=True, help="Path to the stereo calibration result file (.npz).")
    parser.add_argument('--img_dir', type=str, required=True, help="Directory containing the captured image pairs (e.g., pair_XX_rgb.png, pair_XX_ir.png).")
    parser.add_argument('--cols', type=int, required=True, help="Number of inner corners horizontally.")
    parser.add_argument('--rows', type=int, required=True, help="Number of inner corners vertically.")
    parser.add_argument('--size', type=float, required=True, help="Size of a chessboard square in mm.")
    args = parser.parse_args()

    # --- 1. Load Calibration Data ---
    if not os.path.exists(args.calib_file):
        print(f"错误：标定文件未找到: {args.calib_file}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.img_dir):
        print(f"错误：图像目录未找到: {args.img_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        calib_data = np.load(args.calib_file)
        M1 = calib_data['mtx1']
        d1 = calib_data['dist1']
        M2 = calib_data['mtx2']
        d2 = calib_data['dist2']
        R = calib_data['R']
        T = calib_data['T']
        # 尝试加载可能保存的 rvecs/tvecs (假设它们是相对于相机1/RGB的)
        # 注意：stereoCalibrate 本身不直接返回 rvecs/tvecs 列表，这些可能来自初始单目标定
        # 如果 .npz 文件中没有，我们需要重新计算它们，但这会复杂化分析脚本
        # 暂时假设它们存在，如果不存在则需要修改原标定脚本保存或此处重新计算
        if 'rvecs' in calib_data and 'tvecs' in calib_data:
            rvecs = calib_data['rvecs']
            tvecs = calib_data['tvecs']
            print("已加载 rvecs 和 tvecs 用于误差计算。")
            use_extrinsics = True
        else:
            print("警告：未在 .npz 文件中找到 rvecs/tvecs。将无法计算精确的每对图像重投影误差。", file=sys.stderr)
            print("将仅尝试重新检测角点以检查图像质量。")
            use_extrinsics = False

    except KeyError as e:
        print(f"错误：标定文件 {args.calib_file} 缺少必需的键: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：加载标定文件时出错: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 2. Prepare Object Points and Find Image Pairs ---
    CHECKERBOARD_SIZE = (args.cols, args.rows)
    SQUARE_SIZE = args.size
    object_points_3D = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    object_points_3D[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    object_points_3D *= SQUARE_SIZE

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find image pairs based on naming convention
    rgb_images = sorted(glob.glob(os.path.join(args.img_dir, 'pair_*_rgb.png')))
    ir_images = sorted(glob.glob(os.path.join(args.img_dir, 'pair_*_ir.png')))

    if len(rgb_images) != len(ir_images):
        print("警告：RGB 和 IR 图像文件数量不匹配！", file=sys.stderr)
        # Attempt to match based on prefix
        # This part could be more robust
    if len(rgb_images) == 0:
         print(f"错误：在目录 {args.img_dir} 中未找到 'pair_*_rgb.png' 图像。", file=sys.stderr)
         sys.exit(1)
    if use_extrinsics and len(rvecs) != len(rgb_images):
         print(f"警告：加载的 rvecs/tvecs 数量 ({len(rvecs)}) 与找到的图像对数量 ({len(rgb_images)}) 不匹配！无法计算误差。", file=sys.stderr)
         use_extrinsics = False # Disable error calculation if counts don't match

    print(f"找到 {len(rgb_images)} 对图像。")

    # --- 3. Analyze Each Pair ---
    results = [] # Store (pair_index, error_rgb, error_ir, combined_error, found_rgb, found_ir)
    object_points_list = []
    image_points_rgb = []
    image_points_ir = []
    valid_pair_indices = [] # Store original indices of pairs where corners were found

    for i in range(len(rgb_images)):
        rgb_path = rgb_images[i]
        pair_num = int(os.path.basename(rgb_path).split('_')[1]) # Extract pair number
        ir_path = os.path.join(args.img_dir, f"pair_{pair_num:02d}_ir.png")

        if not os.path.exists(ir_path):
            print(f"警告：找不到对应的 IR 图像: {ir_path}", file=sys.stderr)
            continue

        img_rgb_gray = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        img_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE) # Assume saved IR is already suitable gray

        if img_rgb_gray is None or img_ir_gray is None:
            print(f"警告：无法加载图像对 {pair_num}", file=sys.stderr)
            continue

        # Find corners
        ret_rgb, corners_rgb = cv2.findChessboardCorners(img_rgb_gray, CHECKERBOARD_SIZE, None)
        ret_ir, corners_ir = cv2.findChessboardCorners(img_ir_gray, CHECKERBOARD_SIZE, None)

        error_rgb = float('inf')
        error_ir = float('inf')
        combined_error = float('inf')

        if ret_rgb and ret_ir:
            # Append points for potential re-calibration check later
            object_points_list.append(object_points_3D)
            corners_subpix_rgb = cv2.cornerSubPix(img_rgb_gray, corners_rgb, (11, 11), (-1, -1), criteria)
            corners_subpix_ir = cv2.cornerSubPix(img_ir_gray, corners_ir, (11, 11), (-1, -1), criteria)
            image_points_rgb.append(corners_subpix_rgb)
            image_points_ir.append(corners_subpix_ir)
            valid_pair_indices.append(pair_num) # Store original index


            if use_extrinsics and i < len(rvecs): # Check index bounds
                # Calculate reprojection error for RGB
                try:
                    projected_points_rgb, _ = cv2.projectPoints(object_points_3D, rvecs[i], tvecs[i], M1, d1)
                    error_rgb = cv2.norm(corners_subpix_rgb, projected_points_rgb, cv2.NORM_L2) / len(projected_points_rgb)
                except Exception as e_proj_rgb:
                    print(f"警告：计算 RGB 投影误差时出错 (Pair {pair_num}): {e_proj_rgb}", file=sys.stderr)
                    error_rgb = float('inf')


                # Calculate reprojection error for IR
                # Need extrinsics for IR camera (relative to world)
                # Transform extrinsics from Cam1 (RGB) to Cam2 (IR)
                # R_ir_world = R_rgb_world * R_ir_rgb.T
                # T_ir_world = -R_ir_world * T_ir_rgb
                # Note: R is from IR to RGB, T is from IR to RGB origin in RGB coords
                # So, R_rgb_ir = R, T_rgb_ir = T
                # World origin in IR coords = -R.T @ T
                # World rotation wrt IR = R.T
                # -> Object in IR coords: X_ir = R.T @ (X_world - T)
                # -> Extrinsics for IR: rvec_ir, tvec_ir where T_ir = -R_ir.T @ T_world ; R_ir = R_world @ R_ir.T ??? This is complex.

                # Alternative: Use stereoRectify? No, that's for rectification.

                # Simplest approximation: Assume R/T represent transformation FROM IR TO RGB coords.
                # Use projectPoints with M2, d2 and find extrinsics relative to IR.
                # How to find rvecs/tvecs for IR camera for view i?
                # Maybe use cv2.solvePnP with object_points_3D and image_points_ir[index_in_list] using M2, d2?
                # Let's try solvePnP to get an idea of IR error independently for now.
                try:
                    # Find pose of the board relative to the IR camera for this view
                    valid_idx = valid_pair_indices.index(pair_num) # Find index in the valid lists
                    ret_pnp, rvec_ir_est, tvec_ir_est = cv2.solvePnP(object_points_3D, image_points_ir[valid_idx], M2, d2)
                    if ret_pnp:
                         projected_points_ir, _ = cv2.projectPoints(object_points_3D, rvec_ir_est, tvec_ir_est, M2, d2)
                         error_ir = cv2.norm(image_points_ir[valid_idx], projected_points_ir, cv2.NORM_L2) / len(projected_points_ir)
                    else:
                         error_ir = float('inf') # solvePnP failed
                except IndexError:
                    print(f"内部错误：无法找到配对 {pair_num} 的有效索引。")
                    error_ir = float('inf')
                except Exception as e_proj_ir:
                    print(f"警告：计算 IR 投影误差时出错 (Pair {pair_num}): {e_proj_ir}", file=sys.stderr)
                    error_ir = float('inf')

                # Combined error (geometric mean, handles potential inf)
                if error_rgb != float('inf') and error_ir != float('inf'):
                    combined_error = np.sqrt(error_rgb * error_ir) # Or np.sqrt(error_rgb**2 + error_ir**2)
                else:
                    combined_error = float('inf')

            else: # Cannot calculate error if rvecs/tvecs are missing
                 error_rgb = -1.0 # Indicate error cannot be calculated
                 error_ir = -1.0
                 combined_error = -1.0

        # Store result for this pair number
        results.append({
            "pair": pair_num,
            "found_rgb": ret_rgb,
            "found_ir": ret_ir,
            "error_rgb": error_rgb if use_extrinsics else -1.0,
            "error_ir": error_ir if use_extrinsics else -1.0,
            "combined_error": combined_error if use_extrinsics else -1.0
        })

    # --- 4. Display Results ---
    print("\n--- 图像对分析结果 ---")
    print("Pair | Found RGB | Found IR | RGB Err (px) | IR Err (px) | Combined Err (px)")
    print("----------------------------------------------------------------------------")

    # Sort results by combined error (descending, inf/unavailable first)
    results.sort(key=lambda x: x['combined_error'] if x['combined_error'] != -1.0 else float('inf'), reverse=True)

    high_error_pairs = []
    for res in results:
        found_status_rgb = "Yes" if res['found_rgb'] else "No"
        found_status_ir = "Yes" if res['found_ir'] else "No"
        err_rgb_str = f"{res['error_rgb']:.4f}" if res['error_rgb'] != -1.0 and res['error_rgb'] != float('inf') else "N/A"
        err_ir_str = f"{res['error_ir']:.4f}" if res['error_ir'] != -1.0 and res['error_ir'] != float('inf') else "N/A"
        comb_err_str = f"{res['combined_error']:.4f}" if res['combined_error'] != -1.0 and res['combined_error'] != float('inf') else "N/A"

        print(f"{res['pair']:<4} | {found_status_rgb:<9} | {found_status_ir:<8} | {err_rgb_str:<12} | {err_ir_str:<11} | {comb_err_str:<17}")

        # Identify potentially bad pairs (not found or high error)
        if not res['found_rgb'] or not res['found_ir'] or (use_extrinsics and res['combined_error'] == float('inf')):
            high_error_pairs.append(res['pair'])
        elif use_extrinsics and res['combined_error'] > 1.5: # Threshold for high error (adjust as needed)
             high_error_pairs.append(res['pair'])


    print("\n--- 总结 ---")
    if use_extrinsics:
         valid_errors = [r['combined_error'] for r in results if r['combined_error'] != -1.0 and r['combined_error'] != float('inf')]
         if valid_errors:
             avg_combined_error = np.mean(valid_errors)
             print(f"基于成功检测并计算误差的图像对的平均组合误差: {avg_combined_error:.4f} px")
         else:
             print("未能计算任何有效的组合误差。")
    else:
         print("未能加载外参 (rvecs/tvecs)，无法计算数值误差。请检查标定文件或手动检查图像质量。")

    print(f"\n识别出的潜在问题图像对 (未找到角点或误差 > 1.5): {sorted(list(set(high_error_pairs)))}")
    print("建议检查对应的 pair_XX_rgb.png 和 pair_XX_ir.png 文件，并考虑在重新标定时排除它们。")

if __name__ == "__main__":
    main()