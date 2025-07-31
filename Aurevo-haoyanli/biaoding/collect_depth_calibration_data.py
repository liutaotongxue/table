import cv2
import numpy as np
import time
import requests
import struct
import argparse
import os
import json  # For saving data
import sys  # IMPORTED SYS MODULE

# --- MaixSense Communication & Decoding Functions ---
DEFAULT_HOST = "192.168.233.1"
DEFAULT_PORT = 80
DEPTH_W, DEPTH_H = 320, 240

# --- GLOBAL CONSTANTS ---
MAX_DEPTH_MM = (
    1500.0  # Default, can be adjusted if needed for filtering reported values
)
MIN_DEPTH_MM = 200.0  # Default, can be adjusted
# These are for the *initial* conversion from raw 8-bit/16-bit to preliminary mm values
# The depth calibration process will then find a model to correct *these* preliminary values.
DEPTH_8BIT_DIVISOR_PRE_CORRECTION = 5.1
DEPTH_16BIT_SCALE_PRE_CORRECTION = (
    0.25  # If raw 16-bit value / 4 = mm, then mm = raw_value * 0.25
)
# --- END GLOBAL CONSTANTS ---


def frame_config_decode(frame_config):
    try:
        if frame_config is None or len(frame_config) != 12:
            return None
        return struct.unpack("<BBBBBBBBi", frame_config)
    except struct.error:
        return None


def frame_config_encode(
    trigger_mode=1,
    deep_mode=1,  # Default to a specific depth mode for this script
    deep_shift=255,
    ir_mode=255,  # Attempt to disable IR by setting to a likely invalid/off mode
    status_mode=255,  # Attempt to disable Status by setting to a likely invalid/off mode
    status_mask=0,  # Typically 0 or 7 if status_mode is active, 0 should be safe if status_mode is off
    rgb_mode=2,  # 2 is NULL for RGB
    rgb_res=0,
    expose_time=0,
):
    params = [
        trigger_mode,
        deep_mode,
        deep_shift,
        ir_mode,
        status_mode,
        status_mask,
        rgb_mode,
        rgb_res,
        expose_time,
    ]
    if not all(isinstance(p, int) for p in params):
        raise ValueError("所有配置参数必须是整数。")
    return struct.pack("<BBBBBBBBi", *params)


def frame_payload_decode_for_depth(frame_data: bytes, with_config: tuple):
    if with_config is None:
        return None
    try:
        if frame_data is None or len(frame_data) < 8:
            return None
        deep_data_size_header, _ = struct.unpack("<ii", frame_data[:8])
        frame_payload = frame_data[8:]
        total_payload_len = len(frame_payload)

        if deep_data_size_header < 0 or deep_data_size_header > total_payload_len:
            return None

        cam_deep_mode = with_config[1]
        if not (0 <= cam_deep_mode <= 1):
            return None

        expected_depth_bytes = (DEPTH_W * DEPTH_H * 2) >> cam_deep_mode

        # We expect deep_data_size_header to match expected_depth_bytes if only depth is sent
        bytes_to_read_for_depth = min(
            expected_depth_bytes, deep_data_size_header, len(frame_payload)
        )

        if (
            bytes_to_read_for_depth == expected_depth_bytes
            and bytes_to_read_for_depth > 0
        ):
            depth_bytes = frame_payload[:bytes_to_read_for_depth]
            return depth_bytes
        else:
            # print(f"FPD_Depth: Mismatch or zero size. Read: {bytes_to_read_for_depth}, ExpectedFull: {expected_depth_bytes}, HeaderDSize: {deep_data_size_header}", file=sys.stderr)
            return None

    except Exception as e:
        print(f"FPD_Depth: Error during payload decoding: {e}", file=sys.stderr)
        return None


def post_encode_config(config, host, port):
    if config is None:
        return False
    try:
        r = requests.post(f"http://{host}:{port}/set_cfg", data=config, timeout=5)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e_req:
        print(f"错误(post_encode_config): 请求失败 {e_req}", file=sys.stderr)
        return False
    except Exception as e_gen:
        print(f"错误(post_encode_config): 意外错误 {e_gen}", file=sys.stderr)
        return False


def get_frame_from_http(host, port):
    try:
        r = requests.get(f"http://{host}:{port}/getdeep", timeout=1.0)
        r.raise_for_status()
        if len(r.content) < 28:
            return None
        return r.content
    except requests.exceptions.RequestException:
        return None  # Common error, less verbose
    except Exception as e_gen:
        # print(f"错误(get_frame_from_http): 意外错误 {e_gen}", file=sys.stderr) # Can be verbose
        return None


# --- END MaixSense Functions ---


def raw_depth_to_reported_mm(depth_map_raw_bytes, cam_deep_mode_from_config):
    if depth_map_raw_bytes is None:
        return None
    try:
        depth_dtype = np.uint16 if cam_deep_mode_from_config == 0 else np.uint8
        expected_buffer_len = DEPTH_W * DEPTH_H * (2 if depth_dtype == np.uint16 else 1)

        if len(depth_map_raw_bytes) != expected_buffer_len:
            # print(f"raw_depth_to_reported_mm: Mismatch len {len(depth_map_raw_bytes)} vs exp {expected_buffer_len} for mode {cam_deep_mode_from_config}", file=sys.stderr)
            return None

        depth_map_raw_np = np.frombuffer(
            depth_map_raw_bytes, dtype=depth_dtype
        ).reshape((DEPTH_H, DEPTH_W))

        reported_mm = None
        if depth_map_raw_np.dtype == np.uint8:  # 8-bit depth
            depth_map_float = depth_map_raw_np.astype(np.float32)
            if DEPTH_8BIT_DIVISOR_PRE_CORRECTION > 1e-6:
                with np.errstate(divide="ignore", invalid="ignore"):
                    reported_mm = np.square(
                        depth_map_float / DEPTH_8BIT_DIVISOR_PRE_CORRECTION
                    )
                    reported_mm[~np.isfinite(reported_mm)] = 0
            else:
                reported_mm = np.zeros_like(depth_map_float)
        elif depth_map_raw_np.dtype == np.uint16:  # 16-bit depth
            reported_mm = (
                depth_map_raw_np.astype(np.float32) * DEPTH_16BIT_SCALE_PRE_CORRECTION
            )

        return reported_mm
    except Exception as e:
        print(f"raw_depth_to_reported_mm error: {e}", file=sys.stderr)
        return None


def print_capture_instructions():
    print("\n操作指南:")
    print("  - 将相机对准平面，并精确测量相机到平面的物理距离。")
    print("  - 在终端中输入该真实距离 (单位：毫米 mm)。")
    print("  - 按 Enter 键确认输入，脚本将采集当前帧的中心区域深度值。")
    print(
        "  - 在OpenCV窗口激活时，按 'q' 键可以中断当前帧的处理并尝试获取下一帧（不会退出采集）。"
    )
    print("  - 建议在多个不同距离处采集数据点 (例如 300mm, 500mm, 700mm ... 1500mm)。")
    print(
        "  - 在终端提示输入时，输入 's' (save) 然后按 Enter 可以保存当前数据并退出程序。"
    )
    print(
        "  - 在终端提示输入时，输入 'q' (quit) 然后按 Enter 可以不保存当前数据点并退出程序。"
    )
    print("-----------------------------------------")


def main_collect_data():
    parser = argparse.ArgumentParser(
        description="Collect data for depth camera calibration."
    )
    parser.add_argument(
        "--host", type=str, default=DEFAULT_HOST, help="Camera IP address."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Camera port.")
    parser.add_argument(
        "--depth_mode_cam",
        type=int,
        default=1,
        choices=[0, 1],
        help="Depth mode to configure camera (0: 16-bit, 1: 8-bit).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="depth_calibration_data.json",
        help="JSON file to save collected (true_dist, reported_dist) pairs.",
    )
    parser.add_argument(
        "--center_roi_ratio",
        type=float,
        default=0.2,
        help="Ratio of image width/height to use for central ROI.",
    )
    args = parser.parse_args()

    print("--- MaixSense深度校准数据采集 ---")
    print(f"将配置相机为深度模式: {'16-bit' if args.depth_mode_cam == 0 else '8-bit'}")
    print(f"数据将保存到: {args.output_file}")
    print("请将相机对准一个平整的平面（如墙壁）。")

    cam_config_bytes = frame_config_encode(
        deep_mode=args.depth_mode_cam,
        ir_mode=255,
        status_mode=255,
        rgb_mode=2,
        status_mask=0,
    )
    if not post_encode_config(cam_config_bytes, args.host, args.port):
        print("错误：配置相机失败！请检查连接和相机状态。", file=sys.stderr)
        sys.exit(1)
    print("相机配置成功。")
    time.sleep(0.5)

    collected_data = []
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r") as f:
                loaded_items = json.load(f)
                if isinstance(loaded_items, list):
                    collected_data = loaded_items
                    print(
                        f"已从 '{args.output_file}' 加载 {len(collected_data)} 个先前采集的数据点。"
                    )
                else:
                    print(
                        f"警告：文件 '{args.output_file}' 内容不是预期的列表格式。",
                        file=sys.stderr,
                    )
        except Exception as e:
            print(
                f"警告：无法加载先前的数据文件 '{args.output_file}': {e}",
                file=sys.stderr,
            )
            collected_data = []

    cv2.namedWindow("Depth Stream (ROI in Green)", cv2.WINDOW_AUTOSIZE)
    roi_w = int(DEPTH_W * args.center_roi_ratio)
    roi_h = int(DEPTH_H * args.center_roi_ratio)
    roi_x = (DEPTH_W - roi_w) // 2
    roi_y = (DEPTH_H - roi_h) // 2

    print_capture_instructions()

    loop_counter = 0
    last_avg_reported_depth = 0.0
    reported_mm_map_for_display = None  # Keep last valid map for display

    while True:
        loop_counter += 1
        raw_frame = get_frame_from_http(args.host, args.port)

        current_frame_roi_median = None

        if raw_frame:
            config_tuple_echo = frame_config_decode(raw_frame[16:28])
            if config_tuple_echo:
                if config_tuple_echo[1] != args.depth_mode_cam:
                    if loop_counter % 30 == 0:
                        print(
                            f"警告：相机返回深度模式 ({config_tuple_echo[1]}) 与请求 ({args.depth_mode_cam}) 不符! 可能需重启或重新配置。",
                            file=sys.stderr,
                        )

                depth_bytes_payload = frame_payload_decode_for_depth(
                    raw_frame[28:], config_tuple_echo
                )
                if depth_bytes_payload:
                    reported_mm_map_current = raw_depth_to_reported_mm(
                        depth_bytes_payload, config_tuple_echo[1]
                    )
                    if reported_mm_map_current is not None:
                        reported_mm_map_for_display = (
                            reported_mm_map_current  # Update display map
                        )
                        roi = reported_mm_map_current[
                            roi_y : roi_y + roi_h, roi_x : roi_x + roi_w
                        ]
                        valid_roi_pixels = roi[
                            roi > MIN_DEPTH_MM * 0.1
                        ]  # Filter out very small/zero depths from ROI

                        if (
                            len(valid_roi_pixels) > roi.size * 0.05
                        ):  # Ensure some valid pixels in ROI
                            current_frame_roi_median = float(
                                np.median(valid_roi_pixels)
                            )  # Convert to Python float
                            last_avg_reported_depth = current_frame_roi_median

        # Display logic using reported_mm_map_for_display
        display_depth_bgr = np.zeros((DEPTH_H, DEPTH_W, 3), dtype=np.uint8)
        if reported_mm_map_for_display is not None:
            display_depth_norm = cv2.normalize(
                reported_mm_map_for_display,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            display_depth_bgr = cv2.applyColorMap(display_depth_norm, cv2.COLORMAP_JET)

        cv2.rectangle(
            display_depth_bgr,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            (0, 255, 0),
            1,
        )
        display_text = f"ROI Median: {last_avg_reported_depth:.1f} mm"
        if raw_frame is None:
            display_text += " (No Frame)"
        elif config_tuple_echo is None:
            display_text += " (Config Err)"
        elif depth_bytes_payload is None:
            display_text += " (Depth Payload Err)"
        elif reported_mm_map_current is None:
            display_text += " (Depth Convert Err)"  # Assuming reported_mm_map_current was from this iter
        elif current_frame_roi_median is None:
            display_text += " (ROI Invalid!)"

        cv2.putText(
            display_depth_bgr,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow("Depth Stream (ROI in Green)", display_depth_bgr)

        key = cv2.waitKey(1) & 0xFF  # Check key for OpenCV window
        if key == ord("q"):
            print("  (OpenCV窗口 'q' 按下，将尝试刷新帧并等待终端输入...)")
            continue  # Skip terminal input for this iteration, refresh frame

        # Terminal input for distance
        try:
            prompt_message = f"已采集 {len(collected_data)} 点。当前ROI报告中位深度: {last_avg_reported_depth:.1f}mm. \n请输入真实距离(mm), 或 's'保存退出, 'q'不保存退出: "
            true_dist_str = input(prompt_message)

            if true_dist_str.lower() == "q":
                print("用户选择退出，不保存当前点。")
                break
            if true_dist_str.lower() == "s":
                # Save the point associated with last_avg_reported_depth if it was valid
                if last_avg_reported_depth > MIN_DEPTH_MM * 0.1:
                    final_dist_str = input(
                        f"  为最后有效ROI中位值({last_avg_reported_depth:.1f}mm) 输入真实距离(mm)以保存: "
                    )
                    try:
                        true_distance_mm_final = float(final_dist_str)
                        collected_data.append(
                            {
                                "true_mm": float(true_distance_mm_final),
                                "reported_mm_roi_median": float(
                                    last_avg_reported_depth
                                ),
                            }
                        )
                        print(
                            f"  最终数据点已添加: (真实: {true_distance_mm_final:.1f} mm, 报告: {last_avg_reported_depth:.1f} mm)"
                        )
                    except ValueError:
                        print("  无效的最终距离输入，此点未保存。")
                else:
                    print("  没有有效的最后ROI深度值可供保存。")
                print("用户选择保存并退出。")
                break

            true_distance_mm = float(true_dist_str)
            # Use last_avg_reported_depth for consistency with what user saw when prompted
            if last_avg_reported_depth > MIN_DEPTH_MM * 0.1:
                collected_data.append(
                    {
                        "true_mm": float(true_distance_mm),
                        "reported_mm_roi_median": float(last_avg_reported_depth),
                    }
                )
                print(
                    f"  数据点已添加: (真实: {true_distance_mm:.1f} mm, 报告: {last_avg_reported_depth:.1f} mm)"
                )
                try:
                    with open(args.output_file, "w") as f:
                        json.dump(collected_data, f, indent=4)
                    # print(f"  数据已增量更新到 '{args.output_file}'") # Less verbose
                except Exception as e_save_inc:
                    print(f"错误：保存增量数据时: {e_save_inc}", file=sys.stderr)
            else:
                print(
                    f"  未记录数据点，因相机报告的中心区域深度无效 ({last_avg_reported_depth=}) 或过小。请确保ROI对准平面且在有效范围内。"
                )

        except ValueError:
            print("  输入无效，请输入一个数字作为距离，或 'q'/'s'。")
        except EOFError:
            print("\n检测到输入结束，退出。")
            break
        except KeyboardInterrupt:
            print("\n用户通过Ctrl+C中断，退出。")
            break
        except Exception as e_input:
            print(f"  处理输入时发生错误: {e_input}")
            break

    cv2.destroyAllWindows()
    print(f"\n数据采集结束。总共采集了 {len(collected_data)} 个数据点。")

    if collected_data:
        try:
            # Final conversion check (already done when appending, but good for safety)
            python_native_final_data = []
            for item in collected_data:
                python_native_final_data.append(
                    {
                        "true_mm": float(item["true_mm"]),
                        "reported_mm_roi_median": float(
                            item.get("reported_mm_roi_median", 0.0)
                        ),
                    }
                )
            with open(args.output_file, "w") as f:
                json.dump(python_native_final_data, f, indent=4)
            print(f"最终数据已保存到 '{args.output_file}'")
        except Exception as e_save_final:
            print(f"错误：保存最终数据时: {e_save_final}", file=sys.stderr)
    else:
        print("没有采集到有效数据。")


if __name__ == "__main__":
    main_collect_data()
