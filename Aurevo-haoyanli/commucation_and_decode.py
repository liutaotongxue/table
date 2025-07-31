import struct
import requests
from config import ParamatersSetting as PS
import cv2
import numpy as np


def frame_config_decode(frame_config):
    try:
        if frame_config is None or len(frame_config) != 12:
            return None
        return struct.unpack("<BBBBBBBBi", frame_config)
    except struct.error:
        return None


def frame_config_encode(
    trigger_mode=1,
    deep_mode=None,  # 将使用全局DEPTH_MODE
    deep_shift=255,
    ir_mode=255,
    status_mode=255,
    status_mask=0,
    rgb_mode=PS.RGB_MODE,
    rgb_res=0,
    expose_time=0,
):
    if deep_mode is None:
        deep_mode = PS.DEPTH_MODE

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
        print("错误: 所有配置参数必须是整数。")
        return None
    try:
        return struct.pack("<BBBBBBBBi", *params)
    except struct.error as e:
        print(f"错误: 打包配置时出错: {e}")
        return None


def frame_payload_decode(frame_data: bytes, with_config: tuple):
    if with_config is None:
        return (None, None, None, None)
    try:
        if frame_data is None or len(frame_data) < 8:
            return (None, None, None, None)
        deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
        frame_payload = frame_data[8:]
        total_payload_len = len(frame_payload)
        current_offset = 0
        if (
            deep_data_size < 0
            or rgb_data_size < 0
            or deep_data_size > total_payload_len
            or (deep_data_size + rgb_data_size) > total_payload_len
        ):
            return (None, None, None, None)

        depth_bytes, rgb_numpy_bgr = None, None

        cam_deep_mode = with_config[1]
        expected_depth_size = 0
        if 0 <= cam_deep_mode <= 1:  # 0:16bit, 1:8bit
            expected_depth_size = (PS.DEPTH_W * PS.DEPTH_H * 2) >> cam_deep_mode
        else:
            return (None, None, None, None)

        actual_depth_payload_size = min(
            expected_depth_size,
            deep_data_size - current_offset,
            total_payload_len - current_offset,
        )

        if (
            actual_depth_payload_size == expected_depth_size
            and actual_depth_payload_size > 0
        ):
            depth_bytes = frame_payload[
                current_offset : current_offset + actual_depth_payload_size
            ]

        # RGB数据解析
        rgb_payload_start_offset = deep_data_size
        actual_rgb_available = total_payload_len - rgb_payload_start_offset

        if rgb_data_size > 0 and actual_rgb_available >= rgb_data_size:
            rgb_img_bytes = frame_payload[
                rgb_payload_start_offset : rgb_payload_start_offset + rgb_data_size
            ]
            if with_config[6] == 1:  # RGB_MODE for JPEG
                try:
                    jpeg = cv2.imdecode(
                        np.frombuffer(rgb_img_bytes, "uint8"), cv2.IMREAD_COLOR
                    )
                    if jpeg is not None and jpeg.ndim == 3 and jpeg.shape[2] == 3:
                        rgb_numpy_bgr = jpeg
                except Exception:
                    pass

        return (depth_bytes, None, None, rgb_numpy_bgr)
    except Exception:
        return (None, None, None, None)


def post_encode_config(config, host, port):
    if config is None:
        return False
    try:
        r = requests.post(f"http://{host}:{port}/set_cfg", data=config, timeout=2)
        r.raise_for_status()
        return True
    except Exception:
        return False


def get_frame_from_http(host, port):
    try:
        r = requests.get(f"http://{host}:{port}/getdeep", timeout=0.5)
        r.raise_for_status()
        return r.content if len(r.content) >= 28 else None
    except Exception:
        return None
