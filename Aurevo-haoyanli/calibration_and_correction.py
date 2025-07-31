import time
import json
from config import ParamatersSetting as PS
import os
import numpy as np
import cv2
import open3d as o3d

def save_plane_calibration(
    plane_model, filename=PS.TABLE_PLANE_CALIBRATION_FILE
):  # <--- MODIFIED: default filename
    """
    保存平面标定数据到文件
    """
    try:
        if plane_model is not None and len(plane_model) >= 4:
            plane_data = {
                "plane_model": plane_model.tolist(),  # Convert numpy array to list for JSON
                "timestamp": time.time(),
                "description": "Table plane calibration data, coefficients [a, b, c, d] for ax+by+cz+d=0",
            }
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(plane_data, f, indent=4, ensure_ascii=False)
            print(f"平面标定数据已保存到: {filename}")
            return True
    except Exception as e:
        print(f"保存平面标定数据失败: {e}")
    return False

def load_plane_calibration(
    filename=PS.TABLE_PLANE_CALIBRATION_FILE,
):  # <--- MODIFIED: default filename
    """
    从文件加载平面标定数据
    """
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                plane_data = json.load(f)

            plane_model_list = plane_data.get("plane_model")
            if (
                plane_model_list is not None
                and isinstance(plane_model_list, list)
                and len(plane_model_list) == 4
            ):
                plane_model_np = np.array(plane_model_list, dtype=np.float64)
                print(f"平面标定数据加载成功: {filename}")
                return plane_model_np
            else:
                print(f"警告: {filename} 中的平面模型数据格式无效。")
                return None
        else:
            # print(f"平面标定文件不存在: {filename}") # Be less verbose if file not found is normal
            return None
    except Exception as e:
        print(f"加载平面标定数据失败: {e}")
    return None



def load_calibration_data(use_16bit_mode=False):
    """
    根据深度模式加载相应的标定数据
    """


    if use_16bit_mode:
        calib_file = PS.CURRENT_CALIB_FILE_16BIT
        print(f"加载16bit模式标定数据: {calib_file}")
    else:
        calib_file = PS.CURRENT_CALIB_FILE_8BIT
        print(f"加载8bit模式标定数据: {calib_file}")

    if not os.path.exists(calib_file):
        print(f"错误: 标定文件不存在: {calib_file}")
        return False

    try:
        calib_data = np.load(calib_file)

        # RGB相机参数 (mtx1)
        PS.mtx_rgb = calib_data["mtx1"]
        PS.dist_rgb = calib_data["dist1"]
        PS.fx_rgb, PS.fy_rgb = PS.mtx_rgb[0, 0], PS.mtx_rgb[1, 1]
        PS.cx_rgb, PS.cy_rgb = PS.mtx_rgb[0, 2], PS.mtx_rgb[1, 2]

        # IR/深度相机参数 (mtx2)
        PS.mtx_ir = calib_data["mtx2"]
        PS.dist_ir = calib_data["dist2"]
        PS.fx_ir, PS.fy_ir = PS.mtx_ir[0, 0], PS.mtx_ir[1, 1]
        PS.cx_ir, PS.cy_ir = PS.mtx_ir[0, 2], PS.mtx_ir[1, 2]

        # 立体参数
        PS.R_stereo = calib_data["R"]
        PS.T_stereo = calib_data["T"]

        # 处理不同标定文件的图像尺寸字段
        if use_16bit_mode:
            if "rgb_image_shape_wh" in calib_data:
                PS.calib_rgb_w, PS.calib_rgb_h = calib_data["rgb_image_shape_wh"]
            else:
                PS.calib_rgb_w, PS.calib_rgb_h = PS.RGB_W_OUTPUT, PS.RGB_H_OUTPUT
        else:
            if "rgb_size" in calib_data:
                rgb_size = calib_data["rgb_size"]
                if len(rgb_size) >= 2:
                    PS.calib_rgb_w, PS.calib_rgb_h = int(rgb_size[0]), int(rgb_size[1])
                else:
                    PS.calib_rgb_w, PS.calib_rgb_h = PS.RGB_W_OUTPUT, PS.RGB_H_OUTPUT
            else:
                PS.calib_rgb_w, PS.calib_rgb_h = PS.RGB_W_OUTPUT, PS.RGB_H_OUTPUT  # 创建去畸变映射
        PS.new_mtx_rgb_undistort, roi_rgb = cv2.getOptimalNewCameraMatrix(
            PS.mtx_rgb, PS.dist_rgb, (PS.calib_rgb_w, PS.calib_rgb_h), 1, (PS.calib_rgb_w, PS.calib_rgb_h)
        )
        PS.mapx_rgb, PS.mapy_rgb = cv2.initUndistortRectifyMap(
            PS.mtx_rgb,
            PS.dist_rgb,
            np.eye(3),  # 使用单位矩阵替代None
            PS.new_mtx_rgb_undistort,
            (PS.calib_rgb_w, PS.calib_rgb_h),
            cv2.CV_16SC2,
        )

        PS.mapx_ir, PS.mapy_ir = cv2.initUndistortRectifyMap(
            PS.mtx_ir,
            PS.dist_ir,
            np.eye(3),
            PS.mtx_ir,
            (PS.DEPTH_W, PS.DEPTH_H),
            cv2.CV_16SC2,  # 使用单位矩阵替代None
        )

        # 创建Open3D内参
        PS.o3d_intrinsic_ir = o3d.camera.PinholeCameraIntrinsic(
            PS.DEPTH_W, PS.DEPTH_H, PS.fx_ir, PS.fy_ir, PS.cx_ir, PS.cy_ir
        )

        print("标定数据加载成功:")
        print(
            f"  RGB相机内参: fx={PS.fx_rgb:.2f}, fy={PS.fy_rgb:.2f}, cx={PS.cx_rgb:.2f}, cy={PS.cy_rgb:.2f}"
        )
        print(
            f"  IR相机内参: fx={PS.fx_ir:.2f}, fy={PS.fy_ir:.2f}, cx={PS.cx_ir:.2f}, cy={PS.cy_ir:.2f}"
        )
        print(f"  RGB图像尺寸: {PS.calib_rgb_w}x{PS.calib_rgb_h}")

        return True

    except Exception as e:
        print(f"错误: 加载标定数据失败: {e}")
        return False


def load_depth_correction_model(use_16bit_mode=False):
    """
    根据深度模式加载相应的深度校正模型
    """

    if use_16bit_mode:
        correction_file = PS.CURRENT_DEPTH_CORRECTION_16BIT
        print(f"加载16bit深度校正模型: {correction_file}")
    else:
        correction_file = PS.CURRENT_DEPTH_CORRECTION_8BIT
        print(f"加载8bit深度校正模型: {correction_file}")

    if not os.path.exists(correction_file):
        print(f"警告: 深度校正文件不存在: {correction_file}")
        PS.DEPTH_CORRECTION_MODEL = None
        return False

    try:
        with open(correction_file, "r") as f:
            PS.DEPTH_CORRECTION_MODEL = json.load(f)

        print("深度校正模型加载成功:")
        print(f"  模型类型: {PS.DEPTH_CORRECTION_MODEL.get('model_type', 'unknown')}")
        print(f"  参数: {PS.DEPTH_CORRECTION_MODEL.get('params', [])}")
        print(f"  RMSE: {PS.DEPTH_CORRECTION_MODEL.get('rmse_mm', 'unknown')} mm")

        return True

    except Exception as e:
        print(f"错误: 加载深度校正模型失败: {e}")
        PS.DEPTH_CORRECTION_MODEL = None
        return False

def depth_bytes_to_mm_array(depth_bytes, cam_deep_mode):
    """
    将深度字节数据转换为毫米深度数组（预校正）
    """
    if depth_bytes is None:
        return None

    try:
        if cam_deep_mode == 1:  # 8-bit
            depth_raw = np.frombuffer(depth_bytes, dtype=np.uint8).reshape(
                (PS.DEPTH_H, PS.DEPTH_W)
            )  # 8bit: mm = (raw / divisor)^2
            depth_mm = np.where(
                depth_raw.astype(np.float32) > 0,
                np.power(
                    depth_raw.astype(np.float32) / PS.DEPTH_8BIT_DIVISOR_PRE_CORRECTION, 2
                ),
                0,
            )
        else:  # 16-bit (cam_deep_mode == 0)
            depth_raw = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
                (PS.DEPTH_H, PS.DEPTH_W)
            )
            # 16bit: mm = raw * scale
            depth_mm = depth_raw.astype(np.float32) * PS.DEPTH_16BIT_SCALE_PRE_CORRECTION

        return depth_mm
    except Exception as e:
        print(f"错误: 深度数据转换失败: {e}")
        return None


def apply_depth_correction(depth_mm_pre_correction):
    """
    应用深度校正模型
    """

    if PS.DEPTH_CORRECTION_MODEL is None or depth_mm_pre_correction is None:
        return depth_mm_pre_correction

    try:
        model_type = PS.DEPTH_CORRECTION_MODEL.get("model_type", "")
        params = PS.DEPTH_CORRECTION_MODEL.get("params", [])

        if model_type == "linear" and len(params) >= 2:
            # linear: corrected = a * raw + b
            a, b = params[0], params[1]
            corrected = a * depth_mm_pre_correction + b
        elif model_type == "quadratic" and len(params) >= 3:
            # quadratic: corrected = a * raw^2 + b * raw + c
            a, b, c = params[0], params[1], params[2]
            corrected = (
                a * np.power(depth_mm_pre_correction, 2)
                + b * depth_mm_pre_correction
                + c
            )
        else:
            print(f"警告: 未知的校正模型类型或参数不足: {model_type}")
            return depth_mm_pre_correction

        # 应用深度范围限制
        corrected = np.where(
            (corrected >= PS.MIN_DEPTH_MM) & (corrected <= PS.MAX_DEPTH_MM), corrected, 0
        )

        return corrected

    except Exception as e:
        print(f"错误: 深度校正失败: {e}")
        return depth_mm_pre_correction


def apply_temporal_filter(depth_mm_corrected):
    """
    应用时域滤波
    """

    if depth_mm_corrected is None:
        return None

    PS.depth_history.append(depth_mm_corrected.copy())
    if len(PS.depth_history) > PS.TEMPORAL_FILTER_SIZE:
        PS.depth_history.pop(0)

    if len(PS.depth_history) == 1:
        return PS.depth_history[0]

    # 计算中值滤波
    depth_stack = np.stack(PS.depth_history, axis=0)
    filtered = np.median(depth_stack, axis=0)

    return filtered


def apply_spatial_filter(depth_mm_filtered):
    """
    应用空间滤波
    """
    if depth_mm_filtered is None:
        return None

    try:
        # 高斯滤波
        filtered = cv2.GaussianBlur(
            depth_mm_filtered,
            (PS.SPATIAL_FILTER_SIZE, PS.SPATIAL_FILTER_SIZE),
            PS.SPATIAL_FILTER_SIGMA,
        )
        return filtered
    except Exception as e:
        print(f"错误: 空间滤波失败: {e}")
        return depth_mm_filtered

if __name__ == '__main__':
    res=load_plane_calibration()
    print(res)