from config import ParamatersSetting as PS
import math


def calculate_point_to_plane_distance(point_3d, plane_model):
    """
    计算点到平面的距离
    """
    if point_3d is None or plane_model is None or len(plane_model) != 4:
        return None

    try:
        a, b, c, d = plane_model
        x, y, z = point_3d

        # 点到平面距离公式: |ax + by + cz + d| / sqrt(a² + b² + c²)
        numerator = abs(a * x + b * y + c * z + d)
        denominator = math.sqrt(a**2 + b**2 + c**2)

        if denominator == 0:  # Should not happen for a valid plane
            return None

        distance = numerator / denominator
        return distance * 1000.0  # 转换为毫米

    except Exception as e:
        print(f"错误: 计算点到平面距离失败: {e}")
        return None


def depth_to_3d_point(depth_mm_value, pixel_x, pixel_y):
    """
    将深度像素坐标转换为3D点 (使用IR相机内参)
    """
    if depth_mm_value is None or depth_mm_value <= 0:
        return None
    if PS.fx_ir is None or PS.fy_ir is None or PS.cx_ir is None or PS.cy_ir is None:
        print("错误: 深度转3D点时IR相机内参未初始化。")
        return None
    if PS.fx_ir == 0 or PS.fy_ir == 0:  # Avoid division by zero
        print("错误: IR相机焦距(fx_ir, fy_ir)为0。")
        return None

    try:
        z_m = depth_mm_value / 1000.0  # 转换为米
        x_m = (pixel_x - PS.cx_ir) * z_m / PS.fx_ir
        y_m = (pixel_y - PS.cy_ir) * z_m / PS.fy_ir

        return [x_m, y_m, z_m]

    except Exception as e:
        print(f"错误: 深度转3D点失败: {e}")
        return None
