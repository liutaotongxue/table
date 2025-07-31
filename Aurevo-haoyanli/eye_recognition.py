import mediapipe as mp
import numpy as np
import cv2
from config import ParamatersSetting as PS


try:
    MEDIAPIPE_AVAILABLE = True
    mp_face_mesh = None
    mp_drawing = None  # Try to access the traditional face_mesh and drawing_utils
    try:
        mp_face_mesh = mp.solutions.face_mesh  # type: ignore
        mp_drawing = mp.solutions.drawing_utils  # type: ignore
        print("MediaPipe legacy API loaded successfully")
    except AttributeError:
        print("MediaPipe legacy API not available, trying new API...")

        # Try newer task-based API
        try:
            from mediapipe.tasks.python import vision

            mp_face_mesh = vision.FaceLandmarker  # type: ignore
            print("MediaPipe new task API loaded successfully")
        except (ImportError, AttributeError) as e:
            print(f"MediaPipe new API also failed: {e}")
            mp_face_mesh = None

    if mp_face_mesh is None:
        print("Warning: MediaPipe face_mesh not available in any API version")
        MEDIAPIPE_AVAILABLE = False

except ImportError as e:
    print(f"Warning: MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp = None
    mp_face_mesh = None
    mp_drawing = None

print(">>>> UNIFIED EYE-TO-TABLE DETECTION (CONFIGURABLE DEPTH MODE) <<<<")

face_mesh_model=None
def initialize_mediapipe():
    """初始化MediaPipe面部网格模型"""
    global face_mesh_model
    mp_face_mesh = mp.solutions.face_mesh

    try:
        if not MEDIAPIPE_AVAILABLE or mp_face_mesh is None:
            print("警告: MediaPipe不可用，跳过面部网格初始化")
            return True  # Return true as it's not a critical failure

        # Use the traditional API if available and it's the FaceMesh class
        if hasattr(mp_face_mesh, "FaceMesh") and callable(mp_face_mesh.FaceMesh):
            face_mesh_model = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print(
                "MediaPipe FaceMesh (legacy API) 初始化成功"
            )  # Check if mp_face_mesh is the FaceLandmarker class from the new API
        elif (
            MEDIAPIPE_AVAILABLE
            and mp is not None
            and hasattr(mp, "tasks")
            and mp_face_mesh is not None
        ):
            # For the new API, initialization might be different, e.g. using FaceLandmarker.create_from_options
            # This example assumes the old API structure for simplicity as per original code context.
            # If new API is strictly mp_face_mesh = vision.FaceLandmarker, then usage in process_frame needs update too.
            # For now, let's assume the old API was intended or a compatible wrapper.
            print(
                "警告: MediaPipe FaceLandmarker (new API) detected, but example uses legacy FaceMesh. Process might fail."
            )
            # If you intend to use the new API, the model creation and processing calls need to be adapted.
            # As a placeholder, we'll prevent it from crashing here if it's the class itself.
            face_mesh_model = None  # Or attempt new API init if logic is added.
            print("MediaPipe面部网格模型 (new API) 未完全适配于当前处理流程。")
            return True  # Allow continuation
        else:
            print(
                "警告: MediaPipe API版本不兼容或face_mesh未正确初始化，跳过面部网格。"
            )
            face_mesh_model = None
            return True

        if face_mesh_model:
            print("MediaPipe面部网格模型初始化成功")
        return True
    except Exception as e:
        print(f"错误: MediaPipe初始化失败: {e}")
        face_mesh_model = None  # Ensure it's None on failure
        return False


# ==============================================================================
# === MediaPipe眼部检测函数 ===
# ==============================================================================
def detect_eyes_with_mediapipe(rgb_image):
    """
    使用MediaPipe检测眼部关键点
    """
    global face_mesh_model

    if face_mesh_model is None or rgb_image is None:
        return None, None

    try:
        # 转换为RGB
        rgb_for_mp = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_for_mp.flags.writeable = False  # Optimization        # MediaPipe处理
        results = face_mesh_model.process(rgb_for_mp)  # type: ignore
        rgb_for_mp.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, None

        # 获取第一个检测到的面部
        face_landmarks = results.multi_face_landmarks[0]

        # 眼部关键点索引 (MediaPipe 468个关键点)
        # These are commonly used sets of points around the eyes.
        LEFT_EYE_INDICES = PS.LEFT_EYE_INDICES
        RIGHT_EYE_INDICES = PS.RIGHT_EYE_INDICES
        # For eye centers, sometimes specific pupil landmarks are better if available and reliable
        # For example, left eye: 473, right eye: 468 (if refine_landmarks=True)
        # Using mean of eye outline points as a robust proxy for eye center.

        h, w = rgb_image.shape[:2]

        # 计算左眼中心
        left_eye_points = []
        for idx in LEFT_EYE_INDICES:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                left_eye_points.append([x, y])

        # 计算右眼中心
        right_eye_points = []
        for idx in RIGHT_EYE_INDICES:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                right_eye_points.append([x, y])

        if (
            not left_eye_points or not right_eye_points
        ):  # Check if points were actually extracted
            return None, None

        # 计算眼部中心
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

        return left_eye_center, right_eye_center

    except Exception as e:
        print(f"错误: 眼部检测失败: {e}")
        return None, None


def get_eye_depth_from_neighbor(
    depth_mm, eye_center_rgb, neighbor_size=PS.EYE_DEPTH_NEIGHBOR_SIZE
):
    """
    从眼部邻域获取深度值
    """
    if depth_mm is None or eye_center_rgb is None:
        return None
    if (
        PS.calib_rgb_w == 0 or PS.calib_rgb_h == 0
    ):  # Avoid division by zero if calib data not loaded
        print("警告: RGB标定尺寸为0，无法转换眼部坐标。")
        return None

    try:
        # 将RGB坐标转换为深度坐标
        # 简单的比例缩放（假设RGB和深度图像对齐或已校正对齐）
        rgb_h, rgb_w = PS.calib_rgb_h, PS.calib_rgb_w
        depth_h, depth_w = PS.DEPTH_H, PS.DEPTH_W

        scale_x = depth_w / rgb_w
        scale_y = depth_h / rgb_h

        eye_x_depth = int(eye_center_rgb[0] * scale_x)
        eye_y_depth = int(eye_center_rgb[1] * scale_y)

        # 检查边界
        if not (0 <= eye_x_depth < depth_w and 0 <= eye_y_depth < depth_h):
            return None

        # 定义邻域
        half_size = neighbor_size // 2
        y_min = max(0, eye_y_depth - half_size)
        y_max = min(depth_h, eye_y_depth + half_size + 1)
        x_min = max(0, eye_x_depth - half_size)
        x_max = min(depth_w, eye_x_depth + half_size + 1)

        # 提取邻域深度值
        neighborhood = depth_mm[y_min:y_max, x_min:x_max]
        valid_depths = neighborhood[
            (neighborhood > PS.MIN_DEPTH_MM) & (neighborhood < PS.MAX_DEPTH_MM)
        ]

        if len(valid_depths) == 0:
            return None

        # 返回中值深度
        return np.median(valid_depths)

    except Exception as e:
        print(f"错误: 获取眼部深度失败: {e}")
        return None


if __name__ == '__main__':
    initialize_mediapipe()