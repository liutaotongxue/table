from typing import Optional, Any

class ParamatersSetting():
    # ---------- 配置参数 ---------- #
    DEFAULT_HOST = "192.168.233.1"
    DEFAULT_PORT = 80
    RGB_MODE = 1
    DEPTH_W, DEPTH_H = 320, 240
    RGB_W_OUTPUT, RGB_H_OUTPUT = 640, 480
    MIN_DEPTH_MM = 10.0
    MAX_DEPTH_MM = 2000.0

    # 深度转换参数（预校正）
    DEPTH_8BIT_DIVISOR_PRE_CORRECTION = 5.1
    DEPTH_16BIT_SCALE_PRE_CORRECTION = 0.25

    # 滤波和平滑参数
    TEMPORAL_FILTER_SIZE = 3
    SPATIAL_FILTER_SIZE = 5
    SPATIAL_FILTER_SIGMA = 1.0
    EYE_DEPTH_NEIGHBOR_SIZE = 5
    EMA_ALPHA_EYE_TABLE_DIST = 0.1

    # YOLO和平面拟合参数
    YOLO_MODEL_NAME = "Aurevo-haoyanli/model/yolov8n-seg.pt"
    TABLE_CLASS_ID = [41, 60]  # COCO 'dining table'
    YOLO_CONF_THRESHOLD = 0.25
    YOLO_IOU_THRESHOLD = 0.45
    PLANE_FIT_DISTANCE_THRESHOLD_M = 0.02
    MIN_POINTS_FOR_PLANE_FIT = 100
    MIN_INLIERS_FOR_PLANE_FIT = 50

    # 掩码后处理参数
    MASK_OPEN_KERNEL_SIZE = 5
    MASK_CLOSE_KERNEL_SIZE = 7
    MASK_MIN_AREA_PERCENT = 0.01

    # --- 全局变量 ---
    # 深度模式配置（可切换）
    DEPTH_MODE: int = 0  # 0 for 16-bit, 1 for 8-bit（切换为16bit）
    CURRENT_CALIB_FILE_8BIT = "Aurevo-haoyanli/my_stereo_data.npz"
    CURRENT_CALIB_FILE_16BIT = "Aurevo-haoyanli/biaoding/stereo_calibration_data16BIT.npz"
    CURRENT_DEPTH_CORRECTION_8BIT = "Aurevo-haoyanli/biaoding/depth_correction_quadratic_params_8bit.json"
    CURRENT_DEPTH_CORRECTION_16BIT = "Aurevo-haoyanli/biaoding/depth_correction_quadratic_params_16bit.json"

    # 眼部点索引
    LEFT_EYE_INDICES=[33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    RIGHT_EYE_INDICES=[362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]


    # 配置文件路径
    CONFIG_FILE = "config.json"
    TABLE_PLANE_CALIBRATION_FILE = (
        "Aurevo-haoyanli/parameters/table_plane_calibration.json"  # <--- ADDED: Filename for saved plane
    )

    # 标定数据
    mtx_rgb, dist_rgb, fx_rgb, fy_rgb, cx_rgb, cy_rgb = [None] * 6
    mtx_ir, dist_ir, fx_ir, fy_ir, cx_ir, cy_ir, o3d_intrinsic_ir = [None] * 7
    R_stereo, T_stereo = None, None
    mapx_rgb, mapy_rgb, new_mtx_rgb_undistort = None, None, None
    mapx_ir, mapy_ir = None, None
    calib_rgb_w, calib_rgb_h = RGB_W_OUTPUT, RGB_H_OUTPUT

    # 深度校正模型
    DEPTH_CORRECTION_MODEL = None

    # AI模型
    yolo_model, yolo_device = None, "cpu"  # 强制使用CPU以避免CUDA兼容性问题
    face_mesh_model = None

    # 历史数据和平滑
    depth_history = []
    ema_eye_to_table_dist_mm: Optional[float] = None

    # --- 锁定模式和自动校准变量 ---
    PLANE_IS_LOCKED: bool = False
    LOCKED_PLANE_MODEL: Optional[Any] = None
    IN_CALIBRATION_MODE: bool = True  # 程序启动时进入校准模式
    CALIBRATION_FRAMES_TOTAL = 100  # 校准总帧数
    best_plane_model: Optional[Any] = None
    best_plane_score: float = -1
    calibration_frame_count: int = 0
