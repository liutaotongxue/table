import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import sys


def linear_model(z_reported, a, b):
    return a * z_reported + b


def quadratic_model(z_reported, a, b, c):
    return a * z_reported**2 + b * z_reported + c


def main_analyze():
    parser = argparse.ArgumentParser(
        description="Analyze collected depth data and fit a correction model."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSON file containing (true_mm, reported_mm_roi_median) data points.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="linear",
        choices=["linear", "quadratic"],
        help="Type of correction model to fit.",
    )
    parser.add_argument(
        "--output_params_file",
        type=str,
        default="depth_correction_params.json",
        help="JSON file to save the fitted model parameters.",
    )
    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误：无法加载数据文件 '{args.input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    if not data or len(data) < 3:  # Need at least 2 for linear, 3 for quadratic
        print("错误：数据点不足以进行模型拟合。", file=sys.stderr)
        sys.exit(1)

    true_distances = np.array([d["true_mm"] for d in data])
    reported_distances = np.array([d["reported_mm_roi_median"] for d in data])

    print(f"加载了 {len(true_distances)} 个数据点。")

    params = None
    model_func = None
    param_names = []

    try:
        if args.model_type == "linear":
            model_func = linear_model
            param_names = ["a", "b"]
            # Initial guess for linear: a=1, b=0
            popt, pcov = curve_fit(
                model_func, reported_distances, true_distances, p0=[1.0, 0.0]
            )
            params = popt
            print(
                f"线性模型拟合参数 (Z_corrected = a*Z_reported + b): a={params[0]:.6f}, b={params[1]:.6f}"
            )
        elif args.model_type == "quadratic":
            model_func = quadratic_model
            param_names = ["a", "b", "c"]
            # Initial guess for quadratic: a=0, b=1, c=0
            popt, pcov = curve_fit(
                model_func, reported_distances, true_distances, p0=[0.0, 1.0, 0.0]
            )
            params = popt
            print(
                f"二次模型拟合参数 (Z_corr = a*Z_rep^2 + b*Z_rep + c): a={params[0]:.8f}, b={params[1]:.6f}, c={params[2]:.6f}"
            )
        else:
            print(f"错误：不支持的模型类型 '{args.model_type}'", file=sys.stderr)
            sys.exit(1)

        # --- 评估模型 ---
        corrected_distances_pred = model_func(reported_distances, *params)
        residuals = true_distances - corrected_distances_pred
        rmse = np.sqrt(np.mean(residuals**2))
        print(f"拟合模型的均方根误差 (RMSE): {rmse:.2f} mm")
        max_abs_error = np.max(np.abs(residuals))
        print(f"最大绝对误差: {max_abs_error:.2f} mm")

        # --- 可视化 ---
        plt.figure(figsize=(10, 6))
        plt.scatter(
            reported_distances,
            true_distances,
            label="原始数据 (真实 vs. 相机报告)",
            color="blue",
        )

        # 生成用于绘制拟合曲线的X值
        x_fit = np.linspace(min(reported_distances), max(reported_distances), 100)
        y_fit = model_func(x_fit, *params)
        plt.plot(
            x_fit,
            y_fit,
            label=f"拟合模型 ({args.model_type})",
            color="red",
            linestyle="--",
        )

        plt.plot(
            [min(true_distances), max(true_distances)],
            [min(true_distances), max(true_distances)],
            label="理想情况 (真实 = 报告)",
            color="green",
            linestyle=":",
        )

        plt.xlabel("相机报告的深度 (mm)")
        plt.ylabel("真实深度 (mm)")
        plt.title(
            f"深度校正模型拟合 ({args.model_type.capitalize()}) - RMSE: {rmse:.2f}mm"
        )
        plt.legend()
        plt.grid(True)
        plt.axis("equal")  # Force equal scaling for x and y axes
        plt.show()

        # --- 保存参数 ---
        correction_params = {
            "model_type": args.model_type,
            "params": params.tolist(),
            "param_names": param_names,
            "rmse_mm": rmse,
        }
        with open(args.output_params_file, "w") as f:
            json.dump(correction_params, f, indent=4)
        print(f"校正模型参数已保存到: {args.output_params_file}")

    except RuntimeError as e_fit:
        print(
            f"错误：模型拟合失败。可能是数据点太少或分布不佳: {e_fit}", file=sys.stderr
        )
    except Exception as e_analyze:
        print(f"分析数据或保存参数时发生错误: {e_analyze}", file=sys.stderr)


if __name__ == "__main__":
    # Example usage:
    # First run collect_depth_calibration_data.py to get e.g. my_8bit_depth_data.json
    # Then run this script:
    # python analyze_and_fit_depth_correction.py --input_file my_8bit_depth_data.json --model_type linear --output_params_file 8bit_linear_correction.json
    # python analyze_and_fit_depth_correction.py --input_file my_8bit_depth_data.json --model_type quadratic --output_params_file 8bit_quadratic_correction.json
    main_analyze()
