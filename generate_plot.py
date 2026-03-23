from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BLOCK_CONFIGURATIONS = [(8, 8), (16, 16), (32, 32), (32, 8)]
STREAM_COUNTS = [1, 2, 4, 8, 16]


def validate_columns(dataframe: pd.DataFrame, required_columns: list[str], dataframe_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing_columns}")


def get_largest_resolution(dataframe: pd.DataFrame) -> tuple[int, int]:
    validate_columns(dataframe, ["w", "h"], "dataframe")

    resolutions = (
        dataframe[["w", "h"]]
        .dropna()
        .drop_duplicates()
        .assign(pixels=lambda rows: rows["w"] * rows["h"])
        .sort_values("pixels")
    )
    if resolutions.empty:
        raise ValueError("No valid resolution found in dataframe")

    largest_row = resolutions.iloc[-1]
    return int(largest_row["w"]), int(largest_row["h"])


def apply_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def save_plot(output_directory: Path, filename: str) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    return output_path


def plot_gaussian_method_ratio(benchmark_df: pd.DataFrame, output_directory: Path) -> Path:
    validate_columns(
        benchmark_df,
        ["algo", "impl", "variant", "w", "h", "kernel", "t2d_ms", "tsep_ms"],
        "benchmark_df",
    )

    gaussian_gpu = benchmark_df[
        (benchmark_df["algo"] == "gaussian") & (benchmark_df["impl"] == "gpu")
    ].copy()
    gaussian_gpu = gaussian_gpu.dropna(subset=["w", "h", "kernel", "t2d_ms", "tsep_ms", "variant"])
    if gaussian_gpu.empty:
        raise ValueError("No Gaussian GPU rows available for the ratio plot")

    gaussian_gpu["pixels"] = gaussian_gpu["w"] * gaussian_gpu["h"]
    gaussian_gpu["separable_to_2d_ratio"] = gaussian_gpu["tsep_ms"] / gaussian_gpu["t2d_ms"]

    apply_plot_style()
    plt.figure()

    variant_styles = [
        ("baseline", "#4C72B0", "Baseline GPU"),
        ("fast", "#55A868", "Optimized GPU"),
    ]

    for variant_name, color, label in variant_styles:
        curve = (
            gaussian_gpu[gaussian_gpu["variant"] == variant_name]
            .groupby("pixels")["separable_to_2d_ratio"]
            .median()
            .reset_index()
            .sort_values("pixels")
        )
        if curve.empty:
            continue

        plt.plot(
            curve["pixels"],
            curve["separable_to_2d_ratio"],
            marker="o",
            linewidth=2.4,
            label=label,
            color=color,
        )

    plt.axhline(1.0, linestyle="--", linewidth=1.4, color="#333333")
    plt.text(512 * 512, 1.02, "Parity line", color="#333333")
    plt.xticks(
        [512 * 512, 1024 * 1024, 2048 * 2048],
        ["512x512", "1024x1024", "2048x2048"],
    )
    plt.xlabel("Image Resolution")
    plt.ylabel("Separable Time / 2D Time")
    plt.title("Gaussian GPU: Separable vs 2D Time Ratio")
    plt.legend()

    return save_plot(output_directory, "gaussian_method_ratio.png")


def plot_gaussian_throughput_scaling(
    benchmark_df: pd.DataFrame,
    output_directory: Path,
    kernel_size: int = 31,
) -> Path:
    validate_columns(
        benchmark_df,
        ["algo", "impl", "variant", "w", "h", "kernel", "t2d_ms", "tsep_ms"],
        "benchmark_df",
    )

    gaussian_gpu = benchmark_df[
        (benchmark_df["algo"] == "gaussian")
        & (benchmark_df["impl"] == "gpu")
        & (benchmark_df["kernel"] == kernel_size)
    ].copy()
    gaussian_gpu = gaussian_gpu.dropna(subset=["w", "h", "t2d_ms", "tsep_ms", "variant"])
    if gaussian_gpu.empty:
        raise ValueError(f"No Gaussian GPU rows available for K={kernel_size}")

    gaussian_gpu["pixels"] = gaussian_gpu["w"] * gaussian_gpu["h"]
    gaussian_gpu["throughput_2d"] = gaussian_gpu["pixels"] / gaussian_gpu["t2d_ms"] / 1000.0
    gaussian_gpu["throughput_separable"] = gaussian_gpu["pixels"] / gaussian_gpu["tsep_ms"] / 1000.0

    apply_plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    variant_panels = [("baseline", axes[0], "Baseline GPU"), ("fast", axes[1], "Optimized GPU")]

    for variant_name, axis, panel_title in variant_panels:
        curve = (
            gaussian_gpu[gaussian_gpu["variant"] == variant_name]
            .groupby("pixels")[["throughput_2d", "throughput_separable"]]
            .median()
            .reset_index()
            .sort_values("pixels")
        )
        if curve.empty:
            continue

        axis.plot(curve["pixels"], curve["throughput_2d"], marker="o", linewidth=2.2, color="#C44E52", label="2D")
        axis.plot(
            curve["pixels"],
            curve["throughput_separable"],
            marker="o",
            linewidth=2.2,
            color="#4C72B0",
            label="Separable",
        )
        axis.set_xticks([512 * 512, 1024 * 1024, 2048 * 2048], ["512x512", "1024x1024", "2048x2048"])
        axis.set_title(panel_title)
        axis.set_xlabel("Image Resolution")
        axis.legend()

    axes[0].set_ylabel("Throughput (MPix/s)")
    figure.suptitle(f"Gaussian GPU Throughput Scaling (K={kernel_size})", fontweight="bold")

    return save_plot(output_directory, "gaussian_throughput_scaling.png")


def plot_sobel_block_performance(
    tuning_df: pd.DataFrame,
    output_directory: Path,
    resolution: tuple[int, int] | None = None,
) -> Path:
    validate_columns(
        tuning_df,
        ["algo", "type", "w", "h", "bx", "by", "time_ms"],
        "tuning_df",
    )

    if resolution is None:
        resolution = get_largest_resolution(
            tuning_df[(tuning_df["algo"] == "sobel") & (tuning_df["type"] == "block")]
        )
    width, height = resolution

    sobel_tuning = tuning_df[
        (tuning_df["algo"] == "sobel")
        & (tuning_df["type"] == "block")
        & (tuning_df["w"] == width)
        & (tuning_df["h"] == height)
    ].copy()
    sobel_tuning = sobel_tuning.dropna(subset=["bx", "by", "time_ms"])
    sobel_tuning["bx"] = sobel_tuning["bx"].astype(int)
    sobel_tuning["by"] = sobel_tuning["by"].astype(int)
    sobel_tuning = sobel_tuning[sobel_tuning[["bx", "by"]].apply(tuple, axis=1).isin(BLOCK_CONFIGURATIONS)]
    if sobel_tuning.empty:
        raise ValueError(f"No Sobel block-tuning rows available at {width}x{height}")

    median_times = sobel_tuning.groupby(["bx", "by"])["time_ms"].median().reset_index()
    median_times["label"] = median_times.apply(lambda row: f"{int(row['bx'])}x{int(row['by'])}", axis=1)
    median_times["throughput_mpix_s"] = (width * height) / median_times["time_ms"] / 1000.0

    ordered_rows = []
    for block_x, block_y in BLOCK_CONFIGURATIONS:
        match = median_times[(median_times["bx"] == block_x) & (median_times["by"] == block_y)]
        if not match.empty:
            ordered_rows.append(match.iloc[0])

    median_times = pd.DataFrame(ordered_rows)
    if median_times.empty:
        raise ValueError("No ordered Sobel block configurations available to plot")

    colors = ["#4C72B0" if label != "32x32" else "#C44E52" for label in median_times["label"]]

    apply_plot_style()
    plt.figure()
    bars = plt.bar(median_times["label"], median_times["throughput_mpix_s"], color=colors, width=0.62)
    plt.xlabel("Block Configuration")
    plt.ylabel("Throughput (MPix/s)")
    plt.title(f"Sobel GPU Block Configuration Performance ({width}x{height})")

    best_index = median_times["throughput_mpix_s"].idxmax()
    best_label = median_times.loc[best_index, "label"]
    best_value = median_times.loc[best_index, "throughput_mpix_s"]

    for bar, value in zip(bars, median_times["throughput_mpix_s"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.annotate(
        f"Best configuration: {best_label}",
        xy=(median_times.index.get_loc(best_index), best_value),
        xytext=(median_times.index.get_loc(best_index), best_value * 1.06),
        ha="center",
        color="#2E8B57",
        fontweight="bold",
    )

    return save_plot(output_directory, "sobel_block_performance.png")


def plot_rgb2yuv_stream_scaling(
    tuning_df: pd.DataFrame,
    output_directory: Path,
    resolution: tuple[int, int] | None = None,
) -> Path:
    validate_columns(
        tuning_df,
        ["algo", "type", "w", "h", "streams", "time_ms"],
        "tuning_df",
    )

    if resolution is None:
        resolution = get_largest_resolution(
            tuning_df[(tuning_df["algo"] == "rgb2yuv") & (tuning_df["type"] == "stream")]
        )
    width, height = resolution

    rgb2yuv_streams = tuning_df[
        (tuning_df["algo"] == "rgb2yuv")
        & (tuning_df["type"] == "stream")
        & (tuning_df["w"] == width)
        & (tuning_df["h"] == height)
    ].copy()
    rgb2yuv_streams = rgb2yuv_streams.dropna(subset=["streams", "time_ms"])
    rgb2yuv_streams["streams"] = rgb2yuv_streams["streams"].astype(int)
    rgb2yuv_streams = rgb2yuv_streams[rgb2yuv_streams["streams"].isin(STREAM_COUNTS)]
    if rgb2yuv_streams.empty:
        raise ValueError(f"No RGB2YUV stream-tuning rows available at {width}x{height}")

    median_times = (
        rgb2yuv_streams.groupby("streams")["time_ms"]
        .median()
        .reindex(STREAM_COUNTS)
        .dropna()
        .reset_index()
    )
    if median_times.empty:
        raise ValueError("No RGB2YUV stream series available to plot")

    apply_plot_style()
    plt.figure()
    plt.plot(
        median_times["streams"],
        median_times["time_ms"],
        marker="o",
        linewidth=2.4,
        color="#64B5CD",
    )

    for stream_count, elapsed_time in zip(median_times["streams"], median_times["time_ms"]):
        plt.text(
            stream_count,
            elapsed_time + 0.015,
            f"{elapsed_time:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2F2F2F",
        )

    if 4 in median_times["streams"].values:
        inflection_point = median_times[median_times["streams"] == 4].iloc[0]
        plt.scatter([inflection_point["streams"]], [inflection_point["time_ms"]], color="#2E8B57", s=80, zorder=3)
        plt.annotate(
            "Strong scaling region",
            xy=(inflection_point["streams"], inflection_point["time_ms"]),
            xytext=(inflection_point["streams"] + 0.45, inflection_point["time_ms"] * 1.05),
            arrowprops={"arrowstyle": "->", "color": "#2E8B57", "lw": 1.2},
            color="#2E8B57",
            fontweight="bold",
        )

    if 8 in median_times["streams"].values and 16 in median_times["streams"].values:
        value_at_8 = median_times[median_times["streams"] == 8].iloc[0]
        value_at_16 = median_times[median_times["streams"] == 16].iloc[0]
        plt.axvspan(8, 16, color="#DDDDDD", alpha=0.18)
        plt.scatter([value_at_16["streams"]], [value_at_16["time_ms"]], color="#C44E52", s=80, zorder=3)
        plt.annotate(
            "Saturation region",
            xy=(value_at_16["streams"], value_at_16["time_ms"]),
            xytext=(value_at_16["streams"] - 3.3, max(value_at_8["time_ms"], value_at_16["time_ms"]) * 1.06),
            arrowprops={"arrowstyle": "->", "color": "#C44E52", "lw": 1.2},
            color="#C44E52",
            fontweight="bold",
        )

    plt.xticks(STREAM_COUNTS)
    plt.xlabel("Number of CUDA Streams")
    plt.ylabel("Average End-to-End Time per Pass (ms)")
    plt.title(f"RGB2YUV Multi-Stream Scaling ({width}x{height})")

    return save_plot(output_directory, "rgb2yuv_stream_scaling.png")


def plot_rgb2yuv_end_to_end_comparison(
    benchmark_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    output_directory: Path,
    resolution: tuple[int, int] | None = None,
) -> Path:
    validate_columns(
        benchmark_df,
        ["algo", "impl", "variant", "w", "h", "avg_ms", "h2d_ms", "d2h_ms"],
        "benchmark_df",
    )
    validate_columns(
        tuning_df,
        ["algo", "type", "w", "h", "streams", "time_ms"],
        "tuning_df",
    )

    if resolution is None:
        resolution = get_largest_resolution(benchmark_df[benchmark_df["algo"] == "rgb2yuv"])
    width, height = resolution

    rgb2yuv_single_stream = benchmark_df[
        (benchmark_df["algo"] == "rgb2yuv")
        & (benchmark_df["impl"] == "gpu")
        & (benchmark_df["variant"] == "fast")
        & (benchmark_df["w"] == width)
        & (benchmark_df["h"] == height)
    ].copy()
    rgb2yuv_single_stream = rgb2yuv_single_stream.dropna(subset=["avg_ms", "h2d_ms", "d2h_ms"])
    if rgb2yuv_single_stream.empty:
        raise ValueError(f"No RGB2YUV GPU benchmark rows available at {width}x{height}")

    rgb2yuv_single_stream["single_stream_e2e_ms"] = (
        rgb2yuv_single_stream["avg_ms"] + rgb2yuv_single_stream["h2d_ms"] + rgb2yuv_single_stream["d2h_ms"]
    )
    single_stream_time = rgb2yuv_single_stream["single_stream_e2e_ms"].median()

    rgb2yuv_pipeline = tuning_df[
        (tuning_df["algo"] == "rgb2yuv")
        & (tuning_df["type"] == "stream")
        & (tuning_df["w"] == width)
        & (tuning_df["h"] == height)
    ].copy()
    rgb2yuv_pipeline = rgb2yuv_pipeline.dropna(subset=["streams", "time_ms"])
    rgb2yuv_pipeline["streams"] = rgb2yuv_pipeline["streams"].astype(int)
    rgb2yuv_pipeline = (
        rgb2yuv_pipeline.groupby("streams")["time_ms"]
        .median()
        .reindex(STREAM_COUNTS)
        .dropna()
        .reset_index()
    )
    if rgb2yuv_pipeline.empty:
        raise ValueError(f"No RGB2YUV pipeline rows available at {width}x{height}")

    labels = ["Single-stream end-to-end"] + [
        f"Pipeline {int(stream_count)} stream" + ("s" if int(stream_count) != 1 else "")
        for stream_count in rgb2yuv_pipeline["streams"]
    ]
    values = [single_stream_time] + rgb2yuv_pipeline["time_ms"].tolist()
    colors = ["#C44E52"] + ["#64B5CD"] * len(rgb2yuv_pipeline)

    apply_plot_style()
    plt.figure(figsize=(10.5, 5.8))
    bars = plt.bar(labels, values, color=colors, width=0.62)
    plt.ylabel("Average Time per Pass (ms)")
    plt.xlabel("Execution Mode")
    plt.title(f"RGB2YUV End-to-End Comparison ({width}x{height})")
    plt.xticks(rotation=15, ha="right")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.annotate(
        "H2D + Kernel + D2H",
        xy=(0, values[0]),
        xytext=(0.55, max(values) * 0.55),
        arrowprops={"arrowstyle": "->", "color": "#C44E52", "lw": 1.2},
        color="#C44E52",
        fontweight="bold",
    )

    if len(values) > 1:
        plt.annotate(
            "Chunked multi-stream pipeline",
            xy=(1, values[1]),
            xytext=(1.7, max(values) * 0.82),
            arrowprops={"arrowstyle": "->", "color": "#4C72B0", "lw": 1.2},
            color="#4C72B0",
            fontweight="bold",
        )

    return save_plot(output_directory, "rgb2yuv_end_to_end_comparison.png")


def generate_all_plots(
    benchmark_csv: Path,
    tuning_csv: Path,
    output_directory: Path,
    resolution: tuple[int, int] | None = None,
) -> list[Path]:
    benchmark_df = pd.read_csv(benchmark_csv)
    tuning_df = pd.read_csv(tuning_csv)

    return [
        plot_gaussian_method_ratio(benchmark_df, output_directory),
        plot_sobel_block_performance(tuning_df, output_directory, resolution=resolution),
        plot_rgb2yuv_stream_scaling(tuning_df, output_directory, resolution=resolution),
        plot_rgb2yuv_end_to_end_comparison(benchmark_df, tuning_df, output_directory, resolution=resolution),
        plot_gaussian_throughput_scaling(benchmark_df, output_directory),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate five clean plots for the Phase 3 report."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("outputs") / "benchmark_results_selected_fast.csv",
        help="Path to benchmark_results_selected_fast.csv",
    )
    parser.add_argument(
        "--tuning-csv",
        type=Path,
        default=Path("outputs") / "tuning_results.csv",
        help="Path to tuning_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "plots",
        help="Directory where the PNG plots will be written",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Optional resolution filter in the form WIDTHxHEIGHT, for example 2048x2048",
    )
    args = parser.parse_args()

    selected_resolution = None
    if args.resolution:
        try:
            width_text, height_text = args.resolution.lower().split("x", maxsplit=1)
            selected_resolution = (int(width_text), int(height_text))
        except ValueError as exc:
            raise SystemExit("--resolution must look like 2048x2048") from exc

    generated_paths = generate_all_plots(
        benchmark_csv=args.benchmark_csv,
        tuning_csv=args.tuning_csv,
        output_directory=args.output_dir,
        resolution=selected_resolution,
    )

    for output_path in generated_paths:
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
