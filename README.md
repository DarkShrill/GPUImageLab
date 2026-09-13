# Parallel & Distributed Systems — CUDA Image Filters (CPU vs GPU)

This project (developed in C/C++ on Google Colab using CUDA) reimplements and compares
the sequential (CPU) and parallel (GPU CUDA) versions of several basic image processing filters:

- Color Space Conversion: RGB → YUV
- Gaussian Blur (Standard 2D)
- Gaussian Blur (Separable)
- Sobel Edge Detection

The goal is to evaluate both correctness and performance across:
1) baseline CPU implementations
2) optimized CPU implementations
3) optimized CUDA GPU implementations


OBJECTIVES
- Implement the same image filters on CPU and GPU (CUDA)
- Measure and compare execution times and speedups
- Analyze Host↔Device transfer overhead
- Validate correctness of GPU outputs against CPU results

## 📊 Performance Results

The project evaluates CPU and CUDA implementations across multiple image sizes and configurations, measuring:

- execution time;
- CPU vs GPU speedup;
- Host-to-Device and Device-to-Host transfer overhead;
- CUDA block configuration performance;
- multi-stream scaling;
- processing throughput in MPix/s.

Detailed benchmark results and experimental analysis are available in the accompanying project paper and notebook.

## ⚙️ Performance Analysis

The benchmarking pipeline explores several GPU optimization strategies:

- baseline vs optimized CUDA implementations;
- 2D vs separable Gaussian convolution;
- CUDA block-size tuning;
- asynchronous multi-stream execution;
- end-to-end timing including memory transfers.

The included `generate_plot.py` script can generate performance plots directly from the benchmark CSV outputs.

IMPLEMENTED FILTERS

1) RGB → YUV
Per-pixel conversion from RGB (0–255) to YUV (BT.601 or equivalent standard).

2) Gaussian Blur (Standard 2D)
2D convolution using a Gaussian kernel (configurable size: 3x3, 5x5, 7x7, etc.).

3) Gaussian Blur (Separable)
Separable Gaussian implementation:
- horizontal 1D pass
- vertical 1D pass
This reduces computational complexity compared to the full 2D kernel.

4) Sobel Edge Detection
Computation of horizontal and vertical gradients (Gx, Gy) and edge magnitude.

# Dataset
## Kodak Image Dataset

To evaluate the image processing filters implemented in CUDA C++, we use the Kodak Image Dataset, a well-known benchmark dataset widely adopted in the image processing and computer vision literature.

The dataset consists of 24 uncompressed color images with a resolution of 768×512 pixels, featuring a wide variety of:

 - scene content (natural scenes, objects, people),

 - color distributions,

 - textures and edge structures.

These characteristics make the Kodak Image Dataset particularly suitable for evaluating:

 - RGB to YUV color space conversion,

 - Gaussian Blur (standard and separable implementations),

 - Sobel Edge Detection.

The same set of images is used across all experiments to ensure fair comparison, reproducibility, and consistent performance evaluation between different filtering approaches and implementations (CPU vs GPU).

Official dataset source:
http://r0k.us/graphics/kodak/
