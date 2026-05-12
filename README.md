# TensorRT Release

一个尽量易懂、可复现的 TensorRT 图像分类示例项目。

这个仓库演示了如何把一个 `torchvision` 预训练模型从 `PyTorch` 导出到 `ONNX`，再构建为 `TensorRT Engine`，并完成推理、性能测试和输出一致性验证。默认模型是 `ResNet50`，默认任务是图像分类。

适合这些人：

- 第一次接触 `TensorRT`，想看一条完整部署链路的人
- 想把 `PyTorch -> ONNX -> TensorRT` 跑通的人
- 需要一个可直接改造、可继续扩展的实验模板的人

## 项目目标

本项目覆盖下面这条常见部署流程：

```text
PyTorch -> ONNX -> TensorRT Engine -> Inference -> Benchmark -> Validation -> Report
```

你可以把它理解成一个“最小但完整”的 TensorRT 入门项目：

- 有基线：`PyTorch`
- 有中间格式：`ONNX`
- 有部署产物：`TensorRT Engine`
- 有结果对比：延迟、吞吐、一致性
- 有实验输出：`results/` 下的 CSV、JSON、Markdown

## 你能从这个仓库获得什么

- 一套能直接运行的 Python 脚本
- 一个默认支持动态 batch 的 ONNX 导出流程
- FP32 / FP16 TensorRT engine 构建与基准测试命令
- PyTorch、ONNX Runtime、TensorRT Python Runtime 的输出一致性验证
- 一份已经生成好的实验结果示例

## 项目结构

```text
tensorRT_release/
├─ README.md
├─ README_self.md
├─ requirements.txt
├─ assets/
│  ├─ figures/
│  └─ sample_images/
├─ notes/
│  ├─ handoff_2026-05-12.md
│  ├─ issues.md
│  └─ learning_log.md
├─ results/
│  ├─ accuracy_compare.csv
│  ├─ benchmark_results.csv
│  ├─ report.md
│  └─ raw/
└─ scripts/
   ├─ benchmark.sh
   ├─ build_and_benchmark.py
   ├─ build_engines.sh
   ├─ check_environment.py
   ├─ collect_results.py
   ├─ common.py
   ├─ export_onnx.py
   ├─ preprocess.py
   ├─ run_onnx_infer.py
   ├─ run_pytorch_infer.py
   ├─ run_trt_infer.py
   └─ validate_outputs.py
```

说明：

- `assets/sample_images/` 放测试图片
- `scripts/` 放完整实验脚本
- `results/` 放汇总结果
- `results/raw/` 放原始 JSON、日志和中间结果
- `models/` 目录会在导出 ONNX 或构建 engine 时自动创建

## 环境要求

推荐环境：

- Linux / WSL2 Ubuntu
- NVIDIA GPU
- Python 3.10+
- CUDA 11.8
- TensorRT 10.x

本仓库当前已经验证过的一组环境如下：

- 日期：2026-05-11 到 2026-05-12
- Python：`3.10.8`
- GPU：`NVIDIA GeForce RTX 4090`
- CUDA：`11.8`
- PyTorch：`2.3.1+cu118`
- ONNX：`1.16.2`
- ONNX Runtime：`1.18.1`
- TensorRT：`10.13.0.35`

如果你想少踩坑，优先建议：

1. 使用 NVIDIA 官方容器
2. 或者先安装好系统级 TensorRT 与 `trtexec`，再安装 Python 依赖

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已经包含了 `cu118` 的 PyTorch 索引。

### 2. 安装系统级组件

下面这些通常不能只靠 `pip` 解决：

- `TensorRT` 系统库
- `trtexec`
- `Nsight Systems`（如果你要做 profiling）

如果你是手动安装 TensorRT，可以像这样配置环境变量：

```bash
export TENSORRT_ROOT=/usr/local/TensorRT-10.13.0.35
export PATH=$TENSORRT_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH

trtexec --version
```

## 快速开始

如果你只想先把主流程跑通，按照下面顺序执行即可。

### 1. 检查环境

```bash
python scripts/check_environment.py
```

输出会保存到：

```text
results/raw/environment.json
```

### 2. 运行 PyTorch 基线

```bash
python scripts/run_pytorch_infer.py \
  --input assets/sample_images \
  --output-prefix results/raw/pytorch
```

会生成：

- `results/raw/pytorch_logits.npy`
- `results/raw/pytorch_predictions.json`
- `results/raw/pytorch_benchmark.json`

### 3. 导出 ONNX

```bash
python scripts/export_onnx.py \
  --model-name resnet50 \
  --batch-size 1 \
  --image-size 224 \
  --output models/model.onnx
```

默认行为：

- 导出动态 batch ONNX
- 输入名默认是 `input`
- 输出名默认是 `logits`

如果你明确需要固定 batch，可以加上：

```bash
--static-batch
```

### 4. 构建 TensorRT Engine

```bash
python scripts/build_and_benchmark.py build \
  --onnx-path models/model.onnx \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --build-fp16
```

或者直接使用封装脚本：

```bash
bash scripts/build_engines.sh
```

默认会生成：

- `models/model_fp32.engine`
- `models/model_fp16.engine`
- `results/raw/build_fp32.log`
- `results/raw/build_fp16.log`

### 5. 运行 ONNX Runtime 推理

```bash
python scripts/run_onnx_infer.py \
  --input assets/sample_images \
  --onnx-path models/model.onnx \
  --provider auto \
  --output-prefix results/raw/onnx
```

### 6. 运行 TensorRT Python Runtime 推理

FP32:

```bash
python scripts/run_trt_infer.py \
  --engine models/model_fp32.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp32
```

FP16:

```bash
python scripts/run_trt_infer.py \
  --engine models/model_fp16.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp16
```

### 7. 用 `trtexec` 做 benchmark

```bash
python scripts/build_and_benchmark.py benchmark \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --benchmark-fp16
```

或者：

```bash
bash scripts/benchmark.sh
```

### 8. 验证输出一致性

PyTorch vs ONNX:

```bash
python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/onnx_logits.npy \
  --name pytorch_vs_onnx
```

PyTorch vs TensorRT FP32:

```bash
python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp32_logits.npy \
  --name pytorch_vs_trt_fp32
```

PyTorch vs TensorRT FP16:

```bash
python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp16_logits.npy \
  --name pytorch_vs_trt_fp16
```

### 9. 汇总结果

```bash
python scripts/collect_results.py
```

汇总产物：

- `results/benchmark_results.csv`
- `results/accuracy_compare.csv`
- `results/raw/summary.md`

## 一条完整复现命令链

如果你想从头完整跑一遍，可以按这个顺序：

```bash
python scripts/check_environment.py
python scripts/run_pytorch_infer.py --input assets/sample_images --output-prefix results/raw/pytorch
python scripts/export_onnx.py --model-name resnet50 --batch-size 1 --image-size 224 --output models/model.onnx
python scripts/build_and_benchmark.py build --onnx-path models/model.onnx --input-name input --batch-size 1 --image-size 224 --build-fp16
python scripts/run_onnx_infer.py --input assets/sample_images --onnx-path models/model.onnx --provider auto --output-prefix results/raw/onnx
python scripts/run_trt_infer.py --engine models/model_fp32.engine --input assets/sample_images --output-prefix results/raw/trt_fp32
python scripts/run_trt_infer.py --engine models/model_fp16.engine --input assets/sample_images --output-prefix results/raw/trt_fp16
python scripts/build_and_benchmark.py benchmark --input-name input --batch-size 1 --image-size 224 --benchmark-fp16
python scripts/validate_outputs.py --baseline results/raw/pytorch_logits.npy --candidate results/raw/onnx_logits.npy --name pytorch_vs_onnx
python scripts/validate_outputs.py --baseline results/raw/pytorch_logits.npy --candidate results/raw/trt_fp32_logits.npy --name pytorch_vs_trt_fp32
python scripts/validate_outputs.py --baseline results/raw/pytorch_logits.npy --candidate results/raw/trt_fp16_logits.npy --name pytorch_vs_trt_fp16
python scripts/collect_results.py
```

## 当前仓库内已附带的示例结果

下面这些结果来自仓库当前保存的实验输出：

### Benchmark 摘要

| Backend | Precision | Batch Size | Mean Latency (ms) | Throughput (qps) |
| --- | --- | --- | --- | --- |
| PyTorch | FP32 | 18 | 6.7862 | 2652.46 |
| ONNX Runtime | CUDA | 18 | 7.6392 | 2356.27 |
| TensorRT Python Runtime | FP32 | 18 | 22.9942 | 782.80 |
| TensorRT Python Runtime | FP16 | 18 | 23.2285 | 774.91 |
| `trtexec` | FP32 | 1 | N/A | 1187.77 |
| `trtexec` | FP16 | 1 | N/A | 3250.98 |

### 输出一致性摘要

| Compare | Max Abs Diff | Mean Abs Diff | Top-1 Agreement | Top-5 Overlap |
| --- | --- | --- | --- | --- |
| PyTorch vs ONNX | 0.015849 | 0.001616 | 1.0 | 1.0 |
| PyTorch vs TensorRT FP32 | 0.019328 | 0.001891 | 1.0 | 1.0 |
| PyTorch vs TensorRT FP16 | 0.036550 | 0.003061 | 1.0 | 1.0 |

说明：

- 这些数字只是当前实现与当前环境下的一次实验结果
- `TensorRT Python Runtime` 与 `trtexec` 的统计口径不同，不能直接横向比较
- 当前 Python 版 TensorRT 推理路径更偏“教学与验证”，不是极致优化版本

## 一些重要设计说明

### 1. ONNX 默认是动态 batch

这意味着：

- 多张图片输入更自然
- 不容易因为导出时写死 batch=1 而报 shape 错误
- 构建 TensorRT engine 时，脚本会自动补 `--minShapes/--optShapes/--maxShapes`

### 2. 即使 engine 只接受 batch 1，也能处理多张图片

`run_onnx_infer.py` 和 `run_trt_infer.py` 都做了兼容：

- 如果模型或 engine 只能接收 `batch=1`
- 但你输入了多张图片
- 脚本会自动拆成多次单张推理，再把结果拼起来

这样更适合初学者直接上手。

### 3. 重新导出 ONNX 后，建议重新构建 engine

如果你重新执行了 `export_onnx.py`，建议同时重新生成下面这些文件：

- `models/model_fp32.engine`
- `models/model_fp16.engine`
- `results/raw/onnx_logits.npy`
- `results/raw/trt_fp32_logits.npy`
- `results/raw/trt_fp16_logits.npy`
- `results/raw/validation_*.json`

## 常见问题

### `trtexec` 找不到

说明系统级 TensorRT 没装好，或者 `PATH` 没配好。先确认：

```bash
trtexec --version
```

### ONNX / TensorRT 报 shape 不匹配

常见原因：

- ONNX 是固定 batch 导出的
- engine 构建时使用的 shape 与当前输入不一致

优先解决方式：

1. 重新导出动态 batch ONNX
2. 重新构建 TensorRT engine

### 为什么 TensorRT Python Runtime 目前比 PyTorch 慢

这不是 TensorRT 本身一定更慢，而是因为当前仓库里的 Python runtime 实现更偏演示性质，包含了显式的 host/device 数据搬运与 Python 层控制逻辑。更适合把链路跑通、验证输出和理解流程，不代表最终部署上限。

## 后续可以继续扩展

- 支持 `INT8`
- 接入更严格的 profiling
- 加入 C++ Runtime 版本
- 改造成检测、分割等其他视觉任务
- 接入 Triton Inference Server

## 致谢

这个项目默认使用 `torchvision` 预训练模型与常见图像分类预处理流程，目的是帮助更多人低门槛理解 TensorRT 的基本使用方式。

如果这个仓库对你有帮助，欢迎在此基础上继续扩展。
