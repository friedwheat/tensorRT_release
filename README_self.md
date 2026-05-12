# TensorRT First Project

这是一个按 SOP 直接落地的 TensorRT 图像分类项目模板，默认使用 `ResNet50`，目标是把下面这条链路完整跑通：

`PyTorch -> ONNX -> TensorRT Engine -> Benchmark -> Output Validation -> Profiling -> Report`

这套代码优先面向 `Ubuntu / WSL2 Ubuntu + NVIDIA GPU + Docker`。如果你后面严格按这个 README 走，基本就能把 SOP 要求的“最小完成版”和“标准完成版”复现出来。

## 1. 项目结构

```text
tensorrt-first-project/
├─ README.md
├─ requirements.txt
├─ assets/
│  ├─ sample_images/
│  └─ figures/
├─ models/
├─ scripts/
│  ├─ build_and_benchmark.py
│  ├─ build_engines.sh
│  ├─ benchmark.sh
│  ├─ check_environment.py
│  ├─ collect_results.py
│  ├─ common.py
│  ├─ export_onnx.py
│  ├─ preprocess.py
│  ├─ run_onnx_infer.py
│  ├─ run_pytorch_infer.py
│  ├─ run_trt_infer.py
│  └─ validate_outputs.py
├─ results/
│  ├─ benchmark_results.csv
│  ├─ accuracy_compare.csv
│  ├─ nsys/
│  ├─ raw/
│  └─ report.md
└─ notes/
   ├─ issues.md
   └─ learning_log.md
```

## 2. 推荐环境

推荐直接使用 NVIDIA 官方容器，或者在 WSL2 Ubuntu 中手动装：

- Python 3.10+
- CUDA Toolkit
- TensorRT
- PyTorch
- ONNX Runtime GPU
- `trtexec`
- Nsight Systems

如果你想走最稳的路线，建议优先用 NVIDIA PyTorch / TensorRT 容器。

## 3. 安装依赖

先安装 Python 依赖。

如果你是 `Python 3.10 + CUDA 11.8 + RTX 4090`，当前 `requirements.txt` 已经内置了 `cu118` 索引，直接执行：

```bash
pip install -r requirements.txt
```

注意：

- `requirements.txt` 已经包含 `cu118` 索引，用来约束 `torch` / `torchvision` 的安装来源。
- `trtexec` 不会随着 `pip install -r requirements.txt` 自动可用。它通常由 NVIDIA TensorRT 官方安装包或官方容器提供。
- 如果你本机直接 `pip install tensorrt-cu11` 不顺利，不要硬怼，优先改用 NVIDIA 官方 TensorRT 安装包或容器。

### 3.1 不建议只靠 pip 安装的组件

下面这些组件建议单独安装，不要指望 `requirements.txt` 一步解决：

- `trtexec`
- TensorRT 系统库和开发工具
- `nsys`（Nsight Systems）

推荐方案 1：使用 NVIDIA 官方 TensorRT tar 包或 deb 包安装系统组件，然后单独安装 Python 依赖。

如果你已经把 TensorRT 解压到 `/usr/local/TensorRT-10.x.x.x`，可以这样加入环境变量：

```bash
export TENSORRT_ROOT=/usr/local/TensorRT-10.x.x.x
export PATH=$TENSORRT_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_ROOT/lib:$LD_LIBRARY_PATH
trtexec --version
```

推荐方案 2：直接使用 NVIDIA 官方容器。这是最稳的方式，尤其适合 benchmark 和 profiling。

### 3.2 一组可直接执行的本机安装命令

```bash
conda create -n trt310 python=3.10 -y
conda activate trt310

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如果你已经单独安装好了 TensorRT 系统包，但还没装 Python 绑定，也可以只安装这一组：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  numpy==1.26.4 \
  Pillow==10.4.0 \
  torch==2.3.1 \
  torchvision==0.18.1 \
  onnx==1.16.2 \
  onnxruntime-gpu==1.18.1 \
  cuda-python==11.8.7 \
  pandas==2.2.3
```

## 4. 准备样例图片

把 5 到 20 张测试图片放到：

```text
assets/sample_images/
```

建议统一是 `.jpg` 或 `.png`。

## 5. 阶段 0：检查环境

```bash
python scripts/check_environment.py
```

输出会写到：

- `results/raw/environment.json`

重点确认：

- `torch.cuda.is_available()` 为 `true`
- 能看到 GPU 名称
- 能读到 TensorRT 版本

## 6. 阶段 1：跑通 PyTorch baseline

```bash
python scripts/run_pytorch_infer.py \
  --input assets/sample_images \
  --output-prefix results/raw/pytorch
```

输出包括：

- `results/raw/pytorch_logits.npy`
- `results/raw/pytorch_predictions.json`
- `results/raw/pytorch_benchmark.json`

## 7. 阶段 2：导出 ONNX

```bash
python scripts/export_onnx.py \
  --model-name resnet50 \
  --batch-size 1 \
  --image-size 224 \
  --output models/model.onnx
```

默认会导出动态 batch 的 ONNX，后续无论输入图片数量是多少，都不会因为 batch 维度固定为 `1` 而报错。
如果你确实需要固定 batch 的 ONNX，再额外加：

```bash
--static-batch
```

输出包括：

- `models/model.onnx`
- `results/raw/onnx_export.json`

如果你重新导出了 ONNX，下面这些产物都应该重新生成：

- `models/model_fp32.engine`
- `models/model_fp16.engine`
- `results/raw/onnx_logits.npy`
- `results/raw/trt_fp32_logits.npy`
- `results/raw/trt_fp16_logits.npy`
- `results/raw/validation_*.json`

## 8. 阶段 3：构建 TensorRT Engine

先构建 FP32 和 FP16：

```bash
python scripts/build_and_benchmark.py build \
  --onnx-path models/model.onnx \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --build-fp16
```

也可以直接用 shell 包装脚本：

```bash
bash scripts/build_engines.sh
```

如果当前 ONNX 是动态 batch，脚本会自动为 `trtexec` 生成对应的 optimization profile。

输出包括：

- `models/model_fp32.engine`
- `models/model_fp16.engine`
- `results/raw/build_fp32.log`
- `results/raw/build_fp16.log`

## 9. 阶段 4：跑通 ONNX / TensorRT 推理

### 9.1 ONNX Runtime

```bash
python scripts/run_onnx_infer.py \
  --input assets/sample_images \
  --onnx-path models/model.onnx \
  --output-prefix results/raw/onnx
```

动态 batch ONNX 会直接按整批图片推理，不需要再手动控制输入图片数量。

### 9.2 TensorRT FP32

```bash
python scripts/run_trt_infer.py \
  --engine models/model_fp32.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp32
```

如果 engine 当前是固定 `batch=1` profile，脚本会自动把多张图片拆成多次单图推理，避免因为输入图片数量变化而报 shape mismatch。

### 9.3 TensorRT FP16

```bash
python scripts/run_trt_infer.py \
  --engine models/model_fp16.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp16
```

## 10. 阶段 5：做 benchmark

用 `trtexec` 跑 engine benchmark：

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

输出包括：

- `results/raw/trtexec_benchmark_fp32.json`
- `results/raw/trtexec_benchmark_fp16.json`
- `results/raw/benchmark_fp32.log`
- `results/raw/benchmark_fp16.log`

## 11. 阶段 6：做输出一致性验证

```bash
python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/onnx_logits.npy \
  --name pytorch_vs_onnx

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp32_logits.npy \
  --name pytorch_vs_trt_fp32

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp16_logits.npy \
  --name pytorch_vs_trt_fp16
```

输出包括：

- `results/raw/validation_pytorch_vs_onnx.json`
- `results/raw/validation_pytorch_vs_trt_fp32.json`
- `results/raw/validation_pytorch_vs_trt_fp16.json`

## 12. 阶段 7：汇总结果

```bash
python scripts/collect_results.py
```

输出包括：

- `results/benchmark_results.csv`
- `results/accuracy_compare.csv`
- `results/raw/summary.md`

## 13. 阶段 8：做 Profiling

建议先对 FP16 engine 做一次 `nsys`：

```bash
nsys profile \
  -o results/nsys/trt_fp16_profile \
  trtexec \
  --loadEngine=models/model_fp16.engine \
  --shapes=input:1x3x224x224 \
  --warmUp=500 \
  --iterations=100 \
  --duration=10
```

如果你是直接对动态 ONNX 做 profiling，而不是对已经保存好的 engine 做 profiling，优先使用：

```bash
trtexec \
  --onnx=models/model.onnx \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:1x3x224x224 \
  --maxShapes=input:1x3x224x224 \
  --fp16
```

然后把观察结果整理到 `results/report.md`。

## 14. 一键走完整链路

如果你只是想先把主线一次跑通：

```bash
python scripts/check_environment.py

python scripts/run_pytorch_infer.py \
  --input assets/sample_images \
  --output-prefix results/raw/pytorch

python scripts/export_onnx.py

python scripts/build_and_benchmark.py build --build-fp16

python scripts/run_onnx_infer.py \
  --input assets/sample_images \
  --output-prefix results/raw/onnx

python scripts/run_trt_infer.py \
  --engine models/model_fp32.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp32

python scripts/run_trt_infer.py \
  --engine models/model_fp16.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp16

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/onnx_logits.npy \
  --name pytorch_vs_onnx

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp32_logits.npy \
  --name pytorch_vs_trt_fp32

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp16_logits.npy \
  --name pytorch_vs_trt_fp16

python scripts/build_and_benchmark.py benchmark --benchmark-fp16

python scripts/collect_results.py
```

## 15. 改动后需要重新跑哪些命令

如果你已经有旧的静态 ONNX 或旧 engine，在本次代码修改后，建议至少重新执行这一组：

```bash
python scripts/export_onnx.py \
  --model-name resnet50 \
  --batch-size 1 \
  --image-size 224 \
  --output models/model.onnx

python scripts/build_and_benchmark.py build \
  --onnx-path models/model.onnx \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --build-fp16

python scripts/run_onnx_infer.py \
  --input assets/sample_images \
  --onnx-path models/model.onnx \
  --output-prefix results/raw/onnx

python scripts/run_trt_infer.py \
  --engine models/model_fp32.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp32

python scripts/run_trt_infer.py \
  --engine models/model_fp16.engine \
  --input assets/sample_images \
  --output-prefix results/raw/trt_fp16

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/onnx_logits.npy \
  --name pytorch_vs_onnx

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp32_logits.npy \
  --name pytorch_vs_trt_fp32

python scripts/validate_outputs.py \
  --baseline results/raw/pytorch_logits.npy \
  --candidate results/raw/trt_fp16_logits.npy \
  --name pytorch_vs_trt_fp16

python scripts/build_and_benchmark.py benchmark \
  --onnx-path models/model.onnx \
  --input-name input \
  --batch-size 1 \
  --image-size 224 \
  --benchmark-fp16

python scripts/collect_results.py
```

## 15. 你最后要交付什么

至少保证这些文件存在并有内容：

- `models/model.onnx`
- `models/model_fp32.engine`
- `models/model_fp16.engine`
- `results/benchmark_results.csv`
- `results/accuracy_compare.csv`
- `results/report.md`

## 16. 推荐面试表达

可以直接改成你自己的口吻：

> 基于 PyTorch 完成图像分类模型的 ONNX 导出与 TensorRT 部署，构建 FP32 / FP16 推理 engine，并在统一输入条件下对比延迟、吞吐和输出一致性；结合 profiling 工具分析推理链路瓶颈，形成完整的部署、测试与优化闭环。

## 17. 参考文档

- [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [TensorRT trtexec Command-Line Programs](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)
- [TensorRT Benchmarking Guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html)
