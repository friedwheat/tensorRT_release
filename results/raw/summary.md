# Experiment Summary

Generated at: 2026-05-12 00:03:27

## Benchmark Results

| runner | precision | batch_size | latency_mean_ms | throughput_qps |
| --- | --- | --- | --- | --- |
| onnxruntime |  | 18 | 7.6392073929309845 | 2356.265391702348 |
| onnxruntime |  | 18 | 7.167009748518467 | 2511.5076763668253 |
| pytorch | fp32 | 18 | 6.786159351468086 | 2652.457607867101 |
| tensorrt_python | fp16 | 18 | 23.22847057133913 | 774.9111137006853 |
| tensorrt_python | fp32 | 18 | 22.99423798918724 | 782.804805641495 |
| trtexec | fp16 | 1 | None | 3250.98 |
| trtexec | fp32 | 1 | None | 1187.77 |

## Accuracy Comparison

| name | max_abs_diff | mean_abs_diff | top1_agreement | top5_overlap |
| --- | --- | --- | --- | --- |
| pytorch_vs_onnx | 0.01584947109222412 | 0.001616181107237935 | 1.0 | 1.0 |
| pytorch_vs_trt_fp16 | 0.036550164222717285 | 0.003061165800318122 | 1.0 | 1.0 |
| pytorch_vs_trt_fp32 | 0.019327878952026367 | 0.0018910857615992427 | 1.0 | 1.0 |