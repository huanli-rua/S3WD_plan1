# S3WD v02 · Reference Tuple 主线说明

本说明文档概述 v02 版本在原 S3WD-GWB 流程上的升级要点：

- **参考元组 (Reference Tuple)**：基于分桶样本，分别维护 Γ_pos/Ψ_neg 参考集合，结合 GWB 置信度挑选典型样本。
- **混合相似度**：使用数值 RBF（径向基函数）与类别匹配权重合成样本与参考元组的相关度。
- **批级小网格阈值搜索**：以期望成本或 Fβ 为目标，在护栏约束下搜索 (α, β)，并通过 EMA 平滑 + 步长限幅稳定阈值。
- **漂移闭环**：对接 PSI/TV/PosRate/性能四类指标，触发 S1/S2/S3 分级响应（调 σ、重建参考元组、重建 GWB 索引与收紧护栏）。
- **兼容旧主线**：通过 YAML 中的 `SWITCH.enable_ref_tuple / enable_pso` 可切换至 v02 主线或原 PSO 主线。

## 快速上手

1. 更新 `configs/s3wd_airline.yaml` 至文档所列的 v02 键位。
2. 运行 `notebooks/02_s3wd_gwb.ipynb`，Notebook 会依次完成数据加载、参考元组构建、流式主循环、漂移响应、可视化与 CSV 导出。
3. 输出文件默认存放于 `DATA.data_dir` 指定目录，包括：
   - `threshold_trace_v02.csv`
   - `window_metrics.csv`
   - `drift_events.csv`

## 模块一览

| 模块 | 作用 |
| --- | --- |
| `s3wdlib/bucketizer.py` | 分桶与回退逻辑 |
| `s3wdlib/ref_tuple.py` | 参考元组构建/合并 |
| `s3wdlib/similarity.py` | 混合相似度与集合相关度 |
| `s3wdlib/batch_measure.py` | 三域概率与批级指标 |
| `s3wdlib/threshold_selector.py` | 小网格阈值搜索与约束 |
| `s3wdlib/smoothing.py` | EMA 平滑 + 步长限幅 |
| `s3wdlib/drift_controller.py` | 漂移判级与分级响应 |

更多细节请参阅 Notebook 中的中文注释与日志。
