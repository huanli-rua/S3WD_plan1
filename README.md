# S3WD

S3WD+GWB 的最小实验脚手架，配套了基于 YAML 的配置加载器以及流式动态组件。

## 配置文件

`configs/` 目录下提供了若干示例，其中 `s3wd_airline_dynamic.yaml` 展示了完整的分组写法。
YAML 结构划分为以下区块：

| 分组 | 作用 |
| --- | --- |
| `DATA`/`LEVEL`/`KWB`/`GWB`/`S3WD`/`PSO` | 传统静态训练阶段所需参数，与原始脚本保持一致 |
| `DYN` | 阈值自适应策略配置，例如窗口化 PSO 或规则法 |
| `DRIFT` | 概念漂移检测器参数 |
| `INCR` | 后验增量维护器（FAISS 缓存）的行为设置 |

加载器会自动将这些区块映射到 `s3wdlib.streaming` 中的 dataclass，方便后续直接构建运行实例：

```python
from s3wdlib.config_loader import load_yaml_cfg, extract_dynamic_configs
from s3wdlib.streaming import build_dynamic_components

cfg = load_yaml_cfg("configs/s3wd_airline_dynamic.yaml")
dyn_cfg, drift_cfg, incr_cfg = extract_dynamic_configs(cfg)
pso_params, detector, updater = build_dynamic_components(
    dyn_cfg,
    drift_cfg,
    incr_cfg,
    estimator_factory=make_estimator,  # 自行提供的构造函数
)
```

### DYN / DRIFT / INCR 字段对照

| 分组 | 关键字段 | 对应接口 |
| --- | --- | --- |
| `DYN` | `strategy`, `step`, `window_size`, `target_bnd`, `ema_alpha`, `median_window`, `keep_gap`, `fallback_rule`,<br>`gamma_last`, `stall_rounds` | `s3wdlib.streaming.DynamicLoopConfig` / `PSOParams` / `adapt_thresholds_windowed_pso` |
| `DRIFT` | `method`, `window_size`, `stat_size`, `significance`, `delta`, `cooldown`, `min_window_length` | `s3wdlib.drift.DriftDetector` |
| `INCR` | `buffer_size`, `cache_strategy`, `rebuild_interval`, `min_rebuild_interval`, `drift_shrink`, `immediate_rebuild_methods`, `enable_faiss_append`, `random_state` | `s3wdlib.incremental.PosteriorUpdater` |

示例 YAML (`configs/s3wd_airline_dynamic.yaml`) 已给出全部字段，可直接按需修改。

## 开启 / 关闭动态组件

- **开启动态策略**：在 YAML 中保留 `DYN`、`DRIFT`、`INCR` 块即可，对应字段会被加载并注入运行组件。
- **关闭单独模块**：删除或注释掉相应分组即可；加载器会自动跳过缺失的区块。例如去掉 `DRIFT` 就会禁用漂移检测，`build_dynamic_components` 将返回 `None`。
- **快速切换策略**：`DYN.strategy` 支持 `windowed_pso`（默认）和 `rule_based` 等自定义值；`DYN.step` 为每次重新评估的批大小，`DYN.target_bnd` 用于业务方设定的边界占比目标。

如需静态运行，只保留基础分组（`DATA`~`PSO`）即可，旧版 YAML 仍受支持。
