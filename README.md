# S3WD
S3WD+GWB+动态阈值

## 动态配置示例

新增的 `configs/s3wd_airline_dynamic.yaml` 在原有基准配置基础上引入 `DYN`/`DRIFT`/`INCR` 三个分组，用于集中管理动态阈值、漂移检测与增量样本设置。需要启用某个组件时，将对应分组的 `enabled` 设为 `true` 并调整细粒度参数；若要关闭，直接将 `enabled` 设为 `false` 即可。

### DYN（动态阈值）

- `strategy`：`windowed_pso`（窗口化粒子群优化）或 `rule_based`（经验规则）。
- `step`：每隔多少个批次/迭代刷新一次阈值，取值 ≤0 时表示每轮都更新。
- `target_bnd`：希望保持的中间区域占比，动态搜索会对不足部分施加惩罚。
- 其余字段（`ema_alpha`、`median_window`、`keep_gap`、`window_size`、`stall_rounds`、`fallback_rule`）映射到 `DynamicAdaptConfig`，并在 `run_dynamic_thresholds` 中传递给阈值自适应算法，实现滚动平滑与兜底策略。【F:s3wdlib/dyn_threshold.py†L32-L220】【F:s3wdlib/dyn_threshold.py†L522-L585】

### DRIFT（漂移检测）

- `strategy` 支持 `kswin` 与 `adwin`，其余参数对齐 `DriftDetector` 构造函数。
- 加载 YAML 后，可通过 `DriftConfig.build_detector()` 快速实例化检测器，或借助 `DriftConfig.apply()` 在运行中动态调整阈值。【F:s3wdlib/drift.py†L1-L120】

### INCR（增量样本）

- `step`：增量样本的刷新节奏；当启用时，`run_dynamic_thresholds` 会在 `step` 未到达前跳过阈值更新。
- `buffer_size`：建议保留的窗口大小，可在未显式指定 `DYN.window_size` 时复用。
- `max_samples`：自定义上限，便于控制滑动窗口累计量。【F:s3wdlib/dyn_threshold.py†L82-L148】【F:s3wdlib/config_loader.py†L12-L120】

## 运行入口

```python
from s3wdlib.config_loader import load_yaml_cfg
from s3wdlib.dyn_threshold import run_dynamic_thresholds, DynamicAdaptConfig, IncrementalUpdateConfig
from s3wdlib.drift import DriftConfig

cfg = load_yaml_cfg("configs/s3wd_airline_dynamic.yaml")
dyn = DynamicAdaptConfig.from_mapping(cfg["DYN"])
incr = IncrementalUpdateConfig.from_mapping(cfg["INCR"])
drift = DriftConfig.from_mapping(cfg["DRIFT"])

# 迭代中按需调用
result = run_dynamic_thresholds(prob_levels, y, s3_params, dynamic=dyn, incremental=incr, iteration=step_idx)
detector = drift.build_detector()
```

通过 YAML 控制，可在不改动代码的情况下，精细调整动态模块的开关与超参数；`extract_vars` 亦会导出同名扁平键，便于遗留脚本复用。【F:s3wdlib/config_loader.py†L122-L216】
