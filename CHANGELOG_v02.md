# v02 变更记录

- 新增 Reference Tuple / 混合相似度 主线，并保留原 PSO 主线切换开关。
- 引入批级三域概率 & 小网格阈值搜索，支持 Expected Cost / Fβ 两种目标。
- 增加 EMA 平滑 + 单窗步长限幅，保证阈值轨迹平稳。
- 构建漂移判级控制器，支持 S1/S2/S3 分级响应与护栏收紧。
- Notebook `02_s3wd_gwb.ipynb` 覆盖数据加载→训练资产→主循环→漂移闭环→导出全流程。
- YAML 增补 BUCKET / REF_TUPLE / SIMILARITY / MEASURE / SMOOTH / DRIFT / SWITCH 键位，缺键自动回退默认值。
