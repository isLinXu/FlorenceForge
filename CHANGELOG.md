# Changelog

本项目所有显著变更均记录于此文件。

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) ，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/) 。

## [Unreleased]

### Added
- **依赖分层**：拆分 `requirements.txt` 为 `requirements-core.txt` / `requirements-optional.txt` / `requirements-dev.txt`，
  并将旧 `requirements.txt` 改为聚合入口，安装时可按需选择最小核心或全量依赖。
- **CHANGELOG.md**：开始按 Keep a Changelog 规范跟踪发布历史。
- **pre-commit hooks**：新增 `.pre-commit-config.yaml`，集成 black / isort / ruff / mypy 检查。
- **CLI 模块化**：`cli/main.py`（1543 行）拆分为三个文件 —— `cli/_helpers.py`
  （共享常量 `TASK_CONFIG_MAPPING`/`TASK_DESCRIPTIONS` 与图像/统计纯函数）、
  `cli/commands.py`（`run_inference_task` / `run_serve_task` / `run_eval_task` /
  `run_data_conversion` / `run_training_task` 等重型 handler），`main.py` 精简至 798 行，
  仅保留 argparse 解析、调度、doctor 诊断与 list/validate/generate。
  所有 handler 在 `main.py` 中回导，保证 `from florence_forge.cli.main import ...`
  历史导入路径与测试 monkeypatch 行为完全兼容。
- **设备一致性加固**：`BaseVLMBackend.generate()` / `forward()` 新增
  `_move_tensors_to_device()`，递归地将 `**kwargs` 中任意张量（如 `decoder_input_ids`）
  同步到模型设备，消除 CPU tensor + CUDA model 组合下的潜在崩溃。
- **评估 padding 正确性**：`MultiTaskEvaluator` 抽取 `_resolve_pad_token_id()`，
  回退构造 `Florence2Collator` 时使用模型真实 pad token id，避免硬编码 0 在
  非零 pad token 模型上导致 padding 错位。
- **架构解析器合并**：`ArchitectureResolver` 改为 `VLMBackendRegistry` 之上的薄壳，
  自动从注册表派生路由表，消除两套机制并发偏差的风险；保留旧 API 以向后兼容。
- **GitHub Actions Lint**：新增 `.github/workflows/lint.yml`，
  对 PR 自动跑 `ruff check` + `black --check` + `isort --check`。
- **v1 → v2 迁移指南**：`docs/MIGRATION_v1_to_v2.md` 阐明双训练栈现状、并行期、对齐计划与用户迁移路径。
- **MoE 实验模块文档**：`florence_forge/experimental/moe/README.md` 补充使用示例、设计动机、已知限制。

### Changed
- **.gitignore 强化**：忽略 `scripts/infer/results_*`、`scripts/infer/images_caption/`、
  `runs/`、`evaluation/`、`.benchmarks/` 以及 `*.onnx` 等推理/训练产物，避免大文件意外入库。
- **temp/ 中有价值素材**已提升到 `scripts/data-conversion/`（`convert_ocr_from_txt.py`、
  `example_ocr_data.txt`），原 `temp/` 仅作为本地草稿目录（已在 .gitignore 中）。

### Fixed
- （待补充）

### Removed
- （待补充）

---

## [1.0.0] - 2026-05-21

首个稳定版本。FlorenceForge 提供 Florence-2 多任务微调一站式框架。

### 核心特性

- **VLM 后端抽象**：基于 `BaseVLMBackend` 抽象与 `VLMBackendRegistry` 注册表，
  内置 4 个后端：Florence-2、PaliGemma、YouTuVL、Generic HuggingFace VLM。
- **13 种 Florence-2 任务**：图像描述（含 detailed/more_detailed）、目标检测、
  开放词汇检测、区域提议/分类/描述、OCR（含 with-region）、引用表达分割等。
- **训练引擎 v1**（默认）：单文件 `MultiTaskTrainer` 提供 FSDP / DeepSpeed / DDP、
  激活重计算 4 档、异步 checkpoint、梯度验证、内存监控、任务调度、LoRA 管理、
  模型合并、三平台监控集成（WandB / SwanLab / TensorBoard）。
- **训练引擎 v2**（重构中）：`trainer_refactored.py` + `training_loop.py` +
  `checkpoint_manager.py` 模块化拆分，职责更清晰。
- **数据管线**：`MultiTaskDataset`（多任务采样 + LRU 图像缓存按字节预算）、
  `DataFormatConverter`（YOLO/COCO/CSV/VOC/OCR → Florence-2 五种格式转换）、
  `MultiDatasetManager`（多数据集权重分配/优先级调度）。
- **评估体系**：基础评估、基准评测、统计分析、HTML/PDF 报告、并行评测、缓存、
  以及高级指标层（语义、多模态、鲁棒性、效率、检测、字幕）。
- **部署服务**：`InferenceEngine`（AMP / 批量 / 编译）、`ModelServer`（FastAPI REST）、
  `ModelExporter`（ONNX / TorchScript / SafeTensors）、`ModelOptimizer`（图优化）、
  `ModelQuantizer`（bnb-4bit/8bit、GPTQ、AWQ、dynamic-int8）。
- **配置体系**：Pydantic v2，字段级约束 + 交叉字段校验，未知字段告警而不报错。
- **CLI**：`florence_forge_cli` / `florence-forge` 单入口，覆盖
  `train` / `evaluate` / `infer` / `serve` / `convert-data` / `list-tasks` /
  `validate` / `generate-config` / `benchmark` / `merge-and-benchmark`。
- **安全 / 稳健性**：`safe_torch_load()` 默认 `weights_only=True`；
  Flash-Attention 在 MPS/CPU 场景的兼容补丁；CPU 回退仅对设备/精度错误生效。

### 文档与脚本

- `configs/` 提供 quick_start / production / distributed_training /
  paligemma_caption 等多种模板，以及 13 种任务示例配置。
- `scripts/` 覆盖训练 / 推理 / 数据转换 / 性能分析 / 冒烟测试。
- `docs/` 含架构图与多份深度分析报告。

---

## 版本约定

- **MAJOR**（X.0.0）：包含破坏性 API 变更或不兼容的配置调整。
- **MINOR**（0.X.0）：向后兼容的新功能、显著优化、新后端 / 任务支持。
- **PATCH**（0.0.X）：向后兼容的 bug 修复、文档勘误、依赖范围更新。

## 链接约定

- 每个版本段尾的 `[Unreleased]` 与 `[X.Y.Z]` 标题建议在发布时挂上 GitHub 比较链接：
  `https://github.com/florenceforge/florence-forge/compare/vX.Y.Z...HEAD`
