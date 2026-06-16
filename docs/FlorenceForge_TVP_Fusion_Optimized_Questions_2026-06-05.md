# FlorenceForge x Thinking with Visual Primitives 融合评审报告（优化版）

> 分析日期: 2026-06-05  
> 项目基线: FlorenceForge 当前工作区源码  
> 参考报告: `FlorenceForge x Thinking with Visual Primitives 融合可行性分析`  
> 目标: 在原报告基础上修正表达、补齐源码核验、明确阶段门槛，并新增可转化为研发 issue 的问题清单

---

## 0. 修订摘要

原报告的核心判断是正确的：FlorenceForge 的 Florence-2 多任务训练管线与 TVP 的视觉原语思想存在明显互补。但原报告更像“方案设想”，缺少几类决策前必须回答的问题：当前源码是否已有承载点、VP 任务如何注册、数据与评估如何闭环、训练栈是否足够稳定、MVP 到专家蒸馏/RL 的阶段门槛是什么。

本优化版做了五类调整：

1. **明确状态**: TVP/VP 在本报告成稿时仍是融合研究方案；本次后续实现已补入 Layer 1 MVP 的 VP task、`VisualPrimitiveConverter`、VP parser、指标接入，以及真实 Florence 权重 + COCO128 YOLO 标注的一步训练 smoke 和 LoRA smoke，GRPO/RM 仍未实现。
2. **修正口径**: 当前 `FLORENCE2_TASKS` 是 14 个任务，不是原报告中的 13 个。
3. **修复格式**: 原报告中多个 Markdown 表格被 `<|box|>`、`<|ref|>` 里的竖线截断，本版在表格中改用“ref + box”等描述。
4. **降低结论强度**: 从“高度可行”调整为“条件高度可行”。Layer 1 MVP 可推进；Layer 2/3 需等数据、评估和训练栈稳定后再开。
5. **新增问题清单**: 新增 P0/P1/P2 问题矩阵，覆盖模型、数据、训练、评估、部署、产品价值和论文价值。

结论先行：建议先做 **Layer 1 VP 输出格式 MVP**，但不要直接进入专家蒸馏或 RL。第一阶段的成功标准不是“模型变聪明”，而是证明 Florence-2 能稳定学习、生成、解析、评估视觉原语格式，并且不破坏现有 14 个任务。

---

## 1. 优化后的核心结论

### 1.1 可行性判定

| 层级 | 原报告判断 | 优化后判断 | 推荐动作 |
| --- | --- | --- | --- |
| Layer 1: VP token + 数据格式 + task 扩展 | 近期可做 | **可做，但需先补 task registry、converter、parser、评估闭环** | 立即做 MVP |
| Layer 2: Box/Point 专家 + OPD 蒸馏 | 中期可做 | **暂缓，先证明 VP 格式比 Florence-2 原生 loc 格式有增益** | MVP 通过后再开 |
| Layer 3: GRPO + Reward Model | 远期可做 | **研究项，不应进入近期工程路线** | 等 reward、baseline、数据规模齐备 |

### 1.2 推荐改写后的总判断

FlorenceForge 与 TVP 的融合是一个值得推进的研究工程方向，但它不是单纯“加几个 token”的改造。真正的风险集中在四个闭环：

1. **格式闭环**: VP token 能否被 tokenizer、decoder、parser、postprocessor 一致处理。
2. **数据闭环**: COCO/YOLO/VOC 等标注能否可靠转换为坐标归一化后的 VP 训练样本。
3. **评估闭环**: 现有指标能否比较 Florence 原生 loc 输出与 VP 输出，而不是只统计格式正确率。
4. **训练闭环**: 当前训练栈能否承载新任务、混合任务、LoRA adapter 切换、缓存和端到端 smoke test。

因此，本项目不应把目标写成“复制 TVP”，而应写成：

> 在 Florence-2 encoder-decoder 架构上，验证显式视觉原语输出是否能提升定位、计数、空间关系和可解释性，并将其封装为 FlorenceForge 的可选增强任务族。

---

## 2. 当前源码核验补充

| 核验点 | 当前状态 | 影响 |
| --- | --- | --- |
| 任务数量 | `florence_forge/core/tasks.py` 当前定义 14 个 Florence-2 任务（见 22-159 行） | 原报告“13 种任务”需修正 |
| VP 相关实现 | 本次已新增 VP task、VP converter、VP parser、检测指标接入；Reward Model、GRPO trainer 未实现 | Layer 1 MVP 已启动，Layer 2/3 仍应保持实验规划状态 |
| 数据转换格式 | `DataFormatConverter` 当前输出 Florence-2 JSON suffix，如 `{"<OD>": {"bboxes": ..., "labels": ...}}`（见 `converter.py` 189-205 行） | 不能直接产生 VP 思维链 |
| task 校验 | `MultiTaskDataset._validate_configs()` 使用固定任务注册表校验（见 `dataset.py` 154-164 行）；本次已将 `OD_VP`、`COUNT_VP`、`PHRASE_GROUNDING_VP` 注册进任务表 | VP 任务已可被 dataset 加载 |
| Florence-2 编码约束 | `Florence2Backend.encode_with_task()` 说明 task token 必须独占 text（见 `florence2_backend.py` 128-136 行） | 推理模板不能简单拼到 prompt；应放在 answer/suffix 侧 |
| tokenizer 扩展 | 本次已新增 `ModelConfig.enable_visual_primitives`，开启后 Florence-2 backend 会添加 VP token 并 resize embedding | 默认关闭，不影响现有训练/推理 |
| 检测指标解析 | 本次已新增 VP parser，并在 `DetectionMetrics` 中作为 fallback 解析 VP ref/box 输出 | VP 输出已可进入现有 detection precision/recall/mAP 流程 |
| 真实训练 smoke | 本次已新增 `scripts/smoke/real_florence_vp_training_smoke.py`，并在 2026-06-05 用本机 Florence 权重和 COCO128 YOLO 标注完成 1 step MPS 训练；参数切片 smoke: `final_loss=6.34699`，`trainable_param_delta_norm=0.00156`；LoRA smoke: `final_loss=5.95749`，`trainable_param_delta_norm=0.00141`；8 step JSON adapter smoke: `final_loss=2.33649`；12 step `loc_tokens` adapter smoke: `final_loss=2.76191` | 已证明真实权重、真实数据、VP token resize、dataset/collator、PEFT LoRA、forward/backward/optimizer step 链路可跑通；短训生成仍偏向 Florence 原生 `<loc_*>`，本次进一步新增结构化 VP decoder，将原生 `label<loc_*>` 确定性包装为 `ref + box` 证据链，使解析、评估和可视化可先稳定落地 |
| VP box 格式 | 已支持 JSON 坐标和 Florence 原生 `<loc_*>` token 两种 VP box 载荷；`VisualPrimitiveParser` 和 VP metrics 均可解析两者 | `loc_tokens` 更贴合 Florence 先验，适合作为下一阶段默认实验格式；但 wrapper 生成仍需额外约束 |
| 结构化 VP 解码 | 本次新增 `StructuredVisualPrimitiveDecoder` 与 `FlorenceNativeDetectionParser`，可将 `zebra<loc_0><loc_51><loc_690><loc_938>` 转成 `<\|ref\|>zebra<\|/ref\|><\|box\|><loc_0><loc_51><loc_690><loc_938><\|/box\|>`；并已接入 `VisualPrimitiveDetectionMetrics` 与 CLI `infer --structured-vp-decode` JSON 输出 | 当前最佳工程路径是“模型负责定位，decoder 负责 VP 结构化”；这不等同于模型已内化 VP wrapper，但足以让 Florence-VP 的证据链评估和 bad-case 可视化先投入使用 |
| 训练完备性审计 | 本次新增 `VPTrainingAuditThresholds`、`build_vp_training_audit()`、`scripts/experiments/audit_florence_vp_training.py` 与 `scripts/experiments/run_florence_vp_training_experiment.py`，可从训练 summary 和推理 summary 生成 gate report，并自动补 adapter/base baseline 推理对照 | 将“方案是否完备”变成可执行判断：raw wrapper 内化、structured 可用性、decoder 依赖、baseline 缺失、样本规模不足都会被显式标记 |
| 统一训练器 | ~~`UnifiedMultiTaskTrainer`~~ 已于 2026-06-06 删除（为不可运行的空桩，缺关键 import，实例化即 `NameError`）。当前训练入口为 v1 `MultiTaskTrainer`（默认）和 v2 `trainer_refactored.MultiTaskTrainer`（模块化） | VP 应挂载在 v1 `MultiTaskTrainer` 或 v2 训练栈上 |

---

## 3. 优化后的任务映射

| FlorenceForge 任务 | VP 适配方式 | 收益预期 | 备注 |
| --- | --- | --- | --- |
| `OD` | ref + box | 高 | MVP 首选，标注和指标最容易闭环 |
| `OPEN_VOCABULARY_DETECTION` | ref + box | 高 | 需处理开放词汇别名与 label canonicalization |
| `CAPTION_TO_PHRASE_GROUNDING` | ref + box | 高 | 与 Reference Gap 最直接相关 |
| `DENSE_REGION_CAPTION` | 多组 ref + box + text | 中高 | 输出更长，需控制 token budget |
| `OCR_WITH_REGION` | ref/text + box | 中高 | 可提升文字区域可解释性 |
| `REGION_PROPOSAL` | box list | 中高 | 可作为 box expert 冷启动数据 |
| `REGION_TO_CATEGORY` | 输入 box，输出 category | 中 | VP 更多用于输入约束，而非输出 |
| `REGION_TO_DESCRIPTION` | 输入 box，输出 description | 中 | 适合评估区域描述是否被 box 锚定 |
| `REGION_TO_SEGMENTATION` | box/point/polygon/mask | 不确定 | 单点不足以表达 mask，需另设格式 |
| `REFERRING_EXPRESSION_SEGMENTATION` | ref + point/polygon/mask | 不确定 | TVP point 思路可借鉴，但不应替代分割标注 |
| `CAPTION` | 可选 box evidence | 低中 | 容易增加输出长度，需验证收益 |
| `DETAILED_CAPTION` | 可选多 box evidence | 中 | 适合可解释 caption，但未必提升指标 |
| `MORE_DETAILED_CAPTION` | 可选多 box evidence | 中 | 需防止思维链过长导致训练不稳定 |
| `OCR` | 可选 text region grounding | 低中 | 若无区域标注，收益有限 |

建议 MVP 只覆盖三类任务：

1. `OD_VP`: 检测输出 VP box。
2. `COUNT_VP`: 从多实例 box 得出 count。
3. `PHRASE_GROUNDING_VP`: 短语到 box grounding。

这三类任务能同时验证格式生成、坐标归一化、计数推理和 Reference Gap 缓解，不会过早引入分割/RL 的复杂性。

---

## 4. 新增问题清单（按优先级）

> 本次实现已覆盖 VP-P0-01、VP-P0-02、VP-P0-04、VP-P0-05、VP-P0-06、VP-P0-08 的 Layer 1 最小闭环；VP-P0-03 通过”task prompt 保持原生、VP 内容放入 suffix”规避；VP-P0-07 仍建议使用 v1 `MultiTaskTrainer` 或 v2 训练栈（~~`UnifiedMultiTaskTrainer`~~ 已删除）。

### 4.1 P0：启动 MVP 前必须解决

| ID | 新增问题 | 为什么重要 | 建议处理 |
| --- | --- | --- | --- |
| VP-P0-01 | 新增 VP 任务注册后，`MultiTaskDataset` 仍会校验失败 | 当前 task registry 是固定字典，`OD_VP` 不在其中 | 增加 `VISUAL_PRIMITIVE_TASKS` 并合并进注册/校验入口 |
| VP-P0-02 | VP token 添加与 model embedding resize 的时机未定义 | 只改 tokenizer 不 resize embedding 会导致新 token 无可训练参数 | 在 backend load 完 model + processor 后统一执行 token add/resize |
| VP-P0-03 | Florence-2 task token 必须独占 text，与“推理模板 prompt”冲突 | `encode_with_task()` 已明确不能把 task token 与自然语言拼接 | 将 reasoning template 放进 answer/suffix，而不是 processor text |
| VP-P0-04 | 现有 converter 输出 Florence JSON，不输出 VP 格式 | 没有训练数据，模型无法学习 VP | 新增 `VisualPrimitiveConverter`，先支持 COCO/YOLO OD |
| VP-P0-05 | 现有检测指标无法解析 VP 输出 | 如果 evaluator 读不懂 VP，只能评格式，无法评精度 | 新增 `VisualPrimitiveParser`，解析 ref/box/point 到标准结构 |
| VP-P0-06 | 坐标归一化/反归一化规范缺失 | `[0,999]` 坐标与图像实际尺寸、Florence loc 坐标可能混淆 | 建立 `normalize_bbox()`、`denormalize_bbox()` 单测，覆盖边界值 |
| VP-P0-07 | Unified trainer 当前不宜作为 VP 承载点 | 最小实例化报 `_create_accelerator` 缺失；后续还有未导入名和 `output_dir` 顺序风险 | MVP 先挂稳定的 `MultiTaskTrainer`，并单独修复统一训练器 |
| VP-P0-08 | 缺少 VP 格式 golden tests | 格式一旦漂移，训练和评估会静默失真 | 为 token、parser、converter、dataset、metrics 增加最小样例测试 |

### 4.2 P1：MVP 期间必须回答

| ID | 新增问题 | 为什么重要 | 建议处理 |
| --- | --- | --- | --- |
| VP-P1-01 | VP 相比 Florence 原生 loc 格式是否真的更好 | 如果只是换格式，没有收益，工程复杂度不成立 | 做 baseline vs VP ablation：OD mAP、count acc、phrase grounding acc |
| VP-P1-02 | LoRA 是否足以学习新 token embedding | LoRA 通常不训练 embedding；新 token 可能学不动 | 明确是否训练 `embed_tokens`/`lm_head`，并记录新 token embedding norm |
| VP-P1-03 | 多任务 batch 的 adapter 切换可能只看第一个 task | 训练器部分路径会取 `task_types[0]`，混合 VP/非 VP 时可能误切 adapter | MVP 阶段先按任务分 batch，或在 trainer 中支持 per-task loss 分组 |
| VP-P1-04 | VP 输出长度会挤占 generation budget | 多 box + reasoning template 可能超过 `max_new_tokens` | 为每类 VP task 单独配置 token budget 和截断策略 |
| VP-P1-05 | 计数任务如何处理遮挡、小目标、重复框 | count 依赖检测召回，检测错误会放大成计数错误 | 在评估中分离 detection recall 与 final count accuracy |
| VP-P1-06 | 开放词汇 label 如何规范化 | `person`、`people`、`man` 会影响 ref 匹配和 reward | 建立 label canonicalization 和 synonym map |
| VP-P1-07 | 空标注/无目标样本如何表达 | 只训练有框样本会让模型过度输出目标 | 设计 `no object` VP 格式和负样本比例 |
| VP-P1-08 | 推理后处理是否保持向后兼容 | 现有用户可能依赖 Florence 原始 JSON/loc 输出 | VP 作为 opt-in task，不改变原任务输出 |

### 4.3 P2：专家蒸馏/RL 前再解决

| ID | 新增问题 | 为什么重要 | 建议处理 |
| --- | --- | --- | --- |
| VP-P2-01 | Box Expert 与 Point Expert 是否真的需要拆分 | Florence-2 小模型容量有限，专家拆分可能增加维护成本 | 先用单模型 SFT 做上限，再决定是否专家化 |
| VP-P2-02 | Point primitive 是否适合 FlorenceForge 的现有任务 | 当前主任务多是 box/segmentation，maze/path 数据不在框架主场景 | Point Expert 暂不进入第一阶段 |
| VP-P2-03 | Reward Model 的 truth source 从哪里来 | 无稳定 parser/metrics 时，RM 会奖励错格式或错坐标 | 先让 deterministic metrics 成熟，再做 RL |
| VP-P2-04 | GRPO 训练成本是否匹配 Florence-2 小模型收益 | RL 工程成本高，可能不如高质量 SFT 数据 | 设定“只有 SFT plateau 后才开 RL”的门槛 |
| VP-P2-05 | OPD 蒸馏是否会丢失原始 Florence 能力 | 统一模型可能在 caption/OCR 上退化 | 增加 non-VP regression suite |
| VP-P2-06 | 视觉推理链是否需要暴露给最终用户 | 公开 chain 可能增加延迟和输出噪声 | 支持 `return_visual_trace=True/False` |

---

## 5. 建议加入原报告的新章节

### 5.1 阶段门槛（Stage Gates）

| 阶段 | 进入条件 | 退出标准 |
| --- | --- | --- |
| Phase 0: 工程承载修复 | 当前代码可安装、训练入口明确 | VP task 能被 dataset 加载，converter/parser 单测通过 |
| Phase 1: VP MVP | 具备 COCO/YOLO -> VP 数据 | Florence-2-small 可稳定生成 ref + box，格式有效率 >= 95% |
| Phase 2: 效果评估 | MVP 可训练可评估 | 至少一个核心指标显著优于 baseline，且非 VP 任务无明显回退 |
| Phase 3: 专家/蒸馏 | SFT 单模型达到 plateau | Box/Point 专家相对单模型有可重复收益 |
| Phase 4: RL | deterministic reward 已稳定 | RL 带来可复现收益，且训练成本可接受 |

### 5.2 MVP 成功指标

| 指标 | 最低门槛 | 说明 |
| --- | --- | --- |
| VP 格式有效率 | >= 95% | parser 能成功解析 ref/box |
| 坐标合法率 | >= 99% | 所有坐标在 `[0,999]` 内，且 `x1 < x2`, `y1 < y2` |
| OD mAP | 不低于 baseline 95% | MVP 初期允许略低，但不能明显崩 |
| Counting Accuracy | 高于 baseline | 这是 VP 最应该体现价值的任务 |
| 训练稳定性 | 3 次 smoke run 无崩溃 | 覆盖 tokenizer resize、dataset、collator、trainer |
| 非 VP 任务回归 | 无明显下降 | 原有 `OD`、`CAPTION`、`OCR` 仍可用 |

### 5.3 推荐的最小实现范围

第一版只新增这些模块：

```text
florence_forge/core/visual_primitives.py
florence_forge/data/vp_converter.py
florence_forge/evaluation/visual_primitive_parser.py
tests/test_visual_primitives.py
tests/test_vp_converter.py
tests/test_vp_parser.py
tests/test_vp_dataset_integration.py
```

第一版不要新增这些模块：

```text
florence_forge/experimental/grpo_trainer.py
florence_forge/experimental/reward_models.py
florence_forge/training/distillation_trainer.py
```

原因是 MVP 还没有证明 VP 格式收益，过早引入 RL/蒸馏会让问题定位变得困难。

### 5.4 结构化 VP 解码的阶段性结论

真实权重 + 真实 COCO128 数据的短训结果显示，Florence-2 已具备很强的原生定位先验，但在少量 LoRA step 下并不会自然稳定生成 `<|ref|>`、`<|box|>` wrapper。继续强推 wrapper SFT 会把问题混在一起：一部分是“框是否准”，另一部分是“格式是否会写”。

因此当前更稳妥的工程路径是新增结构化 decoder：

1. 保留 Florence 原生 `<loc_*>` 生成作为定位能力来源。
2. 用 `FlorenceNativeDetectionParser` 解析 `label<loc_*>` 序列。
3. 用 `StructuredVisualPrimitiveDecoder` 确定性输出 VP `ref + box` 文本。
4. 再交给 `VisualPrimitiveParser`、VP metrics 和可视化工具统一消费。

这条路径的好处是立即获得可解析、可评分、可回放的视觉证据链，同时不掩盖研究事实：模型本身尚未充分内化 VP wrapper。后续训练目标应从“能否定位”拆成两个可测问题：第一，原生定位是否保持或提升；第二，模型是否能在更高质量数据和更长训练下稳定直接生成 VP wrapper。

### 5.5 训练完备性的可执行 Gate

为了避免“训练方案完备”只停留在主观判断，本次新增训练审计入口：

```bash
python3 scripts/experiments/audit_florence_vp_training.py \
  --training-summary .codex_reports/florence_vp_loc_token_smoke/real_florence_vp_training_smoke_summary.json \
  --inference-summary .codex_reports/florence_vp_structured_decoder_visualizations/train/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_training_audit
```

审计会输出 `vp_training_audit.json` 和 `vp_training_audit.md`，并检查以下 gate：

1. `training_smoke_passed`: 真实权重、真实数据、梯度更新是否跑通。
2. `loc_token_format`: 是否使用更贴合 Florence 先验的 `loc_tokens`。
3. `vp_head_trainable`: LoRA 是否同时训练/保存 `lm_head` 与 shared embedding。
4. `raw_vp_internalized`: raw `vp_format_valid_ratio` 是否达到 0.95。
5. `structured_vp_usable`: `structured_vp_format_valid_ratio` 是否达到 0.95。
6. `decoder_dependency_low`: `structured_vp_decoder_ratio` 是否低于 0.50。
7. `baseline_present`: 是否有 base Florence 对照 summary。

当前状态如果 raw VP 仍为 0、structured VP 为 1，会被判定为 `engineering_mvp_ready_needs_wrapper_training`。只有 raw VP wrapper 生成率达标且有 baseline 对照时，才会进入 `candidate_training_complete`。

本次进一步新增实验编排入口：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --training-summary .codex_reports/florence_vp_loc_token_smoke/real_florence_vp_training_smoke_summary.json \
  --manifest-path .codex_reports/florence_vp_loc_token_smoke/vp_real_data_manifest.json \
  --output-dir .codex_reports/florence_vp_experiment \
  --max-samples 2
```

该入口默认复用已有训练 summary 和 adapter，依次跑 adapter 推理、base Florence 推理和 audit。当前 2 样本真实运行结果显示：adapter 与 base Florence 的 raw `vp_format_valid_ratio` 都是 0，`structured_vp_format_valid_ratio` 都是 1，`structured_vp_decoder_ratio` 都是 1。也就是说，短训 adapter 在这组样本上没有比 base Florence 更会写 VP wrapper；它仍主要继承 Florence 原生定位能力，并依赖 decoder 完成 VP 结构化。

进一步运行 64 step 的 loc-token LoRA wrapper-focused 小实验：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_wrapper_sft_64step \
  --training-steps 64 \
  --max-train-samples 8 \
  --max-val-samples 2 \
  --max-samples 2
```

结果仍为 raw `vp_format_valid_ratio=0.0`、`structured_vp_format_valid_ratio=1.0`、`structured_vp_decoder_ratio=1.0`。同时，强制 decoder prefix 为 `<|ref|>{label}<|/ref|><|box|>` 后，模型仍不会稳定生成 `<|/box|>`，并会在 box 前插入噪声词。这说明当前瓶颈不是单纯采样不到起始 wrapper token，而是 VP wrapper 的闭合结构尚未被模型学稳。后续训练应考虑更强的 wrapper curriculum、更多纯 OD_VP 样本、暂时去掉 COUNT_VP 混合干扰，或引入受约束解码/后处理作为产品路径。

随后在 64 step adapter 上补跑 val split 对照，并清理已验证中间权重：

- train adapter、train baseline、forced-prefix、val adapter、val baseline 的 raw `vp_format_valid_ratio` 均为 0。
- 对应 `structured_vp_format_valid_ratio` 均为 1，`structured_vp_decoder_ratio` 均为 1。
- val audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`，`raw_vp_internalized` 未通过，`baseline_present` 已通过。
- 已删除 `.codex_reports` 中经过验证的 LoRA adapter 权重目录，仅保留 summary、manifest、可视化图和 audit；64 step 实验目录从约 616MB 降至约 4.9MB。

进一步将 64 step 实验改为纯 `OD_VP`（不混入 `COUNT_VP`）后，结论仍未改变：

- 训练完成：`steps_executed=64`，`final_loss=9.76650`，`trainable_param_delta_norm=0.57369`。
- adapter 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- baseline 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`。
- 本次由 `--cleanup-adapter-after-audit` 自动删除 `.codex_reports/florence_vp_od_only_64step/training/adapter`，避免保留已验证中间权重。

为定位 wrapper 未内化的原因，本次新增 token/logit 诊断入口：

```bash
python3 scripts/experiments/probe_florence_vp_tokens.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --manifest-path .codex_reports/florence_vp_od_only_64step/training/vp_real_data_manifest.json \
  --output-dir .codex_reports/florence_vp_token_probe \
  --max-samples 1 \
  --max-new-tokens 8
```

真实 probe 输出 `status=generation_prior_blocks_wrapper`：

- VP marker 均为单 token：`<|ref|>` 到 `<|/point|>` 对应 tokenizer id `51289-51294`。
- 模型 embedding 已 resize：tokenizer vocab size 与 model vocab size 都是 `51295`。
- 训练 label 确实监督了 wrapper：1 个样本中有 4 个 VP marker token、4 个 `<loc_*>` token。
- 生成第一内容 token 仍是 Florence 原生 `cat`，随后直接进入 `<loc_*>`；`<|ref|>` 在该步排名约 `23084`，概率约 `1.96e-11`，`<|box|>` 排名约 `23081`。

因此，当前瓶颈不是“VP token 没加上”或“labels 没监督到”，而是 Florence 原生 OD 生成先验压过了新 wrapper。下一步不宜只加训练步数，应优先尝试：受约束 VP 解码、wrapper-only curriculum、降低 `<OD>` 原生任务先验的替代 prompt、或用非特殊普通文本标记（如 `<ref>`/`</ref>`）做对照实验。

本次继续推进了普通文本 wrapper 对照：

- 新增 `marker_style` 维度，保留默认 `special` 的 `<|ref|>`/`<|box|>`，同时支持 `plain` 的 `<ref>`/`<box>`。
- converter、parser、metrics、structured decoder、CLI、可视化脚本、训练 smoke、实验 runner、token probe 都已接入 `marker_style`。
- 已物化 COCO128 YOLO 的 plain-marker loc-token 数据：`.codex_reports/florence_vp_plain_marker_probe/coco128_yolo_od_vp_plain_loc.jsonl`，共 126 行，大小约 76KB；前 5 行 parser/metrics 验证 `vp_format_valid_ratio=1.0`、`vp_coordinate_valid_ratio=1.0`、`vp_box_count_exact_match=1.0`。
- plain marker token probe 显示 `<ref>` 并非单 token，但 label 中能统计到 marker 序列：1 个样本里 8 个 wrapper marker sequence、28 个 `<loc_*>` token。
- plain marker 的生成先验有所改善但仍被原生 OD 压制：第一内容 token 是 `bus`，`<ref>` 首 token probe key `<ref>[0]` 排名约 `11720`，概率约 `8.46e-10`；相比 special `<|ref|>` 的约 `23084` 有提升，但仍远低于可自然生成的范围。

因此，普通文本 marker 不是“立刻解决 wrapper 内化”的银弹，但它降低了新 special token 的冷启动难度。下一轮如果继续训练，应优先跑一个**很短的 plain-marker LoRA 对照**，并用 `--cleanup-adapter-after-audit` 删除中间 adapter；若 plain 仍无法 raw internalize，则应把工程主路径固定为 structured/constrained decoding，把 raw wrapper 生成作为研究分支。

随后执行了 32 step plain-marker LoRA 对照（本轮 adapter 已自动删除）：

- 训练完成：`steps_executed=32`，`final_loss=6.92600`，`trainable_param_delta_norm=0.23192`。
- adapter 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- baseline 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`。
- token probe 中 plain opening marker 首 token `<ref>[0]` 在第一内容步排名约 `3220`，比未训练 plain baseline 的约 `11720` 有明显改善，但模型实际仍先输出 `cat<loc_*>`，raw wrapper 未内化。
- cleanup 成功删除 `.codex_reports/florence_vp_plain_marker_32step/training/adapter`。

这次 plain 对照还暴露了一个 tokenizer 细节：`</ref><box>` 连写时会被 BPE 合成类似 `><` 的子词，导致 `</ref>` 与 `<box>` marker sequence 不能被完整统计。已将 plain formatter 改为 `<ref>label</ref> <box>...</box>`，即在 ref span 与 box span 之间加入一个空格；新的 spaced plain 数据 probe 已能完整统计 4 组 marker sequence：`<ref>`、`</ref>`、`<box>`、`</box>` 各 4 次。后续如果继续 plain-marker 训练，应使用这个 spaced 版本重新跑短训对照。

随后用 spaced plain formatter 重跑 32 step LoRA 对照，并在 audit 后自动清理 adapter：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_plain_spaced_32step \
  --training-steps 32 \
  --max-train-samples 8 \
  --max-val-samples 2 \
  --max-samples 2 \
  --vp-marker-style plain \
  --structured-vp-marker-style plain \
  --device auto \
  --torch-dtype float32 \
  --max-new-tokens 128 \
  --num-beams 1 \
  --token-probe-max-samples 1 \
  --token-probe-max-new-tokens 8
```

结果确认了 plain marker 的方向性改善，但仍未达到 raw wrapper 内化：

- 训练完成：`steps_executed=32`，`final_loss=6.75500`，`trainable_param_delta_norm=0.22506`。
- 第一批 answer preview 已使用 spaced plain 格式：`<ref>cat</ref> <box><loc_7><loc_77><loc_559><loc_768></box>`。
- adapter 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- baseline 推理：raw `vp_format_valid_ratio=0.0`，`structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`。
- token probe 状态仍为 `generation_prior_blocks_wrapper`；1 个样本中 `<ref>`、`</ref>`、`<box>`、`</box>` 均被完整统计到 1 次。
- `<ref>[0]` 在第一内容步排名约 `3514`，比未训练 plain baseline 的约 `11720` 明显改善，但仍不足以让模型自然先生成 wrapper。
- cleanup 成功删除 `.codex_reports/florence_vp_plain_spaced_32step/training/adapter`，未保留已验证中间权重。

阶段性判断：spaced plain 已修复 tokenization 统计问题，并降低了起始 marker 的 prior barrier；但 32 step 短训仍无法改变 Florence-2 的原生 `label<loc_*>` 生成路径。下一阶段不应继续只堆短训 step，而应把工程主路径固定为 **structured VP decoding**，并把 raw wrapper 内化拆成单独研究分支：受约束 decoder prefix 诊断、wrapper-only/text-first curriculum、更大纯 `OD_VP` 数据和更长训练。

为支持后续受约束诊断，实验 runner 已新增 `--decoder-prefix` 透传到可视化推理脚本。示例：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --training-summary .codex_reports/florence_vp_plain_spaced_32step/training/real_florence_vp_training_smoke_summary.json \
  --manifest-path .codex_reports/florence_vp_plain_spaced_32step/training/vp_real_data_manifest.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_prefix_probe \
  --adapter-dir <adapter-dir-if-kept> \
  --decoder-prefix '<ref>{label}</ref> <box>' \
  --vp-marker-style plain \
  --structured-vp-marker-style plain \
  --max-samples 2
```

注意：若前序实验使用 `--cleanup-adapter-after-audit`，adapter 已被删除，不能再补跑 adapter forced-prefix；这类 probe 应在同一轮实验中开启，或仅对 baseline 做诊断。

基于已删除 adapter 后的实际情况，本次用 base Florence 跑了 2 样本 plain forced-prefix baseline：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --training-summary .codex_reports/florence_vp_plain_spaced_32step/training/real_florence_vp_training_smoke_summary.json \
  --manifest-path .codex_reports/florence_vp_plain_spaced_32step/training/vp_real_data_manifest.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_prefix_baseline \
  --adapter-dir "" \
  --skip-baseline \
  --decoder-prefix '<ref>{label}</ref> <box>' \
  --vp-marker-style plain \
  --structured-vp-marker-style plain \
  --max-samples 2
```

诊断结果：raw `vp_format_valid_ratio=0.0`，structured `structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。也就是说，强制起手能让输出靠近 VP 形状，但模型仍不会自然闭合 `</box>`，并会在 `<box>` 后继续走 Florence 原生 label + loc 生成模式。为减少这种半闭合 VP 对 structured decoder 的污染，已优化 `FlorenceNativeDetectionParser`：当 native loc 前缀中存在显式 `<ref>...</ref>` 或 `<|ref|>...<|/ref|>` 时，优先使用 ref span 里的 label，而不是把 `<box>` 后的原生续写拼进 label。修复后，上述 raw 输出会被结构化为干净的 `<ref>cat</ref> ... <ref>footwear</ref>` 和 `<ref>zebra</ref>`。

本次进一步将 structured VP 解码固化到 CLI 推理路径：

- `infer` 新增 `--structured-vp-mode {off,auto,on}`，其中 `off` 保持后向兼容，`on` 强制输出 structured VP，`auto` 在 `OD`、`OD_VP`、`COUNT_VP`、`PHRASE_GROUNDING_VP`、`OPEN_VOCABULARY_DETECTION`、`CAPTION_TO_PHRASE_GROUNDING` 等定位类任务上自动输出。
- 旧参数 `--structured-vp-decode` 继续保留，等价于 `--structured-vp-mode on`。
- 可视化推理脚本和实验 runner 同步支持 `structured_vp_mode`；runner 默认仍为 `on`，保持既有 audit 行为。
- 真实 CLI 验证命令已跑通：

```bash
python3 -m florence_forge.cli.main infer \
  --model /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --input /Users/gatilin/PycharmProjects/datasets/coco128/images/train2017/000000000575.jpg \
  --output .codex_reports/florence_vp_cli_auto_infer \
  --device auto \
  --task-prompt '<OD>' \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain
```

输出 JSON 中，raw result 仍是 `cat<loc_0><loc_78><loc_564><loc_774>footwear<loc_515><loc_156><loc_729><loc_560>`，但 `structured_vp` 自动生成：

```text
<ref>cat</ref> <box><loc_0><loc_78><loc_564><loc_774></box>
<ref>footwear</ref> <box><loc_515><loc_156><loc_729><loc_560></box>
```

并且 `format_valid=true`、`source=florence_native`、`used_structured_decoder=true`。这说明当前工程产品路径已经可以在真实权重和真实图像上稳定输出 VP JSON。

真实 CLI 验证还暴露并修复了一个部署级兼容问题：`InferenceEngine` 原先没有 patch Transformers 4.50+ 下 Florence remote-code 模型缺失 `generate()` 的情况，导致 CLI 真实推理返回空字符串。本次已将 `GenerationMixin` 兼容 patch 下沉到 `Florence2Backend`，并在 LoRA 注入后再次执行，因此 CLI、server、evaluation 和后续训练脚本都会共享这条修复。

随后执行了更长的 96 step plain-spaced LoRA 对照，继续使用真实 Florence 权重、COCO128 YOLO 数据、plain spaced wrapper，并在 audit 后自动删除 adapter：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step \
  --training-steps 96 \
  --max-train-samples 16 \
  --max-val-samples 4 \
  --max-samples 3 \
  --vp-marker-style plain \
  --structured-vp-marker-style plain \
  --structured-vp-mode on \
  --device auto \
  --torch-dtype float32 \
  --max-new-tokens 128 \
  --num-beams 1 \
  --token-probe-max-samples 2 \
  --token-probe-max-new-tokens 8
```

结果：

- 训练完成：`steps_executed=96`，`final_loss=5.60351`，`trainable_param_delta_norm=0.80443`，训练数据为 16 个 `OD_VP` 样本、4 个 val 样本。
- adapter 推理：raw `vp_format_valid_ratio=0.0`，structured `structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- baseline 推理：raw `vp_format_valid_ratio=0.0`，structured `structured_vp_format_valid_ratio=1.0`，`structured_vp_decoder_ratio=1.0`。
- audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`，`raw_vp_internalized=false`，`decoder_dependency_low=false`，`baseline_present=true`。
- token probe 状态仍为 `generation_prior_blocks_wrapper`。第 1 个样本第一内容步仍生成 `cat`，`<ref>[0]` rank 约 `3491`、概率约 `8.66e-10`；这与 32 step spaced plain 的 rank 约 `3514` 基本持平，说明继续增加到 96 step 并没有实质突破起始 wrapper barrier。
- 第 2 个 probe 样本第一内容步生成 `z`，`<ref>[0]` rank 约 `1980`，但生成路径仍是原生 label/token，而不是 VP wrapper。
- cleanup 成功删除 `.codex_reports/florence_vp_plain_spaced_96step/training/adapter`；最终实验目录约 4.2MB。

这次 96 step 还暴露了一个新的质量风险：native loc 输出会在部分样本上过生成长 `<loc_*>` 串。第 3 个样本中 adapter structured 预测框数为 14、baseline 为 15，而 GT 只有 1。为让这类问题进入自动审计，本次新增 box-count overgeneration 指标：

- `avg_pred_boxes`
- `avg_gt_boxes`
- `box_count_overgeneration_ratio`
- audit gate: `box_count_not_overgenerated`

对 96 step 实验重新审计后：adapter `avg_pred_boxes=5.67`，`avg_gt_boxes=1.0`，`box_count_overgeneration_ratio=0.667`；baseline `avg_pred_boxes=6.0`，`box_count_overgeneration_ratio=0.667`。这说明 structured VP 能稳定包装 native 输出，但不能替代检测质量控制。下一步工程优先级应转向 **constrained decoding / post-filtering**：限制每个 label 的 box 数、按 GT-free 规则过滤异常长 loc 串、或使用 processor 原生 post_process_generation 结果作为 structured VP 的 box source。

本次继续推进了 structured VP 的 post-filtering 路径，新增两个 opt-in 约束参数：

- `--structured-vp-max-boxes-per-label`: 每个 label 最多保留多少个 box。
- `--structured-vp-max-total-boxes`: 整体最多保留多少个 box。

两个参数已接入：

- `StructuredVisualPrimitiveDecoder`
- CLI `infer`
- `visualize_florence_vp_adapter.py`
- `run_florence_vp_training_experiment.py`

默认不启用过滤，避免改变已有实验口径。启用后 decoder 会按原始生成顺序保留前面的 box，并记录 `raw_detection_count` 与 `filtered_detection_count`，方便审计。

对 96 step 已有 raw prediction 做离线重放验证：

| 过滤策略 | adapter avg pred boxes | adapter overgeneration | baseline avg pred boxes | baseline overgeneration |
| --- | ---: | ---: | ---: | ---: |
| 不过滤 | 5.67 | 0.667 | 6.00 | 0.667 |
| `max_boxes_per_label=1,max_total_boxes=2` | 1.67 | 0.667 | 1.67 | 0.667 |
| `max_total_boxes=1` | 1.00 | 0.000 | 1.00 | 0.000 |

这说明“每 label 限制”只能压掉同一 label 下的长 loc 串，但不能处理额外类别（如 `cat + footwear`）带来的多框；对当前短样本集这种单目标评估，`max_total_boxes=1` 最有效。真实 CLI 也已验证：

```bash
python3 -m florence_forge.cli.main infer \
  --model /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --input /Users/gatilin/PycharmProjects/datasets/coco128/images/train2017/000000000575.jpg \
  --output .codex_reports/florence_vp_cli_auto_infer_filtered \
  --device auto \
  --task-prompt '<OD>' \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain \
  --structured-vp-max-total-boxes 1
```

输出中 raw native detection count 为 2，过滤后 structured VP 只保留 1 个 `cat` box，`filtered_detection_count=1`。这条路径可以作为面向单目标 grounding/OD_VP 评估的保守模式；通用 OD 场景则应配置更高的 `max_total_boxes` 或改用 processor 原生 post-processing/NMS。

2026-06-06 继续推进后，已将上述离线重放固化为正式脚本：

```bash
python3 scripts/experiments/replay_structured_vp_filters.py \
  --inference-summary adapter=.codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --inference-summary baseline=.codex_reports/florence_vp_plain_spaced_96step/baseline_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/postfilter_replay_v2 \
  --structured-vp-marker-style plain
```

该脚本会从已有 `raw_prediction` 重算 structured VP，不重新加载模型、不生成 adapter；默认输出 `unfiltered`、`total1`、`per_label1_total2` 三组 policy，并为每组生成可再次喂给 audit 的 `vp_inference_visualization_summary.json`。已新增到 `run_florence_vp_training_experiment.py` 的可选阶段：`--run-filter-replay`。

用 `total1` replay summary 重新审计 96 step 实验后，`box_count_not_overgenerated` gate 从原来的失败变为通过：

- adapter: `avg_pred_boxes=1.0`，`avg_gt_boxes=1.0`，`box_count_overgeneration_ratio=0.0`。
- baseline: `avg_pred_boxes=1.0`，`avg_gt_boxes=1.0`，`box_count_overgeneration_ratio=0.0`。
- audit 状态仍为 `engineering_mvp_ready_needs_wrapper_training`，因为 raw VP wrapper 仍未内化，`structured_vp_decoder_ratio=1.0`。

因此当前结论应更精确地表述为：**Florence-VP 的训练方案尚不完备；但真实权重 + 真实数据下，结构化解码与单目标 post-filter 已经能形成可验证、低成本、可审计的工程 MVP。**

随后进一步将单目标过滤产品化为显式策略参数：

- CLI `infer`: `--structured-vp-filter-policy {none,auto,single-target,nms}`。
- 可视化脚本: `--structured-vp-filter-policy {none,auto,single-target,nms}`。
- 实验 runner: `--structured-vp-filter-policy {none,auto,single-target,nms}`，并透传到 adapter/baseline 可视化阶段。

策略含义：

- `none`: 默认行为，不自动添加后过滤上限。
- `single-target`: 强制 `max_total_boxes=1`，除非用户显式传入更高的 `--structured-vp-max-total-boxes`。
- `nms`: 默认使用 `nms_iou_threshold=0.5`，用于去除重复框，不强行压成单目标。
- `auto`: 只对 `OD_VP`、`PHRASE_GROUNDING_VP`、`CAPTION_TO_PHRASE_GROUNDING` 这类单目标 VP/grounding prompt 自动解析为 `max_total_boxes=1`；普通 `<OD>` 不改变默认输出。

真实权重 CLI 已验证无需手写 `--structured-vp-max-total-boxes 1`：

```bash
python3 -m florence_forge.cli.main infer \
  --model /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --input /Users/gatilin/PycharmProjects/datasets/coco128/images/train2017/000000000575.jpg \
  --output .codex_reports/florence_vp_cli_single_target_policy \
  --device auto \
  --task-prompt '<OD>' \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy single-target
```

结果：`source=florence_native`，`format_valid=true`，`filter_policy=single-target`，`max_total_boxes=1`，`raw_detection_count=2`，`filtered_detection_count=1`，最终 VP 输出为单个 `cat` 框。这把上一轮实验中的有效手工参数收敛成了可复用的推理策略。

2026-06-06 继续补齐了 VP 检测质量评估闭环，新增：

- `florence_forge/evaluation/vp_detection_quality.py`
- `scripts/experiments/evaluate_vp_detection_quality.py`
- `run_florence_vp_training_experiment.py --run-quality-eval`

该评估不只看 VP 格式是否可解析，还会按 label + IoU 贪心匹配 prediction/target，输出：

- `precision`、`recall`、`f1`
- `true_positives`、`false_positives`、`false_negatives`
- `mean_matched_iou`
- `box_count_exact_match_ratio`
- `box_count_overgeneration_ratio`
- `single_target_hit_ratio`
- `single_target_exact_hit_ratio`
- bad cases 列表，包括 `false_positive`、`false_negative`、`overgenerated`、`undergenerated`

对 96 step adapter 既有真实推理 summary 重新评估：

```bash
python3 scripts/experiments/evaluate_vp_detection_quality.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/quality_unfiltered \
  --structured-vp-marker-style plain

python3 scripts/experiments/evaluate_vp_detection_quality.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/quality_single_target \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy single-target
```

结果进一步解释了上一轮 box-count 现象：

| 策略 | precision | recall | f1 | mean IoU | avg pred boxes | overgeneration | single-target exact hit | bad cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 不过滤 | 0.1765 | 1.0000 | 0.3000 | 0.9636 | 5.67 | 0.667 | 0.333 | 2 |
| `single-target` | 1.0000 | 1.0000 | 1.0000 | 0.9636 | 1.00 | 0.000 | 1.000 | 0 |

这说明当前模型/decoder 的首个 native box 往往已经能对齐 GT，主要问题是额外类别和长 loc 串带来的 false positives。`single-target` 不是通用 OD 解法，但对当前 `OD_VP`/grounding 单目标评估是可审计、有效、低成本的工程策略。完整 VP 实现的下一块应转向：多目标 OD 的置信度/NMS/processor post-process 对齐，而不是继续无约束地增加短训步数。

随后补入多目标方向的第一步：structured VP per-label NMS。

新增参数：

- CLI `infer`: `--structured-vp-filter-policy nms`
- CLI `infer`: `--structured-vp-nms-iou-threshold`
- 可视化脚本、质量评估脚本、实验 runner 同步支持同名参数。
- 离线 replay 支持 `--filter-config nms:nms_iou_threshold=0.5`。

实现口径：NMS 只在同一 label 内按 IoU 去重，保留生成顺序中先出现的框；没有置信度时不重排，也不跨 label 删除。因此它是多目标 duplicate suppression，而不是额外类别过滤。

对 96 step adapter summary 运行：

```bash
python3 scripts/experiments/evaluate_vp_detection_quality.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/quality_nms \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5
```

结果：

| 策略 | precision | recall | f1 | avg pred boxes | overgeneration | filtered detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不过滤 | 0.1765 | 1.0000 | 0.3000 | 5.67 | 0.667 | 0 |
| NMS 0.5 | 0.1875 | 1.0000 | 0.3158 | 5.33 | 0.667 | 1 |
| `single-target` | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.000 | 14 |

这说明 NMS 可以作为通用多目标 VP 的基础清洗层，但当前 COCO128 bad case 的主要错误来自额外类别（例如 `cat + footwear`、`dog + footwear`），不是单纯重复框。下一步如果继续完善通用 OD_VP，应引入 processor 原生 post-process / 类别白名单 / open-vocabulary text constraint，而不是只调 NMS 阈值。

继续推进后，已补入显式 label allow-list 约束：

- CLI `infer`: `--structured-vp-allowed-labels`
- 可视化脚本: `--structured-vp-allowed-labels`
- 质量评估脚本: `--structured-vp-allowed-labels`
- 实验 runner: `--structured-vp-allowed-labels`
- 离线 replay: `--filter-config allowed_labels=...`，由于 filter config 自身用逗号分隔键值项，label 列表推荐用 `|` 或 `;`，例如 `allowed_labels=cat|zebra|dog`。

CLI 对开放词汇/phrase grounding 做了一个保守便利：如果没有显式传 `--structured-vp-allowed-labels`，且任务是 `OPEN_VOCABULARY_DETECTION` 或 `CAPTION_TO_PHRASE_GROUNDING`，会尝试复用 `--text-input` 作为 allow-list。普通 `<OD>` 不自动猜类别，避免误伤通用检测。

对同一份 96 step adapter summary 运行：

```bash
python3 scripts/experiments/evaluate_vp_detection_quality.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/quality_allowed_labels \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels cat,zebra,dog
```

结果：

| 策略 | precision | recall | f1 | avg pred boxes | overgeneration | filtered detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不过滤 | 0.1765 | 1.0000 | 0.3000 | 5.67 | 0.667 | 0 |
| NMS 0.5 | 0.1875 | 1.0000 | 0.3158 | 5.33 | 0.667 | 1 |
| allow-list `cat,zebra,dog` | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.000 | 14 |
| `single-target` | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.000 | 14 |

这个实验使用的是三张样本的目标类别集合，因此更接近“显式类别约束/oracle label set”验证，而不是无条件通用 OD 能力。但它证明了一点：当前 false positives 主要可以被类别约束解决，VP 工程路径已经具备从 raw native output 到 constrained structured VP report 的完整可审计闭环。下一步若面向真实 open-vocabulary/grounding 产品，应优先把用户 query/text_input 变成 allow-list；若面向通用 OD，则需要 processor 原生后处理或类别置信度，而不是依赖 allow-list。

继续推进后，已新增策略比较器：

- 核心函数：`compare_vp_quality_reports`、`recommend_vp_policy`、`render_vp_policy_comparison_markdown`
- CLI：`scripts/experiments/compare_vp_quality_policies.py`
- 实验 runner：`--run-policy-comparison`、`--policy-comparison-report name=path`

对 96 step 的四组质量报告运行：

```bash
python3 scripts/experiments/compare_vp_quality_policies.py \
  --report none=.codex_reports/florence_vp_plain_spaced_96step/quality_unfiltered/vp_detection_quality.json \
  --report nms=.codex_reports/florence_vp_plain_spaced_96step/quality_nms/vp_detection_quality.json \
  --report allowed=.codex_reports/florence_vp_plain_spaced_96step/quality_allowed_labels/vp_detection_quality.json \
  --report single=.codex_reports/florence_vp_plain_spaced_96step/quality_single_target/vp_detection_quality.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/policy_comparison
```

自动比较结论：

| rank | policy | kind | precision | recall | f1 | avg pred | overgen | 适用边界 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `allowed` | `allowed_labels` | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.0000 | 需要显式 query/category allow-list |
| 2 | `single` | `single_target` | 1.0000 | 1.0000 | 1.0000 | 1.00 | 0.0000 | 仅适合单目标 grounding/detection |
| 3 | `nms` | `nms` | 0.1875 | 1.0000 | 0.3158 | 5.33 | 0.6667 | 多目标 OD 的保守 fallback |
| 4 | `none` | `none` | 0.1765 | 1.0000 | 0.3000 | 5.67 | 0.6667 | 无约束 baseline |

机器推荐为 `allowed`，置信度标记为 `exploratory`，原因是样本数只有 3。比较器同时给出 caveat：`allowed` 不能作为无约束 OD 能力证明；若是无显式类别列表的多目标检测，应优先使用 `nms` 或后续接入 processor 原生 post-process / 类别置信度。

继续补齐后，已把 query/category 约束做成逐样本能力：

- `evaluate_vp_detection_quality.py` 新增 `--structured-vp-allowed-labels-field`
- `VPDetectionQualityConfig` 新增 `allowed_labels_field`
- 可使用 `text_input` / `query` 等记录字段作为每条样本自己的 allow-list
- 可使用 `target_labels` / `reference_labels` / `gt_labels` 作为 oracle 上限诊断，但不能作为无约束 OD 证据
- structured decoder 现在不仅能过滤 Florence native `label<loc_*>`，也能过滤模型已经输出的 VP wrapper

同时新增一键 sweep：

```bash
python3 scripts/experiments/sweep_vp_quality_policies.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/policy_sweep \
  --structured-vp-marker-style plain \
  --include-target-label-oracle
```

这会一次性生成：

- `none/vp_detection_quality.json`
- `nms/vp_detection_quality.json`
- `single/vp_detection_quality.json`
- `target_oracle/vp_detection_quality.json`
- `vp_policy_comparison.json`
- `vp_quality_policy_sweep.json`

对现有 3 样本 summary 的 sweep 结果仍是探索性证据：`single` 和 `target_oracle` 都达到 F1=1.0，`nms` 为 0.3158，`none` 为 0.3000。由于三张样本全是单目标，`single` 排名第一；但报告会保留 caveat，提示多目标无约束 OD fallback 仍是 `nms`。

推荐的下一轮 held-out 命令形态：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --run-policy-sweep \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --training-summary .codex_reports/florence_vp_plain_spaced_96step/training/real_florence_vp_training_smoke_summary.json \
  --manifest-path .codex_reports/florence_vp_plain_spaced_96step/training/vp_real_data_manifest.json \
  --output-dir .codex_reports/florence_vp_heldout_policy_sweep \
  --split val \
  --max-samples 50 \
  --structured-vp-mode on \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field text_input \
  --policy-sweep-include-target-label-oracle \
  --device mps \
  --torch-dtype float32
```

如果 held-out 数据没有 `text_input`，则 `text_input` 约束不会生效；这时应把 open-vocabulary/phrase-grounding 数据转换阶段补上 query 字段，或者只把 `target_oracle` 当作“理论上类别约束能带来的上限”。

继续推进后，已补齐 query-grounding 数据构造层：

- `VisualPrimitiveConverter.coco_to_vp_grounding`
- `VisualPrimitiveConverter.yolo_to_vp_grounding`
- `VisualPrimitiveConverter.vp_od_jsonl_to_query_grounding`
- CLI: `convert vp-coco-grounding`、`convert vp-yolo-grounding`、`convert vp-jsonl-grounding`
- 脚本: `scripts/data-conversion/convert_visual_primitives.py jsonl-grounding`
- 训练 smoke: `--include-grounding`、`--grounding-task-type`
- 实验 runner: `--include-grounding`、`--manifest-data-key val_grounding_path`
- 可视化推理: `--data-key`、`--structured-vp-allowed-labels-field`

query-grounding 样本定义为“每张图、每个类别一条”：

```json
{
  "prefix": "<CAPTION_TO_PHRASE_GROUNDING>",
  "text_input": "cat",
  "suffix": "<ref>cat</ref> <box><loc_281><loc_258><loc_757><loc_829></box>"
}
```

这条路径把“类别约束”从评估技巧推进成了正式数据协议：模型推理接收 `text_input`，structured VP 解码可以按同一字段过滤，quality sweep 也能按同一字段复现。

已用现有 96 step 的 `val_od_vp.jsonl` 派生真实 query held-out：

```bash
python3 scripts/data-conversion/convert_visual_primitives.py jsonl-grounding \
  --input .codex_reports/florence_vp_plain_spaced_96step/training/val_od_vp.jsonl \
  --output .codex_reports/florence_vp_plain_spaced_96step/query_grounding_val_vp.jsonl \
  --task-type PHRASE_GROUNDING_VP \
  --box-format loc_tokens \
  --marker-style plain
```

产物：`.codex_reports/florence_vp_plain_spaced_96step/query_grounding_val_vp.jsonl`，共 7 条 query 样本，类别包括 `giraffe`、`cat`、`car`、`dog`、`boat`、`fork`、`cake`。

下一次真实 held-out 建议直接用：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --include-grounding \
  --cleanup-adapter-after-audit \
  --run-policy-sweep \
  --manifest-data-key val_grounding_path \
  --structured-vp-allowed-labels-field text_input \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_query_grounding_heldout \
  --split val \
  --max-samples 50 \
  --max-train-samples 128 \
  --max-val-samples 50 \
  --device mps \
  --torch-dtype float32
```

如果只想评估已有模型/已有 manifest，可以跳过 `--run-training`，并用 `--manifest-data-key val_grounding_path` 指定 query held-out。

进一步验证后，已在真实 Florence 基座上跑通 query-grounding baseline 推理，不使用 adapter、不产生权重：

```bash
python3 scripts/infer/visualize_florence_vp_adapter.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_grounding_val_vp.jsonl \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/query_grounding_baseline_inference \
  --max-samples 7 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1 \
  --structured-vp-mode on \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field text_input
```

推理 summary：

- `num_samples`: 7
- `vp_format_valid_ratio`: 0.0
- `structured_vp_format_valid_ratio`: 1.0
- `structured_vp_decoder_ratio`: 1.0
- `structured_source_counts`: `{"florence_native": 7}`
- `avg_pred_boxes`: 1.1429
- `avg_gt_boxes`: 1.1429
- `box_count_overgeneration_ratio`: 0.0

说明：模型仍输出 Florence native `label<loc_*>`，不是原生 VP wrapper；但在 query-grounding 场景下，native 输出已经能稳定被 structured decoder 包装为 VP 证据链。

随后对该 summary 运行 policy sweep：

```bash
python3 scripts/experiments/sweep_vp_quality_policies.py \
  --summary .codex_reports/florence_vp_plain_spaced_96step/query_grounding_baseline_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/query_grounding_baseline_policy_sweep \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field text_input \
  --include-target-label-oracle
```

结果：

| policy | precision | recall | f1 | mean IoU | avg pred | avg GT | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 1.0000 | 1.0000 | 1.0000 | 0.9013 | 1.14 | 1.14 | 0 | 0 |
| `nms` | 1.0000 | 1.0000 | 1.0000 | 0.9013 | 1.14 | 1.14 | 0 | 0 |
| `text_input_allowed` | 1.0000 | 1.0000 | 1.0000 | 0.9013 | 1.14 | 1.14 | 0 | 0 |
| `target_oracle` | 1.0000 | 1.0000 | 1.0000 | 0.9013 | 1.14 | 1.14 | 0 | 0 |
| `single` | 1.0000 | 0.8750 | 0.9333 | 0.9251 | 1.00 | 1.14 | 0 | 1 |

这个结果很关键：在 query-grounding held-out 上，类别约束不只是 post-filter 补救，Florence 基座本身已经能响应 `text_input` 并输出正确类别框；`single` 反而会误删多实例 query（giraffe 有 2 个框），因此 query 产品路径不应默认使用 `single-target`，而应保留多框输出并用 `text_input` 作为可选安全过滤。

继续扩展到 50 条 query held-out 后，为控制磁盘，推理脚本新增 `--visualization-limit`，本次只保存前 5 张 PNG：

```bash
python3 scripts/data-conversion/convert_visual_primitives.py jsonl-grounding \
  --input .codex_reports/florence_vp_plain_spaced_96step/training/coco128_yolo_od_vp_all.jsonl \
  --output .codex_reports/florence_vp_plain_spaced_96step/query_grounding_all_vp.jsonl \
  --task-type PHRASE_GROUNDING_VP \
  --box-format loc_tokens \
  --marker-style plain

python3 scripts/infer/visualize_florence_vp_adapter.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_grounding_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/query_grounding50_baseline_inference \
  --max-samples 50 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1 \
  --visualization-limit 5 \
  --structured-vp-mode on \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field text_input
```

50 条 baseline summary：

- `structured_vp_format_valid_ratio`: 0.98
- `structured_vp_decoder_ratio`: 1.0
- `avg_pred_boxes`: 1.28
- `avg_gt_boxes`: 2.70
- `box_count_overgeneration_ratio`: 0.0
- `structured_filtered_detection_count`: 1

50 条 policy sweep：

| policy | precision | recall | f1 | mean IoU | avg pred | avg GT | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `text_input_allowed` | 0.8281 | 0.3926 | 0.5327 | 0.8831 | 1.28 | 2.70 | 11 | 82 |
| `target_oracle` | 0.8281 | 0.3926 | 0.5327 | 0.8831 | 1.28 | 2.70 | 11 | 82 |
| `none` | 0.8154 | 0.3926 | 0.5300 | 0.8831 | 1.30 | 2.70 | 12 | 82 |
| `nms` | 0.8125 | 0.3852 | 0.5226 | 0.8834 | 1.28 | 2.70 | 12 | 83 |
| `single` | 0.7600 | 0.2815 | 0.4108 | 0.8802 | 1.00 | 2.70 | 12 | 97 |

按 GT 实例数切片：

| GT boxes | samples | precision | recall | f1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27 | 0.8519 | 0.8519 | 0.8519 |
| 2-3 | 13 | 0.7778 | 0.4242 | 0.5490 |
| 4+ | 10 | 0.8421 | 0.2133 | 0.3404 |

这说明 query-grounding 的第一瓶颈已经从“类别误检”转为“多实例召回”：Florence 基座通常能找到正确类别和高 IoU 框，但经常只返回 1 个或少量实例。后续若训练 Florence-VP adapter，应该把 curriculum 明确转向 multi-instance query grounding，而不是继续默认 `single-target`。

随后继续做了两项优化验证：

1. 新增可选的 `contains` label match mode，用于 query/category-constrained 诊断中的短语包含关系，例如 `cup` vs `coffee cup`。默认仍是 `strict`，不会改变历史实验。
2. 将同源 query 数据从 `CAPTION_TO_PHRASE_GROUNDING` 改成 `OPEN_VOCABULARY_DETECTION`，验证 prompt 选择是否影响多实例召回。

新增参数：

- `scripts/experiments/evaluate_vp_detection_quality.py`: `--vp-label-match-mode {strict,contains}`、`--structured-vp-allowed-label-match-mode {strict,contains}`。
- `scripts/experiments/sweep_vp_quality_policies.py`: 同上，并新增 `--include-phrase-label-policy`，会额外输出 `text_input_phrase_allowed`。
- `scripts/infer/visualize_florence_vp_adapter.py`: `--structured-vp-allowed-label-match-mode {strict,contains}`。

50 条 phrase baseline 重新 sweep 后，`text_input_phrase_allowed` 没有提升：F1 从 `0.5327` 变为 `0.5300`，FP 从 `11` 变为 `12`。因此当前主要问题不是 label alias，而是 prompt/训练对多实例输出的召回能力。

同样 50 条样本改用 `OPEN_VOCABULARY_DETECTION` 后：

```bash
python3 scripts/data-conversion/convert_visual_primitives.py jsonl-grounding \
  --input .codex_reports/florence_vp_plain_spaced_96step/training/coco128_yolo_od_vp_all.jsonl \
  --output .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --task-type OPEN_VOCABULARY_DETECTION \
  --box-format loc_tokens \
  --marker-style plain

python3 scripts/infer/visualize_florence_vp_adapter.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_plain_spaced_96step/query_ovd50_baseline_inference \
  --max-samples 50 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1 \
  --visualization-limit 5 \
  --structured-vp-mode on \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field text_input
```

OVD 50 条 baseline summary：

- `structured_vp_format_valid_ratio`: 1.0
- `structured_vp_decoder_ratio`: 1.0
- `avg_pred_boxes`: 1.88
- `avg_gt_boxes`: 2.70
- `box_count_overgeneration_ratio`: 0.10

OVD 50 条 policy sweep：

| policy | precision | recall | f1 | mean IoU | avg pred | avg GT | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nms` | 0.8444 | 0.5630 | 0.6756 | 0.8420 | 1.80 | 2.70 | 14 | 59 |
| `none` | 0.8085 | 0.5630 | 0.6638 | 0.8542 | 1.88 | 2.70 | 18 | 59 |
| `text_input_allowed` | 0.8085 | 0.5630 | 0.6638 | 0.8542 | 1.88 | 2.70 | 18 | 59 |
| `text_input_phrase_allowed` | 0.8085 | 0.5630 | 0.6638 | 0.8542 | 1.88 | 2.70 | 18 | 59 |
| `single` | 0.8200 | 0.3037 | 0.4432 | 0.8536 | 1.00 | 2.70 | 9 | 94 |

统一比较报告 `.codex_reports/florence_vp_plain_spaced_96step/query_prompt_comparison/vp_policy_comparison.md` 的推荐策略为 `ovd_nms`。相对 phrase baseline 的 `text_input_allowed`，OVD + NMS 将 F1 从 `0.5327` 提升到 `0.6756`，召回从 `0.3926` 提升到 `0.5630`，说明下一步 Florence-VP 的 query-grounding 产品/训练路径应优先采用 `OPEN_VOCABULARY_DETECTION` prompt，并把 NMS 作为默认候选 policy；`contains` label match 保留为诊断工具，不应默认打开。

继续推进后，已将这一路径固化为 experiment runner preset：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --query-grounding-preset ovd-nms \
  --run-training \
  --run-policy-sweep \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_nms_curriculum_run \
  --device mps
```

`ovd-nms` preset 自动填充：

- `--include-grounding`
- `--grounding-task-type OPEN_VOCABULARY_DETECTION`
- `--grounding-selection multi-instance`
- `--grounding-curriculum multi-instance`
- `--training-data-order grounding-first`
- `--manifest-data-key val_grounding_path`
- `--structured-vp-filter-policy nms`
- `--structured-vp-nms-iou-threshold 0.5`
- `--structured-vp-allowed-labels-field text_input`
- `--structured-vp-marker-style plain`
- `--vp-marker-style plain`
- `--visualization-limit 5`
- `--max-new-tokens 64`

为了直接优化多实例召回，还新增了 curriculum 构造脚本：

```bash
python3 scripts/data-conversion/build_vp_query_curriculum.py \
  --input .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --single-weight 1 \
  --medium-weight 2 \
  --dense-weight 3
```

真实 362 条 OVD query 数据的 curriculum v1 结果：

| bucket | input | output | weight |
| --- | ---: | ---: | ---: |
| `single` | 216 | 216 | 1 |
| `medium` | 82 | 164 | 2 |
| `dense` | 64 | 192 | 3 |

- 输出总行数：572
- `avg_query_box_count_input`: 2.5663
- `avg_query_box_count_output`: 3.7745
- `max_query_box_count`: 13

这给下一轮 adapter 训练一个更明确的目标：不是继续让模型学 wrapper，而是用 OVD prompt 和多实例 curriculum 训练，让 `avg_pred_boxes` 从 1.8 更接近 GT 2.7，同时控制 NMS 后 FP 不显著升高。

### 5.7 OVD curriculum 8-step 短训复盘与训练入口修正

已完成一次真实权重、真实 COCO128 数据的 8-step OVD curriculum LoRA 短训：

- run root: `.codex_reports/florence_vp_ovd_curriculum_8step_20260607`
- training summary: `.codex_reports/florence_vp_ovd_curriculum_8step_20260607/training/real_florence_vp_training_smoke_summary.json`
- adapter inference: `.codex_reports/florence_vp_ovd_curriculum_8step_20260607/query_ovd50_adapter_inference`
- adapter policy sweep: `.codex_reports/florence_vp_ovd_curriculum_8step_20260607/query_ovd50_adapter_policy_sweep`
- comparison: `.codex_reports/florence_vp_ovd_curriculum_8step_20260607/base_vs_adapter_policy_comparison`

结果表明，这次 8-step adapter 没有超过 base OVD+NMS：

| model/policy | precision | recall | f1 | avg pred | avg GT | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_ovd_nms` | 0.8444 | 0.5630 | 0.6756 | 1.80 | 2.70 | 14 | 59 |
| `adapter_8step_ovd_nms` | 0.8409 | 0.5481 | 0.6637 | 1.76 | 2.70 | 14 | 61 |

这不是 OVD+VP 路线失败，而是一次数据入口暴露出的失败。训练 smoke 原先先用 shortest-row 选择 OD 子集，再从这个子集派生 query grounding；同时 DataLoader 使用 `shuffle=False`，`data_configs` 又默认 OD 在前、grounding 在后。8-step run 因此不仅 train grounding curriculum 几乎没有 dense bucket，短步数训练还很可能主要消费 OD 样本，实际没有训练到我们想优化的多实例 query-grounding 召回瓶颈。

为修正这个问题，训练入口已经新增两类能力：

1. `--grounding-selection multi-instance`：从完整派生 grounding 行中按 `query_box_count` 优先选择训练样本，避免短训天然偏向单实例。
2. `--grounding-train-path/--grounding-val-path`：直接使用外部 query-grounding JSONL。传入已构造好的 curriculum 时，该路径会作为 effective train path 使用，避免重复 over-sampling。
3. `--training-data-order grounding-first`：让短步数训练优先消费 query-grounding curriculum，而不是先被 OD wrapper 样本占满。

下一轮更合理的可复现实验命令：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --query-grounding-preset ovd-nms \
  --run-training \
  --run-policy-sweep \
  --cleanup-adapter-after-audit \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_curriculum_full_v1 \
  --training-steps 32 \
  --max-samples 50 \
  --device mps
```

空间处理：8-step run 的已验证中间 adapter 已删除，保留训练 summary、推理 summary、policy sweep 和对比报告；当前 `.codex_reports` 下未保留 `.safetensors/.pt/.pth/.bin` 模型权重。

### 5.8 Full curriculum 16-step 真实短训

修正训练入口后，又执行了一次更可信的 16-step OVD curriculum LoRA 短训：

- run root: `.codex_reports/florence_vp_ovd_curriculum_full_v1_16step_20260607`
- training summary: `.codex_reports/florence_vp_ovd_curriculum_full_v1_16step_20260607/training/real_florence_vp_training_smoke_summary.json`
- adapter inference: `.codex_reports/florence_vp_ovd_curriculum_full_v1_16step_20260607/adapter_inference`
- adapter policy sweep: `.codex_reports/florence_vp_ovd_curriculum_full_v1_16step_20260607/policy_sweep/adapter`
- base-vs-adapter comparison: `.codex_reports/florence_vp_ovd_curriculum_full_v1_16step_20260607/base_vs_adapter_policy_comparison`

这次训练入口已经对准 query grounding：

- `first_batch_task_type`: `OPEN_VOCABULARY_DETECTION`
- `training_data_order`: `grounding-first`
- `shuffle_train_data`: `true`
- `shuffle_seed`: `42`
- effective train grounding: 572 rows
- train grounding buckets: `single=216`, `medium=164`, `dense=192`
- `avg_query_box_count`: 3.7745
- `max_query_box_count`: 13

16-step adapter 的 50 条 OVD query 推理与 policy sweep 结果：

| model/policy | precision | recall | f1 | mean IoU | avg pred | avg GT | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_ovd_nms` | 0.8444 | 0.5630 | 0.6756 | 0.8420 | 1.80 | 2.70 | 14 | 59 |
| `adapter_16step_ovd_nms` | 0.8409 | 0.5481 | 0.6637 | 0.8407 | 1.76 | 2.70 | 14 | 61 |

结论：这次不再是“没喂到 grounding”的问题，训练确实消费了 OVD grounding curriculum；但 16 step 仍未改善多实例召回，推荐策略仍是 `base_ovd_nms`。这说明下一步应把问题从“数据入口是否正确”推进到“训练强度与可训练参数是否足够”：需要更长 schedule、分段评估、学习率/LoRA rank/`modules_to_save` 消融，或者考虑 constrained decoding / box-count aware loss，而不是只靠 8/16 step smoke 得出训练完备结论。

空间处理：本次 run 的 adapter 已由 `--cleanup-adapter-after-audit` 自动删除，仅保留 1.7M 左右的 summary、可视化和 policy sweep 报告。

### 5.9 Box-count bucket 误差定位

为了避免只看整体 F1，`vp_detection_quality` 已新增 `box_count_bucket_summary`，按 query/GT box count 分成：

- `single`: 1 个目标
- `medium`: 2-3 个目标
- `dense`: 4 个及以上目标

基于已有 50 条 OVD query summary 重新评估后，base 与 16-step adapter 的分桶结果如下：

| model | bucket | samples | precision | recall | f1 | avg pred | avg GT | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_ovd_nms` | `single` | 27 | 0.8276 | 0.8889 | 0.8571 | 1.07 | 1.00 | 5 | 3 |
| `base_ovd_nms` | `medium` | 13 | 0.8000 | 0.6061 | 0.6897 | 1.92 | 2.54 | 5 | 13 |
| `base_ovd_nms` | `dense` | 10 | 0.8889 | 0.4267 | 0.5766 | 3.60 | 7.50 | 4 | 43 |
| `adapter_16step_ovd_nms` | `single` | 27 | 0.8276 | 0.8889 | 0.8571 | 1.07 | 1.00 | 5 | 3 |
| `adapter_16step_ovd_nms` | `medium` | 13 | 0.8000 | 0.6061 | 0.6897 | 1.92 | 2.54 | 5 | 13 |
| `adapter_16step_ovd_nms` | `dense` | 10 | 0.8824 | 0.4000 | 0.5505 | 3.40 | 7.50 | 4 | 45 |

这说明 16-step adapter 并没有改变 single/medium 行为；整体 F1 下降完全来自 dense bucket 少召回 2 个目标。当前瓶颈可以更精确地表述为：Florence-VP query grounding 的 single/medium 已经接近可用，但 dense 样本明显 under-generate，平均只输出 3.4-3.6 个框，而 GT 平均 7.5 个框。

下一步优先级因此应调整为：

1. 不再用整体 OVD 50 条作为唯一决策指标，必须同时看 dense recall。
2. 训练采样从 `single:medium:dense = 1:2:3` 进一步提高 dense 权重，或做 dense-only adapter smoke。
3. 推理侧尝试 box-count aware prompt / constrained generation，而不仅是 NMS 后处理，因为后处理无法补回未生成的框。

已进一步生成 dense-only curriculum：

```bash
python3 scripts/data-conversion/build_vp_query_curriculum.py \
  --input .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output .codex_reports/florence_vp_plain_spaced_96step/query_ovd_dense_curriculum_v1.jsonl \
  --single-weight 1 \
  --medium-weight 2 \
  --dense-weight 3 \
  --min-query-boxes 4
```

dense-only v1 统计：

- input rows: 362
- output rows: 192
- skipped rows: 298
- dense source rows: 64
- dense output rows: 192
- `avg_query_box_count_output`: 8.0781
- `max_query_box_count`: 13

同时，训练 smoke/experiment runner 已支持直接过滤 effective grounding train rows：

- `--grounding-min-query-boxes`
- `--grounding-max-query-boxes`

下一轮 dense-only 真实短训可以不预先生成新文件，直接复用 full curriculum：

```bash
python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --query-grounding-preset ovd-nms \
  --run-training \
  --run-policy-sweep \
  --skip-baseline \
  --cleanup-adapter-after-audit \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_dense_only_32step_20260607 \
  --training-steps 32 \
  --max-samples 50 \
  --device mps \
  --shuffle-train-data \
  --shuffle-seed 43
```

### 5.10 Dense-aware policy comparison

为了让后续实验不再被整体 F1 稀释，`compare_vp_quality_reports`、`compare_vp_quality_policies.py`、`sweep_vp_quality_policies.py` 已新增 focus bucket 排序：

- `--focus-bucket single`
- `--focus-bucket medium`
- `--focus-bucket dense`

experiment runner 也新增透传参数：

- `--policy-comparison-focus-bucket`
- `--policy-sweep-focus-bucket`

这意味着下一轮 dense-only adapter 不只会生成整体 policy sweep，还能直接生成 dense-aware 推荐。

已用现有 base 与 16-step adapter 的 bucket quality report 生成 dense-focus 对比：

```bash
python3 scripts/experiments/compare_vp_quality_policies.py \
  --focus-bucket dense \
  --report base_ovd_nms=.codex_reports/florence_vp_box_count_bucket_analysis/base_ovd_nms/vp_detection_quality.json \
  --report adapter_16step_ovd_nms=.codex_reports/florence_vp_box_count_bucket_analysis/adapter_16step_ovd_nms/vp_detection_quality.json \
  --output-dir .codex_reports/florence_vp_box_count_bucket_analysis/base_vs_adapter_dense_focus
```

dense-focus 对比结果：

| rank | policy | overall f1 | dense precision | dense recall | dense f1 | dense avg pred | dense avg GT | dense FP | dense FN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `base_ovd_nms` | 0.6756 | 0.8889 | 0.4267 | 0.5766 | 3.60 | 7.50 | 4 | 43 |
| 2 | `adapter_16step_ovd_nms` | 0.6637 | 0.8824 | 0.4000 | 0.5505 | 3.40 | 7.50 | 4 | 45 |

推荐仍是 `base_ovd_nms`。后续 dense-only 训练的成功标准应改为：`dense recall`、`dense f1`、`dense avg_pred_boxes` 至少有一项超过 base，同时不能让 FP 明显增加。

### 5.11 Dense-only 8-step 真实训练复盘

已按上一节方向实际跑完一次 dense-only OVD grounding LoRA 短训。注意：默认 Python 3.11 加载本机 Florence remote-code 权重时会报 `RuntimeError: Tensor.item() cannot be called on meta tensors`，因此真实训练需显式使用 CLT Python 3.9：

```bash
TMPDIR=/Users/gatilin/PycharmProjects/FlorenceForge/.session_tmps/vp_tmp \
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/experiments/run_florence_vp_training_experiment.py \
  --query-grounding-preset ovd-nms \
  --run-training \
  --run-policy-sweep \
  --skip-baseline \
  --cleanup-adapter-after-audit \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607 \
  --training-steps 8 \
  --max-samples 50 \
  --device mps \
  --shuffle-train-data \
  --shuffle-seed 43 \
  --policy-sweep-focus-bucket dense
```

训练入口与数据状态：

- run root: `.codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607`
- training summary: `.codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607/training/real_florence_vp_training_smoke_summary.json`
- adapter inference: `.codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607/adapter_inference`
- dense policy sweep: `.codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607/policy_sweep/adapter`
- base-vs-adapter dense comparison: `.codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607/base_vs_adapter_dense_focus`
- `first_batch_task_type`: `OPEN_VOCABULARY_DETECTION`
- `steps_executed`: 8
- `final_loss`: 5.24621
- effective train grounding rows: 192
- effective train buckets: `single=0`, `medium=0`, `dense=192`
- effective `avg_query_box_count`: 8.0781
- effective `max_query_box_count`: 13

50 条 OVD query 的 adapter policy sweep 仍未超过 base：

| policy | overall precision | overall recall | overall f1 | dense precision | dense recall | dense f1 | dense avg pred | dense avg GT | dense FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base_ovd_nms` | 0.8444 | 0.5630 | 0.6756 | 0.8889 | 0.4267 | 0.5766 | 3.60 | 7.50 | 43 |
| `dense8_adapter` | 0.8409 | 0.5481 | 0.6637 | 0.8824 | 0.4000 | 0.5505 | 3.40 | 7.50 | 45 |

本次 dense-only 训练证明数据入口已经真正对准 dense query grounding，但 8 step 仍没有改善 dense under-generation；平均预测框从 base 的 3.60 降到 3.40，dense false negatives 从 43 增到 45。也就是说，问题不再是“没训练到 dense 样本”，而更可能是 generation budget、停止行为、box-count prompt、可训练参数或训练强度不足。

空间处理：本次 run 使用 `--cleanup-adapter-after-audit`，adapter 权重目录已自动删除；保留训练 summary、推理可视化、policy sweep 和 dense 对比报告。复核时 `.codex_reports` 下没有残留 `.safetensors/.pt/.pth/.bin` 中间权重。

### 5.12 Dense under-generation 诊断修复

为了避免继续靠猜测调参，本次新增两类推理诊断能力：

1. `visualize_florence_vp_adapter.py` 新增 `--min-query-boxes/--max-query-boxes`，可直接筛选 dense query rows；同时在 summary/record 中记录 `max_new_tokens`、`raw_prediction_token_count`、`raw_loc_token_count`、`generation_budget_hit_ratio`、`dense_generation_budget_near_hit_ratio` 等字段。
2. 新增 `scripts/experiments/sweep_vp_generation_budgets.py`，可批量扫 `max_new_tokens`、`num_beams`，自动接 policy sweep，并输出 dense-focus JSON/Markdown。
3. 新增 `--text-input-template`，用于诊断 box-count-aware prompt，例如 `all {text_input}`；该模板只影响生成 prompt，不改原始 GT 和评估字段。

真实 dense6 小样本诊断：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/experiments/sweep_vp_generation_budgets.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_generation_budget_diagnostic_dense6_20260607 \
  --max-samples 6 \
  --min-query-boxes 4 \
  --max-new-tokens-list 64,128 \
  --device mps \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5 \
  --structured-vp-allowed-labels-field text_input \
  --focus-bucket dense
```

| run | dense F1 | dense recall | dense avg pred | dense avg GT | dense near budget | max raw loc tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tokens64_beams1` | 0.6234 | 0.4706 | 4.33 | 8.50 | 0.1667 | 52 |
| `tokens128_beams1` | 0.5556 | 0.3922 | 3.50 | 8.50 | 0.0000 | 20 |

这个结果排除了一个直觉但错误的修复方向：dense 漏框不是简单的 64-token 硬截断。64 token 在这 6 条 dense 样本上没有硬命中 budget，反而比 128 token 召回更高。因此 `ovd-nms` preset 不应默认改成更长输出；更长 budget 只能作为诊断参数，而不是修复。

进一步测试 `--text-input-template 'all {text_input}'` 后，模型会输出 `all knife`、`all umbrella`、`all person` 这类 phrase label。严格 label matching 下 dense F1 归零；即使这些框有些位置合理，也不能作为默认产品策略。`contains`/label canonicalization 可以作为诊断分支，但不能替代严格 grounding 指标。

更新后的判断：dense under-generation 的主要瓶颈更像是 Florence OVD 解码先验和多实例召回能力本身，而不是 token budget 或简单自然语言前缀。下一步更合理的是做 beam/length penalty 诊断、box-count-conditioned training target、或在结构化 decoder 侧探索可解释但不篡改标签的候选框补全策略。

### 5.13 Generation search 诊断

在上一节基础上，本次进一步将 generation search 参数接入 VP 推理链：

- `visualize_florence_vp_adapter.py`: 新增 `--length-penalty`、`--repetition-penalty`、`--no-repeat-ngram-size`、`--early-stopping`，显式传入时才影响 `model.generate()`。
- `sweep_vp_generation_budgets.py`: 支持 `--length-penalty-list`、`--repetition-penalty-list`、`--no-repeat-ngram-size-list` 的组合 sweep。
- `run_florence_vp_training_experiment.py`: adapter/base 推理也可透传上述 search 参数，并会写入 `experiment_summary.json`。

真实 dense6 beam search 诊断：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/experiments/sweep_vp_generation_budgets.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_generation_search_dense6_beam_20260607 \
  --max-samples 6 \
  --min-query-boxes 4 \
  --max-new-tokens-list 64 \
  --num-beams-list 1,3 \
  --device mps \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5 \
  --structured-vp-allowed-labels-field text_input \
  --focus-bucket dense
```

| run | dense F1 | dense recall | dense avg pred | dense avg GT | dense FP | dense FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `num_beams=1` | 0.7000 | 0.5490 | 4.83 | 8.50 | 1 | 23 |
| `num_beams=3` | 0.5946 | 0.4314 | 3.83 | 8.50 | 1 | 29 |

beam search 在这组 dense 样本上更慢且更保守，平均预测框从 4.83 降到 3.83，dense recall 从 0.5490 降到 0.4314。因此当前 Florence-VP OVD+NMS 默认不应改成 beam search。后续若继续搜 generation 参数，应优先小样本验证 length penalty/repetition penalty，再扩到 50 条，避免把高成本 beam search 当成默认修复。

### 5.14 Count-hint prompt 的宽松标签诊断

上一轮 `--text-input-template 'all {text_input}'` 在严格 label matching 下 F1 为 0，因为模型会输出 `all person`、`all knife` 这类 phrase label。为了区分“只是标签漂移”还是“框召回也变差”，本次将 `sweep_vp_generation_budgets.py` 新增透传：

- `--include-phrase-label-policy`
- `--include-target-label-oracle`

这样同一次 sweep 可以同时保留严格默认指标，并额外生成 `text_input_phrase_allowed` 诊断策略：`allowed_labels_field=text_input`、`label_match_mode=contains`、`allowed_label_match_mode=contains`。

真实 dense6 count-hint prompt 诊断：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/experiments/sweep_vp_generation_budgets.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_prompt_all_dense6_contains_20260607 \
  --max-samples 6 \
  --min-query-boxes 4 \
  --max-new-tokens-list 64 \
  --num-beams-list 1 \
  --text-input-template 'all {text_input}' \
  --device mps \
  --structured-vp-mode auto \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5 \
  --structured-vp-allowed-labels-field text_input \
  --include-phrase-label-policy \
  --focus-bucket dense
```

| prompt/policy | dense F1 | dense recall | dense avg pred | dense avg GT | dense FP | dense FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original query, strict/NMS | 0.7000 | 0.5490 | 4.83 | 8.50 | 1 | 23 |
| `all {text_input}`, phrase-contained diagnostic | 0.4857 | 0.3333 | 3.17 | 8.50 | 2 | 34 |

结论：`all {text_input}` 不只是标签字符串漂移；即使用 contains policy 宽松评估，它的 dense recall 仍明显低于原始 query。因此 count-hint prompt 不应作为默认推理修复。更有希望的方向是训练侧显式构造 box-count-conditioned target 或引入候选框补全诊断，而不是在未训练的 Florence OVD prompt 前追加自然语言数量提示。

### 5.15 多搜索候选框 ensemble 诊断

为了测试“不同 generation 配置是否只是各自漏掉不同实例”，本次新增：

```bash
scripts/experiments/ensemble_vp_inference_summaries.py
```

该脚本从多个 `vp_inference_visualization_summary.json` 中读取同一批样本的 raw prediction，使用 structured VP decoder 解析候选框，然后在不使用 GT 的情况下做 union + NMS，生成新的 ensemble summary，再交给现有 `sweep_vp_quality_policies.py` 评估。

真实 dense6 search-union 诊断：

```bash
python3 scripts/experiments/ensemble_vp_inference_summaries.py \
  --summary greedy64=.codex_reports/florence_vp_generation_search_dense6_beam_20260607/tokens64_beams1_lpdefault_rpdefault_ngramdefault/inference/vp_inference_visualization_summary.json \
  --summary tokens128=.codex_reports/florence_vp_generation_budget_diagnostic_dense6_20260607/tokens128_beams1/inference/vp_inference_visualization_summary.json \
  --summary beam3=.codex_reports/florence_vp_generation_search_dense6_beam_20260607/tokens64_beams3_lpdefault_rpdefault_ngramdefault/inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_ensemble_dense6_search_union_20260607 \
  --structured-vp-marker-style plain \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5 \
  --structured-vp-allowed-labels-field text_input
```

| run | dense F1 | dense recall | dense avg pred | dense avg GT | dense FP | dense FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| greedy64 | 0.7000 | 0.5490 | 4.83 | 8.50 | 1 | 23 |
| search-union ensemble | 0.7000 | 0.5490 | 4.83 | 8.50 | 1 | 23 |

ensemble 未超过 greedy64。虽然输入候选来自 greedy64、tokens128、beam3 三组推理，但 NMS 后保留下来的有效候选与 greedy64 基本一致。这说明当前 dense under-generation 不是“多个搜索配置各自找到不同实例，合并即可补齐”的问题，而是 Florence OVD 在这些 dense query 上本身没有稳定产生足够多的不同候选框。

因此，剩余主线应进一步收敛到训练侧：构造显式 dense/box-count-conditioned target、更长且分段审计的 dense-only 训练、或引入外部 proposal/检测器作为候选框 teacher。推理侧继续靠 beam、长输出、自然语言 `all` 前缀或多搜索 ensemble 的收益都已被小样本诊断压低优先级。

### 5.16 OVD query 训练 prompt 修复与真实小实验

进一步排查训练链路时发现一个关键训练语义问题：`MultiTaskDataset` 已经能从样本构造完整 prompt，例如 `<OPEN_VOCABULARY_DETECTION>knife`，但走 Florence-2 backend 编码路径时只把 `suffix/answer` 作为 `text_input` 传给 `encode_with_task`。由于 Florence-2 processor 要求 task token 独占 processor text，后端此前采用了 `task + answer` 的手动拼接；这会导致 OVD/grounding 样本中的 query label 没有进入模型输入 prompt。

本次修复为 Florence-2 backend 增加显式三段式训练编码：

```text
task token + query prompt + answer suffix
```

其中 `task token + query prompt` 在 labels 中置为 `-100`，只监督 `answer suffix`。旧调用 `text_input=answer` 保持兼容；新训练路径通过 `answer_text=...` 显式区分 query 与 answer。新增单测覆盖：

- `Florence2Backend.encode_with_task(..., text_input=query, answer_text=answer)` 会把 query 放入 `input_ids`，并把 query span mask 为 `-100`。
- `MultiTaskDataset` 在支持三段式编码的后端上会把 JSONL 的 `text_input/query_label` 作为 query，把 `suffix` 作为 answer。
- `probe_florence_vp_tokens.py` 增加 `--task-type` 与 `--manifest-data-key`，可直接审计 OVD grounding split，而不是默认只看 `OD_VP`。
- `run_florence_vp_training_experiment.py --run-token-probe` 在 query grounding preset 下会 probe `train_grounding_effective_path,train_grounding_path`。

真实 Florence processor 验证：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/experiments/probe_florence_vp_tokens.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --manifest-path .codex_reports/florence_vp_ovd_dense_only_8step_py39_20260607/training/vp_real_data_manifest.json \
  --manifest-data-key train_grounding_effective_path,train_grounding_path \
  --task-type OPEN_VOCABULARY_DETECTION \
  --marker-style plain \
  --output-dir .codex_reports/florence_vp_ovd_prompt_answer_probe_20260607 \
  --device mps \
  --torch-dtype float32 \
  --max-samples 2 \
  --max-new-tokens 2
```

结果显示真实 tokenizer 下首个监督 token 已从 `<ref>` 开始，query `knife` 不再进入 loss；plain VP marker sequence 与 loc tokens 都被正确监督。

随后执行了一个极小的真实 LoRA 训练 + 推理 + audit + token probe 实验，并在 audit 后删除 adapter 权重：

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --query-grounding-preset ovd-nms \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_prompt_fix_4step_20260607 \
  --training-steps 4 \
  --max-train-samples 12 \
  --max-val-samples 4 \
  --max-samples 8 \
  --min-inference-samples 4 \
  --visualization-limit 4 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1
```

关键结果：

| item | value |
| --- | ---: |
| training steps | 4 |
| first batch task | `OPEN_VOCABULARY_DETECTION` |
| final loss | 5.8864 |
| trainable param delta norm | 0.0565 |
| adapter/baseline samples | 8 |
| adapter structured valid ratio | 1.0000 |
| adapter structured decoder ratio | 1.0000 |
| adapter avg pred boxes / GT boxes | 2.25 / 2.125 |
| dense samples in this smoke | 1 |
| adapter cleanup | deleted |

这次 4-step smoke 的主要价值不是性能提升，而是确认训练链路已从错误的 `task + answer` 修到正确的 `task + query + answer`。在 8 条推理样本上 adapter 与 baseline 完全一致，说明 4 step 还不足以改变生成行为；audit 状态仍是 `engineering_mvp_ready_needs_wrapper_training`，raw VP 仍未内化，wrapper 仍依赖 structured decoder。

下一步不应继续优先调 beam、token budget 或自然语言 prompt，而应基于修复后的训练链路重新做更长的 dense-only OVD LoRA 分段实验，例如 16/32/64 step，且每段都保留：

- OVD query token/label probe；
- dense bucket F1/recall；
- avg predicted box count vs GT count；
- adapter 与 base 的同样本差异；
- audit 后删除已验证 adapter，避免磁盘继续被中间权重吃掉。

### 5.17 修复后 dense-only 16-step 训练与同样本差异诊断

按上一节方向，已基于真实 Florence 权重与真实 COCO128-derived OVD query 数据完成一次更聚焦的 dense-only 16-step LoRA 小实验，并在 audit/quality/token probe 通过后删除 adapter 权重：

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --run-quality-eval \
  --run-policy-comparison \
  --query-grounding-preset ovd-nms \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --inference-min-query-boxes 4 \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_prompt_fix_dense16_lr2e6_20260607 \
  --training-steps 16 \
  --learning-rate 2e-6 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1
```

训练侧确认已经进入修复后的 OVD grounding 路径：

| item | value |
| --- | ---: |
| training steps | 16 |
| first batch task | `OPEN_VOCABULARY_DETECTION` |
| final loss | 5.3199 |
| trainable param delta norm | 0.2789 |
| adapter cleanup | deleted |

16 条 dense OVD query 的 adapter/base quality 对比如下：

| run | dense F1 | precision | recall | avg pred | avg GT | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 0.5155 | 0.7576 | 0.3906 | 4.125 | 8.000 | 50 | 16 | 78 |
| baseline | 0.5208 | 0.7813 | 0.3906 | 4.000 | 8.000 | 50 | 14 | 78 |

结论非常具体：16-step adapter 没有提升 dense recall；TP/FN 与 baseline 完全相同，只是平均预测框数多了 `0.125`，并额外带来 2 个 FP，所以 policy comparison 推荐仍是 `baseline`。

为避免只看 aggregate F1，又新增了 record-level adapter-vs-baseline 诊断：

```bash
python3 scripts/experiments/compare_vp_quality_records.py \
  --candidate-report .codex_reports/florence_vp_ovd_prompt_fix_dense16_lr2e6_20260607/quality/adapter/vp_detection_quality.json \
  --baseline-report .codex_reports/florence_vp_ovd_prompt_fix_dense16_lr2e6_20260607/quality/baseline/vp_detection_quality.json \
  --candidate-name adapter \
  --baseline-name baseline \
  --focus-bucket dense \
  --output-dir .codex_reports/florence_vp_ovd_prompt_fix_dense16_lr2e6_20260607/record_comparison
```

同样本差异显示：

| metric | value |
| --- | ---: |
| compared dense records | 16 |
| unchanged records | 12 |
| TP-improved records | 2 |
| TP-regressed records | 1 |
| FP-increased records | 2 |
| undergeneration fixed records | 1 |
| delta TP / FP / FN | 0 / +2 / 0 |
| delta F1 | -0.0054 |

changed records 只有 4 条：`person@000000000315` 从 6 TP 掉到 4 TP；`person@000000000074` 多找回 1 个 TP；`person@000000000149` 多找回 1 个 TP 但同时多 1 个 FP；`kite@000000000149` TP 不变但多 1 个 FP。也就是说，这次训练表现更像局部扰动，而不是形成了稳定的 dense multi-instance 召回策略。

工程侧同步新增：

- `compare_vp_quality_record_reports(...)` 与 `render_vp_record_comparison_markdown(...)`，用于直接比较两份 `vp_detection_quality.json` 的逐样本变化。
- `scripts/experiments/compare_vp_quality_records.py`，输出 `vp_record_comparison.json/md`。
- `run_florence_vp_training_experiment.py --run-record-comparison`，让下一轮真实训练可自动生成 adapter-vs-base record diff。

更新后的判断：训练入口已经修正，真实 dense-only 训练也确实改变了部分样本，但 16 step 的变化不稳定且未改善总体 recall。因此下一步不应继续依赖短步数 smoke 判断“方案完备”，而应推进两条更有判别力的方向：

1. 跑更保守的 32-step dense-only 分段训练，例如 LR `1e-6`、同样开启 `--run-record-comparison --record-comparison-focus-bucket dense`，看 TP 改善样本是否持续多于 TP 回退样本。
2. 设计 box-count-conditioned / target-count-aware 的训练目标或 decoder constraint，因为当前失败点不是格式解析，而是 Florence OVD 对 dense query 的生成停止/多实例召回先验。

### 5.18 dense-only 32-step LR 1e-6 复验

已按 5.17 的建议执行一次更保守的 dense-only 32-step LoRA 复验。第一次 32-step run 漏传 `--max-samples 16`，只完成了 2 条评估，因此仅作为训练 smoke 保留；有效结论以第二次 eval16 run 为准：

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --run-quality-eval \
  --run-policy-comparison \
  --run-record-comparison \
  --record-comparison-focus-bucket dense \
  --query-grounding-preset ovd-nms \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --inference-min-query-boxes 4 \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_prompt_fix_dense32_lr1e6_eval16_20260607 \
  --training-steps 32 \
  --learning-rate 1e-6 \
  --max-samples 16 \
  --min-inference-samples 4 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1
```

训练与清理状态：

| item | value |
| --- | ---: |
| training steps | 32 |
| first batch task | `OPEN_VOCABULARY_DETECTION` |
| final loss | 6.8491 |
| trainable param delta norm | 0.2256 |
| train sec / total sec | 92.518 / 114.777 |
| adapter cleanup | deleted |

16 条 dense OVD query 的 adapter/base quality：

| run | dense F1 | precision | recall | avg pred | avg GT | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 0.5155 | 0.7576 | 0.3906 | 4.125 | 8.000 | 50 | 16 | 78 |
| baseline | 0.5155 | 0.7576 | 0.3906 | 4.125 | 8.000 | 50 | 16 | 78 |

policy comparison 的 `recommended_policy` 显示为 `adapter`，但这是同分 tie 下的排序结果，不代表 adapter 取得有效提升。record-level diff 更能说明问题：

| metric | value |
| --- | ---: |
| compared dense records | 16 |
| unchanged records | 14 |
| TP-improved records | 0 |
| TP-regressed records | 0 |
| FP-reduced records | 1 |
| FP-increased records | 1 |
| undergeneration fixed records | 0 |
| delta TP / FP / FN | 0 / 0 / 0 |
| delta F1 | 0.0000 |

也就是说，32-step、LR `1e-6` 比 16-step、LR `2e-6` 更稳定，没有再造成 TP 回退，但也完全没有提升 dense recall；变化只剩两个 precision 方向的抵消样本。token probe 仍为 `generation_prior_blocks_wrapper`，generation probe 解码为 `</s><s> <loc_0><loc_0><loc_998><loc_998></s>`，说明模型仍沿 Florence 原生 loc 先验生成，raw VP wrapper 没有内化。

更新后的结论：

- 修复后的 prompt-answer 训练路径已验证正确，但短程 LoRA 只会造成局部扰动或完全贴近 baseline，不能解决 dense under-generation。
- 在当前 OVD+NMS 路径下，继续把 16/32 step 拉长到 64 step 可能仍然成本高、信息量低；更优先的是改变训练目标，而不是只延长 schedule。
- 下一步应实现 box-count-conditioned target 或 count-aware decoding 诊断：让训练样本显式携带目标框数量/停止条件，再看 raw generation 是否能从平均 4 个框向 GT 8 个框移动。

### 5.19 count-aware query hint 训练目标实现

本轮已把 5.18 的下一步收敛成一个可运行的训练入口：在 grounding 数据进入 Florence prompt 之前，可选择把 query 改写成带目标框数量的形式，例如：

```text
knife | count=5
```

核心实现点：

- `scripts/smoke/real_florence_vp_training_smoke.py` 新增 `--grounding-count-hint-template`，通过 `_apply_query_count_hints(...)` 生成 hinted train/val JSONL。
- hinted JSONL 只改 `text_input`，保留干净的 `query_label` 与 `query_box_count`，避免把 `knife | count=5` 当成类别。
- `scripts/experiments/run_florence_vp_training_experiment.py` 同步新增 `--grounding-count-hint-template`，并在启用 count hint 时默认把 structured VP allow-list 字段切到 `query_label`。

这里最关键的是 label/query 的解耦：训练 prompt 可以是 `label + count`，但评估、NMS/allow-list 与 record comparison 必须继续使用 `query_label`。否则 false positive 过滤会把真实类别和数量提示混在一起，导致 structured decoder 无法正确判断 `knife` 这类类别是否命中。

已做一个 no-model 数据探针，输出目录为 `.codex_reports/florence_vp_count_hint_data_probe_20260607`：

| item | value |
| --- | ---: |
| train dense rows | 192 |
| val rows | 362 |
| first train `text_input` | `knife | count=5` |
| first train `query_label` | `knife` |
| first train `query_box_count` | 5 |

新增/更新的测试覆盖：

- `tests/test_visual_primitive_workflow.py`：验证 CLI parser 接受 count hint 模板，并确认 helper 只改 `text_input`、保留 `query_label`。
- `tests/test_vp_experiment_runner.py`：验证 dry-run 会透传 `--grounding-count-hint-template`，且推理/eval allow-list 使用 `query_label`。

本轮完整 VP 回归已通过：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`139 passed, 1 warning`。

更新后的判断：count-aware prompt 不应直接作为未训练推理 prompt 使用，前面的 `all {text_input}` 诊断已经证明自然语言数量提示会伤害 dense recall；但把数量作为训练侧 query 条件是合理的下一步，因为它把“模型应该继续生成多少个同类框”的停止条件显式暴露给 LoRA。下一步需要跑真实 count-hinted dense-only 训练，并用 record-level diff 判断它是否真的修复 under-generation，而不是只制造更多 FP。

### 5.20 count-hinted dense-only 真实训练复验

已执行一次真实权重 + 真实 COCO128 派生 grounding 数据的 count-hinted dense-only LoRA 训练：

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --run-quality-eval \
  --run-policy-comparison \
  --run-record-comparison \
  --record-comparison-focus-bucket dense \
  --query-grounding-preset ovd-nms \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --grounding-count-hint-template "{label} | count={query_box_count}" \
  --inference-min-query-boxes 4 \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_dense16_lr1e6_20260607 \
  --training-steps 16 \
  --learning-rate 1e-6 \
  --max-samples 16 \
  --min-inference-samples 4 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1
```

训练与清理状态：

| item | value |
| --- | ---: |
| training steps | 16 |
| first batch task | `OPEN_VOCABULARY_DETECTION` |
| final loss | 5.9429 |
| trainable param delta norm | 0.1461 |
| train sec / total sec | 76.979 / 105.619 |
| count-hinted train rows | 192 dense rows |
| count-hinted val rows | 362 rows |
| adapter cleanup | deleted |

16 条 dense OVD query 的 adapter/base quality：

| run | dense F1 | precision | recall | avg pred | avg GT | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 0.0000 | 0.0000 | 0.0000 | 0.000 | 8.000 | 0 | 0 | 128 |
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.000 | 8.000 | 0 | 0 | 128 |

record-level diff：

| metric | value |
| --- | ---: |
| compared dense records | 16 |
| unchanged records | 16 |
| TP-improved records | 0 |
| TP-regressed records | 0 |
| FP-increased records | 0 |
| FP-reduced records | 0 |
| undergeneration fixed records | 0 |
| delta TP / FP / FN | 0 / 0 / 0 |
| delta F1 | 0.0000 |

这个结果比 5.18 的“不提升 recall”更极端：adapter 和 baseline 在 count-hinted val prompt 上全部归零。观察原始生成可以解释原因：

- adapter/baseline 都倾向直接复读 `label | count=N`，再输出 `<poly>...` 或极端边界 loc。
- structured decoder 的 allow-list 已正确使用 `query_label`，但模型输出不再是可解析的 `label<loc_*>` 或 VP `<ref>/<box>` 格式。
- token probe 的 generation probe 解码为 `</s><s> <poly><loc_999><loc_999><loc_999><loc_997></s>`，说明生成先验从 OVD native loc 又滑向 Florence 的 polygon/region 表达，VP marker 仍没有被内化。

因此这次实验的结论不是“数量条件无效”，而是 **`{label} | count=N` 不是合适的 Florence OVD query 形式**。它把 query 分布从 `knife`、`person` 这种开放词汇类别，变成 Florence 会复读和误解的复合文本，导致 baseline 也崩掉。后续 count-aware 路线应避免把 count 直接拼进公开 query：

1. 保持推理 query 干净，只在训练 metadata/辅助 loss 中使用 `query_box_count`。
2. 若必须显式提示数量，改成不参与 label span 的专用 control field，例如独立 task prompt 或 decoder constraint，而不是 `text_input` 中的自然语言后缀。
3. 优先实现 target-count-aware decoder 诊断：生成后按 `query_box_count` 统计缺口，并测试是否可用 proposal/NMS 补全；训练侧则用同一指标作为 bucket loss/audit，而不是直接改 query 字符串。

磁盘清理：本次 runner 已删除 adapter 目录，`.codex_reports` 未留下 `.safetensors/.pt/.pth/.bin` 权重文件；额外清理了 `/private/var/folders/.../T/tmp.ZejtHPr3w4/out/pretrain_64m_baseline` 下两个已验证临时 `.pt` checkpoint，释放约 627MB 权重占用。

### 5.21 train-only count hint 诊断开关

基于 5.20 的负结果，已继续实现一个更可判别的实验开关：`--grounding-count-hint-splits`。

支持值：

- `both`：默认旧行为，train/val 都改写为 count-hinted query。
- `train`：只改训练集 query，验证/推理仍使用干净 `text_input`。
- `val`：只改验证集 query，用于复现 prompt 分布污染诊断。

代码变更：

- `scripts/smoke/real_florence_vp_training_smoke.py` 新增 `_apply_query_count_hints_for_splits(...)` 与 CLI 参数 `--grounding-count-hint-splits`。
- `scripts/experiments/run_florence_vp_training_experiment.py` 透传该参数，并在 experiment summary 记录 `grounding_count_hint_splits`。
- 测试验证 `splits=train` 时只生成 `train_grounding_count_hint_vp.jsonl`，`val_grounding_path` 保持原始干净 JSONL。

回归结果：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`139 passed, 1 warning`。

下一轮真实实验建议不再使用 `both`，而是使用：

```bash
--grounding-count-hint-template "{label} | count={query_box_count}" \
--grounding-count-hint-splits train
```

这样 baseline 的验证 prompt 不会被 `| count=N` 污染，可以判断 adapter 是否在干净 query 下仍保持或改善 dense OVD 召回。如果仍然没有改善，count-aware 路线就应从 text prompt 迁移到 decoder constraint/proposal teacher，而不是继续改自然语言模板。

### 5.22 train-only count hint 真实训练复验

已执行 5.21 建议的真实复验：训练集使用 `"{label} | count={query_box_count}"`，验证/推理集保持干净 query。

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/run_florence_vp_training_experiment.py \
  --run-training \
  --run-token-probe \
  --run-quality-eval \
  --run-policy-comparison \
  --run-record-comparison \
  --record-comparison-focus-bucket dense \
  --query-grounding-preset ovd-nms \
  --grounding-train-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_curriculum_v1.jsonl \
  --grounding-val-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --grounding-min-query-boxes 4 \
  --grounding-count-hint-template "{label} | count={query_box_count}" \
  --grounding-count-hint-splits train \
  --inference-min-query-boxes 4 \
  --cleanup-adapter-after-audit \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607 \
  --training-steps 16 \
  --learning-rate 1e-6 \
  --max-samples 16 \
  --min-inference-samples 4 \
  --device mps \
  --torch-dtype float32 \
  --max-new-tokens 64 \
  --num-beams 1
```

训练与清理状态：

| item | value |
| --- | ---: |
| training steps | 16 |
| first batch task | `OPEN_VOCABULARY_DETECTION` |
| final loss | 5.9062 |
| trainable param delta norm | 0.1501 |
| train sec / total sec | 171.924 / 212.956 |
| count hint splits | `train` |
| count-hinted train rows | 192 dense rows |
| val grounding path | clean `.codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl` |
| adapter cleanup | deleted |

16 条 dense OVD query 的 adapter/base quality：

| run | dense F1 | precision | recall | avg pred | avg GT | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 0.5155 | 0.7576 | 0.3906 | 4.125 | 8.000 | 50 | 16 | 78 |
| baseline | 0.5128 | 0.7463 | 0.3906 | 4.188 | 8.000 | 50 | 17 | 78 |

record-level diff：

| metric | value |
| --- | ---: |
| compared dense records | 16 |
| unchanged records | 15 |
| precision-improved records | 1 |
| TP-improved records | 0 |
| TP-regressed records | 0 |
| FP-reduced records | 1 |
| undergeneration fixed records | 0 |
| delta TP / FP / FN | 0 / -1 / 0 |
| delta F1 | +0.0026 |

这个结果把 5.20 的问题拆开了：

- `both` 模式归零主要是验证 prompt 被 `| count=N` 污染；`train-only` 模式在干净 val 上恢复到正常 OVD+NMS 水平。
- 但 adapter 没有提升 dense recall：TP 仍为 50、FN 仍为 78，唯一变化是 `person@000000000149` 少了 1 个 FP。
- token probe 仍显示 count-hinted train prompt 会进入 `<poly>` 先验：`</s><s> <poly><loc_999><loc_999><loc_999><loc_997></s>`，VP wrapper 仍未内化。

因此，`train-only count hint` 是一个有用的诊断开关，但不是当前 dense under-generation 的修复方案。后续优先级应明确转向：

1. 保持用户 query 干净，不再把 count 拼进 `text_input` 作为默认训练/推理模板。
2. 实现 target-count-aware decoder/proposal 诊断：基于 `query_box_count` 判断缺口，先看外部 proposals 或候选框补全是否能提高 dense recall。
3. 如果继续训练侧 count-aware，应使用独立 control token/task prompt 或 auxiliary loss，而不是自然语言 query 后缀。

本次实验结束后 runner 已删除 adapter，`.codex_reports` 未发现 `.safetensors/.pt/.pth/.bin` 权重残留。

### 5.23 target-count gap oracle/proposal teacher 上界诊断

基于 5.22 的结论，已继续实现 target-count gap 上界分析，用来回答一个更具体的问题：如果不再污染 `text_input`，而是由 decoder/proposal teacher 根据 `query_box_count` 补齐缺失目标，dense recall 理论上还有多少可恢复空间？

新增能力：

- `florence_forge/evaluation/vp_detection_quality.py` 新增 `analyze_vp_target_count_gap(...)` 和 `render_vp_target_count_gap_markdown(...)`。
- 新增 CLI：`scripts/experiments/analyze_vp_target_count_gap.py`。
- 实验 runner 新增 `--run-target-count-gap-analysis`、`--target-count-gap-focus-bucket`、`--target-count-gap-max-rows`、`--target-count-gap-output-dir`。
- runner 已校验 `--run-target-count-gap-analysis` 必须搭配 `--run-quality-eval`，避免没有 quality report 时生成空计划。

分析口径：对每条 VP quality record 计算 `target_box_count - pred_box_count` 的缺口；oracle 假设一个完美 proposal teacher 只补齐这些缺失槽位，且补上的框均可匹配 FN，不额外引入 FP。因此它不是实际性能，而是“数量缺口补全”路线的上界诊断。

对 5.22 的 train-only count hint 实验直接复用已有真实质量报告运行：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/analyze_vp_target_count_gap.py \
  --report .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/quality/adapter/vp_detection_quality.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/adapter \
  --focus-bucket dense

/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/analyze_vp_target_count_gap.py \
  --report .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/quality/baseline/vp_detection_quality.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/baseline \
  --focus-bucket dense
```

输出文件：

- `.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/adapter/vp_target_count_gap.json`
- `.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/adapter/vp_target_count_gap.md`
- `.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/baseline/vp_target_count_gap.json`
- `.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_gap/baseline/vp_target_count_gap.md`

结果：

| run | current F1 | current precision | current recall | oracle F1 | oracle precision | oracle recall | recoverable FN / total FN | target deficit | no-slot blocked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adapter | 0.5155 | 0.7576 | 0.3906 | 0.8880 | 0.8779 | 0.8984 | 65 / 78 | 65 | 3 |
| baseline | 0.5128 | 0.7463 | 0.3906 | 0.8846 | 0.8712 | 0.8984 | 65 / 78 | 65 | 3 |

关键样本显示，dense under-generation 主要不是“定位完全不会”，而是生成停止得太早：

| index | label | pred/target | TP/FP/FN | recoverable TP | oracle delta F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | `chair` | 1/9 | 0/1/9 | 8 | 0.8889 |
| 8 | `person` | 1/10 | 1/0/9 | 9 | 0.8182 |
| 12 | `cup` | 1/10 | 1/0/9 | 9 | 0.8182 |
| 11 | `car` | 3/13 | 3/0/10 | 10 | 0.6250 |
| 2 | `person` | 4/13 | 4/0/9 | 9 | 0.5294 |

这个诊断把下一步方向进一步收窄了：

1. count hint 作为自然语言 query 后缀没有训练收益，但 `query_box_count` 本身非常有诊断价值。
2. dense recall 的主要可恢复空间来自“缺框数量”而非单纯 FP 清洗；65/78 个 FN 可由 count-fill oracle 覆盖，召回缺口闭合比例为 0.8333。
3. 下一步优先实现 target-count-aware decoder/proposal teacher：保持用户 query 干净，先生成或收集候选框，再按 label、IoU/NMS、target count 进行补全。
4. 训练侧如果继续推进，应把 count 作为控制 token、decoder constraint 或 auxiliary loss，而不是拼进 `text_input`。

验证：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_detection_quality.py tests/test_vp_experiment_runner.py
```

结果：`34 passed`。

完整 VP 回归：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`142 passed, 1 warning`。

### 5.24 target-count proposal replay 与高预算候选源复验

5.23 只给出了 oracle 上界。为了判断真实候选源能否吃掉这部分上界，已继续实现离线 proposal replay：

- 新增 CLI：`scripts/experiments/replay_vp_target_count_proposals.py`
- 输入：primary inference summary + proposal inference summary。
- 逻辑：按 `index/image/query_label` 对齐记录，解析 primary/proposal 的结构化 VP 框；对每条样本用 `query_box_count` 作为 target count，只从 proposal 中补入与已有框 label 相同且 IoU 不重复的候选框，直到达到目标数量。
- 输出：merged summary、quality report、markdown replay report、quality markdown。

新增单测覆盖了“primary 缺一个框、proposal 有重复框和新框”的场景，验证 replay 会跳过重复框并补齐目标框：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_filter_replay.py
```

结果：`3 passed`。

#### 5.24.1 adapter + baseline 同源候选复验

先用 5.22 的 adapter inference 作为 primary，用同一实验的 baseline inference 作为 proposal：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/replay_vp_target_count_proposals.py \
  --primary-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --proposal-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/baseline_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_proposals/adapter_from_baseline \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field query_label \
  --primary-filter-policy nms \
  --proposal-filter-policy nms \
  --primary-nms-iou-threshold 0.5 \
  --proposal-nms-iou-threshold 0.5 \
  --duplicate-iou-threshold 0.5
```

结果：

| proposal source | added boxes | deficit before | deficit after | closure | F1 | precision | recall | TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline NMS | 0 | 65 | 65 | 0.0000 | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |
| baseline raw | 0 | 65 | 65 | 0.0000 | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |

结论：同一次 Florence base/adapter 输出之间几乎没有互补候选。baseline 不能作为有效 proposal teacher。

#### 5.24.2 base 高生成预算候选复验

继续跑真实 base model 高预算 inference，把 `max_new_tokens` 从 64 提到 160，不保存 PNG，只输出 summary：

```bash
TMPDIR=.session_tmps/vp_tmp /Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/infer/visualize_florence_vp_adapter.py \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --data-path .codex_reports/florence_vp_plain_spaced_96step/query_ovd_all_vp.jsonl \
  --output-dir .codex_reports/florence_vp_ovd_dense16_base_budget160_20260608/baseline_inference \
  --device mps \
  --torch-dtype float32 \
  --max-samples 16 \
  --min-query-boxes 4 \
  --max-new-tokens 160 \
  --num-beams 1 \
  --structured-vp-decode \
  --structured-vp-mode on \
  --structured-vp-filter-policy nms \
  --structured-vp-nms-iou-threshold 0.5 \
  --structured-vp-allowed-labels-field query_label \
  --structured-vp-marker-style plain \
  --visualization-limit 0
```

高预算 inference summary：

| max tokens | avg pred | avg GT | budget hit | near hit | avg raw tokens | max raw loc tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 160 | 4.125 | 8.000 | 0.0000 | 0.0000 | 21.1875 | 52 |

高预算 quality：

| F1 | precision | recall | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5155 | 0.7576 | 0.3906 | 50 | 16 | 78 |

再把高预算 base summary 作为 proposal 源回放：

| proposal source | added boxes | deficit before | deficit after | closure | F1 | precision | recall | TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base budget160 | 0 | 65 | 65 | 0.0000 | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |

结论进一步明确：dense under-generation 不是 `max_new_tokens=64` 的生成预算问题；在 `max_new_tokens=160` 下模型仍提前停止，平均预测框数、F1、recall 与 64 token 完全一致。后续 target-count-aware 路线需要真正不同机制的候选源，例如：

1. region proposal / selective search / SAM boxes 作为 teacher candidates；
2. Florence REGION_PROPOSAL 或 DENSE_REGION_CAPTION 路径蒸馏出 class-agnostic proposals；
3. 训练侧加入独立 count/control token 或 auxiliary loss，让 decoder 学会继续生成同类实例；
4. 若继续做 ensemble，应使用不同 prompt/task family 的候选，而不是同一 OVD query 的 base/adapter 重采样。

本轮完整 VP 回归：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`143 passed, 1 warning`。

### 5.25 image-proposal teacher 原型与候选池上界复验

5.24 已证明同源 Florence 输出和高生成预算都不能提供额外候选。本轮继续实现一个无需新模型权重的 class-agnostic image proposal teacher，用来验证“不同机制候选源”是否能吃掉 5.23 的 count-fill 上界。

新增 CLI：

- `scripts/experiments/generate_vp_image_proposal_summary.py`

输入为已有 inference summary；脚本逐图读取 `image`、`target`、`query_label`、`query_box_count`，生成候选框并包装成标准 VP inference summary。当前候选源包括：

- multi-scale grid proposals；
- `skimage.segmentation.felzenszwalb` region boxes；
- `skimage.segmentation.slic` superpixel boxes。

支持两种排序：

- `objectness`：不看 GT 的真实候选排序，用面积/aspect/source prior 排序。
- `oracle_iou`：只用于诊断，用 GT IoU 排序，估计候选池本身的可用上界。

新增单测：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_image_proposals.py tests/test_vp_filter_replay.py
```

结果：`4 passed`。

#### 5.25.1 objectness image proposals

对 5.22 的 16 条 dense adapter inference 生成每条 300 个 image proposals：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/generate_vp_image_proposal_summary.py \
  --source-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/objectness_grid_seg \
  --max-proposals-per-record 300 \
  --methods grid,felzenszwalb,slic \
  --rank-policy objectness \
  --structured-vp-marker-style plain
```

候选池覆盖：

| rank policy | avg proposals | avg GT | recall@0.25 | recall@0.50 | recall@0.75 | mean best IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| objectness | 300.0 | 8.0 | 0.3413 | 0.2031 | 0.0204 | 0.2272 |

接入 target-count proposal replay：

| source | added boxes | deficit before | deficit after | closure | F1 | precision | recall | TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image objectness | 65 | 65 | 0 | 1.0000 | 0.4093 | 0.4046 | 0.4141 | 53/78/75 |

这个结果很关键：objectness proposals 能把数量缺口补满，但只多恢复 3 个 TP，同时新增大量 FP，F1 反而从 `0.5155` 降到 `0.4093`。因此“按数量补齐”本身不是解法，必须有更强的 proposal ranking / teacher scoring。

#### 5.25.2 oracle-ranked image proposal pool

再用同一个候选池做 oracle_iou 排序诊断：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/generate_vp_image_proposal_summary.py \
  --source-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/oracle_grid_seg \
  --max-proposals-per-record 300 \
  --methods grid,felzenszwalb,slic \
  --rank-policy oracle_iou \
  --structured-vp-marker-style plain
```

候选池上界覆盖：

| rank policy | avg proposals | avg GT | recall@0.25 | recall@0.50 | recall@0.75 | mean best IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_iou | 300.0 | 8.0 | 0.9187 | 0.6813 | 0.2007 | 0.5774 |

接入 target-count proposal replay 后：

| source | added boxes | deficit before | deficit after | closure | F1 | precision | recall | TP/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| image oracle_iou | 65 | 65 | 0 | 1.0000 | 0.7413 | 0.7328 | 0.7500 | 96/35/32 |

这给出了一个比 5.23 更现实的分解：

- count-fill oracle：F1 `0.8880`，recall `0.8984`，这是“完美补框”的理论上界。
- image proposal pool oracle：F1 `0.7413`，recall `0.7500`，说明轻量 grid+seg 候选池中确实有可用框，但覆盖还不够满。
- image proposal objectness：F1 `0.4093`，recall `0.4141`，说明当前无监督排序不能直接用于补框。

因此下一步不应继续堆普通 grid/seg proposals，而应做更强的 teacher ranking：

1. 引入 class-agnostic proposal scorer：基于 Florence visual features、CLIP/SigLIP 相似度或轻量 objectness classifier 对候选排序。
2. 尝试 Florence `REGION_PROPOSAL` / `DENSE_REGION_CAPTION` 作为候选源，再用 query label 做二阶段过滤。
3. 若允许外部模型，优先接 SAM/EdgeBoxes/Detic/GroundingDINO 这类强 proposal teacher，再蒸馏到 Florence-VP。
4. 训练侧应把“补框质量”作为 auxiliary target，而不是只学习 count 或盲目补齐。

本轮完整 VP 回归：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_vp_image_proposals.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`144 passed, 1 warning`。

### 5.26 proposal candidates 元数据与保守补框策略

5.25 证明了 image proposal pool 有上界，但盲目按 target count 补满会制造大量 FP。本轮继续把 image proposal teacher 和 target-count replay 从“文本回放”推进到“候选级可控回放”，重点验证一个更保守的问题：

> Florence-VP 是否可以只在高置信、低风险候选上补少量框，从而同时提升 dense recall 与整体 F1？

#### 5.26.1 实现更新

本轮新增/优化：

1. `scripts/experiments/generate_vp_image_proposal_summary.py`
   - 每条记录新增 `proposal_candidates`，候选保留 `label`、`bbox`、`confidence`、`proposal_source`、`proposal_area_ratio`、`proposal_edge_density`、`proposal_contrast`。
   - 新增 `edge_density` rank policy，用整图边缘图和灰度图计算候选边缘密度/对比度。
   - 将边缘特征从“每个候选重复 sobel”优化为“每张图预计算一次 sobel map”，再按候选框切片统计。

2. `scripts/experiments/replay_vp_target_count_proposals.py`
   - `_decode_record` 优先读取 `proposal_candidates`，旧 summary 仍兼容 `raw_prediction/structured_prediction` 文本解码。
   - 新增 proposal selection/filter CLI：
     - `--proposal-selection-policy {source_order,confidence,edge_density,area_small,area_large}`
     - `--proposal-min-confidence`
     - `--proposal-allowed-sources`
     - `--max-proposal-additions-per-record`
   - 这让 replay 可以表达“只用 slic 候选、按 edge density 排序、每条样本最多补 1 个高分框”这样的保守策略。

新增/更新单测：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_image_proposals.py tests/test_vp_filter_replay.py
```

结果：`6 passed`。

#### 5.26.2 edge-density proposal 复验

用真实 dense16 adapter summary 重新生成 edge-density proposals：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/generate_vp_image_proposal_summary.py \
  --source-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/edge_density_grid_seg \
  --max-proposals-per-record 300 \
  --methods grid,felzenszwalb,slic \
  --rank-policy edge_density \
  --structured-vp-marker-style plain
```

候选池覆盖：

| rank policy | avg proposals | avg GT | recall@0.25 | recall@0.50 | recall@0.75 | mean best IoU | time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| edge_density | 300.0 | 8.0 | 0.3773 | 0.1967 | 0.0125 | 0.2311 | 77.42s |

edge-density 本身没有改善候选覆盖，说明它不是一个足够强的 teacher scorer。但结构化候选元数据让 replay 可以做更细的 source/threshold/cap 消融。

#### 5.26.3 保守 replay 结果

基准 adapter 结果仍为：

| source | F1 | precision | recall | TP/FP/FN |
| --- | ---: | ---: | ---: | ---: |
| adapter | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |
| baseline | 0.5128 | 0.7463 | 0.3906 | 50/17/78 |

本轮 replay 对比：

| proposal summary | replay strategy | added | deficit after | F1 | precision | recall | TP/FP/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| objectness | blind fill, confidence order | 65 | 0 | 0.4093 | 0.4046 | 0.4141 | 53/78/75 |
| edge_density | blind fill, edge order | 65 | 0 | 0.3938 | 0.3893 | 0.3984 | 51/80/77 |
| edge_density | slic only, edge order | 33 | 32 | 0.4670 | 0.5354 | 0.4141 | 53/46/75 |
| edge_density | slic, min_conf=0.725, edge order | 8 | 57 | 0.5149 | 0.7027 | 0.4063 | 52/22/76 |
| edge_density | slic, min_conf=0.725, edge order, max_add=1 | 6 | 59 | 0.5200 | 0.7222 | 0.4063 | 52/20/76 |
| objectness | slic, min_conf=1.25, edge order, max_add=1 | 7 | 58 | 0.5174 | 0.7123 | 0.4063 | 52/21/76 |

最佳真实组合：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/experiments/replay_vp_target_count_proposals.py \
  --primary-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --proposal-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/edge_density_grid_seg/vp_image_proposal_summary.json \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_proposals/adapter_from_image_edge_slic_min0725_max1 \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field query_label \
  --primary-filter-policy nms \
  --proposal-filter-policy none \
  --primary-nms-iou-threshold 0.5 \
  --duplicate-iou-threshold 0.5 \
  --proposal-selection-policy edge_density \
  --proposal-allowed-sources slic \
  --proposal-min-confidence 0.725 \
  --max-proposal-additions-per-record 1
```

这个组合从 adapter 的 `50/16/78` 推进到 `52/20/76`：

- recall: `0.3906 -> 0.4063`
- F1: `0.5155 -> 0.5200`
- 只补 6 个框，新增 2 个 TP 和 4 个 FP

结论也更明确：

1. “补满 target count”会系统性伤害 precision。
2. “保守补框”是当前 image proposal teacher 的可用形态，尤其是 source gating + score threshold + per-record cap。
3. 现有无监督 edge/objectness 分数只能带来很小收益，真实突破需要一个可校准的 teacher scorer。
4. 下一步应从 replay 走向训练侧：把 `proposal_candidates` 变成 distillation 数据，训练 Florence-VP 学习“何时补、补哪个”，而不是只学习“应该有几个”。

本轮完整 VP 回归：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_ensemble_summaries.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_generation_budget_sweep.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_token_probe.py \
  tests/test_vp_detection_quality.py \
  tests/test_vp_filter_replay.py \
  tests/test_vp_image_proposals.py \
  tests/test_cli_inference_helpers.py \
  tests/test_visual_primitive_workflow.py \
  tests/test_structured_vp_decoder.py \
  tests/test_backend.py \
  tests/test_data_pipeline.py \
  tests/test_dataset_cache.py
```

结果：`146 passed, 1 warning`。

### 5.27 proposal replay -> distillation JSONL 训练链路

5.26 的 replay 证明“保守补框”可以在 dense OVD 上带来小幅真实收益。本轮继续把这条推理侧策略推进到训练侧，新增 proposal distillation 数据构造工具，让 Florence-VP 可以学习“何时补、补哪个”，而不是只在推理后处理阶段被动补框。

新增 CLI：

- `scripts/data-conversion/build_vp_proposal_distillation.py`

输入：

- primary inference summary；
- image proposal summary；
- 与 5.26 相同的 proposal selection/filter 参数。

输出：

- 可直接被 `MultiTaskDataset` / real training smoke 加载的 JSONL；
- JSON summary；
- Markdown audit。

关键过滤参数：

- `--min-added-boxes`: 默认只保留实际补过框的样本；
- `--quality-filter {none,non_regression,improvement}`:
  - `none`: 所有补框样本都写入；
  - `non_regression`: teacher 单样本 F1 不低于 primary，且 TP 不减少；
  - `improvement`: teacher 单样本 F1 或 TP 有提升；
- `--max-proposal-additions-per-record`: 与 replay 一致，支持保守蒸馏。

#### 5.27.1 单测与格式验证

新增测试：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_proposal_distillation.py tests/test_vp_image_proposals.py tests/test_vp_filter_replay.py
```

结果：`8 passed`。

#### 5.27.2 真实 dense16 distillation 数据

使用 5.26 的最佳策略生成 improvement 版蒸馏数据：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/data-conversion/build_vp_proposal_distillation.py \
  --primary-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --proposal-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/edge_density_grid_seg/vp_image_proposal_summary.json \
  --output .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/edge_slic_min0725_max1_improvement.jsonl \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field query_label \
  --primary-filter-policy nms \
  --proposal-filter-policy none \
  --primary-nms-iou-threshold 0.5 \
  --duplicate-iou-threshold 0.5 \
  --proposal-selection-policy edge_density \
  --proposal-allowed-sources slic \
  --proposal-min-confidence 0.725 \
  --max-proposal-additions-per-record 1 \
  --quality-filter improvement \
  --min-added-boxes 1
```

结果：

| distill set | rows | added boxes | delta TP | delta FP | delta FN | labels | bucket |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| improvement | 2 | 2 | +2 | 0 | -2 | person x2 | dense |
| non_regression | 3 | 3 | +2 | +1 | -2 | person x2, cup x1 | dense |

improvement 版很小，但质量干净：只保留真实提升样本，不把 5.26 中的 FP 补框写进训练目标。两条样本均能被 `VisualPrimitiveParser` 解析，且 `MultiTaskDataset` 可直接加载：

```text
rows=2, parsed_box_counts=[2, 2], dataset_len=2,
first_prompt=<OPEN_VOCABULARY_DETECTION>person
```

#### 5.27.3 真实权重训练链路 smoke

继续用 improvement JSONL 跑 1 step 真实训练 smoke，不保存 adapter/checkpoint：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/smoke/real_florence_vp_training_smoke.py \
  --dataset-root /Users/gatilin/PycharmProjects/datasets/coco128 \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/training_smoke_improvement_1step \
  --training-mode lora \
  --lora-r 2 \
  --lora-alpha 4 \
  --lora-target-modules q_proj v_proj \
  --vp-box-format loc_tokens \
  --vp-marker-style plain \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --max-steps 1 \
  --learning-rate 1e-6 \
  --include-grounding \
  --grounding-train-path .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/edge_slic_min0725_max1_improvement.jsonl \
  --grounding-task-type OPEN_VOCABULARY_DETECTION \
  --training-data-order grounding-first \
  --device auto \
  --torch-dtype float32
```

结果：

| item | value |
| --- | ---: |
| ok | true |
| device | mps |
| dataset size | 3 |
| distillation train rows | 2 |
| first batch task | OPEN_VOCABULARY_DETECTION |
| first batch answer | `<ref>person</ref> ...` |
| steps | 1 |
| final loss | 5.1408 |
| grad norm | 220.1410 |
| trainable params | 110,592 |
| trainable delta norm | 0.001411 |

本轮没有保存 adapter，distillation 输出目录约 `132K`，未产生 `.safetensors/.pt/.pth/.bin` 权重文件。

结论：

1. proposal replay 已经可以转成真实 Florence-VP 训练数据；
2. 高质量 improvement 蒸馏样本很少，说明当前 teacher scorer 仍弱，不能直接靠这 16 条训练出稳定提升；
3. 下一步应扩大候选 teacher 覆盖：更多 dense OVD 样本、更强 proposal scorer，或接入 SAM/GroundingDINO/Florence region tasks 生成更高召回候选；
4. 训练侧建议用 `improvement` 集作为 hard-positive seed，用 `non_regression` 或更大 teacher 数据作辅助混合，但不要使用 blind-fill pseudo labels。

本轮完整 VP 回归：`148 passed, 1 warning`。

### 5.28 reference-target hard-positive mix

5.27 的第一版 distillation JSONL 使用 teacher replay 后的 partial prediction 作为 `suffix`。这验证了训练链路，但并不适合作为长期训练目标：teacher 只补少量框，若直接监督 partial target，可能让模型学习欠生成。

本轮做了两个训练侧优化：

1. `build_vp_proposal_distillation.py` 新增 `--distillation-target-mode {teacher,reference}`。
   - `teacher`: 输出 replay 后的 teacher prediction。
   - `reference`: 仍用 proposal improvement 选择 hard example，但训练 `suffix` 改用完整 GT `target`。
2. 新增 `scripts/data-conversion/build_vp_distillation_mix.py`。
   - 将 base dense curriculum 与 high-quality distillation rows 混合。
   - 支持 distillation repeat、同 key base replacement、shuffle 和审计 summary。

新增/更新测试：

```bash
TMPDIR=.session_tmps/vp_tmp python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  --basetemp=.session_tmps/vp_pytest \
  tests/test_vp_proposal_distillation.py \
  tests/test_vp_distillation_mix.py \
  tests/test_vp_filter_replay.py \
  tests/test_vp_image_proposals.py
```

结果：`10 passed`。

#### 5.28.1 reference-mode distillation

基于 5.26 的最佳 replay 策略重新生成 reference target 版：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/data-conversion/build_vp_proposal_distillation.py \
  --primary-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/adapter_inference/vp_inference_visualization_summary.json \
  --proposal-summary .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals/edge_density_grid_seg/vp_image_proposal_summary.json \
  --output .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/edge_slic_min0725_max1_improvement_reference.jsonl \
  --structured-vp-marker-style plain \
  --structured-vp-allowed-labels-field query_label \
  --primary-filter-policy nms \
  --proposal-filter-policy none \
  --proposal-selection-policy edge_density \
  --proposal-allowed-sources slic \
  --proposal-min-confidence 0.725 \
  --max-proposal-additions-per-record 1 \
  --quality-filter improvement \
  --distillation-target-mode reference
```

结果仍然只选出 2 条 true-improvement hard positives，但训练目标改为完整 GT：

| set | rows | selected by | target mode | output box counts | delta TP/FP/FN |
| --- | ---: | --- | --- | --- | ---: |
| improvement_reference | 2 | proposal replay improvement | reference GT | 10, 6 | +2 / 0 / -2 |

#### 5.28.2 dense curriculum + hard-positive mix

将 192 行 dense curriculum 与 2 行 reference hard positives 混合，hard positives 重复 12 次，并替换同 image/query 的 base rows：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/data-conversion/build_vp_distillation_mix.py \
  --base-input .codex_reports/florence_vp_plain_spaced_96step/query_ovd_dense_curriculum_v1.jsonl \
  --distillation-input .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/edge_slic_min0725_max1_improvement_reference.jsonl \
  --output .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/query_ovd_dense_plus_reference_distill_r12.jsonl \
  --base-repeat 1 \
  --distillation-repeat 12 \
  --distillation-min-delta-tp 1 \
  --distillation-target-mode reference \
  --replace-base-on-distillation-key \
  --shuffle \
  --seed 7
```

混合集统计：

| metric | value |
| --- | ---: |
| base input rows | 192 |
| distillation input rows | 2 |
| skipped/replaced base rows | 6 |
| output rows | 210 |
| base output rows | 186 |
| distillation output rows | 24 |
| distillation ratio | 0.1143 |
| avg query boxes | 8.0714 |
| bucket | dense only |

可用性验证：

```text
rows=210, dataset_len=210, min_boxes=4, max_boxes=13,
avg_boxes=8.0714, mix_groups={base:186, distillation:24}
```

#### 5.28.3 mixed JSONL 真实训练 smoke

不保存 adapter/checkpoint，验证 mixed train JSONL 能进入真实 Florence-VP forward/backward：

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 \
  scripts/smoke/real_florence_vp_training_smoke.py \
  --dataset-root /Users/gatilin/PycharmProjects/datasets/coco128 \
  --model-path /Users/gatilin/Downloads/Florence2_det_base_ovd-v3-1751283651704-model \
  --output-dir .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/training_smoke_mix_reference_1step \
  --training-mode lora \
  --lora-r 2 \
  --lora-alpha 4 \
  --lora-target-modules q_proj v_proj \
  --vp-box-format loc_tokens \
  --vp-marker-style plain \
  --max-train-samples 1 \
  --max-val-samples 1 \
  --max-steps 1 \
  --learning-rate 1e-6 \
  --include-grounding \
  --grounding-train-path .codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/query_ovd_dense_plus_reference_distill_r12.jsonl \
  --grounding-task-type OPEN_VOCABULARY_DETECTION \
  --training-data-order grounding-first \
  --device auto \
  --torch-dtype float32
```

结果：

| item | value |
| --- | ---: |
| ok | true |
| device | mps |
| dataset size | 211 |
| train grounding rows | 210 |
| first batch task | OPEN_VOCABULARY_DETECTION |
| steps | 1 |
| final loss | 6.6415 |
| grad norm | 221.3501 |
| trainable params | 110,592 |
| trainable delta norm | 0.001411 |

本轮结论：

1. proposal distillation 现在有两种语义：`teacher` 用于复现后处理行为，`reference` 用于 hard-example mining + 完整监督训练。
2. 当前更推荐 `reference` 模式，因为 COCO/OVD 数据本身有完整 GT，没必要用 partial teacher target 教模型欠生成。
3. mixed JSONL 已能进入真实训练链路；下一步应在磁盘允许时跑一个短训 adapter，并用原 dense16 validation 复验是否超过 5.22 的 adapter F1 `0.5155`。
4. 由于磁盘空间已接近上限，本轮只保存 JSONL/summary，不保存 adapter 权重。

本轮完整 VP 回归：`150 passed, 1 warning`。

### 5.29 encoder-decoder 标签修复、grounding-only overfit 与动态 count cap

本轮继续推进 Florence-VP 的真实权重训练诊断，核心目标是区分三件事：

1. adapter 是否真的参与训练和生成。
2. dense hard-positive 监督是否能让模型输出多框。
3. 当前欠生成/过生成瓶颈是在训练、wrapper 起始，还是解码约束。

#### 5.29.1 训练标签路径修复

发现 Florence-2 是 encoder-decoder 架构，旧训练路径把 `task + query + answer` 都放进 encoder `input_ids`，再用 masked label 监督。这会让 generation 只看到 `task + query` 时与训练形态不一致。

已修复 `Florence2Backend.encode_with_task`：

- encoder `input_ids`: 只包含 `task + query + eos`。
- decoder `labels`: 只包含 `answer + eos`。
- `prepare_labels` 优先返回显式 `labels`。

真实 1-step smoke 验证后的关键形状：

| tensor | shape |
| --- | --- |
| input_ids | `[1, 10]` |
| labels | `[1, 54]` |
| pixel_values | `[1, 3, 768, 768]` |

同时将 VP LoRA smoke/experiment runner 默认 PEFT task type 改为 `SEQ_2_SEQ_LM`，并在 summary 中记录 `lora_task_type`，避免 encoder-decoder 训练继续隐式使用 `CAUSAL_LM`。

#### 5.29.2 hard-first mix 与短训结果

新增 distillation mix 的放置控制：

- `--placement {append,prepend,interleave}`
- `--distillation-repeat-order {grouped,round_robin}`

生成 hard-first r32 round-robin mix：

| metric | value |
| --- | ---: |
| output rows | 250 |
| base rows | 186 |
| distillation rows | 64 |
| distillation ratio | 0.256 |
| placement | prepend |
| repeat order | round_robin |

但 24-step label-fix run 仍与原 dense16 完全一致：

| run | F1 | P | R | TP/FP/FN |
| --- | ---: | ---: | ---: | --- |
| original dense16 | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |
| r32 prepend 24-step label-fix | 0.5155 | 0.7576 | 0.3906 | 50/16/78 |

结论：label fix 是必要修复，但 24-step/`1e-6` 训练强度不足以改变 dense eval generation。

#### 5.29.3 2-row overfit 诊断

为排除大数据 mix 干扰，构造仅 2 条 hard-positive reference target 的 overfit 数据：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/edge_slic_min0725_max1_improvement_reference_only2.jsonl
```

逐步诊断结果：

| run | dataset | task type | steps/lr | final loss | delta norm | F1 | TP/FP/FN | 结论 |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 48-step label fix | 3 | implicit old | 48 / 1e-5 | 6.7572 | 2.4509 | 0.2222 | 2/0/14 | 输出仍 1 框 |
| 48-step seq2seq | 3 | SEQ_2_SEQ_LM | 48 / 1e-5 | 6.5700 | 2.5081 | 0.2222 | 2/0/14 | task type 正确但仍太弱 |
| 96-step grounding-only | 2 | SEQ_2_SEQ_LM | 96 / 1e-4 | 0.2038 | 28.1978 | 0.1111 | 1/1/15 | adapter 确实改变坐标，但 greedy 仍一框即停 |
| 96-step grounding-only + forced prefix | 2 | SEQ_2_SEQ_LM | 96 / 1e-4 | 0.1746 | 30.3156 | 0.6500 | 13/11/3 | 可多框，但明显过生成 |
| forced prefix + query-count cap replay | 2 | eval replay | n/a | n/a | n/a | 0.7500 | 12/4/4 | 数量对齐，过生成率 0 |

关键发现：

1. grounding-only 强 overfit 能把 loss 从约 `6.45` 压到 `0.20`，说明训练链路与 adapter 加载链路是有效的。
2. 普通 greedy generation 仍倾向 Florence 原生 OVD 起始格式 `person<loc...>`，并在 1 个框后停止；这不是 adapter 完全无效，而是原生 OVD prior 压住了 VP wrapper 与多框继续生成。
3. forced decoder prefix `<ref>{label}</ref> <box>` 能让同一训练方案输出多框序列，第一条达到 `9/10` TP，第二条达到 `4/6` TP。
4. forced prefix 会过生成；按 `query_box_count` 动态截断后，box count exact match 达到 `1.0`，F1 从 `0.65` 提升到 `0.75`。

#### 5.29.4 新增实现

新增/优化：

- `scripts/smoke/real_florence_vp_training_smoke.py`
  - 默认 `--lora-task-type SEQ_2_SEQ_LM`。
  - summary 记录 `lora_task_type`。
  - 新增 `--skip-od-training-data`，支持 grounding-only overfit，不再强制混入 OD_VP train rows。
- `scripts/experiments/run_florence_vp_training_experiment.py`
  - 透传 `--lora-task-type`、`--skip-od-training-data`。
  - 新增 `--structured-vp-max-total-boxes-field`。
- `scripts/infer/visualize_florence_vp_adapter.py`
  - 支持从 per-record 字段动态解析 `max_total_boxes`，例如 `query_box_count`。
- `florence_forge/evaluation/vp_detection_quality.py`
  - `VPDetectionQualityConfig.max_total_boxes_field`。
  - 离线 quality replay 支持动态 count cap。
- `florence_forge/evaluation/vp_training_audit.py`
  - audit 记录 grounding rows、`skip_od_training_data`、`lora_task_type`。
  - `data_ready` gate 支持 grounding-only 数据。
- `scripts/experiments/sweep_vp_quality_policies.py`
  - policy sweep 支持动态 count cap 字段。
- `scripts/experiments/sweep_vp_generation_budgets.py`
  - generation budget sweep 透传动态 count cap 字段。

本轮 focused tests：

```bash
python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  tests/test_vp_detection_quality.py \
  tests/test_vp_inference_visualization_helpers.py \
  tests/test_vp_experiment_runner.py \
  tests/test_vp_generation_budget_sweep.py
```

结果：`48 passed, 1 warning`。

```bash
python3 -m pytest -c /dev/null --rootdir=. -p no:cacheprovider \
  tests/test_backend.py::TestFlorence2PromptAnswerEncoding \
  tests/test_data_pipeline.py::TestMultiTaskDataset::test_backend_prompt_answer_encoding_receives_query_and_answer_separately \
  tests/test_dataset_cache.py::TestDatasetCache::test_backend_encoding_path_writes_disk_cache \
  tests/test_vp_training_audit.py
```

结果：`9 passed, 1 warning`。

磁盘与权重清理：

- 所有本轮真实训练 run 均启用或完成 adapter cleanup。
- `.codex_reports` 下未残留 `.safetensors/.pt/.pth/.bin` 中间权重。
- 本轮仅保留 JSON/Markdown/report 产物。

#### 5.29.5 更新后的判断

当前 Florence-VP 已经不再是“训练链路不确定”的状态。更准确的判断是：

1. VP 训练链路可用。
2. LoRA adapter 可改变真实 Florence-2 生成。
3. 强 overfit 能学习 dense reference 坐标序列。
4. 仍未完备的是自由 greedy 生成：模型尚未稳定内化 VP wrapper 起始与停止边界。
5. 最有希望的下一步不是盲目延长 dense training，而是组合：
   - decoder prefix / constrained decoding；
   - query-count 动态 cap；
   - count-hint 或 target-count-aware training；
   - 对 overfit 成功策略做 16-row dense eval。

---

### 5.30 end-to-end qcap 最优结果与生成期 count-stop 诊断

#### 5.30.1 最优 2-row overfit recipe

当时 2-row dense overfit 的最好可复现实验是：

```text
.codex_reports/florence_vp_overfit_reference_only2_groundingonly_96step_lr1e4_forcedprefix_qcap_20260608
```

核心设置：

- `SEQ_2_SEQ_LM`
- grounding-only hard-positive rows
- `--decoder-prefix '<ref>{label}</ref> <box>'`
- `--structured-vp-filter-policy nms`
- `--structured-vp-max-total-boxes-field query_box_count`
- `--structured-vp-allowed-labels-field query_label`
- `--cleanup-adapter-after-audit`

结果：

- training final loss: `0.2121`
- trainable delta norm: `28.8352`
- quality F1 / precision / recall: `0.8750 / 0.8750 / 0.8750`
- TP / FP / FN: `14 / 2 / 2`
- avg pred boxes / avg GT boxes: `8.0 / 8.0`
- box count exact match ratio: `1.0`
- overgeneration ratio: `0.0`
- prediction sources: `visual_primitive=1`, `florence_native=1`
- row 0 (`000000000110.jpg|person`): `10/0/0`, F1 `1.0`
- row 1 (`000000000488.jpg|person`): `4/2/2`, F1 `0.6667`

这说明 qcap 后的剩余错误已经不再是“能不能给够框数”，而主要是局部定位、重复/顺序和 wrapper 内化问题。

#### 5.30.2 新增生成期 VP count stopping

新增实现：

- `scripts/infer/visualize_florence_vp_adapter.py`
  - 新增 `VPBoxCountStoppingCriteria`。
  - 新增 `--stop-after-vp-max-total-boxes`。
  - 使用 tokenizer 中的 `<loc_0>` 到 `<loc_999>` token id 计数。
  - 每 4 个 loc token 视为 1 个 box。
  - 目标框数来自 `--structured-vp-max-total-boxes` 或 per-record `--structured-vp-max-total-boxes-field`。
  - summary 记录 `vp_count_stopping_available_ratio`、`vp_count_stopping_targeted_ratio`、`vp_count_stopping_triggered_ratio`。
- `scripts/experiments/run_florence_vp_training_experiment.py`
  - 透传 `--stop-after-vp-max-total-boxes`。
- `scripts/experiments/sweep_vp_generation_budgets.py`
  - generation budget sweep 透传并记录 count-stop 配置。

这个优化的定位是“减少已经达到目标 count 后的无效尾巴”，不是替代训练本身。它应该在模型过生成时生效；如果模型自己提前 EOS，则不会补框。

#### 5.30.3 count-stop 真实实验诊断

真实 96-step grounding-only + forced-prefix + qcap + count-stop 实验：

```text
.codex_reports/florence_vp_overfit_reference_only2_groundingonly_96step_lr1e4_forcedprefix_qcap_stop_20260608
```

结果：

- training final loss: `0.2855`
- trainable delta norm: `27.2884`
- count-stop available ratio: `1.0`
- count-stop targeted ratio: `1.0`
- avg target boxes: `8.0`
- avg raw loc token count: `24.0`
- max raw loc token count: `32`
- generation budget hit ratio: `0.0`
- quality F1 / precision / recall: `0.7200 / 1.0000 / 0.5625`
- TP / FP / FN: `9 / 0 / 7`
- avg pred boxes / avg GT boxes: `4.5 / 8.0`
- undergeneration ratio: `1.0`
- adapter cleanup: `deleted=true`

解释：

- 这一轮不是 count-stop 带来的提升，而是一个负向诊断样本。
- 两条样本都没有达到目标 loc-token 数：目标分别为 `40` 和 `24` 个 loc token，实际 raw loc token 数为 `32` 和 `16`。
- 因此模型是自己提前结束或未继续生成，而不是被 count-stop 截断。
- 这把下一步问题重新指向“防欠生成/稳定继续生成到 query count”，而不是“进一步压缩 token budget”。

#### 5.30.4 更新后的下一步（已推进）

下一步优先级调整为：

1. ~~对同一个 adapter 做 stop/no-stop 双推理对照，避免训练随机性干扰 count-stop 结论。~~ 已在 `5.31` 完成。
2. 加入 min-count continuation 诊断：当 `raw_loc_box_count < query_box_count` 时，记录欠生成样本并回放更高 `max_new_tokens`、禁用 EOS 或 continuation prompt。
3. 在 dense16 上评估当前 best qcap recipe，而不是直接采用 count-stop。
4. 针对 wrapper 内化继续训练：减少 `structured_vp_decoder_ratio`，让 raw VP valid ratio 接近 structured valid ratio。

---

### 5.31 同 adapter stop/no-stop 双推理对照

#### 5.31.1 目的

上一轮 count-stop 真实实验混入了训练随机性：使用 count-stop 的 run 和不使用 count-stop 的 run 并不是同一个 adapter，因此不能判断 `--stop-after-vp-max-total-boxes` 本身是否有收益。本轮重新训练一次 2-row grounding-only adapter，并在同一个 adapter 上分别跑 no-stop 与 stop 推理。

实验目录：

```text
.codex_reports/florence_vp_count_stop_paired_reference_only2_20260609
```

训练设置仍为：

- `SEQ_2_SEQ_LM`
- grounding-only 2-row reference overfit
- `--decoder-prefix '<ref>{label}</ref> <box>'`
- `--structured-vp-filter-policy nms`
- `--structured-vp-max-total-boxes-field query_box_count`
- `--structured-vp-allowed-labels-field query_label`

训练结果：

- steps: `96`
- final loss: `0.1661`
- trainable delta norm: `28.4188`
- adapter 临时目录大小约 `625MB`，完成对照后已删除。

#### 5.31.2 no-stop vs stop 结果

两套推理的 quality 完全一致：

| setting | F1 | precision | recall | TP/FP/FN | avg pred / GT | exact count | overgen | undergen |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| no-stop | `0.9333` | `1.0000` | `0.8750` | `14/0/2` | `7.0 / 8.0` | `0.5` | `0.0` | `0.5` |
| stop | `0.9333` | `1.0000` | `0.8750` | `14/0/2` | `7.0 / 8.0` | `0.5` | `0.0` | `0.5` |

per-record 对比：

- row 0 (`000000000110.jpg|person`, GT `10`):
  - no-stop raw loc boxes: `11`
  - stop raw loc boxes: `10`
  - stop triggered: `true`
  - quality: both `10/0/0`, F1 `1.0`
  - 解释：count-stop 成功截掉 1 个重复尾框，使 structured filter 从 `1` 降到 `0`，但 qcap/NMS 后质量不变。
- row 1 (`000000000488.jpg|person`, GT `6`):
  - no-stop raw loc boxes: `4`
  - stop raw loc boxes: `4`
  - stop triggered: `false`
  - quality: both `4/0/2`, F1 `0.8`
  - 解释：模型自己提前结束，count-stop 无法补框。

生成诊断：

- no-stop avg raw loc tokens: `30.0`
- stop avg raw loc tokens: `28.0`
- no-stop max raw loc tokens: `44`
- stop max raw loc tokens: `40`
- stop triggered ratio: `0.5`
- record comparison delta:
  - `delta_f1 = 0`
  - `delta_true_positives = 0`
  - `delta_false_positives = 0`
  - `delta_false_negatives = 0`
  - `delta_raw_detection_count = -1`
  - `delta_filtered_detection_count = -1`

#### 5.31.3 更新判断

本轮把 count-stop 的作用边界确认清楚了：

1. `--stop-after-vp-max-total-boxes` 对“已经达到目标 count 后继续吐重复尾巴”的样本有效，可减少无效 raw detections 和后处理过滤压力。
2. 它不会解决欠生成；当模型在 `query_box_count` 之前 EOS，count-stop 不会触发，也不会补框。
3. 当前 2-row overfit 的新 best quality 是 F1 `0.9333`，比上一轮 `0.8750` 更好，但仍保留 row1 的 2 个 recoverable FN。
4. 下一步真正应攻的是 min-count continuation / count-conditioned decoding，而不是把 count-stop 当成默认增益策略。

因此后续默认建议是：

- 保留 count-stop 作为可选 runtime safety/efficiency 开关。
- dense16 泛化实验仍以 no-stop + qcap 作为主 recipe，同时记录 stop ablation。
- 新增 continuation 诊断：对 `raw_loc_box_count < query_box_count` 的样本自动尝试继续生成剩余 boxes。

### 5.32 min-count continuation 与 malformed-tail repair

#### 5.32.1 实现更新

在 `scripts/infer/visualize_florence_vp_adapter.py` 中新增了 continuation 诊断路径：

- `--continue-underfilled-vp-boxes`
- `--vp-continuation-max-rounds`
- `--vp-continuation-max-new-tokens`
- `--vp-continuation-min-missing-boxes`

本轮进一步把 continuation 的触发口径从 raw loc-token 数改为可被下游评估使用的 parseable/structured box 数，并新增：

- `vp_continuation_initial_parseable_box_count`
- `vp_continuation_final_parseable_box_count`
- `vp_continuation_added_parseable_boxes`
- `vp_continuation_count_basis`
- `vp_continuation_last_candidate_raw_prediction`

同时修正 continuation prefix：当输出形如 `<box>...person</box> <<loc_...` 时，prefix 会截到最后一个完整 `<loc_*>` token，去掉 `</box>`、尾随 label 噪声和 malformed tail。

#### 5.32.2 真实 continuation 结果

实验目录：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609
```

训练设置：

- real Florence-2 base + LoRA adapter
- grounding-only 2-row reference overfit
- steps: `96`
- final loss: `0.1381`
- decoder prefix: `<ref>{label}</ref> <box>`

no-continuation 质量：

- F1 / precision / recall: `0.6667 / 1.0000 / 0.5000`
- TP / FP / FN: `8 / 0 / 8`
- avg pred / GT boxes: `4.0 / 8.0`

parseable-count continuation 结果：

- `vp_continuation_attempted_ratio`: `1.0`
- `vp_continuation_applied_ratio`: `0.0`
- `vp_continuation_reached_target_ratio`: `0.0`
- F1 仍为 `0.6667`

解释：

- row0 从 `10` 个 GT 里稳定生成 `7` 个可解析框；continuation candidate 与初始输出相同。
- row1 raw loc-token 有 `4` 个框，但原始 VP parser 只能解析 `1` 个合法框；parseable-count 口径成功暴露了这个差异。
- trim prefix 后，模型仍复现 `person</box> <<loc_...` 的 malformed tail，说明当前 adapter 的局部生成分布倾向于提前关闭 box，而不是继续补足 query count。

#### 5.32.3 malformed-tail repair 消融

新增 opt-in 开关：

- `--structured-vp-repair-malformed-tail`

作用边界：

- 仅当输出已经存在合法 VP box，且其后还有 malformed native loc groups 时触发。
- repair 会用最近的 ref label 给 tail loc groups 补标签，再交给 structured decoder 的 NMS、query cap 和 allowed-label filter。
- 默认关闭，避免改变历史实验语义。

repair-on 可视化目录：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/adapter_inference_continue_parseable_trimmed_repair_tail
```

repair-on quality：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/quality/adapter_continue_parseable_trimmed_repair_tail_visualizer/vp_detection_quality.json
```

结果：

| setting | F1 | precision | recall | TP/FP/FN | avg pred / GT |
| --- | ---: | ---: | ---: | --- | ---: |
| no-cont / no-repair | `0.6667` | `1.0000` | `0.5000` | `8/0/8` | `4.0 / 8.0` |
| continuation only | `0.6667` | `1.0000` | `0.5000` | `8/0/8` | `4.0 / 8.0` |
| continuation + tail repair | `0.7200` | `1.0000` | `0.5625` | `9/0/7` | `4.5 / 8.0` |

per-record：

- row0 (`000000000110.jpg|person`, GT `10`): repair 不触发，仍为 `7/0/3`。
- row1 (`000000000488.jpg|person`, GT `6`): repair 从 malformed tail 回收 1 个有效新框，`1/0/5 -> 2/0/4`，无 FP。

#### 5.32.4 更新判断

本轮结论更清楚：

1. min-count continuation 作为解码后补生成策略目前没有收益；它能诊断欠生成，但不能改变模型在局部 prefix 下的提前闭合倾向。
2. malformed-tail repair 有真实收益，且是 GT-free 后处理：它不猜新框，只回收模型已经生成但格式破损的 loc groups。
3. repair 不能解决 row0 的真实欠生成，也不能补 row1 仍缺失的 4 个实例；下一步仍需要训练侧 count-conditioned decoding 或更强的 dense grounding 数据。
4. 默认 recipe 建议变为：`qcap + allowed-label + NMS + optional malformed-tail repair`；continuation 保留为诊断开关，不作为默认增益策略。

### 5.33 repair policy sweep 与审计字段补齐

#### 5.33.1 实现更新

上一节已经证明 `--structured-vp-repair-malformed-tail` 对 malformed tail 有收益。本轮进一步把它从“单独命令开关”推进成 policy sweep 的正式候选：

- `scripts/experiments/sweep_vp_quality_policies.py` 新增 `--include-repair-policy`。
- 该开关会为当前所有 policy 自动生成 `_repair` 对照项，例如 `none_repair`、`nms_repair`、`query_label_allowed_repair`。
- `scripts/experiments/sweep_vp_generation_budgets.py` 和 `scripts/experiments/run_florence_vp_training_experiment.py` 已透传该开关。
- `StructuredVPDecodeResult` 新增 `repaired_tail_detection_count` 与 `used_tail_repair`。
- quality report 新增：
  - `repaired_tail_detection_count`
  - `repaired_tail_record_ratio`
  - `avg_repaired_tail_detection_count`

同时修正了一个 policy sweep 命名问题：默认 `none` policy 现在显式使用 `filter_policy=none`，不再继承命令行的全局 `--structured-vp-filter-policy nms`。否则会出现“名字叫 none、实际是 nms”的误导。

#### 5.33.2 真实 policy sweep

本轮不重新训练、不重新推理，只基于已保存 summary 做 quality policy sweep：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/adapter_inference_continue_parseable_trimmed/vp_inference_visualization_summary.json
```

输出目录：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/policy_sweep/adapter_continue_parseable_trimmed_repair_ablation
```

命令要点：

- `--structured-vp-filter-policy nms`
- `--structured-vp-max-total-boxes-field query_box_count`
- `--structured-vp-allowed-labels-field query_label`
- `--include-repair-policy`
- `--focus-bucket dense`

结果：

| policy | kind | F1 | precision | recall | TP/FP/FN | repaired tail count | repaired record ratio |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `none_repair` | none | `0.7200` | `1.0000` | `0.5625` | `9/0/7` | `3` | `0.5` |
| `nms_repair` | nms | `0.7200` | `1.0000` | `0.5625` | `9/0/7` | `3` | `0.5` |
| `query_label_allowed_repair` | allowed-label | `0.7200` | `1.0000` | `0.5625` | `9/0/7` | `3` | `0.5` |
| `none` | none | `0.6667` | `1.0000` | `0.5000` | `8/0/8` | `0` | `0.0` |
| `nms` | nms | `0.6667` | `1.0000` | `0.5000` | `8/0/8` | `0` | `0.0` |
| `query_label_allowed` | allowed-label | `0.6667` | `1.0000` | `0.5000` | `8/0/8` | `0` | `0.0` |

recommended policy:

```text
none_repair
```

解释：

- repair variants 稳定领先，收益来自 row1 的 malformed tail 回收。
- `repaired_tail_detection_count=3` 表示从 row1 tail 中识别出 3 个候选 loc groups；其中 NMS/匹配后真正贡献 1 个新增 TP，其余为重复候选或未匹配候选。
- 在这个 2-row dense 子集上，NMS/allowed-label 与否不改变最终质量，因为输出标签单一且无 FP；repair 是主要增益来源。

#### 5.33.3 更新判断

当前 VP 后处理链路更完备了：

1. repair 可以作为标准 policy 进入所有 sweep，而不是一次性手工 ablation。
2. repair 的收益、覆盖率和候选数量已经能被 JSON/Markdown 审计。
3. 下一步应把 `--include-repair-policy` 纳入 dense held-out 默认 sweep，观察 malformed tail 是否是普遍错误模式，还是只出现在当前 2-row overfit。
4. 如果 held-out 中 repair 仍稳定提升且不引入 FP，再考虑把 `--structured-vp-repair-malformed-tail` 提升为推荐默认；目前仍保留 opt-in。

### 5.34 VP report-card 统一完备性诊断

新增轻量诊断入口：

```text
scripts/experiments/build_vp_report_card.py
```

它不重新训练模型，也不生成权重；只读取已有 JSON 产物：

- `vp_detection_quality.json`
- 可选 `vp_quality_policy_sweep.json`
- 可选 `vp_target_count_gap.json`；若未传入，默认从 quality report 中派生 target-count gap。

本次基于 latest dense repair ablation 生成：

```text
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/report_card/vp_report_card.json
.codex_reports/florence_vp_continuation_paired_reference_only2_20260609/report_card/vp_report_card.md
```

命令：

```bash
python3 scripts/experiments/build_vp_report_card.py \
  --quality-report .codex_reports/florence_vp_continuation_paired_reference_only2_20260609/policy_sweep/adapter_continue_parseable_trimmed_repair_ablation/none_repair/vp_detection_quality.json \
  --policy-sweep .codex_reports/florence_vp_continuation_paired_reference_only2_20260609/policy_sweep/adapter_continue_parseable_trimmed_repair_ablation/vp_quality_policy_sweep.json \
  --output-dir .codex_reports/florence_vp_continuation_paired_reference_only2_20260609/report_card \
  --focus-bucket dense
```

report-card 结论：

| item | value |
| --- | --- |
| status | `fail` |
| readiness | `needs_work` |
| precision / recall / F1 | `1.0000 / 0.5625 / 0.7200` |
| avg pred / GT boxes | `4.50 / 8.00` |
| undergeneration ratio | `1.0000` |
| repair record ratio | `0.5000` |
| recommended policy | `none_repair` |
| repair lift | `+0.0533 F1`, `+0.0625 recall` |
| target-count recoverable FN | `7 / 7` |

解释：

- 当前模型的主要瓶颈仍是 dense recall/undergeneration，而不是 FP。
- repair 是有效诊断和后处理，但 `repair_record_ratio=0.5`，还不能把它视为模型本体能力。
- target-count gap 显示 7 个 FN 在 oracle count-fill 下都可恢复，说明下一步优先级应放在 count-conditioned dense decoding、proposal distillation 或 continuation training，而不是盲目加大 NMS/allowed-label 后处理。
- report-card 默认阈值偏实验推进：`min_samples=10`、`min_recall=0.70`、`min_f1=0.75`、`max_undergeneration_ratio=0.35`。当前 2-row 结果只能作为小样本诊断，不能证明 Florence-VP 已完备。

更新后的推进顺序：

1. 把 report-card 纳入每次 VP 实验的固定收尾产物。
2. 在 dense held-out split 上重跑 quality + policy sweep + report-card。
3. 若 report-card 仍显示 recoverable FN 高，优先推进 count-conditioned curriculum 和 proposal distillation。
4. 若 repair 在 held-out 上稳定增益且不引入 FP，再考虑作为推荐默认；否则只保留为审计/诊断开关。

### 5.35 dense16 report-card 与 proposal distillation 诊断

为了避免 2-row overfit 误导，本次对已有 dense16 产物补跑了 policy sweep、report-card 和 proposal replay：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607
```

#### 5.35.1 dense16 report-card

adapter report-card：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/report_card/adapter/vp_report_card.json
```

baseline report-card：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/report_card/baseline/vp_report_card.json
```

结果：

| model | samples | precision | recall | F1 | TP/FP/FN | avg pred / GT | undergen | recommended policy | repair lift | recoverable FN |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| adapter | `16` | `0.7576` | `0.3906` | `0.5155` | `50/16/78` | `4.125 / 8.000` | `0.7500` | `none` | `0.0000` | `65/78` |
| baseline | `16` | `0.7463` | `0.3906` | `0.5128` | `50/17/78` | `4.188 / 8.000` | `0.7500` | `none` | `0.0000` | `65/78` |

判断：

- dense16 上 adapter 相对 baseline 基本没有有效提升，主要只是少 1 个 FP。
- repair 在 dense16 上没有收益，说明 2-row overfit 里的 malformed-tail 模式不是当前 dense held-out 的主瓶颈。
- 65/78 个 FN 属于 target-count deficit 可恢复区域，下一步应优先让模型学会多实例补全，而不是继续优化 malformed-tail repair。

#### 5.35.2 grid proposal upper-bound

由于完整 segmentation proposal 在本机较慢，本轮先用 grid-only oracle ranking 做轻量 upper-bound：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/image_proposals_grid_oracle120
```

proposal coverage：

| method | max proposals | recall@25 | recall@50 | recall@75 | mean best IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid + oracle_iou | `120` | `0.7894` | `0.5172` | `0.0888` | `0.4648` |

target-count replay：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/target_count_proposal_grid_oracle120
```

| setting | precision | recall | F1 | TP/FP/FN | avg pred / GT | added boxes | deficit closure |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| adapter primary | `0.7576` | `0.3906` | `0.5155` | `50/16/78` | `4.125 / 8.000` | `0` | - |
| grid proposal replay | `0.6336` | `0.6484` | `0.6409` | `83/48/45` | `8.188 / 8.000` | `65` | `1.0000` |

解释：

- proposal replay 可以把 count deficit 全部填满，并把 recall 从 `0.3906` 提到 `0.6484`。
- 代价是 FP 从 `16` 增到 `48`，说明 proposal teacher 需要质量过滤或更强 proposal ranking。
- 这验证了 report-card 的判断：count/proposal 方向确实能补召回，但需要控制噪声。

#### 5.35.3 distillation hard positives

基于上面的 grid proposal replay，生成了一份 improvement-only distillation JSONL：

```text
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/grid_oracle120_improvement.jsonl
.codex_reports/florence_vp_ovd_count_hint_trainonly_dense16_lr1e6_20260607/distillation/grid_oracle120_improvement_summary.json
```

结果：

| item | value |
| --- | ---: |
| input records | `16` |
| output rows | `9` |
| added boxes in output | `51` |
| avg added boxes / row | `5.6667` |
| delta TP / FP / FN | `+33 / +18 / -33` |
| avg delta F1 | `0.3506` |
| dense rows | `9` |

top labels：

- `person`: `5`
- `chair`: `2`
- `umbrella`: `1`
- `cup`: `1`

下一步训练建议：

1. 不直接把所有 proposal replay 都混进训练；先用 `quality_filter=improvement` 的 9 条 hard positives。
2. 训练时降低 distillation repeat 或使用 round-robin，避免 proposal 噪声覆盖原始 grounding 数据。
3. 下一轮真实训练对比三组：原始 dense curriculum、原始 + improvement distillation、原始 + reference target-mode oracle。
4. 若 FP 明显上升，再加入 negative/no-object 与 label canonicalization。

### 5.36 real distillation 4-step 训练与 prefix 对齐复核

本轮继续把 5.35 的 proposal distillation 从数据构造推进到真实 Florence 训练链路。先生成短 mix：

```text
.codex_reports/florence_vp_distill_grid_oracle120_shortmix_20260609/query_ovd_distill_mix.jsonl
.codex_reports/florence_vp_distill_grid_oracle120_shortmix_20260609/query_ovd_distill_mix_summary.json
```

mix 摘要：

| item | value |
| --- | ---: |
| output rows | `34` |
| base rows | `16` |
| distillation rows | `18` |
| distillation ratio | `0.5294` |

随后跑真实权重、真实数据、CPU、LoRA 的 4-step smoke：

```text
.codex_reports/florence_vp_distill_grid_oracle120_4step_20260609
```

关键产物：

```text
.codex_reports/florence_vp_distill_grid_oracle120_4step_20260609/training/real_florence_vp_training_smoke_summary.json
.codex_reports/florence_vp_distill_grid_oracle120_4step_20260609/quality/adapter/vp_detection_quality.json
.codex_reports/florence_vp_distill_grid_oracle120_4step_20260609/report_card/adapter/vp_report_card.json
.codex_reports/florence_vp_distill_grid_oracle120_4step_20260609/prefix_comparison/vp_quality_prefix_comparison.json
```

训练链路结论：

| item | value |
| --- | --- |
| training ok | `true` |
| max steps | `4` |
| cleanup adapter | `requested=true, deleted=true` |
| residual weight files | none in latest run |

4-step adapter 在 first8 dense validation 上的 report-card：

| samples | precision | recall | F1 | TP/FP/FN | avg pred / GT | undergen | recoverable FN |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `8` | `0.9063` | `0.4754` | `0.6237` | `29/3/32` | `4.000 / 7.625` | `0.8750` | `29/32` |

为了避免把 8-sample 和 16-sample 结果混在一起解释，本轮新增了前缀对齐比较工具：

```text
scripts/experiments/compare_vp_quality_prefix.py
```

真实 first8 对齐结果：

| report | samples | precision | recall | F1 | TP/FP/FN | delta F1 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| old_adapter | `8` | `0.9063` | `0.4754` | `0.6237` | `29/3/32` | `0.0000` |
| old_baseline | `8` | `0.9063` | `0.4754` | `0.6237` | `29/3/32` | `0.0000` |
| distill_4step | `8` | `0.9063` | `0.4754` | `0.6237` | `29/3/32` | `0.0000` |

判断：

1. 4-step distillation smoke 证明了真实训练、推理、quality、target-count gap、policy sweep、report-card、adapter cleanup 这一整条链路可以跑通。
2. 但 first8 对齐复核显示，4 step 没有产生可观测质量变化；它只能证明 pipeline 可用，不能证明 proposal distillation 已经有效。
3. 当前瓶颈仍是 dense undergeneration，`recoverable FN=29/32`，说明后续应继续围绕 count/proposal/continuation 训练，而不是把短训 smoke 视为收敛结果。
4. 因为磁盘紧张，后续真实训练仍应默认开启 `--cleanup-adapter-after-audit`，只保留 JSON、Markdown、可视化和必要 summary。

本轮还把 distillation mix 纳入实验 runner，后续可一条命令规划：

```text
scripts/experiments/run_florence_vp_training_experiment.py --run-distillation-mix --run-training ...
```

新增 runner 参数覆盖：

- `--distillation-mix-base-input`
- `--distillation-mix-input`
- `--distillation-mix-repeat`
- `--distillation-mix-repeat-order`
- `--distillation-mix-placement`
- `--distillation-mix-min-delta-tp`
- `--distillation-mix-target-mode`
- `--distillation-mix-replace-base-on-key`

更新后的剩余任务：

1. 在磁盘允许时跑 12/16 step 的 distillation mix 训练，并继续使用 prefix comparison 对齐 old firstN。
2. 将 distillation repeat 降低或改为 `round_robin`，观察是否减少 proposal 噪声覆盖。
3. 对比 `teacher` 与 `reference` target-mode，判断是学习后处理行为更稳，还是 hard-example 完整监督更有效。
4. 若继续无变化，转向更强 proposal teacher 或训练参数消融，例如 LoRA rank、modules_to_save、学习率、dense-only curriculum。

---

## 6. 可直接追加到报告末尾的“待讨论问题”

下面这组问题适合放在报告最后，作为评审会讨论提纲：

1. 我们要验证的是“视觉原语格式”本身，还是“带可见推理链的视觉原语格式”？两者训练难度不同。
2. MVP 的 baseline 是 Florence-2 原生 `<loc_*>` 输出，还是当前 FlorenceForge 的 JSON suffix 输出？
3. VP token 应该作为真正的特殊 token 加入 tokenizer，还是先用普通文本标记模拟，降低 embedding 风险？
4. 新 token embedding 是否训练？如果训练，是否需要把 `embed_tokens` 和 `lm_head` 从 LoRA 冻结策略中排除？
5. `OD_VP` 是新任务，还是 `OD` 的输出格式变体？这会影响后向兼容和 dataset 校验。
6. VP reasoning template 是否需要固定三段式，还是只要求最终答案包含可解析的 ref/box？
7. 如果模型输出 box 正确但文本 reasoning 错误，reward/metric 应如何计分？
8. 如果文本答案正确但 box 错误，是否视为失败？这决定 VP 是否真的是“思考单元”。
9. 计数任务是否从 COCO detection 数据自动构造，还是需要人工 count QA 数据？
10. 多实例目标是否要求输出所有框？只输出部分框但 count 正确时如何评分？
11. 坐标空间使用 `[0,999]` 后，是否与 Florence-2 原生 loc token 的离散化方式冲突？
12. 图像 resize/crop/augmentation 后，VP 坐标由谁负责同步变换？
13. VP parser 是否允许宽松格式，例如空格、换行、单框、重复 ref？
14. VP parser 解析失败时，训练/评估是 fail-fast 还是记为 format error？
15. 开放词汇检测中的同义词、复数、大小写如何归一化？
16. OCR_WITH_REGION 的 ref 应该是文字内容、类别 `text`，还是二者都输出？
17. segmentation 是否应使用 point sequence、polygon，还是保持 Florence 原有 mask 表达？
18. ~~`UnifiedMultiTaskTrainer`~~ 已删除（不可运行的空桩）。VP MVP 应使用 v1 `MultiTaskTrainer` 或 v2 训练栈。
19. 混合 VP/非 VP 训练时，LoRA adapter 是共享、分任务，还是分 VP task family？
20. MVP 成功后，是否要把 VP 功能定位为核心功能、experimental 功能，还是论文实验功能？

---

## 7. 优化后的路线图

### Phase 0: 预备修复与接口定义（约 1 周）

1. 修正报告中的任务数量和格式错误。
2. 定义 VP token、坐标规范、parser AST。
3. 选择承载训练器：MVP 用 v1 `MultiTaskTrainer`，后续收敛到 v2 训练栈（~~`UnifiedMultiTaskTrainer`~~ 已删除）。
4. 增加最小单测：token、bbox normalize、parser、converter。

### Phase 1: VP 数据与 SFT MVP（约 2 周）

1. 新增 `VisualPrimitiveConverter`，支持 COCO/YOLO OD -> `OD_VP`。
2. 新增 `OD_VP`、`COUNT_VP`、`PHRASE_GROUNDING_VP` 任务配置。
3. 扩展 Florence-2 backend，支持 opt-in token add + embedding resize。
4. 已完成真实权重 + COCO128 的 1 step 参数切片 smoke train、LoRA smoke train、8 step JSON adapter 可视化和 12 step `loc_tokens` adapter 可视化；本次已增加结构化 VP 解码，将 Florence 原生 `label<loc_*>` 输出包装成 `ref + box` 证据链，并新增 `structured_vp_format_valid_ratio`、`structured_precision/recall/mAP` 作为当前可用性门槛。
5. 输出格式有效率、坐标合法率、OD mAP、count accuracy。

### Phase 2: 消融与可解释性验证（约 2 周）

1. baseline: 原生 Florence OD/loc 或现有 JSON suffix。
2. ablation A: 只输出 VP box，无 reasoning。
3. ablation B: 输出三段式 reasoning + VP box。
4. ablation C: VP token 作为 special token vs 普通文本 token。
5. bad case 可视化：原图 + GT box + predicted VP box + textual trace。

### Phase 3: 专家训练与 RL（延后）

只有当 Phase 2 证明 VP 至少在计数、定位或 grounding 上有稳定收益时，再启动：

1. Box Expert 数据扩展。
2. Point/Polygon task 可行性评估。
3. OPD 蒸馏。
4. deterministic reward -> GRPO。

---

## 8. 推荐结论段落（可替换原报告 7.1）

FlorenceForge 与 TVP 的融合具备明确研究价值和工程差异化，但短期不应以“完整复刻 TVP”为目标。更合理的路线是先把 VP 作为 FlorenceForge 的可选任务增强层：在不破坏现有 14 个 Florence-2 任务的前提下，新增可训练、可解析、可评估的视觉原语输出格式。

若 Layer 1 MVP 能证明模型稳定生成 ref/box，且在计数、定位或 phrase grounding 上超过 baseline，则 FlorenceForge 可以进一步进入专家蒸馏和 RL 阶段。若 MVP 只能提升可解释性而不提升精度，也仍可作为推理可视化和 bad-case 分析能力保留，但不宜投入高成本 RL。

一句话总结：

> FlorenceForge + TVP 的第一目标不是“让模型学会长篇思考”，而是让 Florence-2 的视觉输出从不可控文本变成可解析、可评分、可回放的视觉证据链。
