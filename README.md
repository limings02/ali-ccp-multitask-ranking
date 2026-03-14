# Ali-CCP 电商广告多目标学习优化（CTR / CVR / CTCVR）

> - 这不是单纯的 Ali-CCP 预处理仓库，而是一条从原始样本到 CTR / CVR / CTCVR 联合建模、评估和诊断的完整实验链。
> - 我主要做了 5 件事：entity 级防泄漏切分、长尾 token 截断与 featuremap、ESMM / MMoE / PLE 多任务结构、专家健康与梯度冲突诊断、以及损失与选优策略改造。
> - 当前 README 主展示结果来自 `runs/shared_bottom_esmm_20260215_060628`：CTR AUC `0.5927`，CVR AUC `0.6816`，CTCVR AUC `0.6401`。
> - 如果只看当前 main runs 里的 CTCVR 最优点，`runs/esmm_mmoe_nogate_20260215_080115` 达到 `0.6500`；但从结果平衡和复跑完整度看，我更愿意把上一组作为简历展示主结果。

## 1. 项目简介

这个仓库做的是 Ali-CCP 电商广告场景下的多目标学习项目。核心目标不是“把公开数据处理干净”就结束，而是把 CTR、CVR、CTCVR 这三类目标放到一条统一的训练和评估链路里，看看数据切分、特征表达、共享结构和损失设计分别会把结果推向哪里。

为什么一定要同时看这三个目标：只看 CTR，模型很容易学到点击偏好，但对真正的转化帮助有限；只看 CVR，又会碰到点击样本选择偏差；CTCVR 更接近广告转化链路的最终目标，但它又依赖 CTR 和 CVR 的耦合。这个项目的重点就是把这三个目标放在一起看，而不是把它们拆成互相割裂的单任务。

和普通学生项目或者只跑一个 baseline 的复现不太一样，这个仓库里保留了比较完整的工程痕迹：`canonical -> split -> split-tokens -> process -> vectorize -> train -> eval` 的数据链路、多任务结构切换、诊断指标、配置化实验、测试、以及真实的 `runs/` 和 `reports/` 产物。目录里也能看到 `attempt`、`interview_*`、`main` 这些实验阶段，它们不是刻意整理成“完美历史”，而是我实际迭代时留下来的。

## 2. 项目背景与问题定义

Ali-CCP 是公开的电商广告数据，标签天然带有曝光、点击、转化链路。这个场景下，多任务学习的意义很直接：点击提供更充分的监督，转化提供更接近业务目标的监督，但两者共享表示时又容易互相干扰。

这个仓库里我主要盯着下面几类问题：

- 数据泄漏：Ali-CCP 有公共特征和重复实体，随机切分很容易让 train/valid 共享同一个实体的侧信息，离线结果会偏乐观。
- 长尾特征：`reports/eda/token_truncation_strategy.md` 里统计到每条样本平均约 `629` 个 tokens，`P99` 接近 `1980`，直接全保留很难训。
- 转化稀疏：当前有效验证集里，CTR 正样本率约 `3.88%`，click 后 CVR 正样本率约 `0.53%`，曝光级 CTCVR 正样本率只有 `0.0207%`。
- 任务冲突：CTR 和 CVR / CTCVR 不是简单同向，目标权重、共享结构和路由方式都会影响 trade-off。
- 共享结构不合理导致的负迁移：ESMM 能处理样本选择偏差，但不自动解决共享资源怎么分配的问题，MMoE / PLE 的意义就在这里。

## 3. 我在这个项目里做了什么

### 3.1 数据处理与特征工程链路

这部分我花的时间其实不少。仓库里先用 [`src/data/canonical.py`](src/data/canonical.py) 把原始 `skeleton/common_features` 还原成可追踪的 `samples + tokens` 两张表，再用 [`src/data/split.py`](src/data/split.py) 做稳定哈希切分。简历里如果说“按 user_id 去重防泄漏”，在这个仓库里的实际落地是按 `entity_id` 做切分隔离，因为公开数据里能稳定拿来做防泄漏的键就是它；本质上都是让 train/valid 不共享同实体的公共特征。

后面的重点是怎么处理长尾多值特征。`reports/eda/token_truncation_strategy.md` 里可以看到：Top-4 字段吃掉了绝大部分 token 预算，P99 样本会把 batch 内存直接顶满。我没有另外接一套体量很大的浅层树特征工程，而是把预算收敛到 featuremap 设计上：在 [`configs/dataset/featuremap.yaml`](configs/dataset/featuremap.yaml) 里给不同 field 选 `vocab / hash / hybrid` 编码，给多值 field 设 `max_len`，再通过 [`src/data/token_select.py`](src/data/token_select.py) 做 `auto_mix` / `topk_by_freq` 这类确定性截断。对高基数字段，还做了 `safe4g` 风格的降维、扩桶、缩 `max_len` 调整，保证低资源机器也能把数据跑通。

默认训练使用的是 vectorized CSR 形式，所以我又补了 [`src/data/vectorize_parquet.py`](src/data/vectorize_parquet.py)，把 processed parquet 转成 `data/vectorized/`，这样 dataloader 走 mmap 读起来会更稳。

代码入口：

- 数据读取与 canonical 构建：[`src/data/canonical.py`](src/data/canonical.py)
- 防泄漏切分：[`src/data/split.py`](src/data/split.py)
- EDA 与 featuremap 证据：[`src/eda/aliccp_eda.py`](src/eda/aliccp_eda.py)、[`src/eda/extra`](src/eda/extra)、[`reports/eda/token_truncation_strategy.md`](reports/eda/token_truncation_strategy.md)、[`reports/eda/featuremap_rationale.md`](reports/eda/featuremap_rationale.md)
- processed 数据构建：[`src/data/processed_builder.py`](src/data/processed_builder.py)
- 向量化格式：[`src/data/vectorize_parquet.py`](src/data/vectorize_parquet.py)

### 3.2 多任务建模：ESMM / MMoE / PLE

建模上我没有停在“DeepFM + 一个 dual head baseline”。当前主干是用 [`src/models/backbones/deepfm.py`](src/models/backbones/deepfm.py) 做 backbone，再通过 [`src/models/build.py`](src/models/build.py) 挂接 `sharedbottom`、`mmoe`、`ple` 三类结构。ESMM 的链式关系放在 [`src/loss/bce.py`](src/loss/bce.py) 里处理：`p_ctcvr = p_ctr * p_cvr`，同时支持 `lambda_cvr_aux` 这类 click 子集辅助损失。

我做这部分不是为了盲目堆模型，而是为了把问题拆开看。ESMM 主要解决样本选择偏差；MMoE 想解决的是“共享但不要硬绑死”；PLE 则进一步把 shared experts 和 task-specific experts 分开，让 CTR 和 CVR / CTCVR 有显式的共性区和私有区。仓库里这三条线都能切，配置在 [`configs/experiments/main`](configs/experiments/main) 和 [`configs/experiments/attempts`](configs/experiments/attempts) 里都有保留。

当前主线复跑里，`shared_bottom_esmm` 已经把 CVR 拉到 `0.6816`，说明 ESMM 对这个问题是有帮助的；继续换成 `esmm_mmoe_nogate`，CTCVR 能到 `0.6500`，但 CTR 会有小幅回撤。这也是我后来更重视“目标怎么选 best”和“共享资源怎么诊断”的原因。PLE 这条线我也完整实现了，包括 homogeneous / heterogeneous experts 和配套诊断，不过当前 `configs/experiments/main/esmm_ple.yaml` 这次主线复跑没有产出最终 `eval.json`，所以 README 里不会把它写成现阶段主结果。

代码入口：

- 模型装配：[`src/models/build.py`](src/models/build.py)
- SharedBottom：[`src/models/mtl/shared_bottom.py`](src/models/mtl/shared_bottom.py)
- MMoE：[`src/models/mtl/mmoe.py`](src/models/mtl/mmoe.py)
- PLE / 异构专家：[`src/models/mtl/ple.py`](src/models/mtl/ple.py)
- ESMM loss：[`src/loss/bce.py`](src/loss/bce.py)
- 实验配置：[`configs/experiments/main`](configs/experiments/main)、[`configs/experiments/attempts/interview_chain`](configs/experiments/attempts/interview_chain)

### 3.3 诊断与分析：专家健康度、梯度冲突、跷跷板问题

这部分是我觉得和“只跑出一个 AUC 数字”的复现差别最大的地方。仓库里不只记录 `metrics.jsonl`，还会额外写 `expert_health_diag.jsonl`。[`src/train/grad_diag.py`](src/train/grad_diag.py) 会动态识别 shared parameters，算任务间 gradient cosine 和 conflict rate；[`src/utils/expert_health_diag.py`](src/utils/expert_health_diag.py) 会记录 gate 的 `top1_share`、`mean_weight`、`p95`、`dead experts`、`monopoly experts` 和 `Gini`。

这套诊断的实际价值，是把“跷跷板”从一句抽象描述，变成可以对着日志解释的问题。比如早期 MMoE run [`runs/attempt/test_mmoe_20260211_112003/expert_health_diag.jsonl`](runs/attempt/test_mmoe_20260211_112003/expert_health_diag.jsonl) 里就出现过 `utilization_cvr: high load imbalance (Gini=0.593)` 这类告警；[`reports/attempt_analyse/interview_series_report.md`](reports/attempt_analyse/interview_series_report.md) 也汇总了不同阶段的 dead / monopoly expert 情况。再往后看 [`reports/lambda_sweep/lambda_sweep_summary.md`](reports/lambda_sweep/lambda_sweep_summary.md)，你会发现 `ctcvr_auc` 拉高时 `ctr_auc` 会有回撤，这就是很典型的离线 Pareto 取舍。

我最后没有把诊断当成“附加彩蛋”，而是把它并进了训练选优逻辑里：[`src/train/best_selector.py`](src/train/best_selector.py) 的 `gate` 策略要求主目标提升，同时辅助目标不能明显回撤。这个设计本质上就是把 Pareto 思路做成了工程规则。

代码入口：

- 梯度冲突与共享参数诊断：[`src/train/grad_diag.py`](src/train/grad_diag.py)、[`src/train/grad_conflict_sampler.py`](src/train/grad_conflict_sampler.py)
- 专家健康诊断：[`src/utils/expert_health_diag.py`](src/utils/expert_health_diag.py)
- 训练选优：[`src/train/best_selector.py`](src/train/best_selector.py)
- 诊断样例：[`reports/attempt_analyse/interview_series_report.md`](reports/attempt_analyse/interview_series_report.md)、[`reports/lambda_sweep/lambda_sweep_summary.md`](reports/lambda_sweep/lambda_sweep_summary.md)

### 3.4 损失与优化：Focal Loss、residual 调制、Pareto 改进

转化相关任务最难的地方还是稀疏和耦合。CTCVR 的正样本非常少，而 ESMM 里 `p_ctcvr = p_ctr * p_cvr` 又会让优化更容易偏到“先把 CTR 学稳”。这部分我做了两件事：一是把 Aux Focal 接进 [`src/loss/bce.py`](src/loss/bce.py)，支持 warmup、logits 版本 focal、组件日志和 smoke test；二是加了 [`src/models/residual_head.py`](src/models/residual_head.py)，在 logit 空间对 CTCVR 做 residual 修正，并配合 λ sweep 去看不同目标权重下的 trade-off。

这里我不想把 README 写成“某个 loss 上去就稳定涨点”。当前仓库里的记录更像真实调参过程：在 `lambda=5` 的对比里，`classic_mmoe_lambda5` 的 `ctcvr_auc` 是 `0.6437`，`use_focal_mmoe_lambda5` 反而掉到 `0.6407`；但 `use_resi_mmoe_lambda5` 把 `cvr_auc` 拉到了 `0.6877`。也就是说，Focal、residual、Pareto 这些方向我都做了实现、测试和实验，但它们在当前仓库里更适合拿来讲“我怎么分析优化失衡”，而不是硬写成一个固定涨幅结论。

代码入口：

- ESMM / Focal / residual loss：[`src/loss/bce.py`](src/loss/bce.py)
- residual head：[`src/models/residual_head.py`](src/models/residual_head.py)
- Pareto / λ sweep 配置：[`configs/experiments/attempts/mmoe_optim/pareto_analy`](configs/experiments/attempts/mmoe_optim/pareto_analy)
- 汇总报告：[`reports/lambda_sweep/lambda_sweep_summary.md`](reports/lambda_sweep/lambda_sweep_summary.md)
- Focal 实现和测试：[`docs/aux_focal_summary.md`](docs/aux_focal_summary.md)、[`tests/test_aux_focal_smoke.py`](tests/test_aux_focal_smoke.py)

## 4. 实验结果与结论

下面这张表只放当前仓库里已经有完整 `eval.json` 的主线 run。表里的 `CVR AUC` 都是 click 子集上的 masked 指标。

| 方案 | 主要改动 | CTR AUC | CVR AUC | CTCVR AUC |
| --- | --- | ---: | ---: | ---: |
| `single_task_ctr` | 只训练 CTR head | 0.5999 | - | - |
| `single_task_ctcvr` | 只盯 CTCVR，`lambda_ctr=0` | 0.4786 | 0.6520 | 0.6343 |
| `shared_bottom` | 双头硬共享，不用 ESMM | 0.5536 | 0.6287 | - |
| `shared_bottom_esmm` | SharedBottom + ESMM v2 | **0.5927** | **0.6816** | **0.6401** |
| `esmm_mmoe_nogate` | ESMM v2 + MMoE | 0.5922 | 0.6877 | 0.6500 |

这些数字分别来自：

- [`runs/single_task_ctr_20260215_101108/eval.json`](runs/single_task_ctr_20260215_101108/eval.json)
- [`runs/single_task_ctcvr_20260215_021422/eval.json`](runs/single_task_ctcvr_20260215_021422/eval.json)
- [`runs/shared_bottom_20260215_041605/eval.json`](runs/shared_bottom_20260215_041605/eval.json)
- [`runs/shared_bottom_esmm_20260215_060628/eval.json`](runs/shared_bottom_esmm_20260215_060628/eval.json)
- [`runs/esmm_mmoe_nogate_20260215_080115/eval.json`](runs/esmm_mmoe_nogate_20260215_080115/eval.json)

我自己对这些结果的判断是：

- `shared_bottom -> shared_bottom_esmm` 这一步最能说明项目主问题是什么。没有 ESMM 时，CVR 和整条转化链路都不太好看；一旦引入 ESMM，结果就明显更稳。
- 如果拿来做简历展示，我会把 `shared_bottom_esmm` 作为主结果。原因不是它绝对最高，而是它和当前主线 pipeline、配置、runs 产物对应得最完整，也最适合解释“为什么同时看 CTR / CVR / CTCVR”。
- 如果面试官继续追问“那你后面为什么还做 MMoE / PLE”，我会再展开 `esmm_mmoe_nogate` 和 λ sweep。CTCVR 可以继续往上推，但 CTR 会有回撤，这就是多目标优化里真正要处理的 trade-off。
- PLE 方向在仓库里是有实现、有测试、有早期实验记录的，详见 [`reports/attempt_analyse/interview_series_report.md`](reports/attempt_analyse/interview_series_report.md)。但当前主线 [`configs/experiments/main/esmm_ple.yaml`](configs/experiments/main/esmm_ple.yaml) 对应的复跑 [`runs/esmm_ple_20260215_100903`](runs/esmm_ple_20260215_100903) 没有产出最终 `eval.json`，所以我不把它写成现阶段主结论。

整体上，我会把 `CTR 0.5927 / CVR 0.6816 / CTCVR 0.6401` 这组结果描述成：在当前资源、实现条件和公开数据约束下，一版比较稳定、适合对外展示的复跑结果；不是“最优到可以盖棺定论”的数字，但足够支撑项目思路和工程质量。

## 5. 代码结构总览

如果面试官只想快速定位代码，可以按下面看：

- 想看数据处理：[`src/data`](src/data)
  - `canonical.py`：原始 CSV 转 `samples + tokens`
  - `split.py`：按 `entity_id` 稳定哈希切分，做防泄漏
  - `processed_builder.py`：按 featuremap 生成训练 parquet
  - `token_select.py`：长尾 token 截断策略
  - `vectorize_parquet.py`：把 processed parquet 转成训练默认使用的 vectorized CSR 格式
- 想看模型实现：[`src/models`](src/models)
  - `backbones/deepfm.py`：DeepFM backbone
  - `mtl/shared_bottom.py`、`mtl/mmoe.py`、`mtl/ple.py`：多任务结构
  - `residual_head.py`：CTCVR residual 修正
- 想看训练流程：[`src/train`](src/train)
  - `trainer.py`：训练入口
  - `loops.py`：train / valid 主循环
  - `best_selector.py`：best checkpoint 选择
  - `grad_diag.py`：梯度冲突诊断
- 想看评估流程：[`src/eval`](src/eval)
  - `run_eval.py`：评估入口
  - `metrics.py`：AUC / logloss 等
  - `calibration.py`：ECE
  - `funnel.py`：funnel consistency
- 想看配置：[`configs`](configs)
  - `dataset/`：数据与 featuremap
  - `experiments/main/`：主线复跑配置
  - `experiments/attempts/`：早期尝试、interview chain、Pareto/focal/residual 配置
- 想看测试：[`tests`](tests)
  - 数据链路：`test_data_pipeline.py`、`test_processed_pipeline.py`
  - 模型前向：`test_model_forward.py`
  - 诊断：`test_expert_health_diag.py`、`test_grad_conflict_sampler.py`
  - loss / 训练边界：`test_aux_focal_smoke.py`、`test_train_loop_cvr_zero_mask.py`
- 想看补充材料：[`docs`](docs)、[`reports`](reports)
  - `docs/interview_chain.md`：一条完整的结构演进实验链
  - `reports/eda/*`：featuremap 和 token 截断证据
  - `reports/lambda_sweep/*`：Pareto / λ sweep
  - `reports/attempt_analyse/*`：interview 系列汇总

## 6. 如何快速跑通

下面只保留当前仓库最短、能跑通的链路。默认训练读取 `data/vectorized`，所以做完 `process` 之后还要执行一次 `vectorize_parquet`。

### 环境安装

```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install scikit-learn pytest
```

如果原始数据不在默认位置，可以改 [`configs/dataset/aliccp.yaml`](configs/dataset/aliccp.yaml)，或者用环境变量 `ALICCP_SKELETON_PATH`、`ALICCP_COMMON_FEATURES_PATH` 覆盖。

### 数据预处理

```bash
python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --overwrite
python -m src.cli.main split --config configs/dataset/aliccp.yaml --overwrite
python -m src.cli.main split-tokens --config configs/dataset/aliccp.yaml --overwrite
python -m src.cli.main process --config configs/dataset/featuremap.yaml --split-dir data/splits/aliccp_entity_hash_v1 --out data/processed --batch-size 500000
python -m src.data.vectorize_parquet --processed-root data/processed --source-metadata data/processed/metadata.json --out-root data/vectorized --overwrite
```

### 训练

```bash
python -m src.cli.main train --config configs/experiments/main/shared_bottom_esmm.yaml
```

如果想直接看当前主线里 CTCVR 更高的点，可以改成：

```bash
python -m src.cli.main train --config configs/experiments/main/esmm_mmoe_nogate.yaml
```

### 评估

训练结束后，`runs/` 下会生成新的时间戳目录，再用该目录里的配置和 checkpoint 做评估：

```bash
python -m src.cli.main eval --config runs/<your_run_dir>/config.yaml --ckpt runs/<your_run_dir>/ckpt_best.pt --split valid --save-preds
```

如果只是想复核仓库里已有的主展示结果，可以直接跑：

```bash
python -m src.cli.main eval --config runs/shared_bottom_esmm_20260215_060628/config.yaml --ckpt runs/shared_bottom_esmm_20260215_060628/ckpt_best.pt --split valid --save-preds
```

补充说明：

- 想一次性跑更完整的结构演进实验，可以看 [`scripts/run_interview_chain.py`](scripts/run_interview_chain.py) 或直接 `make interview-chain`。
- 想看 featuremap 证据链，可以额外跑 `eda` / `eda-extra`，入口在 [`src/cli/main.py`](src/cli/main.py)。

## 7. 项目亮点（面试视角）

- 为什么要防 `entity_id` 泄漏：Ali-CCP 的公共特征会复用到同一实体的多条曝光，如果随机切分，valid 很容易偷看到训练时已经出现过的实体侧信息。这里我用的是 entity 级稳定哈希切分，让 valid 更接近“新实体/新样本”场景。
- 为什么 ESMM 上还要继续做 MMoE / PLE：ESMM 解决的是样本选择偏差，不等于共享结构本身就合理了。CTR 和 CVR / CTCVR 仍然可能在共享表示上互相拖累，所以后面才会继续看软共享、shared/private experts 和异构专家。
- 为什么会有跷跷板问题：当前仓库里的 λ sweep 已经能看到 CTCVR 往上推时 CTR 会有回撤，这不是一句“多任务很难”就能带过的事。仓库里我用 gate 分布、dead expert、Gini 和 gradient conflict 去定位这种回撤是不是共享资源分配出了问题。
- 为什么 Focal Loss 对 CVR / CTCVR 是合理方向，但我没把它写成主结论：从标签分布上看它很合理，代码和测试也都接好了；但当前仓库记录显示它不是一上就能稳定涨点的招，所以我把它保留成配置化实验项，而不是“必涨技巧”。
- 为什么要做诊断指标而不是只看最终 AUC：很多坏现象在最终 AUC 之前就已经出现在路由和梯度里了。比如专家塌缩、校准偏差、funnel gap，这些都决定了你怎么解释模型，而不只是最后报一个数字。

## 8. 项目局限与后续可做方向

- 当前主线成功复跑主要集中在 `shared_bottom_esmm` 和 `esmm_mmoe_nogate`，PLE 的主线复跑还没有收敛到稳定结果，说明结构实现虽然有了，但工程稳定性还需要继续打磨。
- 结果全部是公开 Ali-CCP 上的离线指标，和工业真实流量、延迟约束、延迟转化、曝光偏差之间还有距离。
- 现在的评估重点还是 `AUC / logloss / ECE / funnel gap`。更系统的 calibration、counterfactual / off-policy 评估、线上指标映射，这个仓库里还没有做深。
- Focal / residual / Pareto 这些优化方向已经有实现和对比，但还不够稳定，不适合现在就定成默认方案。
- 数据链路里 `process` 和 `vectorize` 还是两步，后面可以继续合并成更顺手的一条 CLI。

## 9. 仓库适合怎么阅读

### 面试官快速阅读路径

1. 先看这份 README 的第 1、3、4、7 节，基本就能知道项目背景、我做了什么、结果怎么解释。
2. 然后直接点这几个文件：[`src/data/split.py`](src/data/split.py)、[`src/data/token_select.py`](src/data/token_select.py)、[`src/models/mtl/mmoe.py`](src/models/mtl/mmoe.py)、[`src/utils/expert_health_diag.py`](src/utils/expert_health_diag.py)。
3. 如果想看一眼真实实验产物，再看 [`runs/shared_bottom_esmm_20260215_060628/eval.json`](runs/shared_bottom_esmm_20260215_060628/eval.json) 和 [`reports/lambda_sweep/lambda_sweep_summary.md`](reports/lambda_sweep/lambda_sweep_summary.md)。
4. 如果要追问“你为什么说这不是盲目堆模型”，就看 [`reports/attempt_analyse/interview_series_report.md`](reports/attempt_analyse/interview_series_report.md) 或 [`docs/interview_chain.md`](docs/interview_chain.md)。

### 想复现实验的阅读路径

1. 先看 [`configs/dataset/aliccp.yaml`](configs/dataset/aliccp.yaml) 和 [`configs/dataset/featuremap.yaml`](configs/dataset/featuremap.yaml)，确认数据路径和 featuremap。
2. 再按顺序看 [`src/data/canonical.py`](src/data/canonical.py) -> [`src/data/split.py`](src/data/split.py) -> [`src/data/processed_builder.py`](src/data/processed_builder.py) -> [`src/data/vectorize_parquet.py`](src/data/vectorize_parquet.py)。
3. 接着看 [`configs/experiments/main`](configs/experiments/main) 里几份主线配置，尤其是 [`configs/experiments/main/shared_bottom_esmm.yaml`](configs/experiments/main/shared_bottom_esmm.yaml) 和 [`configs/experiments/main/esmm_mmoe_nogate.yaml`](configs/experiments/main/esmm_mmoe_nogate.yaml)。
4. 训练入口是 [`src/cli/main.py`](src/cli/main.py) -> [`src/train/trainer.py`](src/train/trainer.py) -> [`src/train/loops.py`](src/train/loops.py)，评估入口是 [`src/eval/run_eval.py`](src/eval/run_eval.py)。
5. 最后再回头看 `runs/`、[`reports/eda`](reports/eda)、[`reports/lambda_sweep`](reports/lambda_sweep) 和 [`tests`](tests)，基本就能把这个仓库的思路串起来。
