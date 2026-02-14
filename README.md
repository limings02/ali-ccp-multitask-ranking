# Ali-CCP MTL Ranker

Canonical preprocessing pipeline (scheme C: samples + tokens) for the Ali-CCP dataset.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # On PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -e .
```

## Generate Canonical Data

The CLI builds canonical data via `src/data/canonical.py` (`build_canonical_aliccp`, seed defaults to 20260121 for stability). Paths in `configs/dataset/aliccp.yaml` are relative to the repo root. Override raw inputs via env vars `ALICCP_SKELETON_PATH` and `ALICCP_COMMON_FEATURES_PATH`.

### Run via Python
```bash
python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
```

### Run via Bash helper
```bash
bash scripts/canonical.sh configs/dataset/aliccp.yaml --nrows 100000 --overwrite
```

Outputs (relative to repo root):

- `data/interim/aliccp_canonical/common_features.sqlite` - entity_id -> feat_str lookup
- `data/interim/aliccp_canonical/samples_train.parquet` - row-wise labels/index columns
- `data/interim/aliccp_canonical/tokens_train/tokens_train.part_*.parquet` - long token table shards
- `data/interim/aliccp_canonical/manifest.json` - reproducibility metadata (paths, params, seed, parts)

### Quick Commands (PowerShell)

- Debug first 100000 rows (canonical):
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
  ```
- Full run (default `nrows=null` in config):
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
  ```

## Compact Tokens (optional)

After canonical, you can compact many small token parquet files into ~100 larger parts:

- Bash helper:
  ```bash
  bash scripts/compact.sh configs/dataset/aliccp.yaml --target-parts 100 --overwrite
  ```
- In-place replace tokens (backup to tokens_train.bak):
  ```powershell
  python -m src.cli.main compact --config configs/dataset/aliccp.yaml --target-parts 100 --overwrite --inplace
  ```

## Split & Split-Tokens

- Split samples by entity hash:
  ```bash
  python -m src.cli.main split --config configs/dataset/aliccp.yaml --overwrite
  ```
- Filter tokens using the split:
  ```bash
  python -m src.cli.main split-tokens --config configs/dataset/aliccp.yaml --overwrite
  ```


## EDA

- Run EDA (pyarrow fallback when duckdb not installed):
  ```bash
  python -m src.cli.main eda --config configs/dataset/aliccp.yaml
  python -m src.cli.main eda --config configs/dataset/aliccp.yaml --overwrite
  ```
- Bash helper:
  ```bash
  bash scripts/eda.sh configs/dataset/aliccp.yaml --overwrite
  ```
- Outputs (under `config.eda.out_dir` / `config.eda.report_dir`):
  - `sanity.json`, `samples_stats_{train,valid}.json`, `entity_freq_{train,valid}.parquet`
  - `row_nnz_hist_{train,valid}.parquet`, `field_stats_train.parquet`, `field_topk_train.parquet`
  - `fid_lift_train.parquet`, `drift_summary.parquet`, `reports/drift_report.md`

## EDA Extra（FeatureMap 精细化证据）

- Debug 100k：`python -m src.cli.main eda-extra --config configs/dataset/aliccp.yaml --in-stats data/stats/eda_v1 --out data/stats/eda_extra_v1 --plots reports/eda/eda_extra_v1 --debug-sample --overwrite`
- Full：`python -m src.cli.main eda-extra --config configs/dataset/aliccp.yaml --in-stats data/stats/eda_v1 --out data/stats/eda_extra_v1 --plots reports/eda/eda_extra_v1 --overwrite`
- 产物（结构化 + 元数据）：`data/stats/eda_extra_v1/*.parquet`, `data/stats/eda_extra_v1/*.json`, `data/stats/eda_extra_v1/metadata.json`
- 可视化与 FeatureMap 建议：`reports/eda/eda_extra_v1/*.png`, `featuremap_patch.yaml`, `featuremap_rationale.md`, `featuremap_diff.md`, `reports/eda/eda_extra_v1/metadata.json`
- 映射规则：
  1) `oov_curve.parquet` 选 head_size 与编码：row/token OOV 低于 2%→vocab/小 hybrid；5% 以内→hybrid；更高→hash。
  2) `truncation_loss_curve.parquet` 选 max_len：最小 K 使 `retained_token_frac>=0.995` 且 `retained_row_full_frac>=0.95`（结合 `field_length_quantiles` p99 校验）。
  3) `hash_collision_est.parquet` 选 hash_bucket：碰撞率 <2%（线上）/<5%（本地）取最小可行 bucket。
  4) `val_psi.parquet` + `val_profile_train_valid.parquet`：若 PSI 偏高或极值比例异常，建模侧需 clip/log/分桶，并将 nan/inf 计入上线监控。


## Process (split -> processed)

将 split 的 samples + tokens 按 featuremap.yaml 处理成训练可直接读取的 processed parquet：

- Python 命令（推荐）：
  ```bash
  python -m src.cli.main process \
    --config configs/dataset/featuremap.yaml \
    --split-dir data/splits/aliccp_entity_hash_v1 \
    --out data/processed/aliccp_entity_hash_v1 \
    --batch-size 500000
  ```
- PowerShell：
  ```powershell
  python -m src.cli.main process --config configs/dataset/featuremap.yaml --split-dir data/splits/aliccp_entity_hash_v1 --out data/processed --batch-size 100000
  ```
- python -m src.cli.main process --config configs/dataset/featuremap.yaml --split-dir data/splits/aliccp_entity_hash_v1 --out data/processed --batch-size 100000 --log-level DEBUG

- Bash helper：
  ```bash
  bash scripts/process.sh
  ```

**输出目录**：
- `data/processed/aliccp_entity_hash_v1/train/*.parquet`
- `data/processed/aliccp_entity_hash_v1/valid/*.parquet`
- `data/processed/aliccp_entity_hash_v1/metadata.json` - 包含 rows、feature_meta、token_policy 等

**列设计**（对齐 featuremap）：
- 标签：`y_ctr`(click), `y_cvr`(conversion), `y_ctcvr`, `click_mask`, `row_id`, `entity_id`
- 单值特征：`f{src}_{field}_idx` (int64), `f{src}_{field}_val` (float32, 若 use_value=True)
- 多值特征：`f{src}_{field}_idx` (list<int64>), `f{src}_{field}_val` (list<float32>), `f{src}_{field}_len` (int32)

**Dataset/Loader**：`src/data/dataset.py` 提供 `ProcessedDataset` 与 `build_dataloader`，collate 时在内存做 padding（落盘不填充）。

```python
from src.data.dataset import build_dataloader

# 注意：IterableDataset 不支持 shuffle，必须显式传 shuffle=False
train_loader = build_dataloader(
    "data/processed/aliccp_entity_hash_v1/train",
    batch_size=1024,
    shuffle=False,  # 必须为 False，否则会抛出 ValueError
    num_workers=4,
)
```

**重要注意事项**：
1. **Hash 索引是 field-aware 的**：相同 fid 在不同 field 会哈希到不同索引，避免跨字段碰撞
2. **Hybrid 编码紧凑**：tail 索引紧跟 vocab 区间，`total_num_embeddings = vocab_num_embeddings + hash_bucket`
3. **Missing value 统一为 1.0**：multi-hot 和 single-hot 的缺失值均为 1.0，便于 weighted pooling
4. **OOV 统计**：仅 vocab/hybrid-head 实际 OOV 计入统计，hash 编码不报告 OOV


## DeepFM Backbone Smoke Run

- 配置文件：`configs/models/backbone_deepfm.yaml`
- 运行命令：
  ```bash
  python -m src.train.run_backbone_deepfm --config configs/model/backbone_deepfm.yaml
  ```
- 预期输出：运行 `runtime.steps` 个 batch（默认 1），日志包含 `h shape=(batch_size, out_dim)`，默认形状为 `(128, 128)`，完成后正常退出。

## Train: DeepFM + SharedBottom test
 
训练 1~2 个 epoch（单机单卡/CPU）：

```bash
python -m src.cli.main train --config configs/experiments/deepfm_sharedbottom_train.yaml
python -m src.cli.main train --config configs/experiments/deepfm_sharedbottom_train_ctr.yaml
python -m src.cli.main train --config configs/experiments/deepfm_sharedbottom_train_cvr.yaml

runtime:
  resume_path: "runs/your_experiment_dir/ckpt_best.pt"
```
## Phase 5 Eval (AUC/LogLoss/ECE/Funnel
- 新增评估模块：src/eval/metrics.py, src/train/infer.py, src/eval/calibration.py, src/eval/funnel.py 
- CVR 指标按 click_mask 过滤（仅在点击样本上计算） 
- 若未安装 sklearn，AUC 字段为 null，程序不会报错 )```

```bash
python -m src.cli.main eval --config runs\test1_classic_mmoe_20260211_205242\config.yaml --ckpt runs\test1_classic_mmoe_20260211_205242\ckpt_best.pt --split valid
```

## MMoE
```bash
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml  
```

## 更新速览
- 新增 MMoE 多任务结构，可通过 `model.mtl=mmoe` 启用（示例：`configs/experiments/mtl_mmoe.yaml` / `configs/model/mtl_mmoe_dual_sparse.yaml`）。
- 优化器框架升级为 OptimizerBundle，支持 `optim.type = single | dual_sparse_dense`，可选稀疏优化器（SparseAdam）；旧配置与旧 ckpt 仍兼容。
- 支持稀疏梯度：在 `embedding.sparse_grad=true` 时 EmbeddingBag 使用稀疏权重并由 SparseAdam 更新；未开启时保持原有 dense 行为。
- 重要护栏：`optim.sparse.enabled=true` 且未找到稀疏参数会报错，除非 `allow_fallback_if_empty=true` 才降级为 dense；日志会输出稀疏/稠密参数统计与加载的优化器类型。

## PLE-Lite（方案1：消融对照组）

PLE-Lite（Progressive Layered Extraction Lite）新增实现，作为 MMoE 的对照组消融模型。

### 核心改动
- **新增文件**：`src/models/mtl/ple.py` - PLE-Lite 模型实现
  - Shared Experts：所有任务共享（默认 4 个）
  - Task-Specific Experts：每个任务专属（默认每任务 1 个）
  - 每个 task 的 gate 只在 (shared + 本任务 specific) experts 上做 softmax
  
- **新增配置**：`configs/model/mtl_ple.yaml` 和 `configs/experiments/mtl_ple.yaml`
  - 完全对标 MMoE 配置，确保公平对比
  - 新增参数 `gate_reg_scope`：控制正则化范围
    - `"shared_only"`（默认）：只对 shared experts 的 gate 权重做 entropy/kl 正则
    - `"all"`：对全部 experts 做正则

- **build.py 新增分支**：`mtl="ple"` 时自动构建 PLE 模型

- **测试用例**：`tests/test_model_forward.py::test_model_forward_ple`
  - 验证 forward 输出格式、gate_reg_loss 有效性、梯度反传

### 运行消融实验
```bash
# PLE-Lite 训练
python -m src.cli.main train --config configs/experiments/mtl_ple_lite.yaml

# 对标 MMoE
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml
```

### 设计特点
- **共享 vs 专属路由**：通过 gate_reg_scope="shared_only" 只约束 shared experts 的路由均匀性，避免过度正则化 task-specific 专家
- **最小侵入**：复用 MMoE 的 gate 稳定化、composer、TaskHead 等接口，仅改变 expert 组织结构
- **可复现性**：与 MMoE 共享 ESMM/Loss/Metrics/Optimizer 完整体系


python tools/run_ablation_main.py && python tools/summarize_main_ablation.py --manifest reports/main_ablation/main_ablation_runs.json --out-dir reports/main_ablation
