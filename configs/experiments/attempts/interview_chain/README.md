# Interview Chain - 快速启动指南

## 🎯 一键运行所有实验

```bash
# 方式一：使用 Makefile（推荐）
make interview-chain

# 方式二：直接调用 Python 脚本
python scripts/run_interview_chain.py
```

## 📊 查看结果

```bash
# 查看汇总表格（可复制到 Excel）
cat runs/interview_chain/summary.csv

# 查看增量分析（ΔAUC 计算）
cat runs/interview_chain/delta_analysis.txt

# 查看完整信息（包含 run_dir 路径）
cat runs/interview_chain/summary.json
```

## 🔧 高级用法

```bash
# 断点续跑（跳过已完成的实验）
make interview-chain-resume
python scripts/run_interview_chain.py --resume
# 只打印命令（调试用）
make interview-chain-dry-run

# 跳过前两个单任务基线（加速调试）
python scripts/run_interview_chain.py --skip E0a,E0b

# 自定义输出目录
python scripts/run_interview_chain.py --output my_interview_chain
```

## ✅ 配置校验

```bash
# 检查 7 个配置文件的公共字段是否一致
python scripts/validate_interview_chain_configs.py
```

## 📖 详细文档

查看 [docs/interview_chain.md](../docs/interview_chain.md) 了解：
- 实验设计原理
- 配置文件说明
- 指标解读
- 面试讲法
- 故障排查

## 🚀 预期结果

| 实验 | 描述 | 预期 CTCVR AUC | 预期提升 |
|-----|------|---------------|---------|
| E0a | 单任务 CTR | - | - |
| E0b | 单任务 CVR | - | - |
| E1 | Hard Sharing | - | baseline |
| E2 | + ESMM v2 | 0.646 | +0.34% |
| E3 | + MMoE | 0.649 | +0.52% |
| E4 | + PLE（同构） | 0.651 | +0.34% |
| E5 | + PLE（异构） | 0.653 | +0.34% |

**累计收益**：E5 相比 E1，CTCVR AUC 提升约 **0.78%**

---

**总耗时**：约 8-12 小时（取决于 GPU 性能）  
**GPU 需求**：单卡 V100/A100（16GB+ 显存）  
**数据集**：Ali-CCP（需提前完成 `canonical` → `process` 步骤）
