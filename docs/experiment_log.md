# HYDRA 复现实验记录

## 实验目标

复现 HYDRA (EMNLP 2025) 在 WOS46985 数据集上的结果，使用与当前实验相同的编码器 (SciBERT) 进行公平对比。

## 数据集

- 来源：官方 WOS46985 (原始 X.txt/YL1.txt/YL2.txt/Y.txt)
- 父类：7 (CS, ECE, Psychology, MAE, Civil, Medical, biochemistry)
- 子类：134（全局编号，Y.txt 0-133）
- 总样本：46985
- 划分：train 30070 / val 7518 / test 9397
- 划分方式：HYDRA 官方策略 — np.random.seed(7) shuffle, train_test_split(test_size=0.2, random_state=0) x2
- 文本预处理：与 HYDRA 官方一致（clean_str: lower, 缩写展开, 多空格合并）

### 数据说明

原始 WOS46985 的 YL1/YL2 使用局部编号（YL2 在每个 parent 下从 0 开始），Y.txt 使用全局编号（0-133）。HYDRA 使用 Y.txt 全局编号作为 child label（134 类）。Data.xlsx 版本有 143 个子类（因为 Depression 和 Schizophrenia 在 Medical 和 Psychology 下各出现一次），与原始 Y.txt 的 134 类不一致。为确保与 HYDRA 论文一致，使用原始 txt 文件。

## 实验配置

| 配置项 | 值 | 说明 |
|--------|------|------|
| 编码器 | allenai/scibert_scivocab_uncased | 与当前实验一致 |
| 层级结构 | 2级 (parent=7, child=134) | |
| 嵌入投影 | Linear(768, 768x2=1536) | |
| 池化方式 | CLS | |
| 分类头 | 2层MLP + LayerNorm + ReLU + Dropout(0.2) | |
| 学习率 | 3.5e-5 | |
| 批大小 | 32 | |
| 最大序列长度 | 256 | 原文512，WOS摘要P95=432 |
| 最大训练轮数 | 30 | |
| 早停耐心 | 5 | |
| 阈值 | 0.5 | |
| Warmup步数 | 500 | |
| 权重衰减 | 0.01 | |
| 梯度裁剪 | 1.0 | |
| 种子 | 42, 1, 2, 3, 4 | 5次取平均 |
| 设备 | MPS (Apple M3 Pro) | sandbox 环境回退 CPU |
| 动态padding | 启用 | 按batch内最长序列pad |

## HYDRA 三种变体

| 变体 | 损失函数 | 论文WOS结果 (RoBERTa) Micro-F1 / Macro-F1 |
|------|----------|------------------------------------------|
| Local Heads Only | Sum BCE_j | 86.90 / 81.18 |
| Local + Global Head | Local + alpha*Global | 86.91 / 81.22 |
| Local + Nested Head | Local + alpha*Nested | 86.90 / 81.14 |

## 模型参数量

| 变体 | 参数量 |
|------|--------|
| HYDRA Local | 120,988,557 |
| HYDRAGlobal | 123,569,178 |
| HYDRANested | 121,426,458 |

## 评估指标

| 指标 | 口径 | 说明 |
|------|------|------|
| Parent Acc | argmax | 父类准确率 |
| Child Acc | argmax | 子类准确率 |
| Child Micro-F1 | argmax | 子类微平均F1（论文主指标） |
| Child Macro-F1 | argmax | 子类宏平均F1 |
| Parent Micro-F1 | argmax | 父类微平均F1 |
| Parent Macro-F1 | argmax | 父类宏平均F1 |
| Overall Micro-F1 | threshold 0.5 | 全标签微平均F1 |
| Overall Macro-F1 | threshold 0.5 | 全标签宏平均F1 |
| Hierarchical Consistency | - | 预测满足层级约束比例 |
| Per-level Precision/Recall | micro+macro | 精确率/召回率 |

## 与当前实验对比

| 方法 | Child Micro-F1 | Child Macro-F1 | 备注 |
|------|----------------|----------------|------|
| 当前方法 (主实验) | 84.99 | 84.44 | SciBERT, lambda=0.2, alpha=0.5 |
| HYDRA (论文, RoBERTa) | 86.91 | 81.22 | 论文报告值 |
| HYDRA (复现, SciBERT) | ? | ? | 本实验 |

## 训练速度

- CPU (sandbox): ~13.3s/batch, ~3.5h/epoch
- MPS 预估: ~2-3s/batch, ~30-50min/epoch
- 预计收敛 10-15 epoch

## 运行方式

```bash
# 单次实验
cd /Users/fuhao10/project/mine/htc
bash scripts/run_hydra_wos.sh local 42

# 全量实验 (3变体 x 5种子)
bash scripts/run_all_experiments.sh
```

## 实验进展

- [x] 环境搭建 (uv, 依赖)
- [x] 数据准备 (原始 X.txt/YL1.txt/YL2.txt/Y.txt, 134 child labels)
- [x] 代码实现 (HYDRA 3变体 + 评估)
- [x] 端到端流程验证 (数据加载、模型前向、训练步骤、评估指标)
- [ ] HYDRA Local Heads Only (seed=42)
- [ ] HYDRA Local + Global Head (seed=42)
- [ ] HYDRA Local + Nested Head (seed=42)
- [ ] 多seed实验
- [ ] 结果整理与分析

## 重要注意事项

1. 当前 sandbox 环境 MPS 不可用，需在用户终端中运行以获得 MPS 加速
2. 不要使用 torch.compile — 在 MPS 上会导致 Metal shader 编译错误
3. 数据划分使用 HYDRA 官方策略 (seed=7 + train_test_split)
4. 当前实验的 child 标签口径可能不同（143 vs 134），后续需统一
