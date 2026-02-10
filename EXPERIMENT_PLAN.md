# REALM 实验复现方案

## 1. 实验概述

本文是 REALM (Real-Time Dual-Stream Pacing for Long-Horizon Consistency) 的论文复现实验。该论文提出了一种双流架构，通过System 1 (Reflex) 快速响应和System 2 (Reflection) 深度推理的结合，实现在保持实时交互响应性的同时维持长期一致性。

## 2. 实验环境

- **服务器**: 8x RTX 3090 (24GB each)
- **可用GPU**: 2, 4, 5, 6, 7 (GPU 0, 1, 3正在使用中)
- **Python**: 3.10
- **Conda环境**: realm

## 3. 论文中的实验设计

### 3.1 主要实验 (Main Results)

#### 表1: All-in-One Main Evidence Table (tab:all-in-one)

| Method | TTFT (ms) ↓ | PNH Acc. (%) ↑ | Win Rate (%) ↑ | Consistency (1-5) ↑ | Naturalness (1-5) ↑ | Waiting Anxiety (1-5) ↓ |
|--------|-------------|----------------|----------------|---------------------|---------------------|------------------------|
| Vanilla RAG | 520 | 54 | 41 | 3.35 | 3.40 | 3.32 |
| Ours w/o State-Awareness | 190 | 65 | - | - | - | - |
| Ours w/o Dual-Stream | 560 | 78 | - | - | - | - |
| REALM (Full) | 210 | 76 | 59 | 4.10 | 4.05 | 2.62 |

**目标指标**:
- TTFT (Time To First Token) < 300ms
- PNH (Psychological Needle-in-Haystack) Accuracy ~76%

### 3.2 消融实验 (Ablation Study)

#### 表2: Ablation Matrix (tab:ablation-matrix)

| # | Dual-Stream | Homeostasis | Motivated Retrieval | Accordion Memory | Parametric Subconscious | TTFT | PNH Acc | Task Score | Drift/Error |
|---|-------------|-------------|---------------------|------------------|------------------------|------|---------|------------|-------------|
| 1 | ✗ | ✗ | ✗ | ✗ | ✗ | 520 | 54 | 0.62 | 14.5 |
| 2 | ✓ | ✗ | ✗ | ✓ | ✓ | 190 | 65 | 0.65 | 11.2 |
| 3 | ✗ | ✓ | ✓ | ✓ | ✓ | 560 | 78 | 0.72 | 7.2 |
| 4 | ✓ | ✓ | ✗ | ✓ | ✓ | 210 | 68 | 0.68 | 9.8 |
| 5 | ✓ | ✓ | ✓ | ✗ | ✓ | 215 | 72 | 0.70 | 8.2 |
| 6 | ✓ | ✓ | ✓ | ✓ | ✗ | 190 | 75 | 0.70 | 6.8 |
| **7** | ✓ | ✓ | ✓ | ✓ | ✓ | 210 | 76 | **0.74** | **6.5** |

### 3.3 基准比较

#### 表3: Broader Baselines (tab:broader-baselines-main)

比较方法包括: MemGPT, Streaming CoT, TiMem, PsyAgent, Sophia, CAIM

## 4. 实验实现

### 4.1 现有代码结构

```
realm/
├── src/
│   ├── realm.py          # 主REALM类
│   ├── state.py          # OU状态控制器
│   └── memory.py         # 记忆管理器
├── experiments/
│   ├── run_server.py     # 服务器实验(vLLM)
│   └── run_simulation.py # 模拟实验
├── data/                 # 数据目录
└── main.tex             # 论文
```

### 4.2 GPU分配策略

根据README_SERVER.md，我们将使用以下GPU分配:

- **GPU 2**: System 1 (Reflex) - Qwen2.5-0.5B-Instruct (~1GB VRAM)
- **GPU 4-7**: System 2 (Reflection) - Qwen2.5-7B-Instruct with Tensor Parallelism (TP=4)

命令:
```bash
CUDA_VISIBLE_DEVICES=2,4,5,6,7 python experiments/run_server.py \
    --tp-size 4 \
    --gpu-util 0.80
```

## 5. 实验执行计划

### 5.1 实验1: 基础模拟实验

**目标**: 验证基础REALM框架功能
**文件**: `experiments/run_simulation.py`
**预期**: 完成5轮对话，观察状态变化

### 5.2 实验2: TTFT测量

**目标**: 测量Time To First Token
**方法**: 
- 运行服务器实验
- 记录System 1 (Reflex)的首次token生成时间
- 目标: < 300ms

### 5.3 实验3: PNH准确率测试

**目标**: 测试Psychological Needle-in-Haystack准确率
**方法**:
- 创建测试数据集
- 插入"针"(关键信息)后延迟提问
- 测量在正确心理状态下的召回率
- 目标: ~76%

### 5.4 实验4: 消融实验

**目标**: 验证各组件的贡献
**变体**:
1. Vanilla RAG (baseline)
2. w/o State-Awareness
3. w/o Dual-Stream
4. w/o Motivated Retrieval
5. w/o Parametric Subconscious
6. REALM (Full)

## 6. 复现检查清单

### 6.1 环境准备
- [ ] Conda环境创建
- [ ] 依赖安装 (torch, transformers, vllm, faiss-gpu等)
- [ ] GPU可用性检查

### 6.2 模型下载
- [ ] Qwen/Qwen2.5-0.5B-Instruct (System 1)
- [ ] Qwen/Qwen2.5-7B-Instruct (System 2)

### 6.3 实验运行
- [ ] 基础模拟实验
- [ ] TTFT测量
- [ ] PNH准确率测试
- [ ] 消融实验

### 6.4 结果分析
- [ ] 指标对比
- [ ] 可视化
- [ ] 与论文结果对比

## 7. 预期结果

根据论文，我们期望得到以下结果:

1. **TTFT**: REALM (210ms) < Vanilla RAG (520ms) < 300ms阈值 ✓
2. **PNH Accuracy**: REALM (76%) >> Vanilla RAG (54%)
3. **人类评估**: REALM在Consistency (4.10)和Naturalness (4.05)上显著优于baseline
4. **消融分析**: Dual-Stream和State-Awareness都是必要的组件

## 8. 失败处理策略

如果实验结果不理想，我们将:

1. **TTFT过高**: 
   - 检查GPU利用率
   - 优化vLLM参数
   - 调整System 1模型大小

2. **PNH准确率过低**:
   - 改进检索质量
   - 调整状态控制器参数
   - 增强记忆管理

3. **内存不足**:
   - 调整TP size
   - 降低gpu-util参数
   - 使用更小的模型

## 9. 结果保存

所有实验结果将保存到:
- `results/` 目录
- 包括: JSON格式的指标、CSV表格、可视化图表
- 复现方法文档: `REPRODUCTION.md`

---

*Generated for REALM Paper Reproduction*
