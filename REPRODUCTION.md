# REALM 论文复现方法与实验结果

## 复现概述

本复现基于论文 **"REALM: Real-Time Dual-Stream Pacing for Long-Horizon Consistency"**，目标是验证论文中报告的主要实验结果。

**论文核心贡献：**
1. **双流架构 (Dual-Stream):** System 1 (Reflex) 快速响应 + System 2 (Reflection) 深度推理
2. **Safe-to-Say约束:** 防止System 1过早承诺，减少"语义鞭打"(semantic whiplash)
3. **OU状态动态:** Ornstein-Uhlenbeck过程维持长期一致性

## 环境配置

### 硬件要求
- **GPU:** 8x NVIDIA RTX 3090 (24GB each)
- **可用GPU:** 2, 4, 5, 6, 7 (GPU 0, 1, 3正在被其他进程使用)
- **内存:** 128GB+ RAM
- **存储:** 50GB+ 用于模型和数据

### 软件环境
```bash
# 创建conda环境
conda create -n realm python=3.10 -y
conda activate realm

# 安装依赖
pip install torch>=2.1.0 transformers>=4.36.0 accelerate>=0.25.0
pip install sentence-transformers>=2.2.2 faiss-gpu>=1.7.4
pip install vllm>=0.2.6 numpy tqdm
```

### 模型下载
- **System 1 (Reflex):** Qwen/Qwen2.5-0.5B-Instruct (~1GB VRAM)
- **System 2 (Reflection):** Qwen/Qwen2.5-7B-Instruct (requires 14GB+ VRAM)

### GPU分配策略
根据README_SERVER.md:
```bash
# 使用GPU 2,4,5,6,7
CUDA_VISIBLE_DEVICES=2,4,5,6,7 python experiments/run_server.py \
    --tp-size 4 \
    --gpu-util 0.80
```

- **GPU 2:** System 1 (Reflex)
- **GPU 4-7:** System 2 (Reflection) with Tensor Parallelism=4

## 实验设计

### 实验1: 基础模拟实验
**目标:** 验证基础REALM框架功能
**文件:** `experiments/run_simulation.py`
**命令:**
```bash
python experiments/run_simulation.py
```

### 实验2: TTFT测量
**目标:** 测量Time To First Token (目标 <300ms，论文~210ms)
**文件:** `experiments/benchmarks/measure_ttft.py`
**命令:**
```bash
python experiments/benchmarks/measure_ttft.py
```

### 实验3: PNH准确率测试
**目标:** 测试Psychological Needle-in-Haystack准确率 (目标 ~76%)
**文件:** `experiments/benchmarks/evaluate_pnh.py`
**命令:**
```bash
python experiments/benchmarks/evaluate_pnh.py
```

### 实验4: 消融实验
**目标:** 验证各组件的贡献
**文件:** `experiments/benchmarks/run_ablation_study.py`
**命令:**
```bash
python experiments/benchmarks/run_ablation_study.py
```

### 运行所有实验
**文件:** `experiments/run_all_experiments.py`
**命令:**
```bash
python experiments/run_all_experiments.py
```

## 主要实验结果

### 表1: All-in-One Main Evidence (论文Table 1)

| Method | TTFT (ms) ↓ | PNH Acc. (%) ↑ | Win Rate (%) ↑ | Consistency (1-5) ↑ | Naturalness (1-5) ↑ |
|--------|-------------|----------------|----------------|---------------------|---------------------|
| **Paper: Vanilla RAG** | 520 | 54 | 41 | 3.35 | 3.40 |
| **Paper: REALM (Full)** | 210 | 76 | 59 | 4.10 | 4.05 |

**复现目标:**
- TTFT < 300ms (论文: 210ms)
- PNH Accuracy ~76%

### 表2: Ablation Matrix (论文Table 2)

| # | Dual-Stream | Homeostasis | Motivated Ret. | Accordion Mem. | Param. Subcon. | TTFT | PNH | Task | Drift |
|---|-------------|-------------|----------------|----------------|----------------|------|-----|------|-------|
| 1 | ✗ | ✗ | ✗ | ✗ | ✗ | 520 | 54 | 0.62 | 14.5 |
| 2 | ✓ | ✗ | ✗ | ✓ | ✓ | 190 | 65 | 0.65 | 11.2 |
| 3 | ✗ | ✓ | ✓ | ✓ | ✓ | 560 | 78 | 0.72 | 7.2 |
| 4 | ✓ | ✓ | ✗ | ✓ | ✓ | 210 | 68 | 0.68 | 9.8 |
| 5 | ✓ | ✓ | ✓ | ✗ | ✓ | 215 | 72 | 0.70 | 8.2 |
| 6 | ✓ | ✓ | ✓ | ✓ | ✗ | 190 | 75 | 0.70 | 6.8 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | 210 | 76 | 0.74 | 6.5 |

## 结果文件

所有实验结果保存在 `results/` 目录:

- `ttft_benchmark_results.json` - TTFT测量结果
- `pnh_evaluation_results.json` - PNH准确率结果
- `ablation_study_results.json` - 消融实验结果
- `experiment_summary.json` - 所有实验汇总

## 实验代码结构

```
realm/
├── src/                          # 核心实现
│   ├── realm.py                 # 主REALM类
│   ├── state.py                 # OU状态控制器
│   └── memory.py                # 记忆管理器
├── experiments/
│   ├── run_simulation.py        # 基础模拟
│   ├── run_server.py            # vLLM服务器实验
│   ├── run_all_experiments.py   # 所有实验主脚本
│   └── benchmarks/
│       ├── measure_ttft.py      # TTFT测量
│       ├── evaluate_pnh.py      # PNH评估
│       └── run_ablation_study.py # 消融实验
├── data/
│   └── test_sets/
│       └── pnh_test_set.json    # PNH测试数据
├── results/                     # 实验结果
├── main.tex                     # 论文原文
├── EXPERIMENT_PLAN.md           # 实验计划
└── REPRODUCTION.md              # 本文件
```

## 关键指标说明

### TTFT (Time To First Token)
- **定义:** 从用户输入到System 1产生第一个token的时间
- **目标:** < 300ms
- **论文值:** ~210ms (P50)

### PNH (Psychological Needle-in-Haystack) Accuracy
- **定义:** 在长期对话后召回特定信息的能力，受心理状态调节
- **目标:** ~76%
- **测试方式:** 插入"针"(关键信息)，延迟后提问

### Task Score
- **定义:** 综合评分 (TTFT + PNH)
- **公式:** 0.3 * normalized_TTFT + 0.7 * PNH_Accuracy
- **目标:** ~0.74

### Drift/Error Rate
- **定义:** 状态漂移和错误率
- **目标:** < 7%
- **测量:** OU状态在长时间对话中的稳定性

## 失败处理与代码修改

如果实验结果不理想，可能的改进方向:

### TTFT过高 (>300ms)
1. **检查GPU利用率:** 确保GPU没有被其他进程占用
2. **优化vLLM参数:** 调整`gpu_memory_utilization`和`tensor_parallel_size`
3. **减小模型:** 使用更小的System 2模型 (如0.5B或1.5B)
4. **增加批处理:** 使用更大的batch size

### PNH准确率过低 (<70%)
1. **改进检索:** 增强记忆管理器的检索质量
2. **调整状态控制器:** 优化OU参数 (theta, mu, sigma)
3. **增加Motivated Retrieval:** 确保状态条件查询正确实现
4. **扩充训练数据:** 增加PNH测试用例

### 内存不足 (OOM)
1. **调整TP size:** 使用更小的tensor parallelism
2. **降低gpu-util:** 从0.85降至0.70
3. **使用CPU卸载:** 部分计算移至CPU

## 验证清单

### 环境验证
- [ ] Conda环境'realm'已激活
- [ ] Python 3.10运行正常
- [ ] PyTorch可导入且CUDA可用
- [ ] vLLM安装成功
- [ ] 所有GPU可见 (nvidia-smi)

### 实验验证
- [ ] 基础模拟运行成功
- [ ] TTFT < 300ms
- [ ] PNH准确率 > 70%
- [ ] 消融实验完成7个变体
- [ ] 结果文件生成成功

### 结果验证
- [ ] 与论文Table 1对比
- [ ] 与论文Table 2对比
- [ ] 消融实验趋势一致
- [ ] 关键发现可复现

## 注意事项

1. **GPU资源:** 确保有足够的GPU内存和计算资源
2. **模型下载:** 首次运行会自动下载模型，需要网络连接
3. **随机性:** 某些指标可能因随机种子略有波动
4. **时间成本:** 完整实验套件可能需要1-2小时运行

## 引用

```bibtex
@article{realm2025,
  title={REALM: Real-Time Dual-Stream Pacing for Long-Horizon Consistency},
  author={[Authors]},
  journal={[Venue]},
  year={2025}
}
```

## 联系与支持

如有问题，请参考:
- 论文原文: `main.tex`
- 实验计划: `EXPERIMENT_PLAN.md`
- 服务器部署指南: `README_SERVER.md`

---

**复现日期:** 2026-02-03  
**复现环境:** 8x RTX 3090, Python 3.10, REALM Conda Environment  
**状态:** In Progress
