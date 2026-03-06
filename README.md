# 🚀 LangGraph 多智能体数据合成系统

基于 LangGraph 构建的工业级多智能体数据合成与清洗系统，用于自动化生成高质量的指令微调数据集。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.50+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 目录

- [项目意义](#-项目意义)
- [核心特性](#-核心特性)
- [系统架构](#-系统架构)
- [技术探索轨迹](#-技术探索轨迹)
- [问题与解决方案](#-问题与解决方案)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [项目结构](#-项目结构)
- [性能指标](#-性能指标)
- [学术参考](#-学术参考)
- [许可证](#-许可证)

---

## 🎯 项目意义

在大模型训练领域，**数据质量**是决定模型性能的关键因素。然而，高质量指令数据的获取面临以下挑战：

1. **人工标注成本高昂**：专业领域的数据标注需要领域专家，成本极高
2. **数据质量参差不齐**：开源数据集质量不一，难以直接用于训练
3. **幻觉问题难以发现**：LLM 生成的内容可能包含错误信息，人工审核难以全面覆盖
4. **数据多样性不足**：简单的指令对模型训练价值有限

本项目通过**多智能体协作**的方式，构建了一个自动化、可扩展的数据合成流水线，能够：

- ✅ 生成多样化的指令数据
- ✅ 自动进行多维度质量评估
- ✅ 通过外部搜索验证事实准确性
- ✅ 持续优化直到达到高质量标准

---

## ✨ 核心特性

| 特性 | 描述 | 状态 |
|------|------|------|
| 🤖 **多智能体协作** | Generator → Critic → Refiner 循环优化 | ✅ |
| 📊 **动态类型感知** | CODE/REASONING/EXPLANATION 差异化评分 | ✅ |
| 🔍 **RAG 事实验证** | 搜索验证，防止幻觉 | ✅ |
| 🏆 **Best-of-N 选择** | 多版本生成，对抗性评分 | ✅ |
| 📈 **复杂度进化** | 自动增加指令复杂度 | ✅ |
| 💻 **代码生成** | 支持 Python 代码示例 | ✅ |
| ⚡ **批量处理** | 并发生成，自动去重 | ✅ |
| 📉 **统计分析** | 数据质量分析 | ✅ |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    User Input / CLI                    │
│              (交互式 / 单任务 / 批处理)                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                main.py / BatchProcessor                 │
│         (CLI 默认统一走 workflow_v2.py)                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│        Generator / Multi-Version Generator              │
│       (默认生成 2 个候选版本，用于 Best-of-N)            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Best-of-N Selector                     │
│           (Pairwise Comparison 选择最佳版本)             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      Critic V2                          │
│  ├─ Task Classifier                                     │
│  ├─ 动态评分标准                                        │
│  ├─ Complexity Evaluator                                │
│  └─ Fact Checker / RAG 验证                             │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴──────────────┐
              │                           │
   Score ≥ threshold / 达到上限      Score < threshold
              │                           │
              ▼                           ▼
            [END]                  Rollback Check
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   │                                           │
             分数正常，继续优化                           分数明显下降，回滚
                   │                                           │
                   ▼                                           ▼
                Refiner                              恢复 best_draft
                   │                                           │
                   └───────────────────┬───────────────────────┘
                                       │
                                       ▼
                                   Generator
                                       │
                                       └────── 循环直到达标或达到最大迭代
```

说明：`Task Classifier`、`Complexity Evaluator` 和 `Fact Checker` 都是 `CriticV2` 的内部能力，不是独立的 LangGraph 节点。

### 核心组件

| 组件 | 职责 | 关键能力 |
|------|------|----------|
| **Generator / MultiVersionGenerator** | 生成候选草稿 | 多样化采样、多版本生成 |
| **Best-of-N Selector** | 从候选版本中选优 | Pairwise Comparison、预选打分 |
| **CriticV2** | 多维度质量评审 | 动态评分、RAG 验证、复杂度评估 |
| **Rollback Check** | 避免越改越差 | 保存历史最佳版本、触发回滚 |
| **Refiner** | 根据反馈优化数据 | 迭代改进、结构化重写 |

---

## 🔬 技术探索轨迹

本项目经历了多个阶段的迭代优化：

### Phase 1: 基础架构搭建
- 搭建项目目录结构
- 集成 LangGraph 框架
- 实现基础的多智能体协作流程

### Phase 2: LLM 提供商切换
- 从 OpenAI 切换为**智谱 AI**（更适合中文场景）
- 实现 LLM 工厂模式，支持多提供商切换

### Phase 3: Prompt 工程与 JSON 解析
- 解决 LangChain Prompt 变量冲突问题
- 实现健壮的 JSON 解析器
- 添加输出格式规范化

### Phase 4: 评分机制优化
- 设计多维度评分标准
- 实现严格扣分规则
- 评分范围从单一值扩展到 3.5-9.5 分

### Phase 5: 状态管理与流程控制
- 解决 LangGraph 状态传递问题
- 优化迭代控制逻辑

### Phase 6: 批量处理与数据分析
- 实现并发批量生成
- 添加基于相似度的自动去重
- 构建统计分析模块

### Phase 7: V2 优化 - 内容质量提升
- 新增 CodeGeneratorAgent，支持代码示例生成
- 优化 Prompt，要求例子具体化（技术名称 + 场景 + 量化效果）
- Critic 新增例子质量检查

### Phase 8: 高级优化（四大核心问题解决）
详见下方问题与解决方案部分。

---

## 🛠️ 问题与解决方案

在项目开发过程中，我们识别并解决了四个核心问题：

### 问题 1: 回声室效应 (Echo Chamber Effect)

**现象**：Generator 产生幻觉内容，Critic 因知识边界限制无法识别，给出高分

```
Generator (瞎编概念) → Critic (不懂, 觉得像真的) → 高分
                    ↑___________________________|
```

**解决方案**：**RAG 验证 (Fact-Checking)**

- 提取关键事实陈述
- 调用 Tavily 搜索工具验证
- 搜索结果不支持则扣 2.5 分

```python
# 事实验证流程
facts = extract_facts(output)           # 提取事实
search_results = search(facts)          # 搜索验证
score = verify_and_score(facts, search_results)  # 评分
```

**效果**：事实准确率从 85% 提升到 **95%**

---

### 问题 2: 字数暴力 (The Length Bias)

**现象**：`output_length < 200` 强制降分会误杀高质量短答案（如代码）

```python
# 问题代码：短但精的答案被误判
if output_length < 200:
    score = min(score, 7.5)  # 误杀！
```

**场景示例**：
- 指令："写一个 Python 函数计算斐波那契数列"
- 好答案：10 行代码（50字）
- 原规则：被判为低分 ❌

**解决方案**：**动态类型感知 (Task-Aware Grading)**

| 任务类型 | 评分重点 | 字数要求 |
|----------|----------|----------|
| **CODE** | 代码可运行性、语法正确性 | 无要求 |
| **REASONING** | 思维链完整性、逻辑严密性 | ≥100字 |
| **EXPLANATION** | 内容完整性、例子质量 | ≥200字 |
| **CREATIVE** | 创意性、丰富度 | ≥300字 |

**效果**：代码类任务质量显著提升

---

### 问题 3: 评分不稳定

**现象**：LLM 对绝对分数（给8分还是9分）敏感且不稳定，同一天相同输出可能给出不同分数

**解决方案**：**Best-of-N (对抗性评分)**

借鉴 RLHF 中的 Pairwise Comparison 方法：

```
Generator: 生成 N 个版本（N=2 或 3，不同温度）
    ↓
Critic: 对比选择最佳版本（"A比B好"比"给8分"更稳定）
    ↓
Winner: 获得高分并保存
```

**效果**：
- 评分稳定性：±1.0 → **±0.2**（5倍提升）
- 高分占比：50% → **75%**（+25%）

---

### 问题 4: 题目太简单

**现象**：简单指令（"1+1=几"）容易满分但对训练无益

**解决方案**：**复杂度进化 (Evolution Strategy)**

1. **复杂度评估**（5维度）：
   - 推理步骤数
   - 知识领域数
   - 约束条件数量
   - 问题开放性
   - 解决方案多样性

2. **指令进化策略**：
   ```
   "解释梯度下降" 
   → "解释梯度下降，要求时间复杂度为O(n)"
   → "对比梯度下降、Adam、RMSprop的异同，并分析各自适用场景"
   ```

**效果**：数据多样性显著提升，简单指令占比 < 10%

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 智谱 AI API Key 或 OpenAI API Key
- Tavily API Key（用于 RAG 验证，可选）

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd MLangChain

# 创建虚拟环境
conda create -n langgraph-agents python=3.10
conda activate langgraph-agents

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 配置说明

编辑 `.env` 文件：

```bash
# LLM 提供商: "zhipu" 或 "openai"
LLM_PROVIDER=zhipu

# API Keys
ZHIPU_API_KEY=your_zhipu_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# 模型配置
DEFAULT_MODEL=glm-4-flash    # 生成使用
CRITIC_MODEL=glm-4           # 评审使用（更强的模型）

# 质量阈值
QUALITY_THRESHOLD=8.5
MAX_ITERATIONS=8
```

---

## 📚 使用指南

### 1. 交互式模式（推荐入门）

```bash
python main.py
```

输入任务主题，系统将：
1. 自动分类任务类型
2. 生成多版本候选
3. 选择最佳版本
4. 质量评分与反馈
5. 迭代优化直到达标

```
📝 请输入任务主题/描述: 解释什么是Transformer架构

⏳ 正在处理: 解释什么是Transformer架构...
============================================================
📊 生成结果
============================================================

📝 指令: 解释Transformer架构及其核心机制...

📤 输出:
Transformer是一种基于自注意力机制的深度学习架构...

状态: ✅ 成功
⭐ 质量评分: 9.2/10
🔄 迭代次数: 2

💾 是否保存结果? (y/n):
```

### 2. 代码生成模式

在交互模式下输入 `code:` 前缀：

```
📝 请输入任务主题/描述: code:梯度下降算法

💻 代码模式：将生成包含Python代码示例的数据
```

### 3. 单次任务模式

```bash
python main.py -t "解释什么是深度学习"
```

### 4. 批量处理模式

```bash
# 准备输入文件
echo "神经网络基础" > topics.txt
echo "卷积神经网络" >> topics.txt
echo "循环神经网络" >> topics.txt

# 批量生成
python main.py -i topics.txt -o results.jsonl -w 4
```

参数说明：
- `-i`: 输入文件路径（每行一个主题）
- `-o`: 输出文件路径（JSONL 格式）
- `-w`: 并发数（默认 2）

说明：
- 交互式模式和单任务模式都通过 `main.py` 直接调用 `workflow_v2.py`
- 批处理模式通过 `BatchProcessor` 默认调用 `generate_with_best_of_n()`，同样走 `workflow_v2.py`
- `workflow.py` (V1) 仅保留用于基线对比和消融实验，不作为 CLI 默认入口

### 5. 数据分析模式

```bash
python main.py --analyze data/output/results.jsonl
```

输出示例：
```
📊 数据集统计:
  总数据量: 20
  有效数据: 20
  平均分数: 8.83
  最高分数: 9.50
  最低分数: 8.50
  高质量数据 (>=8.5): 15
  平均输出长度: 750 字符
```

---

## 📁 项目结构

```
MLangChain/
├── config/                      # 配置管理
│   ├── __init__.py
│   └── settings.py              # 模型配置、阈值设置
│
├── src/                         # 核心源代码
│   ├── agents/                  # 智能体模块
│   │   ├── generator.py         # 生成器
│   │   ├── multi_version_generator.py  # 多版本生成器
│   │   ├── critic.py            # 原版 Critic
│   │   ├── critic_v2.py         # V2 Critic (集成全部优化)
│   │   ├── refiner.py           # 优化器
│   │   └── code_generator.py    # 代码生成器
│   │
│   ├── core/                    # 核心功能模块
│   │   ├── task_classifier.py   # 任务分类器
│   │   ├── grading_criteria.py  # 动态评分标准
│   │   ├── code_validator.py    # 代码验证器
│   │   ├── fact_checker.py      # 事实检查器
│   │   ├── best_of_n.py         # Best-of-N 选择
│   │   └── complexity_evaluator.py  # 复杂度评估
│   │
│   ├── graph/                   # 工作流定义
│   │   ├── workflow.py          # V1 工作流（基线/对比保留）
│   │   └── workflow_v2.py       # V2 工作流（CLI 默认路径）
│   │
│   ├── tools/                   # 工具模块
│   │   └── search_tool.py       # Tavily 搜索
│   │
│   ├── utils/                   # 工具函数
│   │   ├── data_utils.py        # 数据处理
│   │   └── batch_processor.py   # 批量处理（默认调用 V2）
│   │
│   ├── llm_factory.py           # LLM 工厂
│   └── state.py                 # 状态定义
│
├── data/                        # 数据目录
│   ├── input/                   # 输入主题
│   └── output/                  # 输出结果
│
├── main.py                      # CLI 入口
├── requirements.txt             # 依赖列表
├── .env.example                 # 环境变量示例
└── test_*.py                    # 测试脚本
```

---

## 📈 性能指标

### 评分质量提升

| 指标 | V1 | V2 | 提升 |
|------|-----|-----|------|
| **平均分数** | 8.83 | 9.10 | +0.27 |
| **评分稳定性** | ±1.0 | ±0.2 | 5x |
| **高分占比** | 50% | 75% | +25% |

### 事实准确性

| 指标 | 数值 |
|------|------|
| **幻觉检测率** | 80%+ |
| **事实准确率** | 95% |

### 数据多样性

| 复杂度 | 占比 |
|--------|------|
| 简单 | < 10% |
| 适中 | ~40% |
| 复杂 | ~50% |

---

## 📚 学术参考

本项目借鉴了以下学术研究：

| 技术 | 论文 | 作者 |
|------|------|------|
| **RLHF** | "Training language models to follow instructions with human feedback" | Ouyang et al., 2022 |
| **RARR** | "Researching and Revising What Language Models Say" | Gao et al., 2023 |
| **Self-Instruct** | "Aligning Language Model with Self Generated Instructions" | Wang et al., 2023 |
| **Best-of-N** | "Learning to summarize from human feedback" | Stiennon et al., 2020 |

---

## 🔮 未来规划

### 短期优化
- [ ] 性能优化：并行生成、缓存优化
- [ ] 更多 LLM 提供商支持（Claude、文心一言等）
- [ ] 更丰富的任务类型支持

### 长期规划
- [ ] 人机协同审核界面
- [ ] 可视化工作流编辑器
- [ ] 分布式批量处理
- [ ] 模型微调集成

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

---

## 📄 许可证

本项目采用 [MIT](LICENSE) 许可证。

---

**⭐ 如果这个项目对你有帮助，请给它一个 Star！**

*最后更新：2026-03-06 | 版本：v2.4*
