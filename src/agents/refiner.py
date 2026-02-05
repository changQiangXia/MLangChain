"""
Refiner Agent
根据 Critic 的反馈重写和优化内容
"""

import json
from typing import Optional, List, Dict
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.state import AlpacaData, CritiqueResult, GraphState
from src.llm_factory import create_llm
from src.utils.json_utils import safe_json_loads, sanitize_for_json


class RefinerAgent:
    """
    优化者 Agent：根据评审反馈改进生成的数据
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化优化器
        
        Args:
            model_name: 使用的模型名称
        """
        self.model_name = model_name or settings.default_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.5)
    
    def _get_refinement_prompt(self) -> ChatPromptTemplate:
        """
        获取优化任务的 Prompt 模板
        """
        system_template = """你是一个专业的数据优化专家。
你的任务是根据评审反馈，**彻底重写**指令数据，使其达到卓越质量。

## 优化原则
1. **100%解决所有问题**：评审指出的每个问题都必须解决
2. **大幅扩展内容**：从200字扩展到400+字
3. **增加具体例子**：至少3个实际应用案例
4. **深化内容**：解释原理、机制，不只是概念
5. **改进结构**：使用清晰的段落和列表

## 质量标准
- 回答 ≥400字
- 包含定义、原理、应用案例
- 专业、准确、深入
- 易于理解和实用

警告：敷衍的修改会被再次打低分！
"""

        human_template = """请彻底重写以下数据：

原始主题：{task_description}

当前数据（评分 {score}/10，严重不足）：
- 指令: {instruction}
- 输出: {output}

评审反馈（必须全部解决）：
{feedback}

问题列表：
{issues}

## 重写要求（严格按照评审反馈修改）
1. **解决所有问题**：必须解决评审中提到的每个问题
2. **指令进化**：如果反馈中提到"指令过于简单"，必须增加约束条件
   - 添加：时间/空间复杂度要求
   - 添加：多步推理要求
   - 添加：对比分析要求
   - 添加：具体应用场景
3. **例子具体化**：
   - ❌ 错误："用于图像识别"
   - ✅ 正确："ResNet-50在ImageNet 2012图像分类任务中达到76.2%的top-1准确率"
   - 必须包含：**技术名称 + 具体场景 + 量化效果**
4. **增加深度**：
   - 原理部分增加数学公式或算法步骤
   - 解释"为什么"而不仅是"是什么"
5. **回答长度 ≥500字**

## 指令进化模板（如果原指令太简单）
- "{instruction}，要求时间复杂度为O(n)"
- "{instruction}，并对比至少两种方法的优缺点"
- "{instruction}，给出具体的推导过程"
- "{instruction}，并提供实际应用案例"

## 输出格式（严格JSON）
```json
{{
  "instruction": "进化后的指令（增加约束和要求）",
  "input": "",
  "output": "## 概念定义\\n...\\n\\n## 核心原理\\n数学公式/算法步骤...\\n\\n## 具体应用案例\\n1. **[公司/产品]**：[具体场景]，[量化效果]\\n2. **[公司/产品]**：[具体场景]，[量化效果]\\n3. **[公司/产品]**：[具体场景]，[量化效果]\\n\\n## 深入分析\\n优缺点、未来趋势..."
}}
```

记住：目标是**9分以上**！指令要**有挑战性但可完成**！
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def refine(
        self, 
        task_description: str,
        draft: AlpacaData,
        critique: CritiqueResult,
        iteration_history: Optional[List[Dict]] = None
    ) -> AlpacaData:
        """
        根据评审反馈优化数据
        
        Args:
            task_description: 原始任务描述
            draft: 当前数据草稿
            critique: 评审结果
            iteration_history: 迭代历史（用于追踪进步）
            
        Returns:
            优化后的 AlpacaData
        """
        # 构建包含历史信息的 Prompt
        history_context = ""
        if iteration_history and len(iteration_history) > 1:
            history_context = "\n## 历史评分（必须提升！）\n"
            for i, record in enumerate(iteration_history[-3:], 1):  # 最近3次
                history_context += f"迭代 {record.get('iteration', i)}: {record.get('score', 0)}/10\n"
            
            prev_score = iteration_history[-2].get('score', 0) if len(iteration_history) >= 2 else 0
            current_score = critique.score
            
            if current_score <= prev_score:
                history_context += f"\n[警告] 评分没有提升（{prev_score} -> {current_score}）"
                history_context += "\n必须采取更激进的改进措施！\n"
        
        prompt = self._get_refinement_prompt_with_history(history_context)
        chain = prompt | self.llm
        
        # 构建改进指南文本
        improvement_guide_text = "未提供具体改进指南"
        if critique.improvement_guide:
            guide = critique.improvement_guide
            parts = []
            if "keep" in guide:
                parts.append(f"【保留部分】\n{guide['keep']}")
            if "fix" in guide and guide["fix"]:
                parts.append("【需要修复】")
                for i, fix in enumerate(guide["fix"], 1):
                    parts.append(f"{i}. 位置: {fix.get('location', '未知')}")
                    parts.append(f"   问题: {fix.get('problem', '未知')}")
                    parts.append(f"   建议: {fix.get('suggestion', '无')}")
            if "add" in guide:
                parts.append(f"【建议新增】\n{guide['add']}")
            improvement_guide_text = "\n".join(parts)
        
        # 带重试的 API 调用
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                response = chain.invoke({
                    "task_description": task_description,
                    "instruction": draft.instruction,
                    "input_text": draft.input,
                    "output": draft.output,
                    "score": critique.score,
                    "feedback": critique.feedback,
                    "issues": json.dumps(critique.issues, ensure_ascii=False),
                    "history_context": history_context,
                    "improvement_guide": improvement_guide_text
                })
                break  # 成功则跳出循环
            except Exception as e:
                if "timeout" in str(e).lower() or "read operation" in str(e).lower():
                    print(f"[Refiner] API 超时，重试 {attempt + 1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        print(f"[Refiner] 重试次数耗尽，返回原始数据")
                        return draft
                    import time
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise  # 非超时错误直接抛出
        
        # 解析 JSON 输出
        try:
            content = response.content
            
            # 使用安全的 JSON 解析
            data = safe_json_loads(content, default=None)
            
            if data is None:
                raise ValueError("JSON 解析失败")
            
            # 清理 output 字段
            if "output" in data and data["output"]:
                data["output"] = sanitize_for_json(data["output"])
            
            return AlpacaData(**data)
            
        except Exception as e:
            print(f"[Refiner Error] JSON 解析失败: {e}")
            # 如果解析失败，返回原始数据
            return draft
    
    def _get_refinement_prompt_with_history(self, history_context: str) -> ChatPromptTemplate:
        """获取带历史信息的优化 Prompt - 精准手术版本 (Strategy 2)"""
        system_template = """你是一个精准的数据优化专家，像外科医生一样做"微创手术"。

## 核心原则：保留优质，精准修复
❌ 不要：推倒重来，删掉原本好的内容
✅ 要：像医生一样，只修改病灶，保留健康组织

## 手术指南
1. **诊断**：仔细阅读 Critic 的改进指南 (improvement_guide)
2. **保留**：明确标记为"keep"的优质内容必须保留
3. **修复**：只修改"fix"中指出的具体问题位置
4. **新增**：在"add"建议的位置添加内容
5. **评分目标**：新版本评分必须 > {score}

## 质量标准
- 回答 ≥400字
- 保留原有优质结构和内容
- 只修改被指出的问题部分
- 专业、准确、深入

""" + history_context + """

警告：不要"为了改而改"！如果某部分已经是好的，保留它！
"""

        human_template = """请根据评审反馈进行精准优化：

原始主题：{task_description}

当前数据（评分 {score}/10）：
- 指令: {instruction}
- 输出: {output}

评审反馈：
{feedback}

问题列表：
{issues}

改进指南：
{improvement_guide}

## 精准修改要求（Strategy 2）
1. **保留优点**：如果改进指南中提到某些部分是好的，**不要修改**
2. **精准定位**：只在"fix"指出的位置进行修改
3. **具体实施**：按照"suggestion"中的建议修改
4. **增量改进**：在"add"建议的位置添加内容
5. **禁止过度修改**：未指出的问题不要动

## 修改示例
❌ 错误（过度修改）：
- 原文："Transformer是2017年提出的..."
- 修改："BERT是2018年Google推出的..."（完全替换主题）

✅ 正确（精准修改）：
- 问题："缺少具体年份"
- 修改："Transformer是**2017年**Google提出的..."（只添加年份）

## 输出格式（JSON）
```json
{{
  "instruction": "优化后的指令（可能增加约束）",
  "input": "",
  "output": "优化后的回答（保留优质部分，修复问题部分）",
  "modifications": [
    "修改1: 在第X段添加了...",
    "修改2: 修复了...",
    "修改3: 保留了...（因为Critic说这是好的）"
  ]
}}
```

记住：目标是**精准提升**，不是**推倒重来**！
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        作为 LangGraph Node 调用
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print(f"\n[Refiner] 正在根据反馈优化数据...")
        
        task = state["task_description"]
        draft = state.get("current_draft")
        critique = state.get("critique_feedback")
        
        if not draft or not critique:
            state["error_msg"] = "缺少必要的数据进行优化"
            return state
        
        # 获取历史评分信息
        iteration_history = state.get("iteration_history", [])
        
        # 执行优化（传入历史评分）
        refined_draft = self.refine(task, draft, critique, iteration_history)
        
        # 更新状态
        state["current_draft"] = refined_draft
        
        print(f"[Refiner] 优化完成: {refined_draft.instruction[:50]}...")
        
        return state


# 便捷函数
def refine_data(
    task_description: str,
    draft: AlpacaData,
    critique: CritiqueResult
) -> AlpacaData:
    """
    快速优化数据，无需实例化类
    """
    agent = RefinerAgent()
    return agent.refine(task_description, draft, critique)
