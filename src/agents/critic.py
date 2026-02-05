"""
Critic Agent (The Judge) - 改进版
多维度评分系统，更严格、更有区分度
"""

import json
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.state import AlpacaData, CritiqueResult, GraphState
from src.llm_factory import create_llm


class CriticAgent:
    """
    评审者 Agent - 多维度严格评分
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.critic_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.1)
    
    def _get_critique_prompt(self) -> ChatPromptTemplate:
        """
        获取评审任务的 Prompt 模板 - 多维度详细评分
        """
        system_template = """你是一个极其严格的数据质量评审专家。
你的任务是对生成的指令微调数据进行多维度评估和打分。

## 评分维度（每个维度必须独立评分，不能所有维度一样）

### 1. 指令质量 (0-2.5分)
- 2.5分：指令非常清晰、具体、有深度，要求明确
- 2.0分：指令清晰，但可以更具体
- 1.5分：指令基本清楚，有些模糊
- 1.0分：指令较模糊，要求不明确
- 0.5分：指令很差，难以理解

### 2. 回答准确性 (0-2.5分)
- 2.5分：内容100%准确，专业术语正确，无错误
- 2.0分：内容基本准确，有小瑕疵
- 1.5分：有轻微错误或不准确之处
- 1.0分：有明显错误
- 0.5分：有严重错误

### 3. 回答完整性和深度 (0-2.5分)
- 2.5分：非常全面、深入，超过200字，覆盖所有关键点
- 2.0分：比较完整，但缺少一些细节
- 1.5分：基本完整，但深度不够
- 1.0分：不完整，缺少重要内容
- 0.5分：非常简略，敷衍

### 4. 实用性 (0-1.5分)
- 1.5分：例子非常具体（包含公司/产品/量化效果），非常实用
- 1.0分：例子较具体，但缺少公司/量化信息
- 0.5分：例子泛泛而谈，如"用于图像识别"

### 例子质量检查（扣分项）
- ❌ "用于图像识别"、"在自然语言处理中" → 扣0.5分
- ❌ 缺少具体产品/公司名称 → 扣0.3分
- ❌ 缺少量化效果（准确率/速度等） → 扣0.3分
- ✅ "ResNet-50在ImageNet达到76.2%准确率" → 不扣分

### 5. 表达质量 (0-1.0分)
- 1.0分：语言流畅，逻辑清晰，易于理解
- 0.5分：表达一般，有些地方不够清晰

## 严格扣分规则（发现以下问题必须扣分）

- ❌ 回答少于150字：扣1.5分
- ❌ 回答少于100字：扣2.5分
- ❌ 缺少具体例子：扣0.5-1.0分
- ❌ 有事实错误：扣1.0-2.0分
- ❌ 内容空洞/泛泛而谈：扣1.0分
- ❌ 结构混乱：扣0.5分
- ❌ 与指令不匹配：扣1.0-2.0分

## 总分计算
总分 = 维度1 + 维度2 + 维度3 + 维度4 + 维度5 - 扣分

## 评分标准
- 9.0-10分：卓越，可以发布
- 8.0-8.9分：良好，小瑕疵可接受
- 7.0-7.9分：一般，有明显问题
- 6.0-6.9分：较差，需要大幅修改
- <6分：很差，需要重写

## 重要规则
1. 大多数数据应该在 6.0-8.5 分之间
2. 9分以上必须是非常优秀的数据
3. 必须根据实际质量评分，不能给"人情分"
4. 如果数据平庸，必须给 7 分以下
"""

        human_template = """请严格评审以下指令数据：

主题：{task_description}

生成的数据：
- 指令: {instruction}
- 输入: {input_text}
- 输出: {output}

输出字数：{output_length}

请进行多维度评分，以 JSON 格式输出：
```json
{{
  "dimension_scores": {{
    "instruction_quality": <0-2.5>,
    "accuracy": <0-2.5>,
    "completeness": <0-2.5>,
    "practicality": <0-1.5>,
    "expression": <0-1.0>
  }},
  "deductions": <扣分数值>,
  "score": <总分0-10>,
  "feedback": "详细的评审意见，指出具体问题",
  "issues": ["问题1", "问题2", "问题3"]
}}
```

记住：要严格评分，平庸的数据给 7 分以下！
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def critique(
        self, 
        task_description: str,
        draft: AlpacaData
    ) -> CritiqueResult:
        """
        对生成的数据进行多维度评审
        """
        prompt = self._get_critique_prompt()
        chain = prompt | self.llm
        
        output_length = len(draft.output) if draft.output else 0
        
        response = chain.invoke({
            "task_description": task_description,
            "instruction": draft.instruction,
            "input_text": draft.input,
            "output": draft.output,
            "output_length": output_length
        })
        
        # 解析 JSON 输出
        try:
            content = response.content
            # 清理控制字符
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            
            # 尝试从 Markdown 代码块中提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            
            # 验证分数合理性
            score = float(data.get("score", 0))
            
            # 如果 LLM 给分太宽松，进行校准
            if score > 8.5 and output_length < 200:
                score = min(score, 7.5)  # 短内容不能超过 7.5
                data["score"] = score
                data["feedback"] = "【自动校准】" + data.get("feedback", "")
            
            return CritiqueResult(
                score=score,
                feedback=data.get("feedback", ""),
                issues=data.get("issues", [])
            )
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Critic Error] JSON 解析失败: {e}")
            return CritiqueResult(
                score=0.0,
                feedback=f"评审解析失败，错误: {e}",
                issues=["解析错误"]
            )
    
    def __call__(self, state: GraphState) -> GraphState:
        """作为 LangGraph Node 调用"""
        print(f"\n[Critic] 正在严格评审数据质量...")
        
        task = state["task_description"]
        draft = state.get("current_draft")
        
        if not draft:
            state["error_msg"] = "没有可评审的数据草稿"
            state["quality_score"] = 0.0
            return state
        
        # 执行评审
        result = self.critique(task, draft)
        
        # 更新状态
        state["critique_feedback"] = result
        state["quality_score"] = result.score
        
        print(f"[Critic] 评分: {result.score}/10")
        print(f"[Critic] 反馈: {result.feedback[:120]}...")
        
        return state
