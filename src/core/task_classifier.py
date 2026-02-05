"""
Task Classifier
任务类型分类器 - 实现动态类型感知 (Task-Aware Grading)

基于指令内容自动分类任务类型，为不同任务应用不同的评分标准。
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from src.llm_factory import create_llm


class TaskType(str, Enum):
    """任务类型枚举"""
    CODE = "code"                    # 编程/代码生成
    REASONING = "reasoning"          # 逻辑推理/数学证明
    EXPLANATION = "explanation"      # 概念解释/知识问答
    CREATIVE = "creative"            # 创意写作/开放式问答
    CHITCHAT = "chitchat"            # 闲聊/对话
    UNKNOWN = "unknown"              # 未知类型


class TaskClassification(BaseModel):
    """任务分类结果"""
    task_type: TaskType = Field(description="任务类型")
    confidence: float = Field(description="分类置信度", ge=0.0, le=1.0)
    reasoning: str = Field(description="分类理由")


class TaskClassifier:
    """
    任务分类器
    
    使用 LLM 或规则匹配对指令进行分类，支持：
    1. LLM 智能分类
    2. 关键词规则匹配
    3. 混合模式（规则 + LLM 确认）
    """
    
    # 关键词规则映射
    KEYWORD_RULES = {
        TaskType.CODE: [
            "代码", "编程", "python", "java", "c++", "javascript", "函数",
            "写一个", "实现", "算法", "class", "def", "代码示例",
            "write a", "implement", "algorithm", "function", "code"
        ],
        TaskType.REASONING: [
            "证明", "推导", "推理", "计算", "求解", "为什么", "分析",
            "逻辑", "步骤", "过程", "推导过程", "数学证明",
            "prove", "derive", "calculate", "solve", "why", "analyze"
        ],
        TaskType.EXPLANATION: [
            "解释", "什么是", "介绍", "概念", "原理", "机制",
            "说明", "阐述", "定义", "简述", "概述",
            "explain", "what is", "introduce", "concept", "principle"
        ],
        TaskType.CREATIVE: [
            "创作", "写作", "故事", "诗歌", "文章", "创意",
            "编写", "撰写", "构思", "设计",
            "create", "write", "story", "poem", "essay", "creative"
        ],
        TaskType.CHITCHAT: [
            "你好", "谢谢", "再见", "聊天", "对话", "问答",
            "help", "hi", "hello", "thanks", "chat"
        ]
    }
    
    def __init__(self, use_llm: bool = True, confidence_threshold: float = 0.7):
        """
        初始化分类器
        
        Args:
            use_llm: 是否使用 LLM 进行分类（否则仅用规则）
            confidence_threshold: LLM 分类置信度阈值，低于此值使用规则
        """
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold
        if use_llm:
            self.llm = create_llm(temperature=0.1)
    
    def classify(self, instruction: str) -> TaskClassification:
        """
        分类任务类型
        
        Args:
            instruction: 指令文本
            
        Returns:
            TaskClassification: 分类结果
        """
        # 1. 首先使用规则分类
        rule_result = self._classify_by_rules(instruction)
        
        # 2. 如果规则分类明确且不使用 LLM，直接返回
        if rule_result and not self.use_llm:
            return TaskClassification(
                task_type=rule_result,
                confidence=0.8,
                reasoning="基于关键词规则匹配"
            )
        
        # 3. 使用 LLM 分类
        if self.use_llm:
            llm_result = self._classify_by_llm(instruction)
            
            # 4. 如果 LLM 置信度高，使用 LLM 结果
            if llm_result.confidence >= self.confidence_threshold:
                return llm_result
            
            # 5. 如果 LLM 置信度低但有规则结果，使用规则
            if rule_result:
                return TaskClassification(
                    task_type=rule_result,
                    confidence=0.7,
                    reasoning=f"LLM置信度低({llm_result.confidence:.2f})，fallback到规则匹配"
                )
            
            # 6. 否则使用 LLM 结果（即使置信度低）
            return llm_result
        
        # 7. 无法分类
        return TaskClassification(
            task_type=TaskType.UNKNOWN,
            confidence=0.5,
            reasoning="无法匹配任何规则"
        )
    
    def _classify_by_rules(self, instruction: str) -> Optional[TaskType]:
        """基于关键词规则分类"""
        instruction_lower = instruction.lower()
        
        # 统计每个类型的匹配关键词数
        match_counts = {}
        for task_type, keywords in self.KEYWORD_RULES.items():
            count = sum(1 for kw in keywords if kw.lower() in instruction_lower)
            if count > 0:
                match_counts[task_type] = count
        
        # 返回匹配最多的类型
        if match_counts:
            return max(match_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _classify_by_llm(self, instruction: str) -> TaskClassification:
        """使用 LLM 进行分类"""
        prompt = f"""请分析以下指令的任务类型：

指令: "{instruction}"

可选类型：
- CODE: 编程/代码生成（要求写代码、实现算法、函数等）
- REASONING: 逻辑推理/数学证明（需要推导、计算、证明等）
- EXPLANATION: 概念解释/知识问答（解释概念、介绍原理等）
- CREATIVE: 创意写作/开放式问答（写作、创作、构思等）
- CHITCHAT: 闲聊/对话（问候、简单问答等）

请以 JSON 格式输出：
{{
  "task_type": "类型代码",
  "confidence": 0.0-1.0,
  "reasoning": "分类理由（简短说明）"
}}

只输出 JSON，不要其他内容。"""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            import re
            
            content = response.content
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            
            return TaskClassification(
                task_type=TaskType(data.get("task_type", "unknown")),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "LLM分类")
            )
            
        except Exception as e:
            # LLM 分类失败，返回未知
            return TaskClassification(
                task_type=TaskType.UNKNOWN,
                confidence=0.0,
                reasoning=f"LLM分类失败: {str(e)}"
            )


# 便捷函数
def classify_task(instruction: str) -> TaskType:
    """快速分类任务"""
    classifier = TaskClassifier()
    return classifier.classify(instruction).task_type
