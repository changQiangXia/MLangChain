"""
Grading Criteria
动态评分标准 - 基于任务类型的差异化评分

解决"字数暴力"问题，为不同类型任务应用不同评分维度。
"""

from typing import Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum

from src.core.task_classifier import TaskType


class Dimension(BaseModel):
    """评分维度定义"""
    name: str = Field(description="维度名称")
    description: str = Field(description="维度描述")
    max_score: float = Field(description="最高分", ge=0)
    weight: float = Field(description="权重", ge=0, le=1)


class TaskCriteria(BaseModel):
    """任务类型的评分标准"""
    task_type: TaskType
    description: str
    min_length: Optional[int] = Field(default=None, description="最小字数要求（None表示不检查）")
    dimensions: List[Dimension]
    special_checks: List[str] = Field(default_factory=list, description="特殊检查项")
    length_check_enabled: bool = Field(default=True, description="是否启用字数检查")


class GradingCriteria:
    """
    动态评分标准管理器
    
    根据任务类型返回不同的评分维度和标准。
    """
    
    # 预定义的评分标准
    CRITERIA_MAP: Dict[TaskType, TaskCriteria] = {
        TaskType.CODE: TaskCriteria(
            task_type=TaskType.CODE,
            description="编程/代码生成任务",
            min_length=None,  # 代码类不看字数！
            length_check_enabled=False,
            dimensions=[
                Dimension(name="correctness", description="代码正确性（能否正确运行）", max_score=3.0, weight=0.3),
                Dimension(name="readability", description="代码可读性（命名、结构）", max_score=2.0, weight=0.2),
                Dimension(name="efficiency", description="代码效率（时间/空间复杂度）", max_score=2.0, weight=0.2),
                Dimension(name="completeness", description="功能完整性（边界处理）", max_score=2.0, weight=0.2),
                Dimension(name="documentation", description="文档/注释", max_score=1.0, weight=0.1),
            ],
            special_checks=["syntax_valid", "can_execute", "has_comments", "has_test_cases"]
        ),
        
        TaskType.REASONING: TaskCriteria(
            task_type=TaskType.REASONING,
            description="逻辑推理/数学证明任务",
            min_length=100,  # 推理类需要一定长度展示思维链
            length_check_enabled=True,
            dimensions=[
                Dimension(name="logic_chain", description="逻辑链条完整性（CoT）", max_score=3.0, weight=0.3),
                Dimension(name="correctness", description="结论正确性", max_score=2.5, weight=0.25),
                Dimension(name="clarity", description="逻辑清晰度", max_score=2.0, weight=0.2),
                Dimension(name="depth", description="推理深度", max_score=1.5, weight=0.15),
                Dimension(name="expression", description="表达质量", max_score=1.0, weight=0.1),
            ],
            special_checks=["has_cot", "logic_valid"]
        ),
        
        TaskType.EXPLANATION: TaskCriteria(
            task_type=TaskType.EXPLANATION,
            description="概念解释/知识问答任务",
            min_length=200,  # 解释类需要字数保证完整性
            length_check_enabled=True,
            dimensions=[
                Dimension(name="accuracy", description="内容准确性", max_score=2.5, weight=0.25),
                Dimension(name="completeness", description="内容完整性", max_score=2.5, weight=0.25),
                Dimension(name="practicality", description="实用性（具体例子）", max_score=2.0, weight=0.2),
                Dimension(name="depth", description="深度（原理/机制）", max_score=2.0, weight=0.2),
                Dimension(name="expression", description="表达质量", max_score=1.0, weight=0.1),
            ],
            special_checks=["has_examples", "has_principle"]
        ),
        
        TaskType.CREATIVE: TaskCriteria(
            task_type=TaskType.CREATIVE,
            description="创意写作/开放式问答任务",
            min_length=300,  # 创意类需要丰富度
            length_check_enabled=True,
            dimensions=[
                Dimension(name="creativity", description="创意/原创性", max_score=3.0, weight=0.3),
                Dimension(name="coherence", description="连贯性", max_score=2.5, weight=0.25),
                Dimension(name="engagement", description="吸引力/趣味性", max_score=2.0, weight=0.2),
                Dimension(name="richness", description="内容丰富度", max_score=1.5, weight=0.15),
                Dimension(name="expression", description="语言表达", max_score=1.0, weight=0.1),
            ],
            special_checks=["originality", "emotional_impact"]
        ),
        
        TaskType.CHITCHAT: TaskCriteria(
            task_type=TaskType.CHITCHAT,
            description="闲聊/对话任务",
            min_length=20,  # 闲聊类字数要求低
            length_check_enabled=True,
            dimensions=[
                Dimension(name="appropriateness", description="回答得体性", max_score=3.0, weight=0.3),
                Dimension(name="helpfulness", description="有帮助性", max_score=3.0, weight=0.3),
                Dimension(name="naturalness", description="自然度", max_score=2.0, weight=0.2),
                Dimension(name="conciseness", description="简洁性", max_score=1.0, weight=0.1),
                Dimension(name="friendliness", description="友好度", max_score=1.0, weight=0.1),
            ],
            special_checks=[]
        ),
        
        TaskType.UNKNOWN: TaskCriteria(
            task_type=TaskType.UNKNOWN,
            description="未知类型（使用通用标准）",
            min_length=150,
            length_check_enabled=True,
            dimensions=[
                Dimension(name="accuracy", description="准确性", max_score=2.5, weight=0.25),
                Dimension(name="completeness", description="完整性", max_score=2.5, weight=0.25),
                Dimension(name="practicality", description="实用性", max_score=2.0, weight=0.2),
                Dimension(name="depth", description="深度", max_score=2.0, weight=0.2),
                Dimension(name="expression", description="表达", max_score=1.0, weight=0.1),
            ],
            special_checks=[]
        ),
    }
    
    @classmethod
    def get_criteria(cls, task_type: TaskType) -> TaskCriteria:
        """
        获取指定任务类型的评分标准
        
        Args:
            task_type: 任务类型
            
        Returns:
            TaskCriteria: 评分标准
        """
        return cls.CRITERIA_MAP.get(task_type, cls.CRITERIA_MAP[TaskType.UNKNOWN])
    
    @classmethod
    def check_length(cls, task_type: TaskType, output_length: int) -> tuple[bool, float]:
        """
        检查字数是否符合要求
        
        Args:
            task_type: 任务类型
            output_length: 输出字数
            
        Returns:
            (是否通过, 扣分)
        """
        criteria = cls.get_criteria(task_type)
        
        # 如果禁用字数检查，直接通过
        if not criteria.length_check_enabled:
            return True, 0.0
        
        # 如果没有最小字数要求，直接通过
        if criteria.min_length is None:
            return True, 0.0
        
        # 检查字数
        if output_length >= criteria.min_length:
            return True, 0.0
        
        # 计算扣分（低于要求越多，扣分越多）
        ratio = output_length / criteria.min_length
        if ratio >= 0.75:  # 达到75%，扣少量分
            deduction = 0.5
        elif ratio >= 0.5:  # 达到50%，扣中等分
            deduction = 1.0
        else:  # 低于50%，扣大量分
            deduction = 2.0
        
        return False, deduction
    
    @classmethod
    def get_total_score(cls, task_type: TaskType) -> float:
        """获取指定类型的总分"""
        criteria = cls.get_criteria(task_type)
        return sum(d.max_score for d in criteria.dimensions)
    
    @classmethod
    def generate_prompt_section(cls, task_type: TaskType) -> str:
        """
        生成 Prompt 中的评分标准部分
        
        Args:
            task_type: 任务类型
            
        Returns:
            str: Prompt 文本
        """
        criteria = cls.get_criteria(task_type)
        
        lines = [
            f"## 任务类型: {criteria.description}",
            "",
            "### 评分维度",
        ]
        
        for dim in criteria.dimensions:
            lines.append(f"- {dim.name} (0-{dim.max_score}分): {dim.description} [权重{dim.weight:.0%}]")
        
        if criteria.special_checks:
            lines.extend([
                "",
                "### 特殊检查项",
                "必须检查以下项目："
            ])
            for check in criteria.special_checks:
                lines.append(f"- {check}")
        
        if criteria.length_check_enabled and criteria.min_length:
            lines.extend([
                "",
                f"### 字数要求: 最少 {criteria.min_length} 字",
                f"（当前类型 {'启用' if criteria.length_check_enabled else '禁用'} 字数检查）"
            ])
        else:
            lines.extend([
                "",
                "### 字数要求: 无（本类型不看字数，看内容质量）"
            ])
        
        return "\n".join(lines)


# 便捷函数
def get_grading_criteria(task_type: TaskType) -> TaskCriteria:
    """获取评分标准"""
    return GradingCriteria.get_criteria(task_type)
