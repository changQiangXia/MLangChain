"""
Graph State Definition
定义多智能体系统的全局状态管理
"""

from typing import TypedDict, Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class AlpacaData(BaseModel):
    """
    Alpaca 格式的指令微调数据
    """
    instruction: str = Field(description="指令/问题")
    input: str = Field(default="", description="上下文输入（可选）")
    output: str = Field(description="期望的输出/回答")
    
    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output
        }


class CritiqueResult(BaseModel):
    """
    Critic Agent 的评审结果 - 支持建设性CoT
    """
    score: float = Field(description="质量评分 (0-10)", ge=0, le=10)
    feedback: str = Field(description="具体的修改建议")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    improvement_guide: Optional[dict] = Field(default=None, description="结构化改进指南")
    
    def to_dict(self) -> dict:
        result = {
            "score": self.score,
            "feedback": self.feedback,
            "issues": self.issues
        }
        if self.improvement_guide:
            result["improvement_guide"] = self.improvement_guide
        return result


class GraphState(TypedDict, total=False):
    """
    LangGraph 全局状态定义
    """
    task_description: str
    search_results: Optional[List[dict]]
    current_draft: Optional[AlpacaData]
    critique_feedback: Optional[CritiqueResult]
    quality_score: float
    iteration_count: int
    is_complete: bool
    error_msg: Optional[str]
    # 新增元数据字段
    metadata: Optional[dict]  # 存储模型信息、时间戳等
    # 迭代历史追踪
    iteration_history: List[dict]  # 每次迭代的评分历史
    
    # === 模拟退火/回滚机制 (Strategy 1) ===
    best_draft_so_far: Optional[AlpacaData]  # 历史最佳版本
    best_score_so_far: float                 # 历史最佳分数
    retry_count: int                         # 连续重试次数（用于温度调整）
    current_temperature: float               # 当前温度（动态调整）
    needs_rollback: bool

    # Workflow V2 临时字段（需声明，否则 LangGraph 节点间会丢失）
    _generated_versions: List[dict]
    _best_version_index: int
    _best_of_n_score: float


def initialize_state(task_description: str) -> GraphState:
    """初始化一个新的 GraphState"""
    return GraphState(
        task_description=task_description,
        search_results=None,
        current_draft=None,
        critique_feedback=None,
        quality_score=0.0,
        iteration_count=0,
        is_complete=False,
        error_msg=None,
        metadata={
            "start_time": datetime.now().isoformat(),
            "task": task_description
        },
        iteration_history=[],  # 初始化空历史
        # 回滚机制初始化
        best_draft_so_far=None,
        best_score_so_far=0.0,
        retry_count=0,
        current_temperature=0.3,  # 默认温度
        needs_rollback=False,
        _generated_versions=[],
        _best_version_index=0,
        _best_of_n_score=0.0
    )
