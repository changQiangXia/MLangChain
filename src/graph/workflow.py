"""
Workflow Definition
定义 Nodes, Edges 和 Compiler
构建 LangGraph StateGraph
"""

from typing import Literal
from datetime import datetime
from langgraph.graph import StateGraph, END

from config.settings import settings
from src.state import GraphState, AlpacaData, CritiqueResult
from src.agents.generator import GeneratorAgent
from src.agents.critic import CriticAgent
from src.agents.critic_v2 import CriticV2Agent
from src.agents.refiner import RefinerAgent


def create_workflow() -> StateGraph:
    """创建并配置 LangGraph 工作流"""
    workflow = StateGraph(GraphState)
    
    # 初始化 Agents（使用 V2 Critic 支持动态类型感知）
    generator = GeneratorAgent()
    critic = CriticV2Agent()  # 使用新版 Critic
    refiner = RefinerAgent()
    
    # 添加 Nodes
    workflow.add_node("generator", generator)
    workflow.add_node("critic", critic)
    workflow.add_node("refiner", refiner)
    
    # 设置入口点
    workflow.set_entry_point("generator")
    
    # 添加边：Generator -> Critic
    workflow.add_edge("generator", "critic")
    
    # 添加条件边：Critic -> 根据评分决定下一步
    def decide_next_step(state: GraphState) -> Literal["refiner", "end"]:
        """根据评审结果决定下一步"""
        score = state.get("quality_score", 0)
        iteration = state.get("iteration_count", 0)
        max_iter = settings.max_iterations
        
        print(f"\n[Decision] 当前评分: {score}, 迭代次数: {iteration}/{max_iter}")
        
        # 如果评分达到阈值，结束流程
        if score >= settings.quality_threshold:
            print(f"[Decision] 质量达标 (≥{settings.quality_threshold})，流程结束")
            state["is_complete"] = True
            return "end"
        
        # 如果超过最大迭代次数，强制结束
        if iteration >= max_iter:
            print(f"[Decision] 达到最大迭代次数，强制结束")
            state["is_complete"] = True
            return "end"
        
        # 需要继续优化
        print(f"[Decision] 质量未达标，进入优化阶段")
        return "refiner"
    
    workflow.add_conditional_edges(
        "critic",
        decide_next_step,
        {
            "refiner": "refiner",
            "end": END
        }
    )
    
    # 添加边：Refiner -> Generator（形成循环）
    workflow.add_edge("refiner", "generator")
    
    return workflow


def compile_workflow(checkpointer=None):
    """编译工作流图"""
    workflow = create_workflow()
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=[],
        interrupt_after=[]
    )
    return app


def run_workflow(
    task_description: str,
    verbose: bool = True
) -> GraphState:
    """运行完整的工作流"""
    from src.state import initialize_state
    
    initial_state = initialize_state(task_description)
    app = compile_workflow()
    
    if verbose:
        print("=" * 50)
        print(f"开始处理任务: {task_description}")
        print("=" * 50)
    
    config = {"recursion_limit": settings.recursion_limit}
    
    # 使用 invoke 而不是 stream 来正确获取最终状态
    final_state = app.invoke(initial_state, config=config)
    
    if verbose:
        print("\n" + "=" * 50)
        print("工作流执行完成")
        print("=" * 50)
    
    return final_state


def generate_high_quality_data(task_description: str) -> dict:
    """快速生成高质量的指令数据"""
    final_state = run_workflow(task_description, verbose=True)
    
    draft = final_state.get("current_draft")
    critique = final_state.get("critique_feedback")
    metadata = final_state.get("metadata", {})
    
    # 更新元数据
    metadata.update({
        "end_time": datetime.now().isoformat(),
        "model_provider": settings.llm_provider,
        "generator_model": settings.default_model,
        "critic_model": settings.critic_model,
        "quality_threshold": settings.quality_threshold
    })
    
    # 确定成功状态（只要分数达标就算成功）
    score = final_state.get("quality_score", 0)
    success = score >= settings.quality_threshold
    
    return {
        "success": success,
        "score": final_state.get("quality_score", 0),
        "iterations": final_state.get("iteration_count", 0),
        "data": draft.to_dict() if draft else None,
        "feedback": critique.feedback if critique else None,
        "critique_details": critique.to_dict() if critique else None,
        "metadata": metadata,
        "error": final_state.get("error_msg")
    }
