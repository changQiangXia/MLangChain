"""
Workflow V2 - 集成 Best-of-N + 模拟退火回滚机制

核心改进：
1. Generator 生成 N 个版本（N=2 或 3）
2. Selector 使用 Pairwise Comparison 选择最佳
3. **新增**: 模拟退火回滚机制 - 防止分数下降
4. **新增**: 动态温度调整 - 帮助跳出局部最优
"""

from typing import Literal, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END

from config.settings import settings
from src.state import GraphState, initialize_state
from src.agents.multi_version_generator import MultiVersionGeneratorAgent
from src.agents.critic_v2 import CriticV2Agent
from src.agents.refiner import RefinerAgent
from src.core.best_of_n import BestOfNSelector


def create_workflow_v2(
    use_best_of_n: bool = True,
    n_versions: int = 2
) -> StateGraph:
    """
    创建 Workflow V2（集成 Best-of-N + 回滚机制）
    
    Args:
        use_best_of_n: 是否使用 Best-of-N
        n_versions: 生成版本数（2 或 3）
    
    Returns:
        StateGraph: 工作流图
    """
    workflow = StateGraph(GraphState)
    
    # 初始化 Agents
    if use_best_of_n:
        generator = MultiVersionGeneratorAgent()
    else:
        from src.agents.generator import GeneratorAgent
        generator = GeneratorAgent()
    
    critic = CriticV2Agent()
    refiner = RefinerAgent()
    
    if use_best_of_n:
        best_of_n_selector = BestOfNSelector(n=n_versions)
    
    # 添加 Nodes
    workflow.add_node("generator", _create_generator_node(generator, use_best_of_n, n_versions))
    workflow.add_node("selector", _create_selector_node(best_of_n_selector) if use_best_of_n else lambda x: x)
    workflow.add_node("critic", critic)
    workflow.add_node("refiner", _create_refiner_node(refiner))
    workflow.add_node("rollback_check", _create_rollback_check_node())  # 新增：回滚检查节点
    
    # 设置入口点
    workflow.set_entry_point("generator")
    
    # 设置边
    if use_best_of_n:
        workflow.add_edge("generator", "selector")
        workflow.add_edge("selector", "critic")
    else:
        workflow.add_edge("generator", "critic")
    
    # 条件边：Critic -> Rollback Check 或 End
    workflow.add_conditional_edges(
        "critic",
        _decide_after_critic,
        {
            "rollback_check": "rollback_check",
            "end": END
        }
    )
    
    # 条件边：Rollback Check -> Refiner 或 Generator
    workflow.add_conditional_edges(
        "rollback_check",
        _decide_after_rollback,
        {
            "refiner": "refiner",
            "generator": "generator",  # 需要重新生成
            "end": END
        }
    )
    
    # Refiner -> Generator（形成循环）
    workflow.add_edge("refiner", "generator")
    
    return workflow


def _decide_after_critic(state: GraphState) -> Literal["rollback_check", "end"]:
    """Critic 后决策：是否达标或继续"""
    score = state.get("quality_score", 0)
    iteration = state.get("iteration_count", 0)
    max_iter = settings.max_iterations
    
    print(f"\n[Decision-Critic] 当前评分: {score:.1f}, 迭代: {iteration}/{max_iter}")
    
    # 质量达标
    if score >= settings.quality_threshold:
        print(f"[Decision-Critic] 质量达标 (≥{settings.quality_threshold})，流程结束")
        state["is_complete"] = True
        return "end"
    
    # 达到最大迭代次数
    if iteration >= max_iter:
        print(f"[Decision-Critic] 达到最大迭代次数，强制结束")
        state["is_complete"] = True
        return "end"
    
    # 进入回滚检查
    return "rollback_check"


def _create_rollback_check_node():
    """创建回滚检查节点 (Strategy 1: 模拟退火机制)"""
    def rollback_check_node(state: GraphState) -> GraphState:
        print("\n[RollbackCheck] 检查是否需要回滚...")
        
        current_score = state.get("quality_score", 0)
        best_score = state.get("best_score_so_far", 0)
        best_draft = state.get("best_draft_so_far")
        retry_count = state.get("retry_count", 0)
        
        # 情况1: 当前分数是最佳分数，正常继续
        if current_score >= best_score - 0.1:  # 允许0.1的误差
            print(f"[RollbackCheck] 分数正常: {current_score:.1f} (最佳: {best_score:.1f})")
            state["needs_rollback"] = False
            return state
        
        # 情况2: 分数下降超过阈值，执行回滚
        if current_score < best_score - 0.5:
            print(f"[RollbackCheck] [警告] 分数下降: {best_score:.1f} -> {current_score:.1f}")
            
            if best_draft and retry_count < 3:
                # 回滚到最佳版本
                print(f"[RollbackCheck] [回滚] 回滚到最佳版本 (重试{retry_count}/3)")
                state["current_draft"] = best_draft
                state["quality_score"] = best_score
                state["needs_rollback"] = True
            else:
                # 重试次数过多，接受当前结果
                print(f"[RollbackCheck] [警告] 重试次数过多({retry_count})，强制结束")
                state["is_complete"] = True
        
        return state
    
    return rollback_check_node


def _decide_after_rollback(state: GraphState) -> Literal["refiner", "generator", "end"]:
    """回滚检查后决策"""
    if state.get("is_complete"):
        return "end"
    
    if state.get("needs_rollback"):
        # 回滚后需要重新生成（使用更高温度）
        print("[Decision-Rollback] 回滚完成，使用新策略重新生成")
        return "generator"
    
    # 正常进入优化阶段
    return "refiner"


def _create_refiner_node(refiner: RefinerAgent):
    """创建 Refiner Node - 支持动态温度"""
    def refiner_node(state: GraphState) -> GraphState:
        # 获取当前温度
        temperature = state.get("current_temperature", 0.3)
        
        # 如果温度变化较大，重新初始化 LLM
        if abs(temperature - 0.3) > 0.1:
            from src.llm_factory import create_llm
            try:
                # 限制温度在有效范围内 (0.0 - 1.0)
                safe_temp = max(0.0, min(1.0, temperature))
                refiner.llm = create_llm(
                    model_name=refiner.model_name, 
                    temperature=safe_temp
                )
                print(f"[Refiner] 温度调整为: {safe_temp:.2f}")
            except Exception as e:
                print(f"[Refiner] 温度调整失败: {e}，使用默认温度")
        
        # 执行优化
        return refiner(state)
    
    return refiner_node


def _create_generator_node(generator, use_best_of_n: bool, n_versions: int):
    """创建 Generator Node"""
    def generator_node(state: GraphState) -> GraphState:
        print(f"\n[Generator] 正在生成指令数据... (迭代: {state['iteration_count']})")
        
        task = state["task_description"]
        search_results = state.get("search_results")
        
        if use_best_of_n and isinstance(generator, MultiVersionGeneratorAgent):
            # 生成多个版本
            versions = generator.generate_multiple(task, n=n_versions, search_results=search_results)
            state["_generated_versions"] = [v.to_dict() if hasattr(v, 'to_dict') else v for v in versions]
            state["current_draft"] = versions[0]
            print(f"[Generator] 生成 {len(versions)} 个版本，等待选择...")
        else:
            draft = generator.generate(task, search_results)
            state["current_draft"] = draft
            print(f"[Generator] 生成完成: {draft.instruction[:50]}...")
        
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        return state
    
    return generator_node


def _create_selector_node(selector: BestOfNSelector):
    """创建 Best-of-N Selector Node"""
    def selector_node(state: GraphState) -> GraphState:
        print("\n[BestOfN] 从多个版本中选择最佳...")
        
        task = state["task_description"]
        versions_data = state.get("_generated_versions", [])
        
        if not versions_data:
            print("[BestOfN] 没有多个版本，跳过选择")
            return state
        
        from src.state import AlpacaData
        versions = [AlpacaData(**v) for v in versions_data]
        result = selector.select(task, versions)
        
        state["current_draft"] = result.best_version
        state["_best_version_index"] = result.best_index
        state["_best_of_n_score"] = result.final_score
        
        print(f"[BestOfN] 选择版本 {result.best_index + 1}")
        print(f"[BestOfN] 预选分数: {result.final_score}/10")
        
        return state
    
    return selector_node


def run_workflow_v2(
    task_description: str,
    use_best_of_n: bool = True,
    n_versions: int = 2,
    verbose: bool = True
) -> GraphState:
    """运行 Workflow V2"""
    initial_state = initialize_state(task_description)
    
    workflow = create_workflow_v2(use_best_of_n, n_versions)
    app = workflow.compile()
    
    if verbose:
        print("=" * 50)
        print(f"开始处理任务: {task_description}")
        print(f"使用 Best-of-N (N={n_versions}) + 模拟退火回滚")
        print("=" * 50)
    
    config = {"recursion_limit": settings.recursion_limit}
    final_state = app.invoke(initial_state, config=config)
    
    if verbose:
        print("\n" + "=" * 50)
        print("工作流执行完成")
        print(f"最终评分: {final_state.get('quality_score', 0):.1f}/10")
        print(f"最佳评分: {final_state.get('best_score_so_far', 0):.1f}/10")
        print("=" * 50)
    
    return final_state


def generate_with_best_of_n(
    task_description: str,
    n: int = 2
) -> dict:
    """使用 Best-of-N 生成高质量数据"""
    final_state = run_workflow_v2(task_description, use_best_of_n=True, n_versions=n)
    
    # 优先使用最佳版本（如果有回滚）
    best_draft = final_state.get("best_draft_so_far")
    current_draft = final_state.get("current_draft")
    draft = best_draft if best_draft else current_draft
    
    critique = final_state.get("critique_feedback")
    metadata = final_state.get("metadata", {})
    
    metadata.update({
        "end_time": datetime.now().isoformat(),
        "model_provider": settings.llm_provider,
        "generator_model": settings.default_model,
        "critic_model": settings.critic_model,
        "best_of_n_enabled": True,
        "n_versions": n,
        "rollback_enabled": True,
        "selected_version": final_state.get("_best_version_index", 0),
        "retry_count": final_state.get("retry_count", 0),
        "final_temperature": final_state.get("current_temperature", 0.3)
    })
    
    # 使用最佳分数
    score = final_state.get("best_score_so_far", 0)
    if score == 0:
        score = final_state.get("quality_score", 0)
    
    success = score >= settings.quality_threshold
    
    return {
        "success": success,
        "score": score,
        "iterations": final_state.get("iteration_count", 0),
        "data": draft.to_dict() if draft else None,
        "feedback": critique.feedback if critique else None,
        "metadata": metadata,
        "error": final_state.get("error_msg")
    }
