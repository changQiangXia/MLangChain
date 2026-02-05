"""
Complexity Evaluator
复杂度评估器 - 实现"复杂度进化" (Evolution Strategy)

目标：
1. 评估指令复杂度
2. 太简单的指令要求增加约束
3. 提升数据多样性

引用：Self-Instruct: Aligning Language Model with Self Generated Instructions (Wang et al., 2023)
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field

from src.llm_factory import create_llm
from src.core.safe_json_utils import safe_json_loads


class ComplexityLevel(str, Enum):
    """复杂度等级"""
    SIMPLE = "simple"       # 简单（1+1=几）
    MEDIUM = "medium"       # 中等（解释概念）
    COMPLEX = "complex"     # 复杂（多步推理）
    EXPERT = "expert"       # 专家级（开放研究问题）


@dataclass
class ComplexityMetrics:
    """复杂度指标"""
    reasoning_steps: int        # 推理步骤数
    knowledge_domains: int      # 知识领域数
    constraints: int            # 约束条件数
    ambiguity: float            # 模糊程度 (0-1)
    creativity_required: float  # 创造力要求 (0-1)


class ComplexityScore(BaseModel):
    """复杂度评分结果"""
    score: float = Field(description="复杂度分数 (0-1)", ge=0.0, le=1.0)
    level: ComplexityLevel = Field(description="复杂度等级")
    metrics: Dict = Field(description="详细指标")
    reasoning: str = Field(description="评估理由")


class ComplexityEvaluator:
    """
    复杂度评估器
    
    评估指令的复杂度，判断是否需要进行"进化"
    """
    
    # 复杂度阈值
    THRESHOLDS = {
        "evolution_trigger": 0.3,   # 低于此值触发进化
        "ideal_range": (0.4, 0.8),  # 理想复杂度范围
        "too_complex": 0.9          # 高于此值可能无法完成
    }
    
    def __init__(self):
        self.llm = create_llm(temperature=0.1)
    
    def evaluate(self, instruction: str) -> ComplexityScore:
        """
        评估指令复杂度
        
        Args:
            instruction: 指令文本
            
        Returns:
            ComplexityScore: 复杂度评分
        """
        # 方法1: 启发式规则（快速且稳定）
        heuristic_score = self._heuristic_evaluate(instruction)
        
        # 方法2: LLM 评估（精确但可能失败）
        try:
            llm_score = self._llm_evaluate(instruction)
            if llm_score.score > 0:  # LLM 成功返回
                # 综合（加权平均）
                final_score = 0.3 * heuristic_score + 0.7 * llm_score.score
                level = self._determine_level(final_score)
                return ComplexityScore(
                    score=final_score,
                    level=level,
                    metrics=llm_score.metrics,
                    reasoning=llm_score.reasoning
                )
        except Exception as e:
            print(f"[ComplexityEvaluator] LLM评估失败，使用启发式: {e}")
        
        # LLM 失败，仅使用启发式
        level = self._determine_level(heuristic_score)
        return ComplexityScore(
            score=heuristic_score,
            level=level,
            metrics={},
            reasoning="基于启发式规则评估"
        )
    
    def _heuristic_evaluate(self, instruction: str) -> float:
        """启发式规则评估"""
        score = 0.5  # 基础分
        
        # 1. 长度（长指令通常更复杂）
        length = len(instruction)
        if length < 10:
            score -= 0.2
        elif length > 50:
            score += 0.1
        
        # 2. 关键词复杂度
        complex_keywords = [
            "证明", "推导", "分析", "对比", "比较", "优化",
            "设计", "实现", "构建", "开发", "研究",
            "多步", "多维度", "综合考虑", "深入",
            "prove", "derive", "analyze", "compare", "optimize",
            "design", "implement", "research"
        ]
        
        simple_keywords = [
            "是什么", "什么是", "定义", "简单", "简述", "介绍",
            "what is", "define", "simple", "brief"
        ]
        
        for kw in complex_keywords:
            if kw in instruction:
                score += 0.05
        
        for kw in simple_keywords:
            if kw in instruction:
                score -= 0.05
        
        # 3. 问题类型
        if "为什么" in instruction or "why" in instruction.lower():
            score += 0.1  # 解释原因更复杂
        
        if "如何" in instruction or "how to" in instruction.lower():
            score += 0.1  # 过程性问题更复杂
        
        # 4. 约束条件（逗号、分号通常表示多约束）
        constraint_count = instruction.count('，') + instruction.count('、') + instruction.count(';')
        score += constraint_count * 0.02
        
        # 限制范围
        return max(0.0, min(1.0, score))
    
    def _llm_evaluate(self, instruction: str) -> ComplexityScore:
        """使用 LLM 评估复杂度"""
        prompt = f"""请评估以下指令的复杂度。

指令: "{instruction}"

评估维度：
1. reasoning_steps (1-5): 需要多少步推理？
   - 1步: 直接回答（如"什么是X"）
   - 2-3步: 需要解释和举例
   - 4-5步: 需要多步推理、推导、证明

2. knowledge_domains (1-3): 涉及多少知识领域？
   - 1: 单一领域
   - 2: 两个领域结合
   - 3: 多个跨领域知识

3. constraints (0-3): 有多少约束条件？
   - 0: 无约束
   - 1-2: 有具体要求
   - 3+: 多重约束

4. ambiguity (0.0-1.0): 模糊程度
   - 0.0: 非常明确
   - 0.5: 有一定开放性
   - 1.0: 非常开放/模糊

5. creativity_required (0.0-1.0): 创造力要求
   - 0.0: 纯知识性问题
   - 0.5: 需要一定创造
   - 1.0: 高度创造性任务

请以 JSON 格式输出：
{{
  "metrics": {{
    "reasoning_steps": 1-5,
    "knowledge_domains": 1-3,
    "constraints": 0-3,
    "ambiguity": 0.0-1.0,
    "creativity_required": 0.0-1.0
  }},
  "score": 0.0-1.0,
  "level": "simple/medium/complex/expert",
  "reasoning": "评估理由"
}}

评分标准：
- 0.0-0.3: simple (太简单，需要进化)
- 0.3-0.6: medium (适中)
- 0.6-0.8: complex (较复杂)
- 0.8-1.0: expert (专家级)

只输出 JSON。"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 使用安全的 JSON 解析
            data = safe_json_loads(content, default=None)
            
            if data is None:
                raise ValueError("JSON 解析失败")
            
            return ComplexityScore(
                score=float(data.get("score", 0.5)),
                level=ComplexityLevel(data.get("level", "medium")),
                metrics=data.get("metrics", {}),
                reasoning=data.get("reasoning", "")
            )
            
        except Exception as e:
            print(f"[ComplexityEvaluator] LLM评估失败: {e}")
            return ComplexityScore(
                score=0.5,
                level=ComplexityLevel.MEDIUM,
                metrics={},
                reasoning="评估失败"
            )
    
    def _determine_level(self, score: float) -> ComplexityLevel:
        """根据分数确定等级"""
        if score < 0.3:
            return ComplexityLevel.SIMPLE
        elif score < 0.6:
            return ComplexityLevel.MEDIUM
        elif score < 0.8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT
    
    def should_evolve(self, score: ComplexityScore) -> bool:
        """
        判断是否需要进化
        
        Args:
            score: 复杂度评分
            
        Returns:
            bool: 是否需要进化
        """
        return score.score < self.THRESHOLDS["evolution_trigger"]
    
    def is_too_complex(self, score: ComplexityScore) -> bool:
        """判断是否过于复杂（可能无法完成）"""
        return score.score > self.THRESHOLDS["too_complex"]


class InstructionEvolver:
    """
    指令进化器
    
    对太简单的指令进行"进化"，增加复杂度
    """
    
    # 进化策略
    EVOLUTION_STRATEGIES = {
        "add_constraints": {
            "description": "增加约束条件",
            "templates": [
                "{instruction}，要求时间复杂度为 O(n)",
                "{instruction}，要求空间复杂度为 O(1)",
                "{instruction}，不能使用标准库函数",
                "{instruction}，要求给出数学证明",
                "{instruction}，要求考虑边界情况",
            ]
        },
        "multi_step": {
            "description": "要求多步推理",
            "templates": [
                "{instruction}，并详细解释每一步的原理",
                "{instruction}，对比至少两种方法的优缺点",
                "{instruction}，分析为什么这种方法有效",
                "{instruction}，给出具体的推导过程",
                "{instruction}，考虑极端情况下的表现",
            ]
        },
        "comparison": {
            "description": "要求对比分析",
            "templates": [
                "对比 {instruction} 和另一种方法的异同",
                "{instruction}，并与传统方法进行比较",
                "{instruction}，分析不同场景下的适用性",
                "{instruction}，评估其优缺点",
            ]
        },
        "application": {
            "description": "要求实际应用",
            "templates": [
                "{instruction}，并给出具体的应用案例",
                "{instruction}，说明在实际项目中的使用",
                "{instruction}，提供代码实现",
                "{instruction}，给出可运行的示例",
            ]
        },
        "deep_dive": {
            "description": "深入分析",
            "templates": [
                "深入分析 {instruction} 的底层原理",
                "{instruction}，探讨其数学基础",
                "{instruction}，分析时间/空间复杂度",
                "{instruction}，讨论可能的优化方向",
            ]
        }
    }
    
    def __init__(self):
        self.llm = create_llm(temperature=0.7)
        self.evaluator = ComplexityEvaluator()
    
    def evolve(self, instruction: str, target_complexity: float = 0.5) -> str:
        """
        进化指令
        
        Args:
            instruction: 原始指令
            target_complexity: 目标复杂度
            
        Returns:
            str: 进化后的指令
        """
        current_score = self.evaluator.evaluate(instruction)
        
        if not self.evaluator.should_evolve(current_score):
            return instruction  # 不需要进化
        
        print(f"[Evolver] 指令复杂度过低 ({current_score.score:.2f})，开始进化...")
        
        # 选择合适的进化策略
        strategy = self._select_strategy(current_score)
        
        # 应用进化
        evolved = self._apply_evolution(instruction, strategy)
        
        # 验证进化后的复杂度
        new_score = self.evaluator.evaluate(evolved)
        print(f"[Evolver] 进化后复杂度: {new_score.score:.2f} ({new_score.level.value})")
        
        return evolved
    
    def _select_strategy(self, score: ComplexityScore) -> str:
        """选择进化策略"""
        # 根据当前复杂度选择策略
        if score.score < 0.2:
            # 太简单：增加约束或多步推理
            return "multi_step" if score.metrics.get("reasoning_steps", 1) < 2 else "add_constraints"
        elif score.score < 0.3:
            # 比较简单：要求对比或应用
            return "comparison" if score.metrics.get("knowledge_domains", 1) < 2 else "application"
        else:
            # 接近阈值：深入分析
            return "deep_dive"
    
    def _apply_evolution(self, instruction: str, strategy: str) -> str:
        """应用进化策略"""
        import random
        
        strategy_config = self.EVOLUTION_STRATEGIES.get(strategy, self.EVOLUTION_STRATEGIES["multi_step"])
        template = random.choice(strategy_config["templates"])
        
        return template.format(instruction=instruction)
    
    def evolve_with_llm(self, instruction: str) -> str:
        """使用 LLM 进行更智能的进化"""
        prompt = f"""请对以下指令进行"进化"，增加其复杂度和深度。

原始指令: "{instruction}"

当前问题：
- 指令过于简单，缺乏挑战性
- 回答容易流于表面
- 对训练模型帮助有限

进化要求：
1. 增加约束条件（如时间/空间复杂度要求）
2. 要求多步推理或详细推导
3. 要求对比分析不同方法
4. 要求给出具体应用案例
5. 要求考虑边界情况或异常处理

请给出 3 个进化后的版本，每个版本增加不同的复杂度维度：

版本 1（增加约束）: 
版本 2（要求多步推理）: 
版本 3（要求对比分析）: 

只输出进化后的指令，不要其他说明。"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 提取第一个版本
            lines = content.split('\n')
            for line in lines:
                if '版本 1' in line or '版本1' in line:
                    # 提取指令部分
                    if ':' in line:
                        return line.split(':', 1)[1].strip()
            
            # 如果没找到，返回原指令
            return instruction
            
        except Exception as e:
            print(f"[Evolver] LLM进化失败: {e}")
            return instruction


# 便捷函数
def evaluate_complexity(instruction: str) -> ComplexityScore:
    """快速评估复杂度"""
    evaluator = ComplexityEvaluator()
    return evaluator.evaluate(instruction)


def evolve_instruction(instruction: str) -> str:
    """快速进化指令"""
    evolver = InstructionEvolver()
    return evolver.evolve(instruction)
