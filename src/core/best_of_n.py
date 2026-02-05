"""
Best-of-N (对抗性评分)

解决评分不稳定问题：
- LLM 对绝对分数（给8分还是9分）敏感且不稳定
- 但对相对好坏（A比B好）判断极准

方案：
1. Generator 生成 N 个版本（N=2 或 3）
2. Critic 对比选择最佳版本
3. 基于胜利者给出评分

引用：
- "Learning to summarize from human feedback" (Stiennon et al., 2020)
- RLHF (Reinforcement Learning from Human Feedback) 核心方法
"""

import json
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from src.llm_factory import create_llm
from src.state import AlpacaData


class ComparisonResult(BaseModel):
    """对比结果"""
    winner_index: int = Field(description="获胜者索引（0或1）")
    loser_index: int = Field(description="失败者索引")
    confidence: float = Field(description="置信度", ge=0.0, le=1.0)
    reasoning: str = Field(description="选择理由")
    dimension_scores: dict = Field(default_factory=dict, description="各维度对比")


@dataclass
class BestOfNResult:
    """Best-of-N 最终结果"""
    best_version: AlpacaData          # 最佳版本
    best_index: int                   # 最佳版本索引
    all_versions: List[AlpacaData]    # 所有版本
    comparisons: List[ComparisonResult]  # 对比过程
    final_score: float                # 最终分数


class PairwiseComparator:
    """
    成对比较器
    
    对比两个版本，判断哪个更好
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.llm = create_llm(model_name=model_name, temperature=0.1)
    
    def compare(
        self, 
        task_description: str, 
        version_a: AlpacaData, 
        version_b: AlpacaData
    ) -> ComparisonResult:
        """
        对比两个版本
        
        Args:
            task_description: 原始任务描述
            version_a: 版本 A
            version_b: 版本 B
            
        Returns:
            ComparisonResult: 对比结果
        """
        prompt = f"""请对比以下两个回答，选择更好的一个。

[原始任务]
{task_description}

[版本 A]
指令: {version_a.instruction}
输出: {version_a.output[:1500]}...

[版本 B]
指令: {version_b.instruction}
输出: {version_b.output[:1500]}...

评估维度：
1. 准确性：哪个回答更准确、专业术语更正确？
2. 完整性：哪个回答更全面、覆盖了更多关键点？
3. 清晰度：哪个回答更易懂、逻辑更清晰？
4. 实用性：哪个回答更有用、例子更具体？
5. 表达质量：哪个回答语言更流畅、结构更好？

请以 JSON 格式输出：
{{
  "winner": "A" or "B",
  "confidence": 0.0-1.0,
  "reasoning": "详细说明为什么这个版本更好",
  "dimension_scores": {{
    "accuracy": {{"A": 0-10, "B": 0-10}},
    "completeness": {{"A": 0-10, "B": 0-10}},
    "clarity": {{"A": 0-10, "B": 0-10}},
    "practicality": {{"A": 0-10, "B": 0-10}},
    "expression": {{"A": 0-10, "B": 0-10}}
  }}
}}

重要：
- 必须选出胜利者，不能平局
- 如果两个版本都很差，选择相对较好的
- 如果两个版本都很好，选择相对更好的
- confidence 表示你对判断的确定程度

只输出 JSON，不要其他内容。"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            
            winner_str = data.get("winner", "A").upper()
            winner_index = 0 if winner_str == "A" else 1
            loser_index = 1 - winner_index
            
            return ComparisonResult(
                winner_index=winner_index,
                loser_index=loser_index,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "无说明"),
                dimension_scores=data.get("dimension_scores", {})
            )
            
        except Exception as e:
            print(f"[PairwiseComparator] 对比失败: {e}")
            # 默认返回 A
            return ComparisonResult(
                winner_index=0,
                loser_index=1,
                confidence=0.5,
                reasoning=f"解析失败，默认选择A: {e}",
                dimension_scores={}
            )


class BestOfNSelector:
    """
    Best-of-N 选择器
    
    实现锦标赛机制，从 N 个版本中选择最佳
    """
    
    def __init__(self, n: int = 2, model_name: Optional[str] = None):
        """
        初始化
        
        Args:
            n: 生成版本数（2 或 3）
            model_name: LLM 模型名称
        """
        if n < 2:
            raise ValueError("n 必须 >= 2")
        if n > 3:
            print(f"[Warning] n={n} 可能成本较高，建议使用 2 或 3")
        
        self.n = n
        self.comparator = PairwiseComparator(model_name)
    
    def select(
        self, 
        task_description: str, 
        versions: List[AlpacaData]
    ) -> BestOfNResult:
        """
        从 N 个版本中选择最佳
        
        Args:
            task_description: 原始任务描述
            versions: N 个版本的数据
            
        Returns:
            BestOfNResult: 选择结果
        """
        if len(versions) < 2:
            raise ValueError("至少需要 2 个版本")
        
        print(f"[BestOfN] 从 {len(versions)} 个版本中选择最佳...")
        
        # 锦标赛机制
        if len(versions) == 2:
            # 2 个版本：直接对比
            return self._compare_two(task_description, versions[0], versions[1])
        
        elif len(versions) == 3:
            # 3 个版本：两两对比，选出胜者
            return self._compare_three(task_description, versions)
        
        else:
            # N > 3：单淘汰赛
            return self._tournament(task_description, versions)
    
    def _compare_two(
        self, 
        task_description: str, 
        version_a: AlpacaData, 
        version_b: AlpacaData
    ) -> BestOfNResult:
        """对比两个版本"""
        comparison = self.comparator.compare(task_description, version_a, version_b)
        
        best_index = comparison.winner_index
        best_version = version_a if best_index == 0 else version_b
        
        # 基于对比结果计算分数
        # 置信度高则分数高，置信度低则分数中等
        base_score = 8.0
        if comparison.confidence > 0.8:
            final_score = 9.0
        elif comparison.confidence > 0.6:
            final_score = 8.5
        else:
            final_score = 8.0
        
        print(f"[BestOfN] 选择版本 {best_index + 1} (置信度: {comparison.confidence:.2f})")
        
        return BestOfNResult(
            best_version=best_version,
            best_index=best_index,
            all_versions=[version_a, version_b],
            comparisons=[comparison],
            final_score=final_score
        )
    
    def _compare_three(
        self, 
        task_description: str, 
        versions: List[AlpacaData]
    ) -> BestOfNResult:
        """对比三个版本（两两对比）"""
        v0, v1, v2 = versions[0], versions[1], versions[2]
        
        # Round 1: 0 vs 1
        comp1 = self.comparator.compare(task_description, v0, v1)
        winner1 = v0 if comp1.winner_index == 0 else v1
        winner1_idx = 0 if comp1.winner_index == 0 else 1
        
        # Round 2: winner vs 2
        comp2 = self.comparator.compare(task_description, winner1, v2)
        final_winner_idx = winner1_idx if comp2.winner_index == 0 else 2
        
        # 计算最终分数
        avg_confidence = (comp1.confidence + comp2.confidence) / 2
        if avg_confidence > 0.8:
            final_score = 9.0
        elif avg_confidence > 0.6:
            final_score = 8.5
        else:
            final_score = 8.0
        
        print(f"[BestOfN] 选择版本 {final_winner_idx + 1} (平均置信度: {avg_confidence:.2f})")
        
        return BestOfNResult(
            best_version=versions[final_winner_idx],
            best_index=final_winner_idx,
            all_versions=versions,
            comparisons=[comp1, comp2],
            final_score=final_score
        )
    
    def _tournament(
        self, 
        task_description: str, 
        versions: List[AlpacaData]
    ) -> BestOfNResult:
        """锦标赛机制（N > 3）"""
        current_winner = versions[0]
        current_winner_idx = 0
        comparisons = []
        
        for i in range(1, len(versions)):
            comp = self.comparator.compare(
                task_description, 
                current_winner, 
                versions[i]
            )
            comparisons.append(comp)
            
            if comp.winner_index == 1:  # 挑战者获胜
                current_winner = versions[i]
                current_winner_idx = i
        
        # 计算平均置信度
        avg_confidence = sum(c.confidence for c in comparisons) / len(comparisons)
        final_score = 8.5 if avg_confidence > 0.7 else 8.0
        
        print(f"[BestOfN] 经过 {len(comparisons)} 轮对比，选择版本 {current_winner_idx + 1}")
        
        return BestOfNResult(
            best_version=current_winner,
            best_index=current_winner_idx,
            all_versions=versions,
            comparisons=comparisons,
            final_score=final_score
        )


class MultiVersionGenerator:
    """
    多版本生成器
    
    为同一个任务生成 N 个不同版本
    """
    
    def __init__(self, n: int = 2, temperature_range: Tuple[float, float] = (0.5, 0.9)):
        """
        初始化
        
        Args:
            n: 生成版本数
            temperature_range: 温度范围（增加多样性）
        """
        self.n = n
        self.temperature_range = temperature_range
    
    def generate_versions(
        self, 
        generator, 
        task_description: str,
        search_results: Optional[list] = None
    ) -> List[AlpacaData]:
        """
        生成 N 个版本
        
        Args:
            generator: Generator Agent
            task_description: 任务描述
            search_results: 搜索结果（可选）
            
        Returns:
            List[AlpacaData]: N 个版本
        """
        import random
        
        versions = []
        temps = [random.uniform(*self.temperature_range) for _ in range(self.n)]
        
        print(f"[MultiVersionGenerator] 生成 {self.n} 个版本...")
        print(f"[MultiVersionGenerator] 使用温度: {[f'{t:.2f}' for t in temps]}")
        
        for i, temp in enumerate(temps):
            print(f"[MultiVersionGenerator] 生成版本 {i+1}/{self.n} (temperature={temp:.2f})...")
            
            # 临时修改 generator 的温度
            original_temp = getattr(generator, 'temperature', None)
            generator.llm.temperature = temp
            
            try:
                version = generator.generate(task_description, search_results)
                versions.append(version)
            except Exception as e:
                print(f"[MultiVersionGenerator] 版本 {i+1} 生成失败: {e}")
                # 如果失败，使用之前的版本（如果有）
                if versions:
                    versions.append(versions[-1])
                else:
                    raise
            
            # 恢复原始温度
            if original_temp is not None:
                generator.llm.temperature = original_temp
        
        print(f"[MultiVersionGenerator] 完成，生成 {len(versions)} 个版本")
        return versions


# 便捷函数
def select_best_version(
    task_description: str, 
    versions: List[AlpacaData],
    n: int = 2
) -> BestOfNResult:
    """
    从多个版本中选择最佳
    
    Args:
        task_description: 任务描述
        versions: 版本列表
        n: 版本数（用于显示）
        
    Returns:
        BestOfNResult: 最佳版本
    """
    selector = BestOfNSelector(n=len(versions))
    return selector.select(task_description, versions)
