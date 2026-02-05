"""
Progress Tracker
进度追踪器 - 确保每次迭代都有进步

核心功能：
1. 追踪每次迭代的评分历史
2. 检测评分停滞或下降
3. 动态调整策略以强制进步
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class ProgressStatus(Enum):
    """进度状态"""
    IMPROVING = "improving"      # 正在进步
    STALLED = "stalled"          # 停滞
    DEGRADING = "degrading"      # 下降
    OSCILLATING = "oscillating"  # 震荡


@dataclass
class IterationRecord:
    """单次迭代记录"""
    iteration: int
    score: float
    feedback: str
    key_issues: List[str]
    

@dataclass
class ProgressReport:
    """进度报告"""
    status: ProgressStatus
    current_score: float
    best_score: float
    score_trend: List[float]  # 分数趋势
    improvement_rate: float   # 改进率
    suggestions: List[str]    # 改进建议


class ProgressTracker:
    """
    进度追踪器
    
    确保每次迭代都有实质性进步
    """
    
    def __init__(self, min_improvement: float = 0.3):
        """
        初始化
        
        Args:
            min_improvement: 最小改进阈值（分数提升）
        """
        self.min_improvement = min_improvement
        self.history: List[IterationRecord] = []
        self.best_score = 0.0
        self.best_version = None
    
    def record(self, iteration: int, score: float, feedback: str, issues: List[str]):
        """
        记录一次迭代
        
        Args:
            iteration: 迭代次数
            score: 评分
            feedback: 反馈
            issues: 关键问题
        """
        record = IterationRecord(
            iteration=iteration,
            score=score,
            feedback=feedback,
            key_issues=issues
        )
        self.history.append(record)
        
        # 更新最佳分数
        if score > self.best_score:
            self.best_score = score
    
    def analyze(self) -> ProgressReport:
        """
        分析进度状态
        
        Returns:
            ProgressReport: 进度报告
        """
        if len(self.history) < 2:
            return ProgressReport(
                status=ProgressStatus.IMPROVING,
                current_score=self.history[-1].score if self.history else 0,
                best_score=self.best_score,
                score_trend=[r.score for r in self.history],
                improvement_rate=0.0,
                suggestions=["需要更多迭代数据"]
            )
        
        scores = [r.score for r in self.history]
        current = scores[-1]
        previous = scores[-2]
        
        # 计算趋势
        improvement = current - previous
        total_improvement = current - scores[0]
        
        # 检测震荡（上升下降交替）
        if len(scores) >= 3:
            oscillation = sum(1 for i in range(2, len(scores)) 
                            if (scores[i] - scores[i-1]) * (scores[i-1] - scores[i-2]) < 0)
            oscillation_rate = oscillation / (len(scores) - 2)
        else:
            oscillation_rate = 0
        
        # 确定状态
        if oscillation_rate > 0.5:
            status = ProgressStatus.OSCILLATING
        elif improvement < -0.5:
            status = ProgressStatus.DEGRADING
        elif improvement < self.min_improvement:
            status = ProgressStatus.STALLED
        else:
            status = ProgressStatus.IMPROVING
        
        # 生成建议
        suggestions = self._generate_suggestions(status, scores)
        
        return ProgressReport(
            status=status,
            current_score=current,
            best_score=self.best_score,
            score_trend=scores,
            improvement_rate=improvement,
            suggestions=suggestions
        )
    
    def _generate_suggestions(self, status: ProgressStatus, scores: List[float]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if status == ProgressStatus.STALLED:
            suggestions.append("评分停滞，需要更激进的改进策略")
            suggestions.append("建议：增加更多具体例子")
            suggestions.append("建议：补充数学公式或代码")
        
        elif status == ProgressStatus.DEGRADING:
            suggestions.append("评分下降，可能优化方向错误")
            suggestions.append("建议：回到之前的高分版本")
            suggestions.append("建议：重新分析问题本质")
        
        elif status == ProgressStatus.OSCILLATING:
            suggestions.append("评分震荡，改进不稳定")
            suggestions.append("建议：聚焦于前一次的主要问题")
            suggestions.append("建议：避免过度修改")
        
        else:  # IMPROVING
            if scores[-1] < 7.0:
                suggestions.append("正在进步，但分数仍偏低")
                suggestions.append("建议：继续深化内容")
            else:
                suggestions.append("进步良好，继续保持")
        
        return suggestions
    
    def should_change_strategy(self) -> bool:
        """是否需要改变策略"""
        if len(self.history) < 2:
            return False
        
        report = self.analyze()
        return report.status in [ProgressStatus.STALLED, ProgressStatus.DEGRADING, ProgressStatus.OSCILLATING]
    
    def get_best_version_info(self) -> Optional[IterationRecord]:
        """获取最佳版本的信息"""
        if not self.history:
            return None
        
        best = max(self.history, key=lambda r: r.score)
        return best
    
    def get_recent_issues(self, n: int = 3) -> List[str]:
        """获取最近 N 次的主要问题"""
        if not self.history:
            return []
        
        recent = self.history[-n:]
        all_issues = []
        for record in recent:
            all_issues.extend(record.key_issues)
        
        # 去重
        return list(dict.fromkeys(all_issues))[:5]  # 最多5个
    
    def reset(self):
        """重置追踪器"""
        self.history.clear()
        self.best_score = 0.0
        self.best_version = None


class IterationManager:
    """
    迭代管理器
    
    管理整个迭代过程，确保有效收敛
    """
    
    def __init__(
        self,
        max_iterations: int = 8,
        min_improvement: float = 0.3,
        patience: int = 3  # 允许停滞的次数
    ):
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.patience = patience
        self.tracker = ProgressTracker(min_improvement)
        self.stall_count = 0
    
    def should_continue(self, iteration: int, score: float) -> bool:
        """
        判断是否应该继续迭代
        
        Returns:
            bool: 是否继续
        """
        # 达到最大迭代次数
        if iteration >= self.max_iterations:
            return False
        
        # 已经达到目标
        if score >= 8.5:
            return False
        
        # 检测停滞
        if self.tracker.history:
            report = self.tracker.analyze()
            
            if report.status == ProgressStatus.STALLED:
                self.stall_count += 1
                if self.stall_count >= self.patience:
                    print(f"[IterationManager] 连续 {self.patience} 次停滞，停止迭代")
                    return False
            else:
                self.stall_count = 0
        
        return True
    
    def record(self, iteration: int, score: float, feedback: str, issues: List[str]):
        """记录迭代"""
        self.tracker.record(iteration, score, feedback, issues)
    
    def get_progress_report(self) -> ProgressReport:
        """获取进度报告"""
        return self.tracker.analyze()
    
    def get_strategy_adjustment(self) -> Dict:
        """
        获取策略调整建议
        
        Returns:
            调整参数
        """
        report = self.tracker.analyze()
        
        adjustments = {
            "temperature_boost": 0.0,
            "require_examples": False,
            "require_math": False,
            "focus_areas": []
        }
        
        if report.status == ProgressStatus.STALLED:
            adjustments["temperature_boost"] = 0.2  # 增加随机性
            adjustments["require_examples"] = True
            adjustments["focus_areas"] = report.suggestions
        
        elif report.status == ProgressStatus.DEGRADING:
            adjustments["temperature_boost"] = -0.1  # 降低随机性
            adjustments["focus_areas"] = ["回到之前的成功方向"]
        
        return adjustments
