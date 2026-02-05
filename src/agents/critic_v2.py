"""
Critic Agent V2 - 动态类型感知版本

基于任务类型应用不同的评分标准，解决"字数暴力"问题。
"""

import json
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.state import AlpacaData, CritiqueResult, GraphState
from src.llm_factory import create_llm
from src.core.task_classifier import TaskClassifier, TaskType
from src.core.grading_criteria import GradingCriteria
from src.core.code_validator import CodeValidator, extract_code_from_output
from src.core.fact_checker import FactChecker, VerificationStatus
from src.core.complexity_evaluator import ComplexityEvaluator, InstructionEvolver
from src.core.safe_json_utils import safe_json_loads


class CriticV2Agent:
    """
    Critic V2 - 动态类型感知评分
    
    主要改进：
    1. 自动识别任务类型
    2. 根据类型应用不同评分标准
    3. CODE 类型不看字数，看代码质量
    4. REASONING 类型看思维链
    """
    
    def __init__(self, model_name: Optional[str] = None, enable_fact_check: bool = True, enable_complexity_check: bool = True):
        self.model_name = model_name or settings.critic_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.1)
        self.task_classifier = TaskClassifier()
        self.code_validator = CodeValidator()
        self.fact_checker = FactChecker() if enable_fact_check else None
        self.complexity_evaluator = ComplexityEvaluator() if enable_complexity_check else None
        self.instruction_evolver = InstructionEvolver() if enable_complexity_check else None
        self.enable_fact_check = enable_fact_check
        self.enable_complexity_check = enable_complexity_check
    
    def critique(self, task_description: str, draft: AlpacaData) -> CritiqueResult:
        """
        对生成的数据进行多维度评审（动态类型感知）
        """
        # Step 1: 识别任务类型
        classification = self.task_classifier.classify(draft.instruction)
        task_type = classification.task_type
        
        print(f"[CriticV2] 任务类型: {task_type.value} (置信度: {classification.confidence:.2f})")
        print(f"[CriticV2] 分类理由: {classification.reasoning}")
        
        # Step 2: 获取对应类型的评分标准
        criteria = GradingCriteria.get_criteria(task_type)
        
        # Step 3: 特殊处理 CODE 类型
        if task_type == TaskType.CODE:
            return self._critique_code(task_description, draft, criteria)
        
        # Step 4: 通用评分
        return self._critique_general(task_description, draft, criteria, task_type)
    
    def _critique_code(self, task_description: str, draft: AlpacaData, criteria) -> CritiqueResult:
        """专门评审代码类任务"""
        print("[CriticV2] 使用代码评审模式（不看字数，看代码质量）")
        
        # 提取代码
        code = extract_code_from_output(draft.output)
        
        if not code:
            return CritiqueResult(
                score=3.0,
                feedback="未检测到可执行代码块，请使用 ```python 格式提供代码",
                issues=["缺少代码块", "格式不符合要求"]
            )
        
        # 验证代码
        validation = self.code_validator.validate(code)
        
        # 计算分数
        base_score = self.code_validator.calculate_quality_score(validation)
        
        # 构建反馈
        feedback_parts = [
            f"代码验证结果:",
            f"- 语法正确: {'[OK]' if validation.syntax_valid else '[FAIL]'}",
            f"- 可执行: {'[OK]' if validation.can_execute else '[FAIL]'}",
            f"- 有注释: {'[OK]' if validation.has_comments else '[FAIL]'}",
            f"- 有文档字符串: {'[OK]' if validation.has_docstring else '[FAIL]'}",
            f"- 函数数量: {validation.function_count}",
            f"- 代码行数: {validation.line_count}",
        ]
        
        if validation.has_error:
            feedback_parts.append(f"\n错误信息: {validation.has_error}")
        
        if validation.output:
            feedback_parts.append(f"\n执行输出:\n{validation.output}")
        
        # 根据验证结果调整分数
        issues = []
        if not validation.syntax_valid:
            issues.append("代码语法错误")
        if not validation.can_execute:
            issues.append("代码无法执行")
        if not validation.has_comments:
            issues.append("缺少注释")
        if not validation.has_docstring:
            issues.append("缺少文档字符串")
        
        # 最终分数（代码类最高10分）
        final_score = min(base_score, 10.0)
        
        return CritiqueResult(
            score=final_score,
            feedback="\n".join(feedback_parts),
            issues=issues if issues else ["代码质量良好"]
        )
    
    def _critique_general(self, task_description: str, draft: AlpacaData, criteria, task_type: TaskType) -> CritiqueResult:
        """通用评审（非代码类）"""
        output_length = len(draft.output) if draft.output else 0
        
        # 检查字数（动态标准）
        length_ok, length_deduction = GradingCriteria.check_length(task_type, output_length)
        
        if not length_ok:
            print(f"[CriticV2] 字数不足: {output_length} < {criteria.min_length}，将扣分 {length_deduction}")
        
        # ===== 复杂度评估（P3）=====
        complexity_info = None
        if self.enable_complexity_check and self.complexity_evaluator:
            print("[CriticV2] 评估指令复杂度...")
            complexity_score = self.complexity_evaluator.evaluate(draft.instruction)
            complexity_info = {
                "score": complexity_score.score,
                "level": complexity_score.level.value,
                "reasoning": complexity_score.reasoning
            }
            print(f"[CriticV2] 复杂度: {complexity_score.score:.2f} ({complexity_score.level.value})")
            
            # 如果太简单，标记需要进化
            if self.complexity_evaluator.should_evolve(complexity_score):
                print(f"[CriticV2] 指令过于简单，建议进化")
        
        # ===== RAG 验证（P1）===== 
        fact_deduction = 0.0
        fact_report = None
        if self.enable_fact_check and self.fact_checker:
            print("[CriticV2] 启动事实验证 (RAG)...")
            fact_report = self.fact_checker.check(draft.output)
            fact_deduction = fact_report.total_deduction
            print(f"[CriticV2] 事实验证完成，扣分: {fact_deduction}")
        
        # 构建 Prompt
        prompt = self._build_grading_prompt(task_description, draft, criteria, task_type, output_length, fact_report)
        
        # 调用 LLM 评分（带重试）
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                break
            except Exception as e:
                if "timeout" in str(e).lower() or "read operation" in str(e).lower():
                    print(f"[CriticV2] API 超时，重试 {attempt + 1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        # 返回一个默认的低分结果
                        return CritiqueResult(
                            score=0.0,
                            feedback="API 调用超时，无法完成评审",
                            issues=["API 超时"]
                        )
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        # 解析结果 - 使用安全的 JSON 解析
        try:
            content = response.content
            
            # 使用 safe_json_loads 处理各种转义问题
            data = safe_json_loads(content, default=None)
            
            if data is None:
                raise ValueError("JSON 解析失败，返回空数据")
            score = float(data.get("score", 0))
            
            # 应用字数扣分
            score = max(0, score - length_deduction)
            
            # 应用事实验证扣分
            score = max(0, score - fact_deduction)
            
            # 构建反馈（包含复杂度评估和事实验证）
            feedback = data.get("feedback", "")
            
            # 添加复杂度信息
            if complexity_info:
                feedback += f"\n\n[复杂度评估] {complexity_info['level']} ({complexity_info['score']:.2f})"
                if complexity_info['score'] < 0.3:
                    feedback += "\n[警告] 指令过于简单，建议增加约束条件或多步推理要求"
            
            # 添加事实验证信息
            if fact_report and fact_report.facts:
                feedback += f"\n\n[RAG事实验证] {fact_report.summary}"
                for f in fact_report.facts[:3]:  # 最多显示3个
                    emoji = {"verified": "[V]", "contradicted": "[X]", "doubtful": "[?]", "partial": "[~]"}.get(f.status.value, "[-]")
                    feedback += f"\n{emoji} {f.fact[:40]}... ({f.status.value})"
            
            issues = data.get("issues", [])
            # 添加事实问题
            if fact_report:
                for f in fact_report.facts:
                    if f.status == VerificationStatus.CONTRADICTED:
                        issues.append(f"事实错误: {f.fact[:50]}...")
                    elif f.status == VerificationStatus.DOUBTFUL:
                        issues.append(f"事实存疑: {f.fact[:50]}...")
            
            # 解析 improvement_guide（建设性CoT）
            improvement_guide = data.get("improvement_guide")
            
            return CritiqueResult(
                score=score,
                feedback=feedback,
                issues=issues if issues else ["质量良好"],
                improvement_guide=improvement_guide
            )
            
        except Exception as e:
            print(f"[CriticV2 Error] 解析失败: {e}")
            return CritiqueResult(
                score=0.0,
                feedback=f"评审解析失败: {e}",
                issues=["解析错误"]
            )
    
    def _build_grading_prompt(self, task_description: str, draft: AlpacaData, criteria, task_type: TaskType, output_length: int, fact_report=None) -> str:
        """构建评分 Prompt - 建设性CoT版本 (Strategy 2)"""
        
        # 获取评分标准文本
        grading_section = GradingCriteria.generate_prompt_section(task_type)
        
        # 添加 RAG 验证信息
        rag_section = ""
        if fact_report and fact_report.facts:
            rag_section = "\n## RAG 事实验证结果（必须考虑）\n"
            for f in fact_report.facts:
                rag_section += f"- [{f.status.value.upper()}] {f.fact[:60]}... (扣分: {f.deduction})\n"
            rag_section += "\n重要：如果存在 CONTRADICTED（矛盾）的事实，准确性维度必须给低分！\n"
        
        system_template = f"""你是一个极其严格但富有建设性的数据质量评审专家。

{grading_section}
{rag_section}

## 建设性评审原则 (Strategy 2)
不要只批评，要给出**具体的修改建议**：

❌ 错误示例："缺少代码示例"
✅ 正确示例："在第3段后添加Python代码示例，如：\`\`\`python\ndef example():\n    pass\n\`\`\`"

❌ 错误示例："解释不够深入"
✅ 正确示例："在解释Attention机制时，建议补充数学公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V"

## 评分标准
- 9.0-10分：卓越（可直接使用）
- 8.0-8.9分：良好（需微调）
- 7.0-7.9分：一般（需明显改进）
- <7分：较差（需大幅重写）

重要：平庸数据给7分以下，但必须给出具体改进建议！
"""

        human_template = f"""请严格评审以下指令数据，并给出建设性反馈：

任务类型: {criteria.description}

原始主题：{task_description}

生成的数据：
- 指令: {draft.instruction}
- 输入: {draft.input}
- 输出: {draft.output}

输出字数：{output_length} 字
{"字数要求：最少 " + str(criteria.min_length) + " 字" if criteria.min_length else "本类型不看字数，看内容质量"}

## 输出格式（JSON）
```json
{{
  "dimension_scores": {{"""
        
        # 添加各维度
        for dim in criteria.dimensions:
            human_template += f'\n    "{dim.name}": <0-{dim.max_score}>,'
        
        human_template += f"""
  }},
  "score": <总分0-10>,
  "feedback": "详细的评审意见，每个问题都要给出具体修改建议，不要只说问题",
  "issues": ["问题1: 具体描述 + 修改建议", "问题2: 具体描述 + 修改建议"],
  "improvement_guide": {{
    "keep": "必须保留的优秀部分（如：结构清晰、某段解释很好）",
    "fix": [
      {{
        "location": "问题位置（如：第2段、代码块后）",
        "problem": "具体问题",
        "suggestion": "具体修改建议（给出参考文本或伪代码）"
      }}
    ],
    "add": "建议新增的内容（如：添加一个具体例子、补充公式）"
  }}
}}
```

## 重要提示
1. **具体化**：每个建议都要有具体位置和内容
2. **保留优点**：明确指出哪些部分是好的，不要修改
3. **示例代码**：如果涉及代码，给出参考实现
4. **数量化**：如"增加2个例子"而非"增加例子"

记住：你是教练，不是裁判。帮助Refiner成功！
"""

        return f"{system_template}\n\n{human_template}"
    
    def __call__(self, state: GraphState) -> GraphState:
        """作为 LangGraph Node 调用 - 集成模拟退火机制"""
        print(f"\n[CriticV2] 正在严格评审数据质量...")
        
        task = state["task_description"]
        draft = state.get("current_draft")
        iteration = state.get("iteration_count", 0)
        
        if not draft:
            state["error_msg"] = "没有可评审的数据草稿"
            state["quality_score"] = 0.0
            return state
        
        # 执行评审
        result = self.critique(task, draft)
        current_score = result.score
        
        # 更新状态
        state["critique_feedback"] = result
        state["quality_score"] = current_score
        
        # === 模拟退火机制 (Strategy 1) ===
        best_score = state.get("best_score_so_far", 0.0)
        
        if current_score > best_score:
            # [新高分] 新纪录！保存为最佳版本
            print(f"[CriticV2] [新高分] 保存最佳版本: {best_score:.1f} -> {current_score:.1f}")
            state["best_draft_so_far"] = draft
            state["best_score_so_far"] = current_score
            state["retry_count"] = 0  # 重置重试计数
            state["current_temperature"] = 0.3  # 重置温度
        elif current_score < best_score - 0.5:
            # [警告] 分数下降超过0.5，标记需要回滚
            print(f"[CriticV2] [警告] 分数下降: {best_score:.1f} -> {current_score:.1f}，将触发回滚")
            state["retry_count"] = state.get("retry_count", 0) + 1
            # 提高温度以增加随机性，试图跳出局部最优
            state["current_temperature"] = min(0.8, 0.3 + state["retry_count"] * 0.15)
            print(f"[CriticV2] 温度提升至: {state['current_temperature']:.2f} (重试{state['retry_count']}次)")
        else:
            # 分数持平或微降，正常迭代
            print(f"[CriticV2] 分数持平: {current_score:.1f} (最佳: {best_score:.1f})")
        
        # 记录迭代历史
        history = state.get("iteration_history", [])
        history.append({
            "iteration": iteration,
            "score": current_score,
            "feedback": result.feedback,
            "issues": result.issues
        })
        state["iteration_history"] = history
        
        print(f"[CriticV2] 评分: {current_score}/10 | 最佳: {state.get('best_score_so_far', current_score)}/10")
        print(f"[CriticV2] 反馈: {result.feedback[:120]}...")
        
        return state


# 向后兼容
def critique_data(task_description: str, draft: AlpacaData) -> CritiqueResult:
    """快速评审数据"""
    agent = CriticV2Agent()
    return agent.critique(task_description, draft)
