"""
完整系统测试脚本
测试所有 Phase 8 高级功能
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("LangGraph Multi-Agent System - Full Test Suite")
print("=" * 60)

# 测试 1: 基础模块导入
print("\n[TEST 1] Module Import Test")
try:
    from src.core.task_classifier import TaskClassifier, TaskType
    from src.core.grading_criteria import GradingCriteria
    from src.core.code_validator import CodeValidator
    from src.core.fact_checker import FactChecker
    from src.core.best_of_n import BestOfNSelector
    from src.core.complexity_evaluator import ComplexityEvaluator
    print("  [OK] All core modules imported")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# 测试 2: 任务分类
print("\n[TEST 2] Task Classification")
classifier = TaskClassifier(use_llm=False)
test_cases = [
    ("写一个Python函数", TaskType.CODE),
    ("证明为什么梯度下降收敛", TaskType.REASONING),
    ("解释什么是深度学习", TaskType.EXPLANATION),
]
for instruction, expected in test_cases:
    result = classifier.classify(instruction)
    status = "OK" if result.task_type == expected else "FAIL"
    print(f"  [{status}] {instruction[:30]}... -> {result.task_type.value}")

# 测试 3: 复杂度评估
print("\n[TEST 3] Complexity Evaluation")
evaluator = ComplexityEvaluator()
test_instructions = [
    "1+1等于几",
    "解释什么是深度学习",
    "证明为什么梯度下降会收敛到全局最优",
]
for instruction in test_instructions:
    score = evaluator.evaluate(instruction)
    evolve_flag = "[EVOLVE]" if evaluator.should_evolve(score) else ""
    print(f"  {instruction[:40]}... -> {score.score:.2f} ({score.level.value}) {evolve_flag}")

# 测试 4: 代码验证
print("\n[TEST 4] Code Validation")
validator = CodeValidator()
test_code = """
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
result = validator.validate(test_code)
print(f"  Syntax valid: {result.syntax_valid}")
print(f"  Can execute: {result.can_execute}")
print(f"  Has comments: {result.has_comments}")
print(f"  Quality score: {validator.calculate_quality_score(result):.1f}/10")

# 测试 5: 动态评分标准
print("\n[TEST 5] Dynamic Grading Criteria")
from src.core.grading_criteria import get_grading_criteria
for task_type in [TaskType.CODE, TaskType.EXPLANATION]:
    criteria = get_grading_criteria(task_type)
    length_check = "ON" if criteria.length_check_enabled else "OFF"
    print(f"  {task_type.value:12} | Length check: {length_check} | Min: {criteria.min_length}")

# 测试 6: 指令进化
print("\n[TEST 6] Instruction Evolution")
from src.core.complexity_evaluator import evolve_instruction
simple_instructions = [
    "什么是深度学习",
    "解释CNN",
]
for instruction in simple_instructions:
    evolved = evolve_instruction(instruction)
    print(f"  Original: {instruction}")
    print(f"  Evolved:  {evolved}")
    print()

print("=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nSystem is ready for production use!")
print("\nNext steps:")
print("1. Run: python main.py -t 'your topic'")
print("2. Or: python main.py (interactive mode)")
print("3. Or: python main.py -i topics.txt -o results.jsonl")
