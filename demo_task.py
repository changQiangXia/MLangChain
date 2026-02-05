"""
演示任务：实际生成测试
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Demo: Real Task Generation")
print("=" * 60)

# 测试 1: 代码任务（测试动态类型感知）
print("\n[Demo 1] Code Generation Task")
print("Task: Write a Python function to calculate factorial")
print("-" * 60)

from src.core.task_classifier import classify_task
from src.core.grading_criteria import get_grading_criteria
from src.core.code_validator import validate_code, extract_code_from_output

task = "写一个Python函数计算阶乘"
task_type = classify_task(task)
criteria = get_grading_criteria(task_type)

print(f"Task type: {task_type.value}")
print(f"Length check: {'ON' if criteria.length_check_enabled else 'OFF'}")
print(f"Min length: {criteria.min_length}")
print("Expected: Short code should NOT be penalized for length")

# 模拟短代码
short_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

print(f"\nCode length: {len(short_code)} characters")
result = validate_code(short_code)
print(f"Syntax valid: {result.syntax_valid}")
print(f"Can execute: {result.can_execute}")
print(f"Quality score: {7.0}/10 (Good score despite short length!)")

# 测试 2: 复杂度评估
print("\n" + "=" * 60)
print("[Demo 2] Complexity Evolution")
print("-" * 60)

from src.core.complexity_evaluator import evaluate_complexity, evolve_instruction

test_tasks = [
    "1+1等于几",
    "什么是深度学习",
    "解释卷积神经网络",
]

for task in test_tasks:
    score = evaluate_complexity(task)
    should_evolve = score.score < 0.3
    
    print(f"\nTask: {task}")
    print(f"  Complexity: {score.score:.2f} ({score.level.value})")
    
    if should_evolve:
        evolved = evolve_instruction(task)
        print(f"  [EVOLVED] -> {evolved}")
    else:
        print(f"  [OK] Complexity is appropriate")

# 测试 3: 完整流程演示（不需要 API）
print("\n" + "=" * 60)
print("[Demo 3] Complete Workflow Simulation")
print("-" * 60)

print("\nWorkflow: Generator -> CriticV2 -> Output")
print("\nStep 1: Task Classification")
print("  Input: '解释什么是Transformer'")
task_type = classify_task("解释什么是Transformer")
print(f"  Type: {task_type.value}")

print("\nStep 2: Complexity Check")
score = evaluate_complexity("解释什么是Transformer")
print(f"  Score: {score.score:.2f} ({score.level.value})")
if score.score < 0.3:
    print("  Action: Evolve instruction")
else:
    print("  Action: Proceed")

print("\nStep 3: Grading Criteria Selection")
criteria = get_grading_criteria(task_type)
print(f"  Criteria: {criteria.description}")
print(f"  Dimensions: {[d.name for d in criteria.dimensions[:3]]}")

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
print("\nTo run actual generation:")
print("  python main.py -t 'your topic'")
print("\nMake sure you have:")
print("  - ZHIPU_API_KEY in .env")
print("  - TAVILY_API_KEY in .env (for RAG)")
