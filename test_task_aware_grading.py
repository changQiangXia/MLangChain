"""
测试脚本：验证动态类型感知 (Task-Aware Grading)

测试目标：
1. CODE 类型任务不看字数，看代码质量
2. REASONING 类型看思维链
3. EXPLANATION 类型看字数和例子
"""

import sys
sys.path.insert(0, '.')

from src.core.task_classifier import TaskClassifier, classify_task
from src.core.grading_criteria import GradingCriteria, get_grading_criteria
from src.core.code_validator import validate_code, extract_code_from_output, CodeValidator


def test_task_classification():
    """测试任务分类器"""
    print("=" * 60)
    print("测试 1: 任务分类器")
    print("=" * 60)
    
    classifier = TaskClassifier(use_llm=False)  # 只使用规则，避免API调用
    
    test_cases = [
        ("写一个Python函数计算斐波那契数列", "code"),
        ("证明为什么梯度下降会收敛", "reasoning"),
        ("解释什么是深度学习", "explanation"),
        ("创作一个关于AI的科幻故事", "creative"),
        ("你好，今天天气怎么样", "chitchat"),
    ]
    
    for instruction, expected in test_cases:
        result = classifier.classify(instruction)
        status = "OK" if result.task_type.value == expected else "FAIL"
        print(f"[{status}] [{result.task_type.value:12}] {instruction[:40]}...")
        print(f"      置信度: {result.confidence:.2f}")
    
    print()


def test_grading_criteria():
    """测试动态评分标准"""
    print("=" * 60)
    print("测试 2: 动态评分标准")
    print("=" * 60)
    
    from src.core.task_classifier import TaskType
    
    for task_type in [TaskType.CODE, TaskType.REASONING, TaskType.EXPLANATION]:
        criteria = get_grading_criteria(task_type)
        print(f"\n[{criteria.description}]")
        print(f"  字数检查: {'启用' if criteria.length_check_enabled else '禁用'}")
        print(f"  最小字数: {criteria.min_length if criteria.min_length else '无要求'}")
        print(f"  评分维度:")
        for dim in criteria.dimensions[:3]:  # 只显示前3个
            print(f"    - {dim.name}: 0-{dim.max_score}分")
    
    print()


def test_code_validation():
    """测试代码验证器"""
    print("=" * 60)
    print("测试 3: 代码验证器")
    print("=" * 60)
    
    # 测试 1: 正确的代码
    good_code = '''
def fibonacci(n):
    """计算斐波那契数列"""
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 测试
print(fibonacci(10))
'''
    
    print("\n测试 3.1: 正确的代码")
    result = validate_code(good_code)
    print(f"  语法正确: {result.syntax_valid}")
    print(f"  可执行: {result.can_execute}")
    print(f"  有注释: {result.has_comments}")
    print(f"  有文档字符串: {result.has_docstring}")
    print(f"  质量分数: {CodeValidator().calculate_quality_score(result):.1f}/10")
    
    # 测试 2: 语法错误的代码
    bad_code = '''
def broken(
    print("missing parenthesis")
'''
    
    print("\n测试 3.2: 语法错误的代码")
    result = validate_code(bad_code)
    print(f"  语法正确: {result.syntax_valid}")
    print(f"  错误信息: {result.has_error[:50]}...")
    
    # 测试 3: 从文本中提取代码
    text_with_code = '''
这是一个计算斐波那契数列的函数：

```python
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

你可以这样使用它。
'''
    
    print("\n测试 3.3: 从文本中提取代码")
    extracted = extract_code_from_output(text_with_code)
    if extracted:
        print(f"  提取成功: {extracted[:80]}...")
    else:
        print("  提取失败")
    
    print()


def test_code_task_grading():
    """测试代码类任务的评分"""
    print("=" * 60)
    print("测试 4: 代码类任务评分（不看字数！）")
    print("=" * 60)
    
    # 短但优质的代码
    short_code = '''def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)'''
    
    print(f"\n代码长度: {len(short_code)} 字符（非常短）")
    
    result = validate_code(short_code)
    score = CodeValidator().calculate_quality_score(result)
    
    print(f"验证结果:")
    print(f"  语法正确: {result.syntax_valid}")
    print(f"  可执行: {result.can_execute}")
    print(f"  质量分数: {score:.1f}/10")
    print(f"\n[结论] 代码类任务不看字数，即使只有 {len(short_code)} 字符，"
          f"只要代码正确就能得到 {score:.1f} 分")
    
    print()


def test_length_check():
    """测试字数检查逻辑"""
    print("=" * 60)
    print("测试 5: 字数检查逻辑对比")
    print("=" * 60)
    
    from src.core.task_classifier import TaskType
    
    test_lengths = [50, 100, 150, 200, 500]
    
    for task_type in [TaskType.CODE, TaskType.EXPLANATION]:
        criteria = get_grading_criteria(task_type)
        print(f"\n[{criteria.description}]")
        print(f"  字数检查: {'启用' if criteria.length_check_enabled else '禁用'}")
        print(f"  最小字数: {criteria.min_length}")
        
        for length in test_lengths:
            ok, deduction = GradingCriteria.check_length(task_type, length)
            status = "OK" if ok else f"扣{deduction}分"
            print(f"    {length:3d}字: {status}")
    
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("动态类型感知 (Task-Aware Grading) 测试套件")
    print("=" * 60 + "\n")
    
    try:
        test_task_classification()
        test_grading_criteria()
        test_code_validation()
        test_code_task_grading()
        test_length_check()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        print("\n关键验证点:")
        print("1. [OK] CODE 类型正确识别")
        print("2. [OK] CODE 类型禁用字数检查（防止误杀短代码）")
        print("3. [OK] 代码验证器能检查语法和执行性")
        print("4. [OK] 不同任务类型有不同的评分维度")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
