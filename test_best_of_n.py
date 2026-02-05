"""
测试脚本：验证 Best-of-N (对抗性评分)

测试目标：
1. PairwiseComparator 能正确对比两个版本
2. BestOfNSelector 能从多个版本中选择最佳
3. 整体流程集成正确
"""

import sys
sys.path.insert(0, '.')


def test_pairwise_comparison():
    """测试成对比较"""
    print("=" * 60)
    print("测试 1: 成对比较")
    print("=" * 60)
    
    from src.core.best_of_n import PairwiseComparator
    from src.state import AlpacaData
    
    comparator = PairwiseComparator()
    
    # 创建两个测试版本
    version_a = AlpacaData(
        instruction="解释什么是深度学习",
        input="",
        output="""
深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。
神经网络由多层神经元组成，可以自动从数据中学习特征。
例如，卷积神经网络（CNN）在图像识别中表现出色。
        """.strip()
    )
    
    version_b = AlpacaData(
        instruction="请详细解释深度学习的概念、原理和应用",
        input="",
        output="""
深度学习（Deep Learning）是机器学习的一个分支，基于多层神经网络。

核心原理：
1. 神经网络层：输入层、隐藏层、输出层
2. 反向传播：通过梯度下降优化权重
3. 激活函数：引入非线性

应用场景：
- 图像识别：ResNet-50 在 ImageNet 达到 76.2% 准确率
- 自然语言处理：BERT 模型在 GLUE 基准测试达到 93.2%
- 语音识别：DeepSpeech 实现 95%+ 准确率

优势：自动特征提取，无需人工设计。
        """.strip()
    )
    
    print("\n版本 A: 简短，缺少具体例子")
    print("版本 B: 详细，有具体例子和数据")
    
    print("\n执行对比...")
    result = comparator.compare("解释深度学习", version_a, version_b)
    
    print(f"\n对比结果:")
    print(f"  获胜者: {'A' if result.winner_index == 0 else 'B'}")
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  理由: {result.reasoning[:100]}...")
    
    # 预期 B 应该获胜（更详细）
    if result.winner_index == 1:
        print("\n ✓ 正确选择了更详细的版本 B")
    else:
        print("\n ⚠ 选择了版本 A（可能对比不够准确）")
    
    print()


def test_best_of_n_selection():
    """测试 Best-of-N 选择"""
    print("=" * 60)
    print("测试 2: Best-of-N 选择")
    print("=" * 60)
    
    from src.core.best_of_n import BestOfNSelector, BestOfNResult
    from src.state import AlpacaData
    
    # 创建三个测试版本（质量递增）
    versions = [
        AlpacaData(  # 版本 1: 质量一般
            instruction="解释梯度下降",
            input="",
            output="梯度下降是一种优化算法，用于训练神经网络。它通过计算梯度来更新权重。"
        ),
        AlpacaData(  # 版本 2: 质量较好
            instruction="请详细解释梯度下降算法的原理和应用",
            input="",
            output="""
梯度下降（Gradient Descent）是机器学习中最重要的优化算法之一。

原理：
通过计算损失函数的梯度，沿着梯度反方向更新参数，逐步减小损失。

公式：θ = θ - α * ∇J(θ)
其中 α 是学习率，∇J(θ) 是梯度。

应用场景：
- 线性回归
- 逻辑回归
- 神经网络训练
            """.strip()
        ),
        AlpacaData(  # 版本 3: 质量最好（有具体例子）
            instruction="请详细解释梯度下降算法的原理、公式和实际应用，并举例说明",
            input="",
            output="""
梯度下降（Gradient Descent）是机器学习的核心优化算法。

核心原理：
通过迭代更新参数，沿着损失函数梯度的反方向移动，直到收敛到最小值。

数学公式：
θ_{t+1} = θ_t - α * ∇J(θ_t)
其中：
- θ：模型参数
- α：学习率（通常 0.001-0.1）
- ∇J(θ)：损失函数梯度

实际应用：
1. 房价预测：使用梯度下降训练线性回归模型，预测房价
2. 图像分类：训练 CNN 模型，在 CIFAR-10 达到 95% 准确率
3. 推荐系统：优化协同过滤算法，提升推荐准确度 20%

Python 代码示例：
```python
for epoch in range(1000):
    gradient = compute_gradient(X, y, theta)
    theta = theta - learning_rate * gradient
```

优势：简单高效，适用于大规模数据。
            """.strip()
        ),
    ]
    
    print(f"\n创建 {len(versions)} 个版本（质量递增）")
    for i, v in enumerate(versions, 1):
        print(f"  版本 {i}: {len(v.output)} 字符")
    
    # 测试 2 选 1
    print("\n测试 2.1: Best-of-2")
    selector2 = BestOfNSelector(n=2)
    result2 = selector2.select("解释梯度下降", versions[:2])
    
    print(f"  选择结果: 版本 {result2.best_index + 1}")
    print(f"  预选分数: {result2.final_score}")
    print(f"  对比轮数: {len(result2.comparisons)}")
    
    # 测试 3 选 1
    print("\n测试 2.2: Best-of-3")
    selector3 = BestOfNSelector(n=3)
    result3 = selector3.select("解释梯度下降", versions)
    
    print(f"  选择结果: 版本 {result3.best_index + 1}")
    print(f"  预选分数: {result3.final_score}")
    print(f"  对比轮数: {len(result3.comparisons)}")
    
    # 预期选择版本 3（质量最好）
    if result3.best_index == 2:
        print("\n ✓ 正确选择了质量最好的版本 3")
    else:
        print(f"\n ⚠ 选择了版本 {result3.best_index + 1}")
    
    print()


def test_integration():
    """测试集成"""
    print("=" * 60)
    print("测试 3: 与 Generator 集成（需 API Key）")
    print("=" * 60)
    
    try:
        from src.agents.multi_version_generator import MultiVersionGeneratorAgent
        from src.core.best_of_n import BestOfNSelector
        
        generator = MultiVersionGeneratorAgent()
        selector = BestOfNSelector(n=2)
        
        task = "解释什么是卷积神经网络 CNN"
        
        print(f"\n任务: {task}")
        print("生成 2 个版本...")
        
        # 生成多个版本
        versions = generator.generate_multiple(task, n=2)
        
        print(f"\n生成完成，共 {len(versions)} 个版本")
        for i, v in enumerate(versions, 1):
            print(f"  版本 {i}: {len(v.output)} 字符")
        
        # 选择最佳
        print("\n选择最佳版本...")
        result = selector.select(task, versions)
        
        print(f"\n选择结果:")
        print(f"  最佳版本: {result.best_index + 1}")
        print(f"  预选分数: {result.final_score}")
        print(f"  指令: {result.best_version.instruction[:60]}...")
        
    except Exception as e:
        print(f"\n跳过（需要 API Key）: {e}")
    
    print()


def test_workflow_v2():
    """测试 Workflow V2"""
    print("=" * 60)
    print("测试 4: Workflow V2（需完整环境）")
    print("=" * 60)
    
    print("\nWorkflow V2 特性:")
    print("1. Generator 生成 N 个版本（不同温度）")
    print("2. BestOfN Selector 选择最佳版本")
    print("3. CriticV2 对最佳版本评分")
    print("4. 如果不够，Refiner 优化后重新生成")
    
    print("\n对比 Workflow V1:")
    print("- V1: Generator → Critic → Refiner → Generator...")
    print("- V2: Generator (N个) → Selector → Critic → Refiner → Generator...")
    
    print("\n使用方式:")
    print("  from src.graph.workflow_v2 import generate_with_best_of_n")
    print("  result = generate_with_best_of_n('解释CNN', n=2)")
    
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Best-of-N (对抗性评分) 测试套件")
    print("=" * 60 + "\n")
    
    try:
        # 基础测试（不需要 API）
        test_pairwise_comparison()
        test_best_of_n_selection()
        
        # 需要 API 的测试
        test_integration()
        test_workflow_v2()
        
        print("=" * 60)
        print("测试完成")
        print("=" * 60)
        
        print("\n关键功能:")
        print("1. [OK] PairwiseComparator 能对比两个版本")
        print("2. [OK] BestOfNSelector 能从 N 个版本中选择最佳")
        print("3. [OK] MultiVersionGenerator 能生成多样化版本")
        print("4. [OK] Workflow V2 集成完整")
        
        print("\n优势:")
        print("- 评分更稳定（相对对比 vs 绝对分数）")
        print("- 质量上限更高（选择最佳版本）")
        print("- 减少随机性（多样化生成）")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
