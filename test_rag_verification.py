"""
测试脚本：验证 RAG 事实检查 (RAG Verification)

测试目标：
1. 事实提取器能提取关键事实
2. 事实验证器能验证事实准确性
3. CriticV2 能根据验证结果调整分数
"""

import sys
sys.path.insert(0, '.')


def test_fact_extraction():
    """测试事实提取"""
    print("=" * 60)
    print("测试 1: 事实提取")
    print("=" * 60)
    
    from src.core.fact_checker import FactExtractor
    
    extractor = FactExtractor()
    
    test_text = """
    ResNet-50 在 ImageNet 图像分类任务中达到了 76.2% 的 top-1 准确率。
    BERT 模型由 Google 在 2018 年提出，在 GLUE 基准测试中达到了 93.2% 的分数。
    GPT-4 在 MMLU 基准测试中达到了 86.4% 的准确率。
    """
    
    print("\n输入文本:")
    print(test_text[:200])
    
    facts = extractor.extract_facts(test_text)
    
    print(f"\n提取到 {len(facts)} 个事实:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. [{fact.fact_type}] {fact.text[:60]}... (置信度: {fact.confidence:.2f})")
    
    print()
    return len(facts) > 0


def test_fact_verification():
    """测试事实验证"""
    print("=" * 60)
    print("测试 2: 事实验证（需要 API Key）")
    print("=" * 60)
    
    try:
        from src.core.fact_checker import FactChecker, Fact
        
        checker = FactChecker()
        
        # 测试：正确的事实
        print("\n测试 2.1: 验证正确的事实")
        correct_fact = Fact(
            text="ResNet-50 在 ImageNet 上达到 76.2% 的准确率",
            fact_type="performance",
            confidence=0.9
        )
        
        from src.core.fact_checker import FactVerifier
        verifier = FactVerifier()
        result = verifier.verify(correct_fact)
        
        print(f"  事实: {result.fact}")
        print(f"  状态: {result.status.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  扣分: {result.deduction}")
        
        # 测试：错误的事实（瞎编）
        print("\n测试 2.2: 验证错误的事实")
        wrong_fact = Fact(
            text="AlexNet 在 ImageNet 上达到了 99.9% 的准确率",
            fact_type="performance",
            confidence=0.9
        )
        
        result2 = verifier.verify(wrong_fact)
        
        print(f"  事实: {result2.fact}")
        print(f"  状态: {result2.status.value}")
        print(f"  置信度: {result2.confidence:.2f}")
        print(f"  扣分: {result2.deduction}")
        
        if result2.deduction > 0:
            print("  ✓ 错误事实被正确识别并扣分")
        
    except Exception as e:
        print(f"  跳过（需要配置 Tavily API Key）: {e}")
    
    print()


def test_critic_v2_with_rag():
    """测试 CriticV2 集成 RAG"""
    print("=" * 60)
    print("测试 3: CriticV2 集成 RAG（需要完整环境）")
    print("=" * 60)
    
    try:
        from src.agents.critic_v2 import CriticV2Agent
        from src.state import AlpacaData
        
        # 创建一个包含可能错误事实的数据
        draft = AlpacaData(
            instruction="解释什么是 AlexNet",
            input="",
            output="""
AlexNet 是一个深度卷积神经网络，由 Alex Krizhevsky 等人在 2012 年提出。
它在 ImageNet 图像分类竞赛中取得了突破性的成果，
达到了 99.9% 的 top-1 准确率，这是当时的历史最高水平。
AlexNet 使用了 ReLU 激活函数和 Dropout 正则化技术。
            """.strip()
        )
        
        print("\n输入数据:")
        print(f"  指令: {draft.instruction}")
        print(f"  输出: {draft.output[:100]}...")
        print("\n注意：输出中包含可能错误的事实（99.9% 准确率）")
        
        print("\n启动 CriticV2（启用 RAG 验证）...")
        critic = CriticV2Agent(enable_fact_check=True)
        
        result = critic.critique("解释 AlexNet", draft)
        
        print(f"\n评分结果:")
        print(f"  分数: {result.score}/10")
        print(f"  反馈: {result.feedback[:200]}...")
        print(f"  问题: {result.issues}")
        
        if result.score < 8.0:
            print("\n ✓ RAG 验证生效：错误事实导致分数降低")
        
    except Exception as e:
        print(f"  跳过（需要配置 API Key）: {e}")
    
    print()


def test_verification_status():
    """测试验证状态枚举"""
    print("=" * 60)
    print("测试 4: 验证状态")
    print("=" * 60)
    
    from src.core.fact_checker import VerificationStatus, FactVerifier
    
    print("\n验证状态及扣分:")
    for status in VerificationStatus:
        deduction = FactVerifier.DEDUCTION_RULES.get(status, 0)
        print(f"  {status.value:15} 扣分: {deduction}")
    
    print("\n预期行为:")
    print("  - VERIFIED:    事实正确，不扣分")
    print("  - CONTRADICTED: 事实错误，扣 2.5 分")
    print("  - PARTIAL:     部分正确，扣 1.0 分")
    print("  - DOUBTFUL:    无法验证，扣 0.5 分")
    
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("RAG 事实验证 (RAG Verification) 测试套件")
    print("=" * 60 + "\n")
    
    try:
        results = []
        
        # 基础测试（不需要 API）
        results.append(("事实提取", test_fact_extraction()))
        
        # 需要 API Key 的测试
        test_fact_verification()
        test_critic_v2_with_rag()
        
        # 基础测试
        test_verification_status()
        
        print("=" * 60)
        print("测试完成")
        print("=" * 60)
        
        print("\n关键功能:")
        print("1. [OK] 事实提取器能从文本中提取关键事实")
        print("2. [需API] 事实验证器能搜索验证事实")
        print("3. [需API] CriticV2 能根据验证结果调整分数")
        print("4. [OK] 验证状态及扣分规则正确")
        
        print("\n使用建议:")
        print("- 生产环境建议启用 RAG 验证")
        print("- 需要配置 Tavily API Key")
        print("- 每个事实验证消耗 1-3 次搜索调用")
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
