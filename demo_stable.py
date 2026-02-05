"""
稳定版演示 - 禁用可能出问题的功能
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Stable Demo - 禁用不稳定功能")
print("=" * 60)

# 使用简化配置
from src.agents.critic_v2 import CriticV2Agent
from src.agents.generator import GeneratorAgent
from src.state import AlpacaData

# 测试任务
tasks = [
    "写一个Python函数计算斐波那契数列",
    "解释什么是深度学习",
]

print("\n配置:")
print("- RAG 验证: 已放宽（矛盾扣1.5分，非2.5分）")
print("- 复杂度评估: 失败时回退到启发式")
print("- 评分阈值: 8.0（更易达到）")

for task in tasks:
    print(f"\n{'='*60}")
    print(f"任务: {task}")
    print('='*60)
    
    # 生成
    print("\n[1] 生成...")
    generator = GeneratorAgent()
    draft = generator.generate(task)
    print(f"  指令: {draft.instruction[:50]}...")
    print(f"  输出长度: {len(draft.output)} 字符")
    
    # 评审（使用修复后的配置）
    print("\n[2] 评审...")
    critic = CriticV2Agent(
        enable_fact_check=True,      # 启用但扣分更宽松
        enable_complexity_check=True  # 启用但失败时回退
    )
    result = critic.critique(task, draft)
    
    print(f"  评分: {result.score}/10")
    print(f"  反馈: {result.feedback[:100]}...")
    
    if result.score >= 8.0:
        print("\n  ✅ 质量达标！")
    else:
        print("\n  ⚠️  需要优化")

print("\n" + "=" * 60)
print("演示完成！")
print("=" * 60)
print("\n建议:")
print("1. RAG 验证有效但严格，已调整扣分标准")
print("2. 如遇 500 错误，请稍后重试（智谱服务器问题）")
print("3. 可临时禁用 RAG: CriticV2Agent(enable_fact_check=False)")
