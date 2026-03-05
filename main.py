"""
Main Entry Point
程序入口：构建图并启动
"""

import json
import os
import argparse
from typing import Optional

from src.graph.workflow_v2 import run_workflow_v2 as run_workflow, generate_with_best_of_n as generate_high_quality_data
from src.utils.batch_processor import BatchProcessor
from src.utils.data_utils import load_jsonl, calculate_dataset_stats
from src.agents.code_generator import CodeGeneratorAgent


def print_result(result: dict):
    """格式化输出结果"""
    print("\n" + "=" * 60)
    print("📊 生成结果")
    print("=" * 60)
    
    if result.get("error"):
        print(f"❌ 错误: {result['error']}")
        return
    
    data = result.get("data")
    if data:
        print(f"\n📝 指令: {data['instruction']}")
        if data.get('input'):
            print(f"\n📥 输入: {data['input']}")
        print(f"\n📤 输出:\n{data['output'][:500]}...")
    
    print(f"\n" + "-" * 60)
    status = "✅ 成功" if result.get("success") else "⚠️ 未完成"
    print(f"状态: {status}")
    print(f"⭐ 质量评分: {result.get('score', 0)}/10")
    print(f"🔄 迭代次数: {result.get('iterations', 0)}")
    if result.get("feedback"):
        print(f"\n💬 评审反馈: {result['feedback'][:150]}...")
    print("-" * 60)


def save_to_jsonl(data: dict, filename: str):
    """保存结果到 JSONL 文件"""
    try:
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        print(f"✅ 结果已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")


def interactive_mode():
    """交互式模式"""
    print("\n" + "=" * 60)
    print("🚀 LangGraph 多智能体数据合成系统")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'code:' 前缀生成带代码示例的数据\n")
    
    while True:
        try:
            task = input("📝 请输入任务主题/描述: ").strip()
            
            if task.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            if not task:
                print("⚠️  请输入有效的主题\n")
                continue
            
            # 检查是否需要代码示例
            generate_code = False
            if task.lower().startswith('code:'):
                generate_code = True
                task = task[5:].strip()
                print(f"\n💻 代码模式：将生成包含Python代码示例的数据")
            
            print(f"\n⏳ 正在处理: {task}...")
            
            if generate_code:
                # 使用代码生成器
                from src.agents.code_generator import CodeGeneratorAgent
                agent = CodeGeneratorAgent()
                data = agent.generate_with_code(task)
                result = {
                    "success": True,
                    "score": 8.5,  # 代码数据默认给8.5分
                    "iterations": 1,
                    "data": data.to_dict(),
                    "feedback": "包含Python代码示例的指令数据",
                    "metadata": {"mode": "code_generation"}
                }
            else:
                result = generate_high_quality_data(task)
            
            print_result(result)
            
            save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_to_jsonl(result, "data/output/data.jsonl")
            
            print("\n" + "-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")


def batch_mode(input_file: str, output_file: str, max_workers: int = 2):
    """批量处理模式（使用 BatchProcessor）"""
    print(f"\n📂 批量处理模式")
    
    # 读取任务
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            tasks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 使用 BatchProcessor 处理
    processor = BatchProcessor(
        max_workers=max_workers,
        similarity_threshold=0.85,
        min_quality_score=8.0
    )
    
    results = processor.process_batch(tasks)
    processor.save_results(output_file)
    
    # 显示统计
    if results:
        stats = calculate_dataset_stats(results)
        print(f"\n📊 数据集统计:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


def analyze_mode(input_file: str):
    """分析已有数据集"""
    print(f"\n📊 分析数据集: {input_file}")
    
    data = load_jsonl(input_file)
    if not data:
        print("❌ 文件为空或不存在")
        return
    
    stats = calculate_dataset_stats(data)
    
    print(f"\n统计信息:")
    print(f"  总数据量: {stats['total_count']}")
    if stats.get('invalid_count', 0) > 0:
        print(f"  ⚠️  无效数据: {stats['invalid_count']} (已过滤)")
    print(f"  有效数据: {stats.get('valid_count', stats['total_count'])}")
    print(f"  平均分数: {stats['avg_score']:.2f}")
    print(f"  最高分数: {stats['max_score']:.2f}")
    print(f"  最低分数: {stats['min_score']:.2f}")
    print(f"  高质量数据 (>=8.5): {stats['high_quality_count']}")
    print(f"  平均输出长度: {stats['avg_output_length']:.0f} 字符")


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph 多智能体数据合成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式模式
  python main.py
  
  # 单次任务
  python main.py -t "解释什么是深度学习"
  
  # 批量处理
  python main.py -i data/input/topics.txt -o data/output/results.jsonl
  
  # 分析数据集
  python main.py --analyze data/output/data.jsonl
        """
    )
    
    parser.add_argument("-i", "--input", type=str, help="输入文件路径")
    parser.add_argument("-o", "--output", type=str, default="data/output/data.jsonl")
    parser.add_argument("-t", "--task", type=str, help="单次任务主题")
    parser.add_argument("-w", "--workers", type=int, default=2, help="并发数 (默认: 2)")
    parser.add_argument("--analyze", type=str, help="分析数据集文件")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_mode(args.analyze)
    elif args.task:
        result = generate_high_quality_data(args.task)
        print_result(result)
        save_to_jsonl(result, args.output)
    elif args.input:
        batch_mode(args.input, args.output, args.workers)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
