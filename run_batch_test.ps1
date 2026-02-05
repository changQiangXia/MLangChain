# 批量处理测试脚本 - 测试4: 大规模生成
# 生成20条AI/深度学习相关的高质量指令数据

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 批量处理大规模生成测试" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 1. 清空旧数据
Write-Host ""
Write-Host "[1/4] 清空旧数据..." -ForegroundColor Yellow
if (Test-Path "data/output/large_results.jsonl") {
    Clear-Content -Path "data/output/large_results.jsonl" -Force
    Write-Host "✅ 已清空 data/output/large_results.jsonl" -ForegroundColor Green
} else {
    New-Item -ItemType File -Path "data/output/large_results.jsonl" -Force | Out-Null
    Write-Host "✅ 创建新文件 data/output/large_results.jsonl" -ForegroundColor Green
}

# 2. 创建大任务集（20个主题）- 使用Python生成避免编码问题
Write-Host ""
Write-Host "[2/4] 创建20个AI/深度学习主题..." -ForegroundColor Yellow

python -c @"
topics = [
    "解释什么是神经网络",
    "什么是卷积神经网络CNN", 
    "什么是循环神经网络RNN",
    "解释Transformer架构的原理",
    "什么是注意力机制Attention",
    "什么是BERT模型",
    "什么是GPT模型",
    "对比CNN和RNN的优缺点",
    "什么是梯度下降算法",
    "什么是反向传播算法",
    "什么是激活函数及其作用",
    "什么是过拟合和如何解决",
    "什么是欠拟合和如何解决",
    "什么是正则化L1和L2",
    "什么是Dropout技术",
    "什么是批量归一化BatchNorm",
    "什么是学习率调度策略",
    "什么是损失函数CrossEntropy",
    "什么是优化器Adam和SGD",
    "什么是迁移学习和微调"
]

with open('data/input/large_batch.txt', 'w', encoding='utf-8') as f:
    for t in topics:
        f.write(t + '\n')

print('已创建20个主题')
for i, t in enumerate(topics, 1):
    print(f'{i}. {t}')
"@

Write-Host ""
Write-Host "✅ 已创建 data/input/large_batch.txt" -ForegroundColor Green

# 3. 运行批量处理（4并发）
Write-Host ""
Write-Host "[3/4] 开始批量生成 (4并发，预计5-10分钟)..." -ForegroundColor Yellow
Write-Host "⚡ 使用4个并发线程同时处理..." -ForegroundColor Magenta

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    python main.py -i data/input/large_batch.txt -o data/output/large_results.jsonl -w 4
    $stopwatch.Stop()
    
    Write-Host ""
    Write-Host "✅ 批量处理完成！耗时: $($stopwatch.Elapsed.ToString('mm\:ss'))" -ForegroundColor Green
}
catch {
    $stopwatch.Stop()
    Write-Host ""
    Write-Host "❌ 批量处理失败: $_" -ForegroundColor Red
    exit 1
}

# 4. 分析结果
Write-Host ""
Write-Host "[4/4] 分析生成结果..." -ForegroundColor Yellow
python main.py --analyze data/output/large_results.jsonl

# 5. 额外统计
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📊 详细统计报告" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python -c @"
import json
from collections import Counter

# 读取数据
with open('data/output/large_results.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]

if not data:
    print('❌ 没有数据')
    exit()

# 基础统计
total = len(data)
success = sum(1 for d in data if d.get('success'))
failed = total - success
scores = [d.get('score', 0) for d in data]

print(f'总任务数: {total}')
print(f'成功生成: {success} ({success/total*100:.1f}%)')
print(f'失败/低质量: {failed} ({failed/total*100:.1f}%)')
print('')
print(f'分数分布:')
print(f'  9.0-10分 (卓越): {sum(1 for s in scores if s >= 9.0)} 条')
print(f'  8.5-8.9分 (优秀): {sum(1 for s in scores if 8.5 <= s < 9.0)} 条')
print(f'  8.0-8.4分 (良好): {sum(1 for s in scores if 8.0 <= s < 8.5)} 条')
print(f'  7.0-7.9分 (一般): {sum(1 for s in scores if 7.0 <= s < 8.0)} 条')
print(f'  <7分 (较差): {sum(1 for s in scores if s < 7.0)} 条')

# 迭代次数分布
iterations = [d.get('iterations', 0) for d in data]
print('')
print(f'迭代次数分布:')
iter_counter = Counter(iterations)
for it in sorted(iter_counter.keys()):
    print(f'  {it}次迭代: {iter_counter[it]} 条')

# 平均输出长度
output_lengths = []
for d in data:
    if d.get('data') and d['data'].get('output'):
        output_lengths.append(len(d['data']['output']))

if output_lengths:
    avg_len = sum(output_lengths) / len(output_lengths)
    print('')
    print(f'输出长度统计:')
    print(f'  平均: {avg_len:.0f} 字符')
    print(f'  最长: {max(output_lengths)} 字符')
    print(f'  最短: {min(output_lengths)} 字符')
"@

# 6. 显示示例
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📝 示例数据预览（第一条）" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

python -c @"
import json

with open('data/output/large_results.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    if lines:
        d = json.loads(lines[0])
        print(f'📌 指令: {d[\"data\"][\"instruction\"]}')
        print('')
        print(f'⭐ 评分: {d[\"score\"]}/10')
        print(f'🔄 迭代: {d[\"iterations\"]} 次')
        print('')
        print(f'📤 输出预览:')
        output = d['data']['output']
        if len(output) > 400:
            print(output[:400] + '...')
        else:
            print(output)
        print('')
        print(f'💬 评审反馈: {d[\"feedback\"][:150]}...')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "🎉 测试完成！结果保存在:" -ForegroundColor Green
Write-Host "   data/output/large_results.jsonl" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Green

Write-Host ""
Write-Host "后续操作建议:" -ForegroundColor Gray
Write-Host "  查看完整结果: python main.py --analyze data/output/large_results.jsonl" -ForegroundColor Gray
