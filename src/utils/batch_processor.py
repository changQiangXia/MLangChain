"""
批量处理器
提供并发批量生成功能
"""

import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.graph.workflow import generate_high_quality_data
from src.utils.data_utils import is_duplicate, calculate_dataset_stats


class BatchProcessor:
    """批量数据生成处理器"""
    
    def __init__(
        self,
        max_workers: int = 2,
        similarity_threshold: float = 0.85,
        min_quality_score: float = 8.0,
        enable_deduplication: bool = True
    ):
        self.max_workers = max_workers
        self.similarity_threshold = similarity_threshold
        self.min_quality_score = min_quality_score
        self.enable_deduplication = enable_deduplication
        self.results: List[dict] = []
        self.failed_tasks: List[str] = []
    
    def process_single_task(self, task: str) -> dict:
        """处理单个任务"""
        try:
            result = generate_high_quality_data(task)
            return result
        except Exception as e:
            print(f"[Error] 任务 '{task}' 失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task,
                "score": 0,
                "data": None
            }
    
    def process_batch(self, tasks: List[str]) -> List[dict]:
        """批量处理任务"""
        self.results = []
        self.failed_tasks = []
        
        print(f"\n[BatchProcessor] 开始处理 {len(tasks)} 个任务")
        print(f"- 最大并发数: {self.max_workers}")
        print(f"- 去重阈值: {self.similarity_threshold}")
        print(f"- 最低质量分: {self.min_quality_score}\n")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.process_single_task, task): task 
                for task in tasks
            }
            
            for i, future in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[future]
                print(f"[{i}/{len(tasks)}] 处理: {task[:50]}...")
                
                try:
                    result = future.result()
                    
                    if not result.get("success"):
                        self.failed_tasks.append(task)
                        continue
                    
                    if result.get("score", 0) < self.min_quality_score:
                        print(f"  [Filter] 质量分不足 ({result.get('score')})")
                        continue
                    
                    if self.enable_deduplication and result.get("data"):
                        existing_data = [r.get("data") for r in self.results if r.get("data")]
                        if is_duplicate(result["data"], existing_data, self.similarity_threshold):
                            print(f"  [Filter] 重复数据")
                            continue
                    
                    self.results.append(result)
                    print(f"  ✅ 完成 (分数: {result.get('score')})")
                    
                except Exception as e:
                    print(f"  [Error] 异常: {e}")
                    self.failed_tasks.append(task)
        
        # 打印统计
        stats = calculate_dataset_stats(self.results)
        print(f"\n[BatchProcessor] 处理完成")
        print(f"- 成功: {len(self.results)}")
        print(f"- 失败: {len(self.failed_tasks)}")
        if stats:
            print(f"- 平均质量分: {stats.get('avg_score', 0):.2f}")
        
        return self.results
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in self.results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        print(f"\n[BatchProcessor] 结果已保存到: {output_path}")
    
    def get_failed_tasks(self) -> List[str]:
        """获取失败的任务列表"""
        return self.failed_tasks
