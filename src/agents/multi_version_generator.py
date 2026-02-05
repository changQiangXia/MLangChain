"""
Multi-Version Generator
多版本生成器

生成 N 个不同版本，用于 Best-of-N 选择
"""

from typing import List, Optional
from src.agents.generator import GeneratorAgent
from src.state import AlpacaData


class MultiVersionGeneratorAgent(GeneratorAgent):
    """
    多版本生成器 Agent
    
    使用不同温度参数生成多样化的版本
    """
    
    def generate_multiple(
        self, 
        task_description: str,
        n: int = 2,
        search_results: Optional[list] = None
    ) -> List[AlpacaData]:
        """
        生成 N 个版本
        
        Args:
            task_description: 任务描述
            n: 版本数量（2 或 3）
            search_results: 搜索结果
            
        Returns:
            List[AlpacaData]: N 个版本
        """
        print(f"\n[MultiVersionGenerator] 生成 {n} 个多样化版本...")
        
        # 使用不同温度生成多样化版本
        temperatures = [0.5, 0.7, 0.9][:n]
        versions = []
        
        for i, temp in enumerate(temperatures, 1):
            print(f"[MultiVersionGenerator] 版本 {i}/{n} (temperature={temp})...")
            
            # 临时修改温度
            original_temp = self.llm.temperature
            self.llm.temperature = temp
            
            try:
                version = self.generate(task_description, search_results)
                versions.append(version)
                print(f"[MultiVersionGenerator] 版本 {i} 完成: {version.instruction[:50]}...")
            except Exception as e:
                print(f"[MultiVersionGenerator] 版本 {i} 失败: {e}")
                if versions:
                    # 如果失败，复制上一个版本
                    versions.append(versions[-1])
                else:
                    raise
            finally:
                # 恢复温度
                self.llm.temperature = original_temp
        
        print(f"[MultiVersionGenerator] 完成，共 {len(versions)} 个版本")
        return versions
