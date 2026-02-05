"""
Tavily Search Tool Wrapper
封装 Tavily 搜索 API，用于获取外部验证信息
"""

from typing import List, Dict, Optional, Any
from tavily import TavilyClient
from config.settings import settings


class SearchTool:
    """
    Tavily 搜索工具封装类
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化搜索工具
        
        Args:
            api_key: Tavily API Key，如果不提供则从配置读取
        """
        self.api_key = api_key or settings.tavily_api_key
        if not self.api_key:
            raise ValueError("Tavily API Key 未配置。请在 .env 文件中设置 TAVILY_API_KEY")
        
        self.client = TavilyClient(api_key=self.api_key)
    
    def search(
        self, 
        query: str, 
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        执行搜索查询
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            search_depth: 搜索深度 ("basic" 或 "advanced")
            include_answer: 是否包含 AI 生成的摘要答案
            
        Returns:
            搜索结果列表，每个结果包含 title, content, url 等
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer
            )
            
            results = response.get("results", [])
            
            # 如果包含 AI 摘要，添加到结果中
            if include_answer and response.get("answer"):
                results.insert(0, {
                    "title": "AI Summary",
                    "content": response["answer"],
                    "url": "",
                    "score": 1.0
                })
            
            return results
            
        except Exception as e:
            print(f"[SearchTool Error] 搜索失败: {e}")
            return []
    
    def get_context_for_generation(
        self, 
        topic: str,
        max_results: int = 3
    ) -> str:
        """
        获取用于生成指令数据的上下文信息
        
        Args:
            topic: 主题
            max_results: 最大结果数
            
        Returns:
            格式化的上下文字符串
        """
        results = self.search(
            query=topic,
            max_results=max_results,
            search_depth="advanced"
        )
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            content = result.get("content", "")
            if content:
                context_parts.append(f"[{i}] {title}\n{content}")
        
        return "\n\n".join(context_parts)


# 便捷函数：快速搜索
def quick_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    快速搜索函数，无需实例化类
    """
    tool = SearchTool()
    return tool.search(query, max_results=max_results)
