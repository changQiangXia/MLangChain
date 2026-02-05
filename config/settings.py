"""
Configuration and Settings
管理模型配置、阈值设置、API Key 加载
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    应用配置类，自动从环境变量加载
    """
    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    azure_openai_api_key: str = Field(default="", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(default="", alias="AZURE_OPENAI_ENDPOINT")
    zhipu_api_key: str = Field(default="", alias="ZHIPU_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    
    # Provider: "openai" | "zhipu"
    llm_provider: str = Field(default="zhipu", alias="LLM_PROVIDER")
    
    # Model Configuration
    default_model: str = Field(default="glm-4-flash", alias="DEFAULT_MODEL")
    critic_model: str = Field(default="glm-4", alias="CRITIC_MODEL")
    
    # Quality Threshold (Critic 评分阈值) - 更严格
    quality_threshold: float = 8.5
    
    # Iteration Control
    max_iterations: int = 8  # 增加到8次，给更多优化机会
    recursion_limit: int = 50
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例（缓存）
    """
    return Settings()


# 全局配置实例
settings = get_settings()
