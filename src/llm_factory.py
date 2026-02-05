"""
LLM Factory
统一封装 LLM 创建逻辑，支持多提供商
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI
from httpx import Timeout

from config.settings import settings

# 全局超时设置 (连接超时, 读取超时)
DEFAULT_TIMEOUT = Timeout(connect=10.0, read=60.0, write=10.0, pool=5.0)


def create_llm(model_name: Optional[str] = None, temperature: float = 0.7, timeout: Optional[Timeout] = None):
    """
    根据配置创建 LLM 实例
    
    Args:
        model_name: 模型名称，默认使用配置中的模型
        temperature: 温度参数
        timeout: 超时设置，默认 60秒读取超时
        
    Returns:
        ChatOpenAI 或 ChatZhipuAI 实例
    """
    model = model_name or settings.default_model
    provider = settings.llm_provider.lower()
    timeout = timeout or DEFAULT_TIMEOUT
    
    if provider == "zhipu":
        # 使用智谱 AI
        if not settings.zhipu_api_key:
            raise ValueError("Zhipu API Key 未配置。请在 .env 文件中设置 ZHIPU_API_KEY")
        
        return ChatZhipuAI(
            model=model,
            temperature=temperature,
            api_key=settings.zhipu_api_key,
            timeout=timeout
        )
    
    elif provider == "openai":
        # 使用 OpenAI
        if not settings.openai_api_key:
            raise ValueError("OpenAI API Key 未配置。请在 .env 文件中设置 OPENAI_API_KEY")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            timeout=timeout
        )
    
    else:
        raise ValueError(f"不支持的 LLM Provider: {provider}。请选择 'openai' 或 'zhipu'")


def get_default_model() -> str:
    """获取默认模型名称"""
    return settings.default_model


def get_critic_model() -> str:
    """获取 Critic 使用的模型名称"""
    return settings.critic_model
