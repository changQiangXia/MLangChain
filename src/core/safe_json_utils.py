"""
安全的 JSON 解析工具
处理 LLM 输出中的各种 JSON 格式问题
"""

import json
import re
from typing import Any, Optional


def clean_json_string(s: str) -> str:
    """
    清理 JSON 字符串中的非法字符
    """
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    
    # 移除控制字符（保留 \n, \t, \r）
    try:
        s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', s)
    except Exception:
        pass  # 如果失败，保留原字符串
    
    # 修复无效的转义序列
    try:
        s = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', s)
    except Exception:
        pass
    
    return s.strip()


def extract_json_from_markdown(content) -> Optional[str]:
    """
    从 Markdown 代码块中提取 JSON
    """
    # 确保是字符串
    if content is None:
        return None
    
    if not isinstance(content, str):
        if hasattr(content, 'content'):
            content = content.content
        elif hasattr(content, 'text'):
            content = content.text
        else:
            try:
                content = str(content)
            except Exception:
                return None
    
    # 匹配 ```json 或 ``` 包围的 JSON
    patterns = [
        r'```json\s*\n(.*?)\n```',
        r'```\s*\n(.*?)\n```',
        r'`(.*?)`',
    ]
    
    try:
        for pattern in patterns:
            try:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    return match.group(1).strip()
            except Exception:
                continue
    except Exception:
        pass
    
    # 尝试直接找 JSON 对象
    try:
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            return content[start:end+1]
    except Exception:
        pass
    
    return None


def safe_json_loads(
    s, 
    default: Any = None,
    max_retries: int = 3
) -> Any:
    """
    安全的 JSON 解析，带多重回退策略
    
    Args:
        s: 要解析的 JSON 字符串（可以是字符串或带有 content 属性的对象）
        default: 解析失败时的默认值
        max_retries: 最大重试次数
        
    Returns:
        解析后的 Python 对象，失败则返回 default
    """
    # 处理不同类型的输入
    if s is None:
        return default
    
    if not isinstance(s, str):
        # 尝试提取内容
        if hasattr(s, 'content'):
            s = s.content
        elif hasattr(s, 'text'):
            s = s.text
        else:
            s = str(s)
    
    if not s or not s.strip():
        return default
    
    # 策略 1: 直接解析
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # 策略 2: 从 Markdown 中提取
    extracted = extract_json_from_markdown(s)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
    
    # 策略 3: 清理后解析
    cleaned = clean_json_string(extracted if extracted else s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # 策略 4: 修复常见错误后解析
    # 修复尾随逗号
    fixed = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # 策略 5: 使用 ast.literal_eval（更安全）
    try:
        import ast
        return ast.literal_eval(fixed)
    except (ValueError, SyntaxError):
        pass
    
    # 策略 6: 尝试 eval（最后手段，但限制环境）
    try:
        # 只允许字面值
        result = eval(fixed, {"__builtins__": {}}, {})
        if isinstance(result, (dict, list, str, int, float, bool, type(None))):
            return result
    except:
        pass
    
    return default


def sanitize_for_json(text: str) -> str:
    """
    清理文本以确保可以安全地序列化为 JSON
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 转义反斜杠（如果未转义）
    # 但保留已经正确转义的序列
    text = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)
    
    return text


# 便捷函数
parse_json = safe_json_loads
