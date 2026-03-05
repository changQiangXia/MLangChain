"""
JSON 工具函数
解决 JSON 解析中的转义字符和格式问题
"""

import json
import re


def clean_json_string(text: str) -> str:
    """
    清理 JSON 字符串中的非法字符
    
    解决：
    - Invalid \escape
    - 控制字符
    - 非法 Unicode
    """
    if not text:
        return text
    
    # 1. 移除或替换控制字符
    # 保留 \n, \t, \" 等合法转义，移除其他控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # 2. 修复非法的反斜杠转义
    # 将单个反斜杠（非法）替换为双反斜杠（合法）
    # 但保留合法的转义序列: \n, \t, \", \\, \/, \b, \f, \r, \uXXXX
    
    # 先保护合法转义序列
    protected = []
    legal_escapes = r'\n\t\"\\\/\b\f\r'
    
    def protect_legal(match):
        protected.append(match.group(0))
        return f"<<PROTECTED_{len(protected)-1}>>"
    
    # 保护合法转义
    text = re.sub(r'\\[nrtbf\\/\"]', protect_legal, text)
    # 保护 Unicode 转义
    text = re.sub(r'\\u[0-9a-fA-F]{4}', protect_legal, text)
    
    # 3. 将剩余的单反斜杠替换为双反斜杠
    text = text.replace('\\', '\\\\')
    
    # 4. 恢复保护的合法转义
    for i, seq in enumerate(protected):
        text = text.replace(f"<<PROTECTED_{i}>>", seq)
    
    # 5. 修复多行字符串（JSON 不允许）
    # 将未转义的换行替换为 \n
    text = re.sub(r'(?<!\\)\n', '\\n', text)
    text = re.sub(r'(?<!\\)\r', '\\r', text)
    text = re.sub(r'(?<!\\)\t', '\\t', text)
    
    return text


def safe_json_loads(text: str, default=None):
    """
    安全地解析 JSON
    
    Args:
        text: JSON 字符串
        default: 解析失败时的默认值
        
    Returns:
        解析后的 Python 对象，或 default
    """
    if not text:
        return default
    
    try:
        # 首先尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 清理后重试
    try:
        cleaned = clean_json_string(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON 块（从 Markdown 代码块）
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            # 尝试找到最外层的 {}
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end+1]
            else:
                return default
        
        # 清理并解析
        cleaned = clean_json_string(json_str)
        return json.loads(cleaned)
    except Exception:
        return default


def sanitize_for_json(text: str) -> str:
    """
    将文本清理为适合放入 JSON 的格式
    
    用于清理 LLM 生成的 output 字段
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    
    # 替换非法控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    # 确保反斜杠正确转义
    text = text.replace('\\', '\\\\')
    
    # 确保引号正确转义
    text = text.replace('"', '\\"')
    
    # 确保换行正确转义
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    
    return text
