"""
数据工具函数
提供数据去重、多样性控制、质量过滤等功能
"""

import json
from typing import List
from difflib import SequenceMatcher


def calculate_similarity(text1: str, text2: str) -> float:
    """计算两段文本的相似度 (0-1)"""
    return SequenceMatcher(None, text1, text2).ratio()


def is_duplicate(
    new_data: dict,
    existing_data_list: List[dict],
    threshold: float = 0.85
) -> bool:
    """检查新数据是否与现有数据重复"""
    new_text = f"{new_data.get('instruction', '')} {new_data.get('output', '')}"
    
    for existing in existing_data_list:
        existing_text = f"{existing.get('instruction', '')} {existing.get('output', '')}"
        similarity = calculate_similarity(new_text, existing_text)
        if similarity >= threshold:
            return True
    
    return False


def filter_by_quality(data_list: List[dict], min_score: float = 8.0) -> List[dict]:
    """按质量评分过滤数据"""
    return [d for d in data_list if d.get("score", 0) >= min_score]


def deduplicate_dataset(data_list: List[dict], threshold: float = 0.85) -> List[dict]:
    """对整个数据集进行去重"""
    unique_data = []
    for data in data_list:
        if not is_duplicate(data, unique_data, threshold):
            unique_data.append(data)
    return unique_data


def filter_valid_data(data_list: List[dict]) -> List[dict]:
    """过滤掉无效数据（data为null或success为false且score为0的）"""
    valid = []
    for d in data_list:
        # 检查是否有有效数据
        data = d.get("data")
        score = d.get("score", 0)
        # 保留：有实际数据内容 或 有有效分数的
        if (data and isinstance(data, dict)) or score > 0:
            valid.append(d)
    return valid


def calculate_dataset_stats(data_list: List[dict]) -> dict:
    """计算数据集的统计信息"""
    if not data_list:
        return {}
    
    # 先过滤掉无效数据
    valid_data = filter_valid_data(data_list)
    
    if not valid_data:
        return {
            "total_count": len(data_list),
            "valid_count": 0,
            "invalid_count": len(data_list)
        }
    
    scores = [d.get("score", 0) for d in valid_data]
    output_lengths = []
    for d in valid_data:
        data = d.get("data")
        if data and isinstance(data, dict):
            output_lengths.append(len(data.get("output", "")))
        else:
            output_lengths.append(0)
    
    return {
        "total_count": len(data_list),
        "valid_count": len(valid_data),
        "invalid_count": len(data_list) - len(valid_data),
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "high_quality_count": sum(1 for s in scores if s >= 8.5),
        "avg_output_length": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
    }


def load_jsonl(filepath: str) -> List[dict]:
    """加载 JSONL 文件"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except FileNotFoundError:
        pass
    return data


def save_jsonl(data_list: List[dict], filepath: str):
    """保存为 JSONL 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for data in data_list:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
