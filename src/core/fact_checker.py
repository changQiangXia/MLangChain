"""
Fact Checker
事实验证器 - 解决"回声室效应" (Echo Chamber Effect)

核心功能：
1. 从文本中提取可验证的事实陈述
2. 使用搜索工具验证事实
3. 对比判定事实是否准确
4. 生成验证报告

引用论文：RARR: Researching and Revising What Language Models Say (Gao et al., 2023)
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.tools.search_tool import SearchTool
from src.llm_factory import create_llm


class VerificationStatus(Enum):
    """验证状态"""
    VERIFIED = "verified"           # 已验证（搜索结果支持）
    CONTRADICTED = "contradicted"   # 矛盾（搜索结果反对）
    DOUBTFUL = "doubtful"           # 存疑（无法验证）
    PARTIAL = "partial"             # 部分支持


@dataclass
class Fact:
    """事实陈述"""
    text: str                       # 原始文本
    fact_type: str                  # 事实类型（数值、实体、关系等）
    confidence: float               # 提取置信度


@dataclass
class VerificationResult:
    """单个事实的验证结果"""
    fact: str                       # 事实陈述
    status: VerificationStatus      # 验证状态
    evidence: List[Dict]            # 搜索证据
    confidence: float               # 验证置信度
    deduction: float                # 扣分


@dataclass
class FactCheckReport:
    """事实验证报告"""
    facts: List[VerificationResult] # 所有事实的验证结果
    total_deduction: float          # 总扣分
    verified_count: int             # 已验证数量
    contradicted_count: int         # 矛盾数量
    doubtful_count: int             # 存疑数量
    summary: str                    # 总结


class FactExtractor:
    """
    事实提取器
    
    从文本中提取可验证的事实陈述
    """
    
    # 正则模式：提取可能的事实
    FACT_PATTERNS = [
        # 数值 + 单位
        r'\d+(?:\.\d+)?\s*(?:%|percent|百分比|倍|倍率)',
        # 年份 + 事件
        r'(?:19|20)\d{2}\s*年?\s*[\u4e00-\u9fa5]{2,10}',
        # 实体 + 属性
        r'(?:ResNet|BERT|GPT|CNN|RNN|Transformer)[-\w]*\s+(?:达到|实现|获得|在)',
        # 人名/公司 + 成就
        r'(?:Google|OpenAI|Microsoft|Meta|Apple|Tesla|DeepMind)[\u4e00-\u9fa5\w\s]{5,50}',
        # 技术 + 性能指标
        r'(?:准确|精确|召回|F1|AUC|BLEU|ROUGE)[率率]?\s*(?:达到|为|是)?\s*\d+',
    ]
    
    def __init__(self):
        self.llm = create_llm(temperature=0.1)
    
    def extract_facts(self, text: str) -> List[Fact]:
        """
        提取事实陈述
        
        Args:
            text: 输入文本
            
        Returns:
            List[Fact]: 提取的事实列表
        """
        # 方法1: 使用 LLM 提取
        facts = self._extract_by_llm(text)
        
        # 方法2: 使用正则补充
        regex_facts = self._extract_by_regex(text)
        
        # 合并去重
        all_facts = facts + regex_facts
        seen = set()
        unique_facts = []
        for f in all_facts:
            if f.text not in seen:
                seen.add(f.text)
                unique_facts.append(f)
        
        return unique_facts[:10]  # 最多10个事实
    
    def _extract_by_llm(self, text: str) -> List[Fact]:
        """使用 LLM 提取事实"""
        prompt = f"""请从以下文本中提取所有可验证的事实陈述。

文本：
{text[:2000]}  # 限制长度

要求：
1. 提取包含具体数值、日期、实体名称、性能指标的陈述
2. 每个事实应该是独立的、可验证的
3. 不要提取主观观点或常识性陈述

输出格式（JSON）：
{{
  "facts": [
    {{
      "text": "事实陈述",
      "type": "数值/实体/关系/性能",
      "confidence": 0.0-1.0
    }}
  ]
}}

只输出 JSON，不要其他内容。"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            
            facts = []
            for f in data.get("facts", []):
                facts.append(Fact(
                    text=f["text"],
                    fact_type=f.get("type", "unknown"),
                    confidence=f.get("confidence", 0.8)
                ))
            
            return facts
            
        except Exception as e:
            print(f"[FactExtractor] LLM提取失败: {e}")
            return []
    
    def _extract_by_regex(self, text: str) -> List[Fact]:
        """使用正则提取事实"""
        facts = []
        
        for pattern in self.FACT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # 扩展上下文
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end]
                
                facts.append(Fact(
                    text=context,
                    fact_type="regex",
                    confidence=0.6
                ))
        
        return facts


class FactVerifier:
    """
    事实验证器
    
    使用搜索工具验证事实的准确性
    """
    
    # 扣分标准（已调整，更宽松）
    DEDUCTION_RULES = {
        VerificationStatus.CONTRADICTED: 1.5,   # 矛盾：扣 1.5 分（原为 2.5）
        VerificationStatus.DOUBTFUL: 0.3,       # 存疑：扣 0.3 分（原为 0.5）
        VerificationStatus.PARTIAL: 0.5,        # 部分：扣 0.5 分（原为 1.0）
        VerificationStatus.VERIFIED: 0.0,       # 验证：不扣分
    }
    
    def __init__(self):
        self.search_tool = SearchTool()
        self.llm = create_llm(temperature=0.1)
    
    def verify(self, fact: Fact) -> VerificationResult:
        """
        验证单个事实
        
        Args:
            fact: 事实陈述
            
        Returns:
            VerificationResult: 验证结果
        """
        print(f"[FactVerifier] 验证: {fact.text[:50]}...")
        
        # 1. 生成搜索查询
        queries = self._generate_queries(fact)
        
        # 2. 执行搜索
        all_results = []
        for query in queries[:3]:  # 最多3个查询
            try:
                results = self.search_tool.search(query, max_results=3)
                all_results.extend(results)
            except Exception as e:
                print(f"[FactVerifier] 搜索失败: {e}")
        
        if not all_results:
            return VerificationResult(
                fact=fact.text,
                status=VerificationStatus.DOUBTFUL,
                evidence=[],
                confidence=0.0,
                deduction=self.DEDUCTION_RULES[VerificationStatus.DOUBTFUL]
            )
        
        # 3. 对比验证
        status, confidence = self._compare_evidence(fact, all_results)
        
        # 4. 生成结果
        deduction = self.DEDUCTION_RULES[status]
        
        return VerificationResult(
            fact=fact.text,
            status=status,
            evidence=all_results[:5],  # 最多5条证据
            confidence=confidence,
            deduction=deduction
        )
    
    def _generate_queries(self, fact: Fact) -> List[str]:
        """生成搜索查询"""
        # 直接查询
        queries = [fact.text]
        
        # 提取关键词查询
        # 提取实体名称（大写字母开头的词）
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', fact.text)
        if entities:
            queries.append(" ".join(entities[:3]))
        
        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', fact.text)
        if numbers and entities:
            queries.append(f"{entities[0]} {numbers[0]}")
        
        return queries
    
    def _compare_evidence(self, fact: Fact, evidence: List[Dict]) -> Tuple[VerificationStatus, float]:
        """
        对比证据，判定验证状态
        
        Args:
            fact: 事实陈述
            evidence: 搜索证据
            
        Returns:
            (状态, 置信度)
        """
        # 使用 LLM 判断
        evidence_text = "\n".join([
            f"[{i+1}] {e.get('title', '')}: {e.get('content', '')[:200]}"
            for i, e in enumerate(evidence[:5])
        ])
        
        prompt = f"""请判断以下事实陈述是否与搜索结果一致。

[事实陈述]
{fact.text}

[搜索结果]
{evidence_text}

判断标准：
- VERIFIED: 搜索结果明确支持事实陈述
- CONTRADICTED: 搜索结果明确反对事实陈述
- PARTIAL: 搜索结果部分支持，有出入
- DOUBTFUL: 搜索结果无法验证该陈述

输出格式（JSON）：
{{
  "status": "VERIFIED/CONTRADICTED/PARTIAL/DOUBTFUL",
  "confidence": 0.0-1.0,
  "reasoning": "判断理由"
}}

只输出 JSON。"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            
            status = VerificationStatus(data.get("status", "DOUBTFUL").lower())
            confidence = float(data.get("confidence", 0.5))
            
            return status, confidence
            
        except Exception as e:
            print(f"[FactVerifier] 判定失败: {e}")
            return VerificationStatus.DOUBTFUL, 0.0


class FactChecker:
    """
    事实检查器（主类）
    
    整合事实提取和验证的完整流程
    """
    
    def __init__(self):
        self.extractor = FactExtractor()
        self.verifier = FactVerifier()
    
    def check(self, text: str) -> FactCheckReport:
        """
        检查文本中的事实
        
        Args:
            text: 输入文本
            
        Returns:
            FactCheckReport: 验证报告
        """
        print(f"[FactChecker] 开始事实验证...")
        
        # 1. 提取事实
        facts = self.extractor.extract_facts(text)
        print(f"[FactChecker] 提取到 {len(facts)} 个事实")
        
        if not facts:
            return FactCheckReport(
                facts=[],
                total_deduction=0.0,
                verified_count=0,
                contradicted_count=0,
                doubtful_count=0,
                summary="未提取到可验证的事实"
            )
        
        # 2. 验证每个事实
        results = []
        for fact in facts:
            result = self.verifier.verify(fact)
            results.append(result)
            print(f"[FactChecker] {result.status.value}: {result.fact[:40]}... (扣分: {result.deduction})")
        
        # 3. 生成报告
        total_deduction = sum(r.deduction for r in results)
        verified_count = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        contradicted_count = sum(1 for r in results if r.status == VerificationStatus.CONTRADICTED)
        doubtful_count = sum(1 for r in results if r.status == VerificationStatus.DOUBTFUL)
        
        summary = f"验证: {verified_count}, 矛盾: {contradicted_count}, 存疑: {doubtful_count}, 总扣分: {total_deduction}"
        
        print(f"[FactChecker] {summary}")
        
        return FactCheckReport(
            facts=results,
            total_deduction=total_deduction,
            verified_count=verified_count,
            contradicted_count=contradicted_count,
            doubtful_count=doubtful_count,
            summary=summary
        )


# 便捷函数
def check_facts(text: str) -> FactCheckReport:
    """快速检查事实"""
    checker = FactChecker()
    return checker.check(text)
