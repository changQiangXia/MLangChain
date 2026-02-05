"""
Code Validator
代码验证器 - 检查代码的可运行性和质量

解决代码类任务的评分问题，不再依赖字数，而是检查代码本身。
"""

import subprocess
import sys
import tempfile
import os
import ast
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """代码验证结果"""
    is_valid: bool = Field(description="是否有效")
    syntax_valid: bool = Field(description="语法是否正确")
    can_execute: bool = Field(description="能否执行")
    has_error: Optional[str] = Field(default=None, description="错误信息")
    output: Optional[str] = Field(default=None, description="执行输出")
    has_comments: bool = Field(description="是否有注释")
    has_docstring: bool = Field(description="是否有文档字符串")
    function_count: int = Field(default=0, description="函数数量")
    class_count: int = Field(default=0, description="类数量")
    line_count: int = Field(default=0, description="代码行数")
    imports: List[str] = Field(default_factory=list, description="导入的模块")


class CodeValidator:
    """
    代码验证器
    
    验证 Python 代码的：
    1. 语法正确性
    2. 可执行性
    3. 代码质量（注释、文档字符串等）
    """
    
    def __init__(self, timeout: int = 5):
        """
        初始化验证器
        
        Args:
            timeout: 代码执行超时时间（秒）
        """
        self.timeout = timeout
    
    def validate(self, code: str) -> ValidationResult:
        """
        验证代码
        
        Args:
            code: Python 代码字符串
            
        Returns:
            ValidationResult: 验证结果
        """
        result = ValidationResult(
            is_valid=False,
            syntax_valid=False,
            can_execute=False,
            has_comments=False,
            has_docstring=False
        )
        
        # 1. 语法检查
        syntax_valid, error = self._check_syntax(code)
        result.syntax_valid = syntax_valid
        
        if not syntax_valid:
            result.has_error = error
            return result
        
        # 2. 代码质量分析
        quality_metrics = self._analyze_quality(code)
        result.has_comments = quality_metrics["has_comments"]
        result.has_docstring = quality_metrics["has_docstring"]
        result.function_count = quality_metrics["function_count"]
        result.class_count = quality_metrics["class_count"]
        result.line_count = quality_metrics["line_count"]
        result.imports = quality_metrics["imports"]
        
        # 3. 执行检查（在沙箱环境中）
        can_execute, output, error = self._execute_safely(code)
        result.can_execute = can_execute
        result.output = output[:500] if output else None  # 限制输出长度
        
        if error:
            result.has_error = error
        
        # 4. 综合判断
        result.is_valid = result.syntax_valid and result.can_execute
        
        return result
    
    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """检查代码语法"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"语法错误: {str(e)}"
        except Exception as e:
            return False, f"解析错误: {str(e)}"
    
    def _analyze_quality(self, code: str) -> Dict:
        """分析代码质量指标"""
        metrics = {
            "has_comments": False,
            "has_docstring": False,
            "function_count": 0,
            "class_count": 0,
            "line_count": 0,
            "imports": []
        }
        
        lines = code.split('\n')
        metrics["line_count"] = len([l for l in lines if l.strip()])
        
        # 检查注释
        for line in lines:
            if '#' in line:
                metrics["has_comments"] = True
                break
        
        # AST 分析
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics["function_count"] += 1
                    # 检查函数文档字符串
                    if ast.get_docstring(node):
                        metrics["has_docstring"] = True
                elif isinstance(node, ast.ClassDef):
                    metrics["class_count"] += 1
                    # 检查类文档字符串
                    if ast.get_docstring(node):
                        metrics["has_docstring"] = True
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            metrics["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        metrics["imports"].append(node.module)
            
            # 检查模块级文档字符串
            if ast.get_docstring(tree):
                metrics["has_docstring"] = True
                
        except Exception:
            pass
        
        return metrics
    
    def _execute_safely(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        在沙箱环境中安全执行代码
        
        Returns:
            (能否执行, 输出, 错误)
        """
        # 危险操作黑名单
        dangerous_keywords = [
            'os.system', 'subprocess', 'eval(', 'exec(',
            '__import__', 'open(', 'file(', 'input(',
            'raw_input', 'reload(', 'compile(',
        ]
        
        for keyword in dangerous_keywords:
            if keyword in code:
                return False, None, f"包含危险操作: {keyword}"
        
        # 创建临时文件执行
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            # 在子进程中执行
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # 清理临时文件
            os.unlink(temp_path)
            
            if result.returncode == 0:
                return True, result.stdout, None
            else:
                return False, result.stdout, result.stderr[:200]
                
        except subprocess.TimeoutExpired:
            return False, None, f"执行超时（>{self.timeout}秒）"
        except Exception as e:
            return False, None, f"执行错误: {str(e)}"
    
    def calculate_quality_score(self, result: ValidationResult) -> float:
        """
        计算代码质量分数 (0-10)
        
        Args:
            result: 验证结果
            
        Returns:
            float: 质量分数
        """
        score = 0.0
        
        # 基础分：语法正确
        if result.syntax_valid:
            score += 3.0
        
        # 可执行性
        if result.can_execute:
            score += 3.0
        
        # 代码规范
        if result.has_comments:
            score += 1.0
        if result.has_docstring:
            score += 1.0
        
        # 代码结构
        if result.function_count > 0:
            score += 1.0
        if result.class_count > 0:
            score += 0.5
        
        # 代码规模（合理的代码长度）
        if 10 <= result.line_count <= 100:
            score += 0.5
        
        return min(score, 10.0)


def validate_code(code: str, timeout: int = 5) -> ValidationResult:
    """便捷函数：验证代码"""
    validator = CodeValidator(timeout=timeout)
    return validator.validate(code)


def extract_code_from_output(output: str) -> Optional[str]:
    """
    从模型输出中提取代码块
    
    Args:
        output: 模型生成的文本
        
    Returns:
        str: 提取的代码，如果没有则返回 None
    """
    # 尝试提取 markdown 代码块
    if "```python" in output:
        code = output.split("```python")[1].split("```")[0].strip()
        return code
    elif "```" in output:
        code = output.split("```")[1].split("```")[0].strip()
        return code
    
    # 如果没有 markdown 标记，尝试提取缩进代码块
    lines = output.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        if line.startswith('def ') or line.startswith('class ') or line.startswith('import '):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    return None
