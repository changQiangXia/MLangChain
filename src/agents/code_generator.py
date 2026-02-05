"""
Code Generator Agent
专门生成包含代码示例的指令数据
"""

import json
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.state import AlpacaData, GraphState
from src.llm_factory import create_llm


class CodeGeneratorAgent:
    """
    代码生成器 Agent：生成包含实际代码示例的指令数据
    """
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.4)
    
    def _get_code_generation_prompt(self) -> ChatPromptTemplate:
        """获取代码生成任务的 Prompt"""
        
        system_template = """你是一个专业的代码示例生成专家。
你的任务是为技术概念生成包含实际可运行代码的指令数据。

## 代码要求
1. **可运行**：代码必须是完整、可运行的，不能是伪代码
2. **实用性**：代码要展示概念的实际应用
3. **注释清晰**：关键步骤要有中文注释
4. **渐进式**：从简单示例到复杂应用

## 支持的语言
- Python（首选，用于机器学习/深度学习）
- 伪代码（用于解释算法原理）
- 具体框架代码（PyTorch/TensorFlow）

## 格式要求
代码块使用 markdown 格式：
```python
# 代码注释
代码内容
```
"""

        human_template = """为主题 "{task_description}" 生成包含代码示例的指令数据。

要求：
1. 指令要明确要求"请提供Python代码示例"
2. output 必须包含：
   - 概念解释（200字）
   - 简单代码示例（带注释）
   - 实际应用场景代码（可选）
3. 代码必须是可运行的Python代码

输出格式（JSON）：
```json
{{
  "instruction": "请解释{task_description}，并提供Python代码示例",
  "input": "",
  "output": "## 概念解释\\n...\\n\\n## 代码示例\\n```python\\n# 代码\\n```\\n\\n## 实际应用\\n..."
}}
```
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def generate_with_code(
        self, 
        task_description: str
    ) -> AlpacaData:
        """生成包含代码的指令数据"""
        
        prompt = self._get_code_generation_prompt()
        chain = prompt | self.llm
        
        response = chain.invoke({
            "task_description": task_description
        })
        
        try:
            content = response.content
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            return AlpacaData(**data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[CodeGenerator Error] JSON 解析失败: {e}")
            return AlpacaData(
                instruction=f"请解释{task_description}并提供代码示例",
                input="",
                output=f"生成失败，错误: {e}"
            )


def enhance_with_code(existing_data: AlpacaData, task_description: str) -> AlpacaData:
    """
    为现有数据添加代码示例
    """
    generator = CodeGeneratorAgent()
    code_data = generator.generate_with_code(task_description)
    
    # 合并原有内容和代码
    enhanced_output = f"""{existing_data.output}

## 代码示例
{code_data.output}
"""
    
    return AlpacaData(
        instruction=f"{existing_data.instruction}，并提供代码示例",
        input=existing_data.input,
        output=enhanced_output
    )
