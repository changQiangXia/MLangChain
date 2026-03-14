"""
Code Generator Agent
Generate instruction data that includes runnable code examples.
"""

import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.core.safe_json_utils import safe_json_loads
from src.llm_factory import create_llm
from src.state import AlpacaData


class CodeGeneratorAgent:
    """Generate instruction-following data with code examples."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.default_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.4)

    def _get_code_generation_prompt(self) -> ChatPromptTemplate:
        system_template = """你是一个专业的代码示例生成专家。
你的任务是为技术主题生成包含可运行代码的指令数据。

要求：
1. 代码必须完整、可运行，不要输出伪代码
2. 解释要清晰，代码要有必要注释
3. 优先输出 Python 示例
4. 返回严格 JSON，output 字段必须是字符串
"""

        human_template = """请围绕主题“{task_description}”生成一条包含代码示例的指令数据。

输出格式：
```json
{{
  "instruction": "请解释{task_description}，并提供 Python 代码示例",
  "input": "",
  "output": "## 概念解释\\n...\\n\\n## 代码示例\\n```python\\n# code\\n```\\n\\n## 说明\\n..."
}}
```
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])

    def generate_with_code(self, task_description: str) -> AlpacaData:
        prompt = self._get_code_generation_prompt()
        chain = prompt | self.llm

        max_retries = 3
        response = None

        for attempt in range(max_retries):
            try:
                response = chain.invoke({"task_description": task_description})
                break
            except Exception as exc:
                error_text = str(exc).lower()
                if "timeout" in error_text or "read operation" in error_text:
                    print(f"[CodeGenerator] API timeout, retry {attempt + 1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        return self._build_fallback(task_description, f"API timeout: {exc}")
                    import time

                    time.sleep(2 ** attempt)
                    continue

                print(f"[CodeGenerator Error] API call failed: {exc}")
                return self._build_fallback(task_description, f"API call failed: {exc}")

        try:
            content = response.content if response is not None else ""
            content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", content)

            data = safe_json_loads(content, default=None)
            if not isinstance(data, dict):
                data = self._parse_relaxed_response(content)
            if not isinstance(data, dict):
                raise ValueError("JSON parsing returned non-dict data")

            if not isinstance(data.get("output", ""), str):
                data["output"] = str(data.get("output", ""))
            return AlpacaData(**data)
        except Exception as exc:
            print(f"[CodeGenerator Error] JSON parse failed: {exc}")
            return self._build_fallback(task_description, f"JSON parse failed: {exc}")

    def _parse_relaxed_response(self, content: str) -> Optional[dict]:
        """Parse near-JSON responses that contain raw multiline code blocks."""
        body = content.strip()
        if body.startswith("```json"):
            body = body[len("```json"):].strip()
        elif body.startswith("```"):
            body = body[len("```"):].strip()
        if body.endswith("```"):
            body = body[:-3].strip()

        instruction_match = re.search(
            r'"instruction"\s*:\s*"(?P<value>.*?)"\s*,\s*"input"',
            body,
            re.DOTALL,
        )
        input_match = re.search(
            r'"input"\s*:\s*"(?P<value>.*?)"\s*,\s*"output"',
            body,
            re.DOTALL,
        )
        output_match = re.search(
            r'"output"\s*:\s*"(?P<value>.*)"\s*}\s*$',
            body,
            re.DOTALL,
        )

        if not instruction_match or not output_match:
            return None

        return {
            "instruction": self._unescape_text(instruction_match.group("value")),
            "input": self._unescape_text(input_match.group("value")) if input_match else "",
            "output": self._unescape_text(output_match.group("value")),
        }

    def _unescape_text(self, text: str) -> str:
        """Unescape common JSON-style sequences used by model output."""
        return (
            text.replace("\\\\", "\\")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .strip()
        )

    def _build_fallback(self, task_description: str, error: str) -> AlpacaData:
        return AlpacaData(
            instruction=f"请解释{task_description}，并提供 Python 代码示例",
            input="",
            output=f"代码生成失败，请重试。错误: {error}",
        )


def enhance_with_code(existing_data: AlpacaData, task_description: str) -> AlpacaData:
    """Append generated code examples to existing data."""

    generator = CodeGeneratorAgent()
    code_data = generator.generate_with_code(task_description)

    enhanced_output = f"""{existing_data.output}

## 代码示例
{code_data.output}
"""

    return AlpacaData(
        instruction=f"{existing_data.instruction}，并提供代码示例",
        input=existing_data.input,
        output=enhanced_output,
    )
