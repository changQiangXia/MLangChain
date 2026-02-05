"""
Generator Agent
基于搜索结果或知识库生成初始指令数据 (Alpaca格式)
"""

import json
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.state import AlpacaData, GraphState
from src.tools.search_tool import SearchTool
from src.llm_factory import create_llm
from src.core.few_shot_examples import get_few_shot_prompt


class GeneratorAgent:
    """
    生成器 Agent：负责生成高质量的指令微调数据
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            model_name: 使用的模型名称，默认使用配置中的 default_model
        """
        self.model_name = model_name or settings.default_model
        self.llm = create_llm(model_name=self.model_name, temperature=0.7)
        self.search_tool = SearchTool()
        
    def _get_generation_prompt(self, use_few_shot: bool = True) -> ChatPromptTemplate:
        """
        获取生成任务的 Prompt 模板 - 注入 Few-Shot 示例 (Strategy 3)
        """
        # 获取 Few-Shot 示例
        few_shot_text = ""
        if use_few_shot:
            try:
                few_shot_text = get_few_shot_prompt(n=1)
            except Exception as e:
                print(f"[Generator] Few-Shot加载失败: {e}")
        
        system_template = f"""你是一个专业的指令微调数据生成专家。
你的任务是基于提供的主题，生成**卓越质量**的 Alpaca 格式训练数据。

{few_shot_text}

## Alpaca 格式要求
- instruction: 清晰、具体、有深度的指令
- input: 上下文（可为空）
- output: **非常详细、全面、深入**的回答（**至少400字**，参考范例）

## 高质量回答的标准（必须满足）
1. **内容深度**：不只讲概念，要深入解释原理、机制（参考范例的Transformer详解）
2. **具体例子**：至少包含3个具体的实际应用例子（具体到公司/产品/数据）
3. **结构化**：使用标题、段落、列表等方式组织内容
4. **准确性**：专业术语正确，内容无错误
5. **实用性**：读者能从中获得实际价值

## 参考范例的结构特点
- 分多个章节系统讲解
- 包含数学公式或代码（如适用）
- 提供真实应用案例和量化指标
- 分析优缺点

## 警告
- 少于300字的回答会被评为低质量
- 没有具体例子（具体到公司）会被扣分
- 泛泛而谈、空洞的内容会被打低分
"""

        human_template = """主题：{task_description}

{search_context}

请生成一条**卓越质量**的指令数据。

## 严格要求（参考满分范例的质量）
1. 回答必须 **≥400字**，深入、全面、结构化
2. 必须包含 **3个具体例子**（具体到公司/产品/量化数据）
3. 必须涵盖：**定义**、**原理/机制**、**应用场景**、**优缺点**
4. 如果涉及算法，提供**复杂度分析**或**关键代码**
5. 例子格式：**技术名称 + 具体场景 + 量化效果**

## 例子质量要求（重要）
- ❌ 错误例子："用于图像识别"
- ✅ 正确例子："ResNet-50在ImageNet图像分类中达到76.2%准确率"
- ✅ 正确例子："Tesla FSD使用CNN进行实时道路物体检测，误报率<0.1%"
- ✅ 正确例子："GPT-4在MMLU基准测试中达到86.4%准确率"

## 输出格式（严格JSON）
**重要：output 必须是单个字符串，不要输出成对象！**

```json
{{
  "instruction": "请详细解释{task_description}的概念、原理和应用",
  "input": "",
  "output": "## 1. 概念定义\\n...\\n\\n## 2. 工作原理\\n...\\n\\n## 3. 具体应用案例\\n1. **[公司/产品]**：[场景]，[效果]\\n2. **[公司/产品]**：[场景]，[效果]\\n3. **[公司/产品]**：[场景]，[效果]\\n\\n## 4. 优缺点分析\\n...\\n\\n## 5. 总结"
}}
```

记住：目标是**9分以上**的质量，参考范例的深度和结构！
"""

        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template)
        ])
    
    def generate(
        self, 
        task_description: str,
        search_results: Optional[list] = None
    ) -> AlpacaData:
        """
        生成指令数据
        
        Args:
            task_description: 任务描述/主题
            search_results: 可选的搜索结果，用于参考
            
        Returns:
            AlpacaData 对象
        """
        # 准备搜索上下文
        if search_results is None:
            search_context = self.search_tool.get_context_for_generation(task_description)
        else:
            search_context = ""
            if search_results:
                context_parts = []
                for i, result in enumerate(search_results, 1):
                    title = result.get("title", "")
                    content = result.get("content", "")
                    if content:
                        context_parts.append(f"[{i}] {title}\n{content}")
                search_context = "\n\n".join(context_parts)
        
        if search_context:
            search_context = f"参考信息：\n{search_context}"
        
        # 构建 Prompt 并调用 LLM（带重试）
        prompt = self._get_generation_prompt()
        chain = prompt | self.llm
        
        # 带重试的 API 调用
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                response = chain.invoke({
                    "task_description": task_description,
                    "search_context": search_context
                })
                break
            except Exception as e:
                if "timeout" in str(e).lower() or "read operation" in str(e).lower():
                    print(f"[Generator] API 超时，重试 {attempt + 1}/{max_retries}...")
                    if attempt == max_retries - 1:
                        raise  # 最后一次重试失败则抛出异常
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise
        
        # 解析 JSON 输出
        try:
            content = response.content
            # 清理控制字符
            import re
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            
            # 尝试从 Markdown 代码块中提取 JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            data = json.loads(json_str)
            return AlpacaData(**data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"[Generator Error] JSON 解析失败: {e}")
            # 返回一个带有错误信息的默认数据
            return AlpacaData(
                instruction=task_description,
                input="",
                output=f"生成失败，请重试。错误: {e}"
            )
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        作为 LangGraph Node 调用
        
        Args:
            state: 当前状态
            
        Returns:
            更新后的状态
        """
        print(f"\n[Generator] 正在生成指令数据... (迭代: {state['iteration_count']})")
        
        task = state["task_description"]
        search_results = state.get("search_results")
        
        # 生成数据
        draft = self.generate(task, search_results)
        
        # 更新状态
        state["current_draft"] = draft
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        print(f"[Generator] 生成完成: {draft.instruction[:50]}...")
        
        return state


# 便捷函数
def generate_instruction_data(
    task_description: str,
    search_results: Optional[list] = None
) -> AlpacaData:
    """
    快速生成指令数据，无需实例化类
    """
    agent = GeneratorAgent()
    return agent.generate(task_description, search_results)
