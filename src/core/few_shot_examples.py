"""
Few-Shot Examples - Full Score Example Library (Strategy 3)

Inject high-quality full-score examples to improve the Generator's baseline level.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Example:
    """Single Example"""
    instruction: str
    input_text: str
    output: str
    score: float
    highlights: List[str]  # Highlights description


class FewShotLibrary:
    """Few-Shot Example Library"""
    
    # General high-quality examples (suitable for all types)
    GENERAL_EXAMPLES: List[Example] = [
        Example(
            instruction="Explain what the Transformer architecture is and its applications in NLP",
            input_text="",
            output="""## Transformer Architecture Explained

### 1. Core Concepts
Transformer is a deep learning architecture proposed by Vaswani et al. in 2017, revolutionizing NLP. It relies entirely on the Attention Mechanism, abandoning traditional RNNs and CNNs.

### 2. Architecture Components
Transformer consists of two main parts:

**Encoder**:
- Stacked N identical layers (N=6 in original paper)
- Two sub-layers per layer: Multi-Head Self-Attention and Feed-Forward Network
- Uses Residual Connections and Layer Normalization

**Decoder**:
- Also stacked N identical layers
- Three sub-layers: Masked Multi-Head Attention, Encoder-Decoder Attention, FFN

### 3. Key Innovation: Attention Mechanism
Scaled Dot-Product Attention formula:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

Multi-Head Attention: Parallel attention executed h times (h=8)

### 4. Applications in NLP
- Machine Translation: Google Translate adopted Transformer in 2016
- Pre-trained Models: BERT, GPT series, T5
- ChatGPT: Based on GPT-4 architecture with RLHF

### 5. Pros and Cons
**Pros**: Parallel computation, long-range dependency modeling, interpretable
**Cons**: O(n^2) complexity, requires large training data

### 6. Conclusion
Transformer has become the foundation of modern NLP, impacting everything from translation to code generation.""",
            score=9.5,
            highlights=[
                "Clear structure with 6 sections",
                "Includes mathematical formulas",
                "Real-world cases: Google Translate, ChatGPT",
                "Quantitative metrics: 28.4 BLEU on WMT2014",
                "Objective pros/cons analysis"
            ]
        )
    ]
    
    @classmethod
    def get_examples_for_task(cls, task_type: str = "general", n: int = 1) -> List[Example]:
        """Get examples suitable for specific tasks"""
        examples = cls.GENERAL_EXAMPLES[:n]
        return examples if examples else cls.GENERAL_EXAMPLES[:1]
    
    @classmethod
    def format_examples_for_prompt(cls, examples: List[Example]) -> str:
        """Format examples into prompt text"""
        parts = ["\n## Full Score Example Reference (9.5 points)\n"]
        
        for i, ex in enumerate(examples, 1):
            parts.append(f"### Example {i}")
            parts.append(f"Instruction: {ex.instruction}")
            if ex.input_text:
                parts.append(f"Input: {ex.input_text}")
            parts.append(f"Output: {ex.output}")
            parts.append("\nHighlights:")
            for highlight in ex.highlights:
                parts.append(f"  - {highlight}")
            parts.append("")
        
        parts.append("=" * 50)
        parts.append("\nRefer to the [structural depth], [content specificity] and [code standards] of the above examples.\n")
        
        return "\n".join(parts)


def get_few_shot_prompt(n: int = 1, task_type: str = "general") -> str:
    """Get Few-Shot prompt text"""
    examples = FewShotLibrary.get_examples_for_task(task_type, n)
    return FewShotLibrary.format_examples_for_prompt(examples)
