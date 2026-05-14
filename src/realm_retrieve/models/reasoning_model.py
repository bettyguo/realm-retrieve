"""
Reasoning Model Wrapper

Unified interface for large reasoning models:
- DeepSeek-R1-671B / DeepSeek-R1-Distill-Qwen-32B
- OpenAI o1
- QwQ-32B-Preview

Handles both open-weight models (via vLLM) and API-only models.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

# vLLM and openai are heavy optional dependencies. We import them lazily inside
# the subclasses that actually need them so that ``from
# realm_retrieve.models.reasoning_model import ReasoningModelWrapper`` keeps
# working in CPU-only or API-only environments (see Issue #3).


class ReasoningModelWrapper(ABC):
    """Abstract base class for reasoning model wrappers."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 25000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate completion from prompt."""
        pass
    
    @abstractmethod
    def compute_embedding(self, text: str) -> torch.Tensor:
        """Compute text embedding for policy network."""
        pass


class VLLMReasoningModel(ReasoningModelWrapper):
    """
    Wrapper for open-weight models via vLLM.
    Supports: DeepSeek-R1-32B, QwQ-32B
    """
    
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 8,
        device: str = "cuda",
    ):
        from vllm import LLM  # lazy: optional GPU-only dep
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=32768,
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 25000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        from vllm import SamplingParams  # lazy

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    def compute_embedding(self, text: str) -> torch.Tensor:
        embedding = self.embedding_model.encode(
            text, 
            convert_to_tensor=True,
        )
        return embedding


class OpenAIReasoningModel(ReasoningModelWrapper):
    """
    Wrapper for OpenAI o1 (API-only).
    """
    
    def __init__(self, model_name: str = "o1-preview", api_key: Optional[str] = None):
        import openai  # lazy: only required if you actually use the OpenAI backend
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 25000,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content
    
    def compute_embedding(self, text: str) -> torch.Tensor:
        embedding = self.embedding_model.encode(
            text,
            convert_to_tensor=True,
        )
        return embedding


def create_reasoning_model(
    model_type: str,
    model_name: Optional[str] = None,
    **kwargs
) -> ReasoningModelWrapper:
    """
    Factory function to create reasoning model wrapper.
    
    Args:
        model_type: "deepseek", "qwq", "openai"
        model_name: Specific model name (optional)
        **kwargs: Additional arguments
        
    Returns:
        ReasoningModelWrapper instance
    """
    if model_type == "deepseek":
        name = model_name or "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        return VLLMReasoningModel(name, **kwargs)
    
    elif model_type == "qwq":
        name = model_name or "Qwen/QwQ-32B-Preview"
        return VLLMReasoningModel(name, **kwargs)
    
    elif model_type == "openai":
        name = model_name or "o1-preview"
        return OpenAIReasoningModel(name, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
