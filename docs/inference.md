# Inference

End-to-end inference with the *production* components:

```python
from realm_retrieve import (
    ColBERTRetriever,
    RSUSCalculator,
    RetrievalInterventionPolicy,
    ReasoningStepSegmenter,
    create_reasoning_model,
)

retriever  = ColBERTRetriever(index_path="data/indices/colbert/musique.plaid")
segmenter  = ReasoningStepSegmenter.from_pretrained("checkpoints/segmentation")
lrm        = create_reasoning_model("deepseek")             # or "openai", "qwq"
rsus       = RSUSCalculator(lrm, retriever, alpha=0.4, beta=0.35, gamma=0.25)
policy     = RetrievalInterventionPolicy().load("checkpoints/policy/best_model.pt")

# … see evaluate.py for the full per-question loop.
```

## Choosing a reasoning model

| `model_type` | Backed by                             | Recommended for      |
|--------------|----------------------------------------|----------------------|
| `deepseek`   | vLLM + `DeepSeek-R1-Distill-Qwen-32B`  | self-hosted GPU      |
| `qwq`        | vLLM + `Qwen/QwQ-32B-Preview`          | self-hosted GPU      |
| `openai`     | OpenAI API (`o1-preview`)              | no-GPU / quick try   |

Drop in your own by subclassing `ReasoningModelWrapper`.

## Plugging in your own retriever

Implement two methods:

```python
class MyRetriever:
    def retrieve(self, query: str, k: int = 5, return_scores: bool = False): ...
    def get_corpus_size(self) -> int: ...
```

Then pass it everywhere `ColBERTRetriever` is expected.
