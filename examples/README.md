# Examples

Runnable demos for ReaLM-Retrieve.

| Script | What it shows | Hardware |
|--------|---------------|----------|
| [`quickstart.py`](quickstart.py) | Full pipeline (segment → RSUS → policy → retrieve → answer) on a 12-doc toy corpus. | CPU |
| [`bench_plot.py`](bench_plot.py) | Re-render the F1-vs-retrieval-calls trade-off figure from the paper. | CPU |

> **Tip:** Once you've grasped the pipeline, swap `ToyRetriever` →
> `ColBERTRetriever` and `ToyReasoningModel` → `VLLMReasoningModel` in
> [`src/realm_retrieve/cli.py`](../src/realm_retrieve/cli.py) and you have the
> production system.
