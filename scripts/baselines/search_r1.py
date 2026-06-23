"""Search-R1 baseline: reasoning model with search tool calls."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("search_r1")
class SearchR1Runner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        top_k = self.config.get("top_k", 5)
        max_search_calls = self.config.get("max_search_calls", 5)
        retrieval_events = []

        system = (
            "You are a helpful assistant that can search for information. "
            "When you need to look something up, output: <search>query</search>. "
            "After receiving search results, continue reasoning. "
            "Provide your final answer after 'The answer is:'."
        )
        prompt = f"{system}\n\nQuestion: {question}\n"

        for attempt in range(max_search_calls + 1):
            output = self.reasoning_model.generate(
                prompt, max_tokens=self.config.get("max_tokens", 25000), seed=seed,
            )

            import re
            search_match = re.search(r"<search>(.*?)</search>", output.text)
            if not search_match or attempt >= max_search_calls:
                break

            query = search_match.group(1).strip()
            docs, latency_ms = self.retriever.retrieve(query, top_k=top_k)
            retrieval_events.append(RetrievalEvent(
                step_index=attempt,
                position_fraction=search_match.start() / max(len(output.text), 1),
                latency_ms=latency_ms,
            ))

            context = "\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(docs))
            text_before = output.text[:search_match.start()]
            prompt = (
                f"{system}\n\nQuestion: {question}\n"
                f"{text_before}\n"
                f"Search results:\n{context}\n\n"
                f"Continue reasoning. Provide your final answer "
                f"after 'The answer is:'.\n"
            )

        predicted = extract_answer(output.text)
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)

        total_latency = sum(e.latency_ms for e in retrieval_events) / 1000 + output.latency

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="search_r1",
            seed=seed,
            model=self.config["model_name"],
            gold_answer=gold_answer,
            predicted_answer=predicted,
            em=em,
            f1=f1,
            num_retrieval_calls=len(retrieval_events),
            retrieval_events=retrieval_events,
            reasoning_tokens=output.token_count,
            e2e_latency_s=total_latency,
            num_hops=kwargs.get("num_hops"),
        )
