"""FLARE baseline: forward-looking active retrieval triggered by low-confidence tokens."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("flare")
class FLARERunner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        top_k = self.config.get("top_k", 5)
        confidence_threshold = self.config.get("flare_confidence_threshold", 0.5)
        max_retrieval_calls = self.config.get("max_retrieval_calls", 6)
        retrieval_events = []

        prompt = (
            f"Answer the following question. Think step by step, then provide "
            f"your final answer after 'The answer is:'.\n\n"
            f"Question: {question}\n"
        )

        for attempt in range(max_retrieval_calls + 1):
            output = self.reasoning_model.generate(
                prompt, max_tokens=self.config.get("max_tokens", 25000),
                seed=seed, return_logprobs=True,
            )

            low_conf_spans = []
            if hasattr(output, "token_logprobs") and output.token_logprobs:
                import math
                for i, lp in enumerate(output.token_logprobs):
                    if lp is not None and math.exp(lp) < confidence_threshold:
                        low_conf_spans.append(i)

            if not low_conf_spans or attempt >= max_retrieval_calls:
                break

            query_tokens = output.text.split()
            start = max(0, low_conf_spans[0] - 5)
            end = min(len(query_tokens), low_conf_spans[0] + 10)
            query = " ".join(query_tokens[start:end])

            docs, latency_ms = self.retriever.retrieve(query, top_k=top_k)
            retrieval_events.append(RetrievalEvent(
                step_index=attempt,
                position_fraction=low_conf_spans[0] / max(len(query_tokens), 1),
                latency_ms=latency_ms,
            ))

            context = "\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(docs))
            prompt = (
                f"Use the following evidence to continue answering.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Question: {question}\n"
                f"Think step by step, then provide your final answer "
                f"after 'The answer is:'.\n"
            )

        predicted = extract_answer(output.text)
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)

        total_latency = sum(e.latency_ms for e in retrieval_events) / 1000 + output.latency

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="flare",
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
