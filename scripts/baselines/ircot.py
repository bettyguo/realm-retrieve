"""IRCoT baseline: interleave retrieval and chain-of-thought at fixed intervals."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("ircot")
class IRCoTRunner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        top_k = self.config.get("top_k", 5)
        max_hops = self.config.get("max_hops", 4)
        retrieval_events = []
        all_docs = []

        query = question
        for hop in range(max_hops):
            docs, latency_ms = self.retriever.retrieve(query, top_k=top_k)
            all_docs.extend(docs)
            retrieval_events.append(RetrievalEvent(
                step_index=hop,
                position_fraction=hop / max_hops,
                latency_ms=latency_ms,
            ))

            context = "\n".join(
                f"[{i+1}] {d.text}" for i, d in enumerate(all_docs)
            )
            prompt = (
                f"Use the following evidence to reason about the question. "
                f"Continue your chain of thought.\n\n"
                f"Evidence:\n{context}\n\n"
                f"Question: {question}\n"
                f"Think step by step. If you have enough information, "
                f"provide your final answer after 'The answer is:'.\n"
            )
            output = self.reasoning_model.generate(
                prompt, max_tokens=self.config.get("max_tokens", 25000), seed=seed,
            )

            if "the answer is" in output.text.lower():
                break

            query = output.text.split(".")[-2].strip() if "." in output.text else question

        predicted = extract_answer(output.text)
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)

        total_latency = sum(e.latency_ms for e in retrieval_events) / 1000 + output.latency

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="ircot",
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
