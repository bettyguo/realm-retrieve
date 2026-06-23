"""Single-RAG baseline: retrieve once before generation, then answer."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("single_rag")
class SingleRAGRunner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        top_k = self.config.get("top_k", 5)
        docs, latency_ms = self.retriever.retrieve(question, top_k=top_k)

        context = "\n".join(
            f"[{i+1}] {d.text}" for i, d in enumerate(docs)
        )
        prompt = (
            f"Use the following evidence to answer the question. "
            f"Think step by step, then provide your final answer "
            f"after 'The answer is:'.\n\n"
            f"Evidence:\n{context}\n\n"
            f"Question: {question}\n"
        )

        output = self.reasoning_model.generate(
            prompt, max_tokens=self.config.get("max_tokens", 25000), seed=seed,
        )
        predicted = extract_answer(output.text)
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="single_rag",
            seed=seed,
            model=self.config["model_name"],
            gold_answer=gold_answer,
            predicted_answer=predicted,
            em=em,
            f1=f1,
            num_retrieval_calls=1,
            retrieval_events=[RetrievalEvent(
                step_index=0, position_fraction=0.0, latency_ms=latency_ms,
            )],
            reasoning_tokens=output.token_count,
            e2e_latency_s=output.latency + latency_ms / 1000,
            num_hops=kwargs.get("num_hops"),
        )
