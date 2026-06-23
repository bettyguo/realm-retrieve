"""Self-RAG baseline: self-reflective retrieval-augmented generation with critique tokens."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("self_rag")
class SelfRAGRunner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        top_k = self.config.get("top_k", 5)
        max_iterations = self.config.get("self_rag_max_iterations", 3)
        retrieval_events = []

        prompt = (
            f"Answer the following question. At each step, decide whether "
            f"retrieval is needed by outputting [Retrieve] or [No Retrieve]. "
            f"If you retrieve, critique the relevance with [Relevant] or "
            f"[Irrelevant]. Provide your final answer after 'The answer is:'.\n\n"
            f"Question: {question}\n"
        )

        for iteration in range(max_iterations):
            output = self.reasoning_model.generate(
                prompt, max_tokens=self.config.get("max_tokens", 25000), seed=seed,
            )

            if "[Retrieve]" not in output.text:
                break

            retrieve_pos = output.text.index("[Retrieve]")
            context_before = output.text[:retrieve_pos]
            query = context_before.split(".")[-1].strip() if "." in context_before else question

            docs, latency_ms = self.retriever.retrieve(query, top_k=top_k)
            retrieval_events.append(RetrievalEvent(
                step_index=iteration,
                position_fraction=retrieve_pos / max(len(output.text), 1),
                latency_ms=latency_ms,
            ))

            context = "\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(docs))
            prompt = (
                f"{output.text[:retrieve_pos]}\n\n"
                f"Retrieved evidence:\n{context}\n\n"
                f"Continue reasoning. Critique relevance with [Relevant] or "
                f"[Irrelevant]. Provide your final answer after 'The answer is:'.\n"
            )

        predicted = extract_answer(output.text)
        em = compute_exact_match(predicted, gold_answer)
        f1 = compute_f1(predicted, gold_answer)

        total_latency = sum(e.latency_ms for e in retrieval_events) / 1000 + output.latency

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="self_rag",
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
