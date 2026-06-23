"""No-retrieval baseline: LLM answers directly without any retrieval."""

from scripts.baselines.base import (
    BaseRunner, PredictionResult, RetrievalEvent,
    extract_answer, register_baseline,
)
from realm_retrieve.evaluation.metrics import compute_exact_match, compute_f1


@register_baseline("no_retrieval")
class NoRetrievalRunner(BaseRunner):

    def run_question(self, question, question_id, dataset, gold_answer, seed, **kwargs):
        prompt = (
            f"Answer the following question. Think step by step, then provide "
            f"your final answer after 'The answer is:'.\n\n"
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
            method="no_retrieval",
            seed=seed,
            model=self.config["model_name"],
            gold_answer=gold_answer,
            predicted_answer=predicted,
            em=em,
            f1=f1,
            num_retrieval_calls=0,
            reasoning_tokens=output.token_count,
            e2e_latency_s=output.latency,
            num_hops=kwargs.get("num_hops"),
        )
