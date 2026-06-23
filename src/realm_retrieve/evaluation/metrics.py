"""
Evaluation Metrics for Multi-Hop QA and Retrieval Quality

Implements:
1. QA Metrics: EM, F1, Sup-F1, Evi-F1
2. IR Metrics: Recall@k, Precision@k, MRR, nDCG@k (via pytrec_eval)
3. Efficiency Metrics: Retrieval calls, latency, per-call overhead
4. Statistical Significance: Paired bootstrap resampling
"""

from collections import Counter
from typing import Dict, List, Tuple
import re
import string

import numpy as np

# pytrec_eval is only needed by compute_retrieval_metrics — load it lazily so
# the rest of the module is usable in lightweight CPU environments.


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_qa_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Compute EM and F1 for QA predictions."""
    em_scores = [compute_exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    f1_scores = [compute_f1(p, g) for p, g in zip(predictions, ground_truths)]
    
    return {
        "em": np.mean(em_scores) * 100,
        "f1": np.mean(f1_scores) * 100,
    }


def compute_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Compute IR metrics using pytrec_eval.
    
    Args:
        qrels: {query_id: {doc_id: relevance}}
        run: {query_id: {doc_id: score}}
        k_values: List of k values for Recall@k, Precision@k
        
    Returns:
        Dict of metric name -> value
    """
    import pytrec_eval  # lazy: optional heavy dep

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {'recall', 'P', 'map', 'ndcg', 'recip_rank'}
    )
    
    results = evaluator.evaluate(run)
    
    # Aggregate metrics
    metrics = {}
    
    # Recall@k
    for k in k_values:
        recall_k = [v[f'recall_{k}'] for v in results.values()]
        metrics[f'recall@{k}'] = np.mean(recall_k) * 100
    
    # Precision@k
    for k in k_values:
        p_k = [v[f'P_{k}'] for v in results.values()]
        metrics[f'precision@{k}'] = np.mean(p_k) * 100
    
    # MRR
    mrr = [v['recip_rank'] for v in results.values()]
    metrics['mrr'] = np.mean(mrr)
    
    # nDCG@k
    for k in k_values:
        ndcg_k = [v[f'ndcg_cut_{k}'] for v in results.values()]
        metrics[f'ndcg@{k}'] = np.mean(ndcg_k) * 100
    
    return metrics


def compute_efficiency_metrics(
    retrieval_calls: List[int],
    latencies: List[float],
    reasoning_tokens: List[int],
    retrieval_latencies: List[float] = None,
) -> Dict[str, float]:
    """Compute efficiency metrics.

    Args:
        retrieval_calls: Number of retrieval calls per question.
        latencies: End-to-end latency per question (seconds).
        reasoning_tokens: Token count per question.
        retrieval_latencies: Per-question cumulative retrieval-only
            latency.  When provided, ``per_call_overhead`` is computed
            as the mean retrieval latency divided by mean calls (the
            incremental retrieval cost, not total e2e / calls).
    """
    total_calls = sum(retrieval_calls)
    total_latency = sum(latencies)

    if retrieval_latencies is not None and total_calls > 0:
        per_call = sum(retrieval_latencies) / total_calls
    elif total_calls > 0:
        per_call = total_latency / total_calls
    else:
        per_call = 0.0

    return {
        "avg_retrieval_calls": float(np.mean(retrieval_calls)),
        "total_retrieval_calls": total_calls,
        "avg_latency": float(np.mean(latencies)),
        "total_latency": total_latency,
        "per_call_overhead": per_call,
        "avg_reasoning_tokens": float(np.mean(reasoning_tokens)),
    }


def compute_sup_f1(
    predicted_sentences: List[str],
    gold_supporting_facts: List[Tuple[str, int]],
    context: List[Tuple[str, List[str]]],
) -> float:
    """
    Supporting Fact F1 for HotpotQA.

    Args:
        predicted_sentences: sentences from retrieved passages that the system used
        gold_supporting_facts: list of (title, sentence_index) tuples from HotpotQA
        context: list of (title, [sentence_list]) from HotpotQA

    Returns:
        F1 score between predicted and gold supporting facts.
    """
    title_to_sents: Dict[str, List[str]] = {title: sents for title, sents in context}

    gold_set: set = set()
    for title, sent_idx in gold_supporting_facts:
        if title in title_to_sents and sent_idx < len(title_to_sents[title]):
            gold_set.add(normalize_answer(title_to_sents[title][sent_idx]))

    pred_set: set = {normalize_answer(s) for s in predicted_sentences}

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_evi_f1(
    predicted_evidence: List[str],
    gold_evidence: List[str],
) -> float:
    """
    Evidence F1 for 2WikiMultiHopQA.

    Args:
        predicted_evidence: evidence passages/sentences used by the system
        gold_evidence: gold evidence annotations from 2WikiMHQA

    Returns:
        F1 score.
    """
    pred_set = {normalize_answer(e) for e in predicted_evidence}
    gold_set = {normalize_answer(e) for e in gold_evidence}

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def paired_bootstrap_test(
    scores1: List[float],
    scores2: List[float],
    num_iterations: int = 10000,
    alpha: float = 0.05,
    num_comparisons: int = 1,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Paired bootstrap resampling for statistical significance with
    Bonferroni correction for multiple comparisons.

    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        num_iterations: Number of bootstrap iterations
        alpha: Significance level (before Bonferroni correction)
        num_comparisons: Number of comparisons for Bonferroni correction
        seed: Random seed for reproducibility

    Returns:
        mean_diff: Mean difference (scores1 - scores2)
        ci: Confidence interval (lower, upper)
        p_value: Two-tailed p-value (Bonferroni-corrected)
    """
    n = len(scores1)
    assert n == len(scores2), "Score lists must have same length"

    rng = np.random.RandomState(seed)

    # Bonferroni-corrected alpha
    corrected_alpha = alpha / max(num_comparisons, 1)

    # Observed difference
    observed_diff = np.mean(scores1) - np.mean(scores2)

    # Bootstrap resampling
    bootstrap_diffs = np.empty(num_iterations)
    for i in range(num_iterations):
        indices = rng.choice(n, size=n, replace=True)
        sample1 = np.array([scores1[idx] for idx in indices])
        sample2 = np.array([scores2[idx] for idx in indices])
        bootstrap_diffs[i] = np.mean(sample1) - np.mean(sample2)

    # Confidence interval (using corrected alpha)
    ci_lower = float(np.percentile(bootstrap_diffs, corrected_alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_diffs, (1 - corrected_alpha / 2) * 100))

    # P-value: center the null distribution at zero, then compute
    # the fraction of centered bootstrap diffs that exceed observed.
    centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = float(np.mean(np.abs(centered_diffs) >= np.abs(observed_diff)))

    # Apply Bonferroni correction to p-value
    p_value = min(p_value * num_comparisons, 1.0)

    return observed_diff, (ci_lower, ci_upper), p_value
