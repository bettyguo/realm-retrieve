"""
Evaluation Metrics for Multi-Hop QA and Retrieval Quality

Implements:
1. QA Metrics: EM, F1, Sup-F1, Evi-F1
2. IR Metrics: Recall@k, Precision@k, MRR, nDCG@k (via pytrec_eval)
3. Efficiency Metrics: Retrieval calls, latency, per-call overhead
4. Statistical Significance: Paired bootstrap resampling
"""

from typing import List, Dict, Tuple
import numpy as np
import pytrec_eval
from collections import Counter
import re
import string


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
) -> Dict[str, float]:
    """Compute efficiency metrics."""
    total_calls = sum(retrieval_calls)
    total_latency = sum(latencies)
    
    return {
        "avg_retrieval_calls": np.mean(retrieval_calls),
        "total_retrieval_calls": total_calls,
        "avg_latency": np.mean(latencies),
        "total_latency": total_latency,
        "per_call_overhead": total_latency / total_calls if total_calls > 0 else 0.0,
        "avg_reasoning_tokens": np.mean(reasoning_tokens),
    }


def paired_bootstrap_test(
    scores1: List[float],
    scores2: List[float],
    num_iterations: int = 10000,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Paired bootstrap resampling for statistical significance.
    
    Args:
        scores1: Scores from method 1
        scores2: Scores from method 2
        num_iterations: Number of bootstrap iterations
        alpha: Significance level
        
    Returns:
        mean_diff: Mean difference (scores1 - scores2)
        ci: Confidence interval (lower, upper)
        p_value: Two-tailed p-value
    """
    n = len(scores1)
    assert n == len(scores2), "Score lists must have same length"
    
    # Observed difference
    observed_diff = np.mean(scores1) - np.mean(scores2)
    
    # Bootstrap resampling
    bootstrap_diffs = []
    for _ in range(num_iterations):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        sample1 = [scores1[i] for i in indices]
        sample2 = [scores2[i] for i in indices]
        
        diff = np.mean(sample1) - np.mean(sample2)
        bootstrap_diffs.append(diff)
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    
    return observed_diff, (ci_lower, ci_upper), p_value
