#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Standalone training script for Retrieval Intervention Policy with REINFORCE.

Reward: R = F1(a_pi, a*) - lambda1 * n_ret - lambda2 * t_latency
Training: curriculum learning (lambda1: start -> end over total steps)

Two reward modes:

  --use_online_f1   Full LLM-based reward: runs the reasoning model with
                    policy-chosen retrieval interventions and computes F1
                    from the generated answer. Matches R=F1(a_pi,a*) exactly.
                    Requires --model and --index_path.

  (default)         Evidence-coverage proxy reward: estimates F1 from token
                    overlap between retrieved documents and gold answer.
                    Uses pre-recorded reasoning traces with simulated retrieval.
                    Practical for large-scale training (50K steps).

Usage:
    # Default (proxy reward, practical for large-scale training)
    python scripts/training/train_policy_full.py \
        --train_data data/processed/musique/train.jsonl \
        --dev_data data/processed/musique/dev.jsonl \
        --output checkpoints/policy/ --seed 42

    # Full LLM reward (matches paper's R=F1(a_pi,a*))
    python scripts/training/train_policy_full.py \
        --train_data data/processed/musique/train.jsonl \
        --dev_data data/processed/musique/dev.jsonl \
        --output checkpoints/policy/ --seed 42 \
        --use_online_f1 \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --index_path data/indices/colbert/musique.plaid
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from realm_retrieve.models import (
    PolicyAction,
    PolicyState,
    QueryGen,
    REINFORCETrainer,
    RetrievalInterventionPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL data (one JSON object per line)."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _load_encoder(device: str):
    """Load a shared SBERT encoder for producing step embeddings."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device,
    )


_encoder = None


def _get_encoder(device: str):
    global _encoder
    if _encoder is None:
        _encoder = _load_encoder(device)
    return _encoder


# ---------------------------------------------------------------------------
# Retrieval simulation & dynamic F1 helpers
# ---------------------------------------------------------------------------


def _build_passage_index(
    example: Dict,
    encoder,
    device: str,
) -> Tuple[List[Dict], Optional[torch.Tensor]]:
    """Pre-compute passage embeddings from example's context paragraphs.

    Returns (passages, passage_embeddings) where passages is a list of
    ``{"title": ..., "text": ...}`` dicts and passage_embeddings is a
    ``[N, dim]`` tensor, or ``([], None)`` if no context is available.
    """
    context = example.get("context", [])
    passages: List[Dict] = []
    for entry in context:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        title = entry[0]
        sents = entry[1]
        if isinstance(sents, str):
            sents = [s.strip() for s in sents.split(". ") if s.strip()]
        for sent in (sents if isinstance(sents, list) else [sents]):
            if isinstance(sent, str) and sent.strip():
                passages.append({"title": title, "text": sent.strip()})

    if not passages:
        return passages, None

    passage_texts = [p["text"] for p in passages]
    with torch.no_grad():
        passage_embs = encoder.encode(
            passage_texts, convert_to_tensor=True,
        ).to(device)

    return passages, passage_embs


def _retrieve_from_index(
    question: str,
    step_text: str,
    passages: List[Dict],
    passage_embs: Optional[torch.Tensor],
    encoder,
    device: str,
    k: int = 5,
    query_embedding: Optional[torch.Tensor] = None,
) -> List[Dict]:
    """Retrieve top-k passages from a pre-built embedding index.

    When *query_embedding* is provided (from the policy's query generator),
    it is used as the retrieval query vector.  Otherwise falls back to
    encoding ``question + step_text`` with the SBERT encoder.
    """
    if not passages or passage_embs is None:
        return []

    if query_embedding is not None:
        query_emb = query_embedding.detach().to(device)
    else:
        query = f"{question} {step_text}"
        with torch.no_grad():
            query_emb = encoder.encode(query, convert_to_tensor=True).to(device)

    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)

    pe = passage_embs
    if pe.dim() == 1:
        pe = pe.unsqueeze(0)

    sims = torch.nn.functional.cosine_similarity(query_emb, pe, dim=-1)
    topk = min(k, len(passages))
    _, indices = sims.topk(topk)
    return [passages[idx] for idx in indices.tolist()]


def _get_baseline_f1(example: Dict) -> float:
    """Return the no-retrieval baseline F1.

    Uses the predicted_answer (generated without retrieval) when available;
    otherwise falls back to a conservative default so that useful retrieval
    is always incentivised.
    """
    answer = example.get("answer", "")
    if not answer:
        return 0.0

    if "predicted_answer" in example:
        from realm_retrieve.evaluation.metrics import compute_f1
        return compute_f1(example["predicted_answer"], answer)

    if "answer_f1" in example:
        return float(example["answer_f1"])

    return 0.2


def _compute_estimated_f1(
    example: Dict,
    retrieved_evidence: List[Dict],
) -> float:
    """Estimate F1 as a function of retrieved evidence quality.

    When ``supporting_facts`` are present, measures coverage of gold
    evidence titles.  Falls back to token overlap between retrieved
    documents and the ground-truth answer.
    """
    answer = example.get("answer", "")
    if not answer:
        return 0.0

    baseline_f1 = _get_baseline_f1(example)

    if not retrieved_evidence:
        return baseline_f1

    supporting_facts = example.get("supporting_facts", [])

    if supporting_facts:
        sup_titles = {sf[0] for sf in supporting_facts if sf}
        retrieved_titles = {doc.get("title", "") for doc in retrieved_evidence}

        covered = len(sup_titles & retrieved_titles)
        total = max(len(sup_titles), 1)
        coverage = covered / total

        max_improvement = 1.0 - baseline_f1
        return min(baseline_f1 + max_improvement * coverage, 1.0)

    from realm_retrieve.evaluation.metrics import normalize_answer

    answer_tokens = set(normalize_answer(answer).split())
    if not answer_tokens:
        return baseline_f1

    all_doc_text = " ".join(doc.get("text", "") for doc in retrieved_evidence)
    doc_tokens = set(normalize_answer(all_doc_text).split())

    covered = len(answer_tokens & doc_tokens)
    coverage = covered / len(answer_tokens)

    max_improvement = 1.0 - baseline_f1
    improvement = max_improvement * (1 - (1 - coverage) ** 2)
    return min(baseline_f1 + improvement, 1.0)


def _compute_online_f1(
    question: str,
    answer: str,
    retrieved_evidence: List[Dict],
    reasoning_model,
) -> float:
    """Generate an answer with the reasoning model and compute real F1."""
    evidence_text = "\n\n".join(doc.get("text", "") for doc in retrieved_evidence)

    prompt = (
        f"Use the following evidence to help answer the question.\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Question: {question}\n\n"
        f"Think step by step and provide your answer."
    )

    try:
        generated = reasoning_model.generate(prompt, max_tokens=2000)
        gen_answer = generated.strip()
        for marker in ("the answer is", "answer:", "therefore,"):
            if marker in gen_answer.lower():
                gen_answer = gen_answer.lower().split(marker)[-1].strip()
                break

        from realm_retrieve.evaluation.metrics import compute_f1
        return compute_f1(gen_answer, answer)
    except Exception:
        return _get_baseline_f1({"answer": answer})


def extract_states_from_example(
    example: Dict,
    device: str,
    embedding_dim: int = 768,
    rsus_calculator=None,
) -> Tuple[List[PolicyState], int]:
    """
    Extract policy states from a training example using actual SBERT
    embeddings for query and step representations.

    When ``rsus_calculator`` is provided, computes full RSUS components
    (verbalized confidence via LLM, entity entropy via retrieval, consistency
    via multi-sample agreement) matching the paper's formulation.  Otherwise
    falls back to pre-computed values from the data or heuristic proxies.

    Each example is expected to contain reasoning trace segments (either
    from a segmenter or pre-segmented).

    Returns:
        states: list of PolicyState objects
        num_segments: total number of segments
    """
    encoder = _get_encoder(device)

    segments = example.get("segments", example.get("steps", []))
    if not segments and "reasoning_trace" in example:
        trace = example["reasoning_trace"]
        segments = [{"text": s.strip()} for s in trace.split("\n") if s.strip()]

    num_segments = max(len(segments), 1)
    question = example.get("question", "")

    q_emb = encoder.encode(question, convert_to_tensor=True).to(device)

    # Encode each segment
    seg_texts = [
        seg.get("text", seg) if isinstance(seg, dict) else str(seg)
        for seg in segments
    ]
    if seg_texts:
        seg_embs = encoder.encode(seg_texts, convert_to_tensor=True).to(device)
    else:
        seg_embs = torch.zeros(1, 768, device=device)

    states: List[PolicyState] = []
    retrieval_count = 0
    last_retrieval_step = -1

    unique_queries = set()
    cumulative_rsus = 0.0
    max_steps = max(len(seg_texts), 1)

    for i in range(len(seg_texts)):
        step_emb = seg_embs[i]

        # Build retrieval history features from actual retrieval state
        history = torch.zeros(10, device=device)
        history[0] = float(retrieval_count)
        history[1] = float(i - last_retrieval_step) if last_retrieval_step >= 0 else 0.0
        history[2] = float(retrieval_count) / max(i + 1, 1)
        history[3] = 0.0  # average retrieval latency (normalized by 1000ms); not tracked per-step in training
        history[4] = 0.0  # whether last retrieval was useful (1.0 if F1 improved)
        history[5] = float(i) / max(max_steps, 1)  # fraction of total steps completed
        history[6] = 1.0 if (last_retrieval_step == i - 1) else 0.0  # retrieval in last step
        history[7] = cumulative_rsus / max(i + 1, 1)  # cumulative RSUS score (normalized)
        history[8] = float(len(unique_queries)) / max(max_steps, 1)  # unique queries (normalized)
        history[9] = float(retrieval_count) / max(i + 1, 1)  # ratio of retrievals to steps (backward compat)

        # RSUS features: live computation > pre-computed components > scalar decomposition
        rsus_raw = example.get("rsus_scores", None)
        rsus_components = example.get("rsus_components", None)
        if rsus_calculator is not None:
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            context_text = "\n".join(seg_texts[:i])
            score, comp = rsus_calculator.compute_rsus(
                reasoning_step=step_text, context=context_text,
            )
            rsus_feat = torch.tensor(
                [comp.verbalized, comp.entity_entropy, comp.consistency],
                dtype=torch.float, device=device,
            )
            cumulative_rsus += score
        elif rsus_components and i < len(rsus_components):
            # Full 3-component RSUS available from data
            comp = rsus_components[i]
            rsus_feat = torch.tensor(
                [comp[0], comp[1], comp[2]], dtype=torch.float, device=device,
            )
        elif rsus_raw and i < len(rsus_raw):
            # Scalar RSUS only: decompose into 3 components
            u_verb = rsus_raw[i]
            # Estimate U_ent from entity count in the step text
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = None
                if nlp is not None:
                    doc = nlp(step_text)
                    u_ent = min(len(doc.ents) / 10.0, 1.0)
                else:
                    u_ent = min(len(step_text.split()) / 100.0, 1.0)
            except Exception:
                # Fallback: rough word-count heuristic as entity proxy
                u_ent = min(len(step_text.split()) / 100.0, 1.0)
            # Consistency proxy: non-zero if discourse markers present
            discourse_markers = ["however", "but", "although", "therefore",
                                 "moreover", "furthermore", "nevertheless",
                                 "in contrast", "on the other hand"]
            text_lower = step_text.lower()
            has_discourse = any(m in text_lower for m in discourse_markers)
            u_consist = u_verb * 0.5 if has_discourse else 0.0
            rsus_feat = torch.tensor(
                [u_verb, u_ent, u_consist], dtype=torch.float, device=device,
            )
            cumulative_rsus += u_verb
        else:
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            word_count = len(step_text.split())
            u_verb = min(word_count / 50.0, 1.0)
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = None
                if nlp is not None:
                    doc = nlp(step_text)
                    u_ent = min(len(doc.ents) / 10.0, 1.0)
                else:
                    u_ent = min(word_count / 100.0, 1.0)
            except Exception:
                u_ent = min(word_count / 100.0, 1.0)
            discourse_markers = [
                "however", "but", "although", "therefore",
                "moreover", "furthermore", "nevertheless",
                "in contrast", "on the other hand",
            ]
            has_discourse = any(m in step_text.lower() for m in discourse_markers)
            u_consist = u_verb * 0.5 if has_discourse else 0.0
            rsus_feat = torch.tensor(
                [u_verb, u_ent, u_consist], dtype=torch.float, device=device,
            )
            cumulative_rsus += u_verb

        state = PolicyState(
            query_embedding=q_emb.detach(),
            current_step_embedding=step_emb.detach(),
            rsus_features=rsus_feat,
            retrieval_history=history,
            step_number=min(i, 199),
            total_steps_estimate=num_segments,
        )
        states.append(state)

    return states, num_segments


def run_episode(
    policy: RetrievalInterventionPolicy,
    example: Dict,
    device: str,
    embedding_dim: int = 768,
    retriever=None,
    reasoning_model=None,
    rsus_calculator=None,
    deterministic: bool = False,
) -> Tuple[List[PolicyState], List[PolicyAction], float, int, float]:
    """
    Run a single episode with incremental state updates and
    action-dependent reward.

    Unlike pre-computing all states and using a constant F1, this:

    1. Builds states **incrementally** -- retrieved evidence is folded
       into the step embeddings of all subsequent states so that the
       policy observes different observations after a retrieval.
    2. Captures retrieval results (from a live retriever **or**
       simulated from the example's context paragraphs).
    3. Computes F1 as a **function** of the policy's retrieval
       decisions, not as a fixed property of the data record.

    Returns:
        states, actions, f1_score, num_retrievals, total_latency
    """
    encoder = _get_encoder(device)

    # ---- parse segments ----
    segments = example.get("segments", example.get("steps", []))
    if not segments and "reasoning_trace" in example:
        trace = example["reasoning_trace"]
        segments = [{"text": s.strip()} for s in trace.split("\n") if s.strip()]

    num_segments = max(len(segments), 1)
    question = example.get("question", "")
    answer = example.get("answer", "")

    seg_texts = [
        seg.get("text", seg) if isinstance(seg, dict) else str(seg)
        for seg in segments
    ]

    with torch.no_grad():
        q_emb = encoder.encode(question, convert_to_tensor=True).to(device)

    # Pre-build passage index for offline retrieval simulation
    passages: List[Dict] = []
    passage_embs: Optional[torch.Tensor] = None
    if retriever is None:
        passages, passage_embs = _build_passage_index(example, encoder, device)

    states: List[PolicyState] = []
    actions: List[PolicyAction] = []
    num_retrievals = 0
    total_latency = 0.0
    retrieved_evidence: List[Dict] = []

    retrieval_count = 0
    last_retrieval_step = -1
    last_retrieval_useful = 0.0
    unique_queries: set = set()
    cumulative_rsus = 0.0
    max_steps = max(len(seg_texts), 1)

    for i in range(len(seg_texts)):
        # ---- step embedding (augmented by prior retrieval evidence) ----
        with torch.no_grad():
            if retrieved_evidence:
                evidence_summary = " ".join(
                    doc.get("text", "")[:200]
                    for doc in retrieved_evidence[-5:]
                )
                augmented_text = (
                    f"{seg_texts[i]} [Evidence: {evidence_summary}]"
                )
                step_emb = encoder.encode(
                    augmented_text, convert_to_tensor=True,
                ).to(device)
            else:
                step_emb = encoder.encode(
                    seg_texts[i], convert_to_tensor=True,
                ).to(device)

        # ---- retrieval history features ----
        history = torch.zeros(10, device=device)
        history[0] = float(retrieval_count)
        history[1] = (
            float(i - last_retrieval_step) if last_retrieval_step >= 0 else 0.0
        )
        history[2] = float(retrieval_count) / max(i + 1, 1)
        history[3] = (
            total_latency / retrieval_count / 1.0
            if retrieval_count > 0
            else 0.0
        )
        history[4] = last_retrieval_useful
        history[5] = float(i) / max(max_steps, 1)
        history[6] = 1.0 if (last_retrieval_step == i - 1) else 0.0
        history[7] = cumulative_rsus / max(i + 1, 1)
        history[8] = float(len(unique_queries)) / max(max_steps, 1)
        history[9] = float(retrieval_count) / max(i + 1, 1)

        # ---- RSUS features: live > pre-computed > heuristic ----
        rsus_raw = example.get("rsus_scores", None)
        rsus_components = example.get("rsus_components", None)
        if rsus_calculator is not None:
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            context_text = "\n".join(seg_texts[:i])
            score, comp = rsus_calculator.compute_rsus(
                reasoning_step=step_text, context=context_text,
            )
            rsus_feat = torch.tensor(
                [comp.verbalized, comp.entity_entropy, comp.consistency],
                dtype=torch.float, device=device,
            )
            cumulative_rsus += score
        elif rsus_components and i < len(rsus_components):
            comp = rsus_components[i]
            rsus_feat = torch.tensor(
                [comp[0], comp[1], comp[2]], dtype=torch.float, device=device,
            )
        elif rsus_raw and i < len(rsus_raw):
            u_verb = rsus_raw[i]
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = None
                if nlp is not None:
                    doc = nlp(step_text)
                    u_ent = min(len(doc.ents) / 10.0, 1.0)
                else:
                    u_ent = min(len(step_text.split()) / 100.0, 1.0)
            except Exception:
                u_ent = min(len(step_text.split()) / 100.0, 1.0)
            discourse_markers = [
                "however", "but", "although", "therefore",
                "moreover", "furthermore", "nevertheless",
                "in contrast", "on the other hand",
            ]
            text_lower = step_text.lower()
            has_discourse = any(m in text_lower for m in discourse_markers)
            u_consist = u_verb * 0.5 if has_discourse else 0.0
            rsus_feat = torch.tensor(
                [u_verb, u_ent, u_consist], dtype=torch.float, device=device,
            )
            cumulative_rsus += u_verb
        else:
            # Estimate RSUS from step text when no pre-computed values available
            step_text = seg_texts[i] if i < len(seg_texts) else ""
            word_count = len(step_text.split())
            u_verb = min(word_count / 50.0, 1.0)
            try:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = None
                if nlp is not None:
                    doc = nlp(step_text)
                    u_ent = min(len(doc.ents) / 10.0, 1.0)
                else:
                    u_ent = min(word_count / 100.0, 1.0)
            except Exception:
                u_ent = min(word_count / 100.0, 1.0)
            discourse_markers = [
                "however", "but", "although", "therefore",
                "moreover", "furthermore", "nevertheless",
                "in contrast", "on the other hand",
            ]
            has_discourse = any(m in step_text.lower() for m in discourse_markers)
            u_consist = u_verb * 0.5 if has_discourse else 0.0
            rsus_feat = torch.tensor(
                [u_verb, u_ent, u_consist], dtype=torch.float, device=device,
            )
            cumulative_rsus += u_verb

        # ---- build state ----
        state = PolicyState(
            query_embedding=q_emb.detach(),
            current_step_embedding=step_emb.detach(),
            rsus_features=rsus_feat,
            retrieval_history=history,
            step_number=min(i, 199),
            total_steps_estimate=num_segments,
        )
        states.append(state)

        # ---- policy decision ----
        with torch.no_grad():
            action, _info = policy(state, deterministic=deterministic)
        actions.append(action)

        # ---- execute retrieval ----
        if action.should_retrieve:
            num_retrievals += 1
            retrieval_count += 1
            last_retrieval_step = i
            unique_queries.add(question)

            if retriever is not None:
                t0 = time.perf_counter()
                try:
                    fallback_query = f"{question} {seg_texts[i][:200]}"
                    if action.query_embedding is not None:
                        q_emb = action.query_embedding.detach().cpu().numpy()
                        docs = retriever.retrieve_by_embedding(
                            q_emb, fallback_text=fallback_query, k=5,
                        )
                    else:
                        docs = retriever.retrieve(fallback_query, k=5)
                    retrieved_evidence.extend(docs)
                except Exception:
                    docs = []
                total_latency += time.perf_counter() - t0
            else:
                docs = _retrieve_from_index(
                    question, seg_texts[i], passages, passage_embs,
                    encoder, device, k=5,
                    query_embedding=action.query_embedding,
                )
                retrieved_evidence.extend(docs)
                total_latency += 0.25

            # Track whether this retrieval was useful (for history[4])
            if docs:
                supporting_facts = example.get("supporting_facts", [])
                if supporting_facts:
                    sup_titles = {sf[0] for sf in supporting_facts if sf}
                    doc_titles = {d.get("title", "") for d in docs}
                    last_retrieval_useful = (
                        1.0 if (sup_titles & doc_titles) else 0.0
                    )
                else:
                    last_retrieval_useful = 0.5

    # ---- compute F1 as a function of retrieval actions ----
    if reasoning_model is not None and retrieved_evidence:
        f1_score = _compute_online_f1(
            question, answer, retrieved_evidence, reasoning_model,
        )
    elif retrieved_evidence:
        f1_score = _compute_estimated_f1(example, retrieved_evidence)
    else:
        f1_score = _get_baseline_f1(example)

    return states, actions, f1_score, num_retrievals, total_latency


# ---------------------------------------------------------------------------
# Dev evaluation
# ---------------------------------------------------------------------------


def evaluate_dev(
    policy: RetrievalInterventionPolicy,
    dev_data: List[Dict],
    device: str,
    embedding_dim: int = 768,
) -> Dict[str, float]:
    """
    Evaluate policy on the dev set with action-dependent F1.

    Uses the same dynamic episode runner as training so that the dev
    F1 reflects the actual quality of the policy's retrieval decisions.
    """
    policy.eval()
    total_f1 = 0.0
    total_retrievals = 0
    total_segments = 0

    for example in dev_data:
        states, actions, f1_score, num_retrievals, total_latency = run_episode(
            policy, example, device, embedding_dim=embedding_dim,
            deterministic=True,
        )

        total_f1 += f1_score
        total_retrievals += num_retrievals
        total_segments += len(states) if states else 1

    n = max(len(dev_data), 1)
    dev_f1 = total_f1 / n
    retrieval_rate = total_retrievals / max(total_segments, 1)

    policy.train()
    return {"dev_f1": dev_f1, "retrieval_rate": retrieval_rate}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_metrics(
    log_file: Optional[str],
    step: int,
    steps_per_epoch: int,
    loss: float,
    reward: float,
    retrieval_rate: float,
    dev_f1: float,
    lambda1: float,
    lr: float,
) -> None:
    """Append a JSONL log entry."""
    if log_file is None:
        return

    entry = {
        "step": step,
        "epoch": step / max(steps_per_epoch, 1),
        "loss": loss,
        "reward": reward,
        "retrieval_rate": retrieval_rate,
        "dev_f1": dev_f1,
        "lambda1": lambda1,
        "lr": lr,
        "metric_name": "dev_f1",
        "metric_value": dev_f1,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialize policy
    # ------------------------------------------------------------------
    policy = RetrievalInterventionPolicy(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        num_heads=args.policy_heads,
        retrieval_threshold=args.threshold,
    )

    # ------------------------------------------------------------------
    # Initialize trainer
    # ------------------------------------------------------------------
    trainer = REINFORCETrainer(
        policy=policy,
        learning_rate=args.lr,
        lambda1_start=args.lambda1_start,
        lambda1_end=args.lambda1_end,
        lambda2=args.lambda2,
        entropy_coef=args.entropy_coef,
        baseline_momentum=args.baseline_momentum,
        device=str(device),
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading training data ...")
    train_data = load_jsonl(args.train_data)
    print(f"  {len(train_data)} training examples loaded")

    dev_data: Optional[List[Dict]] = None
    if args.dev_data:
        print("Loading dev data ...")
        dev_data = load_jsonl(args.dev_data)
        print(f"  {len(dev_data)} dev examples loaded")

    steps_per_epoch = max(len(train_data), 1)

    # ------------------------------------------------------------------
    # Optional: load retriever and reasoning model for online F1 reward
    # ------------------------------------------------------------------
    retriever = None
    reasoning_model = None
    if args.use_online_f1:
        if not args.model or not args.index_path:
            raise ValueError("--model and --index_path required with --use_online_f1")
        from realm_retrieve.models.reasoning_model import create_reasoning_model
        from realm_retrieve.models.retriever import ColBERTRetriever
        reasoning_model = create_reasoning_model(
            "deepseek", args.model,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        retriever = ColBERTRetriever(
            index_path=args.index_path, k=args.top_k,
        )
        from realm_retrieve.models.rsus import RSUSCalculator
        rsus_calculator = RSUSCalculator(
            reasoning_model=reasoning_model,
            retriever=retriever,
            device=str(device),
        )
        print("Online F1 reward enabled: using LLM + retriever for R=F1(a_pi,a*) and live RSUS")
    else:
        rsus_calculator = None
        print("Using evidence-coverage proxy reward (default; pass --use_online_f1 for full LLM reward)")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    running_reward = 0.0
    running_loss = 0.0
    running_retrievals = 0
    running_segments = 0
    latest_dev_f1 = 0.0

    print(f"\nStarting training for {args.steps} steps ...")
    batch_states: List = []
    batch_actions: List = []
    batch_rewards_data: List = []
    episode_state_counts: List = []

    for step in tqdm(range(args.steps), desc="Training"):
        # Sample a training example
        idx = step % len(train_data)
        example = train_data[idx]

        # Run episode (with LLM + retriever + RSUS when --use_online_f1, proxy otherwise)
        states, actions, f1_score, num_retrievals, total_latency = run_episode(
            policy, example, str(device), embedding_dim=args.embedding_dim,
            retriever=retriever, reasoning_model=reasoning_model,
            rsus_calculator=rsus_calculator,
        )

        batch_states.extend(states)
        batch_actions.extend(actions)
        batch_rewards_data.append((f1_score, num_retrievals, total_latency))
        episode_state_counts.append(len(states))

        # Accumulate a mini-batch before updating
        if len(batch_rewards_data) < args.batch_size and step < args.steps - 1:
            continue

        # Build per-state rewards: each state gets its own episode's reward
        per_state_rewards = []
        for ep_idx, (ep_f1, ep_ret, ep_lat) in enumerate(batch_rewards_data):
            for _ in range(episode_state_counts[ep_idx]):
                per_state_rewards.append((ep_f1, ep_ret, ep_lat))

        # Training step with per-state rewards
        metrics = trainer.train_step(
            states=batch_states,
            actions=batch_actions,
            rewards=per_state_rewards,
        )

        batch_states = []
        batch_actions = []
        batch_rewards_data = []
        episode_state_counts = []

        # Accumulate running stats
        running_reward += metrics["reward"]
        running_loss += metrics["loss"]
        num_segments = len(states) if states else 1
        running_retrievals += num_retrievals
        running_segments += num_segments

        # ----- Logging every 100 steps -----
        if step > 0 and step % 100 == 0:
            retrieval_rate = running_retrievals / max(running_segments, 1)

            log_metrics(
                log_file=args.log_file,
                step=step,
                steps_per_epoch=steps_per_epoch,
                loss=running_loss / 100,
                reward=running_reward / 100,
                retrieval_rate=retrieval_rate,
                dev_f1=latest_dev_f1,
                lambda1=metrics["lambda1"],
                lr=args.lr,
            )

            # Reset running accumulators
            running_reward = 0.0
            running_loss = 0.0
            running_retrievals = 0
            running_segments = 0

        # ----- Dev evaluation & summary every 1000 steps -----
        if step > 0 and step % 1000 == 0:
            if dev_data is not None:
                dev_metrics = evaluate_dev(
                    policy, dev_data, str(device),
                    embedding_dim=args.embedding_dim,
                )
                latest_dev_f1 = dev_metrics["dev_f1"]
                dev_ret_rate = dev_metrics["retrieval_rate"]
            else:
                dev_ret_rate = 0.0

            print(
                f"\n[Step {step}]  "
                f"reward={metrics['reward']:.4f}  "
                f"loss={metrics['loss']:.4f}  "
                f"lambda1={metrics['lambda1']:.4f}  "
                f"dev_f1={latest_dev_f1:.4f}  "
                f"dev_ret_rate={dev_ret_rate:.4f}"
            )

        # ----- Checkpointing -----
        if step > 0 and step % args.save_every == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{step}.pt")
            trainer.save_checkpoint(ckpt_path, step)
            print(f"  Saved checkpoint -> {ckpt_path}")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output, "final_model.pt")
    trainer.save_checkpoint(final_path, args.steps)
    print(f"\nTraining complete! Final model saved to {final_path}")

    # Final dev evaluation
    if dev_data is not None:
        dev_metrics = evaluate_dev(
            policy, dev_data, str(device),
            embedding_dim=args.embedding_dim,
        )
        print(
            f"Final dev_f1={dev_metrics['dev_f1']:.4f}  "
            f"retrieval_rate={dev_metrics['retrieval_rate']:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Retrieval Intervention Policy (standalone, argparse)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        "--train_data", required=True,
        help="Training data JSONL path",
    )
    parser.add_argument(
        "--dev_data", default=None,
        help="Dev/validation data JSONL path",
    )
    parser.add_argument(
        "--segmenter_checkpoint", default=None,
        help="Path to trained segmenter checkpoint",
    )

    # Policy architecture
    parser.add_argument(
        "--policy_hidden_dim", type=int, default=512,
        help="Policy hidden dimension",
    )
    parser.add_argument(
        "--policy_layers", type=int, default=4,
        help="Policy transformer layers",
    )
    parser.add_argument(
        "--policy_heads", type=int, default=8,
        help="Policy attention heads",
    )

    # QueryGen architecture
    parser.add_argument(
        "--querygen_hidden_dim", type=int, default=512,
        help="QueryGen hidden dimension",
    )
    parser.add_argument(
        "--querygen_heads", type=int, default=8,
        help="QueryGen attention heads",
    )
    parser.add_argument(
        "--querygen_output_dim", type=int, default=768,
        help="QueryGen output dimension (matches ColBERTv2 query space)",
    )

    # Embeddings
    parser.add_argument(
        "--embedding_dim", type=int, default=768,
        help="Input embedding dimension (768 for all-mpnet-base-v2)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--steps", type=int, default=50000,
        help="Total training steps",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size",
    )

    # Reward shaping
    parser.add_argument(
        "--lambda1_start", type=float, default=0.5,
        help="Initial lambda1 for retrieval penalty",
    )
    parser.add_argument(
        "--lambda1_end", type=float, default=0.1,
        help="Final lambda1",
    )
    parser.add_argument(
        "--lambda2", type=float, default=0.01,
        help="Latency penalty weight",
    )

    # Regularization
    parser.add_argument(
        "--entropy_coef", type=float, default=0.01,
        help="Entropy regularization coefficient",
    )
    parser.add_argument(
        "--baseline_momentum", type=float, default=0.95,
        help="EMA baseline momentum",
    )

    # Checkpointing / logging
    parser.add_argument(
        "--save_every", type=int, default=5000,
        help="Checkpoint save frequency in steps",
    )
    parser.add_argument(
        "--output", default="checkpoints/policy_full",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--log_file", default=None,
        help="JSONL log file path",
    )

    # Online F1 (full LLM-based reward)
    parser.add_argument(
        "--use_online_f1", action="store_true",
        help="Compute F1 from LLM-generated answers instead of evidence "
             "coverage proxy. Requires --model and --index_path. "
             "Much slower (~100x) but matches the paper's R=F1(a_pi,a*) reward.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Reasoning model name/path (required when --use_online_f1)",
    )
    parser.add_argument(
        "--index_path", default=None,
        help="ColBERT PLAID index path (required when --use_online_f1)",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=8,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of passages to retrieve per query",
    )

    # Misc
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Retrieval threshold",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
