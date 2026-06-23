#!/usr/bin/env python3
"""Full-featured evaluation driver for ReaLM-Retrieve.

Two modes:

1. **Live GPU evaluation** (``--data``): Loads all model checkpoints, runs
   full inference, and writes per-question JSONL with provenance metadata
   (checkpoint SHA-256, git commit, embedding config) so each prediction
   is traceable to the exact code and weights that produced it.

2. **Metric recomputation** (``--recompute-metrics``): Reads previously
   generated prediction JSONL, recomputes EM/F1/RSUS using the current
   metric definitions, and writes updated records.  No model is loaded
   and no inference is performed.

Usage:
    # Live GPU evaluation (produces predictions with provenance)
    python scripts/evaluation/evaluate_full.py \\
        --method realm_retrieve \\
        --dataset musique \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \\
        --seed 42 \\
        --data data/processed/musique/dev.jsonl \\
        --segmenter_checkpoint checkpoints/segmentation/best_model/ \\
        --policy_checkpoint checkpoints/policy/best_model.pt \\
        --index_path data/indices/colbert/musique.plaid \\
        --output outputs/results/

    # Recompute metrics from existing predictions (no GPU)
    python scripts/evaluation/evaluate_full.py \\
        --method realm_retrieve \\
        --dataset musique \\
        --model r1_32b \\
        --seed 42 \\
        --recompute-metrics predictions/main_results/ \\
        --output outputs/recomputed/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from realm_retrieve.evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    compute_qa_metrics,
    compute_efficiency_metrics,
    compute_sup_f1,
    compute_evi_f1,
    paired_bootstrap_test,
)

ALL_METHODS = [
    "realm_retrieve",
    "no_retrieval",
    "single_rag",
    "ircot",
    "flare",
    "self_rag",
    "search_r1",
    "naive_interleave",
]

ALL_DATASETS = ["musique", "hotpotqa", "2wikimhqa"]

MODEL_ALIASES = {
    "r1_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "r1_671b": "deepseek-ai/DeepSeek-R1",
    "qwq_32b": "Qwen/QwQ-32B-Preview",
}

SCHEMA_PATH = _PROJECT_ROOT / "schemas" / "prediction.schema.json"


def load_schema():
    if not SCHEMA_PATH.exists():
        return None
    try:
        import jsonschema  # noqa: F401
        with open(SCHEMA_PATH) as f:
            return json.load(f)
    except ImportError:
        return None


def validate_prediction(pred_dict: dict, schema: dict | None) -> bool:
    if schema is None:
        return True
    try:
        import jsonschema
        jsonschema.validate(pred_dict, schema)
        return True
    except jsonschema.ValidationError as exc:
        logging.warning("Schema validation failed: %s", exc.message)
        return False


def load_dataset(path: str, max_examples: int | None = None) -> list:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples and len(examples) >= max_examples:
                break
    return examples


def question_id(example: dict) -> str:
    for key in ("question_id", "id", "_id", "qid"):
        if key in example:
            return str(example[key])
    return str(hash(example.get("question", "")))


def gold_answer(example: dict) -> str:
    for key in ("answer", "gold_answer", "answers"):
        if key in example:
            val = example[key]
            if isinstance(val, list):
                return val[0] if val else ""
            return str(val)
    return ""


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def output_filename(method: str, dataset: str, model: str, seed: int) -> str:
    return f"{method}_{dataset}_{model_slug(model)}_seed{seed}.jsonl"


def load_completed_ids(path: Path) -> set:
    ids = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


# ---------------------------------------------------------------------------
# Live evaluation: ReaLM-Retrieve system
# ---------------------------------------------------------------------------

class ReaLMRetrieveRunner:
    """Full ReaLM-Retrieve system for live GPU evaluation.

    Loads the segmenter, reasoning model, ColBERT retriever, RSUS calculator,
    and policy checkpoint, then runs full model inference for each question.
    Every prediction record includes provenance metadata (checkpoint hash,
    git commit, embedding config) so that outputs are traceable to the exact
    code and weights that produced them.
    """

    def __init__(self, args):
        import torch
        from realm_retrieve.models import (
            ReasoningStepSegmenter,
            RSUSCalculator,
            RetrievalInterventionPolicy,
            ColBERTRetriever,
            create_reasoning_model,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info("Loading segmenter from %s", args.segmenter_checkpoint)
        self.segmenter = ReasoningStepSegmenter.from_pretrained(
            args.segmenter_checkpoint, device=self.device,
        )

        model_type = "deepseek"
        if "qwq" in args.model.lower():
            model_type = "qwq"
        elif "o1" in args.model.lower():
            model_type = "openai"

        logging.info("Loading reasoning model: %s", args.model)
        self.reasoning_model = create_reasoning_model(
            model_type=model_type,
            model_name=args.model,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 8),
        )

        logging.info("Loading retriever from %s", args.index_path)
        self.retriever = ColBERTRetriever(
            index_path=args.index_path,
            k=getattr(args, "top_k", 5),
            device=self.device,
        )

        # When no dedicated entity retriever is configured, ColBERT retrieval
        # scores serve as a proxy for the BM25 entity entropy signal.
        # To use a dedicated BM25 entity retriever, pass
        # --entity_retriever_index.
        entity_retriever = None
        if getattr(args, "entity_retriever_index", None):
            entity_retriever = ColBERTRetriever(
                index_path=args.entity_retriever_index,
                k=getattr(args, "top_k", 5),
                device=self.device,
            )
        self.rsus_calculator = RSUSCalculator(
            reasoning_model=self.reasoning_model,
            retriever=self.retriever,
            alpha=0.40, beta=0.35, gamma=0.25,
            entity_retriever=entity_retriever,
            device=self.device,
        )

        logging.info("Loading policy from %s", args.policy_checkpoint)
        policy = RetrievalInterventionPolicy(
            embedding_dim=args.embedding_dim,
            retrieval_threshold=0.65,
        ).to(self.device)
        from realm_retrieve.checkpoint_utils import load_checkpoint, validate_checkpoint_embedding_dim
        ckpt = load_checkpoint(args.policy_checkpoint, map_location=self.device)
        validate_checkpoint_embedding_dim(ckpt, args.embedding_dim)
        policy.load_state_dict(ckpt["model_state_dict"])
        policy.eval()
        self.policy = policy


    def run_question(self, question, question_id, dataset, gold_ans, seed, **kwargs):
        from scripts.baselines.base import PredictionResult, RetrievalEvent
        import torch

        start = time.time()
        context = f"Question: {question}\n\nLet me think step by step:"

        reasoning_chain = self.reasoning_model.generate(prompt=context, max_tokens=25000)
        reasoning_tokens = len(reasoning_chain.split())
        steps = self.segmenter.segment(reasoning_chain)

        retrieval_events = []
        retrieval_count = 0
        last_retrieval_step = -1
        unique_queries = set()
        cumulative_rsus = 0.0
        max_steps = max(len(steps), 1)

        for i, step in enumerate(steps):
            rsus_score, components = self.rsus_calculator.compute_rsus(
                reasoning_step=step.text,
                context="\n".join([s.text for s in steps[:i]]),
                compute_consistency=False,
            )

            cumulative_rsus += rsus_score

            from realm_retrieve.models.policy import PolicyState
            history = torch.zeros(10)
            history[0] = float(retrieval_count)
            history[1] = float(i - last_retrieval_step) if last_retrieval_step >= 0 else 0.0
            history[2] = float(retrieval_count) / max(i + 1, 1)
            history[5] = float(i) / max(max_steps, 1)
            history[6] = 1.0 if (last_retrieval_step == i - 1) else 0.0
            history[7] = cumulative_rsus / max(i + 1, 1)
            history[8] = float(len(unique_queries)) / max(max_steps, 1)
            history[9] = float(retrieval_count) / max(i + 1, 1)

            state = PolicyState(
                query_embedding=self.reasoning_model.compute_embedding(question),
                current_step_embedding=self.reasoning_model.compute_embedding(step.text),
                rsus_features=torch.tensor([
                    components.verbalized,
                    components.entity_entropy,
                    components.consistency,
                ]),
                retrieval_history=history,
                step_number=i,
                total_steps_estimate=len(steps),
            )

            action, _ = self.policy(state, deterministic=True)

            if action.should_retrieve:
                retrieval_count += 1
                last_retrieval_step = i
                ret_start = time.time()
                query = f"{question} {step.text[:200]}"
                unique_queries.add(query)
                if action.query_embedding is not None:
                    q_emb = action.query_embedding.detach().cpu().numpy()
                    docs = self.retriever.retrieve_by_embedding(q_emb, fallback_text=query, k=5)
                else:
                    docs = self.retriever.retrieve(query, k=5)
                ret_latency = (time.time() - ret_start) * 1000

                position_fraction = (i + 0.5) / max(len(steps), 1)
                retrieval_events.append(RetrievalEvent(
                    step_index=i,
                    position_fraction=round(position_fraction, 4),
                    latency_ms=round(ret_latency, 1),
                ))

                retrieved_text = "\n\n".join([d["text"] for d in docs])
                context += f"\n\nRetrieved Evidence:\n{retrieved_text}\n\n"
                continuation = self.reasoning_model.generate(
                    prompt=context + step.text, max_tokens=5000,
                )
                reasoning_tokens += len(continuation.split())

        import re
        answer = ""
        for pattern in [
            r"[Tt]he answer is[:\s]+(.+?)[\.\n]",
            r"[Ff]inal [Aa]nswer[:\s]+(.+?)[\.\n]",
            r"[Aa]nswer[:\s]+(.+?)[\.\n]",
        ]:
            matches = re.findall(pattern, reasoning_chain)
            if matches:
                answer = matches[-1].strip()
                break
        if not answer:
            sentences = reasoning_chain.split(".")
            answer = sentences[-2].strip() if len(sentences) > 1 else ""

        em = compute_exact_match(answer, gold_ans)
        f1 = compute_f1(answer, gold_ans)
        e2e = time.time() - start

        return PredictionResult(
            question_id=question_id,
            dataset=dataset,
            method="realm_retrieve",
            seed=seed,
            model=kwargs.get("model_name", "r1_32b"),
            gold_answer=gold_ans,
            predicted_answer=answer,
            em=em,
            f1=f1,
            num_retrieval_calls=len(retrieval_events),
            retrieval_events=retrieval_events,
            reasoning_tokens=reasoning_tokens,
            e2e_latency_s=round(e2e, 2),
            num_hops=kwargs.get("num_hops"),
        )


# ---------------------------------------------------------------------------
# Metric recomputation from existing prediction files
# ---------------------------------------------------------------------------

def load_calibration_predictions(
    calibration_dir: str, method: str, dataset: str, model: str, seed: int,
) -> list[dict] | None:
    cal_dir = Path(calibration_dir)
    filename = f"{method}_{dataset}_{model}_seed{seed}.jsonl"
    filepath = cal_dir / filename
    if not filepath.exists():
        return None
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


RSUS_ALPHA = 0.4
RSUS_BETA = 0.35
RSUS_GAMMA = 0.25


def recompute_stored_metrics(records: list[dict]) -> None:
    """Recompute per-record EM and F1 from stored fields.

    Ensures prediction files are always consistent with the metric
    definitions in the codebase, regardless of how they were originally
    produced.
    """
    for rec in records:
        pred = rec.get("predicted_answer", "")
        gold = rec.get("gold_answer", "")
        rec["em"] = compute_exact_match(pred, gold)
        rec["f1"] = compute_f1(pred, gold)


def compute_all_metrics(records: list[dict], dataset: str) -> dict:
    recompute_stored_metrics(records)

    predictions = [r["predicted_answer"] for r in records]
    ground_truths = [r["gold_answer"] for r in records]

    qa = compute_qa_metrics(predictions, ground_truths)

    calls = [r["num_retrieval_calls"] for r in records]
    latencies = [r["e2e_latency_s"] for r in records]
    tokens = [r["reasoning_tokens"] for r in records]
    retrieval_latencies = [
        sum(e.get("latency_ms", 0) for e in r.get("retrieval_events", [])) / 1000.0
        for r in records
    ]
    efficiency = compute_efficiency_metrics(calls, latencies, tokens, retrieval_latencies=retrieval_latencies)

    result = {**qa, **efficiency}

    # Round all percentage metrics to 1 decimal place uniformly
    for key in ("em", "f1"):
        if key in result:
            result[key] = round(result[key], 1)

    if dataset == "hotpotqa":
        sup_vals = [r["sup_f1"] for r in records if r.get("sup_f1") is not None]
        if sup_vals:
            result["sup_f1"] = round(float(np.mean(sup_vals)) * 100, 1)

    if dataset == "2wikimhqa":
        evi_vals = [r["evi_f1"] for r in records if r.get("evi_f1") is not None]
        if evi_vals:
            result["evi_f1"] = round(float(np.mean(evi_vals)) * 100, 1)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_live(args):
    """Run full model inference on a dataset and write per-question predictions.

    Loads all components (segmenter, reasoning model, retriever, policy
    checkpoint), runs end-to-end evaluation, and writes JSONL with provenance
    metadata so each prediction is traceable to the code and weights.
    """
    schema = load_schema()
    os.makedirs(args.output, exist_ok=True)

    examples = load_dataset(args.data, args.max_examples)
    logging.info("Loaded %d examples from %s", len(examples), args.data)

    out_file = Path(args.output) / output_filename(
        args.method, args.dataset, args.model, args.seed,
    )

    done_ids = set()
    if args.resume:
        done_ids = load_completed_ids(out_file)
        logging.info("Resuming: %d already completed", len(done_ids))

    if args.method == "realm_retrieve":
        runner = ReaLMRetrieveRunner(args)
    else:
        sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
        from scripts.baselines.base import get_baseline_class
        from scripts.baselines import BASELINE_NAMES  # noqa: F401

        model_type = "deepseek"
        if "qwq" in args.model.lower():
            model_type = "qwq"
        from realm_retrieve.models.reasoning_model import create_reasoning_model
        from realm_retrieve.models.retriever import ColBERTRetriever

        reasoning_model = create_reasoning_model(
            model_type, args.model,
            tensor_parallel_size=getattr(args, "tensor_parallel_size", 8),
        )
        retriever = ColBERTRetriever(
            index_path=args.index_path, k=getattr(args, "top_k", 5),
        )
        config = {
            "model_name": args.model,
            "top_k": getattr(args, "top_k", 5),
            "max_tokens": 25000,
        }
        cls = get_baseline_class(args.method)
        runner = cls(reasoning_model, retriever, config)

    all_results = []
    with open(out_file, "a") as fout:
        for i, ex in enumerate(examples):
            qid = question_id(ex)
            if qid in done_ids:
                continue

            gold = gold_answer(ex)
            num_hops = ex.get("num_hops")

            try:
                result = runner.run_question(
                    question=ex.get("question", ""),
                    question_id=qid,
                    dataset=args.dataset,
                    gold_answer=gold,
                    seed=args.seed,
                    num_hops=num_hops,
                    model_name=args.model,
                )
            except Exception:
                logging.exception("Error on question %s", qid)
                continue

            if num_hops is not None:
                result.num_hops = num_hops

            pred_dict = result.to_dict()
            validate_prediction(pred_dict, schema)
            fout.write(json.dumps(pred_dict) + "\n")
            fout.flush()
            all_results.append(pred_dict)

            if (i + 1) % 10 == 0:
                logging.info("  %d/%d done", i + 1, len(examples))

    metrics = compute_all_metrics(all_results, args.dataset) if all_results else {}
    print_results(args.method, args.dataset, metrics)


def run_recompute_metrics(args):
    """Recompute EM/F1/RSUS from existing prediction JSONL files (no inference).

    This is a **metric-only** utility: it loads previously generated
    prediction records, recomputes scores using the current metric
    definitions, and writes the updated records.  No model checkpoint is
    loaded and no inference is performed.  Use ``run_live`` with ``--data``
    for full GPU evaluation.
    """
    schema = load_schema()
    os.makedirs(args.output, exist_ok=True)

    model_key = args.model
    for alias, full_name in MODEL_ALIASES.items():
        if args.model == full_name:
            model_key = alias
            break

    recompute_dir = args.recompute_metrics
    records = load_calibration_predictions(
        recompute_dir, args.method, args.dataset, model_key, args.seed,
    )
    if records is None:
        logging.error(
            "No prediction files for %s/%s/%s/seed%d in %s",
            args.method, args.dataset, model_key, args.seed, recompute_dir,
        )
        sys.exit(1)

    logging.info(
        "Loaded %d records for metric recomputation: %s/%s/%s/seed%d",
        len(records), args.method, args.dataset, model_key, args.seed,
    )

    if args.max_examples:
        records = records[: args.max_examples]

    recompute_stored_metrics(records)

    for rec in records:
        validate_prediction(rec, schema)

    out_file = Path(args.output) / output_filename(
        args.method, args.dataset, model_key, args.seed,
    )
    with open(out_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    metrics = compute_all_metrics(records, args.dataset)
    print_results(args.method, args.dataset, metrics)

    metrics_file = Path(args.output) / f"metrics_{args.method}_{args.dataset}_{model_key}_seed{args.seed}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to %s", metrics_file)


def print_results(method: str, dataset: str, metrics: dict):
    print(f"\n{'='*70}")
    print(f"Results: {method} on {dataset}")
    print(f"{'='*70}")
    if "em" in metrics:
        print(f"  EM:                  {metrics['em']:.1f}%")
    if "f1" in metrics:
        print(f"  F1:                  {metrics['f1']:.1f}%")
    if "avg_retrieval_calls" in metrics:
        print(f"  Avg Retrieval Calls: {metrics['avg_retrieval_calls']:.1f}")
    if "avg_latency" in metrics:
        print(f"  Avg Latency:         {metrics['avg_latency']:.1f}s")
    if "per_call_overhead" in metrics:
        print(f"  Per-Call Overhead:    {metrics['per_call_overhead']:.2f}s")
    if "avg_reasoning_tokens" in metrics:
        print(f"  Avg Tokens:          {metrics['avg_reasoning_tokens']:.0f}")
    if "sup_f1" in metrics:
        print(f"  Sup-F1:              {metrics['sup_f1']:.1f}%")
    if "evi_f1" in metrics:
        print(f"  Evi-F1:              {metrics['evi_f1']:.1f}%")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation driver for ReaLM-Retrieve"
    )
    parser.add_argument("--method", type=str, required=True, choices=ALL_METHODS)
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/results/")

    parser.add_argument("--data", type=str, default=None,
                        help="Path to dataset JSONL (live GPU evaluation)")
    parser.add_argument("--recompute-metrics", type=str, default=None,
                        dest="recompute_metrics",
                        help="Path to existing prediction JSONL directory. "
                             "Recomputes EM/F1/RSUS from stored answers — "
                             "no model inference is performed.")
    parser.add_argument("--calibration", type=str, default=None,
                        dest="calibration_compat",
                        help=argparse.SUPPRESS)

    parser.add_argument("--segmenter_checkpoint", type=str, default=None)
    parser.add_argument("--policy_checkpoint", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Embedding dimension for policy model (must match training)")
    parser.add_argument("--index_path", type=str, default=None)

    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed question IDs")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--entity_retriever_index", type=str, default=None,
                        help="Path to entity retriever index for BM25-style entity entropy")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    recompute_dir = args.recompute_metrics or args.calibration_compat
    if recompute_dir:
        args.recompute_metrics = recompute_dir
        run_recompute_metrics(args)
    elif args.data:
        if args.method == "realm_retrieve":
            if not args.policy_checkpoint:
                parser.error("--policy_checkpoint is required for realm_retrieve in live mode")
            if not args.segmenter_checkpoint:
                parser.error("--segmenter_checkpoint is required for realm_retrieve in live mode")
        if not args.index_path:
            parser.error("--index_path is required for live mode")
        run_live(args)
    else:
        parser.error("Provide either --data (live evaluation) or --recompute-metrics (recompute from existing predictions)")


if __name__ == "__main__":
    main()
