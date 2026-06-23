#!/usr/bin/env python3
"""
Evaluation Script for ReaLM-Retrieve

Evaluates complete system on multi-hop QA benchmarks.
Computes QA metrics, retrieval quality, efficiency, and statistical significance.

Usage:
    python evaluate.py configs/experiments/evaluate.yaml dataset=musique
"""

import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from realm_retrieve.models import (
    ReasoningStepSegmenter,
    RSUSCalculator,
    RetrievalInterventionPolicy,
    ColBERTRetriever,
    create_reasoning_model,
)
from realm_retrieve.models.query_gen import QueryGen
from realm_retrieve.models.compression import ImplicitCompression
from realm_retrieve.models.speculative_cache import SpeculativeCache
from realm_retrieve.models.kv_cache import KVCacheManager
from realm_retrieve.models.injection import RetrievalInjector
from realm_retrieve.evaluation.metrics import (
    compute_qa_metrics,
    compute_retrieval_metrics,
    compute_efficiency_metrics,
    paired_bootstrap_test,
)


class ReaLMRetrieveSystem:
    """Complete ReaLM-Retrieve system."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models
        print("Loading models...")
        
        # Segmentation
        self.segmenter = ReasoningStepSegmenter.from_pretrained(
            cfg.model.segmentation_checkpoint,
            device=self.device,
        )
        
        # Reasoning model
        self.reasoning_model = create_reasoning_model(
            model_type=cfg.model.reasoning_model_type,
            model_name=cfg.model.reasoning_model_name,
            tensor_parallel_size=cfg.model.tensor_parallel_size,
        )
        
        # Retriever
        self.retriever = ColBERTRetriever(
            index_path=cfg.retrieval.index_path,
            checkpoint=cfg.retrieval.checkpoint,
            k=cfg.retrieval.k,
            device=self.device,
        )
        
        # RSUS calculator
        # When no dedicated entity retriever is configured, ColBERT retrieval
        # scores serve as a proxy for the BM25 entity entropy signal.
        # To use a dedicated BM25 entity retriever, set
        # rsus.entity_retriever_index in the config.
        entity_retriever = None
        if hasattr(cfg.rsus, "entity_retriever_index") and cfg.rsus.entity_retriever_index:
            entity_retriever = ColBERTRetriever(
                index_path=cfg.rsus.entity_retriever_index,
                k=cfg.retrieval.k,
                device=self.device,
            )
        self.rsus_calculator = RSUSCalculator(
            reasoning_model=self.reasoning_model,
            retriever=self.retriever,
            alpha=cfg.rsus.alpha,
            beta=cfg.rsus.beta,
            gamma=cfg.rsus.gamma,
            entity_retriever=entity_retriever,
            device=self.device,
        )
        
        # Policy
        policy_model = RetrievalInterventionPolicy(
            embedding_dim=cfg.model.embedding_dim,
            retrieval_threshold=cfg.policy.threshold,
        ).to(self.device)
        
        # Load checkpoint
        from realm_retrieve.checkpoint_utils import load_checkpoint, validate_checkpoint_embedding_dim
        checkpoint = load_checkpoint(
            cfg.model.policy_checkpoint,
            map_location=self.device,
        )
        validate_checkpoint_embedding_dim(checkpoint, cfg.model.embedding_dim)
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        policy_model.eval()
        
        self.policy = policy_model

        # Query generation module
        querygen_path = getattr(cfg.model, "querygen_checkpoint", None)
        if querygen_path is None:
            # Co-located with policy checkpoint: assume sibling directory
            querygen_path = str(Path(cfg.model.policy_checkpoint).parent / "query_gen")
        self.query_gen = QueryGen.from_pretrained(querygen_path, device=self.device)
        self.query_gen.to(self.device)
        self.query_gen.eval()

        # Implicit compression
        compression_tau = getattr(cfg.integration, "compression_tau_rel", 0.45) if hasattr(cfg, "integration") else 0.45
        self.compression = ImplicitCompression(tau_rel=compression_tau)

        # SBERT encoder (shared by SpeculativeCache and ImplicitCompression)
        from sentence_transformers import SentenceTransformer
        sbert_encoder = SentenceTransformer("all-mpnet-base-v2")

        # Speculative cache
        spec_threshold = getattr(cfg.integration, "speculative_similarity_threshold", 0.85) if hasattr(cfg, "integration") else 0.85
        self.speculative_cache = SpeculativeCache(
            retriever=self.retriever,
            encoder=sbert_encoder,
            similarity_threshold=spec_threshold,
        )

        # KV-cache manager
        self.kv_cache_manager = KVCacheManager()

        # Retrieval injector
        self.injector = RetrievalInjector(
            kv_cache_manager=self.kv_cache_manager,
            compression=self.compression,
            speculative_cache=self.speculative_cache,
        )

    @staticmethod
    def _build_retrieval_history(
        retrieval_count, step_idx, last_retrieval_step,
        max_steps, cumulative_rsus, unique_queries,
    ):
        history = torch.zeros(10)
        history[0] = float(retrieval_count)
        history[1] = float(step_idx - last_retrieval_step) if last_retrieval_step >= 0 else 0.0
        history[2] = float(retrieval_count) / max(step_idx + 1, 1)
        history[5] = float(step_idx) / max(max_steps, 1)
        history[6] = 1.0 if (last_retrieval_step == step_idx - 1) else 0.0
        history[7] = cumulative_rsus / max(step_idx + 1, 1)
        history[8] = float(len(unique_queries)) / max(max_steps, 1)
        history[9] = float(retrieval_count) / max(step_idx + 1, 1)
        return history

    def answer_question(self, question: str):
        """Answer a single question using ReaLM-Retrieve."""
        import time

        # Clear speculative cache between questions
        self.speculative_cache.clear()

        # Initialize reasoning
        context = f"Question: {question}\n\nLet me think step by step:"

        # Track metrics
        retrieval_calls = 0
        latency_total = 0.0
        reasoning_tokens = 0
        last_retrieval_step = -1
        unique_queries = set()
        cumulative_rsus = 0.0

        # Generate initial reasoning
        reasoning_chain = self.reasoning_model.generate(
            prompt=context,
            max_tokens=25000,
        )
        reasoning_tokens += len(reasoning_chain.split())

        # Segment into steps
        steps = self.segmenter.segment(reasoning_chain)

        # Pre-compute question embedding (reused across steps)
        question_emb = self.reasoning_model.compute_embedding(question)

        # Process each step
        for i, step in enumerate(steps):
            # Compute RSUS
            rsus_score, rsus_components = self.rsus_calculator.compute_rsus(
                reasoning_step=step.text,
                context="\n".join([s.text for s in steps[:i]]),
                compute_consistency=False,
            )

            # Policy decision
            from realm_retrieve.models.policy import PolicyState

            step_emb = self.reasoning_model.compute_embedding(step.text)

            state = PolicyState(
                query_embedding=question_emb,
                current_step_embedding=step_emb,
                rsus_features=torch.tensor([
                    rsus_components.verbalized,
                    rsus_components.entity_entropy,
                    rsus_components.consistency,
                ]),
                retrieval_history=self._build_retrieval_history(
                    retrieval_calls, i, last_retrieval_step,
                    len(steps), cumulative_rsus, unique_queries,
                ),
                step_number=i,
                total_steps_estimate=len(steps),
            )

            cumulative_rsus += rsus_score

            action, _ = self.policy(state, deterministic=True)

            # Retrieve if policy says so
            if action.should_retrieve:
                retrieval_calls += 1
                last_retrieval_step = i

                start_time = time.time()

                # Use QueryGen to produce a dense query embedding
                # QueryGen expects [B, L, hidden_dim]; unsqueeze to add batch
                # and sequence-length dimensions.
                with torch.no_grad():
                    query_emb = self.query_gen(
                        reasoning_step_emb=step_emb.unsqueeze(0).unsqueeze(0),
                        query_prefix_emb=question_emb.unsqueeze(0).unsqueeze(0),
                    )
                query_emb_np = query_emb.detach().cpu().numpy()

                # Check speculative cache first
                cached = self.speculative_cache.check_cache(query_emb_np)
                if cached is not None:
                    evidence_texts = cached
                else:
                    # Use QueryGen embedding for dense retrieval via ColBERT.
                    # The embedding captures cross-attended information need;
                    # the fallback text is only used when the index does not
                    # support direct embedding lookup.
                    fallback_text = f"{question} {step.text[:200]}"
                    unique_queries.add(fallback_text)
                    docs = self.retriever.retrieve_by_embedding(
                        query_embedding=query_emb_np.squeeze(),
                        fallback_text=fallback_text,
                        k=5,
                    )
                    evidence_texts = [d['text'] for d in docs]

                latency_total += time.time() - start_time

                # Use RetrievalInjector for evidence integration
                prefix = context + "\n".join(s.text for s in steps[:i + 1])
                injection_result = self.injector.inject(
                    prefix=prefix,
                    evidence=evidence_texts,
                    model_type=self.cfg.model.reasoning_model_type,
                    query_embedding=query_emb_np.squeeze(),
                )
                context = injection_result.augmented_input

                # Continue reasoning with injected context
                continuation = self.reasoning_model.generate(
                    prompt=context,
                    max_tokens=5000,
                )
                reasoning_tokens += len(continuation.split())
        
        # Extract final answer
        answer = self._extract_answer(reasoning_chain)
        
        return {
            'answer': answer,
            'retrieval_calls': retrieval_calls,
            'latency': latency_total,
            'reasoning_tokens': reasoning_tokens,
            'reasoning_chain': reasoning_chain,
        }
    
    def _extract_answer(self, reasoning_chain: str) -> str:
        """Extract final answer from reasoning chain."""
        # Look for "The answer is" pattern
        import re
        
        patterns = [
            r"[Tt]he answer is[:\s]+(.+?)[\.\n]",
            r"[Ff]inal [Aa]nswer[:\s]+(.+?)[\.\n]",
            r"[Aa]nswer[:\s]+(.+?)[\.\n]",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, reasoning_chain)
            if matches:
                return matches[-1].strip()
        
        # Fallback: last sentence
        sentences = reasoning_chain.split('.')
        return sentences[-2].strip() if len(sentences) > 1 else ""


@hydra.main(version_base=None, config_path="configs/experiments", config_name="evaluate")
def main(cfg: DictConfig):
    """Main evaluation function."""
    
    print(f"Evaluating on {cfg.dataset} dataset")
    
    # Load system
    system = ReaLMRetrieveSystem(cfg)
    
    # Load test data
    with open(cfg.data.test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # Evaluate
    predictions = []
    ground_truths = []
    retrieval_calls_list = []
    latencies_list = []
    reasoning_tokens_list = []
    
    for example in tqdm(test_data[:cfg.eval.max_examples], desc="Evaluating"):
        result = system.answer_question(example['question'])
        
        predictions.append(result['answer'])
        ground_truths.append(example['answer'])
        retrieval_calls_list.append(result['retrieval_calls'])
        latencies_list.append(result['latency'])
        reasoning_tokens_list.append(result['reasoning_tokens'])
    
    # Compute metrics
    qa_metrics = compute_qa_metrics(predictions, ground_truths)
    efficiency_metrics = compute_efficiency_metrics(
        retrieval_calls_list,
        latencies_list,
        reasoning_tokens_list,
    )
    
    # Print results
    print("\n" + "="*70)
    print(f"Results on {cfg.dataset}")
    print("="*70)
    print(f"EM: {qa_metrics['em']:.1f}%")
    print(f"F1: {qa_metrics['f1']:.1f}%")
    print(f"Avg Retrieval Calls: {efficiency_metrics['avg_retrieval_calls']:.1f}")
    print(f"Avg Latency: {efficiency_metrics['avg_latency']:.1f}s")
    print(f"Per-Call Overhead: {efficiency_metrics['per_call_overhead']:.2f}s")
    print("="*70)
    
    # Save results
    results_dir = Path(cfg.output_dir) / cfg.dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "results.json", 'w') as f:
        json.dump({
            'qa_metrics': qa_metrics,
            'efficiency_metrics': efficiency_metrics,
            'predictions': predictions,
        }, f, indent=2)
    
    print(f"Results saved to {results_dir}/results.json")


if __name__ == "__main__":
    main()
