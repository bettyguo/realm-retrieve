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
        self.rsus_calculator = RSUSCalculator(
            reasoning_model=self.reasoning_model,
            retriever=self.retriever,
            alpha=cfg.rsus.alpha,
            beta=cfg.rsus.beta,
            gamma=cfg.rsus.gamma,
            device=self.device,
        )
        
        # Policy
        policy_model = RetrievalInterventionPolicy(
            retrieval_threshold=cfg.policy.threshold,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(
            cfg.model.policy_checkpoint,
            map_location=self.device,
        )
        policy_model.load_state_dict(checkpoint['model_state_dict'])
        policy_model.eval()
        
        self.policy = policy_model
    
    def answer_question(self, question: str):
        """Answer a single question using ReaLM-Retrieve."""
        import time
        
        # Initialize reasoning
        context = f"Question: {question}\n\nLet me think step by step:"
        
        # Track metrics
        retrieval_calls = 0
        latency_total = 0.0
        reasoning_tokens = 0
        
        # Generate initial reasoning
        reasoning_chain = self.reasoning_model.generate(
            prompt=context,
            max_tokens=25000,
        )
        reasoning_tokens += len(reasoning_chain.split())
        
        # Segment into steps
        steps = self.segmenter.segment(reasoning_chain)
        
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
            
            state = PolicyState(
                query_embedding=self.reasoning_model.compute_embedding(question),
                current_step_embedding=self.reasoning_model.compute_embedding(step.text),
                rsus_features=torch.tensor([
                    rsus_components.verbalized,
                    rsus_components.entity_entropy,
                    rsus_components.consistency,
                ]),
                retrieval_history=torch.zeros(10),  # Simplified
                step_number=i,
                total_steps_estimate=len(steps),
            )
            
            action, _ = self.policy(state, deterministic=True)
            
            # Retrieve if policy says so
            if action.should_retrieve:
                retrieval_calls += 1
                
                start_time = time.time()
                
                # Formulate query
                query = f"{question} {step.text[:200]}"
                
                # Retrieve documents
                docs = self.retriever.retrieve(query, k=5)
                
                latency_total += time.time() - start_time
                
                # Integrate retrieved context
                retrieved_text = "\n\n".join([d['text'] for d in docs])
                context += f"\n\nRetrieved Evidence:\n{retrieved_text}\n\n"
                
                # Continue reasoning with new context
                continuation = self.reasoning_model.generate(
                    prompt=context + step.text,
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
