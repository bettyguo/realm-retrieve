"""
Reasoning Step Uncertainty Score (RSUS) Computation

Implements step-level uncertainty detection through three signals:
1. Verbalized uncertainty (U_verb): Model self-assessment
2. Entity entropy (U_ent): Entity coverage ambiguity
3. Consistency signal (U_cons): Agreement across sampled continuations

RSUS(r_i) = α·U_verb(r_i) + β·U_ent(r_i) + γ·U_cons(r_i)
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# spaCy is loaded lazily inside ``RSUSCalculator.__init__`` (Issue #3) and the
# spaCy English model is downloaded on demand if it is missing (Issue #5).


@dataclass
class RSUSComponents:
    """Individual components of RSUS score."""
    verbalized: float
    entity_entropy: float
    consistency: float
    rsus: float


class RSUSCalculator:
    """
    Calculate Reasoning Step Uncertainty Score.
    
    Usage:
        calculator = RSUSCalculator(
            reasoning_model=model,
            retriever=retriever,
            alpha=0.4, beta=0.35, gamma=0.25
        )
        
        score, components = calculator.compute_rsus(
            reasoning_step="The Berlin Wall fell in 1989...",
            context="Previous reasoning..."
        )
    """
    
    def __init__(
        self,
        reasoning_model,  # ReasoningModelWrapper
        retriever,  # ColBERTRetriever
        alpha: float = 0.4,
        beta: float = 0.35,
        gamma: float = 0.25,
        device: str = "cuda",
    ):
        """
        Args:
            reasoning_model: Wrapper for LRM (DeepSeek-R1, o1, QwQ)
            retriever: ColBERT retriever for entity entropy computation
            alpha: Weight for verbalized uncertainty
            beta: Weight for entity entropy
            gamma: Weight for consistency signal
        """
        # Issue #5: catch a common configuration mistake at construction time
        # rather than producing silently mis-scaled RSUS values.
        for name, val in (("alpha", alpha), ("beta", beta), ("gamma", gamma)):
            if val < 0:
                raise ValueError(f"RSUS weight {name}={val!r} must be >= 0")
        total = alpha + beta + gamma
        if not math.isclose(total, 1.0, abs_tol=1e-3):
            raise ValueError(
                f"RSUS weights must sum to 1; got alpha={alpha}, beta={beta}, "
                f"gamma={gamma} (sum={total:.6f})"
            )

        self.reasoning_model = reasoning_model
        self.retriever = retriever
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Lazy import + auto-fetch the English NER model on first use.
        import spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.device = device
        
        # Verbalized uncertainty prompt template
        self.confidence_prompt = (
            "Given the reasoning so far, rate your confidence that the current "
            "conclusion is factually correct on a scale of 0-100, where 0 means "
            "completely uncertain and 100 means absolutely certain. "
            "Respond with only a number:"
        )
    
    def compute_rsus(
        self,
        reasoning_step: str,
        context: str,
        compute_consistency: bool = True,
    ) -> Tuple[float, RSUSComponents]:
        """
        Compute RSUS for a reasoning step.
        
        Args:
            reasoning_step: Current reasoning step text
            context: Previous reasoning context
            compute_consistency: Whether to compute consistency (expensive, requires sampling)
            
        Returns:
            rsus_score: Combined uncertainty score [0, 1]
            components: Individual component scores
        """
        # 1. Verbalized uncertainty
        u_verb = self._compute_verbalized_uncertainty(reasoning_step, context)
        
        # 2. Entity entropy
        u_ent = self._compute_entity_entropy(reasoning_step)
        
        # 3. Consistency signal (optional, expensive)
        if compute_consistency:
            u_cons = self._compute_consistency_signal(reasoning_step, context)
        else:
            u_cons = 0.0
        
        # Combine using learned weights
        rsus = self.alpha * u_verb + self.beta * u_ent + self.gamma * u_cons
        
        components = RSUSComponents(
            verbalized=u_verb,
            entity_entropy=u_ent,
            consistency=u_cons,
            rsus=rsus,
        )
        
        return rsus, components
    
    def _compute_verbalized_uncertainty(
        self,
        reasoning_step: str,
        context: str,
    ) -> float:
        """
        Compute verbalized uncertainty (U_verb).
        
        Prompts the reasoning model to self-assess confidence.
        For models not supporting mid-generation prompting, uses trained classifier.
        
        Returns:
            u_verb: Normalized uncertainty [0, 1] (0=certain, 1=uncertain)
        """
        # Construct prompt
        full_context = f"{context}\n\n{reasoning_step}\n\n{self.confidence_prompt}"
        
        try:
            # Get confidence rating from model
            response = self.reasoning_model.generate(
                prompt=full_context,
                max_tokens=10,
                temperature=0.3,
                stop_sequences=["\n", ".", ","],
            )
            
            # Extract number
            confidence = self._extract_confidence_number(response)
            
            # Convert to uncertainty (invert and normalize)
            u_verb = 1.0 - (confidence / 100.0)
            
        except Exception as e:
            # Fallback: Use pattern-based classifier
            u_verb = self._fallback_verbalized_uncertainty(reasoning_step)
        
        return np.clip(u_verb, 0.0, 1.0)
    
    def _extract_confidence_number(self, response: str) -> float:
        """Extract confidence number from model response."""
        # Try to find a number
        numbers = re.findall(r'\b\d+\.?\d*\b', response)
        if numbers:
            return float(numbers[0])
        
        # Fallback: Parse qualitative responses
        response_lower = response.lower()
        if any(word in response_lower for word in ["very confident", "certain", "sure"]):
            return 90.0
        elif any(word in response_lower for word in ["confident", "likely"]):
            return 70.0
        elif any(word in response_lower for word in ["uncertain", "unsure", "unclear"]):
            return 30.0
        elif any(word in response_lower for word in ["very uncertain", "don't know"]):
            return 10.0
        
        return 50.0  # Default middle value
    
    def _fallback_verbalized_uncertainty(self, reasoning_step: str) -> float:
        """
        Fallback method using pattern matching for models without mid-generation prompting.
        
        Looks for uncertainty markers in the reasoning text itself.
        """
        text_lower = reasoning_step.lower()
        
        # High uncertainty markers
        high_uncertainty_markers = [
            "i'm not sure", "i don't know", "unclear", "uncertain",
            "might be", "could be", "possibly", "perhaps", "maybe",
            "need to verify", "let me check", "need more information",
        ]
        
        # Low uncertainty markers
        low_uncertainty_markers = [
            "i know", "clearly", "obviously", "definitely", "certainly",
            "confirmed", "verified", "established fact",
        ]
        
        # Count markers
        high_count = sum(1 for marker in high_uncertainty_markers if marker in text_lower)
        low_count = sum(1 for marker in low_uncertainty_markers if marker in text_lower)
        
        if high_count > low_count:
            return 0.7  # High uncertainty
        elif low_count > high_count:
            return 0.2  # Low uncertainty
        else:
            return 0.5  # Neutral
    
    def _compute_entity_entropy(self, reasoning_step: str) -> float:
        """
        Compute entity-based entropy (U_ent).
        
        High entropy indicates entities with ambiguous or sparse corpus coverage.
        
        Formula:
            U_ent(r_i) = -∑_{e ∈ Ent(r_i)} p(e|D) log p(e|D)
        
        Returns:
            u_ent: Normalized entropy [0, 1]
        """
        # Extract named entities
        doc = self.nlp(reasoning_step)
        entities = [ent.text for ent in doc.ents if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"
        ]]
        
        if not entities:
            return 0.0  # No entities = low uncertainty
        
        # Compute retrieval entropy for each entity
        entropies = []
        for entity in entities[:5]:  # Limit to top 5 entities for efficiency
            entropy = self._entity_retrieval_entropy(entity)
            entropies.append(entropy)
        
        # Average entropy across entities
        avg_entropy = np.mean(entropies) if entropies else 0.0
        
        # Normalize to [0, 1] (log(100) ≈ 4.6 is typical max)
        u_ent = np.clip(avg_entropy / 4.6, 0.0, 1.0)
        
        return u_ent
    
    def _entity_retrieval_entropy(self, entity: str, top_k: int = 100) -> float:
        """
        Compute retrieval entropy for a single entity.
        
        Query: "What is [entity]?"
        Compute entropy over BM25 scores of top-k documents.
        """
        query = f"What is {entity}?"
        
        try:
            # Retrieve documents
            results = self.retriever.retrieve(query, k=top_k, return_scores=True)
            
            if not results:
                return 5.0  # High entropy for entities with no coverage
            
            # Extract scores
            scores = np.array([r["score"] for r in results])
            
            # Normalize to probability distribution
            probs = np.exp(scores) / np.sum(np.exp(scores))
            
            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            return entropy
            
        except Exception:
            return 2.5  # Medium entropy on retrieval failure
    
    def _compute_consistency_signal(
        self,
        reasoning_step: str,
        context: str,
        num_samples: int = 3,
    ) -> float:
        """
        Compute consistency signal (U_cons).
        
        Sample k alternative continuations and measure agreement.
        Disagreement indicates uncertainty that retrieval may resolve.
        
        Args:
            reasoning_step: Current step
            context: Previous reasoning
            num_samples: Number of alternative continuations (k=3 in paper)
            
        Returns:
            u_cons: Normalized consistency score [0, 1] (0=consistent, 1=inconsistent)
        """
        # Check if this is a "critical" reasoning step (has discourse markers)
        if not self._is_critical_step(reasoning_step):
            return 0.0  # Skip consistency check for non-critical steps
        
        # Sample alternative continuations
        full_context = f"{context}\n\n{reasoning_step}\n\nTherefore,"
        
        continuations = []
        for _ in range(num_samples):
            continuation = self.reasoning_model.generate(
                prompt=full_context,
                max_tokens=100,
                temperature=0.8,  # Higher temperature for diversity
                stop_sequences=["\n\n"],
            )
            continuations.append(continuation.strip())
        
        # Measure agreement using embedding similarity
        consistency_score = self._measure_agreement(continuations)
        
        # Convert to uncertainty (invert: high consistency = low uncertainty)
        u_cons = 1.0 - consistency_score
        
        return np.clip(u_cons, 0.0, 1.0)
    
    def _is_critical_step(self, reasoning_step: str) -> bool:
        """Check if step has discourse markers indicating it's critical."""
        critical_markers = [
            "therefore", "thus", "hence", "so",
            "let me verify", "let's check", "to confirm",
            "this means", "this shows", "this proves",
        ]
        text_lower = reasoning_step.lower()
        return any(marker in text_lower for marker in critical_markers)
    
    def _measure_agreement(self, continuations: List[str]) -> float:
        """
        Measure agreement across continuations using pairwise similarity.
        
        Returns:
            agreement: [0, 1] where 1 = perfect agreement
        """
        if len(continuations) < 2:
            return 1.0
        
        # Use simple token overlap as proxy for semantic similarity
        # (Could use sentence embeddings for better accuracy)
        similarities = []
        for i in range(len(continuations)):
            for j in range(i + 1, len(continuations)):
                sim = self._token_overlap_similarity(continuations[i], continuations[j])
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _token_overlap_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between token sets."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_rsus_batch(
        self,
        reasoning_steps: List[str],
        contexts: List[str],
    ) -> List[Tuple[float, RSUSComponents]]:
        """
        Compute RSUS for multiple steps (batched for efficiency).
        
        Args:
            reasoning_steps: List of reasoning steps
            contexts: List of corresponding contexts
            
        Returns:
            List of (rsus_score, components) tuples
        """
        results = []
        for step, context in zip(reasoning_steps, contexts):
            rsus, components = self.compute_rsus(
                reasoning_step=step,
                context=context,
                compute_consistency=False,  # Skip expensive consistency for batch
            )
            results.append((rsus, components))
        
        return results
