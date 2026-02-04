"""
Inference Module for LVMH Multi-Task Model

Provides:
- Fast inference with batching
- Integration with existing pipeline
- Fallback to deterministic pipeline
- Caching and optimization
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time

from .model import LVMHMultiTaskModel, create_model
from .data_generator import Tokenizer, LabelEncoder


class ModelInference:
    """
    Inference wrapper for the multi-task model.
    
    Handles:
    - Model loading and warmup
    - Batch inference
    - Result decoding
    - Performance monitoring
    """
    
    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        enable_cache: bool = True,
        fallback_enabled: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing model weights and encoders
            device: "cpu", "gpu", or "auto"
            enable_cache: Whether to cache predictions
            fallback_enabled: Whether to fall back to deterministic pipeline
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.enable_cache = enable_cache
        self.fallback_enabled = fallback_enabled
        
        # Set device
        self._configure_device()
        
        # Load model and encoders
        self.model: Optional[LVMHMultiTaskModel] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.concept_encoder: Optional[LabelEncoder] = None
        self.segment_encoder: Optional[LabelEncoder] = None
        self.action_encoder: Optional[LabelEncoder] = None
        self.entity_encoder: Optional[LabelEncoder] = None
        
        # Cache
        self._cache: Dict[str, Any] = {}
        
        # Stats
        self.stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "total_time_ms": 0.0,
            "fallback_count": 0
        }
    
    def _configure_device(self):
        """Configure TensorFlow device."""
        if self.device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        elif self.device == "gpu":
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
        # "auto" uses default TF behavior
    
    def load(self):
        """Load model and encoders from disk."""
        # Load model config
        config_path = self.model_dir / "model_best_config.json"
        if not config_path.exists():
            config_path = self.model_dir / "model_final_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"No model config found in {self.model_dir}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        self.model = LVMHMultiTaskModel(**config)
        
        # Load weights
        weights_path = self.model_dir / "model_best.weights.h5"
        if not weights_path.exists():
            weights_path = self.model_dir / "model_final.weights.h5"
        
        if weights_path.exists():
            # Build model first
            dummy_input = tf.zeros((1, config["maxlen"]), dtype=tf.int32)
            self.model(dummy_input)
            self.model.load_weights(str(weights_path))
        
        # Load encoders
        self.tokenizer = Tokenizer.load(str(self.model_dir / "tokenizer.json"))
        self.concept_encoder = LabelEncoder.load(str(self.model_dir / "concept_encoder.json"))
        self.segment_encoder = LabelEncoder.load(str(self.model_dir / "segment_encoder.json"))
        self.action_encoder = LabelEncoder.load(str(self.model_dir / "action_encoder.json"))
        self.entity_encoder = LabelEncoder.load(str(self.model_dir / "entity_encoder.json"))
        
        # Warmup
        self._warmup()
        
        print(f"Model loaded from {self.model_dir}")
    
    def _warmup(self, num_samples: int = 5):
        """Warmup model with dummy predictions."""
        if self.model is None:
            return
        
        dummy_texts = ["test"] * num_samples
        self.predict_batch(dummy_texts)
        self._cache.clear()  # Clear warmup cache
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return str(hash(text))
    
    def predict(
        self,
        text: str,
        tasks: Optional[List[str]] = None,
        concept_threshold: float = 0.5,
        top_k_actions: int = 5
    ) -> Dict[str, Any]:
        """
        Predict for a single text.
        
        Args:
            text: Input text
            tasks: Which tasks to run (None = all)
            concept_threshold: Threshold for concept detection
            top_k_actions: Number of top actions to return
        
        Returns:
            Dictionary with predictions for each task
        """
        return self.predict_batch(
            [text],
            tasks=tasks,
            concept_threshold=concept_threshold,
            top_k_actions=top_k_actions
        )[0]
    
    def predict_batch(
        self,
        texts: List[str],
        tasks: Optional[List[str]] = None,
        concept_threshold: float = 0.5,
        top_k_actions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict for a batch of texts.
        
        Args:
            texts: List of input texts
            tasks: Which tasks to run (None = all)
            concept_threshold: Threshold for concept detection
            top_k_actions: Number of top actions to return
        
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            if self.fallback_enabled:
                return self._fallback_predict(texts)
            raise RuntimeError("Model not loaded. Call load() first.")
        
        start_time = time.time()
        
        # Check cache
        results = [None] * len(texts)
        texts_to_predict = []
        indices_to_predict = []
        
        if self.enable_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    results[i] = self._cache[cache_key]
                    self.stats["cache_hits"] += 1
                else:
                    texts_to_predict.append(text)
                    indices_to_predict.append(i)
        else:
            texts_to_predict = texts
            indices_to_predict = list(range(len(texts)))
        
        # Predict uncached texts
        if texts_to_predict:
            predictions = self._run_inference(
                texts_to_predict,
                tasks=tasks,
                concept_threshold=concept_threshold,
                top_k_actions=top_k_actions
            )
            
            for idx, pred in zip(indices_to_predict, predictions):
                results[idx] = pred
                if self.enable_cache:
                    cache_key = self._get_cache_key(texts[idx])
                    self._cache[cache_key] = pred
        
        # Update stats
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["total_predictions"] += len(texts)
        self.stats["total_time_ms"] += elapsed_ms
        
        return results
    
    def _run_inference(
        self,
        texts: List[str],
        tasks: Optional[List[str]] = None,
        concept_threshold: float = 0.5,
        top_k_actions: int = 5
    ) -> List[Dict[str, Any]]:
        """Run model inference."""
        if tasks is None:
            tasks = ["concepts", "segment", "ner", "recommendation"]
        
        # Tokenize
        input_ids = self.tokenizer.encode_batch(texts)
        input_tensor = tf.constant(input_ids, dtype=tf.int32)
        
        # Run model
        outputs = self.model(input_tensor, training=False, tasks=tasks)
        
        # Decode results
        results = []
        for i in range(len(texts)):
            result = {"text": texts[i]}
            
            if "concepts" in outputs:
                probs = outputs["concepts"][i].numpy()
                concepts = self.concept_encoder.decode_multi(probs, threshold=concept_threshold)
                result["concepts"] = concepts
                result["concept_scores"] = {
                    self.concept_encoder.idx2label[j]: float(probs[j])
                    for j in np.argsort(probs)[::-1][:10]
                }
            
            if "segment" in outputs:
                probs = outputs["segment"][i].numpy()
                segment_idx = int(np.argmax(probs))
                result["segment"] = self.segment_encoder.decode(segment_idx)
                result["segment_confidence"] = float(probs[segment_idx])
            
            if "ner" in outputs:
                labels = np.argmax(outputs["ner"][i].numpy(), axis=-1)
                entities = self._decode_ner_labels(texts[i], labels)
                result["entities"] = entities
            
            if "recommendation" in outputs:
                probs = outputs["recommendation"][i].numpy()
                top_indices = np.argsort(probs)[::-1][:top_k_actions]
                result["recommendations"] = [
                    {
                        "action": self.action_encoder.decode(idx),
                        "score": float(probs[idx])
                    }
                    for idx in top_indices
                ]
            
            results.append(result)
        
        return results
    
    def _decode_ner_labels(
        self,
        text: str,
        labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Decode NER labels to entity spans."""
        tokens = self.tokenizer._tokenize(text)
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels[1:])):  # Skip [CLS]
            if label == 0:  # O tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            else:
                # Decode BIO label
                is_begin = (label - 1) % 2 == 0
                type_idx = (label - 1) // 2
                entity_type = self.entity_encoder.decode(type_idx)
                
                if is_begin or current_entity is None or current_entity["type"] != entity_type:
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "type": entity_type,
                        "tokens": [token],
                        "start_token": i
                    }
                else:
                    current_entity["tokens"].append(token)
        
        if current_entity:
            entities.append(current_entity)
        
        # Convert to text spans
        for entity in entities:
            entity["text"] = " ".join(entity["tokens"])
            del entity["tokens"]
        
        return entities
    
    def _fallback_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback to deterministic pipeline."""
        self.stats["fallback_count"] += len(texts)
        
        # Import deterministic pipeline
        try:
            from server.extract.detect_concepts import detect_concepts
            from server.profiling.segment_clients import predict_segment
            from server.actions.recommend_actions import recommend_actions
            
            results = []
            for text in texts:
                result = {"text": text, "source": "fallback"}
                
                # Detect concepts using lexicon
                concepts = detect_concepts(text)
                result["concepts"] = [c["concept"] for c in concepts]
                
                # TODO: Add segment prediction fallback
                result["segment"] = "unknown"
                result["segment_confidence"] = 0.0
                
                # TODO: Add recommendation fallback
                result["recommendations"] = []
                
                results.append(result)
            
            return results
        
        except ImportError:
            return [{"text": t, "error": "Fallback not available"} for t in texts]
    
    def get_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """Get text embeddings for downstream tasks."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        input_ids = self.tokenizer.encode_batch(texts)
        input_tensor = tf.constant(input_ids, dtype=tf.int32)
        
        embeddings = self.model.get_embeddings(input_tensor)
        return embeddings.numpy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_time = (
            self.stats["total_time_ms"] / max(self.stats["total_predictions"], 1)
        )
        cache_rate = (
            self.stats["cache_hits"] / max(self.stats["total_predictions"], 1)
        )
        
        return {
            **self.stats,
            "avg_time_ms": avg_time,
            "cache_hit_rate": cache_rate
        }
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._cache.clear()


class HybridPipeline:
    """
    Hybrid pipeline that combines ML model with deterministic rules.
    
    Uses ML for:
    - Fast initial predictions
    - Embedding-based similarity
    
    Uses deterministic rules for:
    - High-confidence concept matching
    - Explainability
    - Fallback on edge cases
    """
    
    def __init__(
        self,
        model_dir: str,
        lexicon_path: str,
        ml_weight: float = 0.6,
        concept_threshold: float = 0.5
    ):
        """
        Initialize hybrid pipeline.
        
        Args:
            model_dir: Path to trained ML model
            lexicon_path: Path to concept lexicon
            ml_weight: Weight for ML predictions (0-1)
            concept_threshold: Threshold for concept detection
        """
        self.ml_weight = ml_weight
        self.concept_threshold = concept_threshold
        
        # Initialize ML inference
        self.ml_inference = ModelInference(model_dir)
        
        # Load lexicon for deterministic matching
        self.lexicon = self._load_lexicon(lexicon_path)
    
    def _load_lexicon(self, path: str) -> Dict:
        """Load concept lexicon."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load(self):
        """Load ML model."""
        try:
            self.ml_inference.load()
            self.ml_available = True
        except FileNotFoundError:
            print("Warning: ML model not found, using deterministic pipeline only")
            self.ml_available = False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run hybrid prediction.
        
        Combines ML predictions with deterministic lexicon matching.
        """
        result = {"text": text}
        
        # Deterministic concept detection
        det_concepts = self._deterministic_concepts(text)
        
        if self.ml_available:
            # ML prediction
            ml_result = self.ml_inference.predict(
                text,
                concept_threshold=self.concept_threshold
            )
            
            # Combine concepts (union with weighted scoring)
            combined_concepts = self._combine_concepts(
                det_concepts,
                ml_result.get("concepts", []),
                ml_result.get("concept_scores", {})
            )
            
            result["concepts"] = combined_concepts
            result["segment"] = ml_result.get("segment", "unknown")
            result["segment_confidence"] = ml_result.get("segment_confidence", 0.0)
            result["recommendations"] = ml_result.get("recommendations", [])
            result["entities"] = ml_result.get("entities", [])
            result["source"] = "hybrid"
        else:
            result["concepts"] = det_concepts
            result["source"] = "deterministic"
        
        return result
    
    def _deterministic_concepts(self, text: str) -> List[str]:
        """Detect concepts using lexicon matching."""
        text_lower = text.lower()
        found_concepts = []
        
        for concept, data in self.lexicon.items():
            aliases = data.get("aliases", [concept])
            for alias in aliases:
                if alias.lower() in text_lower:
                    found_concepts.append(concept)
                    break
        
        return found_concepts
    
    def _combine_concepts(
        self,
        det_concepts: List[str],
        ml_concepts: List[str],
        ml_scores: Dict[str, float]
    ) -> List[str]:
        """Combine deterministic and ML concepts."""
        # Union of concepts
        all_concepts = set(det_concepts) | set(ml_concepts)
        
        # Score each concept
        scored = []
        for concept in all_concepts:
            det_score = 1.0 if concept in det_concepts else 0.0
            ml_score = ml_scores.get(concept, 0.0)
            
            combined_score = (
                self.ml_weight * ml_score +
                (1 - self.ml_weight) * det_score
            )
            
            if combined_score >= self.concept_threshold:
                scored.append((concept, combined_score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored]


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models" / "lvmh_multitask"
    
    # Test inference
    inference = ModelInference(str(model_dir), fallback_enabled=True)
    
    try:
        inference.load()
    except FileNotFoundError:
        print("Model not trained yet. Using fallback.")
    
    # Test prediction
    texts = [
        "Le client recherche un sac en cuir pour un anniversaire",
        "VIP client interested in high jewelry collection"
    ]
    
    results = inference.predict_batch(texts)
    
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"  Concepts: {result.get('concepts', [])}")
        print(f"  Segment: {result.get('segment', 'N/A')}")
    
    print(f"\nStats: {inference.get_stats()}")
