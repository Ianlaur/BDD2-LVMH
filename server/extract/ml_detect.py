"""
ML-Enhanced Concept Detection - Uses trained models for better accuracy.

This module provides an enhanced concept detector that combines:
1. Traditional rule-based matching (fast, high precision)
2. ML model predictions (better recall, handles variations)

Usage:
    # Check available models
    python -m server.extract.ml_detect list-models
    
    # Run concept detection with ML enhancement
    python -m server.extract.ml_detect --model concept_model_base_20260205_161623
    
    # Fall back to rule-based if no model
    python -m server.extract.ml_detect
"""
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from server.shared.config import (
    DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR, MODELS_DIR,
    MAX_ALIAS_MATCHES_PER_NOTE
)
from server.extract.detect_concepts import (
    load_lexicon, build_alias_to_concept_map, find_matches_in_text
)


class MLConceptDetector:
    """
    ML-enhanced concept detector that combines rule-based and model-based detection.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize detector with optional ML model.
        
        Args:
            model_path: Path to trained model directory. If None, uses rule-based only.
        """
        self.model_path = model_path
        self.model_metadata = None
        self.concept_mapping = None
        self.use_ml = False
        
        if model_path and model_path.exists():
            self._load_model(model_path)
    
    def _load_model(self, model_path: Path):
        """Load model metadata and mappings."""
        metadata_path = model_path / "metadata.json"
        mapping_path = model_path / "concept_mapping.json"
        
        if not metadata_path.exists() or not mapping_path.exists():
            logger.warning(f"Model files not found in {model_path}, falling back to rule-based")
            return
        
        try:
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            
            with open(mapping_path, 'r') as f:
                self.concept_mapping = json.load(f)
            
            self.use_ml = True
            
            logger.info(f"✅ Loaded ML model: {model_path.name}")
            logger.info(f"   - Accuracy: {self.model_metadata.get('final_accuracy', 0):.2%}")
            logger.info(f"   - Concepts: {self.model_metadata.get('num_concepts', 0)}")
            logger.info(f"   - Training date: {self.model_metadata.get('training_date', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.use_ml = False
    
    def detect_concepts(
        self,
        notes_df: pd.DataFrame,
        alias_map: Dict[str, str],
        use_ml_enhancement: bool = True
    ) -> pd.DataFrame:
        """
        Detect concepts in notes using rule-based + optional ML enhancement.
        
        Args:
            notes_df: DataFrame with 'note_id' and 'text' columns
            alias_map: Dictionary mapping aliases to concept IDs
            use_ml_enhancement: If True and model loaded, use ML to find additional matches
            
        Returns:
            DataFrame with detected concepts
        """
        logger.info(f"Detecting concepts in {len(notes_df)} notes...")
        logger.info(f"ML enhancement: {'ENABLED' if self.use_ml and use_ml_enhancement else 'DISABLED'}")
        
        all_matches = []
        
        for idx, row in notes_df.iterrows():
            note_id = row['note_id']
            text = row['text']
            
            # 1. Rule-based matching (fast, high precision)
            rule_matches = find_matches_in_text(text, alias_map, MAX_ALIAS_MATCHES_PER_NOTE)
            
            # Add note_id to each match
            for match in rule_matches:
                match['note_id'] = note_id
                match['detection_method'] = 'rule-based'
                match['confidence'] = 1.0  # Rule matches have 100% confidence
            
            all_matches.extend(rule_matches)
            
            # 2. ML enhancement (optional, better recall)
            if self.use_ml and use_ml_enhancement:
                ml_matches = self._detect_with_ml(note_id, text, rule_matches, alias_map)
                all_matches.extend(ml_matches)
        
        # Convert to DataFrame
        if not all_matches:
            logger.warning("No concepts detected!")
            return pd.DataFrame(columns=['note_id', 'concept_id', 'matched_alias', 
                                         'start', 'end', 'detection_method', 'confidence'])
        
        matches_df = pd.DataFrame(all_matches)
        
        # Stats
        total_matches = len(matches_df)
        rule_based = len(matches_df[matches_df['detection_method'] == 'rule-based'])
        ml_based = len(matches_df[matches_df['detection_method'] == 'ml-enhanced'])
        
        logger.info(f"✅ Concept detection complete:")
        logger.info(f"   - Total matches: {total_matches}")
        logger.info(f"   - Rule-based: {rule_based} ({rule_based/total_matches*100:.1f}%)")
        if ml_based > 0:
            logger.info(f"   - ML-enhanced: {ml_based} ({ml_based/total_matches*100:.1f}%)")
        logger.info(f"   - Unique concepts: {matches_df['concept_id'].nunique()}")
        logger.info(f"   - Notes with concepts: {matches_df['note_id'].nunique()}")
        
        return matches_df
    
    def _detect_with_ml(
        self,
        note_id: str,
        text: str,
        rule_matches: List[Dict],
        alias_map: Dict[str, str]
    ) -> List[Dict]:
        """
        Use ML model to find additional concept matches that rules might miss.
        
        This is a placeholder for actual ML inference. In a real implementation:
        1. Generate embeddings for the text
        2. Use model to predict concept probabilities
        3. Return high-confidence predictions not already found by rules
        
        For now, returns empty list (ML inference not yet implemented).
        """
        # TODO: Implement actual ML inference
        # This would involve:
        # - Loading sentence transformer model
        # - Encoding the text
        # - Running through trained classifier
        # - Filtering by confidence threshold
        # - Avoiding duplicates with rule_matches
        
        return []  # Placeholder


def list_available_models() -> List[Tuple[Path, Dict]]:
    """List all available trained models."""
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and (model_dir / "metadata.json").exists():
            with open(model_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            models.append((model_dir, metadata))
    
    # Sort by training date (newest first)
    models.sort(key=lambda x: x[1].get('training_date', ''), reverse=True)
    
    return models


def get_best_model() -> Optional[Path]:
    """Get the best (most accurate) trained model."""
    models = list_available_models()
    
    if not models:
        return None
    
    # Sort by accuracy
    models_sorted = sorted(models, key=lambda x: x[1].get('final_accuracy', 0), reverse=True)
    
    return models_sorted[0][0]


def detect_concepts_with_ml(model_name: Optional[str] = None, use_ml: bool = True):
    """
    Run concept detection with optional ML enhancement.
    
    Args:
        model_name: Name of model to use, or None for best model
        use_ml: If False, use rule-based only
    """
    logger.info("=" * 80)
    logger.info("ML-ENHANCED CONCEPT DETECTION")
    logger.info("=" * 80)
    
    # Load notes
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes not found: {notes_path}. Run ingestion first.")
    
    notes_df = pd.read_parquet(notes_path)
    logger.info(f"Loaded {len(notes_df)} notes")
    
    # Load lexicon
    lexicon_df = load_lexicon()
    alias_map = build_alias_to_concept_map(lexicon_df)
    logger.info(f"Loaded {len(lexicon_df)} concepts with {len(alias_map)} aliases")
    
    # Load model if requested
    model_path = None
    if use_ml:
        if model_name:
            model_path = MODELS_DIR / model_name
            if not model_path.exists():
                logger.warning(f"Model {model_name} not found, looking for best model...")
                model_path = get_best_model()
        else:
            model_path = get_best_model()
        
        if model_path:
            logger.info(f"Using model: {model_path.name}")
        else:
            logger.warning("No trained models found, using rule-based only")
    else:
        logger.info("ML enhancement disabled, using rule-based only")
    
    # Initialize detector
    detector = MLConceptDetector(model_path)
    
    # Detect concepts
    matches_df = detector.detect_concepts(notes_df, alias_map, use_ml_enhancement=use_ml)
    
    # Save results
    output_path = DATA_OUTPUTS / "note_concepts.csv"
    matches_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved concept matches to: {output_path}")
    
    # Also save with ML suffix if using ML
    if use_ml and model_path:
        ml_output_path = DATA_OUTPUTS / f"note_concepts_ml_{model_path.name}.csv"
        matches_df.to_csv(ml_output_path, index=False)
        logger.info(f"✅ Also saved to: {ml_output_path}")
    
    logger.info("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="ML-Enhanced Concept Detection")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available trained models")
    
    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Run concept detection")
    detect_parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (default: best model by accuracy)"
    )
    detect_parser.add_argument(
        "--no-ml",
        action="store_true",
        help="Disable ML enhancement, use rule-based only"
    )
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        models = list_available_models()
        
        if not models:
            print("No trained models found.")
            print(f"Train a model with: python -m server.ml.cli train --size base --epochs 20")
            return
        
        print("\n" + "=" * 80)
        print("AVAILABLE TRAINED MODELS")
        print("=" * 80)
        print()
        
        for i, (model_dir, metadata) in enumerate(models, 1):
            accuracy = metadata.get('final_accuracy', 0)
            date = metadata.get('training_date', 'unknown')
            concepts = metadata.get('num_concepts', 0)
            
            print(f"{i}. {model_dir.name}")
            print(f"   Accuracy: {accuracy:.2%}")
            print(f"   Concepts: {concepts}")
            print(f"   Trained: {date}")
            print()
        
        best = get_best_model()
        if best:
            print(f"✅ Best model (by accuracy): {best.name}")
        
        print("=" * 80)
    
    elif args.command == "detect":
        detect_concepts_with_ml(
            model_name=args.model,
            use_ml=not args.no_ml
        )
    
    else:
        # Default: run detection with best model
        detect_concepts_with_ml()


if __name__ == "__main__":
    main()
