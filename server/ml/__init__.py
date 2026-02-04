"""
LVMH Client Intelligence - TensorFlow Multi-Task Model

A supervised learning approach for:
- Multi-label concept classification
- Client segment prediction
- Named Entity Recognition (NER)
- Action recommendation

Works alongside the deterministic pipeline, with optional fallback.

Usage:
    # Training
    from server.ml import train_model, DataGenerator
    
    data_gen = DataGenerator(data_dir="data/")
    data_gen.load_data()
    train_model(data_gen, output_dir="models/lvmh_multitask")
    
    # Inference
    from server.ml import ModelInference
    
    inference = ModelInference("models/lvmh_multitask")
    inference.load()
    result = inference.predict("Client VIP recherche montre")
    
    # CLI
    python -m server.ml.cli train --size base --epochs 30
    python -m server.ml.cli predict "Client cherche cadeau anniversaire"
"""

from .model import (
    LVMHMultiTaskModel,
    create_model,
    SharedEncoder,
    ConceptHead,
    SegmentHead,
    NERHead,
    RecommendationHead
)
from .data_generator import DataGenerator, Tokenizer, LabelEncoder
from .trainer import ModelTrainer, MultiTaskLoss, train_model
from .inference import ModelInference, HybridPipeline

__all__ = [
    # Model
    "LVMHMultiTaskModel",
    "create_model",
    "SharedEncoder",
    "ConceptHead",
    "SegmentHead",
    "NERHead",
    "RecommendationHead",
    # Data
    "DataGenerator",
    "Tokenizer",
    "LabelEncoder",
    # Training
    "ModelTrainer",
    "MultiTaskLoss",
    "train_model",
    # Inference
    "ModelInference",
    "HybridPipeline",
]
