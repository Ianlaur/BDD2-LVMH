"""
CLI for LVMH Multi-Task Model

Commands:
- train: Train the model
- evaluate: Evaluate on test data
- predict: Run predictions
- export: Export model for deployment
"""

import argparse
import sys
from pathlib import Path
import json


def train_command(args):
    """Train the multi-task model."""
    from .trainer import train_model
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "models" / "lvmh_multitask"
    
    if args.output:
        output_dir = Path(args.output)
    
    print(f"Training LVMH Multi-Task Model")
    print(f"  Data dir: {data_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Model size: {args.size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Train model (it handles data loading internally)
    train_model(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        maxlen=args.maxlen
    )
    
    print(f"\nModel saved to {output_dir}")


def evaluate_command(args):
    """Evaluate model on test data."""
    from .inference import ModelInference
    from .data_generator import DataGenerator
    import numpy as np
    
    model_dir = Path(args.model_dir)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    print(f"Evaluating model from {model_dir}")
    
    # Load inference engine
    inference = ModelInference(str(model_dir))
    inference.load()
    
    # Load test data
    data_gen = DataGenerator(str(data_dir))
    data_gen.load_data()
    _, _, test_ds = data_gen.create_datasets(
        batch_size=32,
        test_split=0.1,
        val_split=0.1
    )
    
    # Evaluate
    metrics = {
        "concept_precision": [],
        "concept_recall": [],
        "segment_accuracy": [],
        "action_accuracy": []
    }
    
    for batch in test_ds.take(args.num_batches):
        inputs = batch[0]
        labels = batch[1]
        
        # Get texts (need to decode)
        texts = [
            inference.tokenizer.decode(ids.numpy())
            for ids in inputs
        ]
        
        predictions = inference.predict_batch(texts)
        
        # Compare with labels
        # (Simplified evaluation - real implementation would be more thorough)
        for i, pred in enumerate(predictions):
            true_concepts = set(
                inference.concept_encoder.decode(j)
                for j, v in enumerate(labels["concepts"][i].numpy())
                if v > 0.5
            )
            pred_concepts = set(pred.get("concepts", []))
            
            if pred_concepts:
                precision = len(true_concepts & pred_concepts) / len(pred_concepts)
                metrics["concept_precision"].append(precision)
            
            if true_concepts:
                recall = len(true_concepts & pred_concepts) / len(true_concepts)
                metrics["concept_recall"].append(recall)
            
            true_segment = inference.segment_encoder.decode(
                np.argmax(labels["segment"][i].numpy())
            )
            pred_segment = pred.get("segment", "")
            metrics["segment_accuracy"].append(
                1.0 if true_segment == pred_segment else 0.0
            )
    
    # Print results
    print("\nEvaluation Results:")
    for name, values in metrics.items():
        if values:
            mean = np.mean(values)
            print(f"  {name}: {mean:.4f}")


def predict_command(args):
    """Run predictions on input text."""
    from .inference import ModelInference, HybridPipeline
    
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models" / "lvmh_multitask"
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
    
    if args.hybrid:
        lexicon_path = project_root / "taxonomy" / "lexicon_v1.json"
        pipeline = HybridPipeline(
            model_dir=str(model_dir),
            lexicon_path=str(lexicon_path),
            ml_weight=args.ml_weight
        )
        pipeline.load()
        
        result = pipeline.predict(args.text)
    else:
        inference = ModelInference(str(model_dir), fallback_enabled=args.fallback)
        
        try:
            inference.load()
        except FileNotFoundError:
            if args.fallback:
                print("Model not found, using fallback pipeline")
            else:
                print("Error: Model not found. Train a model first or use --fallback")
                return
        
        result = inference.predict(
            args.text,
            concept_threshold=args.threshold,
            top_k_actions=args.top_k
        )
    
    # Output
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\nText: {result['text']}")
        print(f"Source: {result.get('source', 'ml')}")
        
        if "concepts" in result:
            print(f"\nConcepts: {', '.join(result['concepts']) or 'None'}")
        
        if "segment" in result:
            print(f"\nSegment: {result['segment']} ({result.get('segment_confidence', 0):.2%})")
        
        if "entities" in result:
            print(f"\nEntities:")
            for ent in result.get("entities", []):
                print(f"  - {ent['type']}: {ent['text']}")
        
        if "recommendations" in result:
            print(f"\nRecommendations:")
            for rec in result.get("recommendations", [])[:5]:
                print(f"  - {rec['action']} ({rec['score']:.2%})")


def export_command(args):
    """Export model for deployment."""
    import tensorflow as tf
    from .inference import ModelInference
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    
    print(f"Exporting model from {model_dir}")
    print(f"  Format: {args.format}")
    print(f"  Output: {output_dir}")
    
    # Load model
    inference = ModelInference(str(model_dir))
    inference.load()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.format == "saved_model":
        # Export as TensorFlow SavedModel
        tf.saved_model.save(inference.model, str(output_dir / "saved_model"))
        
        # Copy encoders
        import shutil
        for encoder_file in ["tokenizer.json", "concept_encoder.json",
                           "segment_encoder.json", "action_encoder.json",
                           "entity_encoder.json"]:
            src = model_dir / encoder_file
            if src.exists():
                shutil.copy(src, output_dir / encoder_file)
        
        print(f"Saved TensorFlow SavedModel to {output_dir}/saved_model")
    
    elif args.format == "tflite":
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(inference.model)
        
        if args.quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_dir / "model.tflite", "wb") as f:
            f.write(tflite_model)
        
        print(f"Saved TFLite model to {output_dir}/model.tflite")
    
    elif args.format == "onnx":
        try:
            import tf2onnx
            
            model_proto, _ = tf2onnx.convert.from_keras(inference.model)
            with open(output_dir / "model.onnx", "wb") as f:
                f.write(model_proto.SerializeToString())
            
            print(f"Saved ONNX model to {output_dir}/model.onnx")
        except ImportError:
            print("Error: tf2onnx not installed. Run: pip install tf2onnx")
    
    print("Export complete!")


def main():
    parser = argparse.ArgumentParser(
        description="LVMH Multi-Task Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--output", "-o", help="Output directory")
    train_parser.add_argument("--size", choices=["small", "base", "large"],
                             default="base", help="Model size")
    train_parser.add_argument("--epochs", type=int, default=30,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4,
                             help="Learning rate")
    train_parser.add_argument("--maxlen", type=int, default=512,
                             help="Maximum sequence length")
    train_parser.add_argument("--patience", type=int, default=5,
                             help="Early stopping patience")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model-dir", required=True,
                            help="Path to trained model")
    eval_parser.add_argument("--num-batches", type=int, default=10,
                            help="Number of batches to evaluate")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Run prediction")
    pred_parser.add_argument("text", help="Text to analyze")
    pred_parser.add_argument("--model-dir", help="Path to trained model")
    pred_parser.add_argument("--threshold", type=float, default=0.5,
                            help="Concept detection threshold")
    pred_parser.add_argument("--top-k", type=int, default=5,
                            help="Number of top recommendations")
    pred_parser.add_argument("--hybrid", action="store_true",
                            help="Use hybrid pipeline")
    pred_parser.add_argument("--ml-weight", type=float, default=0.6,
                            help="ML weight in hybrid mode")
    pred_parser.add_argument("--fallback", action="store_true",
                            help="Enable fallback to deterministic pipeline")
    pred_parser.add_argument("--json", action="store_true",
                            help="Output as JSON")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model")
    export_parser.add_argument("--model-dir", required=True,
                              help="Path to trained model")
    export_parser.add_argument("--output", "-o", required=True,
                              help="Output directory")
    export_parser.add_argument("--format", choices=["saved_model", "tflite", "onnx"],
                              default="saved_model", help="Export format")
    export_parser.add_argument("--quantize", action="store_true",
                              help="Quantize model (TFLite only)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "export":
        export_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
