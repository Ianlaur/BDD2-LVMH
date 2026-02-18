# LVMH ML Training Module

This module provides machine learning capabilities to train context-aware models for better concept detection and keyword understanding in the LVMH client intelligence pipeline.

## What It Does

The ML training system:
- **Learns from your lexicon**: Uses your existing concepts and aliases to understand relationships
- **Learns from client notes**: Extracts contextual patterns from actual client interactions
- **Improves concept detection**: Trains models to better recognize concepts in multilingual text
- **Understands relationships**: Learns associations between keywords, concepts, and client behaviors

## Quick Start

### Train a Model

```bash
# Basic training (recommended for most cases)
python3 -m server.ml.cli train --size base --epochs 30

# Advanced training (higher accuracy, slower)
python3 -m server.ml.cli train --size large --epochs 50 --batch-size 32
```

### List Trained Models

```bash
python3 -m server.ml.cli list
```

### Evaluate a Model

```bash
python3 -m server.ml.cli evaluate --model-path models/concept_model_base_20260205_153932
```

## Training Parameters

### `--size` (Model Size)
- **base**: 128-dimensional embeddings, faster training, good for most cases
- **large**: 384-dimensional embeddings, higher accuracy, more training time

### `--epochs` (Training Duration)
- **30**: Good starting point, fast convergence
- **50-100**: Better accuracy, takes longer
- Default: 30

### `--batch-size` (Training Batch Size)
- **16**: Good for smaller datasets or limited memory
- **32-64**: Faster training with more memory
- Default: 16

### `--learning-rate` (Optimization Speed)
- **0.001**: Standard rate, good for base models
- **0.0005**: More stable, good for large models
- **0.002**: Faster convergence, risk of instability
- Default: 0.001

## Training Process

The training system:

1. **Loads Your Data**
   - Client notes from `data/processed/notes_clean.parquet`
   - Keyword candidates from `data/processed/candidates.csv`
   - Concepts from `taxonomy/lexicon_v1.json`

2. **Prepares Training Examples**
   - Creates concept-keyword pairs from your lexicon
   - Extracts context examples from client notes
   - Builds multilingual training dataset

3. **Trains the Model**
   - Initializes model architecture (base or large)
   - Runs training for specified epochs
   - Tracks loss and accuracy metrics

4. **Saves Everything**
   - Model weights and embeddings
   - Training metadata and metrics
   - Concept mappings
   - Sample training examples

## Model Output

Each trained model is saved in `models/concept_model_{size}_{timestamp}/`:

```
concept_model_base_20260205_153932/
├── metadata.json           # Training info and metrics
├── concept_mapping.json    # Concept IDs and labels
└── training_examples.json  # Sample training data
```

### metadata.json
Contains:
- Model type and configuration
- Training parameters (epochs, batch size, learning rate)
- Performance metrics (loss, accuracy)
- Training date and dataset info

### concept_mapping.json
Contains:
- List of all concepts learned
- Concept labels and buckets
- Mapping for inference

### training_examples.json
Contains:
- Sample concept-keyword pairs used for training
- Sample context examples from client notes

## Integration with Pipeline

Once trained, these models can be used to:
- **Improve concept detection** in Stage 4 (detect_concepts.py)
- **Better keyword extraction** in Stage 2 (run_candidates.py)
- **Enhanced embeddings** in Stage 5 (build_vectors.py)
- **Smarter client segmentation** in Stage 6 (segment_clients.py)

## Performance Metrics

### Loss
- Lower is better
- Typical range: 0.01 - 0.5
- Target: < 0.1 for good models

### Accuracy
- Higher is better
- Range: 0.0 - 1.0 (0% - 100%)
- Target: > 0.85 for production use

### Evaluation Metrics
- **Precision**: How many detected concepts are correct
- **Recall**: How many actual concepts are detected
- **F1 Score**: Balance between precision and recall

## Examples

### Train a Quick Model
```bash
python3 -m server.ml.cli train --size base --epochs 20
```

### Train a High-Quality Model
```bash
python3 -m server.ml.cli train --size large --epochs 100 --batch-size 32 --learning-rate 0.0005
```

### Compare Models
```bash
# List all models
python3 -m server.ml.cli list

# Evaluate each
python3 -m server.ml.cli evaluate --model-path models/concept_model_base_YYYYMMDD_HHMMSS
python3 -m server.ml.cli evaluate --model-path models/concept_model_large_YYYYMMDD_HHMMSS
```

## Best Practices

1. **Start with base models**: Train a base model first to validate your data
2. **Monitor metrics**: Watch loss and accuracy during training
3. **Use more epochs for large models**: Large models need more training time
4. **Evaluate regularly**: Test models on your actual data
5. **Keep best models**: Save models with highest accuracy for production

## Troubleshooting

### Training fails with "No data found"
- Make sure you've run the pipeline at least once
- Check that `data/processed/notes_clean.parquet` exists
- Verify `taxonomy/lexicon_v1.json` is not empty

### Accuracy not improving
- Try more epochs (50-100)
- Use larger model size
- Check your lexicon quality
- Ensure enough training data (> 50 notes minimum)

### Out of memory errors
- Reduce batch size (try 8 or 4)
- Use base model instead of large
- Close other applications

## Future Enhancements

The ML module is designed to support:
- Custom model architectures
- Transfer learning from pre-trained models
- Real-time inference integration
- A/B testing of models
- Automated hyperparameter tuning

## Requirements

This module uses:
- pandas: Data handling
- numpy: Numerical operations
- Standard ML libraries (to be expanded)

See `requirements.txt` for full dependencies.
