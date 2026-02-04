"""
Model Trainer for LVMH Multi-Task Model

Handles:
- Multi-task loss computation
- Training loop with task weighting
- Learning rate scheduling
- Checkpointing and early stopping
- Metrics tracking per task
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, callbacks
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json
import time
import numpy as np

from .model import LVMHMultiTaskModel


class MultiTaskLoss:
    """
    Combined loss for multi-task learning.
    Supports automatic task weighting using uncertainty weighting.
    Note: Not a keras.losses.Loss subclass for Keras 3.x compatibility.
    """
    
    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = True,
        name: str = "multi_task_loss"
    ):
        self.name = name
        self.task_weights = task_weights or {
            "concepts": 1.0,
            "segment": 1.0,
            "ner": 1.0,
            "recommendation": 1.0
        }
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Learnable log variances for uncertainty weighting
        if use_uncertainty_weighting:
            self.log_vars = {
                task: tf.Variable(0.0, trainable=True, name=f"log_var_{task}")
                for task in self.task_weights.keys()
            }
    
    def compute_task_loss(
        self,
        task: str,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """Compute loss for a specific task."""
        if task == "concepts":
            # Binary cross-entropy for multi-label
            return tf.reduce_mean(
                keras.losses.binary_crossentropy(y_true, y_pred)
            )
        elif task == "segment":
            # Sparse categorical cross-entropy
            return tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            )
        elif task == "ner":
            # Sparse categorical cross-entropy per token
            # Mask padding tokens (label = 0 is valid O tag)
            return tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            )
        elif task == "recommendation":
            # Binary cross-entropy for multi-label actions
            return tf.reduce_mean(
                keras.losses.binary_crossentropy(y_true, y_pred)
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def __call__(
        self,
        y_true: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute combined multi-task loss.
        
        Returns:
            total_loss: Combined weighted loss
            task_losses: Individual task losses
        """
        task_losses = {}
        total_loss = tf.constant(0.0)
        
        for task in self.task_weights.keys():
            if task not in y_pred:
                continue
            
            # Get labels
            label_key = f"{task}_labels" if task != "segment" else "segment_labels"
            if task == "concepts":
                label_key = "concept_labels"
            elif task == "recommendation":
                label_key = "action_labels"
            elif task == "ner":
                label_key = "ner_labels"
            
            if label_key not in y_true:
                continue
            
            # Compute task loss
            loss = self.compute_task_loss(task, y_true[label_key], y_pred[task])
            task_losses[task] = loss
            
            # Apply weighting
            if self.use_uncertainty_weighting:
                # Uncertainty weighting: L = 0.5 * exp(-log_var) * loss + 0.5 * log_var
                precision = tf.exp(-self.log_vars[task])
                weighted_loss = 0.5 * precision * loss + 0.5 * self.log_vars[task]
            else:
                weighted_loss = self.task_weights[task] * loss
            
            total_loss += weighted_loss
        
        return total_loss, task_losses


class ModelTrainer:
    """
    Trainer for the multi-task LVMH model.
    """
    
    def __init__(
        self,
        model: LVMHMultiTaskModel,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        task_weights: Optional[Dict[str, float]] = None,
        use_uncertainty_weighting: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Loss function
        self.loss_fn = MultiTaskLoss(
            task_weights=task_weights,
            use_uncertainty_weighting=use_uncertainty_weighting
        )
        
        # Optimizer with warmup
        self.optimizer = self._create_optimizer()
        
        # Metrics
        self.metrics = self._create_metrics()
        
        # Training state
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        # History
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "task_losses": {task: [] for task in ["concepts", "segment", "ner", "recommendation"]}
        }
    
    def _create_optimizer(self) -> optimizers.Optimizer:
        """Create optimizer with learning rate schedule."""
        # Linear warmup then cosine decay
        lr_schedule = optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=10000,
            alpha=0.1
        )
        
        return optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.01,
            clipnorm=1.0
        )
    
    def _create_metrics(self) -> Dict[str, keras.metrics.Metric]:
        """Create metrics for each task."""
        return {
            "concepts_precision": keras.metrics.Precision(name="concepts_precision"),
            "concepts_recall": keras.metrics.Recall(name="concepts_recall"),
            "segment_accuracy": keras.metrics.SparseCategoricalAccuracy(name="segment_accuracy"),
            "ner_accuracy": keras.metrics.SparseCategoricalAccuracy(name="ner_accuracy"),
            "recommendation_precision": keras.metrics.Precision(name="recommendation_precision"),
        }
    
    @tf.function
    def train_step(
        self,
        batch: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Single training step."""
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(batch["input_ids"], training=True)
            
            # Compute loss
            total_loss, task_losses = self.loss_fn(batch, predictions)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self._update_metrics(batch, predictions)
        
        # Increment step
        self.global_step.assign_add(1)
        
        return total_loss, task_losses
    
    @tf.function
    def validation_step(
        self,
        batch: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Single validation step."""
        # Forward pass
        predictions = self.model(batch["input_ids"], training=False)
        
        # Compute loss
        total_loss, task_losses = self.loss_fn(batch, predictions)
        
        # Update metrics
        self._update_metrics(batch, predictions)
        
        return total_loss, task_losses
    
    def _update_metrics(
        self,
        batch: Dict[str, tf.Tensor],
        predictions: Dict[str, tf.Tensor]
    ):
        """Update task metrics."""
        if "concepts" in predictions and "concept_labels" in batch:
            preds = tf.cast(predictions["concepts"] > 0.5, tf.float32)
            self.metrics["concepts_precision"].update_state(batch["concept_labels"], preds)
            self.metrics["concepts_recall"].update_state(batch["concept_labels"], preds)
        
        if "segment" in predictions and "segment_labels" in batch:
            self.metrics["segment_accuracy"].update_state(batch["segment_labels"], predictions["segment"])
        
        if "ner" in predictions and "ner_labels" in batch:
            self.metrics["ner_accuracy"].update_state(batch["ner_labels"], predictions["ner"])
        
        if "recommendation" in predictions and "action_labels" in batch:
            preds = tf.cast(predictions["recommendation"] > 0.5, tf.float32)
            self.metrics["recommendation_precision"].update_state(batch["action_labels"], preds)
    
    def _reset_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset_state()
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        early_stopping_patience: int = 5,
        verbose: int = 1
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch (None = full dataset)
            validation_steps: Validation steps (None = full dataset)
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            self._reset_metrics()
            train_loss = 0.0
            train_steps = 0
            
            for step, batch in enumerate(train_dataset):
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                
                loss, task_losses = self.train_step(batch)
                train_loss += float(loss)
                train_steps += 1
                
                if verbose >= 2 and step % 100 == 0:
                    print(f"  Step {step}: loss = {float(loss):.4f}")
            
            avg_train_loss = train_loss / max(train_steps, 1)
            self.history["train_loss"].append(avg_train_loss)
            
            # Collect train metrics
            train_metrics = {
                name: float(metric.result())
                for name, metric in self.metrics.items()
            }
            
            # Validation
            if val_dataset is not None:
                self._reset_metrics()
                val_loss = 0.0
                val_steps = 0
                
                for step, batch in enumerate(val_dataset):
                    if validation_steps and step >= validation_steps:
                        break
                    
                    loss, _ = self.validation_step(batch)
                    val_loss += float(loss)
                    val_steps += 1
                
                avg_val_loss = val_loss / max(val_steps, 1)
                self.history["val_loss"].append(avg_val_loss)
                
                # Collect val metrics
                val_metrics = {
                    name: float(metric.result())
                    for name, metric in self.metrics.items()
                }
                
                # Early stopping
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint("best")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= early_stopping_patience:
                        if verbose >= 1:
                            print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
            else:
                avg_val_loss = None
                val_metrics = {}
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            if verbose >= 1:
                val_str = f", val_loss: {avg_val_loss:.4f}" if avg_val_loss else ""
                print(f"Epoch {epoch + 1}/{epochs} - {epoch_time:.1f}s - loss: {avg_train_loss:.4f}{val_str}")
                
                if verbose >= 2:
                    print(f"  Train metrics: {train_metrics}")
                    if val_metrics:
                        print(f"  Val metrics: {val_metrics}")
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint("final")
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        weights_path = self.checkpoint_dir / f"model_{name}.weights.h5"
        self.model.save_weights(str(weights_path))
        
        # Save config
        config_path = self.checkpoint_dir / f"model_{name}_config.json"
        with open(config_path, "w") as f:
            json.dump(self.model.get_config(), f, indent=2)
        
        # Save training state
        state_path = self.checkpoint_dir / f"trainer_{name}_state.json"
        with open(state_path, "w") as f:
            json.dump({
                "global_step": int(self.global_step.numpy()),
                "best_val_loss": self.best_val_loss,
                "history": self.history
            }, f, indent=2)
        
        print(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("No checkpoint directory specified")
        
        # Load model weights
        weights_path = self.checkpoint_dir / f"model_{name}.weights.h5"
        self.model.load_weights(str(weights_path))
        
        # Load training state
        state_path = self.checkpoint_dir / f"trainer_{name}_state.json"
        if state_path.exists():
            with open(state_path, "r") as f:
                state = json.load(f)
                self.global_step.assign(state["global_step"])
                self.best_val_loss = state["best_val_loss"]
                self.history = state["history"]
        
        print(f"Loaded checkpoint: {name}")
    
    def evaluate(
        self,
        dataset: tf.data.Dataset,
        steps: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        self._reset_metrics()
        total_loss = 0.0
        total_steps = 0
        
        for step, batch in enumerate(dataset):
            if steps and step >= steps:
                break
            
            loss, _ = self.validation_step(batch)
            total_loss += float(loss)
            total_steps += 1
        
        results = {
            "loss": total_loss / max(total_steps, 1),
            **{name: float(metric.result()) for name, metric in self.metrics.items()}
        }
        
        return results


def train_model(
    data_dir: str,
    output_dir: str,
    model_size: str = "base",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    vocab_size: int = 30000,
    maxlen: int = 256
):
    """
    Convenience function to train the model.
    
    Args:
        data_dir: Directory with pipeline outputs
        output_dir: Directory to save model
        model_size: "small", "base", or "large"
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        vocab_size: Vocabulary size
        maxlen: Maximum sequence length
    """
    from .model import create_model
    from .data_generator import DataGenerator
    
    # Load data
    print("Loading data...")
    data_gen = DataGenerator(
        data_dir=data_dir,
        vocab_size=vocab_size,
        maxlen=maxlen
    )
    data_gen.load_pipeline_outputs()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = data_gen.create_tf_dataset(batch_size=batch_size)
    
    # Create model
    print("Creating model...")
    model = create_model(
        vocab_size=vocab_size,
        maxlen=maxlen,
        num_concepts=data_gen.concept_encoder.num_labels,
        num_segments=data_gen.segment_encoder.num_labels,
        num_entity_types=data_gen.entity_encoder.num_labels,
        num_actions=data_gen.action_encoder.num_labels,
        model_size=model_size
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = ModelTrainer(
        model=model,
        learning_rate=learning_rate,
        checkpoint_dir=output_dir
    )
    
    # Train
    print("Training...")
    history = trainer.train(
        train_dataset=train_dataset,
        epochs=epochs,
        verbose=1
    )
    
    # Save encoders
    data_gen.save_encoders(output_dir)
    
    print(f"\nTraining complete! Model saved to {output_dir}")
    return model, history


if __name__ == "__main__":
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    
    train_model(
        data_dir=str(project_root / "data"),
        output_dir=str(project_root / "models" / "lvmh_multitask"),
        model_size="small",
        epochs=5,
        batch_size=16
    )
