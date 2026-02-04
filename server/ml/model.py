"""
LVMH Multi-Task TensorFlow Model

Architecture:
- Shared Encoder: Transformer-based text encoder (fine-tunable)
- Task Heads:
  1. ConceptHead: Multi-label classification for concept detection
  2. SegmentHead: Multi-class classification for client segmentation
  3. NERHead: Sequence labeling for entity extraction
  4. RecommendationHead: Action/product recommendation
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Optional, Tuple
import numpy as np


class TransformerBlock(layers.Layer):
    """Custom Transformer block for the encoder."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        # Self-attention with residual
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    """Combined token and positional embedding."""
    
    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, 
            output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen, 
            output_dim=embed_dim
        )

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


class SharedEncoder(layers.Layer):
    """
    Shared text encoder backbone.
    Produces contextual embeddings for all downstream tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        maxlen: int = 512,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_transformer_blocks: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        
        # Embedding layer
        self.embedding = TokenAndPositionEmbedding(
            maxlen=maxlen,
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate,
                name=f"transformer_block_{i}"
            )
            for i in range(num_transformer_blocks)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        # Get embeddings
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=mask)
        
        return x  # Shape: (batch, seq_len, embed_dim)
    
    def get_pooled_output(self, sequence_output):
        """Get [CLS]-like pooled representation (mean pooling)."""
        return tf.reduce_mean(sequence_output, axis=1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "dropout_rate": self.dropout_rate,
        })
        return config


class ConceptHead(layers.Layer):
    """
    Multi-label classification head for concept detection.
    Predicts which concepts are present in the text.
    """
    
    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(hidden_dim // 2, activation="relu")
        self.output_layer = layers.Dense(num_concepts, activation="sigmoid")

    def call(self, pooled_output, training=False):
        x = self.dense1(pooled_output)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_concepts": self.num_concepts,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class SegmentHead(layers.Layer):
    """
    Multi-class classification head for segment prediction.
    Predicts which client segment the text belongs to.
    """
    
    def __init__(
        self,
        num_segments: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_segments, activation="softmax")

    def call(self, pooled_output, training=False):
        x = self.dense1(pooled_output)
        x = self.dropout(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_segments": self.num_segments,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class NERHead(layers.Layer):
    """
    Sequence labeling head for Named Entity Recognition.
    Predicts entity labels for each token (BIO scheme).
    """
    
    def __init__(
        self,
        num_entity_types: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        use_crf: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_entity_types = num_entity_types
        # BIO scheme: B-type, I-type, O for each entity type
        self.num_labels = num_entity_types * 2 + 1  # B, I for each type + O
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.use_crf = use_crf
        
        self.dense = layers.Dense(hidden_dim, activation="relu")
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(self.num_labels)
        
        # Optional CRF layer for structured prediction
        if use_crf:
            self.transition_params = self.add_weight(
                name="transition_params",
                shape=(self.num_labels, self.num_labels),
                initializer="glorot_uniform",
                trainable=True
            )

    def call(self, sequence_output, training=False, sequence_lengths=None):
        x = self.dense(sequence_output)
        x = self.dropout(x, training=training)
        logits = self.output_layer(x)
        
        if not self.use_crf:
            return tf.nn.softmax(logits, axis=-1)
        
        return logits  # Return raw logits for CRF decoding
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_entity_types": self.num_entity_types,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "use_crf": self.use_crf,
        })
        return config


class RecommendationHead(layers.Layer):
    """
    Recommendation head for predicting best actions/products.
    Uses attention over available actions based on client context.
    """
    
    def __init__(
        self,
        num_actions: int,
        action_embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Learnable action embeddings
        self.action_embeddings = self.add_weight(
            name="action_embeddings",
            shape=(num_actions, action_embed_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # Context projection
        self.context_projection = layers.Dense(action_embed_dim)
        
        # Scoring network
        self.score_net = keras.Sequential([
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(1)
        ])

    def call(self, pooled_output, training=False):
        batch_size = tf.shape(pooled_output)[0]
        
        # Project context to action space
        context = self.context_projection(pooled_output)  # (batch, action_embed_dim)
        
        # Expand for broadcasting
        context_expanded = tf.expand_dims(context, 1)  # (batch, 1, action_embed_dim)
        actions_expanded = tf.expand_dims(self.action_embeddings, 0)  # (1, num_actions, action_embed_dim)
        actions_expanded = tf.tile(actions_expanded, [batch_size, 1, 1])
        
        # Combine context with each action
        combined = tf.concat([
            context_expanded * actions_expanded,  # Element-wise interaction
            tf.abs(context_expanded - actions_expanded),  # Difference
        ], axis=-1)  # (batch, num_actions, 2*action_embed_dim)
        
        # Score each action
        scores = tf.squeeze(self.score_net(combined), axis=-1)  # (batch, num_actions)
        
        return tf.nn.softmax(scores, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_actions": self.num_actions,
            "action_embed_dim": self.action_embed_dim,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class LVMHMultiTaskModel(Model):
    """
    Complete multi-task model for LVMH Client Intelligence.
    
    Combines:
    - Shared text encoder
    - Concept detection head
    - Segment prediction head
    - NER head for entity extraction
    - Recommendation head
    
    All heads share the encoder, enabling transfer learning
    and efficient inference.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        maxlen: int = 512,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_transformer_blocks: int = 4,
        num_concepts: int = 100,
        num_segments: int = 8,
        num_entity_types: int = 10,
        num_actions: int = 50,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.config = {
            "vocab_size": vocab_size,
            "maxlen": maxlen,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_transformer_blocks": num_transformer_blocks,
            "num_concepts": num_concepts,
            "num_segments": num_segments,
            "num_entity_types": num_entity_types,
            "num_actions": num_actions,
            "dropout_rate": dropout_rate,
        }
        
        # Shared encoder
        self.encoder = SharedEncoder(
            vocab_size=vocab_size,
            maxlen=maxlen,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            dropout_rate=dropout_rate
        )
        
        # Task heads
        self.concept_head = ConceptHead(
            num_concepts=num_concepts,
            hidden_dim=embed_dim,
            dropout_rate=dropout_rate
        )
        
        self.segment_head = SegmentHead(
            num_segments=num_segments,
            hidden_dim=embed_dim // 2,
            dropout_rate=dropout_rate
        )
        
        self.ner_head = NERHead(
            num_entity_types=num_entity_types,
            hidden_dim=embed_dim // 2,
            dropout_rate=dropout_rate
        )
        
        self.recommendation_head = RecommendationHead(
            num_actions=num_actions,
            action_embed_dim=embed_dim // 4,
            hidden_dim=embed_dim // 2,
            dropout_rate=dropout_rate
        )

    def call(
        self,
        inputs,
        training=False,
        return_embeddings=False,
        tasks: Optional[List[str]] = None
    ):
        """
        Forward pass through the model.
        
        Args:
            inputs: Token IDs (batch, seq_len)
            training: Whether in training mode
            return_embeddings: Whether to return encoder embeddings
            tasks: Which task outputs to compute (None = all)
        
        Returns:
            Dictionary with outputs for each task
        """
        if tasks is None:
            tasks = ["concepts", "segment", "ner", "recommendation"]
        
        # Get encoder outputs
        sequence_output = self.encoder(inputs, training=training)
        pooled_output = self.encoder.get_pooled_output(sequence_output)
        
        outputs = {}
        
        if return_embeddings:
            outputs["embeddings"] = pooled_output
            outputs["sequence_embeddings"] = sequence_output
        
        # Compute requested task outputs
        if "concepts" in tasks:
            outputs["concepts"] = self.concept_head(pooled_output, training=training)
        
        if "segment" in tasks:
            outputs["segment"] = self.segment_head(pooled_output, training=training)
        
        if "ner" in tasks:
            outputs["ner"] = self.ner_head(sequence_output, training=training)
        
        if "recommendation" in tasks:
            outputs["recommendation"] = self.recommendation_head(pooled_output, training=training)
        
        return outputs

    def get_embeddings(self, inputs):
        """Get text embeddings without computing task heads."""
        sequence_output = self.encoder(inputs, training=False)
        return self.encoder.get_pooled_output(sequence_output)
    
    def predict_concepts(self, inputs, threshold: float = 0.5):
        """Predict concepts with threshold."""
        outputs = self.call(inputs, training=False, tasks=["concepts"])
        probs = outputs["concepts"]
        return tf.cast(probs >= threshold, tf.int32), probs
    
    def predict_segment(self, inputs):
        """Predict segment with confidence."""
        outputs = self.call(inputs, training=False, tasks=["segment"])
        probs = outputs["segment"]
        return tf.argmax(probs, axis=-1), tf.reduce_max(probs, axis=-1)
    
    def predict_entities(self, inputs):
        """Predict NER entities."""
        outputs = self.call(inputs, training=False, tasks=["ner"])
        return tf.argmax(outputs["ner"], axis=-1)
    
    def recommend_actions(self, inputs, top_k: int = 5):
        """Get top-k recommended actions."""
        outputs = self.call(inputs, training=False, tasks=["recommendation"])
        probs = outputs["recommendation"]
        top_indices = tf.argsort(probs, axis=-1, direction="DESCENDING")[:, :top_k]
        top_probs = tf.sort(probs, axis=-1, direction="DESCENDING")[:, :top_k]
        return top_indices, top_probs
    
    def get_config(self):
        return self.config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(
    vocab_size: int = 30000,
    maxlen: int = 256,
    num_concepts: int = 100,
    num_segments: int = 8,
    num_entity_types: int = 10,
    num_actions: int = 50,
    model_size: str = "base"
) -> LVMHMultiTaskModel:
    """
    Factory function to create model with preset configurations.
    
    Args:
        vocab_size: Vocabulary size
        maxlen: Maximum sequence length
        num_concepts: Number of concept labels
        num_segments: Number of client segments
        num_entity_types: Number of entity types for NER
        num_actions: Number of possible actions
        model_size: "small", "base", or "large"
    
    Returns:
        Configured LVMHMultiTaskModel
    """
    configs = {
        "small": {
            "embed_dim": 128,
            "num_heads": 4,
            "ff_dim": 256,
            "num_transformer_blocks": 2,
            "dropout_rate": 0.1,
        },
        "base": {
            "embed_dim": 256,
            "num_heads": 8,
            "ff_dim": 512,
            "num_transformer_blocks": 4,
            "dropout_rate": 0.1,
        },
        "large": {
            "embed_dim": 512,
            "num_heads": 8,
            "ff_dim": 1024,
            "num_transformer_blocks": 6,
            "dropout_rate": 0.15,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return LVMHMultiTaskModel(
        vocab_size=vocab_size,
        maxlen=maxlen,
        num_concepts=num_concepts,
        num_segments=num_segments,
        num_entity_types=num_entity_types,
        num_actions=num_actions,
        **config
    )


if __name__ == "__main__":
    # Test model creation
    model = create_model(
        vocab_size=10000,
        maxlen=128,
        num_concepts=50,
        num_segments=8,
        num_entity_types=5,
        num_actions=20,
        model_size="small"
    )
    
    # Test forward pass
    dummy_input = tf.random.uniform((2, 128), minval=0, maxval=10000, dtype=tf.int32)
    outputs = model(dummy_input, training=False)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {model.count_params():,}")
