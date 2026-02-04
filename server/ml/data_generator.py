"""
Data Generator for LVMH Multi-Task Model

Generates training data from:
1. Existing pipeline outputs (pseudo-labels)
2. Lexicon for concept labels
3. Clustering results for segment labels
4. Manual annotations (if available)
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import re


class Tokenizer:
    """
    Simple tokenizer for multilingual text.
    Can be replaced with SentencePiece or BPE for production.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        maxlen: int = 256,
        oov_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]"
    ):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = defaultdict(int)
        
        # Reserve special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        special_tokens = [self.pad_token, self.oov_token, self.cls_token, self.sep_token]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
        self.special_token_count = len(special_tokens)
    
    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        # Count words
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                self.word_counts[token] += 1
        
        # Sort by frequency and take top vocab_size - special_tokens
        sorted_words = sorted(
            self.word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.vocab_size - self.special_token_count]
        
        # Build vocab
        for word, _ in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization: lowercase, split on whitespace and punctuation."""
        text = text.lower()
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b|[.,!?;:\'\"()-]', text)
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize(text)
        
        # Convert to IDs
        ids = [
            self.word2idx.get(token, self.word2idx[self.oov_token])
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            ids = [self.word2idx[self.cls_token]] + ids + [self.word2idx[self.sep_token]]
        
        # Truncate
        if truncation and len(ids) > self.maxlen:
            ids = ids[:self.maxlen - 1] + [self.word2idx[self.sep_token]]
        
        # Pad
        if padding:
            pad_length = self.maxlen - len(ids)
            ids = ids + [self.word2idx[self.pad_token]] * pad_length
        
        return ids
    
    def encode_batch(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode multiple texts."""
        return np.array([self.encode(text, **kwargs) for text in texts])
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special_ids = set([
            self.word2idx[self.pad_token],
            self.word2idx[self.oov_token],
            self.word2idx[self.cls_token],
            self.word2idx[self.sep_token]
        ]) if skip_special_tokens else set()
        
        tokens = [
            self.idx2word.get(idx, self.oov_token)
            for idx in ids
            if idx not in special_ids
        ]
        return " ".join(tokens)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            "vocab_size": self.vocab_size,
            "maxlen": self.maxlen,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load tokenizer from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            maxlen=data["maxlen"]
        )
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(k): v for k, v in enumerate(data["word2idx"].keys())}
        tokenizer.word_counts = defaultdict(int, data["word_counts"])
        return tokenizer


class LabelEncoder:
    """Encode categorical labels to indices."""
    
    def __init__(self):
        self.label2idx: Dict[str, int] = {}
        self.idx2label: Dict[int, str] = {}
    
    def fit(self, labels: List[str]):
        """Build label vocabulary."""
        unique_labels = sorted(set(labels))
        for idx, label in enumerate(unique_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label
    
    def encode(self, label: str) -> int:
        """Encode single label."""
        return self.label2idx.get(label, -1)
    
    def encode_multi(self, labels: List[str]) -> np.ndarray:
        """Encode multi-label (binary vector)."""
        vector = np.zeros(len(self.label2idx), dtype=np.float32)
        for label in labels:
            if label in self.label2idx:
                vector[self.label2idx[label]] = 1.0
        return vector
    
    def decode(self, idx: int) -> str:
        """Decode index to label."""
        return self.idx2label.get(idx, "UNKNOWN")
    
    def decode_multi(self, vector: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Decode binary vector to labels."""
        return [
            self.idx2label[i]
            for i, v in enumerate(vector)
            if v >= threshold
        ]
    
    @property
    def num_labels(self) -> int:
        return len(self.label2idx)
    
    def save(self, path: str):
        """Save encoder to file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"label2idx": self.label2idx}, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        """Load encoder from file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        encoder = cls()
        encoder.label2idx = data["label2idx"]
        encoder.idx2label = {int(v): k for k, v in data["label2idx"].items()}
        return encoder


class DataGenerator:
    """
    Generate training data for the multi-task model.
    
    Uses existing pipeline outputs as pseudo-labels:
    - notes_clean.parquet: Source texts
    - note_concepts.csv: Concept labels
    - client_profiles.csv: Segment labels
    - recommended_actions.csv: Action labels
    """
    
    def __init__(
        self,
        data_dir: str,
        vocab_size: int = 30000,
        maxlen: int = 256
    ):
        self.data_dir = Path(data_dir)
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        
        # Initialize encoders
        self.tokenizer = Tokenizer(vocab_size=vocab_size, maxlen=maxlen)
        self.concept_encoder = LabelEncoder()
        self.segment_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        
        # Data containers
        self.texts: List[str] = []
        self.concept_labels: List[List[str]] = []
        self.segment_labels: List[str] = []
        self.action_labels: List[List[str]] = []
        self.entity_annotations: List[List[Tuple[int, int, str]]] = []
    
    def load_pipeline_outputs(self):
        """Load data from existing pipeline outputs."""
        processed_dir = self.data_dir / "processed"
        outputs_dir = self.data_dir / "outputs"
        
        # Load notes
        notes_path = processed_dir / "notes_clean.parquet"
        if notes_path.exists():
            notes_df = pd.read_parquet(notes_path)
            self.texts = notes_df["text"].tolist()
            note_ids = notes_df["note_id"].tolist()
            print(f"Loaded {len(self.texts)} notes")
        else:
            raise FileNotFoundError(f"Notes file not found: {notes_path}")
        
        # Load concept labels
        concepts_path = outputs_dir / "note_concepts.csv"
        if concepts_path.exists():
            concepts_df = pd.read_csv(concepts_path)
            # Group concepts by note_id - use concept_id column
            concept_col = "concept_id" if "concept_id" in concepts_df.columns else "concept"
            note_concepts = concepts_df.groupby("note_id")[concept_col].apply(list).to_dict()
            self.concept_labels = [
                note_concepts.get(nid, []) for nid in note_ids
            ]
            all_concepts = concepts_df[concept_col].unique().tolist()
            self.concept_encoder.fit(all_concepts)
            print(f"Loaded {len(all_concepts)} unique concepts")
        
        # Load segment labels
        profiles_path = outputs_dir / "client_profiles.csv"
        if profiles_path.exists():
            profiles_df = pd.read_csv(profiles_path)
            # Map client_id to segment/profile_type
            segment_col = "profile_type" if "profile_type" in profiles_df.columns else "segment"
            note_segments = dict(zip(profiles_df["client_id"], profiles_df[segment_col]))
            self.segment_labels = [
                str(note_segments.get(nid.replace("NOTE_", "").split("_")[0] + "_" + nid.split("_")[1] if "_" in nid else nid, "unknown")) 
                for nid in note_ids
            ]
            segments = [str(s) for s in profiles_df[segment_col].unique()]
            self.segment_encoder.fit(segments)
            print(f"Loaded {len(segments)} segments")
        
        # Load action labels
        actions_path = outputs_dir / "recommended_actions.csv"
        if actions_path.exists():
            actions_df = pd.read_csv(actions_path)
            # Group actions by client - use action_id or title
            action_col = "action_id" if "action_id" in actions_df.columns else "action"
            client_actions = actions_df.groupby("client_id")[action_col].apply(list).to_dict()
            self.action_labels = [
                client_actions.get(nid.replace("NOTE_", "").split("_")[0] + "_" + nid.split("_")[1] if "_" in nid else nid, []) 
                for nid in note_ids
            ]
            all_actions = actions_df[action_col].unique().tolist()
            self.action_encoder.fit(all_actions)
            print(f"Loaded {len(all_actions)} unique actions")
        
        # Initialize entity encoder with common types
        entity_types = [
            "BRAND", "PRODUCT", "PERSON", "DATE", "MONEY",
            "LOCATION", "PREFERENCE", "OCCASION", "MATERIAL", "COLOR"
        ]
        self.entity_encoder.fit(entity_types)
        
        # Fit tokenizer on texts
        self.tokenizer.fit(self.texts)
        print(f"Built vocabulary with {len(self.tokenizer.word2idx)} tokens")
    
    def generate_ner_labels_from_lexicon(self, lexicon_path: str):
        """
        Generate NER labels by matching lexicon entries in texts.
        This is a form of distant supervision.
        """
        with open(lexicon_path, "r", encoding="utf-8") as f:
            lexicon = json.load(f)
        
        # Build pattern -> entity type mapping
        patterns = {}
        for concept, data in lexicon.items():
            entity_type = self._infer_entity_type(concept)
            for alias in data.get("aliases", [concept]):
                patterns[alias.lower()] = entity_type
        
        # Find matches in each text
        self.entity_annotations = []
        for text in self.texts:
            text_lower = text.lower()
            annotations = []
            for pattern, entity_type in patterns.items():
                start = 0
                while True:
                    pos = text_lower.find(pattern, start)
                    if pos == -1:
                        break
                    annotations.append((pos, pos + len(pattern), entity_type))
                    start = pos + 1
            self.entity_annotations.append(annotations)
    
    def _infer_entity_type(self, concept: str) -> str:
        """Infer entity type from concept name."""
        concept_lower = concept.lower()
        if any(w in concept_lower for w in ["brand", "marque", "louis", "dior", "fendi"]):
            return "BRAND"
        if any(w in concept_lower for w in ["sac", "bag", "montre", "watch", "bijou"]):
            return "PRODUCT"
        if any(w in concept_lower for w in ["noir", "black", "blanc", "white", "rose"]):
            return "COLOR"
        if any(w in concept_lower for w in ["cuir", "leather", "or", "gold", "argent"]):
            return "MATERIAL"
        if any(w in concept_lower for w in ["anniversaire", "wedding", "birthday"]):
            return "OCCASION"
        return "PREFERENCE"
    
    def create_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        tasks: Optional[List[str]] = None
    ) -> tf.data.Dataset:
        """
        Create TensorFlow dataset for training.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle
            tasks: Which tasks to include labels for
        
        Returns:
            tf.data.Dataset
        """
        if tasks is None:
            tasks = ["concepts", "segment", "ner", "recommendation"]
        
        # Encode texts
        input_ids = self.tokenizer.encode_batch(self.texts)
        
        # Prepare labels
        data = {"input_ids": input_ids}
        
        if "concepts" in tasks and self.concept_labels:
            concept_vectors = np.array([
                self.concept_encoder.encode_multi(labels)
                for labels in self.concept_labels
            ])
            data["concept_labels"] = concept_vectors
        
        if "segment" in tasks and self.segment_labels:
            segment_ids = np.array([
                self.segment_encoder.encode(label)
                for label in self.segment_labels
            ])
            data["segment_labels"] = segment_ids
        
        if "recommendation" in tasks and self.action_labels:
            action_vectors = np.array([
                self.action_encoder.encode_multi(labels)
                for labels in self.action_labels
            ])
            data["action_labels"] = action_vectors
        
        if "ner" in tasks and self.entity_annotations:
            # Convert entity annotations to BIO labels
            ner_labels = self._convert_to_bio_labels()
            data["ner_labels"] = ner_labels
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(data)
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.texts))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _convert_to_bio_labels(self) -> np.ndarray:
        """Convert entity annotations to BIO scheme labels."""
        # BIO: O=0, B-type=1+2*type_idx, I-type=2+2*type_idx
        num_labels = self.entity_encoder.num_labels * 2 + 1
        
        all_labels = []
        for text, annotations in zip(self.texts, self.entity_annotations):
            tokens = self.tokenizer._tokenize(text)
            labels = np.zeros(self.maxlen, dtype=np.int32)
            
            # Map character positions to token positions
            char_to_token = {}
            char_pos = 0
            for tok_idx, token in enumerate(tokens[:self.maxlen - 2]):  # Leave room for special tokens
                for _ in range(len(token)):
                    char_to_token[char_pos] = tok_idx + 1  # +1 for [CLS]
                    char_pos += 1
                char_pos += 1  # Space
            
            # Apply annotations
            for start, end, entity_type in annotations:
                type_idx = self.entity_encoder.encode(entity_type)
                if type_idx == -1:
                    continue
                
                # Get token positions
                for pos in range(start, end):
                    if pos in char_to_token:
                        tok_pos = char_to_token[pos]
                        if tok_pos < self.maxlen:
                            if pos == start:
                                labels[tok_pos] = 1 + 2 * type_idx  # B-type
                            else:
                                labels[tok_pos] = 2 + 2 * type_idx  # I-type
            
            all_labels.append(labels)
        
        return np.array(all_labels)
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        seed: int = 42
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Split data into train and test sets."""
        np.random.seed(seed)
        n = len(self.texts)
        indices = np.random.permutation(n)
        split = int(n * (1 - test_size))
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        def subset_data(indices):
            return DataGenerator(
                data_dir=str(self.data_dir),
                vocab_size=self.vocab_size,
                maxlen=self.maxlen
            )
        
        # For now, return full dataset (proper split implementation would filter)
        return self.create_tf_dataset(), self.create_tf_dataset()
    
    def save_encoders(self, output_dir: str):
        """Save all encoders to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer.save(str(output_path / "tokenizer.json"))
        self.concept_encoder.save(str(output_path / "concept_encoder.json"))
        self.segment_encoder.save(str(output_path / "segment_encoder.json"))
        self.action_encoder.save(str(output_path / "action_encoder.json"))
        self.entity_encoder.save(str(output_path / "entity_encoder.json"))
        
        print(f"Saved encoders to {output_path}")
    
    def load_encoders(self, input_dir: str):
        """Load all encoders from files."""
        input_path = Path(input_dir)
        
        self.tokenizer = Tokenizer.load(str(input_path / "tokenizer.json"))
        self.concept_encoder = LabelEncoder.load(str(input_path / "concept_encoder.json"))
        self.segment_encoder = LabelEncoder.load(str(input_path / "segment_encoder.json"))
        self.action_encoder = LabelEncoder.load(str(input_path / "action_encoder.json"))
        self.entity_encoder = LabelEncoder.load(str(input_path / "entity_encoder.json"))
        
        print(f"Loaded encoders from {input_path}")


if __name__ == "__main__":
    # Test data generator
    import sys
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    
    print(f"Loading data from {data_dir}")
    
    generator = DataGenerator(
        data_dir=str(data_dir),
        vocab_size=10000,
        maxlen=128
    )
    
    try:
        generator.load_pipeline_outputs()
        
        # Create dataset
        dataset = generator.create_tf_dataset(batch_size=4)
        
        # Print sample batch
        for batch in dataset.take(1):
            print("\nSample batch:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
        
        # Save encoders
        generator.save_encoders(str(project_root / "models" / "encoders"))
        
    except FileNotFoundError as e:
        print(f"Note: {e}")
        print("Run the main pipeline first to generate training data.")
