"""
Vector building stage: Embed notes using SentenceTransformer.

This module:
- Loads notes
- Computes embeddings for each note's transcription
- Aggregates to client level (mean for multiple notes)
- Saves note_vectors.parquet

Output: data/outputs/note_vectors.parquet
"""
import sys
from typing import List, Dict
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    DATA_PROCESSED, DATA_OUTPUTS,
    SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMERS_CACHE
)
from src.utils import log_stage, set_all_seeds


def load_sentence_transformer() -> SentenceTransformer:
    """Load the SentenceTransformer model with proper cache folder."""
    log_stage("vectors", f"Loading SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}")
    
    model = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL,
        cache_folder=str(SENTENCE_TRANSFORMERS_CACHE)
    )
    return model


def build_vectors() -> pd.DataFrame:
    """
    Main vector building function.
    
    Returns:
        DataFrame with note vectors
        
    Side effects:
        Writes data/outputs/note_vectors.parquet
    """
    set_all_seeds()
    
    log_stage("vectors", "Starting vector building...")
    
    # Load notes
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes not found: {notes_path}. Run ingest first.")
    
    notes_df = pd.read_parquet(notes_path)
    log_stage("vectors", f"Loaded {len(notes_df)} notes")
    
    # Load model
    model = load_sentence_transformer()
    
    # Get texts
    texts = notes_df["text"].fillna("").tolist()
    note_ids = notes_df["note_id"].tolist()
    client_ids = notes_df["client_id"].tolist()
    
    # Compute embeddings
    log_stage("vectors", f"Computing embeddings for {len(texts)} notes...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )
    
    log_stage("vectors", f"Embedding dimension: {embeddings.shape[1]}")
    
    # Build output DataFrame
    # Store embeddings as lists (JSON-serializable for parquet)
    vectors_data = []
    for i, (note_id, client_id) in enumerate(zip(note_ids, client_ids)):
        vectors_data.append({
            "note_id": note_id,
            "client_id": client_id,
            "embedding": embeddings[i].tolist()
        })
    
    vectors_df = pd.DataFrame(vectors_data)
    
    # For MVP: client_vector = note_vector since 1 note per client
    # Future: aggregate multiple notes per client with mean
    # Check if there are multiple notes per client
    notes_per_client = vectors_df.groupby("client_id").size()
    if notes_per_client.max() > 1:
        log_stage("vectors", f"Note: Some clients have multiple notes (max={notes_per_client.max()})")
        log_stage("vectors", "Client vectors will be mean-aggregated in profiling stage")
    
    # Sort for determinism
    vectors_df = vectors_df.sort_values("note_id").reset_index(drop=True)
    
    # Write output
    output_path = DATA_OUTPUTS / "note_vectors.parquet"
    vectors_df.to_parquet(output_path, index=False)
    log_stage("vectors", f"Wrote {len(vectors_df)} note vectors to {output_path}")
    
    # Basic stats
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    log_stage("vectors", f"Embedding L2 norm: mean={embedding_norms.mean():.4f}, std={embedding_norms.std():.4f}")
    
    log_stage("vectors", "Vector building complete!")
    
    return vectors_df


def main():
    """CLI entry point."""
    try:
        build_vectors()
    except Exception as e:
        log_stage("vectors", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
