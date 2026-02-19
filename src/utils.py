"""
Utility functions for the LVMH Client Intelligence pipeline.
"""
import random
import numpy as np
from pathlib import Path
from typing import Optional

from src.config import RANDOM_SEED, NUMPY_SEED


def set_all_seeds(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility."""
    seed = seed or RANDOM_SEED
    random.seed(seed)
    np.random.seed(NUMPY_SEED)
    

def ensure_directories() -> None:
    """Create all required output directories if they don't exist."""
    from src.config import DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR, ACTIVATIONS_DIR, DEMO_DIR
    
    for d in [DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR, ACTIVATIONS_DIR, DEMO_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def log_stage(stage: str, message: str) -> None:
    """Print a formatted log message for a pipeline stage."""
    print(f"[{stage}] {message}")


def slugify(text: str) -> str:
    """Convert text to a clean slug."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '_', text)
    return text[:50]  # limit length


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, collapse whitespace, strip."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text
