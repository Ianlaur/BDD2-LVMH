"""
Concept detection stage: Match lexicon concepts/aliases in notes.

This module:
- Loads the lexicon
- For each note, matches aliases case-insensitively
- Records evidence spans (start/end indices)
- Limits overlapping matches

Output: data/outputs/note_concepts.csv
"""
import sys
import re
from typing import List, Dict, Tuple, Set
import pandas as pd

from server.shared.config import (
    DATA_PROCESSED, DATA_OUTPUTS, TAXONOMY_DIR,
    MAX_ALIAS_MATCHES_PER_NOTE
)
from server.shared.utils import log_stage, set_all_seeds


def load_lexicon() -> pd.DataFrame:
    """Load the lexicon CSV."""
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.csv"
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon not found: {lexicon_path}. Run lexicon building first.")
    return pd.read_csv(lexicon_path)


def build_alias_to_concept_map(lexicon_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping from alias -> concept_id.
    Includes the label as an alias too.
    """
    alias_map = {}
    
    for _, row in lexicon_df.iterrows():
        concept_id = row["concept_id"]
        label = str(row["label"]).lower().strip()
        
        # Add label as alias
        alias_map[label] = concept_id
        
        # Add other aliases
        aliases_str = row.get("aliases", "")
        if pd.notna(aliases_str) and aliases_str:
            for alias in str(aliases_str).split("|"):
                alias = alias.lower().strip()
                if alias:
                    alias_map[alias] = concept_id
    
    return alias_map


def find_matches_in_text(
    text: str,
    alias_map: Dict[str, str],
    max_matches_per_alias: int = MAX_ALIAS_MATCHES_PER_NOTE
) -> List[Dict]:
    """
    Find all alias matches in text with evidence spans.
    Returns list of match dicts with keys: concept_id, matched_alias, start, end
    """
    text_lower = text.lower()
    matches = []
    
    # Track match counts per alias
    alias_counts: Dict[str, int] = {}
    
    # Sort aliases by length (longer first) to prefer longer matches
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    
    # Track occupied spans to avoid overlapping matches
    occupied_spans: List[Tuple[int, int]] = []
    
    for alias in sorted_aliases:
        if len(alias) < 2:  # Skip very short aliases
            continue
        
        # Build regex pattern for word boundary matching
        # Escape special regex characters
        pattern = re.escape(alias)
        # Use word boundaries where possible
        pattern = r'\b' + pattern + r'\b'
        
        try:
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()
                
                # Check for overlap with existing matches
                overlaps = False
                for occ_start, occ_end in occupied_spans:
                    if not (end <= occ_start or start >= occ_end):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                # Check max matches per alias
                if alias_counts.get(alias, 0) >= max_matches_per_alias:
                    continue
                
                # Record match
                matches.append({
                    "concept_id": alias_map[alias],
                    "matched_alias": alias,
                    "start": start,
                    "end": end
                })
                
                occupied_spans.append((start, end))
                alias_counts[alias] = alias_counts.get(alias, 0) + 1
                
        except re.error:
            # Skip problematic patterns
            continue
    
    # Sort matches by position
    matches.sort(key=lambda m: m["start"])
    
    return matches


def detect_concepts() -> pd.DataFrame:
    """
    Main concept detection function.
    
    Returns:
        DataFrame with note-concept matches
        
    Side effects:
        Writes data/outputs/note_concepts.csv
    """
    set_all_seeds()
    
    log_stage("concepts", "Starting concept detection...")
    
    # Load notes
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    if not notes_path.exists():
        raise FileNotFoundError(f"Notes not found: {notes_path}. Run ingest first.")
    
    notes_df = pd.read_parquet(notes_path)
    log_stage("concepts", f"Loaded {len(notes_df)} notes")
    
    # Load lexicon and build alias map
    lexicon_df = load_lexicon()
    alias_map = build_alias_to_concept_map(lexicon_df)
    log_stage("concepts", f"Built alias map with {len(alias_map)} entries")
    
    if len(alias_map) == 0:
        log_stage("concepts", "WARNING: No aliases in lexicon. Creating empty output.")
        empty_df = pd.DataFrame(columns=[
            "note_id", "client_id", "concept_id", "matched_alias", 
            "evidence_span_start", "evidence_span_end"
        ])
        output_path = DATA_OUTPUTS / "note_concepts.csv"
        empty_df.to_csv(output_path, index=False)
        return empty_df
    
    # Detect concepts in each note
    all_matches = []
    notes_with_concepts = 0
    
    for _, note_row in notes_df.iterrows():
        note_id = note_row["note_id"]
        client_id = note_row["client_id"]
        text = note_row["text"]
        
        if not text:
            continue
        
        matches = find_matches_in_text(text, alias_map)
        
        if matches:
            notes_with_concepts += 1
        
        for match in matches:
            all_matches.append({
                "note_id": note_id,
                "client_id": client_id,
                "concept_id": match["concept_id"],
                "matched_alias": match["matched_alias"],
                "evidence_span_start": match["start"],
                "evidence_span_end": match["end"]
            })
    
    # Build output DataFrame
    concepts_df = pd.DataFrame(all_matches)
    
    # Sort for determinism
    if len(concepts_df) > 0:
        concepts_df = concepts_df.sort_values(
            ["note_id", "evidence_span_start"]
        ).reset_index(drop=True)
    
    # Write output
    output_path = DATA_OUTPUTS / "note_concepts.csv"
    concepts_df.to_csv(output_path, index=False)
    log_stage("concepts", f"Wrote {len(concepts_df)} matches to {output_path}")
    
    # Coverage stats
    coverage = (notes_with_concepts / len(notes_df) * 100) if len(notes_df) > 0 else 0
    log_stage("concepts", f"Notes with concepts: {notes_with_concepts}/{len(notes_df)} ({coverage:.1f}%)")
    
    if len(concepts_df) > 0:
        concepts_per_note = concepts_df.groupby("note_id").size()
        log_stage("concepts", f"Avg concepts per note: {concepts_per_note.mean():.1f}")
        
        # Top concepts
        top_concepts = concepts_df["concept_id"].value_counts().head(10)
        log_stage("concepts", "Top 10 detected concepts:")
        for concept_id, count in top_concepts.items():
            label = lexicon_df[lexicon_df["concept_id"] == concept_id]["label"].values
            label_str = label[0] if len(label) > 0 else concept_id
            log_stage("concepts", f"  {label_str}: {count} matches")
    
    log_stage("concepts", "Concept detection complete!")
    
    return concepts_df


def main():
    """CLI entry point."""
    try:
        detect_concepts()
    except Exception as e:
        log_stage("concepts", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
