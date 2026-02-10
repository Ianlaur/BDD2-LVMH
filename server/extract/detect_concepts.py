"""
Concept detection stage: Match lexicon concepts/aliases in notes.

This module:
- Loads the lexicon
- For each note, matches aliases using Aho-Corasick (single-pass, O(N+M))
- Records evidence spans (start/end indices)
- Limits overlapping matches

Output: data/outputs/note_concepts.csv
"""
import sys
import re
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd

from ahocorasick_rs import AhoCorasick, MatchKind

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


# ── Aho-Corasick automaton (built once, reused for all notes) ────

_WORD_BOUNDARY = re.compile(r'\b', re.UNICODE)


def build_aho_automaton(alias_map: Dict[str, str]):
    """
    Build an Aho-Corasick automaton from the alias map.

    Returns (automaton, patterns_list) where patterns_list[i] is the alias
    corresponding to pattern index i returned by the automaton.
    """
    # Filter out very short aliases, store in a list (index matters)
    patterns = [alias for alias in alias_map if len(alias) >= 2]
    # LeftmostLongest ensures longer matches take priority over shorter ones
    automaton = AhoCorasick(patterns, matchkind=MatchKind.LeftmostLongest)
    return automaton, patterns


def find_matches_in_text(
    text: str,
    alias_map: Dict[str, str],
    max_matches_per_alias: int = MAX_ALIAS_MATCHES_PER_NOTE,
    automaton: Optional[AhoCorasick] = None,
    patterns: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Find all alias matches in text with evidence spans.

    If *automaton* and *patterns* are provided they are used for O(N+M)
    Aho-Corasick matching.  Otherwise falls back to per-alias regex
    (kept for backward compatibility in tests).

    Returns list of match dicts with keys: concept_id, matched_alias, start, end
    """
    text_lower = text.lower()

    # ── fast path: Aho-Corasick ──────────────────────────────────
    if automaton is not None and patterns is not None:
        matches = []
        alias_counts: Dict[str, int] = {}

        # Pre-compute word-boundary positions for the lowered text
        boundary_set = {m.start() for m in _WORD_BOUNDARY.finditer(text_lower)}

        for pat_idx, start, end in automaton.find_matches_as_indexes(text_lower):
            # Word-boundary check: start and end must be at word boundaries
            if start not in boundary_set or end not in boundary_set:
                continue

            alias = patterns[pat_idx]

            if alias_counts.get(alias, 0) >= max_matches_per_alias:
                continue

            matches.append({
                "concept_id": alias_map[alias],
                "matched_alias": alias,
                "start": start,
                "end": end,
            })
            alias_counts[alias] = alias_counts.get(alias, 0) + 1

        matches.sort(key=lambda m: m["start"])
        return matches

    # ── slow path: per-alias regex (backward compat / tests) ─────
    matches = []
    alias_counts: Dict[str, int] = {}
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)
    occupied_spans: List[Tuple[int, int]] = []

    for alias in sorted_aliases:
        if len(alias) < 2:
            continue

        pattern = r'\b' + re.escape(alias) + r'\b'
        try:
            for match in re.finditer(pattern, text_lower):
                start, end = match.start(), match.end()

                overlaps = False
                for occ_start, occ_end in occupied_spans:
                    if not (end <= occ_start or start >= occ_end):
                        overlaps = True
                        break
                if overlaps:
                    continue
                if alias_counts.get(alias, 0) >= max_matches_per_alias:
                    continue

                matches.append({
                    "concept_id": alias_map[alias],
                    "matched_alias": alias,
                    "start": start,
                    "end": end,
                })
                occupied_spans.append((start, end))
                alias_counts[alias] = alias_counts.get(alias, 0) + 1
        except re.error:
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
    
    # Build Aho-Corasick automaton once for all notes
    automaton, patterns = build_aho_automaton(alias_map)
    log_stage("concepts", f"Built Aho-Corasick automaton ({len(patterns)} patterns)")
    
    # Detect concepts in each note
    all_matches = []
    notes_with_concepts = 0
    
    for _, note_row in notes_df.iterrows():
        note_id = note_row["note_id"]
        client_id = note_row["client_id"]
        text = note_row["text"]
        
        if not text:
            continue
        
        matches = find_matches_in_text(
            text, alias_map,
            automaton=automaton, patterns=patterns,
        )
        
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
