"""
Qwen-Enhanced Concept Extraction
==================================

Uses local Ollama (qwen2.5:3b) to semantically analyze client notes
and extract richer concept information that rule-based matching misses.

Key improvements over pure rule-based:
  1. Detects implicit concepts (e.g., "Racing Club" → lifestyle:golf/sports)
  2. Assigns graded confidence scores (0.0-1.0)
  3. Extracts relationships between concepts (gift FOR someone, budget FOR occasion)
  4. Understands context across languages
  5. Identifies client sentiment and urgency

Usage:
    # Enhance all notes (after pipeline runs detection)
    python -m server.extract.qwen_enhance

    # Enhance specific notes
    python -m server.extract.qwen_enhance --limit 10

    # Resume from checkpoint
    python -m server.extract.qwen_enhance --resume
"""
import json
import time
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger("qwen_enhance")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_formatter)
    logger.addHandler(_sh)
    _fh = logging.FileHandler(Path(__file__).resolve().parent.parent.parent / "qwen_enhance.log", mode='a')
    _fh.setFormatter(_formatter)
    logger.addHandler(_fh)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TAXONOMY_DIR = BASE_DIR / "taxonomy"
DATA_DIR = BASE_DIR / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_OUTPUTS = DATA_DIR / "outputs"
CHECKPOINT_DIR = DATA_DIR / "qwen_checkpoints"

OLLAMA_URL = "http://localhost:11434"


# ── Ollama Client ────────────────────────────────────────────────────────

def ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return resp.status_code == 200
    except:
        return False


def ollama_chat(model: str, messages: List[Dict], temperature: float = 0.2,
                format_json: bool = False) -> Optional[str]:
    """Send chat request to Ollama with aggressive timeouts for 8GB RAM."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 2048,      # halved — less RAM pressure
            "num_predict": 512,   # force short answers
            "repeat_penalty": 1.3, # strongly penalize repetition loops
        }
    }
    # NOTE: we do NOT set format=json — it causes generation loops on small models
    if format_json:
        payload["format"] = "json"

    max_retries = 2
    timeout_s = 45  # aggressive — if it can't answer in 45s, skip

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=timeout_s
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout attempt {attempt+1}/{max_retries} ({timeout_s}s)")
            time.sleep(2)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Run: ollama serve")
            return None
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2)
    return None


def parse_json_response(raw: str) -> Optional[Dict]:
    """Parse JSON from LLM response with aggressive cleanup."""
    if not raw:
        return None
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip markdown fences
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Remove JS-style comments (// ...)
    import re
    text = re.sub(r'//[^\n]*', '', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first JSON object
    try:
        start = text.index('{')
        # Find balanced close
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}': depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                candidate = re.sub(r'//[^\n]*', '', candidate)
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                return json.loads(candidate)
    except (ValueError, json.JSONDecodeError):
        pass
    # Last resort: try fixing truncated JSON
    try:
        start = raw.index('{')
        text = raw[start:].rstrip().rstrip(',')
        text = re.sub(r'//[^\n]*', '', text)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        opens = text.count('{') - text.count('}')
        brackets = text.count('[') - text.count(']')
        text += ']' * max(0, brackets) + '}' * max(0, opens)
        return json.loads(text)
    except:
        pass
    return None


# ── Checkpoint System ────────────────────────────────────────────────────

class CheckpointManager:
    """Save/resume progress for long runs."""

    def __init__(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / "qwen_enhance_checkpoint.json"
        self.data = self._load()

    def _load(self) -> Dict:
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {"completed_notes": [], "results": {}, "started": datetime.now().isoformat()}

    def is_done(self, note_id: str) -> bool:
        return note_id in self.data["completed_notes"]

    def save_result(self, note_id: str, result: Dict):
        self.data["completed_notes"].append(note_id)
        self.data["results"][note_id] = result
        with open(self.path, 'w') as f:
            json.dump(self.data, f)

    def get_results(self) -> Dict:
        return self.data["results"]

    def reset(self):
        self.data = {"completed_notes": [], "results": {}, "started": datetime.now().isoformat()}
        if self.path.exists():
            self.path.unlink()


# ── Core Enhancement System ──────────────────────────────────────────────

# ── Core Enhancement System ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are an LVMH luxury retail analyst. Extract client intelligence from advisor notes.

Return a JSON object with this structure:
{"concepts": [{"label": "médecin généraliste", "bucket": "lifestyle", "confidence": 0.8, "evidence": "médecin généraliste, 38 ans"}], "client_summary": "One sentence profile", "urgency": "medium", "sentiment": "positive"}

Buckets: preferences, intent, lifestyle, occasion, constraints, next_action
Confidence: 1.0 verbatim, 0.8 implied, 0.6 inferred, 0.4 weak
Labels must be short natural phrases (2-4 words). Evidence must be exact quotes from the note."""


def enhance_note(model: str, note_id: str, text: str, 
                 rule_concepts: List[str]) -> Optional[Dict]:
    """
    Use Qwen to semantically analyze a single note.
    
    Args:
        model: Ollama model name
        note_id: Note identifier
        text: The note text
        rule_concepts: Already-detected concepts (for context)
        
    Returns:
        Structured extraction result or None
    """
    user_prompt = f"""Analyze this LVMH client note. Return JSON with specific concepts the rules missed.

NOTE ({note_id}):
\"\"\"{text[:500]}\"\"\"

Already detected (skip these): {', '.join(rule_concepts[:10]) if rule_concepts else 'none'}

Examples of good labels: "médecin 38 ans", "budget 4000€", "cadeau mère", "finitions dorées", "allergie nickel", "graver initiales". Be specific, not generic."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    raw = ollama_chat(model, messages, temperature=0.15, format_json=True)
    result = parse_json_response(raw)

    if not result:
        logger.warning(f"  Failed to parse response for {note_id}")
        return None

    # Validate and clean concepts
    concepts = result.get("concepts", [])
    valid_concepts = []
    seen_labels = set()
    valid_buckets = {"preferences", "intent", "lifestyle", "occasion", "constraints", "next_action"}

    for c in concepts:
        if not isinstance(c, dict):
            continue
        label = (c.get("label") or "").strip().lower()
        bucket = (c.get("bucket") or "").strip().lower()
        confidence = c.get("confidence", 0.5)

        if not label or bucket not in valid_buckets:
            continue
        if not isinstance(confidence, (int, float)):
            confidence = 0.5
        confidence = max(0.1, min(1.0, float(confidence)))

        # Quality filters
        # 1. Reject programmatic/snake_case labels
        if label.count('_') >= 2:
            continue
        # 2. Reject labels that are too long (>50 chars = likely garbage)
        if len(label) > 50:
            continue
        # 3. Reject numbered duplicates (e.g., "concept_1", "concept_2")
        if any(label.endswith(f"_{i}") or label.endswith(f" {i}") for i in range(2, 20)):
            continue
        # 4. Dedup — skip if we already have this exact label
        if label in seen_labels:
            continue
        seen_labels.add(label)
        # 5. Verify evidence exists in the original text (anti-hallucination)
        evidence = (c.get("evidence") or "")[:200]
        if evidence and text:
            # Check if at least some words from evidence appear in the note
            evidence_words = set(evidence.lower().split())
            text_words = set(text.lower().split())
            overlap = evidence_words & text_words
            if len(evidence_words) > 2 and len(overlap) < len(evidence_words) * 0.3:
                confidence = max(0.3, confidence - 0.3)  # Penalize weak evidence

        valid_concepts.append({
            "label": label,
            "bucket": bucket,
            "confidence": round(confidence, 2),
            "evidence": evidence,
            "language": (c.get("language") or "?")[:3],
        })

    result["concepts"] = valid_concepts
    return result


def merge_with_rule_based(
    note_id: str,
    rule_matches: pd.DataFrame,
    llm_result: Dict,
    vocab: Dict
) -> List[Dict]:
    """
    Merge rule-based and LLM-detected concepts.
    
    Strategy:
    - Rule-based matches keep confidence=1.0
    - LLM concepts that overlap with rules boost context (keep rule version)  
    - LLM concepts that are NEW get added with LLM confidence
    - Dedup by concept label similarity
    """
    merged = []

    # 1. Keep all rule-based matches (confidence=1.0)
    existing_labels = set()
    for _, row in rule_matches.iterrows():
        merged.append({
            "note_id": note_id,
            "concept_id": row.get("concept_id", ""),
            "matched_alias": row.get("matched_alias", ""),
            "start": row.get("start", 0),
            "end": row.get("end", 0),
            "detection_method": "rule-based",
            "confidence": 1.0,
        })
        existing_labels.add(row.get("matched_alias", "").lower())

    # Build reverse lookup: label → concept_id
    label_to_cid = {}
    for cid, d in vocab.items():
        label_to_cid[d["label"].lower()] = cid
        for alias in d.get("aliases", []):
            label_to_cid[alias.lower()] = cid

    # 2. Add LLM concepts that are genuinely new
    llm_concepts = llm_result.get("concepts", [])
    for c in llm_concepts:
        label = c["label"]
        # Skip if already matched by rules
        if label in existing_labels:
            continue
        # Check if it matches any known concept
        cid = label_to_cid.get(label, "")
        if not cid:
            # Try partial match
            for known_label, known_cid in label_to_cid.items():
                if label in known_label or known_label in label:
                    cid = known_cid
                    break

        if not cid:
            # New concept discovered by LLM — generate a temporary ID
            cid = f"LLM_{label.replace(' ', '_')[:30]}"

        merged.append({
            "note_id": note_id,
            "concept_id": cid,
            "matched_alias": label,
            "start": -1,  # LLM doesn't give character positions
            "end": -1,
            "detection_method": "llm-enhanced",
            "confidence": c["confidence"],
        })
        existing_labels.add(label)

    return merged


# ── Main Enhancement Pipeline ────────────────────────────────────────────

def run_enhancement(
    model: str = "qwen2.5:3b",
    limit: int = 0,
    resume: bool = False,
    batch_size: int = 1,
):
    """
    Run Qwen enhancement on all notes.
    
    Args:
        model: Ollama model name
        limit: Max notes to process (0 = all)
        resume: Resume from checkpoint
        batch_size: Notes per LLM call (1 for quality)
    """
    logger.info("=" * 60)
    logger.info("QWEN-ENHANCED CONCEPT EXTRACTION")
    logger.info("=" * 60)

    # Check Ollama
    if not ollama_available():
        logger.error("Ollama is not running. Start with: ollama serve")
        return

    # Load data
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    vocab_path = TAXONOMY_DIR / "lexicon_v1.json"

    if not notes_path.exists():
        logger.error(f"Notes not found: {notes_path}. Run pipeline first.")
        return

    notes_df = pd.read_parquet(notes_path)
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))

    # Re-run rule-based detection fresh (not from file, which may be stale)
    logger.info("Running fresh rule-based detection...")
    from server.extract.detect_concepts import (
        load_lexicon, build_alias_to_concept_map, find_matches_in_text,
        build_aho_automaton
    )
    from server.shared.config import MAX_ALIAS_MATCHES_PER_NOTE

    lexicon_df = load_lexicon()
    alias_map = build_alias_to_concept_map(lexicon_df)
    automaton, patterns = build_aho_automaton(alias_map)

    rule_concepts_all = []
    for _, row in notes_df.iterrows():
        matches = find_matches_in_text(
            row['text'], alias_map, MAX_ALIAS_MATCHES_PER_NOTE,
            automaton=automaton, patterns=patterns
        )
        for m in matches:
            m['note_id'] = row['note_id']
            m['detection_method'] = 'rule-based'
            m['confidence'] = 1.0
        rule_concepts_all.extend(matches)

    rule_concepts_df = pd.DataFrame(rule_concepts_all) if rule_concepts_all else pd.DataFrame()

    logger.info(f"Loaded {len(notes_df)} notes")
    logger.info(f"Loaded {len(vocab)} concepts in taxonomy")
    logger.info(f"Model: {model}")
    logger.info(f"Fresh rule-based matches: {len(rule_concepts_df)}")

    # Checkpoint
    ckpt = CheckpointManager()
    if not resume:
        ckpt.reset()

    # Process notes
    note_ids = notes_df['note_id'].tolist()
    if limit > 0:
        note_ids = note_ids[:limit]

    total = len(note_ids)
    remaining = [nid for nid in note_ids if not ckpt.is_done(nid)]
    logger.info(f"Notes to process: {len(remaining)}/{total} (resume={resume})")

    all_enhanced = []
    stats = {"processed": 0, "failed": 0, "new_concepts": 0, "total_llm_concepts": 0}
    start_time = time.time()

    for i, note_id in enumerate(remaining):
        note_row = notes_df[notes_df['note_id'] == note_id].iloc[0]
        text = note_row['text']

        # Get existing rule-based concepts for this note
        if len(rule_concepts_df) > 0:
            note_rules = rule_concepts_df[rule_concepts_df['note_id'] == note_id]
            rule_labels = note_rules['matched_alias'].tolist() if len(note_rules) > 0 else []
        else:
            note_rules = pd.DataFrame()
            rule_labels = []

        # Enhance with Qwen
        elapsed = time.time() - start_time
        rate = (i + 1) / max(elapsed, 0.1)
        eta = (len(remaining) - i - 1) / max(rate, 0.001)

        logger.info(f"[{i+1}/{len(remaining)}] {note_id} ({len(text)} chars, {len(rule_labels)} rules) "
                     f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

        result = enhance_note(model, note_id, text, rule_labels)

        if result:
            # Merge rule-based + LLM
            merged = merge_with_rule_based(note_id, note_rules, result, vocab)
            llm_only = [m for m in merged if m["detection_method"] == "llm-enhanced"]

            all_enhanced.extend(merged)
            stats["processed"] += 1
            stats["total_llm_concepts"] += len(llm_only)

            # Save to checkpoint
            ckpt.save_result(note_id, {
                "concepts": result.get("concepts", []),
                "summary": result.get("client_summary", ""),
                "urgency": result.get("urgency", ""),
                "sentiment": result.get("sentiment", ""),
                "llm_concepts_count": len(llm_only),
                "total_concepts_count": len(merged),
            })

            logger.info(f"  → {len(merged)} total ({len(rule_labels)} rule + {len(llm_only)} LLM-new)")
        else:
            stats["failed"] += 1
            # Still include rule-based matches
            for _, row in note_rules.iterrows():
                all_enhanced.append(row.to_dict())
            ckpt.save_result(note_id, {"error": True})

    # ── Save Enhanced Concepts ───────────────────────────────────────────
    # First, always save the clean rule-based baseline
    if len(rule_concepts_df) > 0:
        rule_backup_path = DATA_OUTPUTS / "note_concepts_rules_only.csv"
        rule_concepts_df.to_csv(rule_backup_path, index=False)
        logger.info(f"✅ Saved rule-based baseline to {rule_backup_path}")

    if all_enhanced:
        enhanced_df = pd.DataFrame(all_enhanced)

        # For notes NOT processed by LLM (e.g. limit < total), add their rule-based concepts
        processed_note_ids = set(enhanced_df['note_id'].unique())
        if len(rule_concepts_df) > 0:
            unprocessed_rules = rule_concepts_df[~rule_concepts_df['note_id'].isin(processed_note_ids)]
            if len(unprocessed_rules) > 0:
                enhanced_df = pd.concat([enhanced_df, unprocessed_rules], ignore_index=True)
                logger.info(f"  Added {len(unprocessed_rules)} rule-based matches for {unprocessed_rules['note_id'].nunique()} unprocessed notes")

        # Save
        output_path = DATA_OUTPUTS / "note_concepts_enhanced.csv"
        enhanced_df.to_csv(output_path, index=False)
        logger.info(f"\n✅ Saved enhanced concepts to {output_path}")

        # Also overwrite the main concepts file for pipeline compatibility
        main_path = DATA_OUTPUTS / "note_concepts.csv"
        enhanced_df.to_csv(main_path, index=False)
        logger.info(f"✅ Updated main concepts file: {main_path}")

    # ── Save Client Summaries ────────────────────────────────────────────
    summaries = {}
    for note_id, res in ckpt.get_results().items():
        if isinstance(res, dict) and not res.get("error"):
            summaries[note_id] = {
                "summary": res.get("summary", ""),
                "urgency": res.get("urgency", ""),
                "sentiment": res.get("sentiment", ""),
            }

    if summaries:
        summaries_path = DATA_OUTPUTS / "client_summaries.json"
        with open(summaries_path, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved client summaries to {summaries_path}")

    # ── Final Report ─────────────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCEMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"Notes processed: {stats['processed']}/{len(remaining)}")
    logger.info(f"Failed: {stats['failed']}")

    if all_enhanced:
        edf = pd.DataFrame(all_enhanced)
        rule_count = len(edf[edf['detection_method'] == 'rule-based'])
        llm_count = len(edf[edf['detection_method'] == 'llm-enhanced'])
        logger.info(f"Total concepts: {len(edf)}")
        logger.info(f"  Rule-based: {rule_count} (confidence=1.0)")
        logger.info(f"  LLM-enhanced: {llm_count}")
        if llm_count > 0:
            avg_conf = edf[edf['detection_method'] == 'llm-enhanced']['confidence'].mean()
            logger.info(f"  LLM avg confidence: {avg_conf:.2f}")
        logger.info(f"Unique concepts: {edf['concept_id'].nunique()}")
        logger.info(f"Avg concepts/note: {len(edf) / max(1, edf['note_id'].nunique()):.1f}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen-Enhanced Concept Extraction")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--limit", type=int, default=0, help="Max notes (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    run_enhancement(
        model=args.model,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
