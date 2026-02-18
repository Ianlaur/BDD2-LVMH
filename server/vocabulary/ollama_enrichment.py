"""
Ollama-Powered Overnight Vocabulary Enrichment
================================================

This script connects to a local Ollama LLM to massively expand the concept lexicon.
It analyzes real client notes and generates:
  1. New concepts with multilingual aliases (12 languages)
  2. Proper bucket classification (preferences, intent, lifestyle, occasion, constraints, next_action)
  3. Reclassification of the 96 "other" concepts into proper buckets
  4. Domain-specific LVMH luxury vocabulary

Designed to run overnight with checkpointing so it can resume if interrupted.

Usage:
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --resume
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase reclassify
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase all
"""
import json
import time
import hashlib
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

import requests
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).resolve().parent.parent.parent / "ollama_enrichment.log")
    ]
)
logger = logging.getLogger("ollama_enrichment")

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TAXONOMY_DIR = BASE_DIR / "taxonomy"
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "data" / "ollama_checkpoints"

OLLAMA_URL = "http://localhost:11434"

# ── Bucket definitions ───────────────────────────────────────────────────
BUCKETS = {
    "preferences": "Client material, style, color, finish, product-type preferences (e.g., exotic leather, rose gold, minimalist design, monogram)",
    "intent": "Purchase intent or interest signals (e.g., looking for, interested in, wants to buy, requested viewing)",
    "lifestyle": "Client lifestyle markers and hobbies (e.g., art collector, yacht owner, equestrian, wine connoisseur, philanthropist)",
    "occasion": "Life events, celebrations, gift occasions (e.g., birthday, wedding, anniversary, graduation, gala, holiday, retirement, baby shower)",
    "constraints": "Dietary, health, accessibility, religious, or scheduling constraints (e.g., vegan, halal, wheelchair access, kosher, allergy, gluten-free)",
    "next_action": "Follow-up actions the sales associate should take (e.g., schedule callback, send catalog, arrange private viewing, follow up after event)",
}

LANGUAGES = {
    "FR": "French", "EN": "English", "IT": "Italian", "ES": "Spanish",
    "DE": "German", "PT": "Portuguese", "NL": "Dutch", "AR": "Arabic",
    "KO": "Korean", "ZH": "Chinese", "RU": "Russian", "JA": "Japanese"
}

# ── Ollama Client ────────────────────────────────────────────────────────

def ollama_chat(model: str, messages: List[Dict], temperature: float = 0.3, 
                format_schema = None) -> Optional[str]:
    """Send a chat completion request to Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
            "num_predict": 4096,
        }
    }
    if format_schema is not None:
        payload["format"] = format_schema
    
    for attempt in range(3):
        try:
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}/3 (15min limit), retrying...")
            time.sleep(10)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running? (brew services start ollama)")
            return None
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/3 failed: {e}")
            time.sleep(5)
    
    logger.error("All 3 attempts failed")
    return None


def ollama_generate_json(model: str, prompt: str, system: str, 
                         schema: Dict, temperature: float = 0.2) -> Optional[Dict]:
    """Generate a structured JSON response from Ollama with robust parsing."""
    messages = [
        {"role": "system", "content": system + "\n\nIMPORTANT: Return ONLY valid, complete JSON. No truncation."},
        {"role": "user", "content": prompt}
    ]
    
    # Try with structured output first
    raw = ollama_chat(model, messages, temperature=temperature, format_schema=schema)
    if not raw:
        return None
    
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON object
    try:
        start = raw.index('{')
        end = raw.rindex('}') + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        pass
    
    # If structured output failed, retry with simple format:"json" 
    logger.info("  Retrying with simple JSON mode...")
    raw2 = ollama_chat(model, messages, temperature=temperature, format_schema="json")
    if raw2:
        try:
            return json.loads(raw2)
        except json.JSONDecodeError:
            try:
                start = raw2.index('{')
                end = raw2.rindex('}') + 1
                return json.loads(raw2[start:end])
            except (ValueError, json.JSONDecodeError):
                pass
    
    # Last resort: try to fix truncated JSON by closing brackets
    for attempt_raw in [raw, raw2] if raw2 else [raw]:
        if attempt_raw:
            fixed = _fix_truncated_json(attempt_raw)
            if fixed:
                return fixed
    
    logger.warning(f"Could not parse JSON from response: {raw[:200]}...")
    return None


def _fix_truncated_json(raw: str) -> Optional[Dict]:
    """Try to fix truncated JSON by closing unclosed brackets/braces."""
    try:
        start = raw.index('{')
        text = raw[start:]
        
        # Count open/close brackets
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        # Try closing them
        text = text.rstrip().rstrip(',')
        text += ']' * max(0, open_brackets)
        text += '}' * max(0, open_braces)
        
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None


# ── Checkpoint System ────────────────────────────────────────────────────

class CheckpointManager:
    """Save/resume progress for overnight runs."""
    
    def __init__(self, phase: str):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CHECKPOINT_DIR / f"checkpoint_{phase}.json"
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.path.exists():
            with open(self.path, 'r') as f:
                return json.load(f)
        return {"completed": [], "results": {}, "started_at": datetime.now().isoformat()}
    
    def is_done(self, key: str) -> bool:
        return key in self.data["completed"]
    
    def mark_done(self, key: str, result: Any = None):
        self.data["completed"].append(key)
        if result is not None:
            self.data["results"][key] = result
        self._save()
    
    def _save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def get_results(self) -> Dict:
        return self.data.get("results", {})


# ── Phase 1: Reclassify "other" concepts ────────────────────────────────

RECLASSIFY_SYSTEM = """You are an expert luxury retail taxonomy specialist for LVMH (Louis Vuitton, Moët Hennessy, Dior, Bulgari, Tiffany, etc.).

Your task is to reclassify concepts that are currently in the "other" bucket into the correct taxonomy bucket.

The buckets are:
- preferences: Client material, style, color, finish, product-type preferences
- intent: Purchase intent or interest signals  
- lifestyle: Client lifestyle markers, professions, hobbies
- occasion: Life events, celebrations, gift occasions
- constraints: Dietary, health, accessibility, religious constraints
- next_action: Follow-up actions for sales associates
- other: ONLY if it truly doesn't fit any bucket above

Be generous with classification — most concepts SHOULD be in a specific bucket, not "other"."""

RECLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "bucket": {
            "type": "string",
            "enum": ["preferences", "intent", "lifestyle", "occasion", "constraints", "next_action", "other"]
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"}
    },
    "required": ["bucket", "confidence", "reasoning"]
}


def phase_reclassify(model: str, vocab: Dict) -> Dict:
    """Reclassify concepts currently in 'other' bucket."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Reclassifying 'other' concepts")
    logger.info("=" * 70)
    
    ckpt = CheckpointManager("reclassify")
    
    other_concepts = {
        cid: data for cid, data in vocab.items()
        if data.get("rule") == "bucket=other" or data.get("bucket") == "other"
    }
    
    logger.info(f"Found {len(other_concepts)} concepts in 'other' bucket")
    reclassified = 0
    
    bucket_descriptions = "\n".join(f"- {b}: {d}" for b, d in BUCKETS.items())
    
    for i, (cid, data) in enumerate(other_concepts.items()):
        if ckpt.is_done(cid):
            prev = ckpt.get_results().get(cid, {})
            if prev.get("bucket") != "other":
                reclassified += 1
            continue
        
        label = data.get("label", cid)
        aliases = data.get("aliases", [])
        examples_str = ", ".join(aliases[:10])
        
        prompt = f"""Classify this luxury retail concept into the correct bucket.

Concept: "{label}"
Aliases: {examples_str}
Languages: {data.get('languages', 'unknown')}

Available buckets:
{bucket_descriptions}

Which bucket does this concept belong to?"""
        
        result = ollama_generate_json(model, prompt, RECLASSIFY_SYSTEM, RECLASSIFY_SCHEMA)
        
        if result and result.get("bucket") and result.get("confidence", 0) >= 0.5:
            new_bucket = result["bucket"]
            ckpt.mark_done(cid, result)
            
            if new_bucket != "other":
                reclassified += 1
                logger.info(f"  [{i+1}/{len(other_concepts)}] {label}: other → {new_bucket} "
                          f"(confidence={result.get('confidence', '?')})")
            else:
                logger.info(f"  [{i+1}/{len(other_concepts)}] {label}: stays in 'other'")
        else:
            ckpt.mark_done(cid, {"bucket": "other", "confidence": 0, "reasoning": "LLM could not classify"})
            logger.info(f"  [{i+1}/{len(other_concepts)}] {label}: skipped (no valid response)")
        
        # Small delay to avoid overwhelming Ollama
        time.sleep(0.5)
    
    logger.info(f"\nPhase 1 complete: {reclassified}/{len(other_concepts)} concepts reclassified")
    return ckpt.get_results()


# ── Phase 2: Expand aliases with multilingual translations ───────────────

ALIAS_SYSTEM = """You are a multilingual luxury retail vocabulary expert. You know all LVMH maisons (Louis Vuitton, Dior, Moët Hennessy, Bulgari, Tiffany, Fendi, Givenchy, Loewe, Celine, Berluti, Kenzo, etc.).

Your task is to generate multilingual aliases for luxury retail concepts. These aliases are used for keyword matching in client sales notes written by sales associates worldwide.

Important rules:
- Generate LOWERCASE aliases only
- Include natural phrases as they would appear in sales notes (e.g., "interested in watches" not just "watches")
- Include brand-relevant variations (e.g., for "monogram" → "monogramme personnalisé", "monogramma personalizzato")
- Include verb phrases in each language (e.g., "recherche", "cerca", "busca", "sucht")
- Each alias should be 1-4 words maximum
- Do NOT include proper names, client names, or specific dates"""

ALIAS_SCHEMA = {
    "type": "object",
    "properties": {
        "aliases": {
            "type": "object",
            "properties": {
                "FR": {"type": "array", "items": {"type": "string"}},
                "EN": {"type": "array", "items": {"type": "string"}},
                "IT": {"type": "array", "items": {"type": "string"}},
                "ES": {"type": "array", "items": {"type": "string"}},
                "DE": {"type": "array", "items": {"type": "string"}},
                "PT": {"type": "array", "items": {"type": "string"}},
                "NL": {"type": "array", "items": {"type": "string"}},
                "AR": {"type": "array", "items": {"type": "string"}},
                "KO": {"type": "array", "items": {"type": "string"}},
                "ZH": {"type": "array", "items": {"type": "string"}},
                "RU": {"type": "array", "items": {"type": "string"}},
                "JA": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "required": ["aliases"]
}


def phase_expand_aliases(model: str, vocab: Dict) -> Dict[str, List[str]]:
    """Expand aliases for existing concepts in all 12 languages."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Expanding multilingual aliases")
    logger.info("=" * 70)
    
    ckpt = CheckpointManager("aliases")
    new_aliases = {}
    
    # Process concepts that have < 15 aliases (most need more)
    targets = {
        cid: data for cid, data in vocab.items()
        if len(data.get("aliases", [])) < 30
    }
    
    logger.info(f"Expanding aliases for {len(targets)} concepts")
    
    for i, (cid, data) in enumerate(targets.items()):
        if ckpt.is_done(cid):
            cached = ckpt.get_results().get(cid, [])
            if cached:
                new_aliases[cid] = cached
            continue
        
        label = data.get("label", cid)
        existing = data.get("aliases", [])
        bucket = data.get("rule", "bucket=other").replace("bucket=", "")
        
        prompt = f"""Generate multilingual aliases for this luxury retail concept.

Concept: "{label}"
Bucket: {bucket} ({BUCKETS.get(bucket, 'general concept')})
Existing aliases (do NOT repeat these): {', '.join(existing[:15])}

Generate 3-6 NEW aliases per language. Focus on:
- How a sales associate would write about this in client notes
- Natural phrases (not just single words)
- Luxury-specific vocabulary
- Include verb forms ("recherche", "cerca", "seeks", "sucht")

Generate aliases for all 12 languages: FR, EN, IT, ES, DE, PT, NL, AR, KO, ZH, RU, JA."""
        
        result = ollama_generate_json(model, prompt, ALIAS_SYSTEM, ALIAS_SCHEMA)
        
        if result and "aliases" in result:
            all_new = []
            lang_codes = []
            for lang, aliases_list in result["aliases"].items():
                if isinstance(aliases_list, list):
                    cleaned = [a.strip().lower() for a in aliases_list if a.strip() and len(a.strip()) > 1]
                    # Remove any that duplicate existing
                    existing_lower = set(a.lower() for a in existing)
                    cleaned = [a for a in cleaned if a not in existing_lower]
                    all_new.extend(cleaned)
                    if cleaned:
                        lang_codes.append(lang)
            
            # Deduplicate
            all_new = list(dict.fromkeys(all_new))
            
            ckpt.mark_done(cid, all_new)
            new_aliases[cid] = all_new
            
            logger.info(f"  [{i+1}/{len(targets)}] {label}: +{len(all_new)} new aliases "
                       f"(langs: {','.join(lang_codes)})")
        else:
            ckpt.mark_done(cid, [])
            logger.info(f"  [{i+1}/{len(targets)}] {label}: no new aliases generated")
        
        time.sleep(0.5)
    
    total_new = sum(len(v) for v in new_aliases.values())
    logger.info(f"\nPhase 2 complete: {total_new} new aliases across {len(new_aliases)} concepts")
    return new_aliases


# ── Phase 3: Generate NEW concepts from notes ───────────────────────────

DISCOVERY_SYSTEM = """You are an expert luxury retail analyst for LVMH. You analyze client sales notes to discover hidden concepts, patterns, and vocabulary.

Your task is to identify NEW luxury retail concepts from client notes that are NOT already in the existing lexicon. Focus on:

1. **Occasion** concepts: birthdays, weddings, anniversaries, galas, holidays, graduations, retirements, baby showers, engagements, promotions, housewarming, Ramadan, Chinese New Year, Diwali, Christmas, Valentine's Day, Mother's Day, Father's Day
2. **Constraint** concepts: dietary (vegan, halal, kosher, gluten-free), health (wheelchair access, fragrance sensitivity), religious, scheduling, ethical (sustainable, eco-friendly, cruelty-free)
3. **Lifestyle** concepts: professions (CEO, diplomat, surgeon, architect), hobbies (golf, sailing, skiing, polo, opera), interests (contemporary art, vintage cars, wine, cigars, thoroughbred horses)
4. **Preference** concepts: materials (cashmere, silk, platinum, titanium), styles (minimalist, bohemian, classic, avant-garde), colors (navy, burgundy, champagne gold), product types (clutch, tote, trunk, tiara)
5. **Next action** concepts: follow-up signals (callback requested, send catalog, arrange fitting, book appointment)

Return concepts that are genuinely useful for a luxury retail CRM system.
Each concept must have a clear label, bucket classification, and multilingual aliases."""

DISCOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bucket": {
                        "type": "string",
                        "enum": ["preferences", "intent", "lifestyle", "occasion", "constraints", "next_action"]
                    },
                    "aliases": {
                        "type": "object",
                        "properties": {
                            "FR": {"type": "array", "items": {"type": "string"}},
                            "EN": {"type": "array", "items": {"type": "string"}},
                            "IT": {"type": "array", "items": {"type": "string"}},
                            "ES": {"type": "array", "items": {"type": "string"}},
                            "DE": {"type": "array", "items": {"type": "string"}},
                            "PT": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["label", "bucket", "aliases"]
            }
        }
    },
    "required": ["concepts"]
}


def phase_discover_from_notes(model: str, vocab: Dict) -> List[Dict]:
    """Analyze actual client notes to discover new concepts."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Discovering new concepts from client notes")
    logger.info("=" * 70)
    
    ckpt = CheckpointManager("discover_notes")
    
    # Load notes
    notes_path = DATA_DIR / "processed" / "notes_clean.parquet"
    if not notes_path.exists():
        logger.error(f"Notes file not found: {notes_path}")
        return []
    
    notes_df = pd.read_parquet(notes_path)
    
    # Get the note text column
    note_col = None
    for col in ["note_clean", "note", "notes", "SA Notes", "SA_Notes"]:
        if col in notes_df.columns:
            note_col = col
            break
    
    if not note_col:
        logger.error(f"No note column found. Columns: {list(notes_df.columns)}")
        return []
    
    notes = notes_df[note_col].dropna().tolist()
    logger.info(f"Loaded {len(notes)} client notes")
    
    # Get existing concept labels for deduplication
    existing_labels = set()
    existing_aliases = set()
    for cid, data in vocab.items():
        existing_labels.add(data.get("label", "").lower())
        for alias in data.get("aliases", []):
            existing_aliases.add(alias.lower())
    
    # Process notes in batches of 10
    batch_size = 10
    all_new_concepts = []
    
    for batch_idx in range(0, len(notes), batch_size):
        batch_key = f"batch_{batch_idx}"
        
        if ckpt.is_done(batch_key):
            cached = ckpt.get_results().get(batch_key, [])
            all_new_concepts.extend(cached)
            continue
        
        batch = notes[batch_idx:batch_idx + batch_size]
        notes_text = "\n---\n".join(f"Note {batch_idx + j + 1}: {n}" for j, n in enumerate(batch))
        
        # List some existing concepts so the LLM avoids duplicates
        existing_sample = list(existing_labels)[:50]
        
        prompt = f"""Analyze these {len(batch)} luxury retail client notes and identify NEW concepts not already in our lexicon.

CLIENT NOTES:
{notes_text}

EXISTING CONCEPTS (do NOT duplicate these):
{', '.join(existing_sample)}

Find 3-8 NEW concepts from these notes. Focus especially on:
- Occasions (birthdays, events, holidays)
- Constraints (dietary, health, scheduling)  
- Lifestyle markers (professions, hobbies)
- Specific preferences (materials, styles, products)
- Next actions (follow-up signals)

Each concept needs: label, bucket, and aliases in at least FR, EN, IT, ES, DE, PT."""
        
        result = ollama_generate_json(model, prompt, DISCOVERY_SYSTEM, DISCOVERY_SCHEMA, temperature=0.4)
        
        batch_concepts = []
        if result and "concepts" in result:
            for concept in result["concepts"]:
                label = concept.get("label", "").lower().strip()
                
                # Skip if it already exists
                if label in existing_labels:
                    continue
                
                # Flatten aliases
                flat_aliases = []
                aliases_dict = concept.get("aliases", {})
                if isinstance(aliases_dict, dict):
                    for lang, als in aliases_dict.items():
                        if isinstance(als, list):
                            flat_aliases.extend([a.strip().lower() for a in als if a.strip()])
                elif isinstance(aliases_dict, list):
                    flat_aliases = [a.strip().lower() for a in aliases_dict if a.strip()]
                
                # Skip if too much overlap with existing aliases
                overlap = len(set(flat_aliases) & existing_aliases)
                if overlap > len(flat_aliases) * 0.5 and len(flat_aliases) > 2:
                    continue
                
                # Add to results
                batch_concepts.append({
                    "label": label,
                    "bucket": concept.get("bucket", "other"),
                    "aliases": list(dict.fromkeys(flat_aliases)),  # dedupe
                    "reasoning": concept.get("reasoning", "")
                })
                
                # Track for dedup
                existing_labels.add(label)
                existing_aliases.update(flat_aliases)
        
        ckpt.mark_done(batch_key, batch_concepts)
        all_new_concepts.extend(batch_concepts)
        
        logger.info(f"  Batch {batch_idx//batch_size + 1}/{(len(notes) + batch_size - 1)//batch_size}: "
                    f"found {len(batch_concepts)} new concepts")
        
        time.sleep(1)
    
    logger.info(f"\nPhase 3 complete: {len(all_new_concepts)} new concepts discovered from notes")
    return all_new_concepts


# ── Phase 4: Generate domain knowledge concepts ─────────────────────────

DOMAIN_SYSTEM = """You are a luxury retail taxonomy expert specializing in LVMH brands and the ultra-high-net-worth client segment.

Generate new concepts for a luxury CRM taxonomy. These concepts should be realistic things that would appear in client notes written by sales associates at LVMH maisons worldwide.

Each concept needs multilingual aliases that match how sales associates actually write — short phrases, mixed languages, abbreviations common in luxury retail."""

DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "bucket": {
                        "type": "string",
                        "enum": ["preferences", "intent", "lifestyle", "occasion", "constraints", "next_action"]
                    },
                    "aliases": {
                        "type": "object",
                        "properties": {
                            "FR": {"type": "array", "items": {"type": "string"}},
                            "EN": {"type": "array", "items": {"type": "string"}},
                            "IT": {"type": "array", "items": {"type": "string"}},
                            "ES": {"type": "array", "items": {"type": "string"}},
                            "DE": {"type": "array", "items": {"type": "string"}},
                            "PT": {"type": "array", "items": {"type": "string"}},
                            "NL": {"type": "array", "items": {"type": "string"}},
                            "AR": {"type": "array", "items": {"type": "string"}},
                            "KO": {"type": "array", "items": {"type": "string"}},
                            "ZH": {"type": "array", "items": {"type": "string"}},
                            "RU": {"type": "array", "items": {"type": "string"}},
                            "JA": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "required": ["label", "bucket", "aliases"]
            }
        }
    },
    "required": ["concepts"]
}

# Targeted prompts for each empty/weak bucket
DOMAIN_PROMPTS = [
    # ── OCCASION (currently 0 concepts!) ──────────────────────────────
    {
        "key": "occasion_celebrations",
        "prompt": """Generate 15 luxury retail OCCASION concepts for celebrations and life events.

These are occasions when clients purchase luxury gifts or need special services:
- Religious/cultural: Ramadan, Eid, Diwali, Chinese New Year, Hanukkah, Christmas, Easter
- Life events: wedding, engagement, birth/baby, graduation, retirement, promotion
- Personal: birthday, anniversary, Valentine's Day, Mother's/Father's Day
- Social: gala, charity event, awards ceremony, debutante ball

For each, provide aliases in FR, EN, IT, ES, DE, PT, NL, AR, KO, ZH, RU, JA as they would appear in sales associate notes."""
    },
    {
        "key": "occasion_seasonal",
        "prompt": """Generate 10 luxury retail OCCASION concepts for seasonal and travel events.

- Seasonal: summer collection launch, winter holidays, spring fashion week, fall preview
- Travel: honeymoon, world cruise, ski season, summer villa, yacht week
- Business: corporate gifting season, end-of-year gifts, client appreciation

For each, provide aliases in FR, EN, IT, ES, DE, PT, NL as they would appear in sales associate notes."""
    },
    
    # ── CONSTRAINTS (currently 0 concepts!) ───────────────────────────
    {
        "key": "constraints_dietary",
        "prompt": """Generate 10 luxury retail CONSTRAINT concepts for dietary and health restrictions.

These appear when planning events, gifting food/wine, or noting client sensitivities:
- Dietary: vegan, vegetarian, halal, kosher, gluten-free, lactose intolerant, nut allergy
- Health: fragrance sensitivity, mobility issues, visual impairment
- Religious: no alcohol, fasting period, modest dress

For each, provide aliases in FR, EN, IT, ES, DE, PT, NL, AR as they would appear in sales notes."""
    },
    {
        "key": "constraints_ethical",
        "prompt": """Generate 8 luxury retail CONSTRAINT concepts for ethical and sustainability preferences.

Modern luxury clients increasingly have:
- Sustainability: eco-friendly, sustainable materials, carbon neutral
- Animal welfare: cruelty-free, no fur, no exotic skins
- Sourcing: fair trade, conflict-free diamonds, ethically sourced
- Packaging: minimal packaging, recyclable materials

For each, provide aliases in FR, EN, IT, ES, DE, PT, NL as they would appear in sales notes."""
    },
    
    # ── LIFESTYLE (currently only 5 concepts) ─────────────────────────
    {
        "key": "lifestyle_professions",
        "prompt": """Generate 12 luxury retail LIFESTYLE concepts for client professions/profiles.

High-net-worth client professions seen in LVMH sales notes:
- Business: CEO, entrepreneur, hedge fund manager, venture capitalist, family office
- Creative: architect, fashion designer, film director, gallery owner
- Professional: surgeon, diplomat, ambassador, attorney, judge
- Royalty/nobility: royal family, aristocrat, socialite
- Sports: Formula 1, polo player, thoroughbred owner

For each, provide aliases in FR, EN, IT, ES, DE, PT as they would appear in sales notes."""
    },
    {
        "key": "lifestyle_hobbies",
        "prompt": """Generate 12 luxury retail LIFESTYLE concepts for client hobbies and interests.

Hobbies/interests of ultra-high-net-worth clients:
- Sports: sailing/yacht, equestrian, golf, skiing, tennis, polo
- Culture: opera, contemporary art, antiques, rare books
- Leisure: wine collecting, cigar aficionado, Michelin dining, private aviation
- Collecting: watch collecting, car collecting, jewelry collecting

For each, provide aliases in FR, EN, IT, ES, DE, PT, NL as they would appear in sales notes."""
    },
    
    # ── PREFERENCES (currently only 5 concepts) ──────────────────────
    {
        "key": "preferences_materials",
        "prompt": """Generate 12 luxury retail PREFERENCE concepts for materials and finishes.

Materials and finishes mentioned in LVMH client notes:
- Leathers: crocodile, ostrich, python, lambskin, calfskin, patent leather
- Metals: rose gold, white gold, platinum, titanium, palladium
- Stones: diamond, emerald, sapphire, ruby, jade, mother of pearl
- Fabrics: cashmere, silk, tweed, organza, lace
- Finishes: matte, glossy, brushed, hammered, engraved

For each, provide aliases in FR, EN, IT, ES, DE, PT as they would appear in sales notes."""
    },
    {
        "key": "preferences_styles",
        "prompt": """Generate 10 luxury retail PREFERENCE concepts for style preferences.

Client style preferences noted by LVMH sales associates:
- Classic/timeless, minimalist/modern, bold/statement, bohemian, art deco
- Color preferences: neutral tones, jewel tones, pastels, monochrome, colorful
- Size preferences: petite/small, oversized, custom/bespoke, made-to-measure
- Personalization: engraving, hot stamping, custom color, special order

For each, provide aliases in FR, EN, IT, ES, DE, PT as they would appear in sales notes."""
    },
    
    # ── NEXT_ACTION ───────────────────────────────────────────────────
    {
        "key": "next_action_followup",
        "prompt": """Generate 10 luxury retail NEXT_ACTION concepts for follow-up tasks.

Actions that sales associates note for CRM follow-up:
- Communication: schedule callback, send thank-you note, email product photos
- Service: arrange private viewing, book VIP fitting, reserve item
- Events: send event invitation, RSVP follow-up, post-event thank you
- Sales: send quote, prepare special order, arrange home delivery
- Relationship: birthday reminder, anniversary follow-up, welcome gift

For each, provide aliases in FR, EN, IT, ES, DE, PT as they would appear in sales notes."""
    },
    
    # ── INTENT (strengthen existing) ─────────────────────────────────
    {
        "key": "intent_purchase_signals",
        "prompt": """Generate 10 luxury retail INTENT concepts for purchase signals.

Strong purchase intent signals in client notes:
- Ready to buy, wants to order, confirmed purchase, deposit paid
- Comparing options, shortlisted items, requested price
- Gift shopping for spouse, looking for anniversary gift
- Upgrade from previous model, wants the latest version
- Waiting for restock, on the waitlist, pre-ordered

For each, provide aliases in FR, EN, IT, ES, DE, PT as they would appear in sales notes."""
    },
]


def phase_domain_knowledge(model: str, vocab: Dict) -> List[Dict]:
    """Generate domain-specific concepts for weak/empty buckets."""
    logger.info("=" * 70)
    logger.info("PHASE 4: Generating domain knowledge concepts")
    logger.info("=" * 70)
    
    ckpt = CheckpointManager("domain")
    
    existing_labels = set(data.get("label", "").lower() for data in vocab.values())
    all_concepts = []
    
    for task in DOMAIN_PROMPTS:
        key = task["key"]
        
        if ckpt.is_done(key):
            cached = ckpt.get_results().get(key, [])
            all_concepts.extend(cached)
            logger.info(f"  {key}: loaded {len(cached)} cached concepts")
            continue
        
        logger.info(f"  Generating: {key}...")
        
        result = ollama_generate_json(model, task["prompt"], DOMAIN_SYSTEM, DOMAIN_SCHEMA, temperature=0.3)
        
        batch_concepts = []
        if result and "concepts" in result:
            for concept in result["concepts"]:
                label = concept.get("label", "").lower().strip()
                
                if not label or label in existing_labels:
                    continue
                
                # Flatten aliases
                flat_aliases = []
                aliases_dict = concept.get("aliases", {})
                lang_codes = []
                if isinstance(aliases_dict, dict):
                    for lang, als in aliases_dict.items():
                        if isinstance(als, list):
                            cleaned = [a.strip().lower() for a in als if a.strip() and len(a.strip()) > 1]
                            flat_aliases.extend(cleaned)
                            if cleaned:
                                lang_codes.append(lang)
                
                # Always include the label itself
                if label not in flat_aliases:
                    flat_aliases.insert(0, label)
                
                batch_concepts.append({
                    "label": label,
                    "bucket": concept.get("bucket", "other"),
                    "aliases": list(dict.fromkeys(flat_aliases)),
                    "languages": "|".join(sorted(lang_codes)) if lang_codes else "ALL"
                })
                
                existing_labels.add(label)
        
        ckpt.mark_done(key, batch_concepts)
        all_concepts.extend(batch_concepts)
        logger.info(f"    → {len(batch_concepts)} new concepts")
        
        time.sleep(2)
    
    logger.info(f"\nPhase 4 complete: {len(all_concepts)} domain concepts generated")
    return all_concepts


# ── Phase 5: Deep alias enrichment for CJK/RTL languages ────────────────

CJK_SYSTEM = """You are a multilingual luxury retail vocabulary specialist fluent in Arabic, Korean, Chinese, Russian, and Japanese.

Generate authentic aliases for luxury retail concepts as they would be written by sales associates in these languages. 
- Arabic: Use common transliterations and Arabic script
- Korean: Use Hangul and common luxury terms
- Chinese: Use simplified Chinese characters
- Russian: Use Cyrillic script  
- Japanese: Use kanji/katakana mix as natural in luxury retail

Keep aliases SHORT (1-4 words) and lowercase."""

CJK_SCHEMA = {
    "type": "object",
    "properties": {
        "aliases": {
            "type": "object",
            "properties": {
                "AR": {"type": "array", "items": {"type": "string"}},
                "KO": {"type": "array", "items": {"type": "string"}},
                "ZH": {"type": "array", "items": {"type": "string"}},
                "RU": {"type": "array", "items": {"type": "string"}},
                "JA": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "required": ["aliases"]
}


def phase_cjk_enrichment(model: str, vocab: Dict) -> Dict[str, List[str]]:
    """Add CJK/Arabic/Russian aliases to concepts that lack them."""
    logger.info("=" * 70)
    logger.info("PHASE 5: CJK/Arabic/Russian alias enrichment")
    logger.info("=" * 70)
    
    ckpt = CheckpointManager("cjk")
    new_aliases = {}
    
    # Find concepts missing CJK/RTL coverage
    targets = {}
    for cid, data in vocab.items():
        langs = data.get("languages", "")
        has_cjk = any(l in langs for l in ["AR", "KO", "ZH", "RU", "JA"])
        if not has_cjk and len(data.get("aliases", [])) > 0:
            targets[cid] = data
    
    logger.info(f"Found {len(targets)} concepts needing CJK/Arabic/Russian aliases")
    
    # Process in batches of 5
    target_list = list(targets.items())
    batch_size = 5
    
    for batch_idx in range(0, len(target_list), batch_size):
        batch_key = f"cjk_batch_{batch_idx}"
        
        if ckpt.is_done(batch_key):
            cached = ckpt.get_results().get(batch_key, {})
            new_aliases.update(cached)
            continue
        
        batch = target_list[batch_idx:batch_idx + batch_size]
        
        concepts_desc = "\n".join(
            f"- {cid}: \"{data.get('label', cid)}\" (bucket={data.get('rule', '').replace('bucket=', '')}), "
            f"existing: {', '.join(data.get('aliases', [])[:5])}"
            for cid, data in batch
        )
        
        prompt = f"""Generate Arabic, Korean, Chinese, Russian, and Japanese aliases for these {len(batch)} luxury retail concepts:

{concepts_desc}

For EACH concept, provide 2-4 aliases in each of: AR, KO, ZH, RU, JA.
Return a JSON object where keys are the concept IDs listed above."""
        
        # Use a simpler format for batch processing
        messages = [
            {"role": "system", "content": CJK_SYSTEM},
            {"role": "user", "content": prompt}
        ]
        
        raw = ollama_chat(model, messages, temperature=0.2)
        
        batch_aliases = {}
        if raw:
            try:
                parsed = json.loads(raw)
                for cid, data in batch:
                    if cid in parsed:
                        aliases_data = parsed[cid]
                        flat = []
                        if isinstance(aliases_data, dict):
                            # Could be {aliases: {AR: [...], ...}} or {AR: [...], ...}
                            inner = aliases_data.get("aliases", aliases_data)
                            if isinstance(inner, dict):
                                for lang, als in inner.items():
                                    if isinstance(als, list):
                                        flat.extend([a.strip().lower() for a in als if a.strip()])
                        if flat:
                            batch_aliases[cid] = list(dict.fromkeys(flat))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"  Batch {batch_idx}: parse error: {e}")
        
        ckpt.mark_done(batch_key, batch_aliases)
        new_aliases.update(batch_aliases)
        
        total_in_batch = sum(len(v) for v in batch_aliases.values())
        logger.info(f"  Batch {batch_idx//batch_size + 1}: +{total_in_batch} CJK/RTL aliases")
        
        time.sleep(1)
    
    total = sum(len(v) for v in new_aliases.values())
    logger.info(f"\nPhase 5 complete: {total} CJK/Arabic/Russian aliases added")
    return new_aliases


# ── Apply all results to vocabulary ──────────────────────────────────────

def apply_results(vocab: Dict, 
                  reclassifications: Dict,
                  new_aliases: Dict[str, List[str]],
                  discovered_concepts: List[Dict],
                  domain_concepts: List[Dict],
                  cjk_aliases: Dict[str, List[str]]) -> Dict:
    """Merge all enrichment results into the vocabulary."""
    logger.info("=" * 70)
    logger.info("APPLYING RESULTS TO VOCABULARY")
    logger.info("=" * 70)
    
    enriched = json.loads(json.dumps(vocab))  # deep copy
    stats = defaultdict(int)
    
    # 1. Apply reclassifications
    for cid, result in reclassifications.items():
        if cid in enriched and isinstance(result, dict):
            new_bucket = result.get("bucket", "other")
            if new_bucket != "other" and result.get("confidence", 0) >= 0.5:
                enriched[cid]["rule"] = f"bucket={new_bucket}"
                stats["reclassified"] += 1
    
    # 2. Apply new aliases to existing concepts
    for cid, aliases in new_aliases.items():
        if cid in enriched:
            existing = set(enriched[cid].get("aliases", []))
            new = set(aliases) - existing
            enriched[cid]["aliases"] = list(existing | new)
            stats["aliases_added"] += len(new)
    
    # 3. Apply CJK aliases
    for cid, aliases in cjk_aliases.items():
        if cid in enriched:
            existing = set(enriched[cid].get("aliases", []))
            new = set(aliases) - existing
            enriched[cid]["aliases"] = list(existing | new)
            stats["cjk_aliases_added"] += len(new)
    
    # 4. Add discovered concepts from notes
    for concept in discovered_concepts:
        cid = _generate_concept_id(concept["label"], len(enriched))
        if cid not in enriched:
            enriched[cid] = {
                "label": concept["label"],
                "aliases": concept.get("aliases", [concept["label"]]),
                "languages": concept.get("languages", "ALL"),
                "freq_notes": 0,
                "examples": [],
                "rule": f"bucket={concept['bucket']}"
            }
            stats["discovered"] += 1
    
    # 5. Add domain knowledge concepts
    for concept in domain_concepts:
        cid = _generate_concept_id(concept["label"], len(enriched))
        if cid not in enriched:
            enriched[cid] = {
                "label": concept["label"],
                "aliases": concept.get("aliases", [concept["label"]]),
                "languages": concept.get("languages", "ALL"),
                "freq_notes": 0,
                "examples": [],
                "rule": f"bucket={concept['bucket']}"
            }
            stats["domain_added"] += 1
    
    # Print summary
    logger.info(f"  Concepts reclassified from 'other': {stats['reclassified']}")
    logger.info(f"  Multilingual aliases added: {stats['aliases_added']}")
    logger.info(f"  CJK/Arabic/Russian aliases added: {stats['cjk_aliases_added']}")
    logger.info(f"  New concepts from notes: {stats['discovered']}")
    logger.info(f"  Domain knowledge concepts: {stats['domain_added']}")
    logger.info(f"  Total concepts: {len(vocab)} → {len(enriched)}")
    
    # Bucket distribution
    buckets = defaultdict(int)
    total_aliases = 0
    for cid, data in enriched.items():
        bucket = data.get("rule", "bucket=other").replace("bucket=", "")
        buckets[bucket] += 1
        total_aliases += len(data.get("aliases", []))
    
    logger.info(f"\n  Final bucket distribution:")
    for b in ["preferences", "intent", "lifestyle", "occasion", "constraints", "next_action", "other"]:
        logger.info(f"    {b}: {buckets.get(b, 0)}")
    logger.info(f"  Total aliases: {total_aliases}")
    
    return enriched


def _generate_concept_id(label: str, current_count: int) -> str:
    """Generate a unique concept ID."""
    h = hashlib.md5(label.encode()).hexdigest()[:6]
    idx = current_count
    return f"CONCEPT_{idx:04d}_{h}"


# ── Write enriched vocabulary ────────────────────────────────────────────

def write_enriched_vocab(enriched: Dict):
    """Write the enriched vocabulary to both JSON and CSV formats."""
    
    # Backup originals
    json_path = TAXONOMY_DIR / "lexicon_v1.json"
    csv_path = TAXONOMY_DIR / "lexicon_v1.csv"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if json_path.exists():
        backup = TAXONOMY_DIR / f"lexicon_v1_backup_{timestamp}.json"
        import shutil
        shutil.copy2(json_path, backup)
        logger.info(f"Backed up JSON to {backup}")
    
    if csv_path.exists():
        backup = TAXONOMY_DIR / f"lexicon_v1_backup_{timestamp}.csv"
        import shutil
        shutil.copy2(csv_path, backup)
        logger.info(f"Backed up CSV to {backup}")
    
    # Write JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(enriched)} concepts to {json_path}")
    
    # Write CSV
    rows = []
    for cid, data in enriched.items():
        aliases = data.get("aliases", [])
        examples = data.get("examples", [])
        rows.append({
            "concept_id": cid,
            "label": data.get("label", ""),
            "aliases": "|".join(aliases) if isinstance(aliases, list) else str(aliases),
            "languages": data.get("languages", "ALL"),
            "freq_notes": data.get("freq_notes", 0),
            "examples": "|".join(examples) if isinstance(examples, list) else str(examples),
            "rule": data.get("rule", "bucket=other")
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Wrote {len(df)} concepts to {csv_path}")
    
    # Also update taxonomy_v1.json
    taxonomy = defaultdict(list)
    for cid, data in enriched.items():
        bucket = data.get("rule", "bucket=other").replace("bucket=", "")
        taxonomy[bucket].append(cid)
    
    taxonomy_path = TAXONOMY_DIR / "taxonomy_v1.json"
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(dict(taxonomy), f, indent=2, ensure_ascii=False)
    logger.info(f"Updated taxonomy: {dict((k, len(v)) for k, v in taxonomy.items())}")
    
    # Write enrichment report
    report_path = BASE_DIR / "ENRICHMENT_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Vocabulary Enrichment Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total concepts | {len(enriched)} |\n")
        
        total_aliases = sum(len(d.get('aliases', [])) for d in enriched.values())
        f.write(f"| Total aliases | {total_aliases} |\n\n")
        
        f.write(f"## Bucket Distribution\n\n")
        f.write(f"| Bucket | Count |\n")
        f.write(f"|--------|-------|\n")
        for b in ["preferences", "intent", "lifestyle", "occasion", "constraints", "next_action", "other"]:
            f.write(f"| {b} | {len(taxonomy.get(b, []))} |\n")
    
    logger.info(f"Wrote enrichment report to {report_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ollama-powered overnight vocabulary enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full overnight run (all 5 phases):
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b

    # Resume after interruption:
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --resume

    # Run only specific phase:
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase reclassify
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase aliases
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase discover
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase domain
    python -m server.vocabulary.ollama_enrichment --model qwen2.5:14b --phase cjk
        """
    )
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--phase", default="all", 
                       choices=["all", "reclassify", "aliases", "discover", "domain", "cjk"],
                       help="Which phase to run")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from checkpoints (default: start fresh)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run but don't write final output")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("OLLAMA VOCABULARY ENRICHMENT")
    logger.info(f"Model: {args.model}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Check Ollama is running
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        logger.info(f"Ollama connected. Available models: {models}")
        
        if not any(args.model in m for m in models):
            logger.error(f"Model '{args.model}' not found. Available: {models}")
            logger.info(f"Pull it with: ollama pull {args.model}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Start it with: brew services start ollama")
        sys.exit(1)
    
    # Clean checkpoints if not resuming
    if not args.resume:
        if CHECKPOINT_DIR.exists():
            import shutil
            shutil.rmtree(CHECKPOINT_DIR)
            logger.info("Cleared previous checkpoints")
    
    # Load current vocabulary
    json_path = TAXONOMY_DIR / "lexicon_v1.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    logger.info(f"Loaded vocabulary: {len(vocab)} concepts")
    
    # Initialize result containers
    reclassifications = {}
    new_aliases = {}
    discovered_concepts = []
    domain_concepts = []
    cjk_aliases = {}
    
    # Run phases
    phases = ["reclassify", "aliases", "discover", "domain", "cjk"] if args.phase == "all" else [args.phase]
    
    for phase in phases:
        try:
            if phase == "reclassify":
                reclassifications = phase_reclassify(args.model, vocab)
            elif phase == "aliases":
                new_aliases = phase_expand_aliases(args.model, vocab)
            elif phase == "discover":
                discovered_concepts = phase_discover_from_notes(args.model, vocab)
            elif phase == "domain":
                domain_concepts = phase_domain_knowledge(args.model, vocab)
            elif phase == "cjk":
                cjk_aliases = phase_cjk_enrichment(args.model, vocab)
        except KeyboardInterrupt:
            logger.warning(f"\nInterrupted during phase '{phase}'. Progress saved to checkpoints.")
            logger.info("Resume with: python -m server.vocabulary.ollama_enrichment --resume")
            break
        except Exception as e:
            logger.error(f"Error in phase '{phase}': {e}")
            import traceback
            traceback.print_exc()
            logger.info("Continuing to next phase...")
    
    # Apply all results
    enriched = apply_results(vocab, reclassifications, new_aliases, 
                            discovered_concepts, domain_concepts, cjk_aliases)
    
    # Write output
    if not args.dry_run:
        write_enriched_vocab(enriched)
    else:
        logger.info("DRY RUN: not writing output files")
    
    # Summary
    elapsed = datetime.now() - start_time
    logger.info("=" * 70)
    logger.info(f"ENRICHMENT COMPLETE")
    logger.info(f"Duration: {elapsed}")
    logger.info(f"Concepts: {len(vocab)} → {len(enriched)} (+{len(enriched) - len(vocab)})")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review: python -m server.vocabulary.train_vocabulary stats")
    logger.info("  2. Retrain: python -m server.ml.cli train --size large --epochs 75")
    logger.info("  3. Re-run pipeline: python main.py")
    logger.info("  4. Sync to DB: python -m server.db.setup sync")


if __name__ == "__main__":
    main()
