"""
Action recommendation stage: Generate recommended actions based on playbooks.

This module:
- Loads playbooks.yml (creates default if missing)
- Matches clients to actions based on:
  - Profile type keywords
  - Detected concept matches
  - Bucket-based triggers
- Ranks actions by priority and evidence strength
- Outputs recommended_actions.csv

Output: data/outputs/recommended_actions.csv
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import yaml

from server.shared.config import (
    DATA_OUTPUTS, TAXONOMY_DIR, ACTIVATIONS_DIR, PLAYBOOKS_FILE
)
from server.shared.utils import log_stage, set_all_seeds


# Default playbooks if file doesn't exist
DEFAULT_PLAYBOOKS = {
    "version": "1.0",
    "actions": [
        {
            "action_id": "ACT_001",
            "title": "VIP Event Invitation",
            "channel": "Event",
            "priority": "High",
            "kpi": "Event attendance, subsequent purchase",
            "triggers": {
                "buckets": ["lifestyle", "preferences"],
                "keywords": ["vip", "excellent", "art", "collection", "travel"],
                "min_confidence": 0.5
            },
            "description": "Invite high-value clients to exclusive boutique events"
        },
        {
            "action_id": "ACT_002",
            "title": "New Collection Preview",
            "channel": "CRM",
            "priority": "High",
            "kpi": "Preview attendance, conversion rate",
            "triggers": {
                "buckets": ["intent", "preferences"],
                "keywords": ["new", "collection", "looking", "interested", "preview"],
                "min_confidence": 0.4
            },
            "description": "Personal invitation to preview new seasonal collections"
        },
        {
            "action_id": "ACT_003",
            "title": "Gift Occasion Follow-up",
            "channel": "Email",
            "priority": "High",
            "kpi": "Response rate, gift purchase",
            "triggers": {
                "buckets": ["occasion"],
                "keywords": ["birthday", "anniversary", "gift", "cadeau", "regalo", "geschenk"],
                "min_confidence": 0.3
            },
            "description": "Proactive outreach before known gift occasions"
        },
        {
            "action_id": "ACT_004",
            "title": "Dietary Preference Note",
            "channel": "Client Service",
            "priority": "Medium",
            "kpi": "Client satisfaction, event attendance",
            "triggers": {
                "buckets": ["constraints"],
                "keywords": ["vegan", "vegetarian", "allergy", "allergic", "intolerant", "diet"],
                "min_confidence": 0.3
            },
            "description": "Flag dietary preferences for upcoming event invitations"
        },
        {
            "action_id": "ACT_005",
            "title": "Travel Collection Recommendation",
            "channel": "CRM",
            "priority": "Medium",
            "kpi": "Travel accessory sales",
            "triggers": {
                "buckets": ["lifestyle"],
                "keywords": ["travel", "voyage", "viaggio", "viaje", "trip", "safari", "asia", "europe"],
                "min_confidence": 0.4
            },
            "description": "Recommend travel-friendly pieces based on travel mentions"
        },
        {
            "action_id": "ACT_006",
            "title": "Sport/Hobby Personalization",
            "channel": "Email",
            "priority": "Medium",
            "kpi": "Engagement rate, personalization score",
            "triggers": {
                "buckets": ["lifestyle"],
                "keywords": ["golf", "tennis", "yoga", "sailing", "equestrian", "ski"],
                "min_confidence": 0.4
            },
            "description": "Personalized recommendations based on sports/hobbies"
        },
        {
            "action_id": "ACT_007",
            "title": "Follow-up Appointment",
            "channel": "CRM",
            "priority": "High",
            "kpi": "Appointment conversion, purchase rate",
            "triggers": {
                "buckets": ["next_action", "intent"],
                "keywords": ["follow up", "rappeler", "call", "appointment", "rendez-vous", "next"],
                "min_confidence": 0.3
            },
            "description": "Schedule follow-up based on noted next actions"
        },
        {
            "action_id": "ACT_008",
            "title": "Multi-generational Opportunity",
            "channel": "Event",
            "priority": "Medium",
            "kpi": "Family engagement, cross-generational sales",
            "triggers": {
                "buckets": ["lifestyle", "occasion"],
                "keywords": ["family", "daughter", "son", "mother", "father", "figlia", "hijo", "fille", "fils"],
                "min_confidence": 0.4
            },
            "description": "Family-focused event invitations for multi-generational clients"
        },
        {
            "action_id": "ACT_009",
            "title": "Budget-Sensitive Presentation",
            "channel": "Client Service",
            "priority": "Low",
            "kpi": "Conversion within budget, satisfaction",
            "triggers": {
                "buckets": ["constraints"],
                "keywords": ["budget", "price", "prix", "prezzo", "precio"],
                "min_confidence": 0.3
            },
            "description": "Prepare curated selection within noted budget range"
        },
        {
            "action_id": "ACT_010",
            "title": "Referral Program Outreach",
            "channel": "Email",
            "priority": "Low",
            "kpi": "Referral rate, new client acquisition",
            "triggers": {
                "buckets": ["lifestyle"],
                "keywords": ["referred", "network", "friend", "colleague", "potential"],
                "min_confidence": 0.5
            },
            "description": "Engage high-network clients for referral opportunities"
        }
    ]
}


def ensure_playbooks_file() -> Path:
    """
    Ensure playbooks.yml exists. Create default if missing.
    Returns path to playbooks file.
    """
    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not PLAYBOOKS_FILE.exists():
        log_stage("actions", f"Creating default playbooks at {PLAYBOOKS_FILE}")
        with open(PLAYBOOKS_FILE, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_PLAYBOOKS, f, default_flow_style=False, allow_unicode=True)
    
    return PLAYBOOKS_FILE


def load_playbooks() -> Dict[str, Any]:
    """Load playbooks from YAML file."""
    playbooks_path = ensure_playbooks_file()
    
    with open(playbooks_path, "r", encoding="utf-8") as f:
        playbooks = yaml.safe_load(f)
    
    return playbooks


def get_client_concepts_and_buckets(
    client_id: str,
    note_concepts_df: pd.DataFrame,
    lexicon_df: pd.DataFrame
) -> tuple[Set[str], Set[str], List[str]]:
    """
    Get concepts and buckets for a client.
    Returns: (concept_ids set, buckets set, matched aliases list)
    """
    concept_ids = set()
    buckets = set()
    matched_aliases = []
    
    if note_concepts_df is None or len(note_concepts_df) == 0:
        return concept_ids, buckets, matched_aliases
    
    client_concepts = note_concepts_df[
        note_concepts_df["client_id"].astype(str) == str(client_id)
    ]
    
    for _, row in client_concepts.iterrows():
        cid = row["concept_id"]
        concept_ids.add(cid)
        matched_aliases.append(row.get("matched_alias", ""))
        
        # Get bucket from lexicon
        if lexicon_df is not None:
            concept_row = lexicon_df[lexicon_df["concept_id"] == cid]
            if len(concept_row) > 0:
                rule = concept_row.iloc[0].get("rule", "")
                if "bucket=" in str(rule):
                    bucket = str(rule).split("bucket=")[1].split()[0]
                    buckets.add(bucket)
    
    return concept_ids, buckets, matched_aliases


def match_action_to_client(
    action: Dict[str, Any],
    client_profile: Dict[str, Any],
    client_concepts: Set[str],
    client_buckets: Set[str],
    matched_aliases: List[str],
    notes_text: str
) -> Optional[Dict[str, Any]]:
    """
    Determine if an action applies to a client.
    Returns match info or None if no match.
    """
    triggers = action.get("triggers", {})
    
    trigger_buckets = set(triggers.get("buckets", []))
    trigger_keywords = triggers.get("keywords", [])
    min_confidence = triggers.get("min_confidence", 0.0)
    
    # Check confidence threshold
    client_confidence = client_profile.get("confidence", 0.0)
    if client_confidence < min_confidence:
        return None
    
    # Score matching
    score = 0
    matched_triggers = []
    
    # Bucket match
    bucket_matches = client_buckets & trigger_buckets
    if bucket_matches:
        score += len(bucket_matches) * 2
        matched_triggers.extend([f"bucket:{b}" for b in bucket_matches])
    
    # Keyword match in profile_type, top_concepts, or text
    profile_text = " ".join([
        str(client_profile.get("profile_type", "")),
        str(client_profile.get("top_concepts", "")),
        notes_text.lower()
    ])
    
    for kw in trigger_keywords:
        if kw.lower() in profile_text.lower():
            score += 1
            matched_triggers.append(f"keyword:{kw}")
    
    # Alias match
    for alias in matched_aliases:
        if alias.lower() in notes_text.lower():
            score += 0.5
    
    if score == 0:
        return None
    
    return {
        "score": score,
        "triggers": matched_triggers
    }


def recommend_actions() -> pd.DataFrame:
    """
    Main action recommendation function.
    
    Returns:
        DataFrame with recommended actions
        
    Side effects:
        Writes data/outputs/recommended_actions.csv
    """
    set_all_seeds()
    
    log_stage("actions", "Starting action recommendation...")
    
    # Load playbooks
    playbooks = load_playbooks()
    actions = playbooks.get("actions", [])
    log_stage("actions", f"Loaded {len(actions)} actions from playbooks")
    
    # Load client profiles
    profiles_path = DATA_OUTPUTS / "client_profiles.csv"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profiles not found: {profiles_path}. Run segmentation first.")
    
    profiles_df = pd.read_csv(profiles_path)
    profiles_df["client_id"] = profiles_df["client_id"].astype(str)
    log_stage("actions", f"Loaded {len(profiles_df)} client profiles")
    
    # Load note concepts
    note_concepts_df = None
    concepts_path = DATA_OUTPUTS / "note_concepts.csv"
    if concepts_path.exists():
        note_concepts_df = pd.read_csv(concepts_path)
        note_concepts_df["client_id"] = note_concepts_df["client_id"].astype(str)
    
    # Load lexicon
    lexicon_df = None
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.csv"
    if lexicon_path.exists():
        lexicon_df = pd.read_csv(lexicon_path)
    
    # Load notes for text matching
    from server.shared.config import DATA_PROCESSED
    notes_path = DATA_PROCESSED / "notes_clean.parquet"
    notes_df = pd.read_parquet(notes_path)
    notes_df["client_id"] = notes_df["client_id"].astype(str)
    client_texts = dict(zip(notes_df["client_id"], notes_df["text"]))
    
    # Priority order
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    
    # Generate recommendations
    recommendations = []
    
    for _, profile_row in profiles_df.iterrows():
        client_id = str(profile_row["client_id"])
        
        # Get client context
        client_concepts, client_buckets, matched_aliases = get_client_concepts_and_buckets(
            client_id, note_concepts_df, lexicon_df
        )
        
        client_text = client_texts.get(client_id, "")
        
        client_profile = {
            "client_id": client_id,
            "profile_type": profile_row.get("profile_type", ""),
            "top_concepts": profile_row.get("top_concepts", ""),
            "confidence": profile_row.get("confidence", 0.0)
        }
        
        # Match actions
        client_actions = []
        
        for action in actions:
            match = match_action_to_client(
                action,
                client_profile,
                client_concepts,
                client_buckets,
                matched_aliases,
                client_text
            )
            
            if match:
                priority = action.get("priority", "Low")
                client_actions.append({
                    "client_id": client_id,
                    "action_id": action.get("action_id", ""),
                    "title": action.get("title", ""),
                    "channel": action.get("channel", ""),
                    "priority": priority,
                    "kpi": action.get("kpi", ""),
                    "triggers": " | ".join(match["triggers"][:5]),  # Top 5 triggers
                    "rationale": action.get("description", ""),
                    "_priority_order": priority_order.get(priority, 3),
                    "_score": match["score"]
                })
        
        # Sort by priority then score
        client_actions.sort(key=lambda x: (x["_priority_order"], -x["_score"]))
        
        # Keep all matched actions (could limit to top N if needed)
        for rec in client_actions:
            del rec["_priority_order"]
            del rec["_score"]
            recommendations.append(rec)
    
    # Build output DataFrame
    if recommendations:
        recommendations_df = pd.DataFrame(recommendations)
        # Ensure column order
        recommendations_df = recommendations_df[[
            "client_id", "action_id", "title", "channel", 
            "priority", "kpi", "triggers", "rationale"
        ]]
    else:
        recommendations_df = pd.DataFrame(columns=[
            "client_id", "action_id", "title", "channel",
            "priority", "kpi", "triggers", "rationale"
        ])
    
    # Sort for determinism
    recommendations_df = recommendations_df.sort_values(
        ["client_id", "priority", "action_id"]
    ).reset_index(drop=True)
    
    # Write output
    output_path = DATA_OUTPUTS / "recommended_actions.csv"
    recommendations_df.to_csv(output_path, index=False)
    log_stage("actions", f"Wrote {len(recommendations_df)} recommendations to {output_path}")
    
    # Summary
    if len(recommendations_df) > 0:
        actions_per_client = recommendations_df.groupby("client_id").size()
        log_stage("actions", f"Avg actions per client: {actions_per_client.mean():.1f}")
        
        top_actions = recommendations_df["action_id"].value_counts().head(5)
        log_stage("actions", "Top recommended actions:")
        for action_id, count in top_actions.items():
            title = recommendations_df[recommendations_df["action_id"] == action_id]["title"].iloc[0]
            log_stage("actions", f"  {title}: {count} clients")
    
    log_stage("actions", "Action recommendation complete!")
    
    return recommendations_df


def main():
    """CLI entry point."""
    try:
        recommend_actions()
    except Exception as e:
        log_stage("actions", f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
