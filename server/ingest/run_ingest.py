"""
Ingest stage: Load and validate CSV, produce notes_clean.parquet.

This module handles:
- Loading CSV from data/raw/
- Validating required columns
- Normalizing fields
- Outputting cleaned data to data/processed/notes_clean.parquet
"""
import sys
import re
from pathlib import Path
from typing import Optional, List
import pandas as pd

from server.shared.config import (
    DATA_RAW, DATA_PROCESSED, REQUIRED_COLUMNS, SUPPORTED_LANGUAGES,
    ENABLE_ANONYMIZATION, ANONYMIZATION_AGGRESSIVE, DATA_INPUT
)
from server.shared.utils import log_stage, ensure_directories, set_all_seeds


def find_input_csv(input_path: Optional[str] = None) -> Path:
    """
    Find the input CSV file.
    Priority:
    1. Explicit path if provided
    2. Single CSV in data/raw/
    3. Fallback to data/*.csv if data/raw is empty
    """
    if input_path:
        p = Path(input_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    
    # Check data/raw/
    raw_csvs = list(DATA_RAW.glob("*.csv"))
    if raw_csvs:
        if len(raw_csvs) == 1:
            return raw_csvs[0]
        raise ValueError(f"Multiple CSVs found in {DATA_RAW}. Please specify one.")
    
    # Fallback: check data/ directory (for backward compatibility)
    data_dir = DATA_RAW.parent
    data_csvs = list(data_dir.glob("*.csv"))
    if data_csvs:
        if len(data_csvs) == 1:
            return data_csvs[0]
        # Take first one alphabetically for determinism
        return sorted(data_csvs)[0]
    
    raise FileNotFoundError(
        f"No CSV files found in {DATA_RAW} or {data_dir}. "
        "Please place your input CSV in data/raw/"
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )


def normalize_language(lang: str) -> str:
    """Normalize language code to uppercase 2-letter code."""
    lang = str(lang).strip().upper()
    # Handle common variations
    lang_map = {
        "FRENCH": "FR", "FRANÇAIS": "FR", "FRANCAIS": "FR",
        "ENGLISH": "EN", "ANGLAIS": "EN",
        "ITALIAN": "IT", "ITALIANO": "IT", "ITALIEN": "IT",
        "SPANISH": "ES", "ESPAÑOL": "ES", "ESPAGNOL": "ES",
        "GERMAN": "DE", "DEUTSCH": "DE", "ALLEMAND": "DE",
    }
    return lang_map.get(lang, lang[:2].upper() if len(lang) >= 2 else lang)


def parse_date(date_str: str) -> pd.Timestamp:
    """Parse date string to datetime."""
    try:
        return pd.to_datetime(date_str)
    except Exception:
        return pd.NaT


def clean_text(text: str) -> str:
    """Clean transcription text: normalize whitespace, preserve content."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def anonymize_transcription(text: str) -> str:
    """
    Anonymize sensitive personal information in transcription text.
    Only runs if ENABLE_ANONYMIZATION is True in config.
    """
    if not ENABLE_ANONYMIZATION:
        return text
    
    try:
        from server.privacy import anonymize_text, AnonymizationConfig
        
        config = AnonymizationConfig(
            aggressive=ANONYMIZATION_AGGRESSIVE,
            placeholder_style="[TYPE]",  # Use descriptive placeholders
            # Article 9 sensitive categories (health, religion, etc.) are
            # detected for audit purposes but NOT redacted from the text
            # because many (e.g. "allergie") are legitimate business concepts.
            redact_article9=True,
            article9_mode="log",  # detect-only, no text modification
        )
        
        return anonymize_text(text, config)
    except Exception as e:
        log_stage("ingest", f"Warning: Anonymization failed: {e}")
        return text  # Return original text if anonymization fails


def run_ingest(input_path: Optional[str] = None) -> pd.DataFrame:
    """
    Main ingest function.
    
    Args:
        input_path: Optional path to input CSV. If not provided, auto-detects.
        
    Returns:
        Cleaned DataFrame
        
    Side effects:
        Writes data/processed/notes_clean.parquet
    """
    set_all_seeds()
    ensure_directories()
    
    log_stage("ingest", "Starting ingestion...")
    
    # Find and load CSV
    csv_path = find_input_csv(input_path)
    log_stage("ingest", f"Loading CSV: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV: {e}")
    
    if df.empty:
        raise ValueError(f"Input CSV is empty: {csv_path}")
    
    log_stage("ingest", f"Loaded {len(df)} rows")
    
    # Validate columns
    validate_columns(df)
    log_stage("ingest", "Column validation passed")
    
    # Create normalized dataframe
    notes_df = pd.DataFrame()
    
    # note_id and client_id (MVP: both = ID)
    notes_df["note_id"] = df["ID"].astype(str)
    notes_df["client_id"] = df["ID"].astype(str)  # MVP: treat ID as client_id
    
    # Date normalization
    notes_df["date"] = df["Date"].apply(parse_date)
    invalid_dates = notes_df["date"].isna().sum()
    if invalid_dates > 0:
        log_stage("ingest", f"Warning: {invalid_dates} rows with unparseable dates")
    
    # Language normalization
    notes_df["language"] = df["Language"].apply(normalize_language)
    unknown_langs = notes_df[~notes_df["language"].isin(SUPPORTED_LANGUAGES)]["language"].unique()
    if len(unknown_langs) > 0:
        log_stage("ingest", f"Warning: Unknown language codes: {list(unknown_langs)}")
    
    # Duration and Length (keep as-is, categorical)
    notes_df["duration"] = df["Duration"].astype(str)
    notes_df["length"] = df["Length"].astype(str)
    
    # Text cleaning and anonymization
    log_stage("ingest", f"Anonymization: {'ENABLED' if ENABLE_ANONYMIZATION else 'DISABLED'}")
    notes_df["text"] = df["Transcription"].apply(clean_text).apply(anonymize_transcription)
    empty_texts = (notes_df["text"] == "").sum()
    if empty_texts > 0:
        log_stage("ingest", f"Warning: {empty_texts} rows with empty transcription")
    
    # Sort by note_id for determinism
    notes_df = notes_df.sort_values("note_id").reset_index(drop=True)
    
    # Write output
    output_path = DATA_PROCESSED / "notes_clean.parquet"
    notes_df.to_parquet(output_path, index=False)
    log_stage("ingest", f"Wrote {len(notes_df)} notes to {output_path}")
    
    # Summary stats
    log_stage("ingest", f"Languages: {notes_df['language'].value_counts().to_dict()}")
    log_stage("ingest", "Ingestion complete!")
    
    return notes_df


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Ingest CSV data")
    parser.add_argument("--input", type=str, help="Path to input CSV")
    args = parser.parse_args()
    
    try:
        run_ingest(args.input)
    except Exception as e:
        log_stage("ingest", f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
