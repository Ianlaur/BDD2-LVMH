"""
Adaptive Pipeline - Automatically processes any CSV file structure.

This module enables the pipeline to work with any CSV by:
1. Analyzing the CSV structure to detect columns
2. Identifying text columns (for NLP processing)
3. Identifying ID columns (for client/note identification)
4. Mapping to the standard pipeline format

Usage:
    python -m server.adaptive.csv_analyzer data/input/any_file.csv --analyze-only
    python -m server.adaptive.csv_analyzer data/input/any_file.csv --text-column "Notes"
"""
import pandas as pd
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter

from server.shared.utils import log_stage


class CSVAnalyzer:
    """Analyzes CSV structure to detect column types and mappings."""
    
    # Patterns for detecting column types
    TEXT_PATTERNS = ['text', 'note', 'transcription', 'description', 'comment', 'message', 'content', 'body']
    ID_PATTERNS = ['id', 'client', 'customer', 'user', 'account', 'code', 'number', 'num', 'ref']
    DATE_PATTERNS = ['date', 'time', 'created', 'updated', 'timestamp', 'when']
    LANG_PATTERNS = ['lang', 'language', 'locale', 'country']
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = None
        self.analysis = {}
        
    def load(self) -> pd.DataFrame:
        """Load the CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        log_stage("analyzer", f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        return self.df
    
    def analyze(self) -> Dict[str, Any]:
        """Analyze CSV structure and detect column types."""
        if self.df is None:
            self.load()
        
        self.analysis = {
            'file': str(self.csv_path),
            'rows': len(self.df),
            'columns': list(self.df.columns),
            'detected_types': {},
            'recommendations': {}
        }
        
        for col in self.df.columns:
            col_type = self._detect_column_type(col)
            self.analysis['detected_types'][col] = col_type
        
        # Make recommendations
        self._make_recommendations()
        
        return self.analysis
    
    def _detect_column_type(self, column: str) -> Dict[str, Any]:
        """Detect the type and purpose of a column."""
        col_lower = column.lower()
        sample = self.df[column].dropna()
        
        result = {
            'dtype': str(self.df[column].dtype),
            'null_count': self.df[column].isna().sum(),
            'unique_count': self.df[column].nunique(),
            'purpose': 'unknown'
        }
        
        # Check for ID patterns
        if any(p in col_lower for p in self.ID_PATTERNS):
            result['purpose'] = 'identifier'
        
        # Check for date patterns
        elif any(p in col_lower for p in self.DATE_PATTERNS):
            result['purpose'] = 'date'
        
        # Check for language patterns
        elif any(p in col_lower for p in self.LANG_PATTERNS):
            result['purpose'] = 'language'
        
        # Check for text patterns
        elif any(p in col_lower for p in self.TEXT_PATTERNS):
            result['purpose'] = 'text'
        
        # Infer from content
        else:
            result['purpose'] = self._infer_from_content(sample)
        
        # Add sample values
        if len(sample) > 0:
            result['sample'] = sample.head(3).tolist()
            
            if result['dtype'] == 'object':
                avg_len = sample.astype(str).str.len().mean()
                result['avg_length'] = round(avg_len, 1)
        
        return result
    
    def _infer_from_content(self, sample: pd.Series) -> str:
        """Infer column purpose from content analysis."""
        if len(sample) == 0:
            return 'empty'
        
        # Check if it's likely text content (long strings)
        if sample.dtype == 'object':
            avg_len = sample.astype(str).str.len().mean()
            if avg_len > 100:
                return 'text'
            elif avg_len > 20:
                return 'short_text'
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(sample):
            return 'numeric'
        
        # Check if looks like dates
        try:
            pd.to_datetime(sample.head(10))
            return 'date'
        except:
            pass
        
        return 'categorical'
    
    def _make_recommendations(self):
        """Make recommendations for column mappings."""
        recommendations = {}
        
        # Find best text column
        text_candidates = []
        for col, info in self.analysis['detected_types'].items():
            if info['purpose'] in ['text', 'short_text']:
                score = info.get('avg_length', 0)
                if info['purpose'] == 'text':
                    score *= 2  # Prefer longer text
                text_candidates.append((col, score))
        
        if text_candidates:
            best_text = max(text_candidates, key=lambda x: x[1])
            recommendations['text_column'] = best_text[0]
        
        # Find best ID column
        id_candidates = []
        for col, info in self.analysis['detected_types'].items():
            if info['purpose'] == 'identifier':
                # Prefer columns with high uniqueness
                uniqueness = info['unique_count'] / self.analysis['rows']
                id_candidates.append((col, uniqueness))
        
        if id_candidates:
            best_id = max(id_candidates, key=lambda x: x[1])
            recommendations['id_column'] = best_id[0]
        else:
            # Use index if no ID column found
            recommendations['id_column'] = None
            recommendations['use_index_as_id'] = True
        
        # Find language column
        for col, info in self.analysis['detected_types'].items():
            if info['purpose'] == 'language':
                recommendations['language_column'] = col
                break
        
        # Find date column
        for col, info in self.analysis['detected_types'].items():
            if info['purpose'] == 'date':
                recommendations['date_column'] = col
                break
        
        self.analysis['recommendations'] = recommendations
    
    def print_report(self):
        """Print a human-readable analysis report."""
        if not self.analysis:
            self.analyze()
        
        print("\n" + "=" * 60)
        print("CSV STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"\nFile: {self.analysis['file']}")
        print(f"Rows: {self.analysis['rows']}")
        print(f"Columns: {len(self.analysis['columns'])}")
        
        print("\n" + "-" * 40)
        print("COLUMN DETAILS")
        print("-" * 40)
        
        for col in self.analysis['columns']:
            info = self.analysis['detected_types'][col]
            print(f"\n  {col}")
            print(f"    Type: {info['dtype']}")
            print(f"    Purpose: {info['purpose']}")
            print(f"    Nulls: {info['null_count']}")
            print(f"    Unique: {info['unique_count']}")
            if 'avg_length' in info:
                print(f"    Avg Length: {info['avg_length']}")
            if 'sample' in info:
                sample_str = str(info['sample'][0])[:50] if info['sample'] else 'N/A'
                print(f"    Sample: {sample_str}...")
        
        print("\n" + "-" * 40)
        print("RECOMMENDATIONS")
        print("-" * 40)
        
        recs = self.analysis['recommendations']
        if recs.get('text_column'):
            print(f"  Text column: {recs['text_column']}")
        else:
            print("  Text column: NOT FOUND (specify with --text-column)")
        
        if recs.get('id_column'):
            print(f"  ID column: {recs['id_column']}")
        elif recs.get('use_index_as_id'):
            print("  ID column: Will use row index")
        
        if recs.get('language_column'):
            print(f"  Language column: {recs['language_column']}")
        
        if recs.get('date_column'):
            print(f"  Date column: {recs['date_column']}")
        
        print("\n" + "=" * 60)


def run_adaptive_pipeline(csv_path: str, text_column: str = None, 
                          id_column: str = None, analyze_only: bool = False):
    """
    Run the pipeline with adaptive column detection.
    
    Args:
        csv_path: Path to CSV file
        text_column: Override for text column name
        id_column: Override for ID column name
        analyze_only: If True, only analyze without processing
    """
    total_start = time.time()
    timings = {}
    
    # Load and analyze CSV
    load_start = time.time()
    analyzer = CSVAnalyzer(csv_path)
    analysis = analyzer.analyze()
    timings['load_and_analyze'] = time.time() - load_start
    
    if analyze_only:
        analyzer.print_report()
        print(f"\n⏱️  Analysis time: {timings['load_and_analyze']:.2f}s")
        return analysis
    
    # Get column mappings
    recs = analysis['recommendations']
    
    text_col = text_column or recs.get('text_column')
    if not text_col:
        raise ValueError("No text column detected. Specify with --text-column")
    
    id_col = id_column or recs.get('id_column')
    lang_col = recs.get('language_column')
    date_col = recs.get('date_column')
    
    log_stage("adaptive", f"Using text column: {text_col}")
    log_stage("adaptive", f"Using ID column: {id_col or 'index'}")
    
    # Create standardized DataFrame
    df = analyzer.df.copy()
    
    # Map to standard schema
    standardized = pd.DataFrame()
    
    # ID
    if id_col:
        standardized['note_id'] = df[id_col].astype(str)
    else:
        standardized['note_id'] = [f"NOTE_{i:04d}" for i in range(len(df))]
    
    # Client ID (same as note_id in MVP)
    standardized['client_id'] = standardized['note_id']
    
    # Text
    standardized['text'] = df[text_col].fillna('')
    
    # Language
    if lang_col:
        standardized['language'] = df[lang_col].fillna('EN')
    else:
        standardized['language'] = 'EN'  # Default
    
    # Date
    if date_col:
        try:
            standardized['date'] = pd.to_datetime(df[date_col])
        except:
            standardized['date'] = pd.Timestamp.now()
    else:
        standardized['date'] = pd.Timestamp.now()
    
    log_stage("adaptive", f"Standardized {len(standardized)} rows")
    
    # Save to processed folder
    save_start = time.time()
    from server.shared.config import DATA_PROCESSED
    output_path = DATA_PROCESSED / "notes_clean.parquet"
    standardized.to_parquet(output_path)
    timings['save'] = time.time() - save_start
    log_stage("adaptive", f"Saved to {output_path}")
    
    # Print timing summary
    total_time = time.time() - total_start
    print("\n" + "=" * 50)
    print("⏱️  CSV PROCESSING TIMING")
    print("=" * 50)
    print(f"  Load & Analyze: {timings['load_and_analyze']:.2f}s")
    print(f"  Save Output:    {timings['save']:.2f}s")
    print(f"  ─────────────────────────")
    print(f"  Total:          {total_time:.2f}s")
    print(f"  Rows processed: {len(standardized)}")
    print(f"  Rate:           {len(standardized)/total_time:.0f} rows/sec")
    print("=" * 50 + "\n")
    
    return standardized


def main():
    parser = argparse.ArgumentParser(description='Analyze and process any CSV file')
    parser.add_argument('csv', help='Path to CSV file')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not process')
    parser.add_argument('--text-column', help='Override text column name')
    parser.add_argument('--id-column', help='Override ID column name')
    
    args = parser.parse_args()
    
    run_adaptive_pipeline(
        args.csv,
        text_column=args.text_column,
        id_column=args.id_column,
        analyze_only=args.analyze_only
    )


if __name__ == '__main__':
    main()
