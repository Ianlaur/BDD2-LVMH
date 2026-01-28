"""
Vocabulary Training CLI - Manage and train the concept vocabulary.

Usage:
    python -m server.vocabulary.train_vocabulary stats
    python -m server.vocabulary.train_vocabulary add "term" "Label" "bucket" --aliases "alias1,alias2"
    python -m server.vocabulary.train_vocabulary list --bucket preferences
    python -m server.vocabulary.train_vocabulary import keywords.json
    python -m server.vocabulary.train_vocabulary export vocabulary.json
    python -m server.vocabulary.train_vocabulary remove "concept_id"
"""
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

from server.shared.config import TAXONOMY_DIR
from server.shared.utils import log_stage


def load_vocabulary() -> Dict[str, Any]:
    """Load the current vocabulary/lexicon."""
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.json"
    
    if not lexicon_path.exists():
        return {}
    
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_vocabulary(vocab: Dict[str, Any]):
    """Save vocabulary to lexicon file."""
    lexicon_path = TAXONOMY_DIR / "lexicon_v1.json"
    
    # Ensure directory exists
    TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(lexicon_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    log_stage("vocab", f"Saved {len(vocab)} concepts to {lexicon_path}")
    
    # Also update taxonomy
    update_taxonomy(vocab)


def update_taxonomy(vocab: Dict[str, Any]):
    """Update taxonomy_v1.json based on vocabulary buckets."""
    taxonomy = defaultdict(list)
    
    for concept_id, concept_data in vocab.items():
        bucket = concept_data.get('bucket', 'other')
        taxonomy[bucket].append(concept_id)
    
    taxonomy_path = TAXONOMY_DIR / "taxonomy_v1.json"
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        json.dump(dict(taxonomy), f, indent=2, ensure_ascii=False)
    
    log_stage("vocab", f"Updated taxonomy: {dict((k, len(v)) for k, v in taxonomy.items())}")


def generate_concept_id(term: str) -> str:
    """Generate a stable concept ID from a term."""
    slug = term.lower().replace(' ', '_').replace('-', '_')
    slug = ''.join(c for c in slug if c.isalnum() or c == '_')
    return slug[:30]  # Limit length


def cmd_stats(args):
    """Show vocabulary statistics."""
    vocab = load_vocabulary()
    
    if not vocab:
        print("No vocabulary found. Run the pipeline first or import a vocabulary.")
        return
    
    # Count by bucket
    buckets = defaultdict(int)
    languages = set()
    total_aliases = 0
    
    for concept_id, data in vocab.items():
        bucket = data.get('bucket', 'other')
        buckets[bucket] += 1
        
        aliases = data.get('aliases', [])
        total_aliases += len(aliases)
        
        lang = data.get('languages', '')
        if lang == 'ALL':
            languages.add('multilingual')
        elif lang:
            languages.update(lang.split('|'))
    
    print("\n" + "=" * 50)
    print("VOCABULARY STATISTICS")
    print("=" * 50)
    print(f"\nTotal concepts: {len(vocab)}")
    print(f"Total aliases: {total_aliases}")
    print(f"Languages: {len(languages)}")
    
    print("\nConcepts by bucket:")
    for bucket in ['preferences', 'intent', 'lifestyle', 'occasion', 'constraints', 'next_action', 'other']:
        count = buckets.get(bucket, 0)
        if count > 0:
            print(f"  {bucket}: {count}")
    
    print("\nLanguages detected:")
    for lang in sorted(languages):
        print(f"  - {lang}")


def cmd_list(args):
    """List concepts, optionally filtered by bucket."""
    vocab = load_vocabulary()
    
    if not vocab:
        print("No vocabulary found.")
        return
    
    # Filter by bucket if specified
    if args.bucket:
        filtered = {k: v for k, v in vocab.items() if v.get('bucket') == args.bucket}
        print(f"\nConcepts in bucket '{args.bucket}' ({len(filtered)}):")
    else:
        filtered = vocab
        print(f"\nAll concepts ({len(filtered)}):")
    
    # Sort by frequency
    sorted_concepts = sorted(
        filtered.items(),
        key=lambda x: x[1].get('freq_notes', 0),
        reverse=True
    )
    
    for concept_id, data in sorted_concepts[:args.limit]:
        label = data.get('label', concept_id)
        bucket = data.get('bucket', 'other')
        freq = data.get('freq_notes', 0)
        aliases_count = len(data.get('aliases', []))
        print(f"  [{bucket}] {concept_id}: {label} (freq={freq}, aliases={aliases_count})")


def cmd_add(args):
    """Add a new concept to the vocabulary."""
    vocab = load_vocabulary()
    
    # Generate concept ID
    concept_id = generate_concept_id(args.term)
    
    if concept_id in vocab:
        print(f"Concept '{concept_id}' already exists. Use 'update' to modify.")
        return
    
    # Parse aliases
    aliases = []
    if args.aliases:
        aliases = [a.strip() for a in args.aliases.split(',') if a.strip()]
    
    # Always include the original term as an alias
    if args.term not in aliases:
        aliases.insert(0, args.term)
    
    # Create concept
    vocab[concept_id] = {
        'label': args.label,
        'aliases': aliases,
        'languages': args.languages or 'ALL',
        'freq_notes': 0,
        'bucket': args.bucket
    }
    
    save_vocabulary(vocab)
    print(f"Added concept: {concept_id}")
    print(f"  Label: {args.label}")
    print(f"  Bucket: {args.bucket}")
    print(f"  Aliases: {aliases}")


def cmd_remove(args):
    """Remove a concept from the vocabulary."""
    vocab = load_vocabulary()
    
    if args.concept_id not in vocab:
        print(f"Concept '{args.concept_id}' not found.")
        return
    
    del vocab[args.concept_id]
    save_vocabulary(vocab)
    print(f"Removed concept: {args.concept_id}")


def cmd_import(args):
    """Import concepts from a JSON file."""
    import_path = Path(args.file)
    
    if not import_path.exists():
        print(f"File not found: {import_path}")
        return
    
    with open(import_path, 'r', encoding='utf-8') as f:
        import_data = json.load(f)
    
    vocab = load_vocabulary()
    added = 0
    updated = 0
    
    # Handle different import formats
    if isinstance(import_data, list):
        # List of concepts
        for item in import_data:
            concept_id = item.get('id') or generate_concept_id(item.get('term', item.get('label', '')))
            if concept_id in vocab:
                # Merge aliases
                existing_aliases = set(vocab[concept_id].get('aliases', []))
                new_aliases = set(item.get('aliases', []))
                vocab[concept_id]['aliases'] = list(existing_aliases | new_aliases)
                updated += 1
            else:
                vocab[concept_id] = {
                    'label': item.get('label', item.get('term', concept_id)),
                    'aliases': item.get('aliases', []),
                    'languages': item.get('languages', 'ALL'),
                    'freq_notes': item.get('freq_notes', 0),
                    'bucket': item.get('bucket', 'other')
                }
                added += 1
    elif isinstance(import_data, dict):
        # Dictionary format (same as lexicon)
        for concept_id, data in import_data.items():
            if concept_id in vocab:
                # Merge
                existing_aliases = set(vocab[concept_id].get('aliases', []))
                new_aliases = set(data.get('aliases', []))
                vocab[concept_id]['aliases'] = list(existing_aliases | new_aliases)
                updated += 1
            else:
                vocab[concept_id] = data
                added += 1
    
    save_vocabulary(vocab)
    print(f"Import complete: {added} added, {updated} updated")


def cmd_export(args):
    """Export vocabulary to a JSON file."""
    vocab = load_vocabulary()
    
    export_path = Path(args.file)
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(vocab)} concepts to {export_path}")


def cmd_search(args):
    """Search for concepts by term or alias."""
    vocab = load_vocabulary()
    query = args.query.lower()
    
    matches = []
    for concept_id, data in vocab.items():
        # Search in concept_id
        if query in concept_id.lower():
            matches.append((concept_id, data, 'id'))
            continue
        
        # Search in label
        if query in data.get('label', '').lower():
            matches.append((concept_id, data, 'label'))
            continue
        
        # Search in aliases
        for alias in data.get('aliases', []):
            if query in alias.lower():
                matches.append((concept_id, data, f'alias: {alias}'))
                break
    
    if not matches:
        print(f"No concepts found matching '{args.query}'")
        return
    
    print(f"\nFound {len(matches)} concepts matching '{args.query}':\n")
    for concept_id, data, match_type in matches:
        print(f"  {concept_id}")
        print(f"    Label: {data.get('label')}")
        print(f"    Bucket: {data.get('bucket')}")
        print(f"    Matched: {match_type}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Vocabulary Training CLI - Manage concept vocabulary',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show vocabulary statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # list command
    list_parser = subparsers.add_parser('list', help='List concepts')
    list_parser.add_argument('--bucket', '-b', help='Filter by bucket')
    list_parser.add_argument('--limit', '-n', type=int, default=50, help='Max concepts to show')
    list_parser.set_defaults(func=cmd_list)
    
    # add command
    add_parser = subparsers.add_parser('add', help='Add a new concept')
    add_parser.add_argument('term', help='The term to add')
    add_parser.add_argument('label', help='Human-readable label')
    add_parser.add_argument('bucket', help='Taxonomy bucket (preferences, intent, lifestyle, occasion, constraints, next_action, other)')
    add_parser.add_argument('--aliases', '-a', help='Comma-separated aliases')
    add_parser.add_argument('--languages', '-l', help='Languages (default: ALL)')
    add_parser.set_defaults(func=cmd_add)
    
    # remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a concept')
    remove_parser.add_argument('concept_id', help='Concept ID to remove')
    remove_parser.set_defaults(func=cmd_remove)
    
    # import command
    import_parser = subparsers.add_parser('import', help='Import concepts from JSON')
    import_parser.add_argument('file', help='JSON file to import')
    import_parser.set_defaults(func=cmd_import)
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export vocabulary to JSON')
    export_parser.add_argument('file', help='Output JSON file')
    export_parser.set_defaults(func=cmd_export)
    
    # search command
    search_parser = subparsers.add_parser('search', help='Search for concepts')
    search_parser.add_argument('query', help='Search query')
    search_parser.set_defaults(func=cmd_search)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
