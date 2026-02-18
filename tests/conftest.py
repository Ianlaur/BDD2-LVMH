"""
Shared pytest fixtures for Big O / complexity tests.

Provides synthetic data generators at configurable sizes so that
empirical timing tests can measure how runtime scales with N.
Also includes a pytest plugin that auto-generates the Big O PNG graph
after every successful test session and archives old reports.
"""
import random
import string
import numpy as np
import pandas as pd
import pytest

# ── reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── helpers ──────────────────────────────────────────────────────

SAMPLE_WORDS = [
    "leather", "handbag", "perfume", "silk", "cashmere", "watch",
    "jewellery", "scarf", "monogram", "wallet", "sunglasses",
    "bracelet", "trunk", "couture", "cosmetics", "skincare",
    "foundation", "lipstick", "fragrance", "necklace", "ring",
    "earrings", "shoes", "belt", "travel", "champagne", "cognac",
    "tote", "clutch", "heels", "sneakers", "loafers", "boots",
]

CONCEPT_IDS = [f"C{str(i).zfill(4)}" for i in range(1, 51)]
LABELS = [
    "luxury_leather", "premium_fragrance", "haute_couture",
    "fine_jewellery", "classic_watch", "silk_accessory",
    "travel_essential", "skincare_premium", "heritage_monogram",
    "limited_edition", "art_de_vivre", "bridal_collection",
    "evening_wear", "casual_chic", "sport_luxe",
    "resort_collection", "denim_luxe", "sustainable_fashion",
    "bespoke_tailoring", "iconic_print", "savoir_faire",
    "petite_maroquinerie", "haute_horlogerie", "joaillerie",
    "pret_a_porter", "defile_collection", "capsule_collection",
    "festive_gift", "mens_accessories", "womens_accessories",
    "unisex_fragrance", "limited_edition_watch", "vintage_trunk",
    "artisan_craft", "vegan_leather", "recycled_material",
    "diamond_setting", "gold_plating", "enamel_work",
    "hand_stitched", "made_in_france", "made_in_italy",
    "japanese_craft", "swiss_movement", "mother_of_pearl",
    "alligator_leather", "ostrich_leather", "canvas_monogram",
    "damier_pattern", "toile_check",
]


def _random_text(n_words: int = 30) -> str:
    """Generate a pseudo-realistic note."""
    return " ".join(random.choices(SAMPLE_WORDS, k=n_words))


def _random_alias() -> str:
    """Generate a random alias string (1-3 words)."""
    k = random.randint(1, 3)
    return " ".join(random.choices(SAMPLE_WORDS, k=k))


# ── fixtures ─────────────────────────────────────────────────────

@pytest.fixture(params=[100, 500, 1_000, 5_000])
def notes_df(request) -> pd.DataFrame:
    """Synthetic notes DataFrame with configurable row count."""
    n = request.param
    return pd.DataFrame({
        "note_id": [f"N{i:06d}" for i in range(n)],
        "client_id": [f"CA{random.randint(1, max(1, n // 5)):05d}" for _ in range(n)],
        "text": [_random_text() for _ in range(n)],
    })


@pytest.fixture(params=[100, 500, 1_000, 5_000])
def vectors_df(request) -> pd.DataFrame:
    """Synthetic vectors DataFrame with configurable row count."""
    n = request.param
    dim = 64  # smaller dim keeps tests fast
    return pd.DataFrame({
        "note_id": [f"N{i:06d}" for i in range(n)],
        "client_id": [f"CA{random.randint(1, max(1, n // 5)):05d}" for _ in range(n)],
        "embedding": [np.random.randn(dim).tolist() for _ in range(n)],
    })


@pytest.fixture(params=[50, 200, 500, 1_000])
def alias_map(request) -> dict:
    """Synthetic alias→concept_id map of configurable size."""
    n = request.param
    mapping = {}
    for i in range(n):
        concept_id = CONCEPT_IDS[i % len(CONCEPT_IDS)]
        alias = _random_alias() + f" {i}"  # ensure uniqueness
        mapping[alias.lower()] = concept_id
    return mapping


@pytest.fixture(params=[50, 200, 500, 2_000])
def embeddings_array(request) -> np.ndarray:
    """Random embedding matrix (n × 64) for clustering tests."""
    n = request.param
    return np.random.randn(n, 64).astype(np.float32)


def build_lexicon_df(n_concepts: int = 50) -> pd.DataFrame:
    """Build a synthetic lexicon DataFrame."""
    rows = []
    for i in range(n_concepts):
        rows.append({
            "concept_id": CONCEPT_IDS[i % len(CONCEPT_IDS)],
            "label": LABELS[i % len(LABELS)],
            "aliases": "|".join(_random_alias() for _ in range(random.randint(1, 5))),
            "category": random.choice(["leather", "fragrance", "fashion", "jewellery"]),
        })
    return pd.DataFrame(rows)


# ── Pytest plugin: auto-generate Big O graph after test session ──

def pytest_sessionfinish(session, exitstatus):
    """
    After all tests pass, automatically:
      1. Archive old Big O PNGs/JSONs into tests/archive/
      2. Generate a fresh timestamped Big O graph
    Only runs when ALL tests passed (exitstatus == 0).
    """
    if exitstatus != 0:
        return  # skip graph generation if tests failed

    try:
        from tests.generate_big_o_graph import archive_old_graphs, collect_all_measurements, render_graph
        print("\n\n" + "=" * 60)
        print("  AUTO-GENERATING BIG O GRAPH")
        print("=" * 60)

        print("\n📦 Archiving previous results...")
        archive_old_graphs()

        print("⏱️  Running benchmarks...")
        results = collect_all_measurements()

        print("📊 Rendering graph...")
        png_path = render_graph(results)

        print(f"\n✅ Graph: {png_path}")
        print(f"   JSON:  {png_path.with_suffix('.json')}")
        print("=" * 60)
    except Exception as exc:
        print(f"\n⚠️  Graph generation failed: {exc}")
        import traceback
        traceback.print_exc()
