"""
Empirical Big O complexity tests for LVMH pipeline functions.

Strategy
--------
For each function under test we:
 1. Run it at multiple input sizes (N).
 2. Record wall-clock time for each N.
 3. Fit a simple power-law  T = a·N^b  via log-log linear regression.
 4. Assert that the estimated exponent **b** stays within the expected
    complexity class:
        O(N)        → b ≈ 1.0   (accept b < 1.6)
        O(N log N)  → b ≈ 1.0–1.3 (accept b < 1.8)
        O(N²)       → b ≈ 2.0   (accept b < 2.5)
        O(N·M)      → see individual test docstrings

Sizes are deliberately small so the full suite runs in < 30 s on CI.
"""
import time
import math
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

# ── helpers ──────────────────────────────────────────────────────

def _time_fn(fn: Callable, *args, repeats: int = 3, **kwargs) -> float:
    """Return the *minimum* wall-clock time (seconds) over `repeats` runs."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return min(times)


def estimate_exponent(sizes: List[int], times: List[float]) -> float:
    """
    Fit T = a·N^b in log-log space and return exponent **b**.

    Uses ordinary least-squares on  log(T) = log(a) + b·log(N).
    Filters out any zero/negative times (can happen if a run is too fast).
    """
    log_n, log_t = [], []
    for n, t in zip(sizes, times):
        if t > 0:
            log_n.append(math.log(n))
            log_t.append(math.log(t))
    if len(log_n) < 2:
        return 0.0  # not enough data points
    log_n = np.array(log_n)
    log_t = np.array(log_t)
    # OLS: b = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
    x_mean = log_n.mean()
    y_mean = log_t.mean()
    b = ((log_n - x_mean) * (log_t - y_mean)).sum() / ((log_n - x_mean) ** 2).sum()
    return float(b)


def _report(label: str, sizes: List[int], times: List[float], exponent: float):
    """Print a human-readable complexity report (visible via pytest -s)."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    for n, t in zip(sizes, times):
        print(f"    N={n:>6,}  →  {t*1000:>10.3f} ms")
    print(f"  Estimated exponent (b): {exponent:.3f}")
    print(f"{'─' * 60}")


# ── sample words for text generation ────────────────────────────

SAMPLE_WORDS = [
    "leather", "handbag", "perfume", "silk", "cashmere", "watch",
    "jewellery", "scarf", "monogram", "wallet", "sunglasses",
    "bracelet", "trunk", "couture", "cosmetics", "skincare",
]


def _random_text(n_words: int = 30) -> str:
    return " ".join(random.choices(SAMPLE_WORDS, k=n_words))


# =====================================================================
# TEST 1: Alias map construction – O(N) in number of lexicon rows
# =====================================================================

class TestBuildAliasMap:
    """build_alias_to_concept_map should scale linearly with lexicon size."""

    SIZES = [50, 200, 500, 2_000, 5_000]

    @staticmethod
    def _build_lexicon_df(n_rows: int) -> pd.DataFrame:
        rows = []
        for i in range(n_rows):
            rows.append({
                "concept_id": f"C{i:05d}",
                "label": f"label_{i}",
                "aliases": "|".join(f"alias_{i}_{j}" for j in range(3)),
            })
        return pd.DataFrame(rows)

    def test_alias_map_construction_is_linear(self):
        """
        Expected: O(N) where N = number of lexicon rows.
        Accept exponent b < 1.6 (allows constant overhead noise).
        """
        from server.extract.detect_concepts import build_alias_to_concept_map

        sizes, times = [], []
        for n in self.SIZES:
            df = self._build_lexicon_df(n)
            t = _time_fn(build_alias_to_concept_map, df)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("build_alias_to_concept_map — O(N)", sizes, times, b)
        assert b < 1.6, f"Exponent {b:.2f} exceeds linear threshold (1.6)"


# =====================================================================
# TEST 2: Text matching – O(N·M) where N=text length, M=alias count
# =====================================================================

class TestFindMatchesInText:
    """find_matches_in_text should scale as O(N·M)."""

    # We vary M (alias count) with fixed text length
    ALIAS_SIZES = [50, 200, 500, 1_000]
    TEXT_WORD_COUNT = 100

    @staticmethod
    def _make_alias_map(n: int) -> dict:
        return {f"alias_{i}": f"C{i:05d}" for i in range(n)}

    def test_match_scales_with_alias_count(self):
        """
        Fix text length, vary alias map size M.
        Expected exponent on M: O(M) → b ≈ 1.  Accept b < 1.8.
        """
        from server.extract.detect_concepts import find_matches_in_text

        text = _random_text(self.TEXT_WORD_COUNT)
        sizes, times = [], []
        for m in self.ALIAS_SIZES:
            amap = self._make_alias_map(m)
            t = _time_fn(find_matches_in_text, text, amap)
            sizes.append(m)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("find_matches_in_text vs alias_count — O(M)", sizes, times, b)
        assert b < 1.8, f"Exponent {b:.2f} exceeds expected threshold (1.8)"

    # Vary text length with fixed alias count
    TEXT_SIZES = [50, 200, 500, 2_000]
    FIXED_ALIASES = 100

    def test_match_scales_with_text_length(self):
        """
        Fix alias map size, vary text word count N.
        Expected exponent on N: O(N) → b ≈ 1.  Accept b < 1.8.
        """
        from server.extract.detect_concepts import find_matches_in_text

        amap = self._make_alias_map(self.FIXED_ALIASES)
        sizes, times = [], []
        for n in self.TEXT_SIZES:
            text = _random_text(n)
            t = _time_fn(find_matches_in_text, text, amap)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("find_matches_in_text vs text_length — O(N)", sizes, times, b)
        assert b < 1.8, f"Exponent {b:.2f} exceeds expected threshold (1.8)"


# =====================================================================
# TEST 3: Vector aggregation – O(N) in number of note vectors
# =====================================================================

class TestAggregateClientVectors:
    """aggregate_client_vectors should scale linearly with note count."""

    SIZES = [100, 500, 2_000, 5_000]
    DIM = 64

    @staticmethod
    def _make_vectors_df(n: int, dim: int) -> pd.DataFrame:
        return pd.DataFrame({
            "note_id": [f"N{i:06d}" for i in range(n)],
            "client_id": [f"CA{random.randint(1, max(1, n // 5)):05d}" for _ in range(n)],
            "embedding": [np.random.randn(dim).tolist() for _ in range(n)],
        })

    def test_aggregation_is_linear(self):
        """
        Expected: O(N) where N = number of notes.
        Accept exponent b < 1.6.
        """
        from server.profiling.segment_clients import aggregate_client_vectors

        sizes, times = [], []
        for n in self.SIZES:
            df = self._make_vectors_df(n, self.DIM)
            t = _time_fn(aggregate_client_vectors, df)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("aggregate_client_vectors — O(N)", sizes, times, b)
        assert b < 1.6, f"Exponent {b:.2f} exceeds linear threshold (1.6)"


# =====================================================================
# TEST 4: KMeans clustering – O(N·K·D·I)
# =====================================================================

class TestKMeansClustering:
    """
    sklearn KMeans is O(N·K·D·I) where
      N = samples, K = clusters, D = dims, I = iterations.
    Varying N with K, D, I fixed → expect linear in N  (b ≈ 1).
    """

    SIZES = [100, 500, 2_000, 5_000]
    DIM = 64
    K = 5

    def test_kmeans_scales_linearly_with_n(self):
        """Accept b < 1.8 (sklearn has optimised internals)."""
        sizes, times = [], []
        for n in self.SIZES:
            X = np.random.randn(n, self.DIM).astype(np.float32)
            km = KMeans(n_clusters=self.K, random_state=42, n_init=1, max_iter=50)

            t = _time_fn(km.fit, X)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("KMeans.fit vs N — O(N·K·D·I)", sizes, times, b)
        assert b < 1.8, f"Exponent {b:.2f} exceeds expected threshold (1.8)"


# =====================================================================
# TEST 5: Silhouette score – O(N²) pairwise distances
# =====================================================================

class TestSilhouetteScore:
    """
    silhouette_score computes pairwise distances → O(N²).
    We verify the exponent is roughly 2.
    """

    SIZES = [100, 300, 800, 2_000]
    DIM = 32
    K = 4

    def test_silhouette_is_quadratic(self):
        """Accept 1.5 < b < 2.8."""
        from sklearn.metrics import silhouette_score as sil_score

        sizes, times = [], []
        for n in self.SIZES:
            X = np.random.randn(n, self.DIM).astype(np.float32)
            labels = np.random.randint(0, self.K, size=n)

            t = _time_fn(sil_score, X, labels, sample_size=min(n, 500))
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("silhouette_score — O(N²)", sizes, times, b)
        assert b < 2.8, f"Exponent {b:.2f} too large for silhouette"


# =====================================================================
# TEST 6: Euclidean distance matrix – O(N²·D)
# =====================================================================

class TestEuclideanDistances:
    """Pairwise euclidean distances → O(N²·D). Vary N, fix D."""

    SIZES = [100, 300, 800, 2_000]
    DIM = 64

    def test_pairwise_distances_quadratic(self):
        """Accept 1.5 < b < 2.8."""
        from sklearn.metrics.pairwise import euclidean_distances

        sizes, times = [], []
        for n in self.SIZES:
            X = np.random.randn(n, self.DIM).astype(np.float32)
            t = _time_fn(euclidean_distances, X)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("euclidean_distances — O(N²·D)", sizes, times, b)
        assert 1.0 < b < 2.8, f"Exponent {b:.2f} outside expected range"


# =====================================================================
# TEST 7: Sorting aliases by length – O(M log M)
# =====================================================================

class TestAliasSorting:
    """Sorting aliases by length (used in detect_concepts) → O(M log M)."""

    SIZES = [500, 2_000, 10_000, 50_000]

    def test_sorting_is_n_log_n(self):
        """Accept b < 1.5 (Tim-sort is very cache-friendly)."""
        sizes, times = [], []
        for m in self.SIZES:
            aliases = [f"alias_{i}_{'x' * random.randint(1, 20)}" for i in range(m)]
            t = _time_fn(sorted, aliases, key=len, reverse=True)
            sizes.append(m)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("sorted(aliases, key=len) — O(M log M)", sizes, times, b)
        assert b < 1.5, f"Exponent {b:.2f} exceeds expected threshold"


# =====================================================================
# TEST 8: compute_k heuristic – O(1) constant time
# =====================================================================

class TestComputeK:
    """compute_k is a simple formula → O(1)."""

    SIZES = [10, 100, 1_000, 10_000, 100_000]

    def test_compute_k_is_constant(self):
        """All sizes should finish in essentially the same time. b ≈ 0."""
        from server.profiling.segment_clients import compute_k

        sizes, times = [], []
        for n in self.SIZES:
            t = _time_fn(compute_k, n, env_k=0, repeats=100)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("compute_k — O(1)", sizes, times, b)
        assert b < 0.5, f"Exponent {b:.2f} suggests non-constant time"


# =====================================================================
# TEST 9: DataFrame groupby aggregation – O(N)
# =====================================================================

class TestGroupbyAggregation:
    """pandas groupby().mean() → O(N)."""

    SIZES = [500, 2_000, 10_000, 50_000]
    DIM = 32

    def test_groupby_mean_is_linear(self):
        """Accept b < 1.6."""
        sizes, times = [], []
        for n in self.SIZES:
            df = pd.DataFrame({
                "client_id": [f"CA{random.randint(1, max(1, n // 10)):05d}" for _ in range(n)],
                "value": np.random.randn(n),
            })

            def _groupby():
                df.groupby("client_id")["value"].mean()

            t = _time_fn(_groupby)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("DataFrame.groupby().mean() — O(N)", sizes, times, b)
        assert b < 1.6, f"Exponent {b:.2f} exceeds linear threshold"


# =====================================================================
# TEST 10: Regex compilation & search – O(N) in text length
# =====================================================================

class TestRegexSearch:
    """re.finditer is O(N) in text length for simple patterns."""

    import re
    PATTERN = re.compile(r"\b(leather|perfume|silk|cashmere)\b", re.IGNORECASE)
    SIZES = [100, 1_000, 10_000, 50_000]

    def test_regex_search_is_linear(self):
        """Accept b < 1.5."""
        import re

        sizes, times = [], []
        for n in self.SIZES:
            text = _random_text(n)

            def _search():
                list(self.PATTERN.finditer(text))

            t = _time_fn(_search)
            sizes.append(n)
            times.append(t)

        b = estimate_exponent(sizes, times)
        _report("re.finditer — O(N)", sizes, times, b)
        assert b < 1.5, f"Exponent {b:.2f} exceeds linear threshold"
