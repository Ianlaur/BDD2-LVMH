"""
Big O Notation Visualization Generator

Generates a multi-panel PNG graph showing empirical Big O complexity
for every function tested in test_big_o.py.

Features:
- Runs all measurements and plots actual data points + fitted curves
- Annotates each subplot with the complexity class and exponent
- Colour-codes by complexity: green=O(1), blue=O(N), orange=O(N log N), red=O(N²)
- Timestamps the output file
- Auto-archives previous graphs into tests/archive/

Usage:
    python -m tests.generate_big_o_graph          # from project root
    python tests/generate_big_o_graph.py           # direct
"""
import os
import sys
import time
import math
import shutil
import random
import glob
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Output dirs ──────────────────────────────────────────────────
TESTS_DIR = PROJECT_ROOT / "tests"
ARCHIVE_DIR = TESTS_DIR / "archive"
RESULTS_DIR = TESTS_DIR / "results"

# ── Helpers (same as test_big_o.py) ──────────────────────────────

def _time_fn(fn: Callable, *args, repeats: int = 3, **kwargs) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return min(times)


def estimate_exponent(sizes: List[int], times: List[float]) -> Tuple[float, float]:
    """Return (exponent_b, intercept_a) from log-log fit."""
    log_n, log_t = [], []
    for n, t in zip(sizes, times):
        if t > 0:
            log_n.append(math.log(n))
            log_t.append(math.log(t))
    if len(log_n) < 2:
        return 0.0, 0.0
    log_n = np.array(log_n)
    log_t = np.array(log_t)
    x_mean = log_n.mean()
    y_mean = log_t.mean()
    b = float(((log_n - x_mean) * (log_t - y_mean)).sum() / ((log_n - x_mean) ** 2).sum())
    a = float(math.exp(y_mean - b * x_mean))
    return b, a


SAMPLE_WORDS = [
    "leather", "handbag", "perfume", "silk", "cashmere", "watch",
    "jewellery", "scarf", "monogram", "wallet", "sunglasses",
    "bracelet", "trunk", "couture", "cosmetics", "skincare",
]

def _random_text(n_words: int = 30) -> str:
    return " ".join(random.choices(SAMPLE_WORDS, k=n_words))

# ── Complexity colour mapping ────────────────────────────────────

def _complexity_color(label: str) -> str:
    label_lower = label.lower()
    if "o(1)" in label_lower:
        return "#2ecc71"   # green
    elif "n²" in label_lower or "n^2" in label_lower or "quadratic" in label_lower:
        return "#e74c3c"   # red
    elif "n log" in label_lower or "nlogn" in label_lower:
        return "#f39c12"   # orange
    else:
        return "#3498db"   # blue  (linear)


def _complexity_description(label: str, b: float) -> str:
    """Return a human-friendly explanation for the subplot."""
    if b < 0.3:
        return "O(1) – Constant time.\nRuntime does not grow with input size."
    elif b < 1.3:
        return f"O(N) – Linear time (b={b:.2f}).\nRuntime grows proportionally to input."
    elif b < 1.6:
        return f"O(N log N) – Log-linear (b={b:.2f}).\nSlightly super-linear, very efficient."
    elif b < 2.3:
        return f"O(N²) – Quadratic (b={b:.2f}).\nRuntime grows with the square of input."
    else:
        return f"O(N^{b:.1f}) – Super-quadratic (b={b:.2f}).\nMay need optimisation for large N."


# =====================================================================
# Measurement definitions — one entry per subplot
# =====================================================================

def collect_all_measurements() -> List[Dict]:
    """Run every benchmark and return a list of result dicts."""
    random.seed(42)
    np.random.seed(42)

    results = []

    # ── 1. build_alias_to_concept_map ────────────────────────────
    from server.extract.detect_concepts import build_alias_to_concept_map
    sizes = [50, 200, 500, 2_000, 5_000]
    times = []
    for n in sizes:
        rows = [{"concept_id": f"C{i:05d}", "label": f"label_{i}",
                 "aliases": "|".join(f"alias_{i}_{j}" for j in range(3))}
                for i in range(n)]
        df = pd.DataFrame(rows)
        times.append(_time_fn(build_alias_to_concept_map, df))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "build_alias_to_concept_map\nO(N) – Linear",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 2a. find_matches_in_text vs alias count ──────────────────
    from server.extract.detect_concepts import find_matches_in_text
    alias_sizes = [50, 200, 500, 1_000]
    text = _random_text(100)
    times = []
    for m in alias_sizes:
        amap = {f"alias_{i}": f"C{i:05d}" for i in range(m)}
        times.append(_time_fn(find_matches_in_text, text, amap))
    b, a = estimate_exponent(alias_sizes, times)
    results.append({"label": "find_matches_in_text\nvs alias count – O(M)",
                    "sizes": alias_sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 2b. find_matches_in_text vs text length ──────────────────
    text_sizes = [50, 200, 500, 2_000]
    amap = {f"alias_{i}": f"C{i:05d}" for i in range(100)}
    times = []
    for n in text_sizes:
        txt = _random_text(n)
        times.append(_time_fn(find_matches_in_text, txt, amap))
    b, a = estimate_exponent(text_sizes, times)
    results.append({"label": "find_matches_in_text\nvs text length – O(N)",
                    "sizes": text_sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 3. aggregate_client_vectors ──────────────────────────────
    from server.profiling.segment_clients import aggregate_client_vectors
    sizes = [100, 500, 2_000, 5_000]
    dim = 64
    times = []
    for n in sizes:
        df = pd.DataFrame({
            "note_id": [f"N{i:06d}" for i in range(n)],
            "client_id": [f"CA{random.randint(1, max(1, n//5)):05d}" for _ in range(n)],
            "embedding": [np.random.randn(dim).tolist() for _ in range(n)],
        })
        times.append(_time_fn(aggregate_client_vectors, df))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "aggregate_client_vectors\nO(N) – Linear",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 4. KMeans clustering ─────────────────────────────────────
    from sklearn.cluster import KMeans
    sizes = [100, 500, 2_000, 5_000]
    times = []
    for n in sizes:
        X = np.random.randn(n, 64).astype(np.float32)
        km = KMeans(n_clusters=5, random_state=42, n_init=1, max_iter=50)
        times.append(_time_fn(km.fit, X))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "KMeans.fit()\nO(N·K·D·I) – Linear in N",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 5. Silhouette score ──────────────────────────────────────
    from sklearn.metrics import silhouette_score as sil_score
    sizes = [100, 300, 800, 2_000]
    times = []
    for n in sizes:
        X = np.random.randn(n, 32).astype(np.float32)
        labels = np.random.randint(0, 4, size=n)
        times.append(_time_fn(sil_score, X, labels, sample_size=min(n, 500)))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "silhouette_score\nO(N²) – Quadratic",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N²)"})

    # ── 6. Euclidean distances ───────────────────────────────────
    from sklearn.metrics.pairwise import euclidean_distances
    sizes = [100, 300, 800, 2_000]
    times = []
    for n in sizes:
        X = np.random.randn(n, 64).astype(np.float32)
        times.append(_time_fn(euclidean_distances, X))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "euclidean_distances\nO(N²·D) – Quadratic",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N²)"})

    # ── 7. Sorting aliases ───────────────────────────────────────
    sizes = [500, 2_000, 10_000, 50_000]
    times = []
    for m in sizes:
        aliases = [f"alias_{i}_{'x'*random.randint(1,20)}" for i in range(m)]
        times.append(_time_fn(sorted, aliases, key=len, reverse=True))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "sorted(aliases, key=len)\nO(M log M) – Log-linear",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N log N)"})

    # ── 8. compute_k ─────────────────────────────────────────────
    from server.profiling.segment_clients import compute_k
    sizes = [10, 100, 1_000, 10_000, 100_000]
    times = []
    for n in sizes:
        times.append(_time_fn(compute_k, n, env_k=0, repeats=100))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "compute_k()\nO(1) – Constant",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(1)"})

    # ── 9. Groupby aggregation ───────────────────────────────────
    sizes = [500, 2_000, 10_000, 50_000]
    times = []
    for n in sizes:
        df = pd.DataFrame({
            "client_id": [f"CA{random.randint(1, max(1, n//10)):05d}" for _ in range(n)],
            "value": np.random.randn(n),
        })
        def _gb(d=df):
            d.groupby("client_id")["value"].mean()
        times.append(_time_fn(_gb))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "DataFrame.groupby().mean()\nO(N) – Linear",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    # ── 10. Regex search ─────────────────────────────────────────
    import re
    pattern = re.compile(r"\b(leather|perfume|silk|cashmere)\b", re.IGNORECASE)
    sizes = [100, 1_000, 10_000, 50_000]
    times = []
    for n in sizes:
        text = _random_text(n)
        def _search(t=text):
            list(pattern.finditer(t))
        times.append(_time_fn(_search))
    b, a = estimate_exponent(sizes, times)
    results.append({"label": "re.finditer()\nO(N) – Linear",
                    "sizes": sizes, "times": times, "b": b, "a": a,
                    "expected": "O(N)"})

    return results


# =====================================================================
# Graph rendering
# =====================================================================

def archive_old_graphs():
    """Move any existing big_o_*.png files into tests/archive/."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    for pattern in ["big_o_*.png", "big_o_*.json"]:
        for old_file in RESULTS_DIR.glob(pattern):
            dest = ARCHIVE_DIR / old_file.name
            shutil.move(str(old_file), str(dest))
            print(f"  📦 Archived: {old_file.name} → archive/")


def render_graph(results: List[Dict]) -> Path:
    """Render a multi-panel Big O visualization and return the PNG path."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    n_plots = len(results)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.5 * rows))
    fig.suptitle(
        f"LVMH Pipeline – Big O Complexity Analysis\n"
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
        fontsize=18, fontweight="bold", y=0.98
    )
    
    # Flatten axes for easy iteration
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]
    
    for idx, (res, ax) in enumerate(zip(results, axes_flat)):
        sizes = res["sizes"]
        times_ms = [t * 1000 for t in res["times"]]  # convert to ms
        b = res["b"]
        a = res["a"]
        color = _complexity_color(res["label"])
        
        # Plot actual data points
        ax.plot(sizes, times_ms, "o-", color=color, markersize=8,
                linewidth=2, label="Measured", zorder=3)
        
        # Plot fitted power-law curve
        if a > 0 and any(t > 0 for t in res["times"]):
            fit_x = np.linspace(min(sizes), max(sizes), 200)
            fit_y = [a * (x ** b) * 1000 for x in fit_x]
            ax.plot(fit_x, fit_y, "--", color=color, alpha=0.4,
                    linewidth=1.5, label=f"Fit: T = a·N^{b:.2f}")
        
        # Formatting
        ax.set_title(res["label"], fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Input size (N)", fontsize=9)
        ax.set_ylabel("Time (ms)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        
        # Log scale if range is large
        if max(sizes) / min(sizes) > 50:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"))
        else:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{int(x):,}"))
        
        # Annotation box with complexity explanation
        desc = _complexity_description(res["label"], b)
        ax.text(0.97, 0.03, desc, transform=ax.transAxes,
                fontsize=7.5, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color,
                          alpha=0.15, edgecolor=color))
        
        # Exponent badge
        ax.text(0.03, 0.97, f"b = {b:.3f}", transform=ax.transAxes,
                fontsize=10, fontweight="bold", verticalalignment="top",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9))
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    # Add legend explaining the colour code at the bottom
    legend_elements = [
        Line2D([0], [0], marker="o", color="#2ecc71", label="O(1) Constant",
               markersize=8, linewidth=2),
        Line2D([0], [0], marker="o", color="#3498db", label="O(N) Linear",
               markersize=8, linewidth=2),
        Line2D([0], [0], marker="o", color="#f39c12", label="O(N log N) Log-linear",
               markersize=8, linewidth=2),
        Line2D([0], [0], marker="o", color="#e74c3c", label="O(N²) Quadratic",
               markersize=8, linewidth=2),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=10, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / f"big_o_report_{timestamp}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    
    # Also save a JSON summary
    import json
    summary = []
    for r in results:
        summary.append({
            "label": r["label"].replace("\n", " | "),
            "expected": r["expected"],
            "measured_exponent": round(r["b"], 4),
            "sizes": r["sizes"],
            "times_ms": [round(t * 1000, 4) for t in r["times"]],
        })
    json_path = RESULTS_DIR / f"big_o_report_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({"generated": timestamp, "results": summary}, f, indent=2)
    
    return png_path


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("  LVMH Pipeline – Big O Complexity Graph Generator")
    print("=" * 60)
    
    # Step 1: Archive old results
    print("\n📦 Archiving previous results...")
    archive_old_graphs()
    
    # Step 2: Run all benchmarks
    print("\n⏱️  Running benchmarks (this takes ~5-10 seconds)...")
    results = collect_all_measurements()
    
    # Step 3: Generate graph
    print("\n📊 Generating graph...")
    png_path = render_graph(results)
    
    print(f"\n✅ Graph saved to: {png_path}")
    print(f"   JSON saved to:  {png_path.with_suffix('.json')}")
    print(f"\n   Old reports archived in: tests/archive/")
    print("=" * 60)
    
    return png_path


if __name__ == "__main__":
    main()
