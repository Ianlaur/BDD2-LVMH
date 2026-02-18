# Unit Tests – Big O Complexity Analysis

## Overview

This test suite empirically measures the **time complexity (Big O)** of critical functions in the LVMH Voice-to-Tag pipeline. Instead of theoretical analysis alone, we *run* each function at several input sizes, fit a power-law model $T = a \cdot N^b$, and assert the exponent $b$ stays within the expected complexity class.

**After every successful test run, a timestamped PNG graph is automatically generated.** Old graphs are archived into `tests/archive/` so you always have a history.

## Directory structure

```
tests/
├── __init__.py                  # Package init
├── conftest.py                  # Fixtures + auto-graph-generation hook
├── test_big_o.py                # 11 empirical Big O tests
├── generate_big_o_graph.py      # Standalone graph generator
├── README.md                    # This file
├── results/                     # Current graph + JSON report
│   ├── big_o_report_20260210_170000.png
│   └── big_o_report_20260210_170000.json
└── archive/                     # Auto-archived old reports
    ├── big_o_report_20260210_160000.png
    └── big_o_report_20260210_160000.json
```

## What's tested

| # | Function / Operation | Expected | Threshold |
|---|---|---|---|
| 1 | `build_alias_to_concept_map` | O(N) | b < 1.6 |
| 2a | `find_matches_in_text` vs alias count | O(M) | b < 1.8 |
| 2b | `find_matches_in_text` vs text length | O(N) | b < 1.8 |
| 3 | `aggregate_client_vectors` | O(N) | b < 1.6 |
| 4 | KMeans `.fit()` vs N | O(N·K·D·I) | b < 1.8 |
| 5 | `silhouette_score` | O(N²) | b < 2.8 |
| 6 | Pairwise `euclidean_distances` | O(N²·D) | 1.0 < b < 2.8 |
| 7 | Alias sorting by length | O(M log M) | b < 1.5 |
| 8 | `compute_k` heuristic | O(1) | b < 0.5 |
| 9 | DataFrame `groupby().mean()` | O(N) | b < 1.6 |
| 10 | `re.finditer` regex search | O(N) | b < 1.5 |

## Running

```bash
# Run all Big O tests + auto-generate graph
python -m pytest tests/test_big_o.py -v -s

# Generate graph only (without running tests)
python -m tests.generate_big_o_graph

# Run a single test class (graph still auto-generates if all pass)
python -m pytest tests/test_big_o.py::TestKMeansClustering -v -s

# Run with timing stats
python -m pytest tests/test_big_o.py -v -s --tb=short
```

## Graph output

Every run produces:
- **`tests/results/big_o_report_YYYYMMDD_HHMMSS.png`** – Multi-panel visualization
- **`tests/results/big_o_report_YYYYMMDD_HHMMSS.json`** – Machine-readable results

Previous reports are automatically moved to **`tests/archive/`** before a new one is generated.

You can also generate a graph manually without running tests:
```bash
python -m tests.generate_big_o_graph
```

## Interpreting results

Each test prints a report like:

```
──────────────────────────────────────────────────────────────
  build_alias_to_concept_map — O(N)
──────────────────────────────────────────────────────────────
    N=    50  →       0.042 ms
    N=   200  →       0.156 ms
    N=   500  →       0.389 ms
    N= 2,000  →       1.534 ms
    N= 5,000  →       3.821 ms
  Estimated exponent (b): 1.02
──────────────────────────────────────────────────────────────
```

- **b ≈ 1.0** → linear O(N)
- **b ≈ 1.3** → close to O(N log N)
- **b ≈ 2.0** → quadratic O(N²)
- **b ≈ 0.0** → constant O(1)

The thresholds are intentionally generous to avoid flaky failures on slow CI runners or cold caches.

## Adding new tests

1. Pick sizes that span at least 1–2 orders of magnitude.
2. Use `_time_fn(fn, *args, repeats=3)` to reduce noise.
3. Use `estimate_exponent()` to fit the power law.
4. Use `_report()` for readable output.
5. Assert the exponent is within the expected range.
