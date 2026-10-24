"""Build a RING index on synthetic data, register a few historical ranges, and query them.

Run from the repository root after building (see README):  python examples/quickstart.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ring  # noqa: E402

rng = np.random.default_rng(7)
n, dim = 50_000, 32
vectors = rng.standard_normal((n, dim), dtype=np.float32)
attrs = rng.uniform(0.0, 1.0, size=(n, 2))  # two numerical attributes

# 1. Base index over the whole collection.
index = ring.FullIndex(vectors, attrs, workers=8)
started = time.perf_counter()
index.fit_base()
print(f"base index: {len(index.nodes)} tree nodes in {time.perf_counter() - started:.1f} s")

# 2. Register query ranges seen in the historical workload (about 1,000 objects each here).
corners = [(0.10, 0.10), (0.50, 0.60), (0.25, 0.70), (0.70, 0.20)]
ranges = [ring.window(attrs, [0, 1], (x, y), (x + 0.15, y + 0.15), f"range-{i}") for i, (x, y) in enumerate(corners)]
started = time.perf_counter()
receipt = index.add_windows(ranges)
ring.prepare(index)
print(f"registered {receipt['window_count']} ranges ({receipt['extracted_window_graphs']} new range graphs) "
      f"in {time.perf_counter() - started:.1f} s")


def exact_top10(query, ids):
    distances = ((vectors[ids] - query) ** 2).sum(axis=1)
    return set(ids[np.argsort(distances, kind="stable")[:10]].tolist())


# 3. Query each range with new vectors and compare with exact search over the eligible objects.
for budget in (16, 64):
    hits, total, elapsed = 0, 0, 0.0
    for w in ranges:
        eligible = ring.members(attrs, w["box"])
        for _ in range(50):
            query = rng.standard_normal(dim, dtype=np.float32)
            started = time.perf_counter()
            ids, _ = ring.search_window(index, query, w["box"], k=10, budget=budget)
            elapsed += time.perf_counter() - started
            assert np.isin(ids, eligible).all()  # every answer satisfies the range
            hits += len(set(ids.tolist()) & exact_top10(query, eligible))
            total += 10
    print(f"budget {budget:3d}: Recall@10 {hits / total:.3f}, {4 * 50 / elapsed:,.0f} queries/s")

# The experiment code queries with corpus objects through the same route:
answer = index.search(dict(query_vector_id=123, boxes=[ranges[0]["box"]]), 64)
print("corpus-object query on range-0:", answer["ids"][:5], "...")
