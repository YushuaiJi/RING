# RING

RING (Range-Indexed Navigation Graph) answers approximate nearest-neighbor queries whose candidates
must satisfy range predicates on several numerical attributes. It builds proximity graphs over a
hierarchy of attribute partitions and adds query-specific range graphs for ranges seen in the
historical query workload, so a query on such a range searches only objects inside it. Historical
cross-graph connections give entry points between ranges for multi-range queries.

This repository contains the RING implementation used in our PVLDB submission: the C++ core, the
Python interface, and a small runnable example.

## Requirements

- Linux on x86-64 (the distance kernels use AVX), GCC 8+ or Clang with C++17
- CMake 3.18+
- Python 3.9+ with NumPy and pybind11

## Build

```bash
pip install numpy pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j
python examples/quickstart.py
```

The extension (`_frans_v11*.so`) is written into `ring/`; add the repository root to `PYTHONPATH`
(or run from it) to `import ring`. Pass `-DRING_NATIVE_ARCH=OFF` to build without `-march=native`.

## Usage

```python
import numpy as np
import ring

index = ring.FullIndex(vectors, attrs, workers=16)    # float32 [N, D], float64 [N, M]
index.fit_base()                                      # partition hierarchy + base graphs
ranges = [ring.window(attrs, dims=[0, 1], lower=lo, upper=hi, window_id=i) for i, (lo, hi) in enumerate(history)]
index.add_windows(ranges)                             # query-specific range graphs
ring.prepare(index)                                   # pack graphs, enable optimized kernels

ids, dist = ring.search_window(index, query, ranges[0]["box"], k=10, budget=64)
```

- `budget` is the search beam width; larger values trade speed for recall.
- `index.search(record, budget)` is the interface used in the experiments: `record` holds
  `query_vector_id` (a corpus object) and `boxes` (one or more ranges).
- Multi-range queries use connections learned from historical queries:
  `index.learn_bridges(history, history_budget=128, bridge_h=4)`, where each history record has
  `query_vector_id` and `boxes`.
- `index.save(root, phase, vector_path=..., attr_path=...)` and `ring.FullIndex.load(root, phase)`
  persist and reload an index (vectors and attributes stay in the given `.npy` files).

Default construction parameters (`ring.PARAMETERS`): leaf size 1,024, candidate budget 64,
alpha 1.2, maximum degree 32, 8 entry points per graph.

## Layout

| Path | Contents |
|---|---|
| `cpp/bounded_blowup/` | core C++ library: graph construction, leaf repair, graph search |
| `src/ring_native.cpp` | Python extension: corpus, graphs, single- and multi-graph search |
| `ring/` | Python index (`FullIndex`) and helpers; `frans_*` modules are internal stages |
| `examples/quickstart.py` | end-to-end example on synthetic data |
