"""Small helpers around FullIndex: describe ranges, finalize the index, query with any vector."""
from __future__ import annotations

import hashlib

import numpy as np

from .frans_v12.model import raw_box_signature


def members(attrs, box):
    """Object IDs whose attributes lie inside the closed box (unconstrained dimensions are free)."""
    dims = list(box["dims"])
    values = np.asarray(attrs)[:, dims]
    inside = np.all((values >= np.asarray(box["lower"])) & (values <= np.asarray(box["upper"])), axis=1)
    return np.flatnonzero(inside).astype(np.int64)


def window(attrs, dims, lower, upper, window_id):
    """A historical query range in the form FullIndex.add_windows expects."""
    box = dict(dims=[int(d) for d in dims], lower=[float(v) for v in lower], upper=[float(v) for v in upper])
    ids = members(attrs, box)
    return dict(window_id=str(window_id), box=box, legal_count=int(len(ids)),
                membership_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest())


def prepare(index, engine_mode=7):
    """Finalize after add_windows: pack registered range graphs and enable the optimized search
    kernels (the setting used in the paper's experiments)."""
    for key in {row["graph_id"] for row in index.range_directory}:
        index.search_index.pack_single(index.graph_native_ids[key], True)
    search = index.search_index
    search.set_engineering_mode(engine_mode)
    search.set_joint_optimization(engine_mode)
    search.set_single_optimization(engine_mode)
    return index


def search_window(index, query, box, *, k=10, budget=64, entry_mode=None):
    """k nearest neighbors of an arbitrary query vector within a registered range.

    The range must have been registered with add_windows (same box); the search then runs on that
    range's graph only, so every returned object satisfies the range. Returns (ids, squared L2).
    """
    route = index.window_routes.get(raw_box_signature(box))
    if route is None:
        raise ValueError("range is not registered; add it with FullIndex.add_windows first")
    vector = np.ascontiguousarray(query, dtype=np.float32)
    result = index.search_index.search_single(vector, route["native_graph_id"], k=int(k), beam=int(budget),
                                              entry_mode=entry_mode or index.entry_mode)
    return np.asarray(result["ids"], dtype=np.int64), np.asarray(result["distances"], dtype=np.float32)
