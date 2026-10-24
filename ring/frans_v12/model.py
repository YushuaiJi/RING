"""V12 variable-cardinality/dimensionality routing using unchanged V11 native.

The V10 KD construction and graph registration policy are reused as ordinary
inherited methods, never by replacing a module in ``sys.modules``.  Only the
frozen exact windows have a V12 request path; unmatched predicates are
rejected, not silently sent to an all-legal graph.  No result/query-ID cache is
used.  Every graph's complete membership is verified against a global raw
closed-window mask before any timed search.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ring.frans_v10.model import FRANS as _V10FRANS
from ring.frans_v9.model import Box, Node, query_boxes, query_vector_id


def raw_box_signature(raw: dict) -> tuple:
    """Hash the actual supplied predicate, not a query ID or prefilled gid."""
    dims = tuple(int(x) for x in raw.get("dims", ()))
    lower = tuple(float(x) for x in raw.get("lower", ()))
    upper = tuple(float(x) for x in raw.get("upper", ()))
    if not dims or dims != tuple(range(len(dims))) or len(lower) != len(dims) or len(upper) != len(dims):
        raise ValueError("V12 requires one window on the first m raw attributes")
    return dims, lower, upper


class FRANS(_V10FRANS):
    """Reuse immutable V10 construction, with a predicate-free single search."""

    def __init__(self, vectors: np.ndarray, attrs: np.ndarray, *, workers: int = 16,
                 leaf_size: int = 1024, candidate_size: int = 64, alpha: float = 1.2,
                 entries: int = 8, degree_cap: int = 32, seed: int = 20260918,
                 progress: Callable[[str, dict], None] | None = None):
        import _frans_v11

        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.attrs = np.ascontiguousarray(attrs, dtype=np.float64)
        if (self.vectors.ndim != 2 or self.attrs.ndim != 2 or
                len(self.attrs) != len(self.vectors) or self.attrs.shape[1] < 2):
            raise ValueError("aligned vectors and at least two raw attributes are required")
        if not len(self.vectors) or not np.isfinite(self.vectors).all() or not np.isfinite(self.attrs).all():
            raise ValueError("empty/nonfinite corpus")
        if not 1 <= workers <= 16 or min(leaf_size, candidate_size, entries, degree_cap) < 1 or alpha <= 1:
            raise ValueError("invalid construction configuration")
        self._native_module = _frans_v11
        self.native = _frans_v11.Corpus(self.vectors)
        self.members = np.arange(len(self.attrs), dtype=np.int64)
        self.active_mask = np.ones(len(self.attrs), dtype=np.bool_)
        self.domain_lower, self.domain_upper = self.attrs.min(axis=0), self.attrs.max(axis=0)
        self.domain = Box(self.domain_lower, self.domain_upper, tuple(range(self.attrs.shape[1])))
        self.workers, self.leaf_size = workers, leaf_size
        self.candidate_size, self.alpha, self.entries_count = candidate_size, alpha, entries
        self.degree_cap, self.seed = degree_cap, seed
        self.progress = progress or (lambda _stage, _details: None)
        self.nodes: list[Node] = []
        self.root: Node | None = None
        self.graphs: dict[str, object] = {}
        self.graph_members: dict[str, np.ndarray] = {}
        self.graph_native_ids: dict[str, int] = {}
        self.native_graph_keys: dict[int, str] = {}
        self.graph_bboxes: dict[str, Box] = {}
        self.graph_metadata: dict[str, dict] = {}
        self.presets: dict[str, list[int]] = {}
        self.range_directory: list[dict] = []
        self._range_hash: dict[tuple, list[dict]] = {}
        self.bridges: dict = {}
        self._bridge_edges: list = []
        self.search_index = None
        self.frozen = False
        self.receipt: dict = {}
        self.window_routes: dict[tuple, dict] = {}
        self.window_directory: list[dict] = []
        self.entry_mode = "all"

    def prepare_tree(self) -> list[Node]:
        """Geometry only: windows may be selected before any graph/search work."""
        if self.root is None:
            self.root = self._tree(self.members)
        return self.nodes

    def fit_windows(self, windows: list[dict]) -> dict:
        if self.frozen or self.search_index is not None:
            raise RuntimeError("one immutable construction per process")
        if not 1 <= len(windows) <= 160:
            raise ValueError("expected one to 160 offline windows across experimental cells")
        started = time.perf_counter()
        self.prepare_tree()
        self.search_index = self._native_module.SearchIndex(self.native, self.attrs)
        self.progress("base_start", {"members": len(self.members), "tree_nodes": len(self.nodes)})
        self._build_node(self.root)
        base_seconds = time.perf_counter() - started
        self.progress("base_complete", {"seconds": base_seconds})
        range_started = time.perf_counter()
        built = reused = 0
        for window in windows:
            raw = window["box"]
            signature = raw_box_signature(raw)
            box = query_boxes({"boxes": [raw]}, self.domain_lower, self.domain_upper)[0]
            # This offline whole-corpus check intentionally does not reuse a
            # window's claimed member IDs, a tree's membership, or truth IDs.
            members = np.flatnonzero(box.mask(self.attrs)).astype(np.int64)
            if len(members) < 10 or len(members) != int(window["legal_count"]):
                raise ValueError("frozen small window membership changed")
            key = self._member_key(members)
            existed = key in self.graphs
            source = self._base_containing(box)
            if not existed:
                graph = self.graphs[source.graph_id].extract_repair(members.tolist())
                key = self._register(graph, members)
                built += 1
                construction = "history_window_extract_repair_prune"
            else:
                reused += 1
                construction = "reused_base_graph"
            actual = np.sort(np.asarray(self.graphs[key].members, dtype=np.int64))
            if not np.array_equal(members, actual):
                raise RuntimeError("native graph membership != global raw-window mask")
            if not self._fully_inside([box], key):
                raise RuntimeError("graph bbox is not fully inside its exact raw window")
            if signature in self.window_routes:
                raise ValueError("duplicate raw-window route")
            route = {"window_id": window["window_id"], "graph_key": key,
                     "native_graph_id": self.graph_native_ids[key],
                     "selected_graph_size": len(members), "fully_inside": True,
                     "route": "raw_box_hash_exact_small_window", "construction": construction,
                     "source_base_node": source.id, "source_base_members": len(source.members),
                     "membership_sha256": window["membership_sha256"], "box": raw,
                     "selection_source": window["selection_source"]}
            self.window_routes[signature] = route
            self.window_directory.append(route)
            self.range_directory.append({"family": window["window_id"], "level": 0,
                                         "scale": 1.0, "box": box, "graph_id": key})
            self.progress("window_registered", route)
        self.search_index.set_bridges([], [], [])
        self.frozen = True
        self.receipt = {
            "method": "FRANS", "version": "v12", "native_module": "_frans_v11",
            "protocol": "FRANS_V12_EXACT_WINDOW_FIG23",
            "synthetic_targeted_mechanism_benchmark": True, "independent_holdout": False,
            "corpus_members": len(self.vectors), "active_members": len(self.members),
            "vector_dimension": self.vectors.shape[1],
            "window_attribute_counts": sorted({len(w["box"]["dims"]) for w in windows}),
            "offline_kd_attribute_count": self.attrs.shape[1],
            "base_tree_nodes": len(self.nodes), "unique_graphs": len(self.graphs),
            "window_count": len(windows), "extracted_window_graphs": built,
            "reused_window_graphs": reused, "degree_cap": self.degree_cap,
            "base_build_seconds": base_seconds,
            "window_extract_verify_seconds": time.perf_counter() - range_started,
            "total_build_learning_seconds": time.perf_counter() - started,
            "bridge_edges": 0, "bridge_learning_seconds": 0.,
            "parameters": {"leaf_size": self.leaf_size, "candidate_size": self.candidate_size,
                           "alpha": self.alpha, "entries": self.entries_count,
                           "build_workers": self.workers, "query_workers": 1,
                           "degree_cap": self.degree_cap, "seed": self.seed},
            "online_allowed_mask": False, "online_predicate_checks": 0,
            "online_bridges": False, "result_cache": False,
            "membership_verification": "all frozen windows: native members == full-corpus raw float64 closed-window mask",
            "online_routing": "actual raw box hash to exact graph; unmatched predicates fail closed",
            "search": "native single graph ANN; full graph membership legal, finite beam and approximate stop",
            "distance_kernel": self.search_index.distance_kernel,
            "estimated_graph_bytes": sum(self._graph_bytes(key) for key in self.graphs),
            "persistence": "in_process_only",
        }
        return self.receipt

    def search(self, record: dict, budget: int, *, entry_mode: str | None = None) -> dict:
        if not self.frozen:
            raise RuntimeError("freeze and verify windows before search")
        raw_boxes = record.get("boxes", [])
        if len(raw_boxes) != 1:
            raise ValueError("V12 exact-window path requires one raw box")
        route = self.window_routes.get(raw_box_signature(raw_boxes[0]))
        if route is None:
            raise ValueError("predicate is not an exactly registered small window")
        vector_id = query_vector_id(record)
        if not 0 <= vector_id < len(self.vectors):
            raise ValueError("query vector outside corpus")
        result = self.search_index.search_single(self.vectors[vector_id], route["native_graph_id"],
                                                k=10, beam=int(budget),
                                                entry_mode=entry_mode or self.entry_mode)
        counters = dict(result["counters"])
        if counters["predicate_checks"] or counters["bridge_starts"]:
            raise RuntimeError("single exact-graph path used predicates or bridges")
        return {"ids": result["ids"], "distances": result["distances"],
                "expanded": int(counters["expansions"]), "graphs_searched": 1,
                "range_graph_used": route["construction"] != "reused_base_graph",
                "bridge_starts": 0, "predicate_checks": 0,
                "distance_evaluations": int(counters["distance_evaluations"]),
                "stop_reason": counters["stop_reason"], "effective_beam": counters["effective_beam"],
                "selected_graph_size": route["selected_graph_size"], "fully_inside": True,
                "route": route["route"], "entry_mode": counters["entry_mode"],
                "native_graph_id": route["native_graph_id"],
                "entry_distance_evaluations": counters["entry_distance_evaluations"],
                "native_counters": counters}

    def directory_record(self) -> dict:
        return {"version": "v11", "windows": self.window_directory,
                "base_nodes": [{"node_id": n.id, "members": len(n.members),
                                "native_graph_id": self.graph_native_ids.get(n.graph_id)}
                               for n in self.nodes], "bridges": [],
                "graph_metadata": self.graph_metadata}


__all__ = ["FRANS", "raw_box_signature"]

