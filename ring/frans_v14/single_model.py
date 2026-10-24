"""Prepared-window ablation on one immutable V11 index.

The on arm is the unchanged V11 exact-window request path. The off arm only
routes through the original KD tree: maximal wholly covered subtrees plus
intersecting boundary leaves. It never consults prepared-window routes when
selecting its graphs, materializes an online object mask, or rebuilds graphs.
An exact one-graph KD cover keeps the same all-legal single-graph fast path.
"""
from __future__ import annotations

import numpy as np

from ring.frans_v11.model import FRANS as _WindowFRANS, raw_box_signature
from ring.frans_v9.model import query_boxes, query_vector_id


class FRANS(_WindowFRANS):
    """Fit once, then toggle only availability of extra prepared-window graphs.

    ``prepared_windows=True`` uses the original V11 path. ``False`` keeps
    every original KD graph, including any graph whose membership already
    exactly matches a frozen window. Boundary covers use the existing native
    filtered global-beam path with bridges disabled, not independent searches
    with a separate budget per graph.

    Both paths preserve the requested ``all`` or ``nearest2`` entry policy.
    The independent cpp/frans_v14 snapshot still imports as ``_frans_v11``;
    it adds only optional multi-graph entry selection to the original kernel.
    """

    def fit_windows(self, windows: list[dict]) -> dict:
        source_revision = getattr(self._native_module, "source_revision", "")
        if source_revision != "frans_v14_v11_snapshot_multigraph_entry_modes":
            raise RuntimeError("prepend the cpp/frans_v14 build directory to PYTHONPATH "
                               "before importing _frans_v11 for this ablation")
        receipt = super().fit_windows(windows)
        # Node graph IDs refer to the base graphs built before window extract/
        # repair. Reused exact-window graphs therefore remain in the off arm.
        self.base_graph_keys = frozenset(node.graph_id for node in self.nodes)
        if not self.base_graph_keys or not self.base_graph_keys.issubset(self.graphs):
            raise RuntimeError("base KD graph registry is incomplete")
        self.prepared_only_graph_keys = frozenset(self.graphs) - self.base_graph_keys
        receipt["native_source_revision"] = source_revision
        receipt["native_module_file"] = self._native_module.__file__
        receipt["native_source_directory"] = "cpp/frans_v14"
        receipt["prepared_window_ablation"] = {
            "same_immutable_index": True,
            "base_graphs_retained": len(self.base_graph_keys),
            "extra_prepared_graphs": len(self.prepared_only_graph_keys),
            "off_routing": "KD wholly-covered subtrees and intersecting boundary leaves",
            "off_exact_base_fast_path": "same native search_single and entry_mode",
            "off_boundary_search": "native global beam; lazy predicates; same entry mode; no bridges",
            "online_object_mask": False,
            "query_result_cache": False,
            "budget": "same requested global beam; never a beam per selected graph",
        }
        return receipt

    def search(self, record: dict, budget: int, *, prepared_windows: bool = True,
               entry_mode: str | None = None) -> dict:
        if not self.frozen:
            raise RuntimeError("fit and freeze windows before either ablation arm")
        mode = entry_mode or self.entry_mode
        if mode not in ("all", "nearest2"):
            raise ValueError("entry_mode must be all or nearest2")
        if int(budget) < 1:
            raise ValueError("beam must be positive")
        if prepared_windows:
            result = super().search(record, budget, entry_mode=mode)
            result.update(prepared_windows=True,
                          graph_ids=[result["native_graph_id"]],
                          per_graph_fully_inside=[True])
            return result

        raw = record.get("boxes", [])
        if len(raw) != 1:
            raise ValueError("Figure 1 requires exactly one raw window")
        # Validate the same raw attribute schema without looking up a prepared
        # window graph, a query ID, or a cached query route.
        _, lower, upper = raw_box_signature(raw[0])
        if (not np.isfinite(lower).all() or not np.isfinite(upper).all() or
                any(a > b for a, b in zip(lower, upper))):
            raise ValueError("raw window bounds must be finite and lower<=upper")
        vector_id = query_vector_id(record)
        if not 0 <= vector_id < len(self.vectors):
            raise ValueError("query vector outside corpus")
        boxes = query_boxes(record, self.domain_lower, self.domain_upper)
        # Inherited V10 KD covering descends only intersecting branches; it
        # stops at fully covered subtrees or boundary leaves. No root fallback.
        keys = self._required(boxes)
        if any(key not in self.base_graph_keys for key in keys):
            raise RuntimeError("KD-only arm selected an extra prepared graph")
        inside = [self._fully_inside(boxes, key) for key in keys]
        native_ids = [self.graph_native_ids[key] for key in keys]
        selected_size = sum(len(self.graph_members[key]) for key in keys)

        if len(keys) == 1 and inside[0]:
            # The complete KD cover contains one wholly legal graph: all legal
            # corpus objects are in it. This is the same precondition used by
            # V11 search_single, established by routing rather than a mask.
            raw_result = self.search_index.search_single(
                self.vectors[vector_id], native_ids[0], k=10,
                beam=int(budget), entry_mode=mode)
            counters = dict(raw_result["counters"])
            result = {
                "ids": raw_result["ids"], "distances": raw_result["distances"],
                "expanded": int(counters["expansions"]),
                "graphs_searched": int(counters["graphs_searched"]),
                "bridge_starts": int(counters["bridge_starts"]),
                "predicate_checks": int(counters["predicate_checks"]),
                "distance_evaluations": int(counters["distance_evaluations"]),
                "stop_reason": counters["stop_reason"],
                "effective_beam": counters["effective_beam"],
                "entry_mode": counters["entry_mode"],
                "entry_distance_evaluations": counters["entry_distance_evaluations"],
                "native_counters": counters,
                "route": "base_kd_exact_single_graph",
            }
        else:
            bounds_lower = np.ascontiguousarray(
                [box.lower for box in boxes], dtype=np.float64).reshape(-1, self.attrs.shape[1])
            bounds_upper = np.ascontiguousarray(
                [box.upper for box in boxes], dtype=np.float64).reshape(-1, self.attrs.shape[1])
            raw_result = self.search_index.search(
                self.vectors[vector_id], native_ids, bounds_lower, bounds_upper, inside,
                k=10, beam=int(budget), max_expansions=0, use_bridges=False,
                trace=False, entry_mode=mode)
            counters = dict(raw_result["counters"])
            result = {
                "ids": raw_result["ids"], "distances": raw_result["distances"],
                "expanded": int(counters["expansions"]),
                "graphs_searched": int(counters["graphs_searched"]),
                "bridge_starts": int(counters["bridge_starts"]),
                "predicate_checks": int(counters["predicate_checks"]),
                "distance_evaluations": int(counters["distance_evaluations"]),
                "stop_reason": counters["stop_reason"],
                "effective_beam": counters["effective_beam"],
                "effective_max_expansions": counters["effective_max_expansions"],
                "entry_mode": counters["entry_mode"],
                "entry_distance_evaluations": counters["entry_distance_evaluations"],
                "native_counters": counters,
                "route": "base_kd_cover_native_filter",
            }
        result.update(prepared_windows=False, range_graph_used=False,
                      selected_graph_size=selected_size,
                      fully_inside=bool(keys) and all(inside),
                      per_graph_fully_inside=inside, graph_ids=native_ids,
                      native_graph_id=native_ids[0] if len(native_ids) == 1 else None)
        return result


__all__ = ["FRANS"]
