"""FRANS V10 orchestration over the ``_frans_v10`` native seam.

The V10 implementation deliberately keeps the V9 tree, range-learning policy,
and small graph-directory routing in Python, but makes graph registration and
multi-graph search scheduling deep operations of ``SearchIndex``.  In
particular, object eligibility is passed as raw attribute bounds to native
search; Python never materialises an all-member mask on the request path.

This module is intentionally in-process.  It is an engineering variant for
the V10 rerun, not a second query runner or a graph serialization format.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Callable

import numpy as np

from ring.frans_v9.model import (
    Box,
    Node,
    FRANS as _V9FRANS,
    query_boxes,
    query_vector_id,
    weighted_kmeans,
)


class FRANS(_V9FRANS):
    """FRANS with degree-capped graphs and a native global-beam index.

    The public model interface remains compatible with V9 ``fit`` and
    ``search`` records.  The native interface is the seam: callers provide a
    query vector plus raw closed boxes, and receive top-k results/counters;
    Python selects the small set of tree/directory graph IDs, while lazy
    boundary filtering, bridge starts, and global beam scheduling remain
    inside ``_frans_v10.SearchIndex``.
    """

    def __init__(self, vectors: np.ndarray, attrs: np.ndarray, *, workers: int = 16,
                 leaf_size: int = 1024, candidate_size: int = 64, alpha: float = 1.2,
                 entries: int = 8, clusters: int = 2, position_weight: float = 0.5,
                 gamma: float = 1.5, storage_fraction: float = 0.25, bridge_h: int = 4,
                 degree_cap: int = 32, seed: int = 20260917,
                 members: np.ndarray | None = None,
                 distance_kernel: str = "scalar",
                 progress: Callable[[str, dict], None] | None = None):
        import _frans_v10

        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.attrs = np.ascontiguousarray(attrs, dtype=np.float64)
        if self.vectors.ndim != 2 or self.attrs.ndim != 2 or len(self.attrs) != len(self.vectors):
            raise ValueError("vectors/attributes must be aligned two-dimensional arrays")
        if not np.isfinite(self.attrs).all() or not np.isfinite(self.vectors).all():
            raise ValueError("corpus contains nonfinite values")
        if not 0 < position_weight < 1 or gamma <= 1 or min(leaf_size, entries, clusters) < 1:
            raise ValueError("invalid model configuration")
        if not 1 <= workers <= 16 or not 0 <= storage_fraction:
            raise ValueError("workers must be 1..16 and storage fraction nonnegative")
        if distance_kernel not in ("scalar", "avx2_batch4"):
            raise ValueError("unsupported distance kernel")
        if int(degree_cap) < 1:
            raise ValueError("degree_cap must be positive")

        self._native_module = _frans_v10
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        self.attrs = np.ascontiguousarray(self.attrs, dtype=np.float64)
        self.distance_kernel_requested = distance_kernel
        supports_avx2 = getattr(_frans_v10, "supports_avx2_batch4", lambda: False)()
        self.distance_kernel = ("avx2_batch4" if distance_kernel == "avx2_batch4" and supports_avx2
                                else "scalar")
        # SearchIndex owns the distance path; this field is retained for run
        # config compatibility, but no per-query option is passed to native.
        self.search_kernel_options: dict = {}
        self.native = _frans_v10.Corpus(self.vectors)

        self.members = (np.arange(len(attrs), dtype=np.int64) if members is None
                        else np.unique(members).astype(np.int64))
        if not len(self.members) or self.members[0] < 0 or self.members[-1] >= len(attrs):
            raise ValueError("invalid active members")
        self.active_mask = np.zeros(len(attrs), dtype=np.bool_)
        self.active_mask[self.members] = True
        self.domain_lower = self.attrs.min(axis=0)
        self.domain_upper = self.attrs.max(axis=0)
        self.domain = Box(self.domain_lower, self.domain_upper, tuple(range(self.attrs.shape[1])))

        self.workers, self.leaf_size = workers, leaf_size
        self.candidate_size, self.alpha = candidate_size, alpha
        self.entries_count, self.clusters = entries, clusters
        self.position_weight = position_weight
        self.gamma, self.storage_fraction = gamma, storage_fraction
        self.bridge_h, self.degree_cap, self.seed = bridge_h, int(degree_cap), seed
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
        self.bridges: dict[int, dict[int, list[int]]] = {}
        self._bridge_edges: list[tuple[int, int, int]] = []
        self.search_index = None
        self.frozen = False
        self.receipt: dict = {}

    @staticmethod
    def _edge_count(graph: object) -> int:
        return int(getattr(graph, "edge_count", 0))

    @staticmethod
    def _safe_metadata(graph: object) -> dict:
        raw = getattr(graph, "metadata", {})
        try:
            return dict(raw)
        except (TypeError, ValueError):
            return {"native_metadata_repr": repr(raw)}

    @staticmethod
    def _box_signature(box: Box) -> tuple:
        return (tuple(int(x) for x in box.dims),
                tuple(float(x) for x in np.asarray(box.lower)),
                tuple(float(x) for x in np.asarray(box.upper)))

    def _register(self, graph: object, members: np.ndarray) -> str:
        """Prune once, register once, and retain the actual graph bbox.

        The returned Python key is used by the KD/range code; the native gid
        is kept separately so no Python object scan is needed during search.
        Parent graphs are therefore merged from the already-pruned child
        graphs returned by ``_build_node``.
        """
        if self.search_index is None:
            raise RuntimeError("SearchIndex must exist before graph registration")
        pruned = graph.pruned(self.degree_cap)
        key = self._member_key(members)
        if key in self.graphs:
            return key

        member_ids = np.sort(np.asarray(members, dtype=np.int64))
        entries = [int(x) for x in getattr(pruned, "entries", [])]
        sampled = member_ids[np.linspace(0, len(member_ids) - 1,
                                        min(self.entries_count, len(member_ids)), dtype=int)]
        entries = list(dict.fromkeys(entries + [int(x) for x in sampled]))[:self.entries_count]
        native_gid = int(self.search_index.add_graph(pruned, entries))

        values = self.attrs[member_ids]
        bbox = Box(np.asarray(values.min(axis=0), dtype=np.float64),
                   np.asarray(values.max(axis=0), dtype=np.float64), self.domain.dims)
        metadata = self._safe_metadata(pruned)
        metadata.update({"degree_cap": self.degree_cap,
                         "member_count": int(len(member_ids)),
                         "native_graph_id": native_gid})
        self.graphs[key] = pruned
        self.graph_members[key] = member_ids
        self.graph_native_ids[key] = native_gid
        self.native_graph_keys[native_gid] = key
        self.graph_bboxes[key] = bbox
        self.graph_metadata[key] = metadata
        self.presets[key] = entries
        return key

    def _build_node(self, node: Node) -> object:
        if node.left is None:
            graph = self.native.build(node.members.tolist(), candidate_size=self.candidate_size,
                                      alpha=self.alpha, workers=self.workers)
        else:
            left, right = self._build_node(node.left), self._build_node(node.right)
            if not hasattr(left, "merge"):
                raise RuntimeError("V10 needs native Graph.merge")
            # left/right are the pruned child graphs returned by _register.
            graph = left.merge(right)
        node.graph_id = self._register(graph, node.members)
        if node.id == 0 or node.id % 16 == 0:
            self.progress("base_graph", {"node_id": node.id, "members": len(node.members),
                                          "built_graphs": len(self.graphs),
                                          "tree_nodes": len(self.nodes),
                                          "degree_cap": self.degree_cap})
        return self.graphs[node.graph_id]

    def _graph_bytes(self, key: str) -> int:
        return (self._edge_count(self.graphs[key]) * 4 +
                (len(self.graph_members[key]) + 1) * 8 + len(self.graph_members[key]) * 4)

    def _rebuild_range_index(self) -> None:
        directory: dict[tuple, list[dict]] = defaultdict(list)
        for row in self.range_directory:
            directory[self._box_signature(row["box"])].append(row)
        self._range_hash = dict(directory)
        # The directory is deliberately small and static.  This order makes
        # containment lookup choose the smallest complete learned range first.
        self.range_directory.sort(key=lambda row: (len(self.graph_members[row["graph_id"]]),
                                                   row["family"], row["level"]))

    def _containing_graph(self, box: Box) -> tuple[str, bool]:
        """Use exact-range lookup then a tiny bbox directory, never objects."""
        base = self._base_containing(box)
        best, count, learned = base.graph_id, len(base.members), False
        candidates = list(self._range_hash.get(self._box_signature(box), ()))
        candidates.extend(self.range_directory)
        seen: set[int] = set()
        for row in candidates:
            marker = id(row)
            if marker in seen:
                continue
            seen.add(marker)
            key = row["graph_id"]
            size = len(self.graph_members[key])
            if size < count and row["box"].contains(box):
                best, count, learned = key, size, True
        return best, learned

    def _required(self, boxes: list[Box]) -> list[str]:
        """Return KD graphs whose support intersects the OR, without masks."""
        required: list[str] = []

        def visit(node: Node) -> None:
            if not any(box.intersects(node.support) for box in boxes):
                return
            if any(box.contains(node.support) for box in boxes) or node.left is None:
                if node.graph_id:
                    required.append(node.graph_id)
                return
            visit(node.left)
            visit(node.right)

        if self.root is not None:
            visit(self.root)
        return list(dict.fromkeys(required))

    def allowed_mask(self, boxes: list[Box]) -> np.ndarray:
        """Reject the removed V10 online all-member-mask path explicitly."""
        raise RuntimeError("V10 uses SearchIndex raw bounds; allowed_mask is not available")

    def _fully_inside(self, boxes: list[Box], key: str) -> bool:
        bbox = self.graph_bboxes[key]
        return any(box.contains(bbox) for box in boxes)

    @staticmethod
    def _record_value(result: object, key: str, default):
        if isinstance(result, dict):
            return result.get(key, default)
        return getattr(result, key, default)

    def _trace_records(self, result: object, collect_history: bool) -> list[dict]:
        if not collect_history:
            return []
        raw_records = self._record_value(result, "graph_records", []) or []
        records = []
        for raw in raw_records:
            if isinstance(raw, dict):
                gid = raw.get("graph_id")
                last = raw.get("last_expanded")
                ids = raw.get("ids", [])
            else:
                gid = getattr(raw, "graph_id", None)
                last = getattr(raw, "last_expanded", None)
                ids = getattr(raw, "ids", [])
            if gid is None:
                continue
            records.append({"graph_id": int(gid),
                            "last_expanded": None if last is None else int(last),
                            "ids": [int(x) for x in np.asarray(ids).reshape(-1).tolist()]})
        return records

    def _expanded_states(self, result: object, collect_history: bool) -> list[tuple[int, int]]:
        """Normalize native's real cross-graph expansion trace.

        ``graph_records`` is a per-graph summary and is not sufficient to
        infer bridge transitions.  Native additionally returns the ordered
        ``(graph_id, object_id)`` states; fit consumes this sequence directly.
        """
        if not collect_history:
            return []
        raw_states = self._record_value(result, "expanded_states", []) or []
        states = []
        for raw in raw_states:
            try:
                graph_id, object_id = raw
            except (TypeError, ValueError):
                continue
            states.append((int(graph_id), int(object_id)))
        return states

    def _native_search(self, q: np.ndarray, boxes: list[Box], keys: list[str], budget: int,
                       *, use_bridges: bool, collect_history: bool) -> dict:
        if self.search_index is None:
            raise RuntimeError("fit must build SearchIndex before search")
        if not keys:
            return {"ids": [], "distances": [], "expanded": 0, "graphs_searched": 0,
                    "bridge_starts": 0, "distance_evaluations": 0, "predicate_checks": 0,
                    "stop_reason": "no_required_graphs", "effective_beam": int(budget),
                    "local_records": [], "native_counters": {}}
        lower = np.ascontiguousarray(np.stack([box.lower for box in boxes]), dtype=np.float64)
        upper = np.ascontiguousarray(np.stack([box.upper for box in boxes]), dtype=np.float64)
        native_gids = [self.graph_native_ids[key] for key in keys]
        fully_inside = [self._fully_inside(boxes, key) for key in keys]
        result = self.search_index.search(np.ascontiguousarray(q, dtype=np.float32), native_gids,
                                          lower, upper, fully_inside, k=10,
                                          beam=int(budget), max_expansions=0,
                                          use_bridges=bool(use_bridges), trace=bool(collect_history))
        ids = self._record_value(result, "ids", []) or []
        distances = self._record_value(result, "distances", []) or []
        counters = dict(self._record_value(result, "counters", {}) or {})
        records = self._trace_records(result, collect_history)
        expanded_states = self._expanded_states(result, collect_history)
        expanded = int(counters.get("expansions", counters.get("expanded", 0)))
        if not expanded and records:
            expanded = sum(1 for _ in self._record_value(result, "expanded_states", []) or [])
        graphs_searched = int(counters.get("graphs_searched", len(native_gids)))
        bridge_starts = int(counters.get("bridge_starts", 0))
        distance_evaluations = int(counters.get("distance_evaluations",
                                                  counters.get("distance_evals", 0)))
        predicate_checks = int(counters.get("predicate_checks",
                                             counters.get("predicate_evaluations", 0)))
        stop_reason = self._record_value(result, "stop_reason",
                                         counters.get("stop_reason", ""))
        effective_beam = int(counters.get("effective_beam", budget))
        effective_max_expansions = int(counters.get("effective_max_expansions", 0))
        return {"ids": [int(x) for x in np.asarray(ids).reshape(-1).tolist()],
                "distances": [float(x) for x in np.asarray(distances).reshape(-1).tolist()],
                "expanded": expanded, "graphs_searched": graphs_searched,
                "bridge_starts": bridge_starts, "distance_evaluations": distance_evaluations,
                "predicate_checks": predicate_checks, "stop_reason": stop_reason,
                "effective_beam": effective_beam,
                "effective_max_expansions": effective_max_expansions,
                "local_records": records, "expanded_states": expanded_states,
                "native_counters": counters, "graph_ids": native_gids,
                "fully_inside": fully_inside}

    def search(self, record: dict, budget: int, *, use_bridges: bool = True,
               collect_history: bool = False) -> dict:
        boxes = query_boxes(record, self.domain_lower, self.domain_upper)
        if not boxes:
            return {"ids": [], "distances": [], "expanded": 0, "graphs_searched": 0,
                    "range_graph_used": False, "bridge_starts": 0,
                    "distance_evaluations": 0, "predicate_checks": 0,
                    "stop_reason": "empty_predicate", "effective_beam": int(budget),
                    "local_records": []}
        vector_id = query_vector_id(record)
        if vector_id < 0 or vector_id >= len(self.vectors):
            raise ValueError("query_vector_id outside corpus")
        q = self.vectors[vector_id]
        if len(boxes) == 1:
            key, learned = self._containing_graph(boxes[0])
            result = self._native_search(q, boxes, [key], budget,
                                         use_bridges=use_bridges,
                                         collect_history=collect_history)
            result["range_graph_used"] = learned
            return result

        keys = self._required(boxes)
        result = self._native_search(q, boxes, keys, budget,
                                     use_bridges=use_bridges,
                                     collect_history=collect_history)
        result["range_graph_used"] = False
        return result

    def _learn_ranges(self, learning: list[dict]) -> dict:
        """Reuse V9 range selection, publishing the directory only when done."""
        start = time.perf_counter()
        history = []
        for record in learning:
            weight = float(record.get("occurrence", 1))
            if weight <= 0:
                raise ValueError("history occurrence must be positive")
            for box in query_boxes(record, self.domain_lower, self.domain_upper):
                dims = list(box.dims)
                widths = box.upper[dims] - box.lower[dims]
                spans = self.domain_upper[dims] - self.domain_lower[dims]
                if np.any(widths <= 0) or np.any(spans <= 0):
                    continue
                history.append({"box": box, "weight": weight,
                                "base_count": len(self._base_containing(box).members)})

        grouped = defaultdict(list)
        for i, row in enumerate(history):
            grouped[row["box"].dims].append(i)
        candidates = []
        for group_number, (dims_tuple, indices) in enumerate(sorted(grouped.items())):
            dims = list(dims_tuple)
            span = self.domain_upper[dims] - self.domain_lower[dims]
            low = np.asarray([(history[i]["box"].lower[dims] - self.domain_lower[dims]) / span
                              for i in indices])
            high = np.asarray([(history[i]["box"].upper[dims] - self.domain_lower[dims]) / span
                               for i in indices])
            centers, widths = (low + high) / 2, high - low
            weights = np.asarray([history[i]["weight"] for i in indices])
            features = np.concatenate((math.sqrt(self.position_weight) * centers,
                                       math.sqrt(1 - self.position_weight) * widths), axis=1)
            labels = weighted_kmeans(features, weights, self.clusters, self.seed + group_number)
            for label in np.unique(labels):
                chosen = labels == label
                center = np.average(centers[chosen], axis=0, weights=weights[chosen])
                radius = np.average(widths[chosen], axis=0, weights=weights[chosen]) / 2
                safe_radius = np.where(radius > 0, radius, 1.0)
                needed = np.max(np.maximum(abs(low[chosen] - center),
                                           abs(high[chosen] - center)) / safe_radius, axis=1)
                scale, largest = float(needed.min()), float(needed.max())
                levels = [scale]
                while levels[-1] < largest:
                    levels.append(levels[-1] * self.gamma)
                selected_levels = set(np.searchsorted(np.asarray(levels), needed, side="left").tolist())
                for level in sorted(selected_levels):
                    lo, hi = self.domain_lower.copy(), self.domain_upper.copy()
                    lo[dims] = self.domain_lower[dims] + np.maximum(
                        0, center - levels[level] * radius) * span
                    hi[dims] = self.domain_lower[dims] + np.minimum(
                        1, center + levels[level] * radius) * span
                    box = Box(lo, hi, dims_tuple)
                    members = self.members[box.mask(self.attrs[self.members])]
                    if not len(members):
                        continue
                    coverage = np.asarray([i for i in indices if box.contains(history[i]["box"])], dtype=int)
                    if not len(coverage):
                        continue
                    candidates.append({"family": f"{group_number}:{int(label)}", "level": level,
                                       "scale": levels[level], "box": box, "members": members,
                                       "coverage": coverage, "key": self._member_key(members)})

        base_bytes = sum(self._graph_bytes(key) for key in self.graphs)
        directory_bytes = 64 + self.attrs.shape[1] * 16
        base_bytes += len(self.nodes) * directory_bytes
        cap = int(base_bytes * self.storage_fraction)
        total_members = max(1, sum(len(m) for m in self.graph_members.values()))
        degree = sum(self._edge_count(g) for g in self.graphs.values()) / total_members
        best_counts = np.asarray([row["base_count"] for row in history], dtype=float)
        weights = np.asarray([row["weight"] for row in history])
        selected, existing, estimate = [], set(self.graphs), 0
        remaining = candidates.copy()
        while remaining:
            choices = []
            for row in remaining:
                cost = directory_bytes if row["key"] in existing else (
                    directory_bytes + int(len(row["members"]) * (degree * 4 + 12) + 8))
                covered = row["coverage"]
                gain = float(np.sum(np.maximum(0, best_counts[covered] - len(row["members"])) * weights[covered]))
                if gain > 0 and estimate + cost <= cap:
                    choices.append((gain / max(1, cost), gain, cost, row))
            if not choices:
                break
            _, _, cost, row = max(choices, key=lambda item: (item[0], item[1]))
            remaining = [candidate for candidate in remaining if candidate is not row]
            selected.append(row)
            existing.add(row["key"])
            estimate += cost
            best_counts[row["coverage"]] = np.minimum(best_counts[row["coverage"]], len(row["members"]))

        construction_start = time.perf_counter()
        used, built, reused, skipped = 0, 0, 0, 0
        previous = {}
        pending_directory: list[dict] = []
        # Do not expose partially built range rows to _containing_graph.  A
        # family can use its immediately prior graph; all other sources fall
        # back to the immutable base KD directory during offline learning.
        for row in sorted(selected, key=lambda item: (item["family"], item["level"])):
            key, members = row["key"], row["members"]
            fresh = key not in self.graphs
            if not fresh:
                graph = self.graphs[key]
            elif row["family"] in previous:
                graph = self.graphs[previous[row["family"]]].expand(members.tolist())
            else:
                source = self._base_containing(row["box"]).graph_id
                graph = self.graphs[source].extract_repair(members.tolist())
            cost = directory_bytes + (self._edge_count(graph) * 4 + (len(members) + 1) * 8
                                      + len(members) * 4 if fresh else 0)
            if used + cost > cap:
                skipped += 1
                continue
            key = self._register(graph, members)
            pending_directory.append({"family": row["family"], "level": row["level"],
                                      "scale": row["scale"], "box": row["box"], "graph_id": key})
            previous[row["family"]] = key
            used += cost
            built += int(fresh)
            reused += int(not fresh)

        self.range_directory = pending_directory
        self._rebuild_range_index()
        return {"history_windows": len(history), "candidate_levels": len(candidates),
                "selected_estimated_levels": len(selected), "stored_levels": len(self.range_directory),
                "new_graphs": built, "shared_graph_levels": reused,
                "actual_size_rejected_levels": skipped,
                "base_adjacency_directory_estimated_bytes": base_bytes,
                "range_storage_budget_bytes": cap,
                "range_adjacency_directory_estimated_bytes": used,
                "range_selection_seconds": construction_start - start,
                "range_extract_expand_seconds": time.perf_counter() - construction_start,
                "storage_accounting": "uint32_edges+uint64_offsets+uint32_members+directory_estimate; not RSS"}

    def fit(self, learning: list[dict], *, history_budget: int = 128) -> dict:
        if self.frozen:
            raise RuntimeError("index already frozen; evaluation must not relearn")
        start = time.perf_counter()
        self.root = self._tree(self.members)
        self.progress("base_start", {"active_members": len(self.members),
                                      "tree_nodes": len(self.nodes)})
        # The index is created before graph registration so every graph gets a
        # native gid at registration time, and before historical learning.
        self.search_index = self._native_module.SearchIndex(self.native, self.attrs)
        self._build_node(self.root)
        base_seconds = time.perf_counter() - start
        self.progress("base_complete", {"seconds": base_seconds})
        ranges = self._learn_ranges(learning)
        self.progress("ranges_complete", ranges)

        bridge_start = time.perf_counter()
        pools: dict[tuple[int, int], set[int]] = defaultdict(set)
        historical_requests, transitions = 0, 0
        for record in learning:
            if len(query_boxes(record, self.domain_lower, self.domain_upper)) < 2:
                continue
            historical_requests += 1
            trace = self.search(record, history_budget, use_bridges=False,
                                collect_history=True)
            traces = trace["local_records"]
            states = trace.get("expanded_states", [])
            target_ids = {int(row["graph_id"]): row["ids"] for row in traces}
            # Use the real ordered expansion states, not graph_records' final
            # last_expanded summaries, to identify cross-graph bridge sources.
            for (source_gid, source_obj), (target_gid, _target_obj) in zip(states, states[1:]):
                if int(source_gid) == int(target_gid):
                    continue
                candidates = target_ids.get(int(target_gid), [])
                if candidates:
                    pools[(int(source_obj), int(target_gid))].update(candidates)
                    transitions += 1

        source_ids: list[int] = []
        destination_gids: list[int] = []
        target_ids: list[int] = []
        for (source, target_gid), candidates in pools.items():
            retained = self.native.alpha_prune(int(source), sorted(candidates), alpha=self.alpha)
            values = [int(x) for x in retained]
            self.bridges.setdefault(int(source), {})[int(target_gid)] = values
            for target in values:
                source_ids.append(int(source))
                destination_gids.append(int(target_gid))
                target_ids.append(int(target))
        # Freeze the complete flat bridge table once; no request can mutate it.
        self.search_index.set_bridges(source_ids, destination_gids, target_ids)
        self._bridge_edges = list(zip(source_ids, destination_gids, target_ids))
        self.frozen = True
        bridge_seconds = time.perf_counter() - bridge_start
        bridge_edges = len(target_ids)
        self.receipt = {
            "method": "FRANS",
            "version": "v10",
            "engineering_variant": "degree-capped-SearchIndex-global-beam",
            "native_module": "_frans_v10",
            "active_members": len(self.members),
            "corpus_members": len(self.vectors),
            "vector_dimension": self.vectors.shape[1],
            "attribute_count": self.attrs.shape[1],
            "base_tree_nodes": len(self.nodes),
            "unique_graphs": len(self.graphs),
            "native_graphs": len(self.graph_native_ids),
            "degree_cap": self.degree_cap,
            "base_build_seconds": base_seconds,
            **ranges,
            "bridge_learning_seconds": bridge_seconds,
            "history_or_requests": historical_requests,
            "observed_transitions": transitions,
            "bridge_candidate_pools": len(pools),
            "bridge_edges": bridge_edges,
            "total_build_learning_seconds": time.perf_counter() - start,
            "parameters": {
                "leaf_size": self.leaf_size, "candidate_size": self.candidate_size,
                "alpha": self.alpha, "entries": self.entries_count,
                "clusters": self.clusters, "position_weight": self.position_weight,
                "gamma": self.gamma, "storage_fraction": self.storage_fraction,
                "bridge_h": self.bridge_h, "history_budget": history_budget,
                "build_workers": self.workers, "query_workers": 1,
                "seed": self.seed, "degree_cap": self.degree_cap,
                "distance_kernel": self.distance_kernel,
                "distance_kernel_requested": self.distance_kernel_requested,
            },
            "entry_order_cache": "native SearchIndex entries; no request object scan",
            "base_construction": "KD median distinct-value split; native leaf build; pruned child merge",
            "range_construction": "extract_repair smallest stored level; expand successive stored levels",
            "range_directory": "exact bounds hash plus static graph-bbox containment directory",
            "or_routing": "tree-support intersection/full-containment graph set; one native global beam",
            "online_filtering": "raw float64 bounds; fully_inside graph bbox flags; native lazy boundary filter",
            "online_allowed_mask": False,
            "global_beam": True,
            "vector_dtype": "float32",
            "attribute_dtype": "float64",
            "distance_path": "_frans_v10 SearchIndex; native float32 squared-L2",
            "all_members_preserved": True,
            "bridge_freeze": "one flat set_bridges call after historical learning",
            "guarantee": "finite ANN candidates/finite search budgets; no complete alpha-coverage or recall guarantee",
            "persistence": "in_process_only; graph serialization is not implemented",
        }
        self.progress("fit_complete", self.receipt)
        return self.receipt

    def directory_record(self) -> dict:
        return {
            "version": "v10",
            "degree_cap": self.degree_cap,
            "ranges": [{**{k: v for k, v in row.items() if k != "box"},
                         "box": row["box"].as_dict(),
                         "members": len(self.graph_members[row["graph_id"]]),
                         "native_graph_id": self.graph_native_ids[row["graph_id"]],
                         "bbox": self.graph_bboxes[row["graph_id"]].as_dict()}
                        for row in self.range_directory],
            "bridges": {str(source): {str(target): ids for target, ids in targets.items()}
                        for source, targets in self.bridges.items()},
            "base_nodes": [{"node_id": node.id, "graph_id": node.graph_id,
                            "native_graph_id": self.graph_native_ids.get(node.graph_id),
                            "members": len(node.members),
                            "left": None if node.left is None else node.left.id,
                            "right": None if node.right is None else node.right.id,
                            "support": node.support.as_dict()}
                           for node in self.nodes],
            "graph_metadata": self.graph_metadata,
            "bridge_edges": [{"source": source, "destination_graph_id": destination,
                              "target": target}
                             for source, destination, target in self._bridge_edges],
        }


__all__ = ["Box", "Node", "FRANS", "query_boxes", "query_vector_id", "weighted_kmeans"]
