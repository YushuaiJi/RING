"""V9 FRANS orchestration; graph construction/search stay in ``_frans_v9``.

The finite-candidate native backend does not assert complete alpha coverage.
This first runner retains the index in one process; it does not serialize graphs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Callable

import numpy as np


@dataclass
class Box:
    lower: np.ndarray
    upper: np.ndarray
    dims: tuple[int, ...]

    def contains(self, other: "Box") -> bool:
        return bool(np.all(self.lower <= other.lower) and np.all(self.upper >= other.upper))

    def intersects(self, other: "Box") -> bool:
        return bool(np.all(self.lower <= other.upper) and np.all(other.lower <= self.upper))

    def mask(self, attrs: np.ndarray) -> np.ndarray:
        return np.all((attrs >= self.lower) & (attrs <= self.upper), axis=1)

    def as_dict(self) -> dict:
        return {"dims": list(self.dims), "lower": self.lower[list(self.dims)].tolist(),
                "upper": self.upper[list(self.dims)].tolist()}


@dataclass
class Node:
    id: int
    members: np.ndarray
    support: Box
    left: "Node | None" = None
    right: "Node | None" = None
    graph_id: str = ""


def query_boxes(record: dict, domain_lower: np.ndarray, domain_upper: np.ndarray) -> list[Box]:
    """Read raw closed boxes; an OR is never replaced by its enclosing box."""
    raw_boxes = record.get("boxes")
    if raw_boxes is None:
        raw_boxes = [record["window"]]
    boxes = []
    for raw in raw_boxes:
        lo, hi = domain_lower.copy(), domain_upper.copy()
        if "lower" in raw or "lo" in raw:
            lower = raw.get("lower", raw.get("lo"))
            upper = raw.get("upper", raw.get("hi"))
            dims = tuple(int(x) for x in raw.get("dims", range(len(lower))))
            if len(dims) != len(lower) or len(dims) != len(upper):
                raise ValueError("box dims/lower/upper lengths differ")
            for dim, a, b in zip(dims, lower, upper):
                if dim < 0 or dim >= len(lo):
                    raise ValueError("box attribute outside corpus")
                lo[dim], hi[dim] = max(lo[dim], float(a)), min(hi[dim], float(b))
        else:
            dims = tuple(sorted(int(x) for x in raw))
            for dim in dims:
                a, b = raw.get(str(dim), raw.get(dim))
                lo[dim], hi[dim] = max(lo[dim], float(a)), min(hi[dim], float(b))
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("nonfinite box")
        if np.all(lo <= hi):
            boxes.append(Box(lo, hi, dims))
    return boxes


def query_vector_id(record: dict) -> int:
    for key in ("query_vector_id", "source_id", "query_source_id", "source_vector_id"):
        if key in record:
            return int(record[key])
    raise ValueError("query must record its raw corpus query_vector_id")


def weighted_kmeans(features: np.ndarray, weights: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Small deterministic weighted Lloyd implementation; no sklearn runtime."""
    k = min(k, len(features))
    rng = np.random.default_rng(seed)
    centers = [features[int(rng.choice(len(features), p=weights / weights.sum()))]]
    for _ in range(1, k):
        dist = np.min(np.sum((features[:, None] - np.asarray(centers)[None, :]) ** 2, axis=2), axis=1)
        mass = dist * weights
        if mass.sum() == 0:
            break
        centers.append(features[int(rng.choice(len(features), p=mass / mass.sum()))])
    centers = np.asarray(centers)
    previous = None
    for _ in range(30):
        labels = np.argmin(np.sum((features[:, None] - centers[None, :]) ** 2, axis=2), axis=1)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
        for group in range(len(centers)):
            selected = labels == group
            if selected.any():
                centers[group] = np.average(features[selected], axis=0, weights=weights[selected])
    return labels


class FRANS:
    def __init__(self, vectors: np.ndarray, attrs: np.ndarray, *, workers: int = 16,
                 leaf_size: int = 1024, candidate_size: int = 64, alpha: float = 1.2,
                 entries: int = 8, clusters: int = 2, position_weight: float = 0.5,
                 gamma: float = 1.5, storage_fraction: float = 0.25, bridge_h: int = 4,
                 seed: int = 20260917, members: np.ndarray | None = None,
                 distance_kernel: str = "scalar",
                 progress: Callable[[str, dict], None] | None = None):
        import _frans_v9

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
        self.distance_kernel_requested = distance_kernel
        self.distance_kernel = ("avx2_batch4" if distance_kernel == "avx2_batch4" and
                                getattr(_frans_v9, "supports_avx2_batch4", lambda: False)() else "scalar")
        self.search_kernel_options = ({"distance_kernel": "avx2_batch4"}
                                      if self.distance_kernel == "avx2_batch4" else {})
        self.native = _frans_v9.Corpus(self.vectors)
        self.members = np.arange(len(attrs), dtype=np.int64) if members is None else np.unique(members).astype(np.int64)
        if not len(self.members) or self.members[0] < 0 or self.members[-1] >= len(attrs):
            raise ValueError("invalid active members")
        self.active_mask = np.zeros(len(attrs), dtype=np.bool_)
        self.active_mask[self.members] = True
        self.domain_lower, self.domain_upper = self.attrs.min(axis=0), self.attrs.max(axis=0)
        self.domain = Box(self.domain_lower, self.domain_upper, tuple(range(self.attrs.shape[1])))
        self.workers, self.leaf_size, self.candidate_size, self.alpha = workers, leaf_size, candidate_size, alpha
        self.entries_count, self.clusters, self.position_weight = entries, clusters, position_weight
        self.gamma, self.storage_fraction, self.bridge_h, self.seed = gamma, storage_fraction, bridge_h, seed
        self.progress = progress or (lambda _stage, _details: None)
        self.nodes: list[Node] = []
        self.graphs: dict[str, object] = {}
        self.graph_members: dict[str, np.ndarray] = {}
        self.presets: dict[str, list[int]] = {}
        self.range_directory: list[dict] = []
        self.bridges: dict[int, dict[str, list[int]]] = {}
        self.frozen = False
        self.receipt: dict = {}

    def _tree(self, members: np.ndarray, depth: int = 0) -> Node:
        values = self.attrs[members]
        node = Node(len(self.nodes), members,
                    Box(values.min(axis=0), values.max(axis=0), self.domain.dims))
        self.nodes.append(node)
        if len(members) <= self.leaf_size:
            return node
        # Keep equal attribute values on one side: closed-boundary ties cannot
        # disappear into a sibling excluded by routing.
        for offset in range(self.attrs.shape[1]):
            dim = (depth + offset) % self.attrs.shape[1]
            order = np.argsort(values[:, dim], kind="stable")
            sorted_values = values[order, dim]
            changes = np.flatnonzero(sorted_values[:-1] < sorted_values[1:]) + 1
            if not len(changes):
                continue
            cut = int(changes[np.argmin(np.abs(changes - len(members) / 2))])
            node.left = self._tree(members[order[:cut]], depth + 1)
            node.right = self._tree(members[order[cut:]], depth + 1)
            break
        return node

    @staticmethod
    def _member_key(members: np.ndarray) -> str:
        return hashlib.blake2b(np.sort(members).astype("<i8").tobytes(), digest_size=12).hexdigest()

    def _register(self, graph: object, members: np.ndarray) -> str:
        key = self._member_key(members)
        if key not in self.graphs:
            self.graphs[key] = graph
            self.graph_members[key] = np.sort(members)
            native_entries = list(graph.entries)
            sampled = self.graph_members[key][np.linspace(0, len(members) - 1,
                                                        min(self.entries_count, len(members)), dtype=int)]
            self.presets[key] = list(dict.fromkeys([int(v) for v in native_entries] + sampled.tolist()))[:self.entries_count]
        return key

    def _build_node(self, node: Node) -> object:
        if node.left is None:
            graph = self.native.build(node.members.tolist(), candidate_size=self.candidate_size,
                                      alpha=self.alpha, workers=self.workers)
        else:
            left, right = self._build_node(node.left), self._build_node(node.right)
            if not hasattr(left, "merge"):
                raise RuntimeError("V9 needs native Graph.merge: do not independently rebuild all KD nodes")
            graph = left.merge(right)
        node.graph_id = self._register(graph, node.members)
        if node.id == 0 or node.id % 16 == 0:
            self.progress("base_graph", {"node_id": node.id, "members": len(node.members),
                                        "built_graphs": len(self.graphs), "tree_nodes": len(self.nodes)})
        return self.graphs[node.graph_id]

    def _base_containing(self, box: Box) -> Node:
        node = self.root
        while node.left is not None:
            child = next((x for x in (node.left, node.right) if x.support.contains(box)), None)
            if child is None:
                break
            node = child
        return node

    def _containing_graph(self, box: Box) -> tuple[str, bool]:
        base = self._base_containing(box)
        best, count, learned = base.graph_id, len(base.members), False
        for row in self.range_directory:
            size = len(self.graph_members[row["graph_id"]])
            if size < count and row["box"].contains(box):
                best, count, learned = row["graph_id"], size, True
        return best, learned

    def _required(self, boxes: list[Box], mask: np.ndarray) -> list[str]:
        required = []

        def visit(node: Node) -> None:
            if not any(box.intersects(node.support) for box in boxes):
                return
            if any(box.contains(node.support) for box in boxes) or node.left is None:
                if np.any(mask[node.members]):
                    required.append(node.graph_id)
                return
            visit(node.left)
            visit(node.right)

        visit(self.root)
        return list(dict.fromkeys(required))

    def allowed_mask(self, boxes: list[Box]) -> np.ndarray:
        # query_boxes fills undeclared dimensions with the corpus domain, so
        # only the declared dimensions can reject a corpus object. Column
        # comparisons avoid N-by-M broadcasts and their per-row reduction.
        # Native search accepts uint8 directly, avoiding a fresh bool cast for
        # every local graph in a multi-graph request.
        mask = np.zeros(len(self.attrs), dtype=np.bool_)
        selected = np.empty(len(self.attrs), dtype=np.bool_)
        comparison = np.empty(len(self.attrs), dtype=np.bool_)
        for box in boxes:
            selected.fill(1)
            for dim in box.dims:
                np.greater_equal(self.attrs[:, dim], box.lower[dim], out=comparison)
                np.logical_and(selected, comparison, out=selected)
                np.less_equal(self.attrs[:, dim], box.upper[dim], out=comparison)
                np.logical_and(selected, comparison, out=selected)
            np.logical_or(mask, selected, out=mask)
        np.logical_and(mask, self.active_mask, out=mask)
        return mask.view(np.uint8)

    def _graph_bytes(self, key: str) -> int:
        # Explicit compact encoding estimate, not a claim about allocator/RSS.
        return int(self.graphs[key].edge_count) * 4 + (len(self.graph_members[key]) + 1) * 8 + len(self.graph_members[key]) * 4

    def _learn_ranges(self, learning: list[dict]) -> dict:
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
            low = np.asarray([(history[i]["box"].lower[dims] - self.domain_lower[dims]) / span for i in indices])
            high = np.asarray([(history[i]["box"].upper[dims] - self.domain_lower[dims]) / span for i in indices])
            centers, widths = (low + high) / 2, high - low
            weights = np.asarray([history[i]["weight"] for i in indices])
            features = np.concatenate((math.sqrt(self.position_weight) * centers,
                                       math.sqrt(1 - self.position_weight) * widths), axis=1)
            labels = weighted_kmeans(features, weights, self.clusters, self.seed + group_number)
            for label in np.unique(labels):
                chosen = labels == label
                center = np.average(centers[chosen], axis=0, weights=weights[chosen])
                radius = np.average(widths[chosen], axis=0, weights=weights[chosen]) / 2
                needed = np.max(np.maximum(abs(low[chosen] - center), abs(high[chosen] - center)) / radius, axis=1)
                scale, largest = float(needed.min()), float(needed.max())
                levels = [scale]
                while levels[-1] < largest:
                    levels.append(levels[-1] * self.gamma)
                # Only levels selected as the smallest containing historical
                # range are candidates; no minimum-frequency cutoff.
                selected_levels = set(np.searchsorted(np.asarray(levels), needed, side="left").tolist())
                for level in sorted(selected_levels):
                    lo, hi = self.domain_lower.copy(), self.domain_upper.copy()
                    lo[dims] = self.domain_lower[dims] + np.maximum(0, center - levels[level] * radius) * span
                    hi[dims] = self.domain_lower[dims] + np.minimum(1, center + levels[level] * radius) * span
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
        degree = sum(int(g.edge_count) for g in self.graphs.values()) / max(1, sum(len(m) for m in self.graph_members.values()))
        best_counts = np.asarray([row["base_count"] for row in history], dtype=float)
        weights = np.asarray([row["weight"] for row in history])
        selected, existing, estimate = [], set(self.graphs), 0
        remaining = candidates.copy()
        while remaining:
            choices = []
            for row in remaining:
                cost = directory_bytes if row["key"] in existing else directory_bytes + int(len(row["members"]) * (degree * 4 + 12) + 8)
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
        for row in sorted(selected, key=lambda item: (item["family"], item["level"])):
            key, members = row["key"], row["members"]
            fresh = key not in self.graphs
            if not fresh:
                graph = self.graphs[key]
            elif row["family"] in previous:
                graph = self.graphs[previous[row["family"]]].expand(members.tolist())
            else:
                source, _ = self._containing_graph(row["box"])
                graph = self.graphs[source].extract_repair(members.tolist())
            cost = directory_bytes + (int(graph.edge_count) * 4 + (len(members) + 1) * 8 + len(members) * 4 if fresh else 0)
            if used + cost > cap:
                skipped += 1
                continue
            key = self._register(graph, members)
            self.range_directory.append({"family": row["family"], "level": row["level"],
                                         "scale": row["scale"], "box": row["box"], "graph_id": key})
            previous[row["family"]] = key
            used += cost
            built += int(fresh)
            reused += int(not fresh)
        return {"history_windows": len(history), "candidate_levels": len(candidates),
                "selected_estimated_levels": len(selected), "stored_levels": len(self.range_directory),
                "new_graphs": built, "shared_graph_levels": reused, "actual_size_rejected_levels": skipped,
                "base_adjacency_directory_estimated_bytes": base_bytes,
                "range_storage_budget_bytes": cap, "range_adjacency_directory_estimated_bytes": used,
                "range_selection_seconds": construction_start - start,
                "range_extract_expand_seconds": time.perf_counter() - construction_start,
                "storage_accounting": "uint32_edges+uint64_offsets+uint32_members+directory_estimate; not RSS"}

    def _nearest(self, q: np.ndarray, ids: list[int]) -> list[int]:
        if not ids:
            return []
        ids_array = np.asarray(list(dict.fromkeys(ids)), dtype=np.int64)
        diff = self.vectors[ids_array].astype(np.float64) - np.asarray(q, dtype=np.float64)
        distances = np.einsum("ij,ij->i", diff, diff)
        return ids_array[np.lexsort((ids_array, distances))].tolist()

    def search(self, record: dict, budget: int, *, use_bridges: bool = True,
               collect_history: bool = False) -> dict:
        boxes = query_boxes(record, self.domain_lower, self.domain_upper)
        q = self.vectors[query_vector_id(record)]
        mask = self.allowed_mask(boxes)
        if not boxes or not mask.any():
            return {"ids": [], "distances": [], "expanded": 0, "graphs_searched": 0,
                    "range_graph_used": False, "bridge_starts": 0, "local_records": []}
        if len(boxes) == 1:
            key, learned = self._containing_graph(boxes[0])
            result = self.graphs[key].search(q, k=10, budget=budget,
                                            entry_ids=self._nearest(q, self.presets[key]),
                                            allowed_mask=mask, trace=collect_history, **self.search_kernel_options)
            counters = dict(result.get("counters", {}))
            expanded = len(result.get("expanded_ids", []))
            expanded = int(counters.get("expanded", counters.get("expansions", expanded)))
            return {"ids": list(result["ids"]), "distances": list(result["distances"]),
                    "expanded": expanded, "graphs_searched": 1, "range_graph_used": learned,
                    "bridge_starts": 0, "local_records": []}
        required = self._required(boxes, mask)
        # The query and preset vectors are immutable during this request.
        # Cache only preset ordering, not masks, answers, or future bridge entries.
        preset_orders = {key: self._nearest(q, self.presets[key]) for key in required}
        remaining = set(required)
        pending: dict[str, set[int]] = defaultdict(set)
        distances: dict[int, float] = {}
        local_records, expanded, bridge_starts = [], 0, 0
        while remaining:
            reached = [(key, self._nearest(q, list(pending[key]))) for key in sorted(remaining) if pending[key]]
            options = reached if reached else [(key, preset_orders[key]) for key in sorted(remaining)]
            firsts = [ids[0] for _, ids in options]
            nearest = self._nearest(q, firsts)[0]
            key, starts = next((key, ids) for key, ids in options if ids[0] == nearest)
            bridge_starts += min(self.entries_count, len(starts)) if reached else 0
            starts = list(dict.fromkeys(starts + preset_orders[key]))[:self.entries_count]
            # Complete this local call before choosing another required graph.
            result = self.graphs[key].search(q, k=10, budget=budget, entry_ids=starts,
                                            allowed_mask=mask, trace=True, **self.search_kernel_options)
            actual_expanded = [int(x) for x in result.get("expanded_ids", [])]
            expanded += len(actual_expanded)
            remaining.remove(key)
            for obj, distance in zip(result["ids"], result["distances"]):
                distances[int(obj)] = float(distance)
            if collect_history:
                local_records.append({"graph_id": key, "last_expanded": actual_expanded[-1] if actual_expanded else None,
                                      "ids": [int(x) for x in result["ids"][:self.bridge_h]]})
            if use_bridges:
                for source in actual_expanded:
                    for target, targets in self.bridges.get(source, {}).items():
                        if target in remaining:
                            pending[target].update(targets)
        ranked = sorted(distances, key=lambda obj: (distances[obj], obj))[:10]
        return {"ids": ranked, "distances": [distances[obj] for obj in ranked], "expanded": expanded,
                "graphs_searched": len(required), "range_graph_used": False,
                "bridge_starts": bridge_starts, "local_records": local_records}

    def fit(self, learning: list[dict], *, history_budget: int = 128) -> dict:
        if self.frozen:
            raise RuntimeError("index already frozen; evaluation must not relearn")
        start = time.perf_counter()
        self.root = self._tree(self.members)
        self.progress("base_start", {"active_members": len(self.members), "tree_nodes": len(self.nodes)})
        self._build_node(self.root)
        base_seconds = time.perf_counter() - start
        self.progress("base_complete", {"seconds": base_seconds})
        ranges = self._learn_ranges(learning)
        self.progress("ranges_complete", ranges)
        bridge_start = time.perf_counter()
        pools: dict[tuple[int, str], set[int]] = defaultdict(set)
        historical_requests, transitions = 0, 0
        for record in learning:
            if len(query_boxes(record, self.domain_lower, self.domain_upper)) < 2:
                continue
            historical_requests += 1
            traces = self.search(record, history_budget, use_bridges=False, collect_history=True)["local_records"]
            for left, right in zip(traces, traces[1:]):
                source = left["last_expanded"]
                if source is not None and right["ids"]:
                    pools[(source, right["graph_id"])].update(right["ids"])
                    transitions += 1
        for (source, target), candidates in pools.items():
            retained = self.native.alpha_prune(int(source), sorted(candidates), alpha=self.alpha)
            self.bridges.setdefault(int(source), {})[target] = [int(x) for x in retained]
        self.frozen = True
        self.receipt = {"method": "FRANS", "version": "v9", "active_members": len(self.members),
                        "corpus_members": len(self.vectors), "vector_dimension": self.vectors.shape[1],
                        "attribute_count": self.attrs.shape[1], "base_tree_nodes": len(self.nodes),
                        "unique_graphs": len(self.graphs), "base_build_seconds": base_seconds,
                        **ranges, "bridge_learning_seconds": time.perf_counter() - bridge_start,
                        "history_or_requests": historical_requests, "observed_transitions": transitions,
                        "bridge_candidate_pools": len(pools), "bridge_edges": sum(len(ids) for targets in self.bridges.values() for ids in targets.values()),
                        "total_build_learning_seconds": time.perf_counter() - start,
                        "parameters": {"leaf_size": self.leaf_size, "candidate_size": self.candidate_size,
                                       "alpha": self.alpha, "entries": self.entries_count, "clusters": self.clusters,
                                       "position_weight": self.position_weight, "gamma": self.gamma,
                                       "storage_fraction": self.storage_fraction, "bridge_h": self.bridge_h,
                                       "history_budget": history_budget, "build_workers": self.workers,
                                       "query_workers": 1, "seed": self.seed,
                                       "distance_kernel": self.distance_kernel,
                                       "distance_kernel_requested": self.distance_kernel_requested},
                        "entry_order_cache": "per_request_preset_order_only; no answer caching",
                        "seed_rule": "native entries, fill with evenly spaced existing member IDs; no vector centroid scan",
                        "base_construction": "KD median distinct-value split; native leaf build; bottom-up merge reuses child edges",
                        "range_construction": "extract_repair smallest stored level; expand successive stored levels",
                        "or_routing": "complete KD decomposition; per-required-graph budget; finish each local search before next",
                        "distance_path": "float32 stored vectors; float64 accumulated squared L2",
                        "guarantee": "finite ANN candidates/finite search budgets; no complete alpha-coverage or recall guarantee",
                        "persistence": "in_process_only; graph serialization is not implemented",
                        "baselines": "not connected in this runner"}
        self.progress("fit_complete", self.receipt)
        return self.receipt

    def directory_record(self) -> dict:
        return {"ranges": [{**{k: v for k, v in row.items() if k != "box"}, "box": row["box"].as_dict(),
                            "members": len(self.graph_members[row["graph_id"]])} for row in self.range_directory],
                "bridges": {str(source): targets for source, targets in self.bridges.items()},
                "base_nodes": [{"node_id": node.id, "graph_id": node.graph_id, "members": len(node.members),
                                "left": None if node.left is None else node.left.id,
                                "right": None if node.right is None else node.right.id,
                                "support": node.support.as_dict()} for node in self.nodes],
                "graph_metadata": {key: dict(graph.metadata) for key, graph in self.graphs.items()}}
