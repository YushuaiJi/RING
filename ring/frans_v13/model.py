"""Familiar-window OR search with frozen, history-only cross-graph bridges."""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from ring.frans_v12.model import FRANS as WindowFRANS, raw_box_signature
from ring.frans_v9.model import query_boxes, query_vector_id


class FRANS(WindowFRANS):
    def search(self, record: dict, budget: int, *, use_bridges: bool = True,
               collect_history: bool = False) -> dict:
        if not self.frozen:
            raise RuntimeError("build exact windows before searching")
        raw = record.get("boxes", [])
        if len(raw) not in (2, 3):
            raise ValueError("Figure4 requires two or three familiar windows")
        routes = [self.window_routes.get(raw_box_signature(box)) for box in raw]
        if any(route is None for route in routes):
            raise ValueError("unregistered window; no fallback or online construction")
        keys = list(dict.fromkeys(route["graph_key"] for route in routes))
        boxes = query_boxes(record, self.domain_lower, self.domain_upper)
        result = self._native_search(self.vectors[query_vector_id(record)], boxes, keys,
                                    int(budget), use_bridges=use_bridges,
                                    collect_history=collect_history)
        if result["predicate_checks"] or not all(result["fully_inside"]):
            raise RuntimeError("exact window route unexpectedly used attribute filtering")
        result.update(selected_graph_size=sum(len(self.graph_members[k]) for k in keys),
                      route="raw_box_hash_exact_multi_window", range_graph_used=True)
        return result

    def learn_bridges(self, history: list[dict], *, history_budget: int = 128,
                      bridge_h: int = 4) -> dict:
        """V10 observed transition policy, with V9's four target candidates.

        Only measured history search traces are consumed.  No truth IDs or
        evaluation rows participate. Bridges are published once afterwards.
        """
        if getattr(self, "bridges_frozen", False):
            raise RuntimeError("history is already frozen")
        started = time.perf_counter()
        pools = defaultdict(set)
        transitions = 0
        for i, row in enumerate(history):
            trace = self.search(row, history_budget, use_bridges=False, collect_history=True)
            targets = {r["graph_id"]: r["ids"][:bridge_h] for r in trace["local_records"]}
            states = trace["expanded_states"]
            for (gid, source), (dest_gid, _) in zip(states, states[1:]):
                if gid != dest_gid and targets.get(dest_gid):
                    pools[(source, dest_gid)].update(targets[dest_gid])
                    transitions += 1
            if (i + 1) % 200 == 0:
                self.progress("history_progress", {"done": i + 1, "total": len(history)})
        edges = []
        for (source, dest_gid), candidates in sorted(pools.items()):
            kept = [int(x) for x in self.native.alpha_prune(source, sorted(candidates), alpha=self.alpha)]
            self.bridges.setdefault(source, {})[dest_gid] = kept
            edges.extend((source, dest_gid, target) for target in kept)
        self.search_index.set_bridges([e[0] for e in edges], [e[1] for e in edges],
                                      [e[2] for e in edges])
        self._bridge_edges = edges
        self.bridges_frozen = True
        details = {"version": "v13", "protocol": "FRANS_V13_FAMILIAR_MULTIRANGE",
                   "history_queries": len(history), "history_budget": history_budget,
                   "bridge_h": bridge_h, "history_transitions": transitions,
                   "bridge_candidate_pools": len(pools), "bridge_edges": len(edges),
                   "bridge_learning_seconds": time.perf_counter() - started,
                   "test_used_for_bridges": False,
                   "search": "native global beam across exact graphs; history bridges and independent entries",
                   "online_bridges": "traverse frozen bridges; no online bridge construction",
                   "budget": "global beam=max(10,B); total expansion cap=max(64,8B,4*range_count)",
                   "entry_mode": "all offline entries of selected graphs",
                   "workload_scope": "synthetic familiar-window combinations; new vectors, not unseen-range holdout"}
        self.receipt.update(details)
        self.receipt["total_build_learning_seconds"] += details["bridge_learning_seconds"]
        return self.receipt

    def directory_record(self) -> dict:
        out = super().directory_record()
        out.update(version="v13", bridges=self._bridge_edges)
        return out
