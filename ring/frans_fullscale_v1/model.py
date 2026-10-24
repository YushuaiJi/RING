"""Full-corpus V12/V13 semantics with reusable, incremental offline snapshots."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from ring.frans_v12.model import FRANS as WindowFRANS, raw_box_signature
from ring.frans_v13.model import FRANS as MultiFRANS
from ring.frans_v14.single_model import FRANS as AblationFRANS
from ring.frans_v9.model import Node, query_boxes
from ring.frans_resources_v15.frans_adapter import _box_state, _restore_box, _jsonable


FORMAT = "frans_fullscale_v1_shared_corpus_graph_objects"
PARAMETERS = dict(leaf_size=1024, candidate_size=64, alpha=1.2,
                  entries=8, degree_cap=32, seed=20260918)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_identity(path):
    path = Path(path).resolve()
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return dict(path=str(path), bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns, shape=list(array.shape), dtype=str(array.dtype))


class FullIndex(WindowFRANS):
    """Only persistence and incremental preparation change; search is inherited."""

    def __init__(self, vectors, attrs, **kwargs):
        super().__init__(vectors, attrs, **kwargs)
        if getattr(self._native_module, "fullscale_lookup_version", None) != 1:
            raise RuntimeError("load cpp/frans_fullscale_v1 native before the source root")
        self.base_graph_keys = frozenset()
        self.bridges_frozen = False

    def fit_base(self):
        if self.search_index is not None:
            raise RuntimeError("base already built/loaded")
        started = time.perf_counter()
        self.prepare_tree()
        self.search_index = self._native_module.SearchIndex(self.native, self.attrs)
        self.progress("base_start", dict(members=len(self.vectors), tree_nodes=len(self.nodes)))
        self._build_node(self.root)
        elapsed = time.perf_counter() - started
        self.base_graph_keys = frozenset(node.graph_id for node in self.nodes)
        self.search_index.set_bridges([], [], [])
        self.frozen = True
        self.receipt = dict(method="FRANS", version="fullscale_v1", corpus_members=len(self.vectors),
                            vector_dimension=self.vectors.shape[1], attribute_count=self.attrs.shape[1],
                            base_tree_nodes=len(self.nodes), base_build_seconds=elapsed,
                            window_extract_verify_seconds=0.0, total_build_learning_seconds=elapsed,
                            bridge_learning_seconds=0.0, bridge_edges=0,
                            window_count=0, extracted_window_graphs=0, reused_window_graphs=0,
                            unique_graphs=len(self.graphs), parameters=self.parameters(),
                            persistence=FORMAT, native_module_file=self._native_module.__file__,
                            native_source_revision=getattr(self._native_module, "fullscale_revision", ""))
        self.progress("base_complete", self.receipt)
        return self.receipt

    def parameters(self):
        return dict(leaf_size=self.leaf_size, candidate_size=self.candidate_size,
                    alpha=self.alpha, entries=self.entries_count, degree_cap=self.degree_cap,
                    seed=self.seed, workers=self.workers)

    def add_windows(self, windows):
        if self.search_index is None:
            self.fit_base()
        started = time.perf_counter()
        built = reused = 0
        for window in windows:
            raw = window["box"]
            signature = raw_box_signature(raw)
            if signature in self.window_routes:
                continue
            box = query_boxes({"boxes": [raw]}, self.domain_lower, self.domain_upper)[0]
            members = np.flatnonzero(box.mask(self.attrs)).astype(np.int64)
            if len(members) < 10 or len(members) != int(window["legal_count"]):
                raise ValueError("window does not match full-corpus membership")
            key = self._member_key(members)
            source = self._base_containing(box)
            if key not in self.graphs:
                graph = self.graphs[source.graph_id].extract_repair(members.tolist())
                key = self._register(graph, members)
                built += 1
                construction = "new_window_extract_repair_prune"
            else:
                reused += 1
                construction = "reused_base_graph" if key in self.base_graph_keys else "reused_window_graph"
            if not np.array_equal(members, self.graph_members[key]) or not self._fully_inside([box], key):
                raise RuntimeError("registered graph is not exactly the legal set")
            route = dict(window_id=window["window_id"], graph_key=key,
                         native_graph_id=self.graph_native_ids[key], selected_graph_size=len(members),
                         fully_inside=True, route="raw_box_hash_exact_window", construction=construction,
                         source_base_node=source.id, source_base_members=len(source.members),
                         membership_sha256=window["membership_sha256"], box=raw,
                         selection_source=window.get("selection_source", "new_full_corpus_geometry"))
            self.window_routes[signature] = route
            self.window_directory.append(route)
            self.range_directory.append(dict(family=window["window_id"], level=0, scale=1., box=box, graph_id=key))
            self.progress("window_registered", route)
        elapsed = time.perf_counter() - started
        self._rebuild_range_index()
        self.receipt["window_extract_verify_seconds"] += elapsed
        self.receipt["total_build_learning_seconds"] += elapsed
        self.receipt["extracted_window_graphs"] += built
        self.receipt["reused_window_graphs"] += reused
        self.receipt.update(window_count=len(self.window_directory), unique_graphs=len(self.graphs))
        self.prepared_only_graph_keys = frozenset(self.graphs) - self.base_graph_keys
        return self.receipt

    def fit_windows(self, windows):
        self.fit_base()
        return self.add_windows(windows)

    def search(self, record, budget, *, entry_mode=None, prepared_windows=True,
               use_bridges=True, collect_history=False):
        if len(record.get("boxes", [])) > 1:
            return MultiFRANS.search(self, record, budget, use_bridges=use_bridges,
                                     collect_history=collect_history)
        if not prepared_windows:
            return AblationFRANS.search(self, record, budget, prepared_windows=False, entry_mode=entry_mode)
        return WindowFRANS.search(self, record, budget, entry_mode=entry_mode)

    learn_bridges = MultiFRANS.learn_bridges

    def save(self, root, phase, *, vector_path, attr_path):
        root = Path(root).resolve()
        snapshot = root / "snapshots" / (phase + ".json")
        if snapshot.exists():
            raise FileExistsError(f"immutable snapshot already exists: {snapshot}")
        started = time.perf_counter()
        objects = root / "graphs"
        objects.mkdir(parents=True, exist_ok=True)
        graphs = []
        for key in sorted(self.graphs, key=self.graph_native_ids.__getitem__):
            file = objects / (key + ".bin")
            if not file.exists():
                temporary = file.with_suffix(".bin.tmp")
                self.graphs[key].save(str(temporary))
                temporary.replace(file)
            graphs.append(dict(key=key, native_graph_id=self.graph_native_ids[key],
                               file=str(file.relative_to(root)), bytes=file.stat().st_size,
                               entries=self.presets[key], bbox=_box_state(self.graph_bboxes[key]),
                               metadata=self.graph_metadata[key]))
        state = dict(format=FORMAT, phase=phase, parameters=self.parameters(), receipt=self.receipt,
                     corpus=dict(vectors=array_identity(vector_path), attrs=array_identity(attr_path)),
                     graphs=graphs, root_id=self.root.id, entry_mode=self.entry_mode,
                     nodes=[dict(id=n.id, graph_key=n.graph_id, support=_box_state(n.support),
                                 left=n.left.id if n.left else None, right=n.right.id if n.right else None)
                            for n in self.nodes], window_directory=self.window_directory,
                     range_directory=[{**r, "box": _box_state(r["box"])} for r in self.range_directory],
                     bridge_edges=self._bridge_edges, bridges_frozen=self.bridges_frozen)
        state["save_seconds"] = time.perf_counter() - started
        state["graph_disk_bytes"] = sum(g["bytes"] for g in graphs)
        state["external_corpus_bytes"] = sum(v["bytes"] for v in state["corpus"].values())
        write_json(snapshot, state)
        self.progress("snapshot_saved", dict(phase=phase, path=str(snapshot), graphs=len(graphs),
                                               seconds=state["save_seconds"]))
        return snapshot

    @classmethod
    def load(cls, root, phase, *, progress=None):
        root = Path(root).resolve()
        state = json.loads((root / "snapshots" / (phase + ".json")).read_text())
        if state["format"] != FORMAT:
            raise ValueError("not a fullscale snapshot")
        for info in state["corpus"].values():
            if array_identity(info["path"]) != info:
                raise ValueError("external corpus changed; cannot reuse saved index")
        vectors = np.load(state["corpus"]["vectors"]["path"], mmap_mode="r", allow_pickle=False)
        attrs = np.load(state["corpus"]["attrs"]["path"], mmap_mode="r", allow_pickle=False)
        model = cls(vectors, attrs, progress=progress, **state["parameters"])
        model.search_index = model._native_module.SearchIndex(model.native, model.attrs)
        for record in state["graphs"]:
            path = root / record["file"]
            if path.stat().st_size != record["bytes"]:
                raise ValueError(f"incomplete saved graph: {path}")
            key = record["key"]
            graph = model._native_module.Graph.load(model.native, str(path))
            gid = int(model.search_index.add_graph(graph, record["entries"]))
            if gid != record["native_graph_id"]:
                raise ValueError("saved graph ordering changed")
            model.graphs[key] = graph
            model.graph_members[key] = np.asarray(graph.members, dtype=np.int64)
            model.graph_native_ids[key], model.native_graph_keys[gid] = gid, key
            model.graph_bboxes[key], model.graph_metadata[key] = _restore_box(record["bbox"]), record["metadata"]
            model.presets[key] = record["entries"]
        by_id = {}
        for row in state["nodes"]:
            node = Node(row["id"], model.graph_members[row["graph_key"]], _restore_box(row["support"]),
                        graph_id=row["graph_key"])
            model.nodes.append(node)
            by_id[node.id] = node
        for row in state["nodes"]:
            by_id[row["id"]].left = by_id.get(row["left"])
            by_id[row["id"]].right = by_id.get(row["right"])
        model.root = by_id[state["root_id"]]
        model.base_graph_keys = frozenset(n.graph_id for n in model.nodes)
        model.prepared_only_graph_keys = frozenset(model.graphs) - model.base_graph_keys
        model.window_directory = state["window_directory"]
        model.window_routes = {raw_box_signature(r["box"]): r for r in model.window_directory}
        model.range_directory = [{**r, "box": _restore_box(r["box"])} for r in state["range_directory"]]
        model._rebuild_range_index()
        model._bridge_edges = state["bridge_edges"]
        for source, dest, target in model._bridge_edges:
            model.bridges.setdefault(source, {}).setdefault(dest, []).append(target)
        model.search_index.set_bridges([e[0] for e in model._bridge_edges],
                                      [e[1] for e in model._bridge_edges],
                                      [e[2] for e in model._bridge_edges])
        model.bridges_frozen = state["bridges_frozen"]
        model.entry_mode, model.receipt = state["entry_mode"], state["receipt"]
        model.frozen = True
        return model
