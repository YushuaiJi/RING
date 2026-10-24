"""V11-faithful FRANS resource build and complete, loadable persistence.

Use the separately built cpp/frans_resource_v15 overlay on PYTHONPATH. The
frozen query path and all graph parameters stay in ring.frans_v11.model.
Build timing includes Corpus/geometry construction; saving is a separate stage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ring.frans_v11.model import FRANS, raw_box_signature
from ring.frans_v9.model import Box, Node


FORMAT = "frans_resource_v15_complete_v1"
PARAMETERS = {"leaf_size": 1024, "candidate_size": 64, "alpha": 1.2,
              "entries": 8, "degree_cap": 32, "seed": 20260918}


def _native():
    import _frans_v11
    if getattr(_frans_v11, "resource_persistence_version", None) != 15:
        raise RuntimeError("put cpp/frans_resource_v15/build first on PYTHONPATH; "
                           "the old V11 module has no complete persistence")
    return _frans_v11


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _box_state(box: Box) -> dict:
    # Unlike Box.as_dict(), retain all raw bounds, including inactive dimensions.
    return {"lower": box.lower.tolist(), "upper": box.upper.tolist(),
            "dims": list(box.dims)}


def _restore_box(state: dict) -> Box:
    return Box(np.asarray(state["lower"], dtype=np.float64),
               np.asarray(state["upper"], dtype=np.float64), tuple(state["dims"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FRANSResourceIndex(FRANS):
    """Original V11 query API, backed by real persisted adjacency and metadata."""

    @classmethod
    def build(cls, vectors, attrs, windows, workers: int = 16, *, progress=None):
        native = _native()
        native.reset_construction_parallel_stats()
        started = time.perf_counter()
        model = cls(vectors, attrs, workers=workers, progress=progress, **PARAMETERS)
        model.fit_windows(windows)
        model.build_seconds = time.perf_counter() - started
        model.receipt.update({
            "resource_revision": "v15_v11_semantics_complete_persistence",
            "build_seconds": model.build_seconds,
            "build_timing_scope": "after_input_load_before_Corpus_init_through_fit_windows",
            "persistence": FORMAT,
            "construction_parallel_stats": dict(native.construction_parallel_stats()),
            "additional_parallel_phases": ["merge_inherited_rows", "extract_retained_rows"],
            "parallel_semantics": "disjoint_output_rows; original candidate/prune/order rules unchanged",
            "build_workers": int(workers),
            "native_module_file": native.__file__,
        })
        return model

    def save(self, path) -> dict:
        """Write one full index into a new/empty directory; never estimate bytes."""
        if not self.frozen:
            raise RuntimeError("build/freeze before save")
        started = time.perf_counter()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(f"index directory is not empty: {destination}")
        graph_dir = destination / "graphs"
        graph_dir.mkdir()
        np.save(destination / "vectors.npy", self.vectors, allow_pickle=False)
        np.save(destination / "attrs.npy", self.attrs, allow_pickle=False)
        graph_records = []
        for key in sorted(self.graphs, key=self.graph_native_ids.__getitem__):
            gid = self.graph_native_ids[key]
            filename = f"graphs/{gid:06d}.bin"
            self.graphs[key].save(str(destination / filename))
            graph_records.append({"key": key, "native_graph_id": gid, "file": filename,
                                  "entries": self.presets[key],
                                  "bbox": _box_state(self.graph_bboxes[key]),
                                  "metadata": self.graph_metadata[key]})
        manifest = {
            "format": FORMAT, "parameters": PARAMETERS, "workers": self.workers,
            "build_receipt": self.receipt, "entry_mode": self.entry_mode,
            "vectors": "vectors.npy", "attrs": "attrs.npy", "graphs": graph_records,
            "root_id": self.root.id,
            "nodes": [{"id": node.id, "graph_key": node.graph_id,
                       "member_count": len(node.members), "support": _box_state(node.support),
                       "left": node.left.id if node.left else None,
                       "right": node.right.id if node.right else None}
                      for node in self.nodes],
            "window_directory": self.window_directory,
            "range_directory": [{**row, "box": _box_state(row["box"])}
                                for row in self.range_directory],
            "bridges": [], "bridge_edges": [],
            "reconstructible_not_stored": ["dense_global_to_local_maps", "query_epoch_scratch",
                                            "corpus_projection_scores", "node_member_order"],
        }
        manifest["payloads"] = [
            {"file": str(file.relative_to(destination)), "bytes": file.stat().st_size,
             "sha256": _sha256(file)}
            for file in sorted(destination.rglob("*")) if file.is_file()]
        with (destination / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(manifest), stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.save_receipt = {
            "format": FORMAT, "save_seconds": time.perf_counter() - started,
            "disk_bytes": sum(file.stat().st_size for file in destination.rglob("*") if file.is_file()),
            "disk_scope": "vectors_once_attrs_once_all_graph_payloads_tree_routes_entries_manifest",
            "serialization_in_build_seconds": False, "path": str(destination.resolve()),
            "graph_count": len(graph_records),
        }
        return dict(self.save_receipt)

    @classmethod
    def load(cls, path):
        """Restore all stored graphs and routing; no graph construction or repair."""
        native = _native()
        started = time.perf_counter()
        source = Path(path)
        with (source / "manifest.json").open(encoding="utf-8") as stream:
            state = json.load(stream)
        if state["format"] != FORMAT or state["parameters"] != PARAMETERS:
            raise ValueError("unsupported FRANS persisted index format/configuration")
        for payload in state["payloads"]:
            file = source / payload["file"]
            if file.stat().st_size != payload["bytes"] or _sha256(file) != payload["sha256"]:
                raise ValueError(f"saved payload checksum mismatch: {payload['file']}")
        vectors = np.load(source / state["vectors"], allow_pickle=False)
        attrs = np.load(source / state["attrs"], allow_pickle=False)
        model = cls(vectors, attrs, workers=int(state["workers"]), **state["parameters"])
        model.search_index = native.SearchIndex(model.native, model.attrs)
        for record in sorted(state["graphs"], key=lambda row: row["native_graph_id"]):
            key = record["key"]
            graph = native.Graph.load(model.native, str(source / record["file"]))
            members = np.asarray(graph.members, dtype=np.int64)
            if model._member_key(members) != key:
                raise ValueError("loaded graph membership key mismatch")
            gid = int(model.search_index.add_graph(graph, record["entries"]))
            if gid != record["native_graph_id"]:
                raise ValueError("native graph ordering changed on load")
            model.graphs[key] = graph
            model.graph_members[key] = members
            model.graph_native_ids[key] = gid
            model.native_graph_keys[gid] = key
            model.graph_bboxes[key] = _restore_box(record["bbox"])
            model.graph_metadata[key] = record["metadata"]
            model.presets[key] = record["entries"]
        by_id = {}
        for record in state["nodes"]:
            members = model.graph_members[record["graph_key"]]
            if len(members) != record["member_count"]:
                raise ValueError("saved node membership count mismatch")
            node = Node(record["id"], members, _restore_box(record["support"]),
                        graph_id=record["graph_key"])
            model.nodes.append(node)
            by_id[node.id] = node
        for record in state["nodes"]:
            node = by_id[record["id"]]
            node.left = by_id.get(record["left"])
            node.right = by_id.get(record["right"])
        model.root = by_id[state["root_id"]]
        model.window_directory = state["window_directory"]
        model.window_routes = {raw_box_signature(route["box"]): route
                               for route in model.window_directory}
        model.range_directory = [{**row, "box": _restore_box(row["box"])}
                                 for row in state["range_directory"]]
        model._rebuild_range_index()
        model.search_index.set_bridges([], [], [])
        model.entry_mode = state["entry_mode"]
        model.receipt = state["build_receipt"]
        model.build_seconds = model.receipt["build_seconds"]
        model.frozen = True
        model.load_seconds = time.perf_counter() - started
        return model


def build(vectors, attrs, windows, workers: int = 16, *, progress=None) -> FRANSResourceIndex:
    return FRANSResourceIndex.build(vectors, attrs, windows, workers=workers, progress=progress)


def load(path) -> FRANSResourceIndex:
    return FRANSResourceIndex.load(path)


__all__ = ["FRANSResourceIndex", "build", "load", "FORMAT", "PARAMETERS"]
