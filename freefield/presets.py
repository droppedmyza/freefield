from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .pipelines import PIPELINES
from .runner import CompiledRun

PACKAGE_DIR = Path(__file__).resolve().parent
PRESETS_DIR = PACKAGE_DIR / "presets"
GRAPHS_DIR = PACKAGE_DIR / "graphs"

INPUT_TYPES = {"image", "text", "video"}
ASSET = "asset:"
GATED_NODES = ("start_save", "grip_save", "breakout_save", "grip_noise", "breakout_noise")
VIDEO_CHECK_NODES = ("shot1_sample_high", "shot1_sample", "shot1_latent", "start_view_crop")
PARAM_TYPES = {"int", "float", "text", "choice"}
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")
_BIND_RE = re.compile(r"^[A-Za-z0-9_:-]+\.[A-Za-z0-9_]+$")


class PresetError(ValueError):
    pass


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    family: str
    description: str
    graph: dict[str, Any]
    inputs: dict[str, dict[str, Any]]
    params: dict[str, dict[str, Any]]
    fixed: dict[str, Any]
    outputs: list[str]
    models: list[str]
    license: str
    folder: Path
    preview: str | None = None
    tags: list[str] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    pipeline: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "inputs": {k: {kk: vv for kk, vv in v.items() if kk != "bind"} for k, v in self.inputs.items()},
            "params": {k: {kk: vv for kk, vv in v.items() if kk != "bind"} for k, v in self.params.items()},
            "models": self.models,
            "license": self.license,
            "preview": self.preview,
            "pipeline": self.pipeline,
            "tags": self.tags,
        }


def _split_bind(bind: str, where: str) -> tuple[str, str]:
    if not isinstance(bind, str) or not _BIND_RE.match(bind):
        raise PresetError(f"{where}: binding {bind!r} must look like '<node_id>.<input_name>'")
    node_id, input_name = bind.split(".", 1)
    return node_id, input_name


def _check_bind(graph: dict[str, Any], bind: str, where: str) -> None:
    node_id, input_name = _split_bind(bind, where)
    node = graph.get(node_id)
    if node is None:
        raise PresetError(f"{where}: binding {bind!r} names node {node_id!r}, which is not in the graph")
    if input_name not in node.get("inputs", {}):
        raise PresetError(
            f"{where}: node {node_id} ({node.get('class_type')}) has no input {input_name!r}; "
            f"it has {sorted(node.get('inputs', {}))}"
        )


def _load_graph(pid: str, graph_ref: Any, folder: Path, graphs_dir: Path) -> dict[str, Any]:
    if isinstance(graph_ref, dict):
        graph = graph_ref
    else:
        graph_path = folder / graph_ref if (folder / graph_ref).exists() else graphs_dir / graph_ref
        if not graph_path.exists():
            raise PresetError(f"{pid}: graph file {graph_ref!r} not found in {folder} or {graphs_dir}")
        graph = json.loads(graph_path.read_text())
    if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n for n in graph.values()):
        raise PresetError(f"{pid}: graph must be ComfyUI API format (node_id -> {{class_type, inputs}})")
    return graph


def _check_inputs(pid: str, inputs: dict[str, Any], graph: dict[str, Any], pipeline: str | None) -> None:
    for name, spec in inputs.items():
        if spec.get("type") not in INPUT_TYPES:
            raise PresetError(f"{pid}: input {name!r} type must be one of {sorted(INPUT_TYPES)}")
        if not spec.get("bind"):
            raise PresetError(f"{pid}: input {name!r} needs at least one binding")
        for b in spec["bind"]:
            if pipeline and b.startswith("pipeline:"):
                continue
            _check_bind(graph, b, f"{pid}: input {name!r}")


def _check_params(pid: str, params: dict[str, Any], graph: dict[str, Any]) -> None:
    for name, spec in params.items():
        ptype = spec.get("type")
        if ptype not in PARAM_TYPES:
            raise PresetError(f"{pid}: param {name!r} type must be one of {sorted(PARAM_TYPES)}")
        if "default" not in spec:
            raise PresetError(f"{pid}: param {name!r} needs a default")
        if ptype == "choice" and spec["default"] not in spec.get("options", []):
            raise PresetError(f"{pid}: param {name!r} default must be one of its options")
        for b in spec.get("bind", []):
            _check_bind(graph, b, f"{pid}: param {name!r}")


def _check_fixed(pid: str, fixed: dict[str, Any], graph: dict[str, Any], folder: Path) -> None:
    for b, value in fixed.items():
        _check_bind(graph, b, f"{pid}: fixed")
        if isinstance(value, str) and value.startswith(ASSET) and not (folder / value[len(ASSET):]).exists():
            raise PresetError(f"{pid}: fixed {b!r} points at missing asset {value[len(ASSET):]!r}")


def _check_pipeline(pid: str, pipeline: str | None, inputs: dict[str, Any]) -> None:
    if pipeline is not None and pipeline not in PIPELINES:
        raise PresetError(f"{pid}: pipeline must be one of {sorted(PIPELINES)}")
    if pipeline is None and any(v.get("type") == "video" for v in inputs.values()):
        raise PresetError(f"{pid}: a video input needs a pipeline that reads video")


def _whole(value: Any, low: int, high: int | None = None) -> bool:
    return isinstance(value, int) and value >= low and (high is None or value <= high)


def _check_gate(pid: str, gate: dict[str, Any], graph: dict[str, Any]) -> None:
    hole = gate.get("hole")
    if not (isinstance(hole, list) and len(hole) == 4 and all(_whole(v, 0) for v in hole)):
        raise PresetError(f"{pid}: gate.hole must be [x, y, w, h] in pixels")
    if not _whole(gate.get("rerolls", 0), 0, 8):
        raise PresetError(f"{pid}: gate.rerolls must be an integer from 0 to 8")
    if gate.get("rail") is not None and not _whole(gate["rail"], 0, 300):
        raise PresetError(f"{pid}: gate.rail must be 0-300 pixels")
    if not _whole(gate.get("frame_border", 0), 0):
        raise PresetError(f"{pid}: gate.frame_border must be a whole number of pixels")
    if gate.get("grip_edge", "bottom") not in ("bottom", "top"):
        raise PresetError(f"{pid}: gate.grip_edge must be 'bottom' or 'top'")
    for n in GATED_NODES:
        if n not in graph:
            raise PresetError(f"{pid}: a gated preset's graph needs a {n!r} node")
    if not _whole(gate.get("video_rerolls", 0), 0, 4):
        raise PresetError(f"{pid}: gate.video_rerolls must be an integer from 0 to 4")
    if gate.get("video_rerolls") or gate.get("shot1_corners"):
        for n in VIDEO_CHECK_NODES:
            if n not in graph:
                raise PresetError(f"{pid}: the video checks need a {n!r} node")


def _check_outputs(pid: str, outputs: list[Any], graph: dict[str, Any]) -> list[str]:
    names = [str(o) for o in outputs]
    for o in names:
        if o not in graph:
            raise PresetError(f"{pid}: output node {o!r} is not in the graph")
    return names


def parse_preset(data: dict[str, Any], folder: Path, graphs_dir: Path = GRAPHS_DIR) -> Preset:
    if not isinstance(data, dict):
        raise PresetError("preset.json must be a JSON object")
    for key in ("id", "name", "graph", "outputs"):
        if key not in data:
            raise PresetError(f"preset.json is missing required field {key!r}")
    pid = data["id"]
    if not isinstance(pid, str) or not _ID_RE.match(pid):
        raise PresetError(f"id {pid!r} must be lowercase letters, digits and dashes (2-49 chars)")
    graph = _load_graph(pid, data["graph"], folder, graphs_dir)
    inputs, params, fixed = data.get("inputs", {}), data.get("params", {}), data.get("fixed", {})
    pipeline, gate = data.get("pipeline"), data.get("gate")
    _check_inputs(pid, inputs, graph, pipeline)
    _check_params(pid, params, graph)
    _check_fixed(pid, fixed, graph, folder)
    _check_pipeline(pid, pipeline, inputs)
    if gate is not None:
        _check_gate(pid, gate, graph)
    outputs = _check_outputs(pid, data["outputs"], graph)
    return Preset(
        id=pid,
        name=data["name"],
        family=data.get("family", "other"),
        description=data.get("description", ""),
        graph=graph,
        inputs=inputs,
        params=params,
        fixed=fixed,
        outputs=outputs,
        models=list(data.get("models", [])),
        license=data.get("license", "unknown"),
        folder=folder,
        preview=data.get("preview"),
        tags=list(data.get("tags", [])),
        gate=gate,
        pipeline=pipeline,
    )


def load_presets(presets_dir: Path = PRESETS_DIR, graphs_dir: Path = GRAPHS_DIR) -> tuple[dict[str, Preset], dict[str, str]]:
    presets: dict[str, Preset] = {}
    errors: dict[str, str] = {}
    if not presets_dir.exists():
        return presets, errors
    for manifest in sorted(presets_dir.glob("*/preset.json")):
        try:
            p = parse_preset(json.loads(manifest.read_text()), manifest.parent, graphs_dir)
        except (PresetError, json.JSONDecodeError) as e:
            errors[manifest.parent.name] = str(e)
            continue
        if p.id in presets:
            errors[manifest.parent.name] = f"duplicate preset id {p.id!r}"
            continue
        presets[p.id] = p
    return presets, errors


def _coerce_param(name: str, spec: dict[str, Any], value: Any) -> Any:
    ptype = spec["type"]
    if ptype == "choice":
        if value not in spec.get("options", []):
            raise PresetError(f"param {name!r} must be one of {spec.get('options')}, got {value!r}")
        return value
    if ptype == "text":
        text, limit = str(value), int(spec.get("max_length", 2000))
        if len(text) > limit:
            raise PresetError(f"param {name!r} is longer than {limit} characters")
        return text
    try:
        number = int(value) if ptype == "int" else float(value)
    except (TypeError, ValueError):
        raise PresetError(f"param {name!r} must be a {ptype}, got {value!r}") from None
    if "min" in spec and number < spec["min"]:
        raise PresetError(f"param {name!r} must be >= {spec['min']}, got {number}")
    if "max" in spec and number > spec["max"]:
        raise PresetError(f"param {name!r} must be <= {spec['max']}, got {number}")
    return number


def _set(graph: dict[str, Any], bind: str, value: Any) -> None:
    node_id, input_name = bind.split(".", 1)
    graph[node_id]["inputs"][input_name] = value


def fill_for_export(graph: dict[str, Any], fixed: dict[str, Any], placeholders: dict[str, str],
                    asset_name: Callable[[str], str] = str) -> dict[str, Any]:
    filled = copy.deepcopy(graph)
    for bind, value in fixed.items():
        _set(filled, bind, asset_name(value[len(ASSET):]) if isinstance(value, str) and value.startswith(ASSET) else value)
    for bind, value in placeholders.items():
        _set(filled, bind, value)
    return filled


def compile_prompt(preset: Preset, inputs: dict[str, Any], params: dict[str, Any] | None = None) -> CompiledRun:
    params = params or {}
    unknown = set(params) - set(preset.params)
    if unknown:
        raise PresetError(f"unknown params {sorted(unknown)}; this preset accepts {sorted(preset.params)}")
    extra = set(inputs) - set(preset.inputs)
    if extra:
        raise PresetError(f"unknown inputs {sorted(extra)}; this preset accepts {sorted(preset.inputs)}")

    graph = copy.deepcopy(preset.graph)
    files: list[tuple[str, str, Path]] = []
    for bind, value in preset.fixed.items():
        if isinstance(value, str) and value.startswith(ASSET):
            node_id, input_name = bind.split(".", 1)
            files.append((node_id, input_name, preset.folder / value[len(ASSET):]))
            continue
        _set(graph, bind, value)
    for name, spec in preset.params.items():
        value = _coerce_param(name, spec, params.get(name, spec["default"]))
        for b in spec.get("bind", []):
            _set(graph, b, value)
    for name, spec in preset.inputs.items():
        value = inputs.get(name)
        if value in (None, ""):
            if spec.get("required", True):
                raise PresetError(f"missing required input {name!r} ({spec.get('label', spec['type'])})")
            continue
        if spec["type"] == "image":
            path = Path(value)
            if not path.is_file():
                raise PresetError(f"input {name!r}: image file {str(path)!r} does not exist")
            for b in spec["bind"]:
                node_id, input_name = b.split(".", 1)
                files.append((node_id, input_name, path))
            continue
        text = str(value)
        if len(text) > int(spec.get("max_length", 500)):
            raise PresetError(f"input {name!r} is longer than {spec.get('max_length', 500)} characters")
        if "template" in spec:
            text = spec["template"].replace("{value}", text)
        for b in spec["bind"]:
            _set(graph, b, text)
    return CompiledRun(graph=graph, files=files, outputs=list(preset.outputs))
