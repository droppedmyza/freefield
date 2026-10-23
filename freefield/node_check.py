from __future__ import annotations

import json
import urllib.request
from pathlib import Path

SCHEMAS = Path(__file__).resolve().parent / "node_schemas.json"


def load_schemas(comfy_url: str | None = "http://127.0.0.1:8188", timeout: float = 5.0) -> tuple[dict, str]:
    if comfy_url:
        try:
            with urllib.request.urlopen(f"{comfy_url.rstrip('/')}/object_info", timeout=timeout) as resp:
                return json.loads(resp.read()), "live"
        except (OSError, ValueError):
            pass
    return json.loads(SCHEMAS.read_text()), "snapshot"


def _spec(schema: dict, name: str) -> list | None:
    for part in ("required", "optional"):
        if name in schema.get("input", {}).get(part, {}):
            return schema["input"][part][name]
    return None


def _options(spec: list) -> list | None:
    t = spec[0]
    if isinstance(t, list):
        return t
    if t == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def validate(graph: dict, fixed: dict, schemas: dict, bound: set[str] = frozenset()) -> list[str]:
    errors: list[str] = []
    filled = set(fixed) | set(bound)
    if not any(schemas.get(v.get("class_type"), {}).get("output_node") for v in graph.values()):
        errors.append("the graph has no output node (nothing would be saved)")
    for nid, n in graph.items():
        ct = n.get("class_type")
        schema = schemas.get(ct)
        if schema is None:
            errors.append(f"{nid}: node type {ct!r} is not installed in ComfyUI")
            continue
        for req in schema.get("input", {}).get("required", {}):
            if req not in n.get("inputs", {}) and f"{nid}.{req}" not in filled:
                errors.append(f"{nid} ({ct}): required input {req!r} is missing")
        for name, value in n.get("inputs", {}).items():
            spec = _spec(schema, name)
            if spec is None:
                if "." not in name:
                    errors.append(f"{nid} ({ct}): unknown input {name!r}")
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                src = graph.get(value[0])
                if src is None:
                    errors.append(f"{nid}.{name}: links to node {value[0]!r}, which is not in the graph")
                    continue
                outs = schemas.get(src.get("class_type"), {}).get("output", [])
                if value[1] >= len(outs):
                    errors.append(f"{nid}.{name}: links to output {value[1]} of {value[0]}, which has {len(outs)}")
                elif isinstance(spec[0], str) and spec[0] != "COMBO" and outs[value[1]] != spec[0] and "*" not in (spec[0], outs[value[1]]):
                    errors.append(f"{nid}.{name}: expects {spec[0]}, but {value[0]} gives {outs[value[1]]}")
                continue
            opts = _options(spec)
            if opts is not None and value not in opts and f"{nid}.{name}" not in filled and value != "":
                kind = "model" if any(str(value).endswith(e) for e in (".safetensors", ".gguf", ".pt", ".ckpt", ".pth")) else "value"
                errors.append(f"{nid}.{name}: {kind} {value!r} is not available"
                              + (" (download it first)" if kind == "model" else f"; choose one of {opts[:6]}"))
            if opts is not None and value == "" and f"{nid}.{name}" not in filled:
                errors.append(f"{nid}.{name}: is empty and nothing fills it")
    return errors
