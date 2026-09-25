"""How a checkpoint is stored on disk.

A model file is just a long run of numbers plus an index that says where each
tensor starts and ends. For .safetensors the index is a JSON header at the
front of the file:

    [8 bytes: header length N, little-endian] [N bytes: JSON] [raw tensor bytes]

For older PyTorch .bin files it is a zip archive holding a pickled index and
one data blob per tensor. This module reads those structures directly so the
UI can show the actual bytes, not a description of them.
"""
import json
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import torch

from ..hub.load import parse_spec
from ..hub.weights import _kind, _layer_of

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1}


def _decode(raw: bytes, dtype: str):
    if dtype == "F32":
        return np.frombuffer(raw, "<f4").astype(float)
    if dtype == "F16":
        return np.frombuffer(raw, "<f2").astype(float)
    if dtype == "BF16":
        u = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
        return u.view("<f4").astype(float)
    if dtype == "F64":
        return np.frombuffer(raw, "<f8").astype(float)
    return np.frombuffer(raw, "u1").astype(float)


def _float_bits(x: float, dtype: str):
    """Sign, exponent and mantissa bits of one value, to show what a weight is."""
    if dtype == "F16":
        b = struct.unpack("<H", struct.pack("<e", x))[0]
        s = f"{b:016b}"
        return {"sign": s[0], "exponent": s[1:6], "mantissa": s[6:]}
    if dtype == "BF16":
        b = struct.unpack("<I", struct.pack("<f", x))[0] >> 16
        s = f"{b:016b}"
        return {"sign": s[0], "exponent": s[1:9], "mantissa": s[9:]}
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    s = f"{b:032b}"
    return {"sign": s[0], "exponent": s[1:9], "mantissa": s[9:]}


def _local_or_download(spec):
    """Return (list of (relative name, local path, size)), plus a readable source label."""
    if os.path.isdir(spec):
        root = Path(spec)
        files = [(p.name, str(p), p.stat().st_size) for p in sorted(root.iterdir()) if p.is_file()]
        return files, str(root)
    from huggingface_hub import HfApi, hf_hub_download

    repo, sub, rev = parse_spec(spec)
    api = HfApi()
    items = [i for i in api.list_repo_tree(repo, path_in_repo=sub, revision=rev) if getattr(i, "size", None) is not None]
    out = []
    for i in items:
        name = i.path.split("/")[-1]
        wanted = name.endswith((".safetensors", ".bin", ".json")) and not name.startswith("training_args")
        path = hf_hub_download(repo, i.path, revision=rev) if wanted and (name.endswith(".json") or name.endswith((".safetensors", ".bin"))) else None
        out.append((name, path, i.size))
    return out, f"huggingface.co/{repo}" + (f"/{sub}" if sub else "") + (f" @ {rev}" if rev else "")


def _hist(values, lo, hi, bins=40):
    h, _ = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    return h.tolist()


def storage_report(spec, progress=lambda f, m: None):
    progress(0.05, f"Listing files for {spec}")
    files, source = _local_or_download(spec)
    listing = [{"name": n, "bytes": int(s or 0)} for n, _, s in files]
    config = None
    for n, p, _ in files:
        if n == "config.json" and p:
            try:
                config = json.loads(Path(p).read_text())
            except Exception:
                config = None
    weight_files = [(n, p) for n, p, _ in files if p and n.endswith(".safetensors")]
    fmt = "safetensors"
    if not weight_files:
        weight_files = [(n, p) for n, p, _ in files if p and n.endswith(".bin")]
        fmt = "pytorch-zip"
    if not weight_files:
        raise FileNotFoundError(f"No weight files found for {spec}.")

    tensors, header_bytes, meta, zip_entries, preview = [], 0, None, [], None
    values_by_kind = {}
    for fi, (name, path) in enumerate(weight_files):
        progress(0.1 + 0.6 * fi / len(weight_files), f"Reading {name}")
        if fmt == "safetensors":
            with open(path, "rb") as f:
                first8 = f.read(8)
                n = struct.unpack("<Q", first8)[0]
                header = json.loads(f.read(n))
                data_start = 8 + n
                header_bytes += n
                meta = header.pop("__metadata__", None) or meta
                entries = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
                for tname, info in entries:
                    a, b = info["data_offsets"]
                    numel = int(np.prod(info["shape"])) if info["shape"] else 1
                    f.seek(data_start + a)
                    head = f.read(min(32, b - a))
                    row = {"name": tname, "file": name, "dtype": info["dtype"], "shape": info["shape"], "numel": numel,
                           "offset": data_start + a, "bytes": b - a, "kind": _kind(tname), "layer": _layer_of(tname)}
                    tensors.append(row)
                    if preview is None and len(info["shape"]) == 2 and info["dtype"] in ("F32", "F16", "BF16"):
                        vals = _decode(head, info["dtype"])[:8]
                        preview = {"tensor": tname, "file": name, "dtype": info["dtype"], "offset": data_start + a,
                                   "header_len": n, "header_prefix": first8.hex(" "),
                                   "hex": head.hex(" "), "values": [float(v) for v in vals],
                                   "bits": _float_bits(float(vals[0]), info["dtype"]) if len(vals) else None,
                                   "header_json": json.dumps({tname: info}, indent=1)}
            from safetensors import safe_open

            with safe_open(path, "pt") as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k)
                    if t.is_floating_point():
                        values_by_kind.setdefault(_kind(k), []).append(t.float().flatten())
        else:
            with zipfile.ZipFile(path) as z:
                zip_entries += [{"name": i.filename, "bytes": i.file_size} for i in z.infolist()][:60]
            sd = torch.load(path, map_location="cpu", weights_only=True)
            for k, t in sd.items():
                if not isinstance(t, torch.Tensor):
                    continue
                dt = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}.get(t.dtype, str(t.dtype))
                tensors.append({"name": k, "file": name, "dtype": dt, "shape": list(t.shape), "numel": t.numel(),
                                "offset": None, "bytes": t.numel() * t.element_size(), "kind": _kind(k), "layer": _layer_of(k)})
                if t.is_floating_point():
                    values_by_kind.setdefault(_kind(k), []).append(t.float().flatten())
                    if preview is None and t.ndim == 2:
                        v = t.flatten()[:8].float().tolist()
                        preview = {"tensor": k, "file": name, "dtype": dt, "values": v, "bits": _float_bits(v[0], dt)}

    progress(0.8, "Summarising values")
    total_bytes = sum(t["bytes"] for t in tensors)
    by_kind = {}
    for t in tensors:
        k = by_kind.setdefault(t["kind"], {"kind": t["kind"], "bytes": 0, "params": 0, "tensors": 0})
        k["bytes"] += t["bytes"]
        k["params"] += t["numel"]
        k["tensors"] += 1
    by_layer = {}
    for t in tensors:
        if t["layer"] is not None:
            by_layer[t["layer"]] = by_layer.get(t["layer"], 0) + t["numel"]
    dtypes = {}
    for t in tensors:
        dtypes[t["dtype"]] = dtypes.get(t["dtype"], 0) + t["numel"]

    hists = []
    for kind, parts in values_by_kind.items():
        v = torch.cat(parts)
        if v.numel() > 2_000_000:
            v = v[torch.randperm(v.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
        v = v.numpy()
        sd = float(v.std()) or 1.0
        mu = float(v.mean())
        lo, hi = mu - 4 * sd, mu + 4 * sd
        hists.append({"kind": kind, "mean": mu, "std": sd, "min": float(v.min()), "max": float(v.max()),
                      "near_zero": float((np.abs(v) < 0.01 * sd).mean()), "lo": lo, "hi": hi, "hist": _hist(v, lo, hi)})
    progress(1.0, "Done")
    return {
        "spec": spec, "source": source, "format": fmt, "files": listing, "config": config,
        "total_params": sum(t["numel"] for t in tensors), "total_bytes": total_bytes, "header_bytes": header_bytes,
        "metadata": meta, "tensors": tensors, "by_kind": sorted(by_kind.values(), key=lambda k: -k["bytes"]),
        "by_layer": [{"layer": k, "params": v} for k, v in sorted(by_layer.items())], "dtypes": dtypes,
        "preview": preview, "zip_entries": zip_entries, "histograms": sorted(hists, key=lambda h: h["kind"]),
    }
