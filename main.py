"""Run with: uv run modal run main.py"""
import csv
import json
import time
import zlib
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable

import modal

app = modal.App("safecompress")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface-hub==0.34.4", "zstandard==0.24.0", "lz4==4.4.4"
)


@dataclass(frozen=True)
class Codec:
    name: str
    compress: Callable[[bytes], bytes]
    decompress: Callable[[bytes], bytes]


def codecs() -> list[Codec]:
    import lz4.frame
    import zstandard as zstd

    compress = zstd.ZstdCompressor(level=3).compress
    decompress = zstd.ZstdDecompressor().decompress
    return [
        Codec("zstd-3", compress, decompress),
        Codec("zstd-9", zstd.ZstdCompressor(level=9).compress, decompress),
        Codec("lz4", lz4.frame.compress, lz4.frame.decompress),
        Codec("deflate-6", zlib.compress, zlib.decompress),
    ]


def benchmark(data: bytes, codec: Codec) -> dict:
    if not data:
        raise ValueError("Empty sample")
    start = time.perf_counter()
    packed = codec.compress(data)
    compress_s = time.perf_counter() - start
    start = time.perf_counter()
    restored = codec.decompress(packed)
    decompress_s = time.perf_counter() - start
    if restored != data:
        raise ValueError(f"{codec.name}: lossless round-trip failed")
    return dict(method=codec.name, raw_bytes=len(data), compressed_bytes=len(packed),
                ratio=len(data) / len(packed), savings_pct=100 * (1 - len(packed) / len(data)),
                compress_s=compress_s, decompress_s=decompress_s)


@app.function(image=image, cpu=2, memory=4096, timeout=3600, max_containers=5)
def run_file(model: str, revision: str, filename: str, sample_mib: int,
             chunk_mib: int, methods: str) -> dict:
    from huggingface_hub import hf_hub_download

    selected = {name.strip() for name in methods.split(",")} if methods else set()
    available = codecs()
    if selected - {c.name for c in available}:
        raise ValueError(f"Unknown methods: {selected - {c.name for c in available}}")
    active = [c for c in available if not selected or c.name in selected]
    totals = {c.name: dict(raw_bytes=0, compressed_bytes=0, compress_s=0., decompress_s=0.)
              for c in active}
    path = hf_hub_download(model, filename, revision=revision)
    file_bytes = Path(path).stat().st_size
    remaining = min(sample_mib * 1024**2, file_bytes) if sample_mib else file_bytes
    chunks = 0
    with open(path, "rb") as f:
        while remaining > 0:
            data = f.read(min(remaining, chunk_mib * 1024**2))
            if not data:
                raise ValueError(f"Unexpected EOF in {filename}")
            for codec in active:
                result = benchmark(data, codec)
                for key in totals[codec.name]:
                    totals[codec.name][key] += result[key]
            remaining -= len(data)
            chunks += 1
            print(f"{model}: {filename}, chunk {chunks} verified", flush=True)
    return dict(totals=totals, file_bytes=file_bytes, chunks=chunks)


@app.function(image=image, cpu=0.25, memory=512, timeout=3600)
def run_model(model: str, sample_mib: int, chunk_mib: int = 64, methods: str = "") -> list[dict]:
    from huggingface_hub import HfApi

    if sample_mib < 0 or chunk_mib < 1:
        raise ValueError("sample-mib must be nonnegative; chunk-mib must be positive")
    info = HfApi().model_info(model)
    files = sorted(f.rfilename for f in info.siblings if f.rfilename.endswith(".safetensors"))
    if not files:
        raise ValueError(f"No safetensors files in {model}")
    files = files[:1] if sample_mib else files
    parts = list(run_file.starmap((model, info.sha, f, sample_mib, chunk_mib, methods)
                                  for f in files))
    rows = []
    for method in parts[0]["totals"]:
        total = {key: sum(p["totals"][method][key] for p in parts)
                 for key in parts[0]["totals"][method]}
        raw, packed = total["raw_bytes"], total["compressed_bytes"]
        if not raw:
            raise ValueError(f"No data in {model}")
        rows.append(dict(model=model, revision=info.sha, files=files,
                         file_bytes=sum(p["file_bytes"] for p in parts),
                         scope="file-prefix" if sample_mib else "all-safetensors",
                         chunk_mib=chunk_mib, chunks=sum(p["chunks"] for p in parts),
                         method=method, **total, ratio=raw / packed,
                         savings_pct=100 * (1 - packed / raw)))
    return rows


def html_table(rows: list[dict]) -> str:
    headers = ("Model", "Scope", "Method", "Input MiB", "Ratio", "Saved",
               "Compress (s)", "Decompress (s)")
    lines = ['<table border="1" cellspacing="0" cellpadding="6">',
             "  <tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cells = (r["model"], r["scope"], r["method"], f"{r['raw_bytes'] / 1024**2:.1f}",
                 f"{r['ratio']:.3f}x", f"{r['savings_pct']:.1f}%",
                 f"{r['compress_s']:.2f}", f"{r['decompress_s']:.2f}")
        lines.append("  <tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
    return "\n".join([*lines, "</table>"]) + "\n"


@app.local_entrypoint()
def main(models: str = "Qwen/Qwen3-0.6B", sample_mib: int = 0,
         chunk_mib: int = 64, output: str = "results", methods: str = ""):
    if sample_mib < 0 or chunk_mib < 1:
        raise ValueError("sample-mib must be nonnegative; chunk-mib must be positive")
    rows = [row for model in models.split(",")
            for row in run_model.remote(model.strip(), sample_mib, chunk_mib, methods)]
    Path(output).mkdir(parents=True, exist_ok=True)
    Path(output, "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with Path(output, "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Model | Method | Input MiB | Ratio (raw/compressed) | Saved | Compress s |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['method']} | {r['raw_bytes'] / 1024**2:.1f} | "
                     f"{r['ratio']:.3f}x | {r['savings_pct']:.1f}% | {r['compress_s']:.3f} |")
    table = "\n".join(lines) + "\n"
    Path(output, "results.md").write_text(table)
    Path(output, "results.html").write_text(html_table(rows))
    print(table)
