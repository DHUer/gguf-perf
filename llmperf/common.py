"""Shared helpers: platform-portable paths, llama-bench discovery, GGUF metadata.

Everything here must work identically on macOS and Windows.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("LLMPERF_MODELS", ROOT / "models"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# llama.cpp stores the KV cache in f16 by default (both K and V).
KV_BYTES_PER_ELEM = 2

# Bump on every change to the metadata parse. Cached entries carry it, so a
# corrected parser invalidates old values instead of serving them forever.
_PARSER_VERSION = 4


def _is_seq(v) -> bool:
    """True for GGUF array fields; str/bytes are sequences but not arrays."""
    return isinstance(v, (list, tuple)) or (
        hasattr(v, "__len__") and hasattr(v, "__getitem__")
        and not isinstance(v, (str, bytes)))


def host_id() -> str:
    """Stable, filesystem-safe identifier for the current machine.

    Results are keyed by this so the three machines can write into one repo
    without colliding, and so analysis can join measurements to calibration.
    """
    name = os.environ.get("LLMPERF_HOST") or platform.node().split(".")[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def platform_tag() -> str:
    return f"{platform.system()}-{platform.machine()}"


def load_model_splits(models_dir: Path = MODELS_DIR,
                      results_dir: Path = RESULTS_DIR) -> dict[str, str]:
    """Load fixed train/test assignments from every available manifest.

    ``models/manifest.json`` is the live download manifest and normally lives
    beside the (gitignored) weights. ``results/model_manifest.json`` is its
    tracked snapshot, so a clean clone can reproduce the split used by the
    committed measurements without downloading hundreds of gigabytes first.

    A disagreement is an error rather than a last-writer-wins merge: silently
    changing a model's split after observing its error would invalidate the
    held-out evaluation.
    """
    paths = [Path(results_dir) / "model_manifest.json",
             Path(models_dir) / "manifest.json"]
    splits: dict[str, str] = {}
    seen_paths: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen_paths or not path.exists():
            continue
        seen_paths.add(key)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"cannot read model manifest {path}: {e}") from e
        if not isinstance(manifest, dict):
            raise ValueError(f"model manifest {path} must contain a JSON object")
        for filename, entry in manifest.items():
            split = entry.get("split") if isinstance(entry, dict) else None
            if split not in {"train", "test"}:
                raise ValueError(
                    f"model manifest {path} has invalid split for {filename!r}: "
                    f"{split!r}")
            previous = splits.get(filename)
            if previous is not None and previous != split:
                raise ValueError(
                    f"conflicting split provenance for {filename!r}: "
                    f"{previous!r} versus {split!r} in {path}")
            splits[filename] = split
    return splits


def load_model_metadata(results_dir: Path = RESULTS_DIR) -> dict[str, dict]:
    """Load the auditable metadata snapshot stored beside committed results.

    The GGUF files remain the primary source whenever they are available. This
    snapshot preserves the exact derived fields needed to replay analysis on a
    clean clone where hundreds of gigabytes of model weights are intentionally
    absent.
    """
    path = Path(results_dir) / "model_metadata.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"cannot read model metadata snapshot {path}: {e}") from e
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ValueError(f"unsupported model metadata snapshot schema in {path}")
    models = doc.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"model metadata snapshot {path} has no models object")

    required = {
        "arch", "file_bytes", "n_params", "n_active_params", "n_layers",
        "n_kv_heads", "head_dim", "n_expert", "n_expert_used", "quant",
        "kv_bytes_by_depth", "vocab_size", "d_model",
    }
    for filename, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"metadata for {filename!r} in {path} is not an object")
        missing = sorted(required - entry.keys())
        if missing:
            raise ValueError(
                f"metadata for {filename!r} in {path} lacks: {', '.join(missing)}")
        kv = entry["kv_bytes_by_depth"]
        if not isinstance(kv, dict) or not kv:
            raise ValueError(
                f"metadata for {filename!r} in {path} has no KV-depth mapping")
    return models


def find_llama_bench() -> Path:
    """Locate the llama-bench binary on macOS or Windows.

    Honours LLAMA_BENCH first so a user can point at a custom build.  On
    Windows, also recognize versioned binaries installed by this project under
    ``%LOCALAPPDATA%\\gguf-perf`` so the runner survives a new shell or reboot.
    """
    override = os.environ.get("LLAMA_BENCH")
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"LLAMA_BENCH={override} is not a file")

    for name in ("llama-bench", "llama-bench.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)

    managed: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if platform.system() == "Windows" and local_appdata:
        managed = sorted(
            (Path(local_appdata) / "gguf-perf").glob(
                "llama-*/llama-bench.exe"),
            reverse=True,
        )

    candidates = managed + [
        Path("/opt/homebrew/bin/llama-bench"),
        Path("/usr/local/bin/llama-bench"),
        ROOT / "llama.cpp" / "build" / "bin" / "llama-bench",
        ROOT / "llama.cpp" / "build" / "bin" / "Release" / "llama-bench.exe",
        Path(r"C:\Program Files\llama.cpp\llama-bench.exe"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(
        "llama-bench not found. Install llama.cpp (macOS: `brew install llama.cpp`; "
        "Windows: download a release from github.com/ggml-org/llama.cpp/releases and "
        "add it to PATH) or set the LLAMA_BENCH environment variable."
    )


@dataclass
class ModelMeta:
    """Architecture facts needed by the performance model, read from the GGUF file."""

    path: str
    name: str
    file_bytes: int
    n_params: int          # total parameter count, summed from tensor shapes
    n_active_params: int   # equals n_params for dense models; smaller for MoE
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_embd: int
    n_expert: int
    n_expert_used: int
    quant: str             # e.g. Q4_K_M, inferred from the filename
    arch: str
    embd_bytes: int = 0    # size of token_embd.weight
    tied_embeddings: bool = True   # True when no separate output.weight exists
    # Per-layer attention shape. GGUF stores head_count_kv as an ARRAY when
    # layers differ, and 0 marks a layer with no KV cache at all (SSM/Mamba).
    kv_heads_per_layer: tuple = ()
    # sliding_window_pattern[i] is True where layer i is windowed rather than
    # global; its KV is capped at sliding_window instead of growing with context.
    swa_pattern: tuple = ()
    sliding_window: int = 0
    ssm_state_bytes: int = 0   # constant per-token state for recurrent layers

    def bytes_per_weight(self) -> float:
        return self.file_bytes / self.n_params if self.n_params else float("nan")

    def streamed_bytes(self) -> int:
        """Weight bytes actually read to produce one token.

        The token embedding table is NOT streamed: decoding gathers a single
        row from it. When the output head is a separate tensor the embedding
        table is therefore dead weight for bandwidth purposes and must be
        subtracted. When embeddings are tied, the same tensor IS read in full
        for the logit projection and must be kept.

        This matters: a large-vocabulary model can carry an embedding table
        worth several percent of the file, and charging it per token
        systematically under-predicts throughput for exactly those models.
        """
        if self.tied_embeddings:
            return self.file_bytes
        return max(1, self.file_bytes - self.embd_bytes)

    def kv_bytes(self, context: int) -> int:
        """KV cache bytes read per token at a given context length.

        Summed PER LAYER, because the uniform-global-attention assumption that
        the textbook roofline makes is false for much of the 2026 open-weight
        cohort:

        * Sliding-window layers cap their cache at the window, so their cost
          stops growing once context exceeds it. Gemma-4 runs 25 of 30 layers
          windowed at 1024 tokens.
        * Recurrent (SSM/Mamba) layers keep a constant-size state and no KV
          cache at all. Nemotron-3-Nano has attention in only ~8 of 52 blocks.
        * Layers within one model can differ in KV head count, which GGUF
          signals by storing head_count_kv as an array rather than a scalar.

        Charging every layer a full global cache overestimates KV traffic by
        more than an order of magnitude on these models at long context, and
        the predictor then invents a memory wall that does not exist.
        """
        per_layer = self.kv_heads_per_layer or ((self.n_kv_heads,) * self.n_layers)
        total = 0
        for i, heads in enumerate(per_layer):
            if not heads:
                continue                      # recurrent layer: no KV cache
            eff = context
            if self.sliding_window and i < len(self.swa_pattern) and self.swa_pattern[i]:
                eff = min(context, self.sliding_window)
            total += 2 * heads * self.head_dim * eff * KV_BYTES_PER_ELEM
        return total + self.ssm_state_bytes

    def working_bytes(self, context: int) -> int:
        """Bytes that must be read to produce one token: weights + KV cache.

        This is the denominator of the decode roofline. For MoE models only the
        active experts are read, so weights are scaled by the active fraction.
        """
        active_frac = self.n_active_params / self.n_params if self.n_params else 1.0
        return int(self.streamed_bytes() * active_frac) + self.kv_bytes(context)


def _quant_from_name(filename: str) -> str:
    stem = Path(filename).stem.upper()
    # Longest-match first so Q4_K_M is not truncated to Q4_K.
    known = [
        "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ3_M",
        "IQ4_XS", "IQ4_NL",
        "Q2_K_L", "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K",
        "Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q4_1",
        "Q5_K_M", "Q5_K_S", "Q5_K", "Q5_0", "Q5_1",
        "Q6_K", "Q8_0", "MXFP4", "NVFP4", "BF16", "F16", "F32",
    ]
    for q in known:
        if q in stem:
            return q
    return "UNKNOWN"


_META_CACHE = RESULTS_DIR / "gguf_meta_cache.json"
_meta_mem: dict = {}


def read_gguf_meta(path: Path, use_cache: bool = True) -> ModelMeta:
    """Cached wrapper around the GGUF header parse.

    Parsing is not free — the set includes a 63 GB file — and analysis re-reads
    every model on each run. Cached on (name, size, mtime) so an edited or
    replaced file is re-parsed automatically, and on the parser version so a
    corrected parser invalidates stale entries rather than serving wrong
    metadata from disk. That last part matters: two metadata bugs in this
    project were only caught after the fact, and a cache without it would have
    kept serving the bad values.
    """
    if not use_cache:
        return _read_gguf_meta_uncached(path)

    st = path.stat()
    key = f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{_PARSER_VERSION}"
    if key in _meta_mem:
        return _meta_mem[key]

    disk = {}
    if _META_CACHE.exists():
        try:
            disk = json.loads(_META_CACHE.read_text(encoding="utf-8"))
        except Exception:
            disk = {}
    if key in disk:
        m = ModelMeta(**{k: (tuple(v) if isinstance(v, list) else v)
                         for k, v in disk[key].items()})
        _meta_mem[key] = m
        return m

    m = _read_gguf_meta_uncached(path)
    disk[key] = {k: (list(v) if isinstance(v, tuple) else v)
                 for k, v in asdict(m).items()}
    try:
        _META_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _META_CACHE.write_text(json.dumps(disk, indent=1), encoding="utf-8")
    except Exception:
        pass
    _meta_mem[key] = m
    return m


def _read_gguf_meta_uncached(path: Path) -> ModelMeta:
    """Extract architecture metadata straight from the GGUF header.

    Reading the file itself avoids a hand-maintained model registry that could
    silently drift out of sync with the weights actually being benchmarked.
    """
    from gguf import GGUFReader  # imported lazily so --help works without deps

    reader = GGUFReader(str(path))

    fields = {}
    for key, field in reader.fields.items():
        try:
            fields[key] = field.contents()
        except Exception:
            fields[key] = None

    arch = fields.get("general.architecture") or "unknown"
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8", "replace")

    def g(suffix, default=0):
        v = fields.get(f"{arch}.{suffix}")
        if isinstance(v, (int, float)):
            return int(v)
        # Per-layer arrays: take the max as the scalar stand-in. Returning the
        # default here is what silently hid gemma-4's and Nemotron's real
        # attention shape behind a fallback to n_heads.
        if _is_seq(v) and len(v):
            try:
                return int(max(v))
            except (TypeError, ValueError):
                return default
        return default

    def g_seq(suffix):
        v = fields.get(f"{arch}.{suffix}")
        if _is_seq(v):
            try:
                return tuple(v)
            except TypeError:
                return ()
        return ()

    n_layers = g("block_count")
    n_heads = g("attention.head_count")
    n_kv_heads = g("attention.head_count_kv") or n_heads
    n_embd = g("embedding_length")
    n_expert = g("expert_count")
    n_expert_used = g("expert_used_count")

    head_dim = g("attention.key_length") or (n_embd // n_heads if n_heads else 0)

    kv_per_layer = tuple(int(x) for x in g_seq("attention.head_count_kv"))
    swa_pattern = tuple(bool(x) for x in g_seq("attention.sliding_window_pattern"))
    sliding_window = g("attention.sliding_window")

    # Recurrent layers carry a fixed-size state instead of a growing cache:
    # a conv window plus the SSM state, per group. Constant in context, so it
    # shifts the intercept rather than the slope.
    ssm_state_bytes = 0
    ssm_inner, ssm_state = g("ssm.inner_size"), g("ssm.state_size")
    ssm_conv, ssm_groups = g("ssm.conv_kernel"), g("ssm.group_count")
    if ssm_inner and ssm_state:
        n_recurrent = sum(1 for h in kv_per_layer if not h) if kv_per_layer else 0
        per = (ssm_inner * ssm_state + ssm_inner * max(ssm_conv, 1)) * max(ssm_groups, 1)
        ssm_state_bytes = per * n_recurrent * KV_BYTES_PER_ELEM

    # Sum real tensor shapes rather than trusting a metadata field that many
    # converters omit or get wrong.
    n_params = 0
    expert_params = 0
    embd_bytes = 0
    has_output = False
    for t in reader.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        n_params += n
        # MoE expert tensors carry the expert count as their leading dimension.
        # Match the "_exps" suffix generically rather than enumerating names:
        # architectures fuse these differently — gemma-4 ships a combined
        # ffn_gate_up_exps where Qwen ships separate gate/up tensors — and an
        # enumerated list silently misses the fused variant, leaving most of
        # the expert weight counted as always-active. That understates the
        # sparsity of the very models the sparsity term exists for.
        if n_expert and "_exps" in t.name:
            expert_params += n
        if t.name == "token_embd.weight":
            embd_bytes = int(getattr(t, "n_bytes", 0) or 0)
        elif t.name == "output.weight":
            has_output = True

    if n_expert and n_expert_used and expert_params:
        inactive = expert_params * (1 - n_expert_used / n_expert)
        n_active_params = int(n_params - inactive)
    else:
        n_active_params = n_params

    # A sparse MoE that comes out mostly-active means the expert tensors were
    # not recognised, not that the model is dense. Silently accepting that
    # would corrupt the bytes-per-token term for exactly the models the
    # sparsity correction exists to handle, so make it loud.
    if n_expert and n_expert_used and n_expert_used < n_expert / 2:
        frac = n_active_params / n_params if n_params else 1.0
        if frac > 0.5:
            import warnings
            warnings.warn(
                f"{path.name}: declares {n_expert_used}/{n_expert} experts active "
                f"but {frac:.0%} of parameters resolve as active. Expert tensors "
                f"were probably not matched — check the tensor names.",
                stacklevel=2)

    return ModelMeta(
        path=str(path),
        name=path.stem,
        file_bytes=path.stat().st_size,
        n_params=n_params,
        n_active_params=n_active_params,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_embd=n_embd,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        quant=_quant_from_name(path.name),
        arch=str(arch),
        embd_bytes=embd_bytes,
        tied_embeddings=not has_output,
        kv_heads_per_layer=kv_per_layer,
        swa_pattern=swa_pattern,
        sliding_window=sliding_window,
        ssm_state_bytes=ssm_state_bytes,
    )


def discover_models(models_dir: Path = MODELS_DIR) -> list[Path]:
    """All single-file GGUFs under models_dir. Multi-part shards are skipped."""
    if not models_dir.is_dir():
        return []
    out = []
    for p in sorted(models_dir.rglob("*.gguf")):
        if "-of-" in p.name and not p.name.endswith("-00001-of-00001.gguf"):
            continue
        out.append(p)
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def git_commit() -> str:
    """Record provenance so every result row can be traced to the exact code."""
    # On a fresh macOS host, /usr/bin/git is a launcher that can open the Xcode
    # Command Line Tools installer.  A source archive has no commit to resolve,
    # so avoid invoking Git at all.  ``exists()`` intentionally accepts either
    # a normal .git directory or the .git pointer file used by worktrees.
    if not (ROOT / ".git").exists():
        return "nogit"
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _selfcheck() -> None:
    """Smallest check that fails if the non-trivial logic here breaks."""
    m = ModelMeta(
        path="x", name="x", file_bytes=1_000_000_000, n_params=1_000_000_000,
        n_active_params=1_000_000_000, n_layers=32, n_heads=32, n_kv_heads=8,
        head_dim=128, n_embd=4096, n_expert=0, n_expert_used=0, quant="Q4_K_M", arch="llama",
    )
    # 2 * 32 layers * 8 kv heads * 128 dim * 1024 ctx * 2 bytes = 134217728
    assert m.kv_bytes(1024) == 134_217_728, m.kv_bytes(1024)
    assert m.kv_bytes(0) == 0
    assert m.working_bytes(0) == 1_000_000_000
    assert abs(m.bytes_per_weight() - 1.0) < 1e-9

    moe = ModelMeta(**{**asdict(m), "n_params": 30_000_000_000,
                       "n_active_params": 3_000_000_000, "file_bytes": 30_000_000_000,
                       "n_expert": 128, "n_expert_used": 8})
    # Only the active tenth of the weights is read per token.
    assert moe.working_bytes(0) == 3_000_000_000, moe.working_bytes(0)

    assert _quant_from_name("qwen2.5-7b-instruct-q4_k_m.gguf") == "Q4_K_M"
    assert _quant_from_name("Llama-3.2-1B-Instruct-Q8_0.gguf") == "Q8_0"
    assert _quant_from_name("model-f16.gguf") == "F16"
    assert _quant_from_name("mystery.gguf") == "UNKNOWN"
    print("common.py selfcheck ok")


if __name__ == "__main__":
    _selfcheck()
