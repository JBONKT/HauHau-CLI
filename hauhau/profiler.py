"""Hardware profiler and llama.cpp parameter recommendation.

Deliberately stdlib-only (``sysctl``/``vm_stat``/``/proc``): on an 8 GB Apple
Silicon box every megabyte of interpreter overhead competes with the model.
Quantization and GPU-offload hill-climbing are *not* offered here - changing
either means re-downloading multi-gigabyte weights, so they are one-shot
recommendations derived from measured hardware, not session-time knobs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field


@dataclass
class HardwareProfile:
    os_name: str
    arch: str
    cpu: str
    cores: int
    ram_gb: float
    gpu: str
    metal: bool
    disk_free_gb: float
    llama_server: str
    llama_version: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    # llama-server prints its version banner to stderr, sysctl to stdout.
    return ((out.stdout or "") + (out.stderr or "")).strip()


def _sysctl(name: str) -> str:
    return _run(["sysctl", "-n", name])


def _darwin_cpu() -> str:
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"


def _darwin_ram_gb() -> float:
    raw = _sysctl("hw.memsize")
    return round(int(raw) / 1024**3, 1) if raw.isdigit() else 0.0


def _linux_info() -> tuple[str, float]:
    cpu, ram_gb = platform.processor() or "unknown", 0.0
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_gb = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    return cpu, ram_gb


def probe() -> HardwareProfile:
    system = platform.system()
    arch = platform.machine()
    cores = os.cpu_count() or 1
    notes: list[str] = []

    if system == "Darwin":
        cpu, ram_gb = _darwin_cpu(), _darwin_ram_gb()
    elif system == "Linux":
        cpu, ram_gb = _linux_info()
    else:
        cpu, ram_gb = platform.processor() or "unknown", 0.0

    metal = system == "Darwin" and arch in ("arm64", "aarch64")
    if metal:
        gpu = "Apple Silicon (Metal, unified memory)"
    else:
        smi = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        gpu = smi.splitlines()[0] if smi else "none detected"
        if not smi:
            notes.append("no nvidia-smi; assuming CPU-only inference")

    try:
        disk_free_gb = round(shutil.disk_usage(os.path.expanduser("~")).free / 1024**3, 1)
    except OSError:
        disk_free_gb = 0.0

    server = _run(["sh", "-c", "command -v llama-server"]) or ""
    version = ""
    if server:
        out = _run([server, "--version"], timeout=20.0)
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                version = line.split(":", 1)[1].strip()
                break

    if ram_gb and ram_gb <= 8:
        notes.append("8 GB class machine: the agent process and model share one memory budget")

    return HardwareProfile(
        os_name=system,
        arch=arch,
        cpu=cpu,
        cores=cores,
        ram_gb=ram_gb,
        gpu=gpu,
        metal=metal,
        disk_free_gb=disk_free_gb,
        llama_server=server,
        llama_version=version,
        notes=notes,
    )


def recommend(profile: HardwareProfile, model_bytes: int = 0) -> dict:
    """Derive llama.cpp parameters. Returns {param: (value, reason)} records."""
    rec: dict[str, tuple[object, str]] = {}

    if profile.metal:
        threads = max(2, min(6, profile.cores // 2))
        rec["n_gpu_layers"] = (99, "Apple Silicon: full Metal offload is strictly faster than partial")
    else:
        threads = max(1, profile.cores - 1)
        rec["n_gpu_layers"] = (0, "no accelerator detected")

    rec["n_threads"] = (threads, f"{profile.cores} logical cores, leaving headroom for the client")

    if not profile.ram_gb:
        # An unreadable total is not a licence to assume a big machine.
        rec["n_ctx"] = (8192, "RAM could not be detected: the conservative 8k context")
    elif profile.ram_gb < 10:
        rec["n_ctx"] = (8192, "8 GB class RAM: 8k context keeps KV cache small")
    elif profile.ram_gb < 18:
        rec["n_ctx"] = (16384, f"{profile.ram_gb:g} GB RAM fits a 16k KV cache")
    else:
        rec["n_ctx"] = (32768, f"{profile.ram_gb:g} GB RAM fits a 32k KV cache")

    # Single-user agent: llama-server's auto parallel often opens 4 slots and
    # multiplies the KV cache. One slot is enough and is critical on 8 GB.
    flags: list[str] = ["-np", "1"]
    reasons = ["one server slot (auto parallel multiplies KV cache)"]
    if profile.metal:
        flags.extend(["-fa", "on"])
        reasons.append("flash attention on Metal")
    if profile.ram_gb and profile.ram_gb <= 8:
        # Stream weights when the model is a large share of unified memory.
        if model_bytes and model_bytes / 1024**3 > profile.ram_gb * 0.35:
            flags = ["-lm", "none", *flags]
            reasons.append("stream weights from SSD to leave RAM for the OS")
    rec["extra_args"] = (flags, "; ".join(reasons))

    if model_bytes and profile.ram_gb and model_bytes / 1024**3 > profile.ram_gb * 0.5:
        rec["warn_quant"] = (
            True,
            f"model is {model_bytes / 1024**3:.1f} GB on {profile.ram_gb:g} GB RAM (>50%): expect swap pressure",
        )
    return rec


def merge_extra_args(current: list[str] | None, recommended: list[str]) -> list[str]:
    """Keep user tokens; ensure recommended flag families are present.

    Families are flag + optional value (``-np 1``, ``-fa on``, ``-lm none``).
    A family already set by the user is left alone; missing families are appended.
    """
    current = list(current or [])
    present: set[str] = set()
    index = 0
    while index < len(current):
        token = current[index]
        present.add(token)
        index += 1
        if index < len(current) and not current[index].startswith("-"):
            index += 1
    out = list(current)
    index = 0
    while index < len(recommended):
        flag = recommended[index]
        index += 1
        value: str | None = None
        if index < len(recommended) and not recommended[index].startswith("-"):
            value = recommended[index]
            index += 1
        if flag in present:
            continue
        # -ctk bundles usually ship with -ctv; if either is present, skip both adds.
        if flag in ("-ctk", "-ctv") and ("-ctk" in present or "-ctv" in present):
            continue
        out.append(flag)
        present.add(flag)
        if value is not None:
            out.append(value)
    return out


def apply_recommendations(cfg, profile: HardwareProfile | None = None) -> list[str]:
    """Fill model params on a Config object. Returns human-readable changes."""
    profile = profile or probe()
    size = 0
    if cfg.model.gguf and os.path.isfile(os.path.expanduser(cfg.model.gguf)):
        size = os.path.getsize(os.path.expanduser(cfg.model.gguf))
    changes = []
    for name, (value, reason) in recommend(profile, size).items():
        if name.startswith("warn_"):
            changes.append(f"warning: {reason}")
            continue
        if name == "extra_args":
            merged = merge_extra_args(cfg.model.extra_args, list(value))
            if merged != list(cfg.model.extra_args or []):
                cfg.model.extra_args = merged
                changes.append(f"extra_args = {merged} ({reason})")
            continue
        if getattr(cfg.model, name, None) != value:
            setattr(cfg.model, name, value)
            changes.append(f"{name} = {value} ({reason})")
    return changes


def default_tune_trials(profile: HardwareProfile | None = None) -> int:
    """Trials per tunable key when the user runs bare ``/tune``.

    8 GB class machines need a larger budget: threads, GPU layers, flags and
    brain.top_k each take several candidates, and the old default of 4 never
    reached past the first groups.
    """
    profile = profile or probe()
    if profile.ram_gb and profile.ram_gb <= 8:
        return 12
    if profile.ram_gb and profile.ram_gb <= 16:
        return 8
    return 6
