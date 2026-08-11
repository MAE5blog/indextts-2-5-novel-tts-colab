"""Resumable long-form IndexTTS 2.5 rendering helpers for Google Colab.

This module is deliberately independent of the VoxCPM2 project.  It creates a
small manifest, renders one segment at a time, and records completion only
after the WAV has been validated and atomically moved into place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


MANIFEST_VERSION = 1
DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_MODEL_ID = "IndexTeam/IndexTTS-2.5"


@dataclass(frozen=True)
class SegmentSettings:
    """Text segmentation and inference settings stored in every manifest."""

    target_chars: int = 60
    hard_chars: int = 80
    min_chars: int = 8
    sentence_pause_ms: int = 160
    paragraph_pause_ms: int = 360
    max_text_tokens_per_segment: int = 80
    interval_silence_ms: int = 160

    def validate(self) -> None:
        if not 20 <= self.target_chars <= 180:
            raise ValueError("target_chars must be between 20 and 180")
        if not self.target_chars <= self.hard_chars <= 240:
            raise ValueError("hard_chars must be between target_chars and 240")
        if not 1 <= self.min_chars <= self.target_chars:
            raise ValueError("min_chars must be between 1 and target_chars")
        if not 32 <= self.max_text_tokens_per_segment <= 120:
            raise ValueError("max_text_tokens_per_segment must be between 32 and 120")
        if self.sentence_pause_ms < 0 or self.paragraph_pause_ms < 0 or self.interval_silence_ms < 0:
            raise ValueError("pause values must not be negative")


@dataclass(frozen=True)
class PlannedSegment:
    text: str
    paragraph_end: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, target)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_text(path: str | Path) -> str:
    """Load a UTF-8/GB18030 text or Markdown manuscript without external OCR."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input text not found: {source}")
    if source.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise ValueError("This lightweight Colab workflow accepts .txt, .md, or .markdown files")
    text = _decode_text(source.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    # Keep headings as ordinary spoken text, but remove obvious Markdown decorations.
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_~]+", "", text)
    if not text.strip():
        raise ValueError("Input text is empty")
    return text.strip()


def _split_oversized(text: str, hard_chars: int) -> list[str]:
    remaining = text.strip()
    pieces: list[str] = []
    while len(remaining) > hard_chars:
        window = remaining[: hard_chars + 1]
        cut = max(window.rfind(mark) for mark in "，,；;：:、")
        if cut < max(8, hard_chars // 3):
            cut = window.rfind(" ")
        if cut < max(8, hard_chars // 3):
            cut = hard_chars
        else:
            cut += 1
        piece = remaining[:cut].strip()
        if not piece:
            cut = hard_chars
            piece = remaining[:cut].strip()
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_text(text: str, settings: SegmentSettings) -> list[PlannedSegment]:
    """Split Chinese-centric prose at sentence boundaries with a strict fallback."""

    settings.validate()
    result: list[PlannedSegment] = []
    paragraphs = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n\s*\n+", text) if item.strip()]
    for paragraph in paragraphs:
        units = [unit.strip() for unit in re.split(r"(?<=[。！？!?…])", paragraph) if unit.strip()]
        prepared: list[str] = []
        for unit in units:
            prepared.extend(_split_oversized(unit, settings.hard_chars) if len(unit) > settings.hard_chars else [unit])
        current = ""
        current_items: list[str] = []
        for unit in prepared:
            separator = " " if current and current[-1].isascii() and unit[0].isascii() else ""
            candidate = current + separator + unit
            if current and len(candidate) > settings.target_chars:
                result.append(PlannedSegment(current, False))
                current, current_items = unit, [unit]
            else:
                current = candidate
                current_items.append(unit)
            if len(current) >= settings.hard_chars:
                result.append(PlannedSegment(current, False))
                current, current_items = "", []
        if current:
            result.append(PlannedSegment(current, True))
        elif result:
            last = result[-1]
            result[-1] = PlannedSegment(last.text, True)
    # Absorb tiny orphan sentences when it does not violate the hard limit.
    merged: list[PlannedSegment] = []
    for item in result:
        if merged and len(item.text) < settings.min_chars:
            previous = merged[-1]
            separator = " " if previous.text[-1].isascii() and item.text[0].isascii() else ""
            joined = previous.text + separator + item.text
            if len(joined) <= settings.hard_chars:
                merged[-1] = PlannedSegment(joined, previous.paragraph_end or item.paragraph_end)
                continue
        merged.append(item)
    return [item for item in merged if item.text.strip()]


def _manifest_signature(source_sha: str, reference_sha: str, settings: SegmentSettings, model_id: str, model_revision: str) -> str:
    payload = {
        "source_sha256": source_sha,
        "reference_sha256": reference_sha,
        "settings": asdict(settings),
        "model_id": model_id,
        "model_revision": model_revision,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def create_or_load_manifest(
    job_dir: str | Path,
    text_path: str | Path,
    reference_audio: str | Path,
    settings: SegmentSettings,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = "main",
    force_new: bool = False,
) -> dict[str, Any]:
    """Plan a new job or safely reopen an identical interrupted job."""

    settings.validate()
    root = Path(job_dir)
    source = Path(text_path)
    reference = Path(reference_audio)
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference}")
    text = load_text(source)
    source_sha = sha256_text(text)
    reference_sha = sha256_file(reference)
    signature = _manifest_signature(source_sha, reference_sha, settings, model_id, model_revision)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force_new:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("version") != MANIFEST_VERSION:
            raise ValueError("Existing manifest uses an unsupported version")
        if existing.get("config_signature") != signature:
            raise ValueError("Job inputs or settings changed; choose a new job name or use --force-new")
        return existing
    if manifest_path.is_file() and force_new:
        archive = root / f"manifest.replaced-{int(time.time())}.json"
        shutil.move(str(manifest_path), str(archive))

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(split_text(text, settings), start=1):
        segment_id = f"s{index:06d}"
        segments.append(
            {
                "id": segment_id,
                "text": item.text,
                "text_sha256": sha256_text(item.text),
                "pause_ms": settings.paragraph_pause_ms if item.paragraph_end else settings.sentence_pause_ms,
                "output_relpath": str(Path("segments") / f"{segment_id}.wav"),
                "status": "pending",
                "attempts": 0,
                "duration_seconds": None,
                "error": None,
            }
        )
    if not segments:
        raise ValueError("No renderable segments were found")
    manifest = {
        "version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "status": "planned",
        "model_id": model_id,
        "model_revision": model_revision,
        "source_path": source.name,
        "source_sha256": source_sha,
        "reference_audio": reference.name,
        "reference_sha256": reference_sha,
        "settings": asdict(settings),
        "config_signature": signature,
        "segments": segments,
        "events": [{"at": utc_now(), "message": f"Planned {len(segments)} segments"}],
    }
    save_manifest(root, manifest, None)
    return manifest


def save_manifest(job_dir: str | Path, manifest: dict[str, Any], event: str | None) -> None:
    manifest["updated_at"] = utc_now()
    if event:
        manifest.setdefault("events", []).append({"at": utc_now(), "message": event})
    write_json_atomic(Path(job_dir) / "manifest.json", manifest)


def manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in manifest.get("segments", []):
        counts[item.get("status", "unknown")] = counts.get(item.get("status", "unknown"), 0) + 1
    return {"status": manifest.get("status"), "segments": len(manifest.get("segments", [])), "counts": counts}


def cuda_preflight() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "reason": "PyTorch is not installed in the IndexTTS environment"}
    if not torch.cuda.is_available():
        return {"cuda_available": False, "reason": "No CUDA GPU is attached"}
    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "cuda_available": True,
        "name": props.name,
        "total_vram_gib": round(props.total_memory / 1024**3, 2),
        "compute_capability": f"{major}.{minor}",
        "native_bf16": major >= 8 and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        "t4_experimental": major == 7 and minor == 5,
    }


def download_checkpoint(model_dir: str | Path, *, model_id: str = DEFAULT_MODEL_ID, revision: str = "main") -> Path:
    """Download the official checkpoint without altering the shared HF cache."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install IndexTTS into its uv environment before downloading checkpoints") from exc
    destination = Path(model_dir)
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(destination))
    if not (destination / "config.yaml").is_file():
        raise RuntimeError("IndexTTS-2.5 checkpoint is missing config.yaml after download")
    return destination


def load_indextts25(model_dir: str | Path) -> Any:
    """Load official IndexTTS 2.5 with T4-safe options and no Qwen emotion model."""

    preflight = cuda_preflight()
    if not preflight.get("cuda_available"):
        raise RuntimeError(f"A CUDA GPU is required: {preflight.get('reason', 'unknown CUDA error')}")
    directory = Path(model_dir)
    config = directory / "config.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config}")
    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ImportError as exc:
        raise RuntimeError("The IndexTTS source tree is not on PYTHONPATH; rerun the Notebook install cell") from exc
    return IndexTTS2(
        cfg_path=str(config),
        model_dir=str(directory),
        use_bf16=False,
        device="cuda:0",
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )


def wave_info(path: str | Path) -> dict[str, int] | None:
    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 44:
        return None
    try:
        with wave.open(str(target), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            if frames <= 0 or rate <= 0:
                return None
            return {"frames": frames, "sample_rate": rate, "channels": handle.getnchannels()}
    except (wave.Error, OSError):
        return None


def _render_one(
    tts: Any,
    record: dict[str, Any],
    reference_audio: Path,
    output_path: Path,
    settings: SegmentSettings,
    *,
    lang: str,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    temporary.unlink(missing_ok=True)
    result = tts.infer(
        spk_audio_prompt=str(reference_audio),
        text=record["text"],
        output_path=str(temporary),
        lang=lang,
        use_emo_text=False,
        use_random=False,
        interval_silence=settings.interval_silence_ms,
        max_text_tokens_per_segment=settings.max_text_tokens_per_segment,
        stream_return=False,
        verbose=False,
    )
    if result is not None and Path(str(result)) != temporary:
        # The official API normally returns output_path.  Accept None for API
        # compatibility but reject an unrelated path so a manifest never lies.
        raise RuntimeError(f"IndexTTS returned an unexpected output path: {result}")
    info = wave_info(temporary)
    if info is None:
        raise RuntimeError("IndexTTS did not produce a valid non-empty WAV")
    os.replace(temporary, output_path)
    return info


def render_pending_segments(
    job_dir: str | Path,
    manifest: dict[str, Any],
    tts: Any,
    reference_audio: str | Path,
    *,
    lang: str = "ZH",
    max_segments: int | None = None,
    retry_failed: bool = False,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Render pending records serially; a repeat call resumes at the next WAV."""

    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be positive")
    root = Path(job_dir)
    reference = Path(reference_audio)
    if not reference.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference}")
    settings = SegmentSettings(**manifest["settings"])
    settings.validate()
    manifest["status"] = "running"
    save_manifest(root, manifest, "Rendering started or resumed")
    completed_this_call = 0
    for record in manifest["segments"]:
        output_path = root / record["output_relpath"]
        valid = wave_info(output_path)
        if record.get("status") == "completed" and valid:
            continue
        if record.get("status") == "failed" and not retry_failed:
            continue
        if max_segments is not None and completed_this_call >= max_segments:
            manifest["status"] = "partial"
            save_manifest(root, manifest, f"Stopped after {completed_this_call} requested segment(s)")
            break
        record["status"] = "running"
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["error"] = None
        save_manifest(root, manifest, f"Rendering {record['id']}")
        try:
            info = _render_one(tts, record, reference, output_path, settings, lang=lang)
            record["status"] = "completed"
            record["duration_seconds"] = round(info["frames"] / info["sample_rate"], 3)
            save_manifest(root, manifest, f"Completed {record['id']}")
            completed_this_call += 1
            if progress:
                complete = sum(1 for item in manifest["segments"] if item.get("status") == "completed")
                progress(record, complete, len(manifest["segments"]))
        except KeyboardInterrupt:
            record["status"] = "pending"
            manifest["status"] = "interrupted"
            save_manifest(root, manifest, f"Interrupted while rendering {record['id']}")
            raise
        except Exception as exc:
            output_path.with_name(output_path.stem + ".partial" + output_path.suffix).unlink(missing_ok=True)
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            save_manifest(root, manifest, f"Failed {record['id']}: {record['error']}")
            raise
    if all(record.get("status") == "completed" and wave_info(root / record["output_relpath"]) for record in manifest["segments"]):
        manifest["status"] = "rendered"
        save_manifest(root, manifest, "All segments completed")
    return manifest


def run_smoke_test(
    model_dir: str | Path,
    reference_audio: str | Path,
    output_path: str | Path,
    *,
    text: str = "这是 IndexTTS 二点五在 Google Colab 上的短句测试。",
    lang: str = "ZH",
) -> dict[str, Any]:
    settings = SegmentSettings()
    tts = load_indextts25(model_dir)
    record = {"id": "smoke", "text": text}
    info = _render_one(tts, record, Path(reference_audio), Path(output_path), settings, lang=lang)
    return {"output": str(Path(output_path)), **info}


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is unavailable; rerun the Notebook setup cell")
    return executable


def _write_silence(path: Path, milliseconds: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    frames = max(1, round(sample_rate * milliseconds / 1000))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def merge_completed_segments(job_dir: str | Path, manifest: dict[str, Any], output_wav: str | Path) -> Path:
    """Concatenate validated segments with recorded punctuation pauses via ffmpeg."""

    root = Path(job_dir)
    records = manifest.get("segments", [])
    if not records or any(record.get("status") != "completed" or not wave_info(root / record["output_relpath"]) for record in records):
        raise ValueError("All segments must be completed before merging")
    work = root / "merge"
    work.mkdir(parents=True, exist_ok=True)
    concat = work / "concat.txt"
    entries: list[Path] = []
    for index, record in enumerate(records):
        entries.append(root / record["output_relpath"])
        if index < len(records) - 1 and int(record.get("pause_ms", 0)):
            silence = work / f"silence_{index:06d}.wav"
            _write_silence(silence, int(record["pause_ms"]))
            entries.append(silence)
    concat.write_text("\n".join(f"file '{item.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for item in entries) + "\n", encoding="utf-8")
    target = Path(output_wav)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    partial.unlink(missing_ok=True)
    subprocess.run([_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "pcm_s16le", str(partial)], check=True)
    os.replace(partial, target)
    return target


def export_m4b(source_wav: str | Path, output_m4b: str | Path) -> Path:
    source = Path(source_wav)
    if wave_info(source) is None:
        raise ValueError(f"Input WAV is missing or invalid: {source}")
    target = Path(output_m4b)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial" + target.suffix)
    partial.unlink(missing_ok=True)
    subprocess.run([_ffmpeg(), "-y", "-i", str(source), "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(partial)], check=True)
    os.replace(partial, target)
    return target


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    download = commands.add_parser("download")
    download.add_argument("--model-dir", required=True)
    download.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    download.add_argument("--revision", default="main")
    plan = commands.add_parser("plan")
    plan.add_argument("--job-dir", required=True)
    plan.add_argument("--text", required=True)
    plan.add_argument("--reference-audio", required=True)
    plan.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    plan.add_argument("--model-revision", default="main")
    plan.add_argument("--target-chars", type=int, default=60)
    plan.add_argument("--hard-chars", type=int, default=80)
    plan.add_argument("--max-tokens", type=int, default=80)
    plan.add_argument("--force-new", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--model-dir", required=True)
    smoke.add_argument("--reference-audio", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--text", default="这是 IndexTTS 二点五在 Google Colab 上的短句测试。")
    smoke.add_argument("--lang", default="ZH")
    render = commands.add_parser("render")
    render.add_argument("--job-dir", required=True)
    render.add_argument("--model-dir", required=True)
    render.add_argument("--reference-audio", required=True)
    render.add_argument("--lang", default="ZH")
    render.add_argument("--limit", type=int)
    render.add_argument("--retry-failed", action="store_true")
    merge = commands.add_parser("merge")
    merge.add_argument("--job-dir", required=True)
    merge.add_argument("--wav", required=True)
    merge.add_argument("--m4b")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        _print_json(cuda_preflight())
        return 0
    if args.command == "download":
        _print_json({"model_dir": str(download_checkpoint(args.model_dir, model_id=args.model_id, revision=args.revision))})
        return 0
    if args.command == "plan":
        settings = SegmentSettings(target_chars=args.target_chars, hard_chars=args.hard_chars, max_text_tokens_per_segment=args.max_tokens)
        _print_json(manifest_summary(create_or_load_manifest(args.job_dir, args.text, args.reference_audio, settings, model_id=args.model_id, model_revision=args.model_revision, force_new=args.force_new)))
        return 0
    if args.command == "smoke":
        _print_json(run_smoke_test(args.model_dir, args.reference_audio, args.output, text=args.text, lang=args.lang))
        return 0
    if args.command == "render":
        root = Path(args.job_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        tts = load_indextts25(args.model_dir)
        def report(record: dict[str, Any], completed: int, total: int) -> None:
            print(f"[{completed}/{total}] {record['id']} {record.get('duration_seconds', 0):.2f}s", flush=True)
        _print_json(manifest_summary(render_pending_segments(root, manifest, tts, args.reference_audio, lang=args.lang, max_segments=args.limit, retry_failed=args.retry_failed, progress=report)))
        return 0
    if args.command == "merge":
        root = Path(args.job_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        wav = merge_completed_segments(root, manifest, args.wav)
        result: dict[str, str] = {"wav": str(wav)}
        if args.m4b:
            result["m4b"] = str(export_m4b(wav, args.m4b))
        _print_json(result)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
