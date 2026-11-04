"""Audio decoding, transcription and output writers."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma", ".webm", ".mp4", ".amr"}
SAMPLE_RATE = 16000
FORMATS = ("txt", "md", "srt", "json")


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg/ffprobe not found on PATH. Install ffmpeg first.")


def find_audio(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def probe(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, channel_count)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=channels:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    channels = int(info.get("streams", [{}])[0].get("channels", 1))
    return float(info.get("format", {}).get("duration", 0.0)), channels


def decode(path: Path, channel: int | None, limit: float | None = None) -> np.ndarray:
    """Decode to 16 kHz mono float32. channel=None mixes all channels; limit = max seconds."""
    af = ["-af", f"pan=mono|c0=c{channel}"] if channel is not None else ["-ac", "1"]
    lim = ["-t", str(limit)] if limit else []
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", *lim, "-i", str(path), *af, "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def channels_differ(path: Path) -> bool:
    """True if the first two channels carry different audio (i.e. likely one speaker each)."""
    a, b = decode(path, 0, 120), decode(path, 1, 120)
    n = min(len(a), len(b))
    if n == 0:
        return False
    diff = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
    level = float(np.sqrt(np.mean(a[:n] ** 2))) or 1e-9
    return diff / level > 0.1


def load_model(name: str, device: str = "auto"):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    return WhisperModel(name, device=device, compute_type=compute), device


def transcribe(model, path: Path, *, split: bool, speakers: list[str], language: str | None,
               on_progress: Callable[[float], None] | None = None) -> list[Segment]:
    """Transcribe a file. on_progress receives seconds of audio processed (cumulative across channels)."""
    jobs = [(0, speakers[0]), (1, speakers[1])] if split else [(None, speakers[0] if speakers else "")]
    segments: list[Segment] = []
    done = 0.0
    for channel, label in jobs:
        audio = decode(path, channel)
        result, _ = model.transcribe(audio, language=language, vad_filter=True, beam_size=5,
                                     condition_on_previous_text=False)
        for s in result:
            if s.text.strip():
                segments.append(Segment(s.start, s.end, label, s.text.strip()))
            if on_progress:
                on_progress(done + s.end)
        done += len(audio) / SAMPLE_RATE
        if on_progress:
            on_progress(done)
    segments.sort(key=lambda s: s.start)
    return segments


# ---------- writers ----------

def _hms(t: float) -> str:
    h, r = divmod(int(t), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _turns(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments from the same speaker into one turn."""
    turns: list[Segment] = []
    for s in segments:
        if turns and turns[-1].speaker == s.speaker:
            turns[-1].text += " " + s.text
            turns[-1].end = s.end
        else:
            turns.append(Segment(s.start, s.end, s.speaker, s.text))
    return turns


def render(fmt: str, segments: list[Segment], source: Path, model_name: str) -> str:
    note = f"Automatic transcript (Whisper {model_name}). May contain errors; check against the audio."
    if fmt == "json":
        return json.dumps({"source": source.name, "model": model_name,
                           "segments": [asdict(s) for s in segments]}, indent=2) + "\n"
    if fmt == "srt":
        blocks = []
        for i, s in enumerate(segments, 1):
            who = f"{s.speaker}: " if s.speaker else ""
            blocks.append(f"{i}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{who}{s.text}\n")
        return "\n".join(blocks)
    turns = _turns(segments)
    if fmt == "md":
        lines = [f"# {source.name}", "", f"_{note}_", ""]
        for t in turns:
            who = f"**{t.speaker}**" if t.speaker else ""
            lines += [f"`{_hms(t.start)}` {who}  ", t.text, ""]
        return "\n".join(lines)
    lines = [f"Recording: {source.name}", note, ""]
    for t in turns:
        who = f" {t.speaker}:" if t.speaker else ""
        lines += [f"[{_hms(t.start)}]{who}", t.text, ""]
    return "\n".join(lines)
