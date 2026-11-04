"""calltx: interactive call-recording transcriber."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import __version__, core

console = Console()

MODELS = [
    ("large-v3-turbo", "best speed/accuracy balance (recommended)"),
    ("large-v3", "most accurate, slowest"),
    ("medium", "good, faster"),
    ("small", "fast, lower accuracy"),
    ("base", "very fast, rough drafts"),
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="calltx",
        description="Turn call recordings into speaker-labelled transcripts with Whisper, fully offline. "
                    "Run with no arguments for the interactive wizard.")
    p.add_argument("input", nargs="?", type=Path, help="audio file or folder")
    p.add_argument("-o", "--output", type=Path, help="output folder (default: <input>/transcripts)")
    p.add_argument("-m", "--model", default="large-v3-turbo", help="Whisper model (default: large-v3-turbo)")
    p.add_argument("-l", "--language", default="en", help="language code, or 'auto' (default: en)")
    p.add_argument("-f", "--format", default="txt", help=f"comma-separated: {','.join(core.FORMATS)} (default: txt)")
    p.add_argument("--channels", choices=["auto", "split", "mix"], default="auto",
                   help="split = one speaker per stereo channel; mix = mono (default: auto-detect)")
    p.add_argument("--speakers", default="Speaker 1,Speaker 2", help="labels for left,right channels")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--overwrite", action="store_true", help="redo files that already have transcripts")
    p.add_argument("-y", "--yes", action="store_true", help="don't prompt; use flags/defaults")
    p.add_argument("-V", "--version", action="version", version=f"calltx {__version__}")
    return p.parse_args(argv)


def wizard(args: argparse.Namespace) -> argparse.Namespace:
    import questionary as q

    def ask(question):
        answer = question.ask()
        if answer is None:  # Ctrl-C
            raise KeyboardInterrupt
        return answer

    console.print(Panel.fit("[bold]calltx[/] — call recordings → transcripts\n"
                            "[dim]Runs locally with Whisper. Nothing is uploaded.[/]"))
    if args.input is None:
        args.input = Path(ask(q.path("Audio file or folder:", only_directories=False,
                                     validate=lambda s: Path(s).expanduser().exists() or "Path not found")
                              )).expanduser()

    files = core.find_audio(args.input)
    if not files:
        raise SystemExit(f"No audio files found in {args.input}")
    console.print(f"Found [bold]{len(files)}[/] audio file(s).")

    _, ch = core.probe(files[0])
    if ch >= 2:
        separate = core.channels_differ(files[0])
        hint = "channels look different — probably one speaker each" if separate else "channels look identical"
        args.channels = ask(q.select(
            f"Recordings are stereo ({hint}). How should speakers be handled?",
            choices=[q.Choice("Split: one speaker per channel", "split"),
                     q.Choice("Mix to mono: no speaker labels", "mix")],
            default="split" if separate else "mix"))
    else:
        args.channels = "mix"

    if args.channels == "split":
        left = ask(q.text("Label for left channel:", default="Speaker 1"))
        right = ask(q.text("Label for right channel:", default="Speaker 2"))
        args.speakers = f"{left},{right}"

    args.model = ask(q.select("Model:", choices=[q.Choice(f"{n}  —  {d}", n) for n, d in MODELS],
                              default=args.model))
    args.language = ask(q.text("Language code (en, fr, de… or 'auto'):", default=args.language))
    fmts = ask(q.checkbox("Output formats:", choices=[
        q.Choice("txt  — plain text conversation", "txt", checked=True),
        q.Choice("md   — Markdown", "md"),
        q.Choice("srt  — subtitles with timestamps", "srt"),
        q.Choice("json — machine-readable segments", "json")],
        validate=lambda a: bool(a) or "Pick at least one"))
    args.format = ",".join(fmts)
    default_out = (args.input if args.input.is_dir() else args.input.parent) / "transcripts"
    args.output = Path(ask(q.path("Output folder:", default=str(args.output or default_out),
                                  only_directories=True))).expanduser()
    return args


def run(args: argparse.Namespace) -> int:
    core.require_ffmpeg()
    files = core.find_audio(args.input)
    if not files:
        console.print(f"[red]No audio files found in {args.input}[/]")
        return 1
    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    bad = [f for f in fmts if f not in core.FORMATS]
    if bad:
        console.print(f"[red]Unknown format(s): {', '.join(bad)}[/]")
        return 1
    out = args.output or (args.input if args.input.is_dir() else args.input.parent) / "transcripts"
    out.mkdir(parents=True, exist_ok=True)
    speakers = [s.strip() for s in args.speakers.split(",")] + ["Speaker 2"]
    language = None if args.language == "auto" else args.language

    plan = []
    for f in files:
        duration, ch = core.probe(f)
        split = {"split": ch >= 2, "mix": False, "auto": ch >= 2 and core.channels_differ(f)}[args.channels]
        targets = [out / f"{f.stem}.{fmt}" for fmt in fmts]
        skip = not args.overwrite and all(t.exists() for t in targets)
        plan.append((f, duration, split, skip))

    table = Table(title="Plan", show_lines=False)
    for col in ("File", "Length", "Speakers", "Status"):
        table.add_column(col)
    for f, d, split, skip in plan:
        table.add_row(f.name, core._hms(d), "split" if split else "mono",
                      "[dim]skip (exists)[/]" if skip else "queued")
    console.print(table)
    todo = [p for p in plan if not p[3]]
    if not todo:
        console.print("Nothing to do.")
        return 0

    if not args.yes and sys.stdin.isatty():
        import questionary as q
        if not q.confirm(f"Transcribe {len(todo)} file(s) to {out}?", default=True).ask():
            return 1

    with console.status(f"Loading model [bold]{args.model}[/] (first run downloads it)…"):
        model, device = core.load_model(args.model, args.device)
    console.print(f"Model ready on [bold]{device}[/].")

    total = sum(d * (2 if split else 1) for _, d, split, _ in todo) or 1
    columns = [TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
               TimeElapsedColumn(), TimeRemainingColumn()]
    started = time.time()
    with Progress(*columns, console=console) as progress:
        overall = progress.add_task("All files", total=total)
        base = 0.0
        for f, d, split, _ in todo:
            work = d * (2 if split else 1) or 1
            task = progress.add_task(f.name, total=work)

            def tick(sec, task=task, base=base, work=work):
                progress.update(task, completed=min(sec, work))
                progress.update(overall, completed=base + min(sec, work))

            try:
                segs = core.transcribe(model, f, split=split, speakers=speakers[:2],
                                       language=language, on_progress=tick)
            except Exception as e:  # keep going on bad files
                progress.console.print(f"[red]✗ {f.name}: {e}[/]")
            else:
                for fmt in fmts:
                    (out / f"{f.stem}.{fmt}").write_text(core.render(fmt, segs, f, args.model))
                progress.console.print(f"[green]✓[/] {f.name}")
            base += work
            progress.update(overall, completed=base)
            progress.remove_task(task)

    console.print(f"\nDone in {core._hms(time.time() - started)}. Transcripts in [bold]{out}[/]")
    return 0


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        if args.input is None and not args.yes and sys.stdin.isatty():
            args = wizard(args)
        elif args.input is None:
            console.print("[red]No input given.[/] Pass a file/folder, or run interactively.")
            sys.exit(2)
        sys.exit(run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        sys.exit(130)


if __name__ == "__main__":
    main()
