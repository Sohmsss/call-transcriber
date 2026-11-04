# calltx

Interactive command-line tool that turns phone or call recordings into **speaker-labelled transcripts** using [Whisper](https://github.com/openai/whisper) (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)).

- **Runs on your own computer.** Your audio is never uploaded. Good for sensitive recordings such as DSAR call logs, complaints and interviews.
- **Splits stereo calls by channel.** Many call systems record each side of the call on its own channel. calltx detects this, transcribes each side separately, and merges them into a timestamped conversation.
- **Interactive wizard** or plain flags for scripting.
- Output as `txt`, `md`, `srt` or `json`.
- Skips files that already have transcripts, so you can stop and resume.

## Install

You need [ffmpeg](https://ffmpeg.org/) and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install --python 3.12 git+https://github.com/Sohmsss/call-transcriber
```

## Use

Interactive:

```bash
calltx
```

The wizard asks for the file or folder, how to handle speakers (detecting stereo automatically), speaker labels, model, language, output formats and output folder. It then shows a plan and a progress bar.

Non-interactive:

```bash
calltx ./Calls -o ./Transcripts --speakers "Agent,Customer" -f txt,srt -y
```

| Option | Default | |
|---|---|---|
| `-o, --output` | `<input>/transcripts` | output folder |
| `-m, --model` | `large-v3-turbo` | `large-v3`, `medium`, `small`, `base`, … |
| `-l, --language` | `en` | language code or `auto` |
| `-f, --format` | `txt` | any of `txt,md,srt,json` |
| `--channels` | `auto` | `split` (one speaker per channel), `mix` (mono) |
| `--speakers` | `Speaker 1,Speaker 2` | labels for left and right channels |
| `--device` | `auto` | `cpu` or `cuda` |
| `--overwrite` | off | redo existing transcripts |
| `-y, --yes` | off | no prompts |

## Example output

```
Recording: call-001.wav
Automatic transcript (Whisper large-v3-turbo). May contain errors; check against the audio.

[00:00:00] Agent:
Good morning, you're through to the support team. Can I take your reference number?

[00:00:05] Customer:
Yes, it's 4-4-1-2...
```

## Notes

- The first run downloads the model, about 1.6 GB for `large-v3-turbo`.
- On CPU, `large-v3-turbo` runs at roughly 3–5× real time on a modern 8–16 core laptop. With an NVIDIA GPU it's much faster.
- Speaker labelling relies on the two sides being on separate channels. Mono recordings are transcribed without speaker labels.
- Transcripts are automatic and can mishear words, especially on 8 kHz phone audio. Check important quotes against the recording.

## License

MIT
