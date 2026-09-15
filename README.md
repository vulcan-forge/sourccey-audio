# Sourccey Voice

Reusable, networked speech interaction for Sourccey. This repository contains
the robot audio relay, speech recognition, conversation runtime, wake-name
handling, and the host protocol. It intentionally contains no voice models,
saved voice presets, generated audio, or desktop voice-design tools.

## Install

Install the required runtime features on the host:

```bash
uv sync --extra voice
```

The robot only needs its audio dependencies:

```bash
uv sync --extra audio
```

## Optional Desktop Voice Runtime

Text-to-speech is an external runtime. A desktop installer may place that
runtime and its models anywhere; the module does not assume a sibling folder
or a particular user name. Set these before launching the host:

```powershell
$env:SOURCCEY_VOICE_RUNTIME_PATH = "D:\\path\\to\\DesktopVoiceRuntime"
$env:SOURCCEY_VOICE_PRESET = "D:\\path\\to\\SourcceyVoiceTest02.json"
```

The default factory is `sourccey_desktop_voice.runtime:create_tts`. A different
external TTS provider can be selected with the `tts.factory` setting.

## Run

On the desktop host:

```bash
uv run --extra voice sourccey-voice --config config/host.toml host
```

On the robot:

```bash
.venv/bin/sourccey-voice --config config/robot.toml robot-audio
```

`config/host.toml` deliberately leaves model and TTS paths blank. Configure
them through your installer, environment, or an ignored machine-local config.

## Development

```bash
uv run pytest
```

Voice authoring utilities and large runtime assets belong in the separate
desktop tools/downloadables distribution, not in this module.

## Listening and wake-name handling

The default requires a leading Sourccey address on every request. Text matching
uses the original transcript and leaves the request unchanged:

- Exact names and configured phonetic spellings (including `source see` and
  `source C`) can wake the robot.
- Ambiguous spellings such as `Mercy` or `Cersei` require a following request,
  such as `can you ...` or `please ...`. Configure these in
  `wake.contextual_aliases`, or set that list to `[]` to disable them.
- A tight spelling fallback applies only to a leading name followed by a
  request. Similarity is a text heuristic, not acoustic confidence.
- A name alone grants one continuation if the next utterance starts within
  `wake.continuation_seconds` (default 2 seconds). That permission is consumed
  once, including by an empty/noisy turn. Set it to 0 to require the name and
  request in the same utterance. This does not identify the next speaker.
- `wake.engaged_seconds = 0` prevents an ongoing conversation window from
  admitting later background speech.

Speech is segmented at pauses, with pre-roll preserving the beginning of the
name. Adaptive energy handling estimates background level between turns and
requires a relative drop from the current voice before treating high-VAD-score
audio as silence. It is a heuristic that must be tested with the actual mic.
The old `always_listen_window_ms` field is accepted for compatibility but no
longer splits speech at arbitrary boundaries. `vad.max_utterance_ms` aborts
overlong input without executing a partial command. Reconnects discard pending
audio and wake permission.

Logs separate `[STT raw]`, `[WAKE]` decisions, and `[AUDIO turn]` levels/clipping.
No recordings are saved automatically. These diagnostics measure the audio
received from the robot, after its audio processing.

### Replay a microphone recording

**Desktop Git Bash**, from this repo, with the usual model-path environment
variables set:

```bash
uv run --extra voice sourccey-voice --config config/host.toml replay /path/to/robot-test.wav
```

Use mono PCM16 WAV at the configured sample rate (normally 16000 Hz). Replay
loads STT and VAD only. It does not load the LLM/TTS runtime, play sound, or
connect to/actuate a robot. It prints original transcripts, wake decisions, and
a JSON result summary. Timers use the recording's audio timeline; trailing
silence is explicitly added at EOF. Processing time excludes model loading and
is not end-to-end response latency. Use recordings with natural pauses before
EOF when assessing endpoint behavior.

Keep recordings outside the repo. Test both addressed requests and ordinary
room conversation. Track missed wake names, accidental activations per hour,
request transcription accuracy, and time from speech end to audible reply.
Keep some recordings out of tuning and use them only for evaluation.

### Next acoustic upgrade

A custom wake-word model for the sound "source-see" can detect the address
before ASR picks a spelling. [openWakeWord](https://github.com/dscripka/openWakeWord)
supports custom models and optional speaker-specific verifiers. That model is
not trained or included here; it should be an external downloadable asset,
evaluated against robot-mic positive examples and background negatives.

Text matching cannot distinguish a real address from someone discussing a
person named Cersei, nor recover a word masked by a second speaker. A verifier
for a known user can reduce accidental wakes, with the tradeoff of rejecting
unfamiliar voices. The conversation LLM should not invent missing command
words or supply evidence that the wake name was spoken.
