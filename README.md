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
