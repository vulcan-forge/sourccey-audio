# Sourccey Voice

`sourccey-voice` is a standalone, local voice interaction package for Sourccey. It captures
and plays audio on the robot, runs the intelligence stack on the client/host, and puts every
robot action behind a deterministic allowlisted command registry.

No cloud API is required. Model files are downloaded once and remain outside Git.

## Safety boundary

The conversational model never receives a motor API, Python evaluator, shell, protobuf
encoder, or raw network sender.

```text
human speech -> STT -> router -> approved command registry -> robot adapter
                            |
                            +-> local LLM -> spoken text
                                   |
                                   +-> requested_action -> registry validation
```

The rule is: AI interprets humans, the registry validates requests, and Sourccey's robotics
software controls the robot. Unknown explicit commands and malformed or invented LLM actions
are rejected. Explicit commands still work if the LLM is unavailable. A handler exception is
reported as failure and is never announced as success.

## Architecture

```text
Sourccey robot                                      Desktop / host

microphone -> WebRTC AEC/NS/AGC -> PCM16 websocket -> jitter-tolerant chunks
                    ^                                      |
                    |                                Silero probability
speaker <- PCM16 playback <- Kokoro <- response <- VAD segment state machine
                    |                                      |
              far-end reference                    Moonshine streaming STT
                                                           |
                           +-------------------------------+------------------+
                           |                                                  |
                    Command: prefix                                  ordinary speech
                           |                                                  |
                 deterministic aliases                               Qwen3-4B GGUF
                           |                                                  |
                 approved registry <----- optional requested_action JSON ----+
                           |
                  RobotCommandAdapter
```

The websocket protocol uses versioned JSON envelopes, sequence numbers, wall-clock timestamps,
message size limits, base64 PCM16 payloads, authentication, and stale-audio rejection. It never
transmits executable code. TTS is split into 500 ms messages to bound frame size and playback
latency.

## Repository fit

The inspected Sourccey stack is Python 3.12+, with ZeroMQ and protobuf between
`SourcceyClient` and `SourcceyHost`, plus a websocket relay for remote robot operation.
The current protobuf has joint and base targets but no semantic `FOLLOW_ME`, `STOP`, `CANCEL`,
or `FREEZE` message. It also defaults omitted arm fields to zero.

`SourcceyHostConfig` reserves ZeroMQ ports 5557-5559 for a future text/audio pipeline, but the
inspected repositories contain no socket setup, message definition, or runtime using those fields.
They are not treated as a working protocol.

For that reason this package does not inject a partial zero-velocity protobuf. The optional
`sourccey_client` adapter first obtains a fresh complete arm observation, resends those hold
positions, and sets base velocity to zero. It exposes only `STOP`, `CANCEL`, and `FREEZE`.
`FOLLOW_ME` is registered for forward compatibility but remains unavailable until the robotics
stack exposes a real follow-me capability.

The adapter requires an exclusive `SourcceyClient` connection because the existing observation
socket is `PUSH/PULL`, not a broadcast feed. Do not run it alongside another teleoperation client.

## Models and libraries

| Stage | Initial backend | Reason | Approximate cache |
| --- | --- | --- | --- |
| VAD | Silero VAD 6 | Small streaming probability model with a simple local API | a few MB |
| STT | Moonshine Medium Streaming | On-device, cross-platform, designed for incremental transcription | up to about 1.1 GB |
| LLM | Qwen3-4B `Q4_K_M` through `llama-cpp-python` | Compact local instruction model with CPU/GPU support | 2.5 GB |
| TTS | Kokoro-82M | Small local model with clear output and configurable voices | a few hundred MB |
| AEC | `pywebrtc-audio` AEC3 | Accepts microphone audio plus the exact outgoing speaker reference | package-sized |

The selected English Moonshine code and streaming model are MIT licensed. Silero VAD is MIT.
Qwen3-4B GGUF and Kokoro are Apache-2.0. Re-check each upstream model card before changing
models, especially for non-English Moonshine assets. See the upstream
[Moonshine repository](https://github.com/moonshine-ai/moonshine),
[Moonshine Medium model](https://huggingface.co/UsefulSensors/moonshine-streaming-medium),
[Qwen3-4B GGUF model card](https://huggingface.co/Qwen/Qwen3-4B-GGUF),
[Kokoro repository](https://github.com/hexgrad/kokoro), and
[Silero VAD repository](https://github.com/snakers4/silero-vad).

## Install for development

From this repository:

```powershell
uv sync --extra dev
uv run pytest
uv run sourccey-voice --config config/default.toml dev
```

Developer mode loads no AI models and touches no robot hardware:

```text
> hello
Sourccey: Hi! What are we building today?
> Command: follow me
COMMAND REQUEST: FOLLOW_ME
```

A one-shot invocation is useful in scripts:

```powershell
uv run sourccey-voice --config config/default.toml dev --once "Command: stop"
```

## Install into Sourccey Desktop

The installer uses the existing Desktop `modules/lerobot-vulcan/.venv` environment and puts the
editable package there. It creates one persistent configuration file and does not edit Desktop
source code.

```powershell
.\scripts\install.ps1 `
  -DesktopPath C:\Users\Theor\Documents\WebsiteCode\VulcanDesktop\sourccey-desktop `
  -WithVoiceDependencies
```

The resulting config is:

```text
sourccey-desktop/modules/sourccey-voice/config.toml
```

Set a random `network.auth_token`, model paths, the desktop listen address, and the robot URL.
Validate it before startup:

```powershell
sourccey-voice --config path\to\config.toml config-check
```

The current Desktop app has no general third-party module lifecycle API. The package includes
`sourccey-module.toml` as the launch contract, but Desktop does not yet discover it. Until a thin
Tauri controller is added, start the host command as a managed external process or from a shell:

```powershell
sourccey-voice --config path\to\config.toml host
```

## Install on the robot

Install only the core and robot audio dependencies in a Python 3.12 environment:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e ".[audio]"
cp config/default.toml config/robot.toml
```

In `config/robot.toml`, set the same authentication token and set `network.robot_url` to the
desktop address, for example `ws://192.168.1.20:8766`. Then run:

```bash
.venv/bin/sourccey-voice --config config/robot.toml robot-audio
```

The robot agent reconnects after disconnects and clears pending playback. The host rejects stale
audio rather than replaying old speech after reconnect.

## Model setup

On the desktop, install the full optional set and warm each cache while internet is available:

```powershell
sourccey-voice --config path\to\config.toml models download stt
sourccey-voice --config path\to\config.toml models download llm
sourccey-voice --config path\to\config.toml models download tts
```

The STT and LLM commands print the exact paths to put in `stt.model_path` and
`llm.model_path`. By default downloads live below:

```text
~/.cache/sourccey-voice/models
```

Override that location with `SOURCCEY_VOICE_MODEL_CACHE`. Moonshine, Hugging Face, and Kokoro
may also maintain their own content-addressed cache entries beneath it. After the download,
startup and inference are offline. Do not copy these caches into Git.

## Configuration

All settings are in one TOML file. The packaged default and `config/default.toml` are equivalent.
Set `SOURCCEY_VOICE_CONFIG` to avoid passing `--config` on each command.

Important groups:

| Table | Controls |
| --- | --- |
| `audio` | input/output device, 16 kHz mono frames, volume, AEC, NS, AGC, delay |
| `network` | host/port, robot URL, token, size/staleness limits, reconnect delay |
| `vad` | Silero threshold, minimum speech, silence timeout, pre-roll, post-roll |
| `stt` | backend, language, model path, streaming architecture and partial interval |
| `llm` | backend, GGUF path, device, context size, response size, temperature |
| `tts` | Kokoro voice, language, speed, future robotic processing controls |
| `wake` | enabled flag, phrases, engaged window, emergency bypass |
| `conversation` | retained turns, character budget, idle expiry, personality |
| `robot` | no-hardware or exclusive Sourccey client adapter and robot IP |
| `commands` | harmless transcription aliases for approved enum commands |

Unknown sections and keys fail configuration loading. The default authentication token is
intentionally invalid so a production service cannot accidentally start with a shared secret.

## Running the full system

1. Start the existing `sourccey-host` process on the robot if using the command adapter.
2. Set `robot.backend = "sourccey_client"` and `robot.remote_ip` in the desktop voice config.
3. Start `sourccey-voice ... host` on the desktop. This loads and keeps all models warm.
4. Start `sourccey-voice ... robot-audio` on Sourccey.
5. Confirm `[NETWORK] robot audio endpoint connected`, then speak near the microphone.

Use `robot.backend = "none"` for conversation-only hardware trials. Explicit commands will be
recognized but reported unavailable.

## Wake behavior

The included detector is a replaceable transcript-level wake detector. With wake mode enabled,
`Hey Sourccey, who are you?` opens an engaged window; later utterances within the configured
interval do not need the phrase. `STOP`, `FREEZE`, and `CANCEL` explicit commands may bypass wake
requirements. `FOLLOW_ME` does not.

This implementation still runs VAD/STT before detecting the wake phrase. A dedicated low-power
wake-word backend can implement the same boundary later without changing routing or conversation
code.

## Adding a command

1. Add an enum member in `CommandName`.
2. Add aliases under `[commands]`.
3. Add an explicit implementation to a `RobotCommandAdapter` and include it in
   `available_actions()` only when the capability is ready.
4. Add parser, unavailable-capability, handler-failure, and priority tests.

Never add a generic action dictionary, generated Python, shell command, or raw protobuf tool to
the LLM. The enum and adapter are the safety review points.

## Adding robot state

Add the real field to `_KNOWN_STATE_FIELDS` in `state.py`, map it from a real Sourccey
observation, and test serialization. State is refreshed per interaction and is not copied into
conversation history. Absent sensors remain absent; they are never filled with invented values.

## Replacing a backend

The interfaces in `types.py` define `SpeechRecognizer`, `ConversationEngine`, `TextToSpeech`,
`Speaker`, and `RobotCommandAdapter`. Add a lazy adapter and select it in `factory.py`. Keep
large imports out of the package import path so developer mode and command tests remain light.

## Diagnostics and latency

Logs record state transitions and text, never raw audio:

```text
[VAD] speech_started
[STT] "Command: stop"
[ROUTER] explicit command detected
[TTS] interrupted
[COMMAND] STOP accepted
[TTS] "Stopped."
```

`logging.debug_partials` enables partial transcripts. Latency logs currently measure speech end
to final transcript and response to TTS submission. A production UI can timestamp the existing
`voice.event.v1` messages to add transcript-to-first-token and end-to-first-speaker-sample metrics.

## Testing

```powershell
uv sync --extra dev
uv run pytest
uv run python -m build
```

The suite covers VAD transitions, command normalization and aliases, priority interruption,
structured action validation, unavailable and invalid actions, LLM/TTS/handler failure behavior,
history limits, state serialization, wire validation and stale frames, wake transitions, and a
mock audio-to-command integration path. No test can move physical hardware.

## Droid Voice Effects

`droid_voice.py` is a reusable local effects layer for turning normal TTS into a
friendly utility-droid character. It is deliberately pitch-conservative: the default
`cute_helper_droid` preset leaves fundamental pitch unchanged, so its charm comes from
machine coloration and timing rather than a childlike voice.

```powershell
python droid_voice.py input.wav output.wav
python droid_voice.py input.wav output.wav --preset cute_helper_droid
python droid_voice.py input.wav renders\robot.wav --ab
```

`--ab` writes four files next to the requested output: A `subtle_droid`, B
`cute_helper_droid`, C `more_synthetic`, and D `maximum_robot`.

For code integration:

```python
from sourccey_voice.droid_voice import DroidVoiceProcessor

processor = DroidVoiceProcessor(preset="cute_helper_droid")
processed_audio = processor.process(audio, sample_rate)

stream = processor.stream(sample_rate)
processed_chunk = stream.process_chunk(pcm_chunk)
processed_pcm16 = stream.process_pcm16_chunk(pcm16_chunk)
```

The presets live in `config/droid_voice.toml`; pass `--config` with a copied file to
tune without changing code. The most useful character controls are:

- `ring_mod_mix` and `ring_mod_frequency_hz`: restrained biased AM adds machine motion
  without erasing consonants.
- `spectral_mix` and `formant_shift`: create compact synthetic formants; keep formant
  shift below about `0.4` semitones to avoid youthfulness.
- `speaker_low_hz`, `speaker_high_hz`, and `micro_delay_*`: make the result feel like a
  small physical speaker rather than a telephone or a distant echo.
- `saturation`, `bit_depth`, and compression: add controlled harmonic density and keep
  quiet consonants intelligible.
- `pitch_semitones` and `pitch_quantization_strength`: use sparingly. Raising these is
  the main route toward an unwanted childlike sound, so every supplied preset keeps them
  at zero.

The offline path includes spectral/formant stages. `DroidVoiceStream` keeps the
low-latency time-domain effects stateful across chunks and skips look-ahead transforms;
its only intentional latency is the configured micro-delay.

## Troubleshooting

- `network.auth_token must be changed`: edit both configs and give them the same non-default value.
- `model_path is empty`: run the matching `models download` command and paste the printed path.
- AEC unavailable: install `[audio]`. With `aec_required = true`, startup fails rather than risking
  a self-conversation loop. With it false, robot capture is gated during playback and barge-in is
  unavailable.
- Robot cannot connect: bind the host to a reachable interface, allow the configured TCP port,
  and use the desktop LAN IP rather than `127.0.0.1` in the robot config.
- Speech clips at the beginning: increase `vad.pre_roll_ms`. Long response delay often means
  `vad.silence_timeout_ms` is too high.
- Echo remains: verify the speaker reference uses the same 16 kHz device stream and tune
  `aec_stream_delay_ms` in 10 ms increments on the physical robot.
- Commands unavailable: use developer mode to verify parsing, then check `robot.backend` and the
  adapter's `available_actions()` result.

## Uninstall

```powershell
.\scripts\uninstall.ps1 `
  -DesktopPath C:\Users\Theor\Documents\WebsiteCode\VulcanDesktop\sourccey-desktop
```

This preserves configuration and model caches. Add `-RemoveConfig` to delete the Desktop module
configuration. Model caches are deliberately never removed automatically because they may be
large, shared, and expensive to download again.

On Linux, pass the environment's Python executable. Configuration removal requires an explicit
path:

```bash
./scripts/uninstall.sh .venv/bin/python
./scripts/uninstall.sh .venv/bin/python --remove-config /opt/sourccey-voice/config
```

## Known limitations

- Physical microphone selection, speaker gain, AEC delay, noise rejection, and barge-in require
  testing on Sourccey's actual audio hardware.
- The current robotics protocol has no semantic command channel or follow-me capability.
- The current Desktop app does not yet auto-discover or supervise `sourccey-module.toml`.
- Wake detection is transcript-level, not a dedicated low-power wake-word engine.
- The built-in Kokoro `robotic_processing` is intentionally subtle. Use `droid_voice.py`
  for a tunable character-processing chain and audition its presets on the robot speaker.
- Only one robot audio websocket is intended per host process.

These constraints are explicit so the module fails safely while the remaining robot and Desktop
integration points are developed.
