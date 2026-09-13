You are working inside the Sourccey robotics codebase.

I want you to DESIGN, IMPLEMENT, TEST, and DOCUMENT a complete standalone voice interaction module for Sourccey.

Do not merely give me instructions or example code. Inspect the existing repository, understand the architecture, determine the correct integration points, and implement the system.

# PRIMARY GOAL

Sourccey should be able to:

1. Hear a person speaking through the robot's microphone.
2. Stream that audio to the client/host computer.
3. Convert speech to text locally on the client.
4. Distinguish explicit robot commands from ordinary conversation.
5. Execute approved robot commands through a strict command interface.
6. Send ordinary conversation to a small local LLM.
7. Allow the LLM to know relevant real robot state.
8. Generate a spoken response using local TTS.
9. Stream/play that response through Sourccey's speaker.
10. Prevent Sourccey's own speaker output from being interpreted as human speech.
11. Allow a human to interrupt Sourccey while it is talking.
12. Eventually support a wake word such as "Sourccey" or "Hey Sourccey."

Everything should run locally. Do not introduce required cloud APIs.

# IMPORTANT ARCHITECTURAL REQUIREMENT

This MUST be implemented as a STANDALONE MODULE/PACKAGE.

I want to be able to install/download this module into Sourccey rather than permanently baking its internals throughout the rest of the Sourccey codebase.

Treat it as something conceptually like:

sourccey_voice/

with its own:

- source code
- configuration
- dependencies
- models/model management
- tests
- logging
- documentation
- install/uninstall instructions
- public interface/API

The rest of Sourccey should interact with it through a small, clean interface.

Avoid modifying unrelated Sourccey systems.

If modifications to the existing codebase are necessary, keep them extremely small and use adapters/interfaces where possible.

The module should be portable enough that it could theoretically be installed on another Sourccey robot without manually modifying a large number of files.

# FIRST: INSPECT THE REPOSITORY

Before implementing anything:

1. Explore the entire relevant Sourccey repository.
2. Determine:
   - programming languages
   - client/host architecture
   - robot-side architecture
   - networking/protocol currently used
   - existing audio support, if any
   - command/control interfaces
   - robot mode/state system
   - configuration system
   - logging system
   - startup/shutdown lifecycle
   - packaging/deployment system
3. Reuse existing infrastructure where sensible.
4. Do NOT replace working Sourccey infrastructure simply because another library would be easier.

Then create an implementation plan based on the actual repository and execute it.

Do not stop after presenting the plan unless there is a genuinely blocking ambiguity that cannot be safely resolved from the repository.

# TARGET PIPELINE

The conceptual architecture should be:

ROBOT MICROPHONE
    ↓
audio capture
    ↓
noise suppression / acoustic echo cancellation
    ↓
audio stream
    ↓
CLIENT/HOST
    ↓
voice activity detection
    ↓
streaming speech-to-text
    ↓
transcript
    ↓
intent / command router
    ↓
    ├── explicit command → command registry → Sourccey API
    │
    └── ordinary conversation → local LLM
                                 ↓
                              response
                                 ↓
                             local TTS
                                 ↓
                              audio
                                 ↓
ROBOT SPEAKER

Implement this with clean boundaries so individual components can later be replaced.

# 1. AUDIO

Implement robot-side microphone capture and client-side receipt using the networking architecture Sourccey already uses whenever practical.

Prioritize speech intelligibility over studio-quality audio.

Use an appropriate speech sample rate and mono audio unless the existing hardware/architecture gives a good reason otherwise.

Make buffering resilient to normal network jitter.

Do not create enormous latency-producing buffers.

The audio subsystem should expose clean interfaces so microphone hardware can later be changed without rewriting STT.

# 2. VOICE ACTIVITY DETECTION

Use Silero VAD unless repository/platform constraints reveal a clearly superior choice.

The system should determine:

- speech started
- speech continuing
- speech ended

Do not continuously submit arbitrary environmental noise as conversations.

Expose configurable values for:

- speech threshold
- minimum speech duration
- silence timeout
- pre-roll
- post-roll

Use sensible defaults.

# 3. SPEECH TO TEXT

Use a LOCAL streaming speech recognition implementation.

Preferred initial candidate:

Moonshine streaming STT.

Prefer the Medium streaming model initially if hardware performance is acceptable.

However, inspect compatibility before committing.

If Moonshine is impractical on the existing platform, implement a clean STT abstraction and use the best suitable local alternative, such as faster-whisper.

The architecture MUST make the STT engine replaceable.

Example conceptual interface:

SpeechRecognizer
    start()
    push_audio(...)
    partial_transcript()
    finalize()
    stop()

Support partial transcripts if the selected backend supports them.

# 4. COMMAND SYSTEM

THIS IS SAFETY CRITICAL.

THE LLM MUST NEVER RECEIVE DIRECT ARBITRARY CONTROL OF ROBOT MOTORS OR ACTUATORS.

Create a strict command registry.

Explicit commands use syntax such as:

"Command: follow me"

Normalize capitalization, punctuation, and harmless transcription differences.

Initial commands should include infrastructure for at least:

FOLLOW_ME
STOP
CANCEL
FREEZE

If the existing Sourccey code already contains corresponding modes/actions, connect to them.

If FOLLOW_ME does not exist yet, create the voice-side command definition and an adapter/stub/interface for the future follow-me system rather than inventing an entire follow-me navigation system unless the existing repository already contains most of it.

Commands should map to explicit functions/events such as:

enter_follow_me_mode()
stop_motion()
cancel_current_action()
freeze()

NOT arbitrary generated code.

Create a registry capable of aliases, for example:

"follow me"
"start following me"
"follow"
"come with me"

→ FOLLOW_ME

However, preserve an especially reliable deterministic path for explicit:

"Command: <command>"

STOP / FREEZE / CANCEL should receive priority handling.

Where practical, allow these high-priority commands to interrupt speech and current actions.

# 5. NATURAL LANGUAGE TOOL REQUESTS

Separately from explicit "Command:" syntax, design an OPTIONAL system allowing conversational requests such as:

"Hey Sourccey, can you follow me?"

The LLM may interpret this as a request for an approved capability.

It must return/request a structured tool/action such as:

{
  "requested_action": "FOLLOW_ME"
}

The command layer then validates it against the approved registry.

The LLM itself never controls motors.

If the capability is unavailable, the robot should say so rather than pretending it executed it.

# 6. CONVERSATIONAL MODEL

Implement a replaceable local LLM backend.

Start by evaluating an appropriately sized Qwen model around the 4B class, with Qwen3-4B as the initial candidate if compatible with the existing machine.

Do not hardwire the architecture to Qwen.

Create an abstraction such as:

ConversationEngine
    generate(messages, robot_state, available_actions)

Optimize for LOW conversational latency.

Sourccey's answers should normally be short.

Default personality:

You are Sourccey, a mobile manipulation robot built by Vulcan Robotics.

Personality:
- curious
- cheerful
- slightly mischievous
- concise
- cute without sounding childish
- aware that you are a robot

Behavior:
- usually answer in one or two sentences
- never claim to have performed an action unless robot state confirms it
- never invent sensor observations
- never claim to see/hear/feel something unless supplied through robot state
- acknowledge successful commands briefly
- clearly state when a capability is unavailable

Put this prompt/personality in configuration rather than deeply hardcoding it.

# 7. ROBOT STATE

Create a RobotState abstraction/adaptor.

The conversational system should be capable of receiving information such as:

battery percentage
current operating mode
whether a person is currently detected
navigation state
left arm state
right arm state
current action
errors/faults

ONLY expose state that actually exists in the current Sourccey repository.

For fields that do not exist yet, make the interface extensible rather than fabricating values.

The LLM should receive a concise structured representation of current state with each relevant interaction.

Example:

{
  "battery": 73,
  "mode": "follow_me",
  "person_visible": true,
  "left_arm": "idle",
  "right_arm": "idle"
}

This should allow Sourccey to conversationally describe what it is actually doing.

# 8. TEXT TO SPEECH

Implement LOCAL TTS.

Start with Kokoro if compatible.

Create a replaceable TTS abstraction.

The desired Sourccey voice is:

- cute
- friendly
- clear
- somewhat robotic/synthetic
- NOT a cartoon baby voice
- NOT an uncanny perfectly-human assistant voice
- easy to understand in a moderately noisy room

If useful, implement OPTIONAL lightweight post-processing such as:

- slight pitch adjustment
- subtle formant/vocoder treatment
- EQ appropriate for the robot speaker
- short startup/listening chirps

Do not sacrifice speech intelligibility for robotic effects.

Make voice and processing parameters configurable.

# 9. ACOUSTIC ECHO CANCELLATION

This is important.

Sourccey's microphone and speaker are physically on the same robot.

The system must avoid:

Sourccey speaks
→ microphone hears Sourccey
→ STT transcribes Sourccey
→ LLM responds to itself
→ infinite conversation

Implement acoustic echo cancellation if reasonably supported by the platform.

Because the system generates the outgoing speaker waveform, use the outgoing audio as the reference signal for echo cancellation.

Design:

microphone signal ──→ AEC ──→ STT
                       ↑
speaker reference ─────┘

Investigate an appropriate local AEC implementation compatible with the repository/platform, such as WebRTC Audio Processing or an equivalent.

Do not simply mute the microphone whenever Sourccey speaks unless necessary as an initial fallback, because I want barge-in.

# 10. BARGE-IN

A human should be able to interrupt Sourccey while Sourccey is speaking.

Example:

Sourccey:
"According to my current sens—"

Human:
"Command: stop."

Sourccey should stop TTS/playback and process the command immediately.

Design the audio system so VAD/STT can continue operating while TTS is playing, with AEC preventing Sourccey's own voice from triggering it.

If robust full-duplex barge-in cannot initially be achieved because of platform constraints, implement the architecture for it and clearly document the temporary limitation.

# 11. WAKE WORD

Design a WakeWordDetector interface.

Target phrases:

"Sourccey"
"Hey Sourccey"

Do not make wake-word support tightly coupled to the rest of the pipeline.

If a suitable lightweight local wake-word engine can be integrated cleanly, implement it.

Otherwise provide the interface/configuration and initially allow wake-word requirement to be disabled.

Support an engaged-conversation window:

"Hey Sourccey"
→ wake

User speaks
Robot responds

For approximately N configurable seconds after an interaction, additional speech can continue the conversation without repeating the wake word.

After inactivity, return to passive listening.

Explicit emergency commands may optionally bypass wake-word requirements.

# 12. CONVERSATION STATE

Maintain a small bounded conversation history.

Do not allow conversation history to grow forever.

Configuration should control:

- number of retained turns
- token/context budget
- inactivity timeout

Robot state should be refreshed rather than permanently copied into conversation history.

# 13. NETWORK PROTOCOL

Reuse Sourccey's existing communication layer where practical.

Conceptually the module needs messages/events equivalent to:

ROBOT → CLIENT

audio_stream
robot_state
voice_events

CLIENT → ROBOT

audio_playback
command_request
mode_change

Use typed/structured messages.

Do not transmit arbitrary executable code.

Handle:

- disconnects
- reconnects
- malformed messages
- stale messages
- shutdown
- interrupted audio

# 14. CONFIGURATION

Provide one obvious configuration location.

I want to be able to change things such as:

STT model
LLM model
TTS model/voice
wake word enabled
wake word phrase
conversation timeout
VAD sensitivity
microphone device
speaker device
audio volume
robotic voice processing
command aliases
LLM personality
model device (CPU/GPU)
logging level

without modifying source code.

Use the configuration conventions already present in Sourccey if appropriate.

# 15. MODEL MANAGEMENT

Do NOT casually dump multi-gigabyte model files into Git.

Create a sensible model download/cache strategy.

Document:

- which models are required
- approximate storage requirements
- how models are downloaded
- where they are cached
- how to change models
- how to operate offline after models have been downloaded

If licensing restrictions affect redistribution of any selected model, document them.

# 16. INSTALLATION

The finished result should behave like a standalone Sourccey module.

Ideally installation should be approximately:

install package/dependencies
download models
configure audio devices
enable module
run Sourccey

Provide an installation script or appropriate package mechanism for the actual repository/platform.

Also provide:

- clean uninstall instructions
- dependency isolation where practical
- environment/config example
- startup integration

Do not require manually copying random files into many Sourccey directories.

# 17. TESTING

Write meaningful automated tests.

At minimum test:

VAD state transitions
command parsing
command aliases
priority STOP/FREEZE/CANCEL behavior
LLM action validation
invalid/unapproved actions
conversation history limits
robot-state serialization
network message validation
TTS interruption
wake/conversation state transitions where applicable

Create mocks/fakes for:

microphone
speaker
STT
LLM
TTS
robot command interface
robot state

The command system should be testable without moving the physical robot.

Add an integration test that simulates:

human audio/text
→ STT result
→ command/conversation routing
→ response/action
→ TTS output

# 18. DIAGNOSTICS

This system will be developed on a real robot, so debugging visibility matters.

Provide useful logs for events such as:

[VAD] speech_started
[STT] "Command: follow me"
[ROUTER] explicit command detected
[COMMAND] FOLLOW_ME accepted
[ROBOT] mode transition requested
[TTS] "Okay, following you."
[VAD] speech_started during playback
[TTS] interrupted
[COMMAND] STOP accepted

Do not spam logs with raw audio data.

Provide an optional debug mode showing partial STT transcripts and timing/latency measurements.

Measure useful latency stages:

speech end → final transcript
transcript → LLM first token
LLM completion → TTS start
speech end → first robot audio

# 19. FAILURE BEHAVIOR

Fail safely.

Examples:

STT unavailable:
→ do not execute guessed commands.

LLM unavailable:
→ explicit commands should STILL WORK if STT works.

TTS unavailable:
→ command execution should still work and log/display acknowledgement.

Network disconnect:
→ do not queue stale movement commands for execution after reconnect.

Malformed LLM action:
→ reject it.

Unknown command:
→ do not approximate dangerous behavior.

Command handler exception:
→ report failure rather than telling the user it succeeded.

# 20. PERFORMANCE

Prioritize perceived conversational responsiveness.

Do not wait unnecessarily for every pipeline stage when work can safely overlap.

Use streaming where appropriate.

Avoid loading/unloading models for every utterance.

Keep STT/LLM/TTS models warm while the module is active.

Where appropriate, preload them during module startup.

Use GPU acceleration if available and compatible, but maintain configurable CPU fallback where practical.

# 21. DOCUMENTATION

Create a README specifically for the Sourccey Voice module.

Include:

- architecture diagram
- installation
- configuration
- model setup
- microphone/speaker setup
- running the module
- adding a new command
- adding robot state
- changing STT backend
- changing LLM backend
- changing TTS backend
- wake-word configuration
- troubleshooting
- testing
- latency/debugging
- uninstalling

Also explain the safety boundary:

AI interprets humans.
The command registry validates requests.
Sourccey's existing robotics software controls the robot.

# 22. DEVELOPER TOOL / TEST MODE

Create a simple developer mode that lets me test the entire intelligence stack WITHOUT talking to the physical robot.

For example, provide a CLI or small existing-framework-compatible interface where I can type:

> hello

and receive:

Sourccey: Hi! What are we building today?

And:

> Command: follow me

produces something like:

COMMAND REQUEST: FOLLOW_ME

without actually moving hardware.

If practical, also provide microphone and TTS test commands.

This will make development dramatically easier.

# 23. IMPLEMENTATION PHILOSOPHY

Do not overengineer this into dozens of microservices.

This is one robot with a robot side and a client/host side.

Prefer:

- clean interfaces
- replaceable backends
- typed structures
- asynchronous streaming where appropriate
- simple state machines
- minimal dependencies
- testability
- readable code

over enterprise architecture.

Do not duplicate infrastructure Sourccey already has.

Do not silently rewrite unrelated systems.

# 24. FINAL ACCEPTANCE TEST

When finished, I want to be able to perform approximately this sequence:

Human:
"Hey Sourccey."

Robot:
listens / indicates attention.

Human:
"Who are you?"

Robot:
"I'm Sourccey, a robot built by Vulcan Robotics."

Human:
"Command: follow me."

System:
recognizes deterministic command
validates FOLLOW_ME
requests Sourccey's follow-me mode
briefly acknowledges command

Robot:
"Okay!"

Human, while robot is speaking:
"Command: stop."

System:
detects human speech despite robot speaker output
interrupts TTS
recognizes STOP
executes the approved stop handler immediately

And ordinary conversation should never be capable of bypassing the command registry and directly controlling motors.

# EXECUTION INSTRUCTIONS

You have autonomy to inspect the repository, choose exact libraries compatible with the existing system, install development dependencies, write code, run tests, debug failures, and revise the implementation.

Do not blindly follow my suggested libraries if repository/platform inspection demonstrates that one is incompatible or substantially inferior. Preserve the architecture and explain substitutions.

When something can reasonably be inferred from the existing codebase, infer it rather than stopping to ask me.

When functionality does not yet exist in Sourccey, create a clean adapter/interface/stub rather than inventing unrelated robotics functionality.

Do not consider the task complete merely because the code compiles.

Before finishing:

1. Run the automated tests.
2. Run the developer/mock mode.
3. Verify package installation.
4. Verify configuration loading.
5. Verify explicit commands function without the LLM.
6. Verify arbitrary LLM output cannot bypass the command registry.
7. Verify graceful behavior when individual AI components fail.
8. Review changes for unnecessary modifications outside the module.
9. Document anything requiring physical hardware testing.
10. Give me a concise final report containing:
   - architecture implemented
   - files created/changed
   - models/libraries selected and why
   - how to install it
   - how to run it
   - how to test it without the robot
   - how to test it on Sourccey
   - known limitations
   - next recommended steps

The objective is not a demo script.

Build this as a real reusable Sourccey Voice module that we can keep developing as part of the robot.