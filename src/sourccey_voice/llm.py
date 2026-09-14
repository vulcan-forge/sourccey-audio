from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from .types import ConversationReply

_THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)


def _strip_thinking(content: str) -> str:
    return _THINKING_BLOCK.sub("", content).strip()


class LlamaCppConversationEngine:
    def __init__(
        self,
        model_path: str,
        personality: str,
        *,
        context_tokens: int = 4096,
        max_response_tokens: int = 128,
        temperature: float = 0.35,
        device: str = "auto",
        structured_response: bool = True,
    ) -> None:
        if not model_path or not Path(model_path).is_file():
            raise RuntimeError(
                "llm.model_path must point to a downloaded GGUF file; "
                "run `sourccey-voice models download llm`"
            )
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except ImportError as exc:
            raise RuntimeError("llama.cpp is unavailable; install sourccey-voice[llm]") from exc
        if device in {"cuda", "gpu"} and not llama_supports_gpu_offload():
            raise RuntimeError(
                "llama.cpp was configured for CUDA, but this llama-cpp-python build has no GPU support"
            )
        gpu_layers = -1 if device in {"auto", "cuda", "gpu"} else 0
        self._model = Llama(
            model_path=model_path,
            n_ctx=context_tokens,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        self._personality = personality.strip()
        self._max_tokens = max_response_tokens
        self._temperature = temperature
        self._structured_response = structured_response

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        robot_state: dict[str, object],
        available_actions: Sequence[str],
    ) -> ConversationReply:
        if self._structured_response:
            actions = list(available_actions)
            instruction = (
                f"{self._personality}\n\n"
                "Reply as one JSON object with exactly these fields: "
                '{"text":"short spoken reply","requested_action":null}. '
                "requested_action may be null or one exact item from AVAILABLE_ACTIONS. "
                "Never create another action name, never say an action succeeded, and do not expose reasoning. "
                "/no_think\n"
                f"AVAILABLE_ACTIONS={json.dumps(actions)}\n"
                f"CURRENT_ROBOT_STATE={json.dumps(robot_state, separators=(',', ':'))}"
            )
        else:
            instruction = (
                f"{self._personality}\n\n"
                "Reply in plain spoken text only. Do not use JSON, markup, or reasoning tags. /no_think"
            )
        chat = [{"role": "system", "content": instruction}, *messages]
        result = self._model.create_chat_completion(
            messages=chat,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        content = _strip_thinking(str(result["choices"][0]["message"]["content"]))
        if not self._structured_response:
            return ConversationReply(text=content, requested_action=None)
        parsed = _parse_reply(content)
        if parsed is None:
            return ConversationReply(text=content, requested_action=None)
        return parsed


def _parse_reply(content: str) -> ConversationReply | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        return None
    action = value.get("requested_action")
    if action is not None and not isinstance(action, str):
        action = None
    return ConversationReply(text=value["text"].strip(), requested_action=action)


class DeveloperConversationEngine:
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        robot_state: dict[str, object],
        available_actions: Sequence[str],
    ) -> ConversationReply:
        del robot_state, available_actions
        text = messages[-1]["content"] if messages else ""
        lowered = text.casefold()
        if "who are you" in lowered:
            return ConversationReply("I'm Sourccey, a robot built by Vulcan Robotics.")
        return ConversationReply("Hi! What are we building today?")
