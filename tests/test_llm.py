from sourccey_voice.llm import _parse_reply, _strip_thinking


def test_thinking_wrapper_is_not_part_of_json_reply():
    content = '<think>internal reasoning</think>\n{"text":"Hello!","requested_action":null}'
    reply = _parse_reply(_strip_thinking(content))
    assert reply is not None
    assert reply.text == "Hello!"
