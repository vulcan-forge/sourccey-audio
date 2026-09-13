import pytest

from sourccey_voice.commands import CommandName, CommandRegistry, normalize_command


@pytest.fixture
def registry() -> CommandRegistry:
    return CommandRegistry()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Command: follow me", CommandName.FOLLOW_ME),
        ("command, STOP!", CommandName.STOP),
        (" COMMAND - hold position. ", CommandName.FREEZE),
        ("Command: never mind", CommandName.CANCEL),
    ],
)
def test_explicit_command_aliases(registry, text, expected):
    assert registry.parse_explicit(text).name == expected


def test_ordinary_alias_is_not_explicit(registry):
    assert registry.parse_explicit("follow me") is None
    assert registry.match_alias("follow me").name == CommandName.FOLLOW_ME


def test_unknown_explicit_command_is_not_approximated(registry):
    assert registry.uses_explicit_syntax("Command: launch")
    assert registry.parse_explicit("Command: launch") is None


def test_unapproved_structured_action_rejected(registry):
    with pytest.raises(ValueError, match="unapproved"):
        registry.validate_action_name("DRIVE_FORWARD")


def test_normalization_handles_punctuation_and_case():
    assert normalize_command("  STOP, Moving!!! ") == "stop moving"


def test_unavailable_action_never_calls_handler(registry):
    called = False

    def handler(action):
        nonlocal called
        called = True
        return True, action

    result = registry.execute(CommandName.FOLLOW_ME, {"STOP"}, handler)
    assert not result.accepted
    assert not called


def test_handler_exception_reports_failure(registry):
    def handler(action):
        raise RuntimeError("motor fault")

    result = registry.execute(CommandName.STOP, {"STOP"}, handler)
    assert not result.accepted
    assert "motor fault" in result.message

