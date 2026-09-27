"""Извикване на Claude (Anthropic API) - цял отговор или на части (стрийминг)."""

from anthropic import Anthropic

CLAUDE_MODEL = "claude-sonnet-5"


def _claude_error(stop_reason) -> ValueError:
    return ValueError(f"Claude върна празен отговор (stop_reason: {stop_reason}). Опитай пак.")


def call_claude(prompt: str, api_key: str, max_tokens: int = 4096) -> str:
    """Една заявка към Claude, връща целия текст."""
    response = Anthropic(api_key=api_key).messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise _claude_error(response.stop_reason)
    return text


def stream_claude(prompt: str, api_key: str, max_tokens: int = 4096):
    """Същото, но текстът идва на части (за st.write_stream) - отговорът се
    вижда веднага, вместо след цялото генериране."""
    got_text = False
    with Anthropic(api_key=api_key).messages.stream(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            got_text = got_text or bool(text.strip())
            yield text
        final = stream.get_final_message()
    if not got_text:
        raise _claude_error(final.stop_reason)
