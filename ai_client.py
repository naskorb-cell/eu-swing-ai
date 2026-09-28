"""Извикване на AI модел - Claude (Anthropic API) или Gemini (Google) - цял
отговор или на части (стрийминг). Кой модел прави анализите, се избира в
приложението (AI_PROVIDERS; първият е по подразбиране)."""

from anthropic import Anthropic

CLAUDE_MODEL = "claude-sonnet-5"
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"  # сменя се без код със secret GEMINI_MODEL

AI_PROVIDERS = ["Gemini", "Claude"]
AI_KEY_SECRETS = {"Gemini": "GEMINI_API_KEY", "Claude": "ANTHROPIC_API_KEY"}


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


def gemini_thinking_config(model: str):
    """Ниско ниво на вътрешно „мислене“ на Gemini - то се таксува като изходен текст,
    а за кратките ни отговори (JSON с новини, план) не е нужно дълго разсъждение.
    Gemini 3.x приема thinking_level; 2.x - бюджет в токени."""
    from google.genai import types
    if model.startswith("gemini-2"):
        return types.ThinkingConfig(thinking_budget=512)
    return types.ThinkingConfig(thinking_level="LOW")


def stream_gemini(prompt: str, api_key: str, model: str = GEMINI_DEFAULT_MODEL):
    """Gemini на части (за st.write_stream). Импортът е тук - без инсталиран
    google-genai приложението работи, пада само тази функция."""
    from google import genai
    from google.genai import types
    # клиентът трябва да е в променлива - временен обект се затваря преди заявката
    client = genai.Client(api_key=api_key)
    got_text = False
    config = types.GenerateContentConfig(thinking_config=gemini_thinking_config(model))
    for chunk in client.models.generate_content_stream(model=model, contents=prompt, config=config):
        text = chunk.text or ""
        got_text = got_text or bool(text.strip())
        yield text
    if not got_text:
        raise ValueError("Gemini върна празен отговор. Опитай пак.")


def stream_ai(prompt: str, provider: str, api_key: str, max_tokens: int = 4096, gemini_model: str = GEMINI_DEFAULT_MODEL):
    """Стрийминг от избрания доставчик ("Gemini" или "Claude")."""
    if provider == "Gemini":
        yield from stream_gemini(prompt, api_key, gemini_model)
    else:
        yield from stream_claude(prompt, api_key, max_tokens=max_tokens)
