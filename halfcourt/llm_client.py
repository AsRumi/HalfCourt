"""
llm_client.py

Thin LLM abstraction supporting OpenAI, Anthropic, and Gemini.
Provider is read from config.yaml - switching providers requires no code change.
"""

import os
import yaml

from halfcourt.exceptions import LLMError


def _load_config() -> dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def _complete_openai(prompt: dict, cfg: dict) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMError("openai package is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=cfg["llm"]["model"],
            max_tokens=cfg["llm"]["max_tokens"],
            temperature=cfg["llm"]["temperature"],
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user",   "content": prompt["user"]},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise LLMError(f"OpenAI API call failed: {e}") from e


def _complete_anthropic(prompt: dict, cfg: dict) -> str:
    try:
        import anthropic
    except ImportError:
        raise LLMError("anthropic package is not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY environment variable is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=cfg["llm"]["model"],
            max_tokens=cfg["llm"]["max_tokens"],
            temperature=cfg["llm"]["temperature"],
            system=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
        return response.content[0].text
    except Exception as e:
        raise LLMError(f"Anthropic API call failed: {e}") from e


def _complete_gemini(prompt: dict, cfg: dict) -> str:
    try:
        import google.generativeai as genai
    except ImportError:
        raise LLMError("google-generativeai package is not installed. Run: pip install google-generativeai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(
            model_name=cfg["llm"]["model"],
            system_instruction=prompt["system"],
        )
        response = model.generate_content(
            prompt["user"],
            generation_config=genai.GenerationConfig(
                max_output_tokens=cfg["llm"]["max_tokens"],
                temperature=cfg["llm"]["temperature"],
            ),
        )
        return response.text
    except Exception as e:
        raise LLMError(f"Gemini API call failed: {e}") from e


_PROVIDERS = {
    "openai":    _complete_openai,
    "anthropic": _complete_anthropic,
    "gemini":    _complete_gemini,
}


def complete(prompt: dict) -> str:
    """
    Send a prompt to the configured LLM provider and return the response text.

    Args:
        prompt: {"system": str, "user": str} as produced by prompt_builder.build_prompt()

    Returns:
        Raw LLM response string.

    Raises:
        LLMError on API failure or misconfiguration.
    """
    cfg = _load_config()
    provider = cfg["llm"]["provider"]

    if provider not in _PROVIDERS:
        raise LLMError(
            f"Unknown LLM provider '{provider}'. Choose from: {list(_PROVIDERS.keys())}"
        )

    return _PROVIDERS[provider](prompt, cfg)
