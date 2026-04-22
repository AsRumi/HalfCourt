"""
commentary_engine.py

Call the LLM and validate that the response contains the four required sections.
"""

import re

from halfcourt.llm_client import complete
from halfcourt.exceptions import LLMError

REQUIRED_SECTIONS = [
    "SCORE & MOMENTUM",
    "KEY PLAYER ANALYSIS",
    "X-FACTORS / FOUL TROUBLE",
    "SECOND-HALF OUTLOOK",
]


def _validate_sections(text: str) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    if missing:
        raise LLMError(
            f"LLM response is missing required sections: {missing}"
        )


def generate_commentary(prompt: dict) -> str:
    """
    Send the prompt to the LLM and return validated commentary text.

    Args:
        prompt: {"system": str, "user": str} from prompt_builder.build_prompt()

    Returns:
        Commentary string containing all four required sections.

    Raises:
        LLMError if the API call fails or required sections are missing.
    """
    raw = complete(prompt)
    _validate_sections(raw)
    return raw.strip()
