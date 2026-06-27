"""Guard against provider-adapter drift across the three tasks.

The adapters are (currently) copy-pasted per task, so a fix landing in one and not
the others silently breaks the others — that is exactly how the Anthropic
max_tokens-vs-thinking-budget and Gemini timeout fixes drifted. This test fails
the moment any adapter diverges between Vision2Code / CodeEdit / QA.
"""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TASKS = ["Vision2Code", "CodeEdit", "QA"]
_ADAPTERS = ["openai", "anthropic", "gemini", "openrouter"]


def test_model_adapters_identical_across_tasks():
    for adapter in _ADAPTERS:
        texts = {
            t: (_ROOT / t / "models" / f"{adapter}_adapter.py").read_text()
            for t in _TASKS
        }
        ref = texts["Vision2Code"]
        for t in _TASKS:
            assert texts[t] == ref, (
                f"{adapter}_adapter.py drifted in {t} vs Vision2Code — keep the "
                f"provider adapters in sync (a fix in one must land in all three)."
            )
