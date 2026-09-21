"""All prompt templates live in this module.

Keeping them together makes prompts reviewable, diffable and testable in one place.
Templates use ``str.format`` placeholders; :func:`render` fails loudly on missing or
unexpected variables instead of silently sending a half-filled prompt to the model.
Stage-specific templates (router, grader, answer, ...) are added as those stages land.
"""

from __future__ import annotations

from string import Formatter


def template_fields(template: str) -> set[str]:
    """Names of the ``{placeholders}`` used in ``template``."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def render(template: str, **values: object) -> str:
    """Fill ``template`` with ``values``; raise if any placeholder is missing or extra."""
    expected = template_fields(template)
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise KeyError(
            f"Prompt variables mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return template.format(**values)


#: Shared identity used as the system instruction for user-facing generations.
SYSTEM_PERSONA = """\
You are NexusRAG, an assistant that answers questions about the user's own documents.
You are precise, cite your sources, and say plainly when the documents do not contain
the answer instead of guessing or relying on general knowledge."""
