"""The chat settings panel (Chainlit input widgets)."""

from __future__ import annotations

from collections.abc import Sequence

from chainlit.input_widget import InputWidget, MultiSelect, Select, Slider, Switch, TextInput

from nexusrag.ui.state import (
    W_COLLECTION,
    W_DOCUMENTS,
    W_HYBRID,
    W_HYDE,
    W_MULTI_QUERY,
    W_NEW_COLLECTION,
    W_RERANK,
    W_SELF_CORRECT,
    W_STYLE,
    W_TOP_K,
    UISettings,
)


def settings_widgets(
    ui: UISettings, collections: Sequence[str], filenames: Sequence[str]
) -> list[InputWidget]:
    """Widgets for the panel, showing ``ui``'s values.

    ``collections`` are the existing knowledge bases and ``filenames`` the documents in
    the current one. The document filter is left out while the knowledge base is empty:
    Chainlit rejects a multiselect with no options.
    """
    widgets: list[InputWidget] = [
        Select(
            id=W_COLLECTION,
            label="Knowledge base",
            values=sorted({*collections, ui.collection}),
            initial_value=ui.collection,
        ),
        TextInput(
            id=W_NEW_COLLECTION,
            label="New knowledge base",
            placeholder="e.g. contracts-2026",
            description="Type a name to create a knowledge base and switch to it.",
        ),
    ]
    if filenames:
        widgets.append(
            MultiSelect(
                id=W_DOCUMENTS,
                label="Only search these documents",
                values=list(filenames),
                initial=[f for f in ui.documents if f in filenames],
                description="Leave empty to search the whole knowledge base.",
            )
        )
    widgets += [
        Slider(
            id=W_TOP_K,
            label="Passages per answer (top-k)",
            initial=ui.top_k,
            min=1,
            max=20,
            step=1,
        ),
        Switch(id=W_HYBRID, label="Hybrid search (dense + BM25)", initial=ui.hybrid),
        Switch(
            id=W_MULTI_QUERY,
            label="Multi-query (search with rephrasings)",
            initial=ui.multi_query,
        ),
        Switch(id=W_HYDE, label="HyDE (search with a hypothetical answer)", initial=ui.hyde),
        Switch(id=W_RERANK, label="Rerank with a cross-encoder", initial=ui.rerank),
        Switch(
            id=W_SELF_CORRECT,
            label="Self-correction (grade, retry, check groundedness)",
            initial=ui.self_correct,
        ),
        Select(
            id=W_STYLE,
            label="Answer style",
            values=["concise", "detailed"],
            initial_value=ui.style,
        ),
    ]
    return widgets
