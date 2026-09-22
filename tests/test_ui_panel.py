"""The settings panel widgets, including an empty knowledge base."""

from __future__ import annotations

from chainlit.input_widget import MultiSelect, Select, Slider

from nexusrag.config import Settings
from nexusrag.ui.panel import settings_widgets
from nexusrag.ui.state import W_CACHE, W_COLLECTION, W_DOCUMENTS, W_STYLE, W_TOP_K, UISettings


def by_id(widgets: list) -> dict:  # type: ignore[type-arg]
    return {w.id: w for w in widgets}


def test_panel_reflects_current_settings(settings: Settings) -> None:
    ui = UISettings.defaults(settings).updated(
        {W_TOP_K: 8, W_STYLE: "concise", W_DOCUMENTS: ["a.pdf", "gone.md"]}
    )
    widgets = by_id(settings_widgets(ui, ["default", "contracts", "default"], ["a.pdf", "b.md"]))
    kb = widgets[W_COLLECTION]
    assert isinstance(kb, Select)
    assert kb.values == ["contracts", "default"]
    assert kb.initial_value == "default"
    docs = widgets[W_DOCUMENTS]
    assert isinstance(docs, MultiSelect)
    assert docs.values == ["a.pdf", "b.md"]
    assert docs.initial == ["a.pdf"]  # a filter on a deleted document is dropped
    top_k = widgets[W_TOP_K]
    assert isinstance(top_k, Slider)
    assert top_k.initial == 8
    assert widgets[W_STYLE].initial_value == "concise"
    assert widgets[W_CACHE].initial is ui.cache


def test_empty_knowledge_base_has_no_document_filter(settings: Settings) -> None:
    ui = UISettings.defaults(settings).updated({W_COLLECTION: "new-kb"})
    widgets = by_id(settings_widgets(ui, ["default"], []))
    assert W_DOCUMENTS not in widgets  # Chainlit rejects a multiselect without options
    assert widgets[W_COLLECTION].values == ["default", "new-kb"]
