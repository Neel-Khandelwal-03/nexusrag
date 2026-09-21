from __future__ import annotations

import pytest

from nexusrag.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_matches_hand_computation() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    expected = {
        "a": 1 / 61,
        "b": 1 / 62 + 1 / 61,
        "c": 1 / 63 + 1 / 62,
        "d": 1 / 63,
    }
    assert [doc for doc, _ in fused] == ["b", "c", "a", "d"]
    for doc, score in fused:
        assert score == pytest.approx(expected[doc])


def test_documents_in_several_lists_beat_a_single_first_place() -> None:
    # "x" is first in one list only; "y" is second in three lists.
    fused = reciprocal_rank_fusion([["x", "y"], ["z", "y"], ["w", "y"]])
    assert fused[0][0] == "y"


def test_single_list_preserves_order() -> None:
    assert [d for d, _ in reciprocal_rank_fusion([["p", "q", "r"]])] == ["p", "q", "r"]


def test_ties_are_deterministic() -> None:
    fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [d for d, _ in fused] == ["x", "y"]  # same best rank -> first seen wins


def test_weights() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])
    assert [d for d, _ in fused] == ["b", "a"]
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_smaller_k_favours_top_ranks() -> None:
    lists = [["a", "b"], ["b", "c"], ["c", "a"]]
    assert reciprocal_rank_fusion(lists, k=1)[0][1] > reciprocal_rank_fusion(lists, k=60)[0][1]


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
