"""Evaluation suite: a reviewed Q/A dataset, hand-written metrics and a config comparison.

* ``python -m eval.generate_dataset``: draft ``eval/dataset.jsonl`` from the indexed
  documents (then review it by hand).
* ``python -m eval.run_eval``: run several pipeline configurations over the dataset and
  write a comparison report to ``eval/reports/``.
"""
