# NexusRAG evaluation report

*2026-09-24 13:45* · 61 questions (53 answerable, 8 unanswerable) · generation `gemini-3.5-flash-lite`, fast `gemini-3.5-flash-lite`, embeddings `gemini-embedding-2` · judge `gemini-3.5-flash-lite` · k = 5

| Configuration | Hit@5 | MRR | Context precision | Context recall | Faithfulness | Answer relevance | Correct refusals | False refusals | Latency p50 / p95 | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense only (baseline) | 100% | 0.93 | 90% | 100% | 100% | 100% | 100% | 0% | 3.2 s / 7.5 s | $0.0009 |
| Hybrid (dense + BM25 + RRF) | 92% | 0.85 | 83% | 92% | 100% | 92% | 100% | 8% | 4.0 s / 14.9 s | $0.0010 |
| Hybrid + reranking | 91% | 0.87 | 86% | 91% | 100% | 91% | 100% | 9% | 12.2 s / 18.7 s | $0.0009 |
| Full (hybrid + reranking + query rewriting + self-correction) | 100% | 0.91 | 89% | 100% | 100% | 96% | 100% | 4% | 16.1 s / 47.3 s | $0.0018 |

Retrieval metrics are computed on the parent sections given to the answer model. Faithfulness skips refusals, answer relevance counts answerable questions only, and cost covers the system's own model calls (not the judge's). Latency excludes questions slowed by rate limiting (retries or model fallbacks): Dense only (baseline) 7, Hybrid (dense + BM25 + RRF) 4, Hybrid + reranking 3, Full (hybrid + reranking + query rewriting + self-correction) 19.

## Hit rate by question type

| Configuration | comparison (4) | fact (6) | identifier (6) | keyword (3) | numeric (18) | paraphrase (10) | table (6) | unanswerable (8) |
|---|---|---|---|---|---|---|---|---|
| Dense only (baseline) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% |
| Hybrid (dense + BM25 + RRF) | 100% | 100% | 100% | 100% | 100% | 60% | 100% | 0% |
| Hybrid + reranking | 100% | 100% | 100% | 100% | 100% | 50% | 100% | 0% |
| Full (hybrid + reranking + query rewriting + self-correction) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% |

(For unanswerable questions the cell shows how often the system *answered*: lower is better.)

## Where `Full (hybrid + reranking + query rewriting + self-correction)` went wrong

- **q048** (paraphrase) How much will Skylark pay toward a desk chair for people working from home?: refused although the documents answer it
- **q052** (paraphrase) How many people is Skylark planning to recruit next quarter, and for what?: refused although the documents answer it
