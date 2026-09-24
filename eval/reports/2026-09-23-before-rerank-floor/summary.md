# NexusRAG evaluation report

*2026-09-23 16:13* · 61 questions (53 answerable, 8 unanswerable) · generation `gemini-3.5-flash-lite`, fast `gemini-3.5-flash-lite`, embeddings `gemini-embedding-2` · judge `gemini-3.5-flash-lite` · k = 5

| Configuration | Hit@5 | MRR | Context precision | Context recall | Faithfulness | Answer relevance | Correct refusals | False refusals | Latency p50 / p95 | Cost / query |
|---|---|---|---|---|---|---|---|---|---|---|
| Dense only (baseline) | 100% | 0.93 | 90% | 100% | 100% | 100% | 100% | 0% | 3.2 s / 7.5 s | $0.0009 |
| Hybrid (dense + BM25 + RRF) | 92% | 0.85 | 83% | 92% | 100% | 92% | 100% | 8% | 4.0 s / 14.9 s | $0.0010 |
| Hybrid + reranking | 87% | 0.84 | 82% | 86% | 100% | 85% | 100% | 15% | 12.6 s / 18.7 s | $0.0008 |
| Full (hybrid + reranking + query rewriting + self-correction) | 96% | 0.89 | 87% | 96% | 100% | 92% | 100% | 8% | 21.5 s / 47.1 s | $0.0017 |

Retrieval metrics are computed on the parent sections given to the answer model. Faithfulness skips refusals, answer relevance counts answerable questions only, and cost covers the system's own model calls (not the judge's). Latency excludes questions slowed by rate limiting (retries or model fallbacks): Dense only (baseline) 7, Hybrid (dense + BM25 + RRF) 4, Hybrid + reranking 6, Full (hybrid + reranking + query rewriting + self-correction) 13.

## Hit rate by question type

| Configuration | comparison (4) | fact (6) | identifier (6) | keyword (3) | numeric (18) | paraphrase (10) | table (6) | unanswerable (8) |
|---|---|---|---|---|---|---|---|---|
| Dense only (baseline) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 0% |
| Hybrid (dense + BM25 + RRF) | 100% | 100% | 100% | 100% | 100% | 60% | 100% | 0% |
| Hybrid + reranking | 100% | 83% | 100% | 100% | 100% | 40% | 100% | 0% |
| Full (hybrid + reranking + query rewriting + self-correction) | 100% | 83% | 100% | 100% | 100% | 90% | 100% | 0% |

(For unanswerable questions the cell shows how often the system *answered*: lower is better.)

## Where `Full (hybrid + reranking + query rewriting + self-correction)` went wrong

- **q001** (fact) What is the document revision and publication date for the Aurora X1 Field Drone specifications?: source section not retrieved, refused although the documents answer it
- **q048** (paraphrase) How much will Skylark pay toward a desk chair for people working from home?: refused although the documents answer it
- **q052** (paraphrase) How many people is Skylark planning to recruit next quarter, and for what?: source section not retrieved, refused although the documents answer it
- **q055** (identifier) Which product does document SKD-PS-BS2 describe?: refused although the documents answer it
