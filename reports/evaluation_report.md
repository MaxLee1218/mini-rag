# RAG Evaluation Report

## Dataset

- Samples: 10
- Date: 2026-07-15T21:12:50+08:00
- Status: completed
- Dataset: evaluation/dataset/eval_dataset.json

## Retrieval Metrics

| Metric | Score | Valid Samples |
| - | -: | -: |
| Hit Rate | 1.0000 | 8 |
| Abstention Accuracy | 1.0000 | 2 |

## RAGAS Metrics

| Metric | Score | Valid Samples |
| - | -: | -: |
| Faithfulness | 0.8000 | 10 |
| Answer Relevancy | 0.6103 | 10 |
| Context Precision | 0.8167 | 10 |
| Context Recall | 0.9000 | 10 |

## Latency

| Stage | p50 (ms) | p95 (ms) | Samples |
| - | -: | -: | -: |
| Embedding | 16.074 | 8592.449 | 10 |
| Retrieval | 3.788 | 5.316 | 10 |
| Generation | 1029.842 | 1193.727 | 10 |
| Total | 1066.110 | 12580.570 | 10 |

p50 describes typical latency for normal user experience; p95 highlights slow-request bottlenecks.

## Failed Examples

None.
