## Quality gate

**PASS** -- no threshold exceeded.

| check | metric | baseline | this run | delta | limit | status |
| --- | --- | --- | --- | --- | --- | --- |
| quality | composite_quality | 0.828333 | 0.828333 | 0.00% | 2.00% | PASS |
| p95 latency | p95_latency_s | 0.011426 | 0.011143 | -2.47% | 20.00% | PASS |
| cost per query | projected_cost_per_query_usd | 0.010471 | 0.010471 | 0.00% | 15.00% | PASS |

### Supporting metrics (not gated)

| metric | baseline | this run |
| --- | --- | --- |
| recall@k | 0.867 | 0.867 |
| nDCG@10 | 0.437 | 0.437 |
| MRR | 0.310 | 0.310 |
| groundedness | 3.87 | 3.87 |
| relevance | 3.80 | 3.80 |
| citation correctness | 5.00 | 5.00 |
| p50 latency (s) | 0.0079 | 0.0080 |
| context tokens/query | 2793 | 2793 |

### Provenance

- baseline run: `20260909T014505Z-4b8a0ad44f15` frozen 2026-09-09T01:45:05Z
- this run: `20260909T015221Z-4b8a0ad44f15`
- corpus: cbam_synthetic (83d3414da810)
- api mode: replay
- cost compared on `projected_cost_per_query_usd`
