# Retrieval ablations and the cost/latency/quality frontier

- configurations measured: 27
- corpus: cbam_synthetic (83d3414da810)
- generator: extractive / extractive-v1
- judge: heuristic / rule-based-v1
- embedder: hashed / hashed-ngram-projection-v1
- api mode: replay
- quality: composite of the judge axes and recall@k, weighted by `configs/gate/`

## Results

`*` marks a configuration on at least one frontier (9 of 27).

|  | config | quality | recall@k | nDCG@10 | MRR | grounded | relevant | citations | p50 ms | p95 ms | ctx tok | measured $/q | projected $/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| * | fixed/hybrid/k=10 | 0.828 | 0.867 | 0.437 | 0.310 | 3.87 | 3.80 | 5.00 | 7.9 | 11.4 | 2793 | 0.000000 | 0.010471 |
| * | fixed/bm25/k=10 | 0.807 | 0.733 | 0.483 | 0.405 | 3.93 | 3.80 | 5.00 | 0.6 | 0.8 | 2786 | 0.000000 | 0.010347 |
| * | section/bm25/k=10 | 0.792 | 0.733 | 0.578 | 0.531 | 3.73 | 3.80 | 5.00 | 0.5 | 0.8 | 1852 | 0.000000 | 0.007510 |
| * | section/hybrid/k=10 | 0.785 | 0.733 | 0.559 | 0.509 | 3.73 | 3.67 | 5.00 | 7.8 | 10.6 | 1895 | 0.000000 | 0.007396 |
|  | recursive/bm25/k=10 | 0.773 | 0.733 | 0.506 | 0.436 | 3.80 | 3.73 | 4.73 | 0.6 | 0.8 | 2557 | 0.000000 | 0.009682 |
| * | fixed/bm25/k=5 | 0.770 | 0.600 | 0.442 | 0.389 | 3.80 | 3.80 | 5.00 | 0.5 | 0.7 | 1398 | 0.000000 | 0.006183 |
|  | section/bm25/k=5 | 0.765 | 0.600 | 0.533 | 0.511 | 3.73 | 3.80 | 5.00 | 0.4 | 0.7 | 1025 | 0.000000 | 0.005030 |
| * | section/bm25/k=3 | 0.765 | 0.600 | 0.533 | 0.511 | 3.73 | 3.80 | 5.00 | 0.4 | 0.7 | 586 | 0.000000 | 0.003712 |
|  | recursive/hybrid/k=10 | 0.750 | 0.667 | 0.466 | 0.407 | 3.80 | 3.53 | 4.73 | 8.3 | 11.2 | 2566 | 0.000000 | 0.009844 |
|  | fixed/bm25/k=3 | 0.743 | 0.467 | 0.384 | 0.356 | 3.80 | 3.80 | 5.00 | 0.6 | 1.1 | 849 | 0.000000 | 0.004534 |
|  | recursive/bm25/k=5 | 0.742 | 0.600 | 0.463 | 0.419 | 3.73 | 3.73 | 4.73 | 0.5 | 0.7 | 1309 | 0.000000 | 0.005938 |
|  | section/hybrid/k=5 | 0.740 | 0.533 | 0.492 | 0.480 | 3.67 | 3.67 | 5.00 | 8.6 | 10.7 | 984 | 0.000000 | 0.004663 |
|  | fixed/hybrid/k=5 | 0.730 | 0.400 | 0.288 | 0.250 | 3.80 | 3.80 | 5.00 | 8.0 | 11.0 | 1399 | 0.000000 | 0.006290 |
|  | fixed/dense/k=10 | 0.727 | 0.533 | 0.356 | 0.297 | 4.07 | 3.20 | 4.73 | 6.5 | 7.8 | 2698 | 0.000000 | 0.009760 |
| * | section/hybrid/k=3 | 0.727 | 0.467 | 0.467 | 0.467 | 3.67 | 3.67 | 5.00 | 8.4 | 18.2 | 570 | 0.000000 | 0.003421 |
| * | recursive/bm25/k=3 | 0.723 | 0.467 | 0.409 | 0.389 | 3.67 | 4.00 | 4.73 | 0.5 | 0.7 | 777 | 0.000000 | 0.004340 |
|  | fixed/dense/k=5 | 0.717 | 0.467 | 0.335 | 0.289 | 3.93 | 3.47 | 4.73 | 6.5 | 8.4 | 1359 | 0.000000 | 0.005742 |
|  | fixed/dense/k=3 | 0.712 | 0.467 | 0.335 | 0.289 | 3.87 | 3.47 | 4.73 | 7.0 | 8.2 | 813 | 0.000000 | 0.004104 |
|  | fixed/hybrid/k=3 | 0.712 | 0.333 | 0.260 | 0.233 | 3.73 | 3.80 | 5.00 | 7.9 | 10.7 | 841 | 0.000000 | 0.004616 |
|  | recursive/hybrid/k=5 | 0.708 | 0.467 | 0.401 | 0.380 | 3.60 | 3.80 | 4.73 | 7.0 | 9.6 | 1262 | 0.000000 | 0.005932 |
|  | recursive/hybrid/k=3 | 0.690 | 0.400 | 0.375 | 0.367 | 3.53 | 3.80 | 4.73 | 7.7 | 11.6 | 767 | 0.000000 | 0.004447 |
|  | recursive/dense/k=10 | 0.687 | 0.467 | 0.349 | 0.313 | 3.53 | 3.87 | 4.47 | 6.6 | 8.3 | 2539 | 0.000000 | 0.009822 |
|  | section/dense/k=10 | 0.683 | 0.467 | 0.409 | 0.389 | 3.53 | 3.80 | 4.47 | 7.5 | 9.5 | 1839 | 0.000000 | 0.007120 |
|  | section/dense/k=5 | 0.668 | 0.467 | 0.409 | 0.389 | 3.33 | 3.80 | 4.47 | 8.0 | 9.5 | 946 | 0.000000 | 0.004441 |
| * | section/dense/k=3 | 0.663 | 0.467 | 0.409 | 0.389 | 3.27 | 3.80 | 4.47 | 7.8 | 9.2 | 571 | 0.000000 | 0.003316 |
|  | recursive/dense/k=5 | 0.658 | 0.400 | 0.329 | 0.306 | 3.33 | 3.87 | 4.47 | 7.0 | 8.0 | 1243 | 0.000000 | 0.005934 |
|  | recursive/dense/k=3 | 0.645 | 0.333 | 0.300 | 0.289 | 3.33 | 3.87 | 4.47 | 6.9 | 7.8 | 753 | 0.000000 | 0.004463 |

## Frontiers

Measured cost is zero for every run above because the generator makes no API call, so the cost frontier is plotted on projected cost: this run's context and answer tokens priced at the configured model's rates. It is a projection, not a measurement, and it is labelled as one wherever it appears.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/pareto_quality_cost_dark.png">
  <img alt="Quality against projected cost per query" src="figures/pareto_quality_cost.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/pareto_quality_latency_dark.png">
  <img alt="Quality against p95 serving latency" src="figures/pareto_quality_latency.png">
</picture>

### Non-dominated configurations

| frontier | config | quality | projected $/q | p95 ms |
| --- | --- | --- | --- | --- |
| cost | fixed/hybrid/k=10 | 0.828 | 0.010471 | 11.4 |
| cost | fixed/bm25/k=10 | 0.807 | 0.010347 | 0.8 |
| cost | section/bm25/k=10 | 0.792 | 0.007510 | 0.8 |
| cost | section/hybrid/k=10 | 0.785 | 0.007396 | 10.6 |
| cost | fixed/bm25/k=5 | 0.770 | 0.006183 | 0.7 |
| cost | section/bm25/k=3 | 0.765 | 0.003712 | 0.7 |
| cost | section/hybrid/k=3 | 0.727 | 0.003421 | 18.2 |
| cost | section/dense/k=3 | 0.663 | 0.003316 | 9.2 |
| latency | fixed/hybrid/k=10 | 0.828 | 0.010471 | 11.4 |
| latency | fixed/bm25/k=10 | 0.807 | 0.010347 | 0.8 |
| latency | section/bm25/k=10 | 0.792 | 0.007510 | 0.8 |
| latency | fixed/bm25/k=5 | 0.770 | 0.006183 | 0.7 |
| latency | section/bm25/k=3 | 0.765 | 0.003712 | 0.7 |
| latency | recursive/bm25/k=3 | 0.723 | 0.004340 | 0.7 |
