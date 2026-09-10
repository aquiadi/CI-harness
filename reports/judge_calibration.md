# Judge calibration

- judge: `qwen/qwen3.8-27b` (provider `api`)
- judge prompt: `judge/rubric_v1.md` (035548ced080)
- scale: 1-5
- axes: groundedness, relevance, citation_correctness
- api mode: live
- chunker: recursive_structural
- corpus: cbam_synthetic (83d3414da810)
- embedder: local / BAAI/bge-small-en-v1.5
- index: 4e406a172dfd
- retriever: hybrid (k=10)
- judgments: 15 primary, 15 swapped-context
- answers graded from: `reference`
- inputs digest: d0e312e9ab7b

## Agreement with seed-author labels

**These are not independent human labels.** The only labels available are `seed-author` ratings, authored alongside the answers they rate (see docs/DECISIONS.md D-0019). Agreement measured against them says almost nothing about whether the judge agrees with a person: the same author decided both what the answer would get wrong and what score that deserved. They exist so this pipeline is runnable and testable before anyone has labelled. Run `make label` and regenerate to get a number that means something.

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 4.267 | 4.267 | 0.000 |
| relevance | 15 | 0.500 | 0.906 | 0.800 | 1.000 | 4.600 | 4.400 | -0.200 |
| citation_correctness | 15 | 0.872 | 0.986 | 0.933 | 1.000 | 3.933 | 4.000 | 0.067 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 12 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 1 | 0 | 0 |
| 4 | 0 | 0 | 1 | 0 | 0 |
| 5 | 0 | 0 | 0 | 1 | 11 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 | 0 | 0 |
| 3 | 0 | 0 | 1 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 10 |

### Worst disagreements (seed-author)

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| seed-a-007 | citation_correctness | 2 | 3 | 1 | at the market price prevailing on the date of the request | Yes. A Member State repurchases any excess certificates on request, with no limit on the quantity, at the market price prevailing on the date of the request [s… |
| seed-a-007 | relevance | 4 | 3 | 1 | Requests can be made at any point in the year | Yes. A Member State repurchases any excess certificates on request, with no limit on the quantity, at the market price prevailing on the date of the request [s… |
| seed-a-009 | relevance | 5 | 4 | 1 | the reporting declarant may modify a submitted CBAM report until two months after the end of the relevant reporting quarter | A submitted report can be modified until two months after the end of the relevant quarter [syn_impl_reg_reporting#0002]. For the first two reporting periods on… |
| seed-a-013 | relevance | 2 | 1 | 1 | Registration opened in 2023 and is free | Operators should register on the CBAM transitional registry and obtain an operator identification number, which importers then quote in their quarterly reports… |

### Bias probes

#### Position swap: does reversing the context change the score?

| axis | n | identical score | mean abs difference | max abs difference |
| --- | --- | --- | --- | --- |
| groundedness | 15 | 1.000 | 0.000 | 0 |
| relevance | 15 | 1.000 | 0.000 | 0 |
| citation_correctness | 15 | 0.933 | 0.133 | 2 |

The answer is unchanged between the two scorings; only the order of the retrieved context differs. Any movement is the judge responding to presentation rather than to quality.

#### Length bias: do longer answers score better?

| axis | n | spearman rho | p | mean answer tokens | note |
| --- | --- | --- | --- | --- | --- |
| groundedness | 15 | 0.050 | 0.859 | 51.800 |  |
| relevance | 15 | 0.180 | 0.521 | 51.800 |  |
| citation_correctness | 15 | -0.154 | 0.583 | 51.800 |  |

A strong positive correlation is the failure mode that rewards padding. It is evidence, not proof: longer answers may genuinely be better.

#### Self-preference: does the judge favour its own model family?

Not run: this probe needs answers to the same questions from two generators (config `probes.contrast_generator=groq_contrast`), graded by the same judge. Produce them with `make ablate` and regenerate.
