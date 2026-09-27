# Judge calibration

- corpus: cbam
- judges on disk: `qwen/qwen3.8-27b`, `rule-based-v1`, `rule-based-v2`
- label sources: aquiadi (delegated)

## Judges compared against aquiadi (delegated) labels

| judge | retriever | embedder | n | kappa groundedness | kappa relevance | kappa citation_correctness |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen/qwen3.8-27b` | hybrid (k=8) | local / BAAI/bge-small-en-v1.5 | 12 | 0.341 | 0.333 | 0.676 |
| `rule-based-v1` | hybrid (k=8) | local / BAAI/bge-small-en-v1.5 | 12 | -0.043 | 0.091 | 0.176 |
| `rule-based-v2` | hybrid (k=8) | local / BAAI/bge-small-en-v1.5 | 12 | -0.043 | 0.091 | 0.200 |

Each judge scored the answers against context from its own retrieval stack. Where the stacks differ, so did what the judges were shown, and the gap between two rows is not only a difference between judges.

## Judge: `qwen/qwen3.8-27b`

- judge: `qwen/qwen3.8-27b` (provider `api`)
- judge prompt: `judge/rubric_v1.md` (035548ced080)
- scale: 1-5
- axes: groundedness, relevance, citation_correctness
- api mode: live
- chunker: recursive_structural
- corpus: cbam (ab0fc00c7468)
- embedder: local / BAAI/bge-small-en-v1.5
- index: db0e88aed000
- retriever: hybrid (k=8)
- judgments: 12 primary, 12 swapped-context
- answers graded from: `reference`
- inputs digest: 546cc96d2060

### Agreement with aquiadi (delegated) labels

**These are not independent human labels.** The repository owner delegated this labelling rather than doing it, so these scores came from the same process that drafted the questions and answers (docs/DECISIONS.md D-0061). Agreement against them says how consistently a rubric was applied, not whether the judge agrees with a person. Replace them with `make label` to get that number.

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 12 | 0.341 | 0.722 | 0.583 | 0.833 | 3.917 | 3.750 | -0.167 |
| relevance | 12 | 0.333 | 0.540 | 0.750 | 0.833 | 4.667 | 4.083 | -0.583 |
| citation_correctness | 12 | 0.676 | 0.764 | 0.750 | 0.833 | 3.333 | 2.833 | -0.500 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 2 | 1 | 1 | 0 | 0 | 1 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 1 | 2 | 5 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 1 | 0 | 1 | 1 | 8 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 0 | 2 | 0 | 0 | 0 |
| 3 | 1 | 0 | 2 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 1 | 0 | 1 | 3 |

#### Worst disagreements (aquiadi (delegated))

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| cbam-a-006 | relevance | 5 | 1 | 4 | Each Member State sets a fixed CBAM certificate price once a year, based on its national carbon tax | Each Member State sets a fixed CBAM certificate price once a year, based on its national carbon tax [reg_2023_956#0108]. |
| cbam-a-001 | citation_correctness | 5 | 2 | 3 | reports due within one month | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |
| cbam-a-010 | groundedness | 2 | 5 | 3 | reg_2023_956#0111 | The retrieved context does not say how many certificates must be held during the year, so I cannot answer this. |
| cbam-a-004 | groundedness | 5 | 3 | 2 | Up to 20% of the total embedded emissions may be determined using estimations | Up to 20 % of the total embedded emissions of complex goods may be based on estimations made available by the operators of the installations [reg_2023_956#0000… |
| cbam-a-006 | citation_correctness | 3 | 1 | 2 | Each Member State sets a fixed CBAM certificate price once a year, based on its national carbon tax | Each Member State sets a fixed CBAM certificate price once a year, based on its national carbon tax [reg_2023_956#0108]. |
| cbam-a-008 | relevance | 5 | 3 | 2 | the declarant is refunded their purchase price | On 1 July each year the Commission cancels certificates bought during the year before the previous calendar year, and the declarant is refunded their purchase … |
| cbam-a-001 | groundedness | 5 | 4 | 1 | reports due within one month | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |
| cbam-a-001 | relevance | 5 | 4 | 1 | reports due within one month | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |
| cbam-a-002 | groundedness | 2 | 1 | 1 | impl_reg_2023_1773#0029 | A reporting declarant may modify a submitted CBAM report until three months after the end of the relevant reporting quarter [impl_reg_2023_1773#0029]. |
| cbam-a-009 | groundedness | 5 | 4 | 1 | three to five times the standard penalty for each certificate | Such a person is liable for a penalty of three to five times the standard penalty for each certificate they should have surrendered. |

### Bias probes

#### Position swap: does reversing the context change the score?

| axis | n | identical score | mean abs difference | max abs difference |
| --- | --- | --- | --- | --- |
| groundedness | 12 | 0.750 | 0.250 | 1 |
| relevance | 12 | 0.833 | 0.167 | 1 |
| citation_correctness | 12 | 1.000 | 0.000 | 0 |

The answer is unchanged between the two scorings; only the order of the retrieved context differs. Any movement is the judge responding to presentation rather than to quality.

#### Length bias: do longer answers score better?

| axis | n | spearman rho | p | mean answer tokens | note |
| --- | --- | --- | --- | --- | --- |
| groundedness | 12 | -0.023 | 0.944 | 38.750 |  |
| relevance | 12 | 0.177 | 0.582 | 38.750 |  |
| citation_correctness | 12 | 0.664 | 0.019 | 38.750 |  |

A strong positive correlation is the failure mode that rewards padding. It is evidence, not proof: longer answers may genuinely be better.

#### Self-preference: does the judge favour its own model family?

Not run: this probe needs this judge's scores for answers to the same questions from two generators, one of them the judge's own model. Measure a run with that generator (`probes.contrast_generator` names it), grade it with `make judge ARGS="judge.answers_from=run judge.run_id=<run>"`, which adds to this judge's scores rather than replacing them, and regenerate.

## Judge: `rule-based-v1`

- judge: `rule-based-v1` (provider `heuristic`)
- judge prompt: `judge/rubric_v1.md` (n/a)
- scale: 1-5
- axes: groundedness, relevance, citation_correctness
- api mode: replay
- chunker: recursive_structural
- corpus: cbam (ab0fc00c7468)
- embedder: local / BAAI/bge-small-en-v1.5
- index: db0e88aed000
- retriever: hybrid (k=8)
- judgments: 12 primary, 12 swapped-context
- answers graded from: `reference`
- inputs digest: d2a20fd9f875

### Agreement with aquiadi (delegated) labels

**These are not independent human labels.** The repository owner delegated this labelling rather than doing it, so these scores came from the same process that drafted the questions and answers (docs/DECISIONS.md D-0061). Agreement against them says how consistently a rubric was applied, not whether the judge agrees with a person. Replace them with `make label` to get that number.

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 12 | -0.043 | -0.136 | 0.000 | 0.333 | 3.917 | 3.250 | -0.667 |
| relevance | 12 | 0.091 | 0.177 | 0.167 | 0.250 | 4.667 | 2.667 | -2.000 |
| citation_correctness | 12 | 0.176 | 0.410 | 0.417 | 0.583 | 3.333 | 3.000 | -0.333 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 0 | 1 | 2 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 2 | 3 | 3 | 0 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 5 | 4 | 1 | 1 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 2 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 3 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 2 | 0 | 0 | 0 | 3 |

#### Worst disagreements (aquiadi (delegated))

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| cbam-a-001 | citation_correctness | 5 | 1 | 4 | rule-based baseline | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |
| cbam-a-012 | citation_correctness | 5 | 1 | 4 | rule-based baseline | The documents do not address the VAT treatment of CBAM certificates, so I cannot answer this from the context provided. |
| cbam-a-003 | relevance | 5 | 2 | 3 | rule-based baseline | The penalty is between EUR 10 and EUR 50 per tonne of unreported emissions, and it increases in line with the European index of consumer prices [impl_reg_2023_… |
| cbam-a-007 | relevance | 5 | 2 | 3 | rule-based baseline | Yes. On request, excess certificates are repurchased, but the number is limited to one third of the certificates the declarant purchased during the previous ca… |
| cbam-a-008 | relevance | 5 | 2 | 3 | rule-based baseline | On 1 July each year the Commission cancels certificates bought during the year before the previous calendar year, and the declarant is refunded their purchase … |
| cbam-a-009 | relevance | 5 | 2 | 3 | rule-based baseline | Such a person is liable for a penalty of three to five times the standard penalty for each certificate they should have surrendered. |
| cbam-a-011 | groundedness | 5 | 2 | 3 | rule-based baseline | Yes. Goods whose value does not exceed EUR 150 per consignment are treated as exempt [guidance_importers#9999]. |
| cbam-a-011 | relevance | 5 | 2 | 3 | rule-based baseline | Yes. Goods whose value does not exceed EUR 150 per consignment are treated as exempt [guidance_importers#9999]. |
| cbam-a-012 | groundedness | 5 | 2 | 3 | rule-based baseline | The documents do not address the VAT treatment of CBAM certificates, so I cannot answer this from the context provided. |
| cbam-a-001 | groundedness | 5 | 3 | 2 | rule-based baseline | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |

### Bias probes

#### Position swap: does reversing the context change the score?

| axis | n | identical score | mean abs difference | max abs difference |
| --- | --- | --- | --- | --- |
| groundedness | 12 | 1.000 | 0.000 | 0 |
| relevance | 12 | 1.000 | 0.000 | 0 |
| citation_correctness | 12 | 1.000 | 0.000 | 0 |

The answer is unchanged between the two scorings; only the order of the retrieved context differs. Any movement is the judge responding to presentation rather than to quality.

#### Length bias: do longer answers score better?

| axis | n | spearman rho | p | mean answer tokens | note |
| --- | --- | --- | --- | --- | --- |
| groundedness | 12 | 0.527 | 0.078 | 38.750 |  |
| relevance | 12 | 0.067 | 0.836 | 38.750 |  |
| citation_correctness | 12 | 0.683 | 0.014 | 38.750 |  |

A strong positive correlation is the failure mode that rewards padding. It is evidence, not proof: longer answers may genuinely be better.

#### Self-preference: does the judge favour its own model family?

Not run: this probe needs this judge's scores for answers to the same questions from two generators, one of them the judge's own model. Measure a run with that generator (`probes.contrast_generator` names it), grade it with `make judge ARGS="judge.answers_from=run judge.run_id=<run>"`, which adds to this judge's scores rather than replacing them, and regenerate.

## Judge: `rule-based-v2`

- judge: `rule-based-v2` (provider `heuristic`)
- judge prompt: `judge/rubric_v1.md` (n/a)
- scale: 1-5
- axes: groundedness, relevance, citation_correctness
- api mode: replay
- chunker: recursive_structural
- corpus: cbam (ab0fc00c7468)
- embedder: local / BAAI/bge-small-en-v1.5
- index: db0e88aed000
- retriever: hybrid (k=8)
- judgments: 12 primary, 12 swapped-context
- answers graded from: `reference`
- inputs digest: beba23ec47e3

### Agreement with aquiadi (delegated) labels

**These are not independent human labels.** The repository owner delegated this labelling rather than doing it, so these scores came from the same process that drafted the questions and answers (docs/DECISIONS.md D-0061). Agreement against them says how consistently a rubric was applied, not whether the judge agrees with a person. Replace them with `make label` to get that number.

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 12 | -0.043 | -0.136 | 0.000 | 0.333 | 3.917 | 3.250 | -0.667 |
| relevance | 12 | 0.091 | 0.177 | 0.167 | 0.250 | 4.667 | 2.667 | -2.000 |
| citation_correctness | 12 | 0.200 | 0.430 | 0.417 | 0.583 | 3.333 | 2.667 | -0.667 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 0 | 1 | 2 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 2 | 3 | 3 | 0 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 5 | 4 | 1 | 1 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 2 | 0 | 0 | 0 | 0 |
| 3 | 1 | 0 | 0 | 0 | 2 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 2 | 0 | 0 | 0 | 3 |

#### Worst disagreements (aquiadi (delegated))

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| cbam-a-001 | citation_correctness | 5 | 1 | 4 | rule-based baseline | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |
| cbam-a-012 | citation_correctness | 5 | 1 | 4 | rule-based baseline | The documents do not address the VAT treatment of CBAM certificates, so I cannot answer this from the context provided. |
| cbam-a-003 | relevance | 5 | 2 | 3 | rule-based baseline | The penalty is between EUR 10 and EUR 50 per tonne of unreported emissions, and it increases in line with the European index of consumer prices [impl_reg_2023_… |
| cbam-a-007 | relevance | 5 | 2 | 3 | rule-based baseline | Yes. On request, excess certificates are repurchased, but the number is limited to one third of the certificates the declarant purchased during the previous ca… |
| cbam-a-008 | relevance | 5 | 2 | 3 | rule-based baseline | On 1 July each year the Commission cancels certificates bought during the year before the previous calendar year, and the declarant is refunded their purchase … |
| cbam-a-009 | relevance | 5 | 2 | 3 | rule-based baseline | Such a person is liable for a penalty of three to five times the standard penalty for each certificate they should have surrendered. |
| cbam-a-011 | groundedness | 5 | 2 | 3 | rule-based baseline | Yes. Goods whose value does not exceed EUR 150 per consignment are treated as exempt [guidance_importers#9999]. |
| cbam-a-011 | relevance | 5 | 2 | 3 | rule-based baseline | Yes. Goods whose value does not exceed EUR 150 per consignment are treated as exempt [guidance_importers#9999]. |
| cbam-a-012 | groundedness | 5 | 2 | 3 | rule-based baseline | The documents do not address the VAT treatment of CBAM certificates, so I cannot answer this from the context provided. |
| cbam-a-001 | groundedness | 5 | 3 | 2 | rule-based baseline | Each quarterly CBAM report must be submitted to the CBAM Transitional Registry no later than one month after the end of the quarter it covers [impl_reg_2023_17… |

### Bias probes

#### Position swap: does reversing the context change the score?

| axis | n | identical score | mean abs difference | max abs difference |
| --- | --- | --- | --- | --- |
| groundedness | 12 | 1.000 | 0.000 | 0 |
| relevance | 12 | 1.000 | 0.000 | 0 |
| citation_correctness | 12 | 1.000 | 0.000 | 0 |

The answer is unchanged between the two scorings; only the order of the retrieved context differs. Any movement is the judge responding to presentation rather than to quality.

#### Length bias: do longer answers score better?

| axis | n | spearman rho | p | mean answer tokens | note |
| --- | --- | --- | --- | --- | --- |
| groundedness | 12 | 0.527 | 0.078 | 38.750 |  |
| relevance | 12 | 0.067 | 0.836 | 38.750 |  |
| citation_correctness | 12 | 0.767 | 0.004 | 38.750 |  |

A strong positive correlation is the failure mode that rewards padding. It is evidence, not proof: longer answers may genuinely be better.

#### Self-preference: does the judge favour its own model family?

Not run: this probe needs this judge's scores for answers to the same questions from two generators, one of them the judge's own model. Measure a run with that generator (`probes.contrast_generator` names it), grade it with `make judge ARGS="judge.answers_from=run judge.run_id=<run>"`, which adds to this judge's scores rather than replacing them, and regenerate.
