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
- inputs digest: 23bbfb962fb0

Labels come from more than one source. Independent human labels and `seed-author` ratings are reported separately; they are never pooled.

## Agreement with aditya labels

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 15 | 0.800 | 0.986 | 0.933 | 1.000 | 4.200 | 4.267 | 0.067 |
| relevance | 15 | 0.348 | 0.888 | 0.733 | 1.000 | 4.533 | 4.400 | -0.133 |
| citation_correctness | 15 | 0.348 | 0.539 | 0.667 | 0.733 | 3.667 | 4.000 | 0.333 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 12 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1 | 0 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 1 | 0 | 0 |
| 5 | 0 | 0 | 0 | 1 | 11 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 2 | 0 | 1 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 1 | 0 | 0 | 0 | 9 |

### Worst disagreements (aditya)

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| seed-a-004 | citation_correctness | 1 | 5 | 4 | the number of CBAM certificates on its account in the registry corresponds to at least 80 per cent of the embedded emissions in all goods it has imported since the beginning of the calendar year | At the end of each quarter the declarant's registry account must hold certificates covering at least 80 per cent of the embedded emissions in all goods importe… |
| seed-a-008 | citation_correctness | 5 | 1 | 4 | Amounts are indexed to the European index of consumer prices | The penalty is between EUR 10 and EUR 50 per tonne of unreported embedded emissions. It is higher where more than two incomplete or incorrect reports have been… |
| seed-a-003 | citation_correctness | 1 | 3 | 2 | syn_reg_main#0006 | The competent authority must decide within 90 calendar days of receiving a complete application [syn_reg_main#0006]. The period is suspended while the authorit… |
| seed-a-007 | citation_correctness | 1 | 3 | 2 | at the market price prevailing on the date of the request | Yes. A Member State repurchases any excess certificates on request, with no limit on the quantity, at the market price prevailing on the date of the request [s… |
| seed-a-003 | groundedness | 1 | 2 | 1 | syn_reg_main#0006 | The competent authority must decide within 90 calendar days of receiving a complete application [syn_reg_main#0006]. The period is suspended while the authorit… |
| seed-a-005 | citation_correctness | 1 | 2 | 1 | the price of the last week with auctions is used | The price is the average of the closing prices of emission allowances on the common auction platform for each calendar week [syn_faq#0000]. Where a week has no… |
| seed-a-007 | relevance | 2 | 3 | 1 | Requests can be made at any point in the year | Yes. A Member State repurchases any excess certificates on request, with no limit on the quantity, at the market price prevailing on the date of the request [s… |
| seed-a-009 | relevance | 5 | 4 | 1 | the reporting declarant may modify a submitted CBAM report until two months after the end of the relevant reporting quarter | A submitted report can be modified until two months after the end of the relevant quarter [syn_impl_reg_reporting#0002]. For the first two reporting periods on… |
| seed-a-013 | relevance | 2 | 1 | 1 | Registration opened in 2023 and is free | Operators should register on the CBAM transitional registry and obtain an operator identification number, which importers then quote in their quarterly reports… |
| seed-a-015 | relevance | 4 | 3 | 1 | no longer satisfies the criteria set out in Article 9 | Status is revoked where the holder no longer meets the authorisation criteria [syn_reg_main#0007]. |

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
