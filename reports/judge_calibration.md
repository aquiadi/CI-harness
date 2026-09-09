# Judge calibration

- judge: `rule-based-v1` (provider `heuristic`)
- judge prompt: `judge/rubric_v1.md` (n/a)
- scale: 1-5
- axes: groundedness, relevance, citation_correctness
- api mode: replay
- chunker: fixed_token
- corpus: cbam_synthetic (83d3414da810)
- embedder: hashed / hashed-ngram-projection-v1
- index: 6af62afaacd7
- retriever: hybrid (k=10)
- judgments: 15 primary, 15 swapped-context
- answers graded from: `reference`
- inputs digest: 1470ce332d52

## Agreement with seed-author labels

**These are not independent human labels.** The only labels available are `seed-author` ratings, authored alongside the answers they rate (see docs/DECISIONS.md D-0019). Agreement measured against them says almost nothing about whether the judge agrees with a person: the same author decided both what the answer would get wrong and what score that deserved. They exist so this pipeline is runnable and testable before anyone has labelled. Run `make label` and regenerate to get a number that means something.

| axis | n | kappa | quadratic kappa | exact | within 1 | mean human | mean judge | mean signed error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groundedness | 15 | 0.030 | 0.269 | 0.133 | 0.733 | 4.267 | 3.867 | -0.400 |
| relevance | 15 | 0.005 | 0.087 | 0.067 | 0.333 | 4.600 | 2.600 | -2.000 |
| citation_correctness | 15 | 0.268 | 0.372 | 0.600 | 0.667 | 3.933 | 3.400 | -0.533 |

`mean signed error` is judge minus human: positive means the judge is more generous than the labeller.

#### Confusion matrix: groundedness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 1 | 1 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 2 | 8 | 2 |

#### Confusion matrix: relevance

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 |
| 3 | 0 | 1 | 0 | 0 | 0 |
| 4 | 0 | 1 | 0 | 0 | 0 |
| 5 | 1 | 4 | 4 | 3 | 0 |

#### Confusion matrix: citation_correctness

| human \ judge | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0 | 0 | 0 | 0 |
| 2 | 1 | 0 | 0 | 0 | 1 |
| 3 | 0 | 0 | 0 | 0 | 1 |
| 4 | 0 | 0 | 0 | 0 | 0 |
| 5 | 3 | 0 | 0 | 0 | 7 |

### Worst disagreements (seed-author)

| example | axis | human | judge | gap | judge rationale | answer |
| --- | --- | --- | --- | --- | --- | --- |
| seed-a-002 | citation_correctness | 5 | 1 | 4 | rule-based baseline | The mechanism does not apply to goods whose total intrinsic value per consignment does not exceed EUR 150 [syn_reg_main#0001]. The threshold is expressed per c… |
| seed-a-009 | citation_correctness | 5 | 1 | 4 | rule-based baseline | A submitted report can be modified until two months after the end of the relevant quarter [syn_impl_reg_reporting#0002]. For the first two reporting periods on… |
| seed-a-010 | citation_correctness | 5 | 1 | 4 | rule-based baseline | No. For iron and steel, aluminium and hydrogen only direct emissions are covered during the transitional period; indirect emissions are covered for cement, fer… |
| seed-a-011 | relevance | 5 | 1 | 4 | rule-based baseline | The retrieved context does not address the VAT treatment of CBAM certificates. It covers how certificates are sold, priced, surrendered, repurchased and cancel… |
| seed-a-005 | citation_correctness | 2 | 5 | 3 | rule-based baseline | The price is the average of the closing prices of emission allowances on the common auction platform for each calendar week [syn_faq#0000]. Where a week has no… |
| seed-a-005 | relevance | 5 | 2 | 3 | rule-based baseline | The price is the average of the closing prices of emission allowances on the common auction platform for each calendar week [syn_faq#0000]. Where a week has no… |
| seed-a-007 | groundedness | 1 | 4 | 3 | rule-based baseline | Yes. A Member State repurchases any excess certificates on request, with no limit on the quantity, at the market price prevailing on the date of the request [s… |
| seed-a-008 | relevance | 5 | 2 | 3 | rule-based baseline | The penalty is between EUR 10 and EUR 50 per tonne of unreported embedded emissions. It is higher where more than two incomplete or incorrect reports have been… |
| seed-a-009 | relevance | 5 | 2 | 3 | rule-based baseline | A submitted report can be modified until two months after the end of the relevant quarter [syn_impl_reg_reporting#0002]. For the first two reporting periods on… |
| seed-a-012 | relevance | 5 | 2 | 3 | rule-based baseline | The indirect customs representative applies where the importer is not established in a Member State [syn_reg_main#0003]. |

### Bias probes

#### Position swap: does reversing the context change the score?

| axis | n | identical score | mean abs difference | max abs difference |
| --- | --- | --- | --- | --- |
| groundedness | 15 | 1.000 | 0.000 | 0 |
| relevance | 15 | 1.000 | 0.000 | 0 |
| citation_correctness | 15 | 1.000 | 0.000 | 0 |

The answer is unchanged between the two scorings; only the order of the retrieved context differs. Any movement is the judge responding to presentation rather than to quality.

#### Length bias: do longer answers score better?

| axis | n | spearman rho | p | mean answer tokens | note |
| --- | --- | --- | --- | --- | --- |
| groundedness | 15 | 0.106 | 0.708 | 51.800 |  |
| relevance | 15 | -0.240 | 0.390 | 51.800 |  |
| citation_correctness | 15 | 0.047 | 0.867 | 51.800 |  |

A strong positive correlation is the failure mode that rewards padding. It is evidence, not proof: longer answers may genuinely be better.

#### Self-preference: does the judge favour its own model family?

Not run: this probe needs answers to the same questions from two generators (config `probes.contrast_generator=haiku`), graded by the same judge. Produce them with `make ablate` and regenerate.
