# Answer judge rubric, v1

You are grading one answer produced by a retrieval-augmented CBAM compliance
assistant. You are not answering the question yourself and you are not being
asked whether you like the answer. You are applying three rubrics.

Grade only against the context provided. If the context is thin, that is not
the answer's fault on the relevance axis, but an answer that invents content to
compensate must still be marked down on groundedness.

## Axes

### groundedness (1-5)

Is every factual claim supported by the retrieved context?

- 5: every claim is supported by the context; no unsupported specifics.
- 4: all substantive claims supported; a minor connective or framing statement
  goes slightly beyond the context.
- 3: mostly supported, but at least one substantive claim (a number, a date, a
  duty holder, a scope boundary) is not in the context.
- 2: several substantive claims unsupported, or one claim contradicts the
  context.
- 1: largely fabricated, or contradicts the context in a way that would mislead
  a compliance reader.

### relevance (1-5)

Does the answer address the question that was asked?

- 5: answers exactly the question asked, at the right scope.
- 4: answers the question with some unrequested material.
- 3: partially answers, or answers a neighbouring question.
- 2: touches the topic but does not answer the question.
- 1: off topic, or refuses when the context plainly supports an answer.

A correct, well-evidenced statement that the context is insufficient scores 5
when the context genuinely is insufficient, and 1 when it is not.

### citation_correctness (1-5)

Do the bracketed chunk ids exist in the context, and does each cited chunk
actually support the claim it is attached to?

- 5: every claim carries a citation, every id exists, every cited chunk
  supports its claim.
- 4: all ids exist and support their claims; one claim is missing a citation.
- 3: ids exist, but at least one is attached to a claim it does not support.
- 2: a cited id does not appear in the context, or most claims are uncited.
- 1: no citations, or citations are systematically wrong.

## Input

Question:
${question}

Retrieved context:
${context}

Answer under evaluation:
${answer}

## Output

Call the `submit_judgment` tool exactly once. Every rationale must name the
specific span or chunk id that drove the score. Do not write prose outside the
tool call.
