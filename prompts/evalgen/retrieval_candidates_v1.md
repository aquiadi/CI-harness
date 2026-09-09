# Retrieval eval candidate generation, v1

You are drafting candidate entries for a retrieval evaluation set. A human
reviews every candidate you produce and discards the ones that are not
answerable from the chunk they are attached to, so precision matters more than
volume: produce fewer, sharper questions rather than filling the quota.

## Source chunk

id: ${chunk_id}
document: ${document_title}

${chunk_text}

## What to produce

Call the `submit_candidates` tool once, with at most ${max_questions}
candidates. For each candidate:

- `question`: answerable from this chunk alone. If it needs a second document,
  do not produce it. Phrase it as a compliance officer, importer or installation
  operator would ask it, in their words rather than the regulation's. Do not
  reuse a distinctive phrase from the chunk: a question sharing a rare
  five-word span with its gold evidence tests string matching, not retrieval.
- `evidence_span`: the sentence from the chunk above that answers the question,
  copied character for character. Do not paraphrase it, do not trim it to a
  fragment, and do not join two separate sentences. This span is what the
  metrics match against, so an inexact copy makes the slot unusable.
- `difficulty`: `lexical` if the question and the evidence share the words that
  matter, `semantic` if answering requires resolving a paraphrase.
- `rationale`: one sentence on what the question tests, for the reviewer.

Prefer questions turning on an obligation, a threshold, a deadline, a scope
boundary or a definition.

If the chunk is boilerplate -- a table of contents, a signature block, a
recitals header, an address, a heading with no substance -- return an empty
list. An empty list is a correct answer and is better than a weak question.
