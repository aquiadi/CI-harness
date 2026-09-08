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

Call the `submit_candidates` tool once with up to ${max_questions} candidate
questions about this chunk. For each question:

- It must be answerable from this chunk alone. If it needs a second document,
  do not produce it.
- It must be a question a compliance officer, importer or installation operator
  would actually ask, phrased in their words rather than in the regulation's.
- It must not quote a distinctive phrase from the chunk. A question that shares
  a rare five-word span with its gold chunk tests string matching, not
  retrieval.
- Prefer questions turning on an obligation, a threshold, a deadline, a scope
  boundary or a definition.
- State the difficulty: `lexical` if the answer shares vocabulary with the
  question, `semantic` if answering requires resolving a paraphrase.

If the chunk is boilerplate (a table of contents, a signature block, a
recitals header, an address), return an empty list.
