# Synthetic corpus

Every file in this directory is **synthetic**. It was written for this
repository. It is not the text of any EU instrument, it has no legal force, and
it must not be relied on for compliance.

It exists because the environment this repo was built in cannot reach EUR-Lex
or the Commission's document servers (see `docs/DECISIONS.md`, D-0009), and a
harness with nothing to measure is not a harness. The documents restate real
CBAM concepts -- transitional reporting, authorised declarants, embedded
emissions, CBAM certificates -- in the structural style of a regulation, so
that chunkers, retrievers and the judge are exercised on text with the shape of
the real thing: article numbering, cross-references, defined terms, thresholds
and dates.

Runs over this corpus are named `cbam_synthetic` and carry their own
`corpus_hash`. Every report states which corpus produced its numbers. Nothing
measured here is presented as a measurement of the system on the real corpus.

To use the real documents instead:

    make corpus                      # fetch and pin the real sources
    make index corpus=cbam           # index them
