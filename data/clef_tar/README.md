# The CLEF review the agents are measured against

The test plan against a standard dataset (TF-03, DS-01 to DS-04 in
`docs/testing.md`) measures the two agents against one Cochrane review
from the CLEF 2019 technology-assisted review collection (Kanoulas et al.,
2019; the collection is `github.com/CLEF-TAR/tar`, MIT licence): topic
CD011571, "Antistreptococcal interventions for guttate and chronic plaque
psoriasis", 146 PubMed identifiers, 15 of them judged relevant at abstract
level by the review's authors.

## What this folder holds

`cd011571_fifty.csv` is the fifty papers a run screens: every one of the 15
the collection's abstract-level judgements include, and 35 drawn from the
131 it excludes with a fixed starting number, 11571, so the draw can be
repeated exactly. The rows are sorted by identifier so the order carries
nothing about the labels. Two columns, `pmid` and `include`, holding
`include` or `exclude`; nothing from any paper. The draw is
`scripts/measure_against_clef.py draw`, given the collection's topic file
`2019-TAR/Task2/Testing/Intervention/topics/CD011571` and its qrels file
`2019-TAR/Task2/Testing/Intervention/qrels/full.test.intervention.abs.2019.qrels`
at the collection's commit dbc13d02bb3e2f8ebc90e62ff47f5eb591e5ca20, read
on 14 September 2026.

`private/` is ignored by the repository. It holds the collection's files as
downloaded, the fifty records as fetched from PubMed with their fingerprint,
and every measurement run's output, which carries the model's summaries and
extracts of the abstracts. None of that is committed, for the reason the
hand-screened sample's abstracts are not (R-12).

The fixed inclusion and exclusion criteria the agents are given are
condensed from the review's protocol and kept outside the repository, with
a note of how they were derived, because the protocol is Cochrane's text.

## Why fifteen in fifty

Keeping every relevant paper fixes the recall figure's denominator at
fifteen before the run. That makes the fifty richer in relevant papers than
a real search would be, 15 in 50 against 15 in 146 for this review, and the
precision figure is reported with that said.
