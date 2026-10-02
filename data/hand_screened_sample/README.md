# Hand-screened sample

Fifty PubMed records drawn at random from the 827 matching
`ivermectin AND (covid-19 OR sars-cov-2)`, using seed 42.

Random rather than hand-picked, because choosing the records myself would bias
the recall figure towards papers that are easy to find.

The seed records how the fifty were chosen from the 827 that matched on the
day. It does not on its own let the draw be repeated: PubMed returns a
different set as papers are added, so drawing again with the same seed would
only give the same fifty if the same 827 came back in the same order. The
sample that the recall figure is measured against is the one in the file here,
which does not change.

`screening_labels.csv` holds the fifty records with the hand-screening
decision in the `include` column and a short reason in `rationale`.

## What is in these files, and what is not

Each file holds three columns: `pmid`, `include` and `rationale`. Nothing else.

They held more until 3 September 2026: the year, journal, publication types,
title and the full abstract of each record. Eighty-seven of the hundred rows
carried an abstract, and publishing somebody else's abstract needs a basis
nobody had established, which is the same question R-11 asked about the saved
service replies. The abstracts were removed and the files renamed to say what
they now hold (R-12).

Nothing needed for the benchmark was lost. The identifier finds the paper at
its source, and the decision and the reason are what the recall figure is
measured against. The draw as it stood, with the abstracts in it, is kept
privately and is what the reduction was checked against, row by row.

| | First sample | Check sample |
|---|---|---|
| File | `screening_labels.csv` | `screening_labels_check.csv` |
| Seed | 42 | 43 |
| Records | 50 | 50 |
| Included | 3 | 5 |
| Drawn | 26 August 2026 | 26 August 2026 |
| Screened | 26 August 2026 | 27 August 2026 |

The dates are the dates of the commits that added the draw and then its
decisions, not values recorded in the files at the time. They say when each was
in the repository, which is the nearest thing to a record of when it was done.
The check sample was drawn one day and screened the next.

## The screening rules

Each record was judged against one question:

> Did the study evaluate ivermectin as a COVID-19 treatment and report
> clinical, virological or safety outcomes relevant to that treatment?

The question is about what the study set out to do. A paper can mention
ivermectin without studying it, for example when listing drugs that patients
happened to receive, and those are excluded.

**Include** original human research that deliberately evaluates ivermectin as a
COVID-19 treatment, alone or in a defined combination, and reports a clinical,
virological or safety outcome tied to the ivermectin exposure or treatment arm.
Null and underpowered studies are included: study quality is a separate
question from relevance, and excluding weak results would bias the sample
towards findings that happen to be positive.

**Exclude** prophylaxis, reviews, guidelines, opinion pieces, letters,
secondary analyses, retraction notices, non-human research, and records where
ivermectin is only mentioned in passing.

**A missing abstract is not by itself a reason to exclude.** Where there is no
abstract the title, publication type and any available full text are used
instead. Treating every record without an abstract as irrelevant would discard
papers on the basis of what the metadata happens to carry rather than what the
study did.

### Decisions worth recording

- **33994641 included.** No abstract, but the title and publication type
  identify an original retrospective treatment cohort.
- **35383577 excluded.** The title reads like a primary study, but the record
  is a letter reporting a secondary analysis.
- **37377231 included.** The full text reports adverse events attributable to
  ivermectin specifically, not only to the drugs it was compared against.

## Result, first sample

Three of the fifty were included.

| PMID | Reason |
|---|---|
| 35738778 | RCT, hospitalisation outcome |
| 35273939 | ivermectin programme, mortality outcome |
| 33137493 | comparative study, ivermectin assessed |

Forty-seven were excluded. The usual reasons were reviews and commentary rather
than primary studies, laboratory work with no patients, no abstract at all, and
papers where ivermectin was mentioned only in passing.

A low include rate is expected. First-pass screening is mostly rejection, and
that is the work the agent is meant to reduce.

## Why the labels come first

These decisions were written before the agent was ever run against the sample.
Labelling afterwards would mean agreeing with whatever the agent returned, and
the recall figure in SC-01 would measure nothing.

## Revision history

An earlier pass included PMID 35232458 and PMID 36851310. Re-reading both
against the screening question showed ivermectin was incidental in each: the
first is a case report about telling COVID-19 and preeclampsia apart, the
second a cohort study of chronic kidney disease after COVID-19. Both were
changed to `exclude` before any agent run.


## The check sample, and why fifty was not enough

The design sets the target as recall of at least 0.80 against a hand-screened
sample of fifty abstracts. That figure is unchanged and is still the criterion
being reported against. A second sample of fifty was screened as well, for a
reason that only became clear after the first was finished.

Only three of the first fifty were relevant. With three, each paper moves the
recall figure by a third: finding two gives 0.67 and finding all three gives
1.00, with nothing in between. A target of 0.80 cannot be met at all except by
finding every one, so a single missed paper is the difference between passing
the criterion and failing it. That is a property of the sample rather than of
the search being measured.

`screening_labels_check.csv` holds a second fifty, drawn with seed 43 and
excluding every id already in the first sample, so the two do not overlap.
Together they contain eight relevant papers in one hundred records, so each one
moves the figure by roughly an eighth. The second sample also shows whether the
first result was a property of the search or of which fifty records happened to
be drawn.

The recall figure is reported for the first fifty as the design specifies, and
for the two samples together as a check on it.

Five of the fifty were included.

| PMID | Reason |
|---|---|
| 33983065 | RCT, recovery and viral outcomes |
| 33994641 | retrospective cohort, treatment outcomes |
| 34999239 | RCT, viral load and safety |
| 37302406 | RCT, long-COVID outcome |
| 37377231 | pharmacovigilance, ivermectin-specific adverse events |

Across both samples that is eight relevant papers in one hundred records, so
each one moves the recall figure by roughly an eighth rather than a third.

### A limitation to report with the recall figure

Two of these five were judged on information the agent will not see. 33994641
has no abstract at all, and 37377231 was confirmed from its full text. Relevance
is a property of the study rather than of the metadata a search happens to
return, so both are correctly marked relevant. It does mean the agent may miss
them for reasons that are not failures of its search, and the recall figure
should be reported with that said rather than left for a reader to notice.
