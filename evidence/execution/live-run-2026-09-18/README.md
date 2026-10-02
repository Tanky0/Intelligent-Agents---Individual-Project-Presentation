# The live run of 18 September 2026: what the screen showed, and what the run log holds

Sixteen screenshots taken at the screen on 18 September 2026 during the
eight attempts, and the run log's own lines for the run that went through.
Copied into the repository on 19 September, as decided that day, as the
evidence of execution the brief asks for. Times are local (UTC+2).

## The eight attempts, and which pictures belong to which

| Attempt | Model | Run | When | What happened | Pictures |
|---|---|---|---|---|---|
| 1 (the 17th, first) | gemini-3.6-flash | 6ad5b2f5db93 | 17 Sep, 09:04 to 09:05 | three searches finished, the fourth refused by Semantic Scholar, the next move refused by the model service (503) | none kept |
| 2 (the 17th, second) | gemini-3.6-flash | bf12f33e84e9 | 17 Sep, 09:36 to 09:38 | the plan refused three times (503) | none kept |
| 3 | gemini-3.6-flash | 2cffa3f1e896 | 18 Sep, 09:23 to 09:27 | the plan answered, three searches finished, 150 papers retrieved, then the next move refused (503, asked 3 times) | `live-01-form.png`, `live-02-headline.png`, `live-03-coverage.png` |
| 4 | gemini-3.7-flash | 6a0d86b57d31 | 18 Sep, 09:30 to 09:56 | the plan request never answered; killed by hand after seventeen minutes (R-25) | none kept |
| 5 | gemini-3.7-flash | 9ac54215d523 | 18 Sep, 09:59 to 10:04 | the plan refused (503, asked 3 times) | `live-03-headline-attempt3-37flash.png`, `live-03-coverage-attempt3-37flash.png` |
| 6 | gemini-3.6-flash | 7f08fdc3f05e | 18 Sep, 18:06 to 18:21 | the plan answered on its third ask, two searches finished, 100 papers retrieved, then the next move refused (503, asked 4 times) | `live-02-headline-evening.png`, `live-03-coverage-evening.png` |
| 7 | gemini-3.7-flash | 6ea168ed08d5 | 18 Sep, 18:26 to 18:36 | the plan refused (503, asked 4 times) | `live-02-headline-evening-37flash.png`, `live-03-coverage-evening-37flash.png` |
| 8 | gemini-3.5-flash-lite | 98af89fa5cdc | 18 Sep, 19:01 to 20:04 | went through: two searches, 100 papers retrieved, 50 screened, 50 decisions, 41 sent back and judged again, 9 stored twice | `live-01-form-evening.png`, `live-02-headline-lite.png`, `live-04-paper.png`, `live-05-revised.png`, `live-06-store.png`, `live-07-log.png`, `live-08-store-again.png` |

Whether a file here travels in the public export is decided at the
export: `research_agent/release.py` names every evidence file one at a
time, and a picture only after somebody has read it and recorded its
fingerprint, because nothing in the check can read the text in a picture.
The sixteen pictures were read on 19 September and are named there with
their fingerprints, with this file and the run log's lines.

One more picture sits beside this folder, `../notice-on-the-search-page-2026-09-20.png`:
the NCBI disclaimer and copyright notice on the Search page, taken on 20
September 2026 at 2880 by 1470 pixels, under the form's two numbers and
above the record's line. The sixteen pictures of the run stop above the
notice, which is drawn after the papers on every page, so this one shows
what they do not (OB-03). Read on 20 September and named in
`research_agent/release.py` with its fingerprint like the sixteen.

| Picture | What it shows | When |
|---|---|---|
| `../notice-on-the-search-page-2026-09-20.png` | the heading "NCBI disclaimer and copyright notice", the two paragraphs in NCBI's wording and the link to the disclaimer page, on the Search page below "Start the run" | 20 Sep |

The picture names follow the task sheet's steps
rather than by attempt: `live-01` is the form, `live-02` the headline
after the run, `live-03` the coverage section, `live-04` a paper's card,
`live-05` the cards after the send-back, `live-06` the results page after
"Display Approved", `live-07` the record page, `live-08` the results page
after "Display Approved" a second time. The suffixes say which attempt.

The screen in the pictures is the working tree of the day, not a
reviewed commit: the changes built between the 16th and the 18th were
uncommitted, and the seven findings of the eighth attempt (G23 to G29 in
`docs/testing.md`) were built after these pictures were taken, so the cards, the send-back and the
record's line look different on the screen as it now is.

## `run_98af89fa5cdc.jsonl`

The 192 lines the run log wrote under the eighth attempt's identifier,
copied out of `evidence/run_log.jsonl`, which is not in the repository
because it holds every run of every day: the request with its question
and criteria (1), the request attempts (54: two to PubMed, two to
Semantic Scholar, fifty to Crossref), the searches (2), the reviewer's
decisions (92), the papers judged again (41) and the two stores of nine
papers. Papers are named by identifier, never by title, and the log
holds no paper text, no key and no contact address; the reviewer's notes
are the words typed at the screen.

## `live-04-paper.png`

The card this picture shows is a withdrawn preprint on ivermectin in
COVID-19, which the agent rated 10 of 10 and recommended for inclusion.
The reviewer flagged it and it was not stored. Crossref returned no
retraction or formal concern for it, so the screen says so; the
publisher's withdrawal notice is a later version of the record and is not
the version the extract was taken from. The card is kept as evidence for
that reason: a high rating and an extract found in the paper are not a
judgement of the study, which is what the reviewer's decision is for. The
extract shown belongs to its authors and publisher. `NOTICE.md` sets
out the condition it is kept under: a short extract beside a paper
identifier, so that a claim can be traced to its source. That is what
this card is. It is not the captured paper text withdrawn under R-11,
which was whole replies held for replay.
