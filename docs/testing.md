# Comprehensive evidence of testing

This report brings together the test approach, results and evidence of execution for the implemented system. Automated checks use invented replies. Separate measurements cover external services and observed use of the interface. Design decisions, dated development and remediation are recorded once in [development.md](development.md).

Each result applies to the date recorded with it.

## 1. Approach and evidence boundary

| Evidence | What it establishes | Limit |
|---|---|---|
| Unit tests in `tests/unit/` | Individual rules: validation, parsing, budgets, deduplication, storage and release checks | Replies and failure conditions are supplied by tests |
| Integration tests in `tests/integration/` | Agents, graph, handover, status checks and revision work together | Model and service behaviour is replaced |
| Functional tests in `tests/functional/` | The real screen and processing chain produce the expected visible and stored results | Invented replies prepared for the project. Not evidence of live model quality |
| Live service runs and dataset measurements | Services and models answered particular requests, with measured outcomes | Dated, quota dependent and limited to those runs |
| Human readings and browser sessions | Observed fidelity, comprehension and interaction on the tested material | The readings were not made independently of the project. Interface evaluation involved one participant familiar with the project. This was not a representative usability study |
| Export and fresh clone verification | The published package holds only what the release rules permit, is complete and can be reproduced from the inventory | Establishes what the exported files are, not the licensing of what they describe |

Tests redirect recording, buffers and storage into temporary locations. They do not call paid or live services. The `network` and `llm` markers are reserved. No tests currently use them. The contamination test finds known test markers. It cannot prove that a recording came from a service. Manifest integrity, replay scope and publication checks address different questions.

Run the automated checks from the repository root with the environment installed as the README describes:

```text
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q -m rationale
.\.venv\Scripts\python.exe -m pip check
```

Record Python and dependency versions, the reviewed commit and any working tree changes with the output. A command that cannot start remains unverified.

Evidence labelled **submitted** below is intended to travel in the release after its release review. **Private** evidence contains paper text or detailed captures and must not be copied into the public package. Private files support the development audit. The submitted report must contain the result, method and limitation needed to understand each claim without obtaining those files.

## 2. What has been run

| Check | Recorded outcome | Evidence and qualification |
|---|---|---|
| Automated baseline, 20 September | 1,188 passed, 3 skipped, 1 deselected. Ruff clean. Rationale 1 passed | The figures at the technical freeze, kept here as the dated record. The saved output that held them was replaced on 24 September, so no file carries this run any more. The check on the generative-AI acknowledgement, added on 21 September, makes 1,189 when counted against this baseline; no run reported that total |
| Automated record, 1 October | 1,212 passed, 3 skipped, 1 deselected. Ruff clean. Rationale 1 passed, 1,215 deselected | `evidence/testing/final_test_output.txt`, the run made on the tree this commit carries. It was rerun on 1 October, when two test files changed, and replaced the freeze baseline on 24 September, because a record must belong to the code it travels with |
| Post baseline corrections | R-28 to R-33 added checks. The identifier codes taken out of the code's comments on 23 September changed comments only and added no test | Dated development records and targeted results. The whole suite ran from the repository root on 23 September, after all of that comment work: 1,208 passed, 3 skipped, 1 deselected. It ran again the same evening, after the release check was made to require the setup files and the record of execution: 1,211 passed, 3 skipped, 1 deselected |
| Functional checks, 14 to 15 September | 10 passed: nine scenarios and the fixture check | `tests/functional/test_functional.py`. Run again with demonstration mode forced both ways to confirm the fixtures choose their own mode |
| Fresh installation, 9 September | README followed from a fresh clone. 954 tests passed at that date | R-15 records failure in a deep Windows path and success in a shorter one. Not a final release installation check |
| Application startup, 16 September | Page served successfully | Checked from the development checkout. A complete live run from a fresh clone was not performed |
| Live workflow, 18 September | Form to retrieval, screening, reviewer revision and storage completed | Run `98af89fa5cdc`, described in section 3 |
| Public export, 1 October | Running the documented test command inside the export on Windows gives 1,206 passed, 9 skipped and 1 deselected. The skips are the three symbolic-link checks and six checks that query git about the repository, which is not included in an export | The export carries the invented demonstration fixtures. R-12 |

The three skipped automated checks make a symbolic link, which Windows does not allow without a privilege the run did not hold. They are recorded as skipped, not as passed.

## 3. Evidence of execution

### The real workflow

On 18 September, run `98af89fa5cdc` used Gemini 3.5 Flash-Lite after seven earlier attempts on the design's models failed to complete: six service refusals and one prolonged wait. R-25/R-26 explain the model change and timeout handling. The successful run used these changes to the original design.

| Operation | Observed outcome |
|---|---|
| Planning and retrieval | Two searches returned 100 papers |
| Screening | The retained allowance sent 50 papers to curation |
| Human decisions and revision | 50 initial decisions. 41 papers sent back with notes and judged again. Four agent decisions changed, all from include to exclude and none the other way. Counted from the run log of that run, `evidence/execution/live-run-2026-09-18/run_98af89fa5cdc.jsonl`, which holds 41 "judged again" lines, four of them with a different answer |
| Storage | Nine approved papers stored. Repeated storage left nine records |
| Request accounting | 13 model requests, including the revision work, counted against the run allowance |
| Evidence retained | 192 run log lines and 16 screenshots |

The submitted execution record is [the live run README](../evidence/execution/live-run-2026-09-18/README.md), with its reviewed screenshots and run specific log. SC-04 reconciles the same nine identifiers in the approval record and store. This does not establish equality of every stored field. Unit and functional tests cover those field level rules.

Earlier submitted checks show Crossref answering on 26 August, PubMed's hundred record batch on 28 August and Semantic Scholar answering with its key on 28 August: `evidence/execution/crossref_live_check_2026-08-26.txt`, `pubmed_batch_size_check_2026-08-28.txt` and `semantic_scholar_live_check_2026-08-28.txt`.

These records show execution on the stated dates. Future availability is not guaranteed. Date filtering has request construction and returned year tests. No dedicated live check of both services' date parameters is recorded. Live planning observations include continue and refine. They do not exercise every graph branch. The four way routing and stop conditions are covered automatically.

### Public demonstration

The public demonstration uses invented papers and answers prepared for the project. They are labelled as invented. The demonstration shows the implemented workflow without redistributing real abstracts. It does not require a live service to answer. It does not supply live retrieval, recall or model fidelity evidence. Private captured replies remain separate and are refused at the public export boundary.

## 4. Automated coverage

Identifiers are retained so code comments and earlier evidence remain traceable. "Covered" means there are tests of the stated rule.

### Unit checks

Test filenames in this table are under `tests/unit/`.

| IDs | Rule and reason | Test files |
|---|---|---|
| UT-01 | Settings, paths and keyed/unkeyed NCBI rates | `test_config.py` |
| UT-02, UT-22 | Retries and spacing, failures returned, attempts logged without request credentials. NCBI hundred attempt limit counts failed attempts and excludes replay | `test_http.py` |
| UT-03 | Capture/approval/replay integrity. Publication basis and private replay are separate. Damaged manifests, changed replies and incomplete promotion refuse safely | `test_saved_replies.py`, `test_saved_replies_are_not_from_tests.py`, `test_publication_rights.py`, `test_approve_replies_command.py` |
| UT-04 to UT-07, UT-23 | Goal to ordered searches. Source selection. Continue/refine/drop/finish. Bounded repairs. Failed and partial searches. Short decision reasons. Private model reasoning is not requested | `test_planner.py` |
| UT-08, UT-09 | PubMed structured/nested abstracts and Semantic Scholar fields, key header and shared limiter | `test_pubmed.py`, `test_semantic_scholar.py` |
| UT-10 | Withdrawn with the unused PMC source on 19 September. Not evidence of implemented full text retrieval | No current test file |
| UT-11 | Crossref status, DOI normalisation, one lookup per distinct DOI, empty batch and failures | `test_crossref.py` |
| UT-12 to UT-16, UT-26 to UT-33 | DOI/PMID/title/fuzzy matching, conflicting identifiers, safe group merges, stable results, retained origins, text/licence pairing and strongest status | `test_dedup.py` |
| UT-17, UT-18 | Criteria based batch screening. One judgement per paper. Cited summary and extract. Prompt markers, bounded paper text and no full text in the request | `test_curator.py` |
| UT-19 | Only approved, identified records with checked extracts stored. Whole batch refusal/rollback, repeated storage, withdrawal and reconciliation. No abstract/full text column | `test_storage.py` |
| UT-20 | UTC queries, filters, attempts and reviewer decisions linked by run. Concurrent writes, unreadable lines and log write failure reported | `test_run_log.py` |
| UT-21, UT-STRICT-01 | JSON handover and record invariants. Wrong types and extra fields refused instead of silently converted | `test_schemas.py` |
| UT-24 | Both services receive date filters. Out of range years removed. Unusable years retained and counted. Request years bounded | `test_schemas.py`, both source tests, `test_planner.py` |
| UT-25 | Count before every model invocation, including failures, repairs, revision and replay. Exhausted run refuses. Bounded busy service retry and progress notices | `test_schemas.py`, `test_llm.py`, `test_planner.py`, integration handover tests |
| UT-34 | Retained paper allowance shared across searches. Duplicates use one place. Unused papers remain reportable | `test_selection.py` |
| UT-35 | Stable relevance ranking with documented tie breakers | `test_ranking.py` |
| UT-36 | Exact extract checking with controlled whitespace/quotation normalisation. Changed words/numbers and cross field quotations refused | `test_validation.py` |
| UT-37 | Failed, partial, dropped and unreached searches kept distinct. Counts, status failures and write failures reach coverage without paper text | `test_audit_log.py` |
| UT-38 | Form, decisions, notes, reruns, storage refusal, revision, run log and visible failure states | `test_app.py` |
| UT-39 | Revision includes the earlier judgement and reviewer note, with the same batch and marker checks as first screening | `test_curator.py` |
| UT-40 | Extract check diagnostic handles word and number boundaries. No population error rate inferred from synthetic text | `test_extract_check_measurement.py` |
| UT-41 | Run identity survives model recording, approval rebuilding and replay. Mixed run captures detected | `test_saved_replies.py`, `test_llm.py`, `test_run_log.py`, replay set checks |

The timeout in UT-25 applies to each connection operation. It does not impose a two minute deadline on the complete model request. Run accounting does not enforce the account wide daily or token allowance.

Additional tests cover screening label fingerprints, sampler fail closed behaviour, approved fixture origin, release allowlists, inventory verification, credential patterns, Git line endings, theme contrast and result reconstruction. Their remediation links are in [development.md](development.md#remediation-activities). Hostile tests cover approval bypasses and injected prompt markers. They do not prove immunity to semantic prompt injection or every secret format.

### Integration checks

| IDs | Interaction tested | Files under `tests/integration/` |
|---|---|---|
| IT-01, IT-02 | Loop, routing, search refinement, bounded stopping and typed LangGraph compatibility | `test_buffer_handover.py`, `test_langgraph_spike.py` |
| IT-03 | Real `ResearchState` revalidated at handover and completion. Deliberately corrupted state refused at the intended boundary | Both files above |
| IT-04, IT-05 | Deduplication/selection at handover. Real planner and curator connected through the graph, with model and sources replaced | `test_buffer_handover.py`, `test_planner_in_the_graph.py` |
| IT-06, IT-07 | Ranked judgements and extract found status reach the completed run | `test_buffer_handover.py` |
| IT-08 | Flagged/rejected papers revised with notes, fresh extract check, cleared human decision and shared allowance | `test_buffer_handover.py` |
| IT-09 | Crossref called once per distinct selected DOI. Answer shared without merging. Failures carried into coverage and log | `test_buffer_handover.py` |

### Functional checks: expected behaviour and observed result

All nine use synthetic replies through the real screen, graph, agents, PubMed adapter, log and temporary SQLite store. A replacement for the model supplies answers and selected failures. This does not verify schema repair by a live model or enforcement of the budget during live requests.

| ID | Scenario | Observed result |
|---|---|---|
| FT-01 | Goal and criteria. Second search repeatedly returns 503 | Three papers ranked 9, 6, 2, with citations, summaries and found extracts. Finished/failed outcomes and incomplete coverage warning |
| FT-02 | Empty search followed by refine | Refined query returns three papers. Moves refine then finish. Completed headline |
| FT-03 | Fetch returns two of three requested papers | Partial outcome names the missing identifier. Two papers screened. No complete coverage claim |
| FT-04 | Source requests time out | No crash. Failed search and no paper warning. Nothing screened |
| FT-05 | Stand in supplies schema failure for initial plan and three repairs | Four requests counted, no source request, clear need for human attention |
| FT-06 | Approve, reject with note, flag without note | Invalid flag refused beside the paper. Only the approved record stored |
| FT-07 | Approve and store one paper | Identifier, title, extract and decision match the expected database row |
| FT-08 | Approve two, reject one | Store and reviewer log contain the same two approved identifiers |
| FT-09 | Send a flagged paper back, then store before/after a new decision | The revised judgement keeps the note. A new decision is required. Only then can it join the stored set |

## 5. Retrieval, precision, extraction and consistency

### SC-01: the hand screened development benchmark

The two committed label files contain 100 distinct records, with three included papers in the first fifty and five in the second. Seven records in the first and six in the second had no abstract. Their labels used title, publication type and available full text. This is a documented departure from the proposal's "50 abstract sample".

Run `4c0d12e8a3ef` on 19 September retrieved all three included papers in the first sample and all eight across both: recall 1.00. It retrieved 84 of the 100 records overall. The result used **1,000 requested results per search**. The default is fifty. Semantic Scholar allows at most a hundred results in one search request, and this implementation makes one request without paging, so a hundred is as far as one of its searches goes. PubMed returned 675 of 676 requested records. Those shortfalls were reported. The earlier live run requested fifty results per search and retrieved none of the eight positives. Recall of 1.00 does not describe performance at the default setting.

This is a known topic development benchmark: the demonstration question is related to the screening topic and both sets of labels have been seen. The second set tests sensitivity to the sampled records. It is not a blind test of generalisation. Recall is relevant retrieved / relevant labelled. It does not measure screening precision or guarantee systematic review completeness.

The labels, sample method and pinned canonical fingerprints are in `data/hand_screened_sample/` and `tests/unit/test_screening_labels.py`. The matching retrieval buffer and run log entries are private. The recorded result and limitations above are the submitted evidence. Do not export their abstracts.

### TF-03: the standard dataset

Three collections were read from their own pages and counted on 14 September before one was chosen. The tutor's feedback asked for a suitable dataset, so why this one suits a PubMed-based tool screening about fifty abstracts a run is part of the evidence.

| Candidate | Where it is, and under what licence | How it fits a PubMed-based tool screening about fifty abstracts a run |
|---|---|---|
| The CLEF eHealth technology-assisted review collections, 2017 to 2019 (Kanoulas et al., 2019; CLEF-TAR, 2019) | `github.com/CLEF-TAR/tar`, no registration, MIT licence. For each topic, a Cochrane review's identifier and title, its Boolean MEDLINE query, the PubMed identifiers the query returned, and relevance judgements at abstract and content level. The 2019 Task 2 intervention test topics hold 61 to 9,160 identifiers each, 4 to 320 judged relevant at abstract level. Task 1 adds each review's eligibility protocol | Fits best. The identifiers are PubMed's, so the fetch retrieves the records the way a run does. The abstract-level judgement is the decision the Curation Agent makes. The review title is a research question and its protocol gives the criteria. The licence is explicit |
| The Cohen et al. (2006) drug-class review collection (Cohen, no date) | `dmice.ohsu.edu/cohenaa/systematic-drug-class-review-data.html`, one tab-separated file, no licence stated. 18,733 rows for 15 drug-class reviews with PubMed identifiers and abstract- and article-level triage. 2,399 rows included at abstract level | Fits in part: no licence, the questions and criteria would have to be reconstructed, and the records are two decades old |
| SYNERGY (De Bruin et al., 2023) | `doi.org/10.34894/HE6NAQ` at DataverseNL, CC0 1.0, built with the `synergy-dataset` package from OpenAlex. 26 systematic reviews, 169,288 records, 2,834 included after full-text screening. A PubMed identifier where OpenAlex had one, from nearly all to almost none of a review's records | Fits least: the labels are full-text inclusion decisions, which a screener reading the abstract cannot be expected to reproduce, and a built copy holds paper text |

Nothing from any of the three was saved into the repository.

CLEF 2019 Task 2, review CD011571, supplies a separate labelled fifty record sample containing fifteen relevant records. The same question, labels and model settings are held fixed where a comparison requires them. The private measurement files retain the attempts, including failures.

| Check | Result | Interpretation |
|---|---|---|
| DS-01 retrieval, 15 September, `bb5ab682ef9c` | 10/15 relevant retrieved = 0.67. Partial | Below 0.80 on this dataset. PubMed missed an identifier and Semantic Scholar was capped. Do not merge this with SC-01's different benchmark |
| DS-02 screening, Gemini 3.6 Flash, 15 September, `518c474c7098` | Four include decisions, all correct: precision 1.00. Relevant found 4/15 = 0.27. Eleven uncertain | Precision alone hides missed positives: seven positives were uncertain and four excluded |
| DS-02 screening, Gemini 3.5 Flash-Lite, 18 September, `0daad4c84a13` | Six includes, all correct: precision 1.00. Relevant found 6/15 = 0.40. Five uncertain | Positive coverage was higher in this run. This does not establish general model superiority |
| DS-03 extract checking | Flash: 43/50 found. Lite: 41/50 found | Flash's seven failures had no abstract. Lite had six no abstract failures and three failures with an abstract |
| DS-03 human reading | Ten judgements from each model read. 10/10 faithful each time | Small readings, and not independent ones. No independent fidelity estimate follows from them. The later sample included a record with only a title |
| DS-04 repeatability, Lite, `807c3c049453` against `0daad4c84a13` | 49/50 decisions unchanged. 14/50 scores changed by 1 to 3 points. Six include decisions unchanged | One repeat at fixed settings. Five earlier attempts on 16 September failed with 503 and yield no consistency figure |

Two boundaries sit behind the screening figures. Precision of 1.00 records agreement
with the collection's abstract-level labels, and the request carries both the inclusion
and the exclusion criteria, so what is measured is agreement with those criteria as
written. The relevance score is separate and uncalibrated: a whole number from 0 to 10
is asked for, no value is given a meaning, and nothing here or elsewhere calibrates
one. It orders papers, and the review list's opening threshold uses that order. Neither
the score nor the threshold is a measurement.

The measurements retain individual attempts, audit samples and failures as development evidence. The public dataset documentation records provenance and sample selection without redistributing abstracts.

### Other success criteria and audits

| ID | Method and result | Limit |
|---|---|---|
| AUD-01 / SC-03 | Twenty initial judgements sampled from the 18 September live run, seed 20260918. No unsupported summary claim or extract mismatch found | The reading was not made independently of the project. Seventeen summaries repeat the extract exactly, so only three of the twenty tested whether a summary stays inside its paper. Revised judgements were not included in the sample |
| AUD-02 / FR-20 | Sixteen captured rationales from seven runs on 15 to 19 September; one of the two refine moves and the finish move came from the retrieval runs of 15 September: one sentence each, 123 to 195 characters. 13 continue, 2 refine, 1 finish | No observed drop. Captures are not a complete account of failed requests. The 300 character schema bound does not enforce one sentence |
| AUD-03 / SC-02 | Human classification of rejected extracts: 0 formatting refusals in 141 checks. 3 in 241 across all inspected runs | 1.2% describes these observations only. Three honest quotations were refused over dropped characters. A changed word extract was correctly refused |
| SC-04 | Nine approved and the same nine stored, with repeated store remaining nine | Identifier reconciliation, supplemented by storage tests. Not a general data integrity guarantee |

Architecture reviews AR-01/AR-02, service review AR-03 and data boundary review SEC-01 informed [the design and implementation decisions](development.md#design-and-implementation-decisions). They document the separation of deterministic processing and model work. They do not establish a classical architecture classification. Residual service limits, free text reviewer notes and publication boundaries remain explicit there.

## 6. The release checks

| Check | Outcome |
|---|---|
| Whole suite and dependencies | The suite, Ruff and the rationale check were run from the repository root on the released code, and their output saved as `evidence/testing/final_test_output.txt` |
| FIG-01 class diagram | Fields, types, optionality and the storage relationships read against the released code, and the figure rendered and read at its published size |
| FIG-02 sequence diagram | Calls, the human gate, revision and the five error paths read against the released code, and the figure rendered |
| Release safety and completeness, R-12 | The export was built from the release commit using the release command and checked against its inventory: 133 files, every path present and every file's bytes unchanged since the export was built. No private capture is included, and no credential pattern was found. Two screenshots from the 18 September run show a paper card with its quoted extract. That extract is third-party material outside the MIT licence, kept under the condition `NOTICE.md` sets out, a short extract beside a paper identifier so that a claim can be traced to its source. This is a project rule, not permission from the authors or publisher, and the extract has no established legal clearance. The evidence README explains why the card is retained. Comparison with a fresh repository clone on 1 October found one difference among the exported files: the recorded-replies manifest lists 20 replies in the repository and 5 in the export |
| Installation and demonstration | The README was followed from a fresh clone on 1 October. The virtual environment was created with Python 3.13.14. Installing the dependencies from `requirements-dev.txt` failed at a deep path because one path inside a packaged dependency exceeded the Windows path-length limit. Repeating the installation from a shorter path succeeded, as recorded in R-15. The offline demonstration was then run from the exported package using the question and settings named in the README, in demonstration mode with no key and no `.env`. It replayed the five fixtures included in the package, screened the three invented papers through the real application interface using Streamlit's AppTest rather than a browser, without calling the model, and kept back the paper rated 5 or below |
| Presentation and transcript | Their figures and limits read against this report and the development account |

"Independent" here means a separate technical confirmation; only one human participant was available. Each check names where its evidence is.

## 7. Interface and user experience

### 7.1 Basis and scope

The interface criteria cover reading at desktop and phone widths, light/dark appearance, contrast, keyboard access, comprehensible decisions and recovery from empty/partial/failed runs. Established accessibility and design system sources listed in the README take precedence over an AI generated style checklist. A minimalist appearance is a preference. Task completion and readable states are the acceptance evidence.

Two sessions involved the same participant, who was already familiar with the project. This is formative evaluation with **n=1**. It is not a representative usability study. The five point ease rating is not SUS. The proposed SUS study and threshold were withdrawn. No SUS score is claimed. The threshold had been set against an average of 68 across many systems (Sauro, 2011); Yu, Liu and Sharmin (2022) report 63.85 for Rayyan's web version and 56.73 for its mobile one, which is the closer comparison and shows how little a single score from one participant would have said.

The existing interface identifiers resolve as follows. These are targets and bounded observations. They do not establish full accessibility compliance.

| Requirement / check | Acceptance focus and evidence |
|---|---|
| UX-01 / UX-CHK-01 | WCAG 2.2 AA target (World Wide Web Consortium, 2024): contrast, labels, focus, target size and reflow. Measured cases and later theme calculations below. The sitting of 9 September on the form is pictured in `evidence/ux-01-form-focus.png` and `evidence/ux-01-form-focus-overlap.png`, both taken before any change was made for them: the first shows the focus indicator, the second the focused field under the top bar. Final applicable criteria assessment is not complete |
| UX-02 / UX-CHK-02 | Human decision, undo, note/refusal feedback and extract beside the choice. Two sessions and app/storage tests |
| UX-03 / UX-CHK-03 | Keyboard access and finding unfinished work. 19 September desktop walk and task times. Baseline keyboard walk not run |
| UX-04 / UX-CHK-04 | Five point ease rating from the participant. Original SUS commitment not met as written. Scale withdrawn, no score or 68 point comparison |
| UX-05 / UX-CHK-05 | Content led layout, hierarchy, light/dark and responsive reading. Requested changes and session scope below |
| UX-06 / UX-CHK-06 | Clear partial/empty/failed coverage at decision time. Prepared states and subsequent app tests. Final revised wording not read cold again |

### 7.2 Sessions and measured outcomes

| | 16 September baseline | 19 September follow up |
|---|---|---|
| Environment | Chrome and Firefox, Windows 11. Five widths, two zoom levels. Real phone also checked | Chrome and Firefox. Widths 1440, 768 and 500. Two zoom levels. No phone |
| Activities | Six tasks and three deliberately prepared run states | Repeated tasks and selected changed state checks |
| Ease rating, 1 easy to 5 difficult | 3 | 2 |
| Extract and identifier | 5 s | 7.63 s |
| Approve then reject | 10 s | 6.90 s |
| Flag twice | 13 s | 7.53 s |
| Attempt approval of unverified extracts | 24 s | 10.96 s |
| Find a paper / distinguish decided from waiting | 10 s and a failed distinction | 9.65 s |

Familiarity, the changed interface and the small sample prevent attributing the time differences solely to the design. The follow up did not repeat every original viewport, phone orientation or phone keyboard interaction. The cold state wording changed again after feedback. Passing automated wording tests is not another cold human reading.

The session sheets and screenshots are retained outside the repository as `ux-session-observation-sheet.md` and `ux-session-2-observation-sheet-19-september.md` with their image folders. Their principal observations, resulting changes and limitations are recorded here. The private sheets are not an extra appendix needed to understand the result.

### 7.3 Contrast and visual checks

Body text measured 12.53:1 in light mode and 18.11:1 in dark mode. Warning text measured 4.74:1 light on the first day and 4.65:1 on the second. The 10.31:1 dark warning value was calculated from the theme. It was not measured in the browser.

The later configuration sets the input border to `#808495` in both appearances. The theme tests calculate the ratio from the settings and assert that a control's boundary clears 3 to 1; they hold neither figure. The figures recorded for the two outlines, 3.72 to 1 in the light appearance and 5.09 to 1 in the dark, are in the settings file's own comments beside the colours they describe, and the dark one is in the testing record of the second session. Earlier notes saying the light outline was left unchanged describe the older screen. They do not describe that configuration. Neither the calculation nor the comments is a new browser measurement.

The requested violet panel, amber flagged state, labelled cards, progress navigation, shorter panel guidance and explicit storage and repair feedback were implemented with tests. Text labels retain meaning when colour is unavailable.

### 7.4 The register of changes and what it shows

The following register preserves all 59 agreed changes and their evidence, including superseded alternatives. This is detailed implementation evidence. It does not establish that every change was tested separately with a participant. Post session faults R-28 to R-33 and their regression checks are summarised once in development.md.

### 7.5 The changes asked for at the screen, 17 to 19 September 2026

The changed screen was read again at the screen through 17 and 18
September with the invented papers, then on the live run of the 18th,
and the second session of the 19th closed the list. Each change is one
piece with its test. The tests named are in `tests/unit/test_app.py`
unless another file is given. Sizes, alternatives and the measurements
behind the colours are not in the table.

| # | What was asked for | Change | Status |
|---|---|---|---|
| 1 | The coverage box as two headed groups, "What the run did" and "What needs attention", side by side when wide and stacked when narrow, "What to do next" under both | `what_the_run_did` and `what_needs_attention` in `audit_log.py`. `_draw_the_coverage` draws them in two columns, which stack below Streamlit's 640-pixel breakpoint, where a table would not reflow | done 17 September. `test_the_coverage_is_two_headed_groups_side_by_side`, `test_the_two_groups_are_the_one_column_form_taken_apart`, `test_the_failure_lists_never_read_is_a_shortfall_not_a_count` |
| 2 | The help line on the three answers once, in the panel above the buttons | `WHAT_THE_ANSWERS_DO` drawn as a caption in the side panel and nowhere on the page | done 17 September. `test_the_screen_says_what_the_three_answers_do_once_in_the_panel` |
| 3 | After storing, say where the papers went, that nothing is sent to anyone, and what to do next | done as a sentence under the store button, then superseded the same day by the results page, which says it in its own words (R2, F6). Neither the sentence nor its test is in the code now | superseded 17 September. The results page's own tests carry it, `test_storing_moves_the_screen_to_the_outcome_as_the_mock_up_has_it` |
| 4 | The identifier as a link: DOI to doi.org, PubMed to the PubMed page, none otherwise | `citation` writes the DOI and the PubMed identifier as links, percent-encoded in the address. The link is followed by the browser, not by the application | done 17 September. `test_the_doi_and_the_pubmed_identifier_are_links_and_nothing_else_is` |
| 5 | The panel's counts per decision | `how_many_decided`: "12 of 50 papers decided, 38 waiting: 8 approved, 3 flagged, 1 rejected." | done 17 September. `test_the_count_says_how_the_decided_papers_split` |
| 6 | The current decision as a coloured word on each paper | `the_answer_in_its_colour`: Streamlit's coloured text, the word carrying the meaning for a reader who cannot tell the colours apart | done 17 September. `test_the_answer_is_a_coloured_word_and_the_word_is_still_there` |
| R1 | The outcome read from the records, not the session, so any stored run can be opened later | `processing/results.py`: `outcome_of(run)` joins the log's last decision on each paper to the store's rows by normalised DOI, PubMed identifier or Semantic Scholar identifier. `runs_with_approvals` lists the runs with an outcome, newest first | done 17 September. Nine tests in `tests/unit/test_results.py`, thirteen since R-30 and R-31 on 22 September |
| R2 | After a store, the page moves to a results screen: the recap, every approved paper, where the set lives, "Back to the paper selection" and "Start a new search" | `_draw_the_results` and `_draw_the_outcome`. A page flag in the session is set by the store button and by the list of earlier runs on the form's screen | done 17 September. `test_storing_moves_the_screen_to_the_outcome_as_the_mock_up_has_it`, `test_back_returns_to_the_papers_with_the_decisions_as_they_were`, `test_a_new_search_clears_the_run_and_draws_the_empty_form`, `test_an_earlier_run_can_be_opened_from_the_form_screen` and three more |
| R3 | The plan and the README to say this screen is the outcome | the plan, the README's Running section, the FR-14 and FR-16 rows, the development log | done 17 September |
| T1 | The help line as three short headed entries plus the Send back line | `WHAT_THE_ANSWERS_DO` as four headed captions, carrying two facts the draft left out: a flag or a rejection without a note is refused on the spot, and the second judgement comes only when Send back is pressed | done 17 September. Superseded by F1 and G3 |
| T2 | Amber for a flagged paper's coloured word, not orange | ":yellow[flagged]", Streamlit's amber: 4.81 to 1 on the light page against the orange's 3.42, under the 4.5 to 1 UX-01 asks of text | done 17 September. The coloured-word test reads the new value |
| T3 | The undecided sentence beside Store: "X papers currently have no decision. Proceeding now will exclude them from your approved list for consultation. Consider reviewing them first." | `what_storing_now_would_do`, with a singular form. "Every paper has a decision." for none | done 17 September. `test_the_store_button_says_what_storing_now_leaves_out` |
| T4 | Rename "Store the approved papers" to "Display Approved" | the label only. The key stays "store" and the press still stores first, which the results page's first line says. The README, the sequence diagram and the sheets follow | done 17 September. `test_the_store_button_is_labelled_for_what_the_reviewer_sees_next` |
| T5 | Each paper as a block of labelled lines: Author, Journal, DOI as the link, Publication status, Relevance, Reason, Summary, Extract as a quotation | `_judgement_lines` returns the title and nine labelled lines in that order. `named_identifier` gives the kind and the linked text to both pages | done 17 September. `test_each_paper_is_a_block_of_labelled_lines_in_the_owners_order` |
| T6 | The extract last, with the summary. The reason always drawn | the block's last line is the extract (SC-03). The launcher's stand-in keeps the old reason in front of the send-back note | done 17 September, with T5 |
| T7 | The results screen | R1 to R3 | done 17 September |
| F1 | The four help entries pushed the panel's buttons below the fold of a 1179 by 731 window | four short sentences, then three (G3) | done 17 September. The fit in a browser is a measurement, made on the 18th (G17) |
| F2 | The results page's two buttons in the left panel, always in view | drawn in the panel under "What next" | done 17 September. The storing test reads the two keys in the panel |
| F3 | Fault: after Back, a paper approved and stored again was listed as "no longer in the store" | R-23: the launcher redirected the write and not the name the results page read. Every function of `storage.py` now looks the store up at the moment of use | done 17 September. `test_the_results_page_reads_the_store_the_launcher_writes_to`, `test_the_read_back_follows_the_store_the_write_went_to` |
| F4 | The run record as a page of its own. The Deploy control hidden without losing the Settings menu | the "Run record" page with the selector, the lines, the two files' paths and a Back. `client.toolbarMode = "viewer"` hides Deploy, Rerun and Clear cache and keeps print, screencast and the theme toggle | done 17 September. `test_the_run_record_is_a_page_of_its_own_off_the_review_page`, `test_the_toolbar_hides_the_deploy_control_and_keeps_the_appearance_menu` |
| F5 | "Start a new search" lands at the bottom of the form page | Streamlit offers no way to move the page from Python, so the rerun embeds one one-pixel frame carrying a line of script that scrolls the main section to the top, once | done 17 September. `test_a_new_search_moves_the_page_to_the_top_once` |
| F6 | The results page as a recap for a person: no file path in the green line | "N papers approved and saved." The paths in a caption inside the record of runs | done 17 September. Reshaped by F7 |
| F7 | Cards for the review and results pages, to a mock-up | the results page: a header counting the approved papers, a bordered box with the saving sentence and a CSV download, a toggle unfolding the flagged and rejected papers, one bordered card per approved paper with a link button to doi.org or PubMed. The review page's cards in the same shape, at first two per row | done 17 September. `test_storing_moves_the_screen_to_the_outcome_as_the_mock_up_has_it`, `test_the_flagged_and_rejected_papers_unfold_under_the_approved_ones`, `test_the_list_downloads_as_one_csv_with_the_link`, `test_the_finish_time_is_the_clock_on_the_wall` |
| F8 | No model request, no service call, each change with a test | kept | |
| G1 | Fault: after Start the run, approve one paper, "Display Approved", the page did not move | not reproduced by AppTest. What the report matches is a fault raised under the button, out of sight below the fold, so every fault of the store is now carried to the top of the page in red and written to the log | done 17 September. `test_storing_asks_storage_and_reports_what_it_refused`, `test_any_fault_while_storing_is_said_at_the_top_and_logged` |
| G2 | The "Run record" button out of every panel. The record reached from one line at the foot of the results page and of the form | `_offer_the_record` draws one tertiary button, "Record of this run, for checking what was done", last on those pages | done 17 September. The storing and earlier-run tests read the line last |
| G3 | The panel help text as three sentences exactly, no Send back line, British spelling | `WHAT_THE_ANSWERS_DO` is the three sentences | done 17 September. The panel test reads the three |
| G4 | After every piece, the whole flow on the invented papers as a test that stays | `test_the_whole_flow_on_the_invented_papers`: form, run, approve, Display Approved, Back, more approvals, Display Approved, new search, run, approve, Display Approved, each page read by its heading | done 17 September |
| G5 | One card per row on the review page, not two | the grid gone. Each paper one bordered container after the other | done 17 September. `test_each_paper_is_a_bordered_card_one_per_row` |
| G6 | The results page's box pale green like the mock-up's | the box keyed `saved-papers`. A second style rule tints it with Streamlit's green at 12 per cent. Body text 11.3 to 1 on the light tint and 15.3 on the dark | done 17 September. The storing test reads the keyed box and the rule |
| G7 | Fault: after Back, Display Approved refused the whole batch because one paper's extract was never looked for | R-24: `decide` refuses an approval of a paper whose extract was never looked for or was not found, at the paper, with what to do instead. Flag and reject stay open | done 17 September. `test_approving_a_paper_whose_extract_is_not_traceable_is_refused_at_the_paper`, `test_the_refused_approval_is_said_at_the_paper_and_nothing_is_recorded` |
| G8 | "Display Approved" in a different colour from the panel's other buttons | Streamlit's primary type for it alone. White on the default red accent measured 3.30 to 1, under 4.5 | done 17 September. Undone by G9 |
| G9 | Not red: white with a thick violet border and violet lettering, the send-back button the same | both plain again. One style rule on their keys. The accent set to the violet | done 17 September. `test_the_panels_two_actions_are_white_with_violet_lettering` |
| G10 | The panel in a violet with white text at least 4.5 to 1 on both appearances, the help sentences white too | `[theme.sidebar]` in `.streamlit/config.toml`. The panel's captions drawn at full opacity. The panel's boxes given white text on a darker violet | done 17 September. `test_the_panels_text_and_lines_clear_their_ratios`, `test_every_setting_is_one_streamlit_knows` |
| G11 | The panel not collapsible where it sits beside the page | one style rule hiding Streamlit's collapse control from 769 pixels up, the width at which the panel stops overlaying the page | done 17 September. `test_the_panel_cannot_be_folded_away_where_it_sits_beside_the_page` |
| G12 | The run selector's option as the local date and time it started, the question's first three words and the identifier last, in place of the identifier and the UTC stamp | `run_log.heading`, the stamp turned to this computer's clock by `started_on_this_clock` | done 17 September. `test_a_run_is_named_by_its_time_its_question_and_its_identifier` and two more |
| G13 | The panel background #440099 (Pantone Violet C), the same as the accent | `[theme] primaryColor` and `[theme.sidebar] backgroundColor` #440099, white on it 12.31 to 1. `[theme.dark] primaryColor` #ad73e5, 5.72 to 1 on the dark page, where #440099 is 1.54. The panel's boxes #6d3fc0, white on it 6.73 | done 17 September. `test_the_accent_clears_three_to_one_against_the_page_it_is_drawn_on`, `test_the_panels_text_and_lines_clear_their_ratios` |
| G14 | Two thin white lines across the panel, under the counts and under the help | two `st.divider()` calls. `[theme.sidebar] borderColor` white | done 17 September. `test_the_panel_is_divided_under_the_count_and_under_the_help` |
| G15 | The help text one size smaller than the sentences round it | already so: captions at 0.875rem against 1rem | needs no work, 17 September |
| G16 | "Display Approved" with a three-pixel border in #ad73e5 and lettering in #440099. The send-back button the same | the rule. Lettering 12.31 to 1 on the face, the border 3.73 against the panel and 3.3 against the face | done 17 September. `test_the_pages_style_rule_draws_the_buttons_in_the_owners_violets` |
| G17 | The panel too tall for a 1179 by 731 window. Once a paper is decided, say what to press | the panel's elements in one container with the small gap and tighter rules, about 650 pixels against about 800. The prompt "When you are done, press Display Approved below." under the count from the first decision on | done 17 September, seen in view on the 18th. `test_the_panel_is_tight_enough_to_fit_a_short_window`, `test_the_panel_says_what_to_press_last_once_a_paper_is_decided` |
| G18 | After "Start the run" the page lands on the fourth card. Every move between pages should open the next page at its top | the scroll frame drawn on the one drawing after any move, each frame numbered so two moves in a row do not leave the first standing | done 17 September. `test_every_move_between_pages_opens_the_next_page_at_its_top` |
| G19 | The prompt in bold. The help sentences in italics | markdown | done 18 September. The panel tests read both |
| G20 | Hover text on the three answer buttons: the panel's three sentences, one per button | `help=` on each button from `WHAT_EACH_ANSWER_DOES`, the panel's entries made from the same three | done 18 September. `test_each_answer_button_carries_its_sentence_as_hover_text` |
| G21 | Each screen titled Search, Assessment or Result, with a progress line "Search -> Assessment -> Result", the current step bold in the violet and the others clickable, three rules and a confirm on returning to Search | the line first on each page. The rules as warnings: "A search needs to be done first.", "Press Display Approved first.", and the reset on a second press of Search (`_forget_the_run`) | done 18 September. `test_each_screen_is_named_for_its_step_under_the_progress_line`, `test_a_step_past_the_search_is_refused_until_a_search_is_done`, `test_the_result_step_is_refused_until_the_papers_are_stored`, `test_going_back_to_search_asks_once_and_then_resets_the_run` |
| G22 | A run waiting for a busy model service cannot be told from one that will never come back | the run and the send-back drawn under a status box the model interface writes into: "The model is busy: asking again in 2 minutes 30 seconds (2 of 4 asks made)" before every half-minute slice. The Stop control checked to act within a slice | done 18 September. `test_the_run_is_watched_from_a_box_the_model_interface_writes_into`, `test_the_screen_is_told_how_long_each_wait_has_left` in `test_llm.py` |
| G23 | (a) The hover text opens upward over the note box | the three answers drawn above the note box, so the tooltip opens over the paper's own lines. The note read from the session before its box is drawn | done 18 September. `test_the_answers_sit_above_the_note_box_so_the_hover_text_never_covers_it` |
| G24 | (b) Number the cards 1, 2, 3 in the title line | the shown cards counted from 1, so a kept-back paper leaves no gap. The widget keys keep the paper's place in the run | done 18 September. `test_each_paper_is_a_bordered_card_one_per_row` reads the numbers |
| G25 | (c) After the send-back the page does not return to the top | the scroll frame set on that move too | done 18 September. `test_the_send_back_opens_the_page_at_its_top_too` |
| G26 | (d) After the send-back, show what the agent changed | the decision and rating before the hand-back kept by paper and drawn in red under the citation, "Judged again: from: include 8 to: exclude 0", or "Judged again: unchanged" | done 18 September. `test_a_card_judged_again_says_what_the_agent_changed` |
| G27 | (e) Forty-one papers to decide again after the send-back: must every retrieved paper be shown? | nothing in the proposal or the brief requires it. The rule given: the Assessment page shows the papers rated above 5 of 10, never more than fifty, and counts the rest in one line, "N papers rated 5 or below, not shown". A paper answered or sent back stays shown. The kept-back papers stay in the run and the coverage report and are never stored. The form offers the number of results asked of each search, 10 to 1,000, default fifty | done 18 September. `test_papers_rated_five_or_below_are_kept_back_from_the_page_and_counted`, `test_a_run_whose_every_paper_was_rated_five_or_below_says_so`, `test_a_paper_sent_back_stays_shown_whatever_its_new_rating`, `test_the_form_offers_the_number_of_results_asked_of_each_search` |
| G28 | (f) Let the person choose whether the rejected papers go back with the flagged | two buttons naming their counts, the flagged alone first. `sent_back` and `judge_again` in `graph.py` take which answers go | done 18 September. `test_the_send_back_offers_the_flagged_alone_and_with_the_rejected`, `test_the_flagged_papers_can_go_back_on_their_own` in `test_buffer_handover.py` |
| G29 | (g) The record line does not read as a link: blue and underlined on both pages | a style rule on the line's key. Streamlit's own link blue for the appearance in use (blue80 on the light page, 5.49 to 1; blue60 on the dark, 6.58), the light blue when the appearance is not known | done 18 September. `test_the_records_line_is_drawn_as_a_link_on_both_pages` |
| H1 | After the send-back the changed values on the card do not catch the eye. The stand-in's reason ran into the note | the badge and the relevance line in red with the old value in brackets, "unchanged" where nothing changed. The launcher's stand-in separates the reason from the note | done 19 September. `test_a_card_judged_again_says_what_the_agent_changed`, `test_a_second_judgement_that_changes_the_decision_alone_is_marked_too` |
| H2 | The state screens read as written for the developer | the reviewer's words. "Search completed". "No paper found" in the headline's own box. "The run failed before it retrieved anything. Details are under 'Run details'." "a service that failed may answer then". The section "Run details" everywhere. The launcher's invented "needs a person" line gone | done 19 September. `test_a_run_that_found_nothing_says_so_and_offers_no_store`, `test_a_run_that_stopped_before_retrieving_anything_says_so`, `test_a_run_whose_searches_all_failed_says_it_failed_without_asking_for_a_person`, FT-04 |
| H3 | In any state with no papers there is no panel, so the reviewer has no way on | a panel headed "What next" with "Start a new search" in the three no-paper states | done 19 September. `test_a_run_with_nothing_to_review_still_offers_a_new_search` |
| H4 | The bold prompt does not stand out on the violet | Streamlit's info box, given the panel's confirmation colours in `.streamlit/config.toml` (white on #6d3fc0, 6.73 to 1; 1.83 to 1 against the panel) | done 19 September. `test_the_panels_text_and_lines_clear_their_ratios`, `test_the_panel_says_what_to_press_last_once_a_paper_is_decided` |
| H5 | The record line missing from the Assessment screen | one call at the foot of the review page with papers | done 19 September. `test_the_record_is_reached_from_the_foot_of_the_review_page_and_back_returns` |
| H6 | Plainer names for the form's two numbers | "Maximum search attempts" as asked. "Maximum results per search" for the second, since it is the number of results asked of each search and not the papers reviewed | done 19 September. `test_the_form_offers_the_number_of_results_asked_of_each_search` |

One sentence the first session showed to be broken, "the agent would
uncertain it", was fixed on 16 September (R-22) and is not in the lists.


## 8. Human verification and interpretation

On 8 September, a written verification record reported 890 tests passed, 3 skipped and 1 deselected. Ruff was clean. The rationale check passed. Terminal output was not retained. The record predates replacement of that code in history. In the browser, the form worked and the NCBI disclaimer and copyright link were visible. Live goal to result behaviour and live human review were not observed that day. The later 18 September run supplies that evidence. These observations are not a final candidate check.

The evidence supports a working bounded planning/retrieval/screening/review/storage chain and identifies concrete failures and corrections. It does not establish complete literature coverage, clinical suitability, universal accessibility, semantic prompt injection resistance, independent fidelity or legal clearance of arbitrary content.
