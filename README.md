# LLM-Powered Academic Research Planning Agent

Coursework for the University of Essex Online MSc, *Intelligent Agents*
(Unit 11, Development Project), based on the Group B design proposal.

## Purpose and implemented solution

This application helps a researcher search and screen health sciences
literature. Given a question and inclusion/exclusion criteria, its Planning
and Retrieval Agent proposes searches, retrieves papers and decides what to
try next. Its Curation Agent screens batches of papers and supplies a rating,
reason, extractive summary and supporting extract.

PubMed and Semantic Scholar supply papers. Crossref supplies publication
status for papers already found. Ordinary code deduplicates records, checks
extracts, ranks results and stores approved records. The researcher approves,
rejects or flags each candidate. No clinical recommendation is made.

The agents run through a LangGraph loop:

`plan → search → decide/refine → checked JSON hand-over → curate → human review`

By default, a run allows at most 18 model requests and 50 papers for screening. Search
attempts are also bounded. The run records incomplete searches and the reason
for a stop. The handover and final state are revalidated.
Graph updates and changes within a list can bypass schema checks.

The [development account](docs/development.md) explains these choices,
departures from the proposal and remediation. The [class
diagram](docs/class_diagram.md) shows the records and storage relationships. The [sequence diagram](docs/sequence_diagram.md) shows interactions and error
paths. The design is influenced by BDI.
It does not implement a BDI interpreter.

## Install

Use **Python 3.13**, the supported range in `pyproject.toml`
(`>=3.13,<3.14`). Development used Python 3.13.14 on Windows. Run the following
PowerShell commands **from the repository root**, preferably in a short path
such as `C:\projects\research-agent`:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Copy the example only when `.env` does not already exist. Edit `.env` locally. Never commit it. `requirements.txt` pins the application packages. `requirements-dev.txt` includes those packages plus the testing tools. The
latter is used above so the same environment can run the supplied checks.

On the development machine plain `python` selected 3.14, so the command names
3.13 explicitly. A deep Windows folder caused a Streamlit installation error. A shorter path resolved it. The `uuid_utils==0.16.2` pin avoids a Windows Smart
App Control failure observed with 0.17.0. Do not remove these constraints
without checking a clean installation.

### Settings

| Setting | Use |
|---|---|
| `DEMO_MODE` | `true` replays approved replies without service or model calls. `false` runs live. |
| `GEMINI_API_KEY` | Required for live model calls. Not required for offline replay. |
| `GEMINI_MODEL` | Defaults to `gemini-3.6-flash`. The live and standard dataset runs of 18 September used `gemini-3.5-flash-lite`. The model is recorded per run. A different model is not an equivalent evaluation. |
| `NCBI_EMAIL`, `NCBI_TOOL` | Identify the caller for live E-utilities requests. |
| `NCBI_API_KEY` | Optional. The configured rate ceiling is three requests/second without it and ten with it. |
| `SEMANTIC_SCHOLAR_API_KEY` | Use a granted key when available. Without one the shared pool may reject requests. |
| `CROSSREF_EMAIL` | Contact address for Crossref's better served request pool. |
| `RECORD_REPLIES` | Defaults to `true`. Writes live replies and model prompts/answers into the private, git ignored capture folder. Capturing a reply does not approve it for publication. |

## Run

The application runs in two ways. **Live** searches PubMed and Semantic
Scholar and calls the model, so it needs a Gemini API key in `.env` and an
internet connection. The **offline demonstration** replays replies saved in
this package, so it needs no key and no network; it shows the whole workflow
on three invented papers and establishes nothing about live results. The
demonstration is set up at the end of this section.

```powershell
.\.venv\Scripts\python.exe -m streamlit run research_agent/ui/app.py
```

Streamlit serves the application locally, normally at
[localhost:8501](http://localhost:8501).

1. **Search:** enter the question and criteria. Start with the default search
   bounds. The maximum results per search affects retrieval. The separate
   screening cap remains 50.
2. **Assessment:** read the completion or partial coverage notice and **Run
   details**. Cards show the papers the agent rated above 5 out of 10, so the
   screen stays readable and attention goes to the likely papers; the rest are
   counted in the run and in the coverage report, but not shown. A paper you
   have already answered stays on screen whatever its rating. Check the
   citation, publication status, reason, summary and extract before deciding.
3. **Approve, reject or flag:** rejection and flagging require a note.
   Selected papers can be sent back for another judgement. Approval is
   refused if the supporting extract was not found or has not been checked.
4. **Result:** **Display Approved** stores the approved set and shows its
   records and download. Repeating it updates the same set.
   It does not create a duplicate. Earlier approved runs can be opened from Search.

The application shows what could not be completed. A limited or failed run is
not evidence that no relevant literature exists.

### Offline demonstration

Set `DEMO_MODE=true` and `GEMINI_MODEL=gemini-3.6-flash` in `.env`, restart
the application, and use these exact values with the default search bounds.
The model name selects the recorded replies. It does not make a model call:

| Field | Value |
|---|---|
| Question | `Does drug X treat condition Y?` |
| Inclusion criteria | `studies of drug X in condition Y reporting an outcome` |
| Exclusion criteria | `reviews and opinion pieces` |

The public demonstration uses synthetic replies prepared for three
invented papers, identified as invented in their journal names. It exercises
the real workflow through replay, **not live retrieval or real model
performance**. Replay requires the matching fixtures and manifest in the
package. Another question or an unmatched request reports a missing reply. It does not invent one.

`scripts/make_demonstration_fixtures.py` is the development tool that creates
these fixtures. Running the application does not regenerate or approve them.
Recordings from external services are excluded from the export.
The exported package was checked from a fresh clone on 1 October, and [the release checks](docs/testing.md#6-the-release-checks) record what that check found.

For a separate interface only check:

```powershell
.\.venv\Scripts\python.exe scripts\show_synthetic_review_states.py full
```

The script supports `complete`, `partial`, `empty`, `failed` and `full`
states. It uses invented papers and temporary storage. It does not test
retrieval or screening.

### Live operation and local records

For live operation, set `DEMO_MODE=false` and provide the settings above.
Service availability and the account's allowance are external prerequisites.
Requests are rate limited.
The 18 request cap applies to each run. Gemini daily and token limits
are not enforced by this application.

Semantic Scholar requests are kept below one per second. Crossref's reported
limit is followed. The NCBI settings above come from Sayers (2022), the
E-utilities guidance, which asks a caller to identify itself by tool and
address, sets three requests a second without a key and ten with one, and
asks that large jobs run in a named window. The application refuses further
NCBI attempts after its 100 attempt per run threshold. This project defined
threshold avoids relying on NCBI's large job time window, which the code does
not check.

Local runtime files include:

| Location | Contents and boundary |
|---|---|
| `data/approved_records.sqlite3` | Approved records, including extracts and reviewer notes. Not complete abstracts. |
| `evidence/run_log.jsonl` | Run history, reviewer decisions, counts, errors, question and criteria. |
| `evidence/sample_run/` | JSON handover buffers, which can contain retrieved abstracts. |
| `data/captured_replies/` | Captured service bodies and model prompts/answers while recording is on. |
| `data/recorded_replies/` and its manifest | Approved replay material. Private recordings are excluded from the public export. Only permitted demonstration fixtures may be exported. |

Runtime folders must be excluded from the submission.
Git ignore rules do not protect a folder copied outside Git. The release
process uses an explicit allowlist and checks. Those controls cannot establish
the authorship or publication rights of arbitrary prose.

## Testing and evidence of execution

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -m rationale -q
.\.venv\Scripts\python.exe -m ruff check .
```

`pytest -q` runs the whole suite; no test in this package calls a live
service. A separate check confirms that no explanation in the code is still
marked as unwritten.

[Comprehensive evidence of testing](docs/testing.md) contains the test
coverage, dated tool outputs, functional results, dataset measurements,
human sessions and their limitations. Execution captures include the
[live run of 18 September](evidence/execution/live-run-2026-09-18/). The
[saved test output](evidence/testing/final_test_output.txt) is the run made
on the code released here. The installation and the offline demonstration
were checked on 1 October, and [the release
checks](docs/testing.md#6-the-release-checks) record what those checks found.

## Important limitations and responsible use

- **Evidence is limited to titles and retrieved abstracts.** No full text is
  retrieved. Missing abstracts lead to judgements based on metadata.
  The study itself remains unread.
- **Retrieval and screening have different limits.** Recall was measured on
  one ivermectin/COVID-19 question using 1,000 requested results per search. It is not evidence of the default 50 result setting or performance across
  research topics. Semantic Scholar returns at most 100 per search.
- **The relevance score is not calibrated.** The model is asked for a whole
  number from 0 to 10, and nothing defines what any point on that scale means.
  Ranking and the review list's opening threshold both rest on it. The number
  orders papers against each other. It does not measure how relevant one is.
- **Screening judges the criteria as written.** Both the inclusion and the
  exclusion criteria go into the request, so a recommendation records
  agreement with those criteria rather than relevance in any wider sense.
- **An exact extract check can refuse a faithful quotation after a small
  character change.** Such records require review.
  Storage refuses them until the extract passes its checks. Results and observed cases are reported in testing.
- **Coverage is incomplete.** Only PubMed and Semantic Scholar are searched.
  Crossref status can lag a publication event.
- **Human evidence is limited.** The browser sessions involved one participant familiar with the project.
  They do not establish usability for a representative population.
  Synthetic tests do not establish live model quality. Their methods and outcomes are kept distinct in testing.
- **Patient data is out of scope and is not sought.** Retrieved literature
  is not screened for every possible patient detail. Do not enter sensitive
  or undisclosable material in questions, criteria or notes.
- **Live calls disclose data to external services.** Searches go to the
  relevant services. Google receives the question, criteria and records
  submitted for screening, and a bounded reviewer note when rejudging.
  Local storage is not uploaded.
  Live processing includes these external requests.
- **Publication rights are separate from access and replay approval.**
  Retrieved content is not covered by this project's MIT code licence.
  Private capture is not permission to redistribute it.
- **Run log locking is within one process.** Do not run multiple application
  processes against the same log file.

The [development account](docs/development.md) gives the evidence and
remaining obligations behind these boundaries.

## Legal, social and ethical factors

The Recommendation on the ethics of artificial intelligence (UNESCO, 2022)
sets out ten principles. The six below bear on a tool that decides which
research a person reads, and each is answered by a mechanism in this code
rather than by a statement of intent. A seventh, fairness and
non-discrimination, is taken after the table.

| Principle | The mechanism | What it does not settle |
|---|---|---|
| Proportionality and do no harm | A run is bounded to 18 model requests and 50 screened papers, 10 abstracts to a request, and no full text is retrieved. The live run of 18 September completed on the smaller Flash-Lite model | A bound on one run is no bound on a day's usage or on an account allowance |
| Safety and security | Retrieved text is passed to the model as information to judge rather than as instructions, and storage checks the extract again instead of trusting the record of the earlier check | Resistance to semantic prompt injection is not established |
| Right to privacy and data protection | Patient data falls outside the scope. The question, criteria and paper fields used for screening (title, year, what was retrieved, publication status and abstract) go to the model service; when a paper is sent back, a bounded copy of the reviewer's note goes with it. Local storage itself is not uploaded | Questions and criteria leave the machine. A review note leaves it only when the corresponding paper is sent back |
| Responsibility and accountability | A paper is stored only after approval, and every stored record keeps its decision, its reason and the extract that supports it | Responsibility for what is stored rests with the reviewer, not with the tool |
| Transparency and explainability | Every judgement keeps its extract and is marked with whether that extract occurs in the title or abstract of the paper it names. The interface refuses to approve a paper whose extract was not found, and storage refuses the whole batch if one gets through | The relevance score has no defined meaning. The model gives one short reason alongside its judgement; this is not its private reasoning process |
| Human oversight and determination | The available decisions are approve, flag or reject. A flagged or rejected paper can be sent back for a fresh judgement, which returns it to waiting. Storage can continue while other papers are still waiting, and the screen shows how many remain | Oversight reaches what is shown, not what was never retrieved |

Fairness and non-discrimination is the principle this design answers least
well. Ranking uses the uncalibrated score described above. The review list
opens with the papers that score highest, while papers rated 5 or below are
held back and only counted, unless the reviewer has already answered one or
sent it back for a fresh judgement. A paper that stays held back is therefore
not shown to the reviewer, even though the search returned it. Searching only PubMed and Semantic Scholar
narrows the field again. This project did not measure language or geographic
coverage. The effect is not neutral, because it decides whose research is
seen. Nothing in the design corrects for it, which is why every run reports
its coverage instead of assuming it.

## Submission contents

| Material | Location |
|---|---|
| Source code and configuration | `research_agent/`, `scripts/`, dependency files and `.env.example` |
| Evidence of development progress over the module | [Development account](docs/development.md) |
| Comprehensive evidence of testing | [Testing account](docs/testing.md), `tests/` and dated `evidence/` files |
| Evidence of execution | `evidence/execution/`, the documented synthetic demonstration |
| Class and sequence diagrams | [Class diagram](docs/class_diagram.md), [sequence diagram](docs/sequence_diagram.md) |
| Presentation, transcript and recording | The slide deck, the written narration and the recorded presentation, supplied with this submission |
| Public repository | <https://github.com/Tanky0/Intelligent-Agents---Individual-Project-Presentation> includes the code, the documentation and the evidence; not the presentation, transcript or recording |

## Sources and acknowledgements

The references below support decisions explained in the development account,
testing account and code.

### Academic and model sources

Bratman, M.E., Israel, D.J. and Pollack, M.E. (1988) 'Plans and
resource-bounded practical reasoning', *Computational Intelligence*, 4(3),
pp. 349–355.

Brooks, R.A. (1991) 'Intelligence without representation', *Artificial
Intelligence*, 47(1-3), pp. 139–159.

Finin, T. et al. (1994) 'KQML as an agent communication language',
*Proceedings of the Third International Conference on Information and Knowledge
Management (CIKM '94)*, Gaithersburg, MD. New York: ACM, pp. 456–463. Available
at: https://doi.org/10.1145/191246.191322

Google (2026) *Troubleshooting guide*, Gemini API documentation. Available
at: https://ai.google.dev/gemini-api/docs/troubleshooting (Accessed:
14 September 2026).

Google DeepMind (2026a) *Gemini 3.5 Flash-Lite: model card*. Available at:
https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/
(Accessed: 19 September 2026).

Google DeepMind (2026b) *Gemini 3.6 Flash: model card*. Available at:
https://deepmind.google/models/model-cards/gemini-3-6-flash/
(Accessed: 27 August 2026).

Page, M.J. et al. (2021) 'The PRISMA 2020 statement: an updated guideline for
reporting systematic reviews', *International Journal of Surgery*, 88, 105906.
Available at: https://doi.org/10.1016/j.ijsu.2021.105906

Rao, A.S. and Georgeff, M.P. (1995) 'BDI agents: from theory to practice',
*Proceedings of the First International Conference on Multiagent Systems
(ICMAS-95)*, San Francisco, CA, 12-14 June. Menlo Park, CA: AAAI Press,
pp. 312–319.

Rethlefsen, M.L. et al. (2021) 'PRISMA-S: an extension to the PRISMA statement
for reporting literature searches in systematic reviews', *Journal of the
Medical Library Association*, 109(2), pp. 174–200. Available at:
https://doi.org/10.5195/jmla.2021.962

Russell, S. and Norvig, P. (2022) *Artificial intelligence: a modern
approach*. 4th edn, global edn. Harlow: Pearson.

Wang, G. et al. (2026) 'Human–large language model collaboration in clinical
medicine: a systematic review and meta-analysis', *npj Digital Medicine*, 9(1),
195. Available at: https://doi.org/10.1038/s41746-026-02382-2

Wang, Z. et al. (2025) 'A foundation model for human-AI collaboration in
medical literature mining', *Nature Communications*, 16(1), 8361. Available at:
https://doi.org/10.1038/s41467-025-62058-5

Wooldridge, M. (2009) *An introduction to multiagent systems*. 2nd edn.
Chichester: John Wiley & Sons.

Yao, S. et al. (2023) 'ReAct: synergizing reasoning and acting in language
models', *11th International Conference on Learning Representations (ICLR
2023)*, Kigali, Rwanda, 1-5 May. Available at:
https://doi.org/10.48550/arXiv.2210.03629

Yu, F., Liu, C. and Sharmin, S. (2022) 'Performance, usability, and user
experience of Rayyan for systematic reviews', *Proceedings of the Association
for Information Science and Technology*, 59(1), pp. 843–844. Available at:
https://doi.org/10.1002/pra2.745.

### Practitioner source

Sauro, J. (2011) 'Measuring usability with the System Usability Scale (SUS)',
*MeasuringU*, 3 February. Available at: https://measuringu.com/sus/
(Accessed: 9 September 2026).

SUS was withdrawn from this project's sessions on 17 September; no project
SUS score is claimed or compared with Rayyan. The source remains because the
earlier acceptance condition used it.

### Standard

UNESCO (2022) *Recommendation on the ethics of artificial intelligence*.
Paris: UNESCO. Available at:
https://unesdoc.unesco.org/ark:/48223/pf0000381137 (Accessed: 30 September
2026).

The ten principles are used as the reference framework for the legal, social
and ethical factors above. Adoption of that framework is not a claim of
conformance with it.

World Wide Web Consortium (2024) *Web Content Accessibility Guidelines (WCAG)
2.2*. W3C Recommendation, 12 December. Available at:
https://www.w3.org/TR/2024/REC-WCAG22-20241212/ (Accessed: 8 September 2026).

WCAG 2.2 was used as an evaluation reference, not a claim of legal
certification or complete conformance.

### Service guidance

National Library of Medicine (2025) *Copyright notice*, PubMed Central.
Available at: https://pmc.ncbi.nlm.nih.gov/about/copyright/ (Accessed:
14 September 2026).

National Library of Medicine (2026) *For developers*, PubMed Central.
Available at: https://pmc.ncbi.nlm.nih.gov/tools/developers/ (Accessed:
14 September 2026).

Sayers, E. (2022) 'A general introduction to the E-utilities', in *Entrez
Programming Utilities Help*. Bethesda, MD: National Center for Biotechnology
Information. Available at: https://www.ncbi.nlm.nih.gov/books/NBK25497/
(Accessed: 9 September 2026).

The PMC guidance is retained for the historical full-text route decision.
That source was removed on 19 September and is not used by the current run.

### Datasets

CLEF-TAR (2019) *Technologically Assisted Reviews in Empirical Medicine*.
Available at: https://github.com/CLEF-TAR/tar (Accessed: 14 September 2026).

Cohen, A.M. (no date) *Systematic Drug Class Review Gold Standard Data*.
Available at:
https://dmice.ohsu.edu/cohenaa/systematic-drug-class-review-data.html
(Accessed: 14 September 2026).

Cohen, A.M. et al. (2006) 'Reducing workload in systematic review preparation
using automated citation classification', *Journal of the American Medical
Informatics Association*, 13(2), pp. 206–219. Available at:
https://doi.org/10.1197/jamia.M1929

De Bruin, J. et al. (2023) *SYNERGY - Open machine learning dataset on study
selection in systematic reviews*. Version 1. DataverseNL. Available at:
https://doi.org/10.34894/HE6NAQ.

Kanoulas, E. et al. (2019) 'CLEF 2019 Technology Assisted Reviews in
Empirical Medicine Overview', in Cappellato, L. et al. (eds) *Working Notes
of CLEF 2019 - Conference and Labs of the Evaluation Forum*, Lugano,
Switzerland, 9-12 September. CEUR Workshop Proceedings, 2380. Available at:
https://ceur-ws.org/Vol-2380/ (Accessed: 14 September 2026).

CLEF TAR 2019 was selected. The testing account explains the comparison with
the other two collections and the evaluation's limits.

### Software and writing tools

The dependency files are authoritative for installed versions. These credits
retain the authors/publishers and release details recorded by the project.
Their exact formatting against the UoEO software-reference guidance has not
been verified.

| Library | Used for |
|---|---|
| Python Software Foundation (2026) *Python* (Version 3.13.14). Available at: https://www.python.org/downloads/release/python-31314/ (Accessed: 9 September 2026) | The language |
| LangChain (2026) *LangGraph* (Version 1.2.11). Available at: https://pypi.org/project/langgraph/1.2.11/ (Accessed: 9 September 2026) | The state graph wiring the two agents together |
| LangChain (2026) *LangChain Core* (Version 1.6.0). Available at: https://pypi.org/project/langchain-core/1.6.0/ (Accessed: 9 September 2026) | The model interface LangGraph builds on |
| LangChain (2026) *langchain-google-genai* (Version 4.3.5). Available at: https://pypi.org/project/langchain-google-genai/4.3.5/ (Accessed: 9 September 2026) | Reaching Gemini through LangChain |
| Google LLC (2026) *google-genai* (Version 2.20.0). Available at: https://pypi.org/project/google-genai/2.20.0/ (Accessed: 9 September 2026) | Google's own library, which the above builds on |
| Colvin, S. et al. (2026) *Pydantic* (Version 2.13.4). Available at: https://pypi.org/project/pydantic/2.13.4/ (Accessed: 9 September 2026) | The checked record formats |
| Colvin, S. et al. (2026) *pydantic-settings* (Version 2.15.0). Available at: https://pypi.org/project/pydantic-settings/2.15.0/ (Accessed: 9 September 2026) | Reading settings from the environment |
| Christie, T. et al. (2024) *HTTPX* (Version 0.28.1). Available at: https://pypi.org/project/httpx/0.28.1/ (Accessed: 9 September 2026) | Every request to a service |
| Lundberg, J. (2026) *RESPX* (Version 0.23.1). Available at: https://pypi.org/project/respx/0.23.1/ (Accessed: 9 September 2026) | Answering those requests in tests without a network |
| Danjou, J. (2026) *Tenacity* (Version 9.1.4). Available at: https://pypi.org/project/tenacity/9.1.4/ (Accessed: 9 September 2026) | Waiting before a retry |
| The pandas development team (2026) *pandas* (Version 3.0.5). Available at: https://pypi.org/project/pandas/3.0.5/ (Accessed: 9 September 2026) | Pinned because Streamlit requires it. No file in this project imports it: the screening files are read and written with the standard library's `csv` |
| Bachmann, M. (2026) *RapidFuzz* (Version 3.14.5). Available at: https://pypi.org/project/RapidFuzz/3.14.5/ (Accessed: 9 September 2026) | Comparing titles when deduplicating (`processing/dedup.py`) |
| Snowflake Inc. (2026) *Streamlit* (Version 1.62.0). Available at: https://pypi.org/project/streamlit/1.62.0/ (Accessed: 9 September 2026) | The reviewer's screen |
| Kumar, S. (2026) *python-dotenv* (Version 1.2.3). Available at: https://pypi.org/project/python-dotenv/1.2.3/ (Accessed: 9 September 2026) | Loading `.env` |
| Larson, S.M. and Glick, D. (2025) *truststore* (Version 0.10.4). Available at: https://pypi.org/project/truststore/0.10.4/ (Accessed: 9 September 2026) | Using the Windows certificate store, so HTTPS works behind the antivirus (R-01) |
| Alaee, A. (2026) *uuid_utils* (Version 0.16.2). Available at: https://pypi.org/project/uuid-utils/0.16.2/ (Accessed: 9 September 2026) | Loaded by LangChain rather than used directly, and pinned for the reason in `requirements.txt` (R-09) |
| Krekel, H. et al. (2026) *pytest* (Version 9.1.1). Available at: https://pypi.org/project/pytest/9.1.1/ (Accessed: 9 September 2026) | Running the tests |
| Schlaich, M. (2026) *pytest-cov* (Version 7.1.0). Available at: https://pypi.org/project/pytest-cov/7.1.0/ (Accessed: 9 September 2026) | Measuring what the tests reach |
| Astral Software Inc. (2026) *Ruff* (Version 0.16.4). Available at: https://pypi.org/project/ruff/0.16.4/ (Accessed: 9 September 2026) | Style checking |
| Lehtosalo, J. (2026) *mypy* (Version 2.3.1). Available at: https://pypi.org/project/mypy/2.3.1/ (Accessed: 9 September 2026) | Type checking |

The diagram tools were used for rendering checks, not installed by the Python
requirements. GitHub uses its own Mermaid renderer. Grammarly was used on
written text, not application code.

| Library | Used for |
|---|---|
| Sveidqvist, K. (2026) *Mermaid* (Version 11.17.2). Available at: https://www.npmjs.com/package/mermaid/v/11.17.2 (Accessed: 14 September 2026) | The notation of both diagrams, and the library that drew them |
| Long, T. (2026) *mermaid-cli* (Version 11.17.0). Available at: https://www.npmjs.com/package/@mermaid-js/mermaid-cli/v/11.17.0 (Accessed: 14 September 2026) | Rendering the diagrams to pictures on the computer used to develop and test the application, to check that they draw |
| Grammarly Inc. (2026) *Grammarly*. Available at: https://www.grammarly.com/ (Accessed: 23 September 2026) | Proofreading and language refinement of the written text. Not part of the application and not installed by it |

## Use of generative AI

AI Acknowledgment

Claude was used to assist in identifying appropriate code solutions and to
proofread and refine the code I produced. Codex was used to verify the
correctness of the coding outputs. Claude was also used to help write and
run test cases during development. Claude and Grammarly supported
proofreading, language refinement, and formatting of my written text. All
final decisions, analyses, interpretations, conclusions, and
recommendations are my own.
