# Class diagram: records and stored data

The three views cover the main data entities, their properties and their relationship to the Pydantic models and SQLite table (TF-01).

Records extend `Checked`. Solid arrows show fields holding records. Labels give multiplicity. The inheritance arrow represents the common base. The dotted arrow in view 3 shows conditional storage. It does not show ownership. `list~X~` means `list[X]`. A default follows `=`. A record may be valid when built and later changed, so the graph revalidates the whole record at handover and completion.

## View 1: The request, plan and run

```mermaid
classDiagram
    direction TB

    class Checked {
        <<base of every Pydantic record here>>
        model_config: strict, extra forbidden, checked on assignment
    }

    class ResearchRequest {
        question: str
        inclusion_criteria: str
        exclusion_criteria: str
        from_year: int or None = None [1500..3000]
        to_year: int or None = None [1500..3000]
        results_per_search: int = 50 [1..10000]
    }

    class ResearchState {
        request: ResearchRequest
        run_id: str = ""
        pending_searches: list~Subgoal~ = []
        results: ResultsBuffer = an empty buffer, view 2
        screened: list~ScreenedPaper~ = [], view 3
        decisions: list~PlanningDecision~ = []
        last_move: NextMove or None = None
        needs_human_attention: bool = False
        attention_reason: str = ""
        budget: RunBudget = the defaults below
        stopped_because: str = ""
        steps_taken: int = 0 [0..]
        step_limit: int = 12 [1..]
    }

    class RunBudget {
        model_request_limit: int = 18 [1..]
        model_requests_used: int = 0 [0..]
        papers_for_curation_limit: int = 50 [1..]
        papers_sent_to_curation: int = 0 [0..]
        has_a_model_request_left() bool
        with_one_model_request_used() RunBudget
        papers_it_can_still_take() int
        with_papers_sent_to_curation(how_many) RunBudget
    }

    class Subgoal {
        <<filled by the model>>
        purpose: str
        source: SearchableSource
        query: str
    }

    class SubgoalList {
        <<answer format for a plan>>
        subgoals: list~Subgoal~
    }

    class NextMove {
        <<answer format after a search>>
        move: Move
        reason: str
        new_query: str = ""
    }

    class PlanningDecision {
        subgoal: Subgoal
        move: Move
        reason: str
    }

    Checked <|-- ResearchState : and the other thirteen Pydantic records
    ResearchState --> "1" ResearchRequest : request
    ResearchState --> "1" RunBudget : budget
    ResearchState --> "0..*" Subgoal : pending_searches
    ResearchState --> "0..*" PlanningDecision : decisions
    ResearchState --> "0..1" NextMove : last_move
    SubgoalList --> "1..*" Subgoal : subgoals
    PlanningDecision --> "1" Subgoal : subgoal
```

## View 2: Retrieved papers and search outcomes

```mermaid
classDiagram
    direction TB

    class ResultsBuffer {
        papers: list~PaperRecord~ = []
        outcomes: list~Outcome~ = []
        ready: bool = False
        papers_before_deduplication: list~PaperRecord~ = []
        papers_beyond_the_screening_limit: list~PaperRecord~ = []
        status_checks_failed: list~str~ = []
    }

    class Outcome {
        subgoal: Subgoal
        result: finished, partial, dropped or failed
        papers_found: int = 0 [0..]
        reason: str = ""
    }

    class Subgoal {
        <<as in view 1>>
        purpose: str
        source: SearchableSource
        query: str
    }

    class PaperRecord {
        title: str
        origins: list~SearchOrigin~
        abstract: str = ""
        full_text: str = ""
        full_text_licence: str = ""
        authors: list~str~ = []
        year: str = ""
        journal: str = ""
        publication_types: list~str~ = []
        pmid: str = ""
        doi: str = ""
        paper_id: str = ""
        publication_status: PublicationStatus = ""
        text_source: TextSource = "metadata"
    }

    class SearchOrigin {
        source: SearchableSource
        query: str = ""
    }

    ResultsBuffer --> "0..*" PaperRecord : papers, and the two kept-aside lists
    ResultsBuffer --> "0..*" Outcome : outcomes
    Outcome --> "1" Subgoal : subgoal
    PaperRecord --> "1..*" SearchOrigin : origins
```

## View 3: Judgement, human decision and SQLite row

```mermaid
classDiagram
    direction LR

    class PaperRecord {
        <<abbreviated, drawn in full in view 2>>
        title, authors, year, journal, publication_status
        doi, pmid, paper_id
        abstract, full_text: never written to the store
    }

    class Screening {
        <<filled by the model>>
        paper_number: int [1..]
        decision: ScreeningChoice
        relevance: int [0..10]
        reason: str
        summary: str
        extract: str
    }

    class ScreeningList {
        <<answer format for a batch>>
        screenings: list~Screening~
    }

    class ScreenedPaper {
        paper: PaperRecord
        screening: Screening
        extract_found: bool or None = None
        review: ReviewChoice = "waiting"
        review_note: str = ""
    }

    class approved_papers {
        <<SQLite table, not a Pydantic record>>
        identifier: TEXT PRIMARY KEY
        doi: TEXT NOT NULL
        pmid: TEXT NOT NULL
        paper_id: TEXT NOT NULL
        title: TEXT NOT NULL
        authors: TEXT NOT NULL
        year: TEXT NOT NULL
        journal: TEXT NOT NULL
        publication_status: TEXT NOT NULL
        decision: TEXT NOT NULL
        relevance: INTEGER NOT NULL
        reason: TEXT NOT NULL
        summary: TEXT NOT NULL
        extract: TEXT NOT NULL
        review_note: TEXT NOT NULL
    }

    ScreeningList --> "1..*" Screening : screenings
    ScreenedPaper --> "1" PaperRecord : paper
    ScreenedPaper --> "1" Screening : screening
    ScreenedPaper ..> "0..1" approved_papers : save_approved, when review is approved
```

## Storage and scope

A judgement is paired with its paper by batch number. Response order is not used. The curator checks that every requested paper is judged exactly once. Storage accepts only approved, identified records whose extract passes its checks. One refusal stops the batch. The paper's abstract and full text are not stored. Repeated approval updates a row, and a withdrawn decision removes it.

These views omit service reply formats and the records used to read stored results and prepare reports. Fixed values and detailed validators are defined in `research_agent/schemas.py`. SQLite columns and identifier rules are in `research_agent/processing/storage.py`. The record keeps a full-text field, which no implemented run fills.

## Verification

The diagrams were read against the code they describe and rendered at the size
they are published in.
