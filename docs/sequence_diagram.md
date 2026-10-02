# Sequence diagram: planning, handover and human review

The three views show agent interactions and error handling (TF-02).

Solid arrows are calls or human actions. Dotted arrows are returns or raised errors. `alt`, `loop` and `break` show branches, repetition and stops. The model lane combines the bounded repair helper and Gemini adapter. The JSON buffer is serialised and validated between graph steps. PubMed illustrates retrieval. Semantic Scholar follows the same planner boundary. Crossref checks publication status. It does not search for papers.

## View 1: Planning and searching

```mermaid
sequenceDiagram
    autonumber
    actor Reviewer
    participant Screen as ui/app.py
    participant Graph as graph.py
    participant Planner as agents/planner.py
    participant Model as asking.py and llm.py, the model
    participant Source as sources/pubmed.py

    Reviewer->>Screen: presses Start the run
    Screen->>Graph: graph.run(request, planner, curator, step_limit)
    Graph->>Planner: plan_searches(state)
    Planner->>Model: ask_for_a_record(prompt, SubgoalList, budget)
    Model-->>Planner: SubgoalList, or OutOfBudget, ModelFailure or None, with the budget
    Planner-->>Graph: the same answer and budget
    break OutOfBudget
        Graph->>Graph: plan writes stopped_because, after_planning routes to the finishing route
    end
    break ModelFailure or None
        Graph->>Graph: plan writes needs_human_attention, after_planning routes to the finishing route
    end
    loop while after_searching routes to search or refine
        Graph->>Planner: run_next_search(state)
        Planner->>Source: search(query, limit, years)
        Source-->>Planner: identifiers, or a FetchError
        Planner->>Source: fetch(identifiers, or the FetchError)
        Source-->>Planner: records, and the batches that failed
        Planner-->>Graph: papers, Outcome finished, partial or failed
        Graph->>Graph: ResultsBuffer(papers, outcomes) checks every record
        break a record or an outcome the format refuses
            Graph->>Graph: search writes needs_human_attention, the searches that ran are kept
        end
        Graph->>Graph: run_log.note_a_search(outcome)
        alt steps_taken has reached step_limit
            Graph->>Graph: no move is asked for, after_searching routes to the hand-over
        else a step is left
            Graph->>Planner: decide_next_move(observed, outcome, papers)
            Planner->>Model: ask_for_a_record(prompt, NextMove, budget)
            Model-->>Planner: NextMove, or OutOfBudget, ModelFailure or None, with the budget
            Planner-->>Graph: the same answer and budget
            break OutOfBudget
                Graph->>Graph: search writes stopped_because, the searches that ran are kept
            end
            break ModelFailure or None
                Graph->>Graph: search writes needs_human_attention, the searches that ran are kept
            end
            Graph->>Graph: last_move and a PlanningDecision are written
            Graph->>Graph: after_searching: finish hands over, refine puts the replacement first,<br/>continue or drop takes the next search, or hands over when the queue is empty
        end
    end
```

## View 2: Hand over, screening and completion

```mermaid
sequenceDiagram
    autonumber
    participant Screen as ui/app.py
    participant Graph as graph.py
    participant Crossref as sources/crossref.py
    participant Buffer as the JSON buffer, text in memory
    participant Curator as agents/curator.py
    participant Model as asking.py and llm.py, the model

    Graph->>Graph: hand_over: ResearchState.model_validate(state.model_dump())
    break the record breaks its own rules
        Graph-->>Screen: ValidationError raised out of graph.run,<br/>nothing below runs and no report is built
    end
    Graph->>Graph: deduplicate(retrieved), then share_out_places<br/>within budget.papers_it_can_still_take()
    Graph->>Crossref: fetch_statuses(dois), one call with the list
    Crossref-->>Graph: one answer for each distinct DOI,<br/>a status or a FetchError, mixed
    Graph->>Graph: _with_the_status_checked: a status into each paper,<br/>a FetchError leaves it blank, listed in status_checks_failed and logged
    Graph->>Buffer: as_json = finished.model_dump_json(), this text is the hand-over
    Graph->>Graph: _write_the_buffer(as_json): a copy to a file as evidence,<br/>an OSError is caught inside it and recorded, the run goes on
    Graph->>Buffer: ResultsBuffer.model_validate_json(as_json),<br/>read back into the state as results
    Graph->>Curator: screen(state)
    loop each batch of ten papers in state.results.papers, until one fails
        Curator->>Model: ask_for_a_record(prompt, ScreeningList, budget)
        Model-->>Curator: ScreeningList, or OutOfBudget, ModelFailure or None, with the budget
        Curator->>Curator: _every_paper_was_judged_once(answer, batch size), when it is a ScreeningList
    end
    alt every batch answered and matched
        Curator-->>Graph: list of ScreenedPaper in the order given, with the budget
        Graph->>Graph: curate writes screened, and the budget
    else a batch was refused, failed, or did not match the papers sent
        Curator-->>Graph: that OutOfBudget, ModelFailure or None, or a ModelFailure<br/>for a batch that did not match, with the budget, the batching stopped there
        Graph->>Graph: curate writes stopped_because for OutOfBudget,<br/>or needs_human_attention for the rest, and the budget
    end
    Note over Screen,Model: the finishing route, for every run that did not raise, whether it stopped early or screened
    Graph->>Graph: run: ResearchState.model_validate(finished)
    break a judgement that is not one, or any other rule broken
        Graph-->>Screen: ValidationError raised out of graph.run, no report is built
    end
    Graph->>Graph: with_the_extracts_checked, then in_rank_order
    Graph-->>Screen: ResearchState, checked and ranked,<br/>with stopped_because or needs_human_attention if it stopped early
    Screen->>Screen: audit_log.report_on(state, failures during the run),<br/>kept in the session with the state
```

## View 3: Reviewer decisions, revision and storage

```mermaid
sequenceDiagram
    autonumber
    actor Reviewer
    participant Screen as ui/app.py
    participant Graph as graph.py
    participant Curator as agents/curator.py
    participant Model as asking.py and llm.py, the model
    participant Store as processing/storage.py

    Screen->>Reviewer: each paper with its judgement, summary, extract,<br/>and whether the extract was found
    Reviewer->>Screen: presses approved, flagged or rejected,<br/>with a note in the box beside it
    Screen->>Screen: decide(judged, choice, note)
    break flagged or rejected with no note, or approved<br/>with an extract not found or never looked for
        Screen-->>Reviewer: ReviewRefused shown, the earlier answer stands
    end
    Screen->>Screen: the answer kept in the session,<br/>run_log.note_a_decision(judged, choice, note)
    Note over Reviewer,Screen: an answer is all this records: the reviewer<br/>may change it, send papers back, or store
    opt the reviewer presses Send back, the flagged alone or with the rejected, offered only when there are some
        Screen->>Graph: judge_again(decided, curator, answers_that_go)
        Graph->>Curator: judge_again(state, sent_back)
        Curator->>Model: ask_for_a_record(revision prompt, ScreeningList, budget),<br/>once for each batch of ten
        Model-->>Curator: ScreeningList, or OutOfBudget, ModelFailure or None,<br/>with the budget
        Note over Curator,Model: the same batching, batch check and stops<br/>as screening in view 2
        Curator-->>Graph: list of ScreenedPaper, or OutOfBudget,<br/>ModelFailure or None, with the budget
        break OutOfBudget, ModelFailure or None
            Graph-->>Screen: state with stopped_because or needs_human_attention,<br/>and the budget as returned
            Screen-->>Reviewer: the reason shown and written to the run log,<br/>papers and answers left as they were
        end
        Graph->>Graph: with_the_extracts_checked, each fresh judgement in place<br/>of the one sent back with its note kept, ResearchState.model_validate
        Graph-->>Screen: state with those papers waiting again
        Screen->>Screen: run_log.note_a_revision for each, the answers on<br/>those papers cleared, the page drawn again
        Screen-->>Reviewer: the fresh judgements, each needing a new decision
    end
    Reviewer->>Screen: presses Display Approved
    Screen->>Store: save_approved(decided_so_far(session, state.screened)),<br/>the answers as they stand now
    break RefusedByStorage: an extract never looked for, not found or no longer in the paper,<br/>a paper with no identifier, or identifiers that disagree with a stored row
        Store-->>Screen: RefusedByStorage raised, nothing written
        Screen-->>Reviewer: the refusal shown and written to the run log
    end
    Store->>Store: one write: rows of papers now rejected, flagged or waiting<br/>withdrawn, each approved paper inserted or updated
    Store-->>Screen: how many were stored
    Screen-->>Reviewer: the results page, the stored line at its top,<br/>the approved papers read back from the store and the run log
```

## Error paths and verification

| Path | Behaviour and evidence |
|---|---|
| Source unavailable | Bounded HTTP retries return a failure. The planner records failed/partial coverage. UT-02/UT-22 and FT-04 |
| Unreadable or wrong shape service reply | Source failure reported. Invalid shared records are refused at graph boundaries and shown/logged by the screen. UT-07, IT-03 |
| Invalid plan | Initial request plus at most three schema repairs, within the run budget. Unresolved failure asks for human attention. UT-06 and FT-05 |
| Budget exhausted | Refusal before spending the next request. Reason carried to the screen. UT-25, IT-05/IT-08 |
| Human flag/rejection | Notes accompany selected papers on revision. Fresh judgements need new decisions. IT-08 and FT-09 |

The integration tests differ in what they replace. Graph tests may use replacement agents. Planner tests in the graph keep the agents and replace model and source calls. AppTest exercises the screen. Storage tests use a temporary database. Passing one does not verify the whole workflow against live services.

The diagram was read against the code it describes and rendered at the size it
is published in.
