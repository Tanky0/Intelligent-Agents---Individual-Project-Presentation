"""The outcome of a run, put together from the log and the store (FR-14, FR-16)."""
# Every test writes its own log and its own store under the folder pytest
# gives it, so none touches the files a real run uses, and none asks a model
# or a service: the outcome is read, never made.
#
# What matters is that the two records are joined honestly: the last decision
# counts, a DOI spelt two ways is one paper, and a paper the store has let go
# is named rather than dropped.

import json

from research_agent.processing import results, run_log, storage
from research_agent.schemas import (
    Outcome,
    PaperRecord,
    ResearchRequest,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
)


def a_paper(title="A trial of drug X", **changes):
    values = {
        "title": title,
        "doi": "https://doi.org/10.1000/ABC",
        "authors": ["Smith J", "Jones A"],
        "year": "2024",
        "journal": "Journal of Trials",
        "abstract": "Forty patients were given drug X.",
        "text_source": "abstract",
        "origins": [SearchOrigin(source="pubmed", query="drug X")],
    }
    values.update(changes)
    return PaperRecord(**values)


def judged(paper=None, review="approved", note=""):
    return ScreenedPaper(
        paper=paper or a_paper(),
        screening=Screening(paper_number=1, decision="include", relevance=8,
                            reason="it reports an outcome for drug X",
                            summary="a trial of drug X in condition Y",
                            extract="Forty patients were given drug X"),
        review=review,
        review_note=note,
        extract_found=True)


def a_request():
    return ResearchRequest(question="Does drug X treat condition Y?",
                           inclusion_criteria="trials",
                           exclusion_criteria="reviews")


def a_search():
    return Outcome(subgoal=Subgoal(purpose="find trials", source="pubmed",
                                   query="drug X trials"),
                   result="finished", papers_found=1)


def a_recorded_run(tmp_path, decisions=(("approved", ""),), run="run-1"):
    """A log with one run in it, and a store holding what its decisions left."""
    # The decisions are written in order, so the last is the one that stands,
    # and the store is written from the last as the screen would write it.
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    run_log.note_the_request(a_request(), run, log)
    run_log.note_a_search(a_search(), a_request(), run, log)
    for choice, note in decisions:
        run_log.note_a_decision(judged(), choice, note, run, log)
    choice, note = decisions[-1]
    storage.save_approved([judged(review=choice, note=note)], store)
    return log, store


def test_the_outcome_holds_the_request_the_searches_and_the_stored_paper(tmp_path):
    """The recap is the log's, the paper is the store's, and the two are
    joined by the DOI although the log spells it as the source did and the
    store as storage normalised it."""
    log, store = a_recorded_run(tmp_path, [("approved", "a sound trial")])

    outcome = results.outcome_of("run-1", log, store)

    assert outcome.question == "Does drug X treat condition Y?"
    assert outcome.include == "trials" and outcome.exclude == "reviews"
    assert outcome.searches == ["pubmed: drug X trials, finished, 1 paper found"]
    assert (outcome.approved, outcome.flagged, outcome.rejected) == (1, 0, 0)
    assert outcome.not_in_the_store == []
    [paper] = outcome.papers
    assert paper.title == "A trial of drug X"
    assert paper.identifier == "10.1000/abc" and paper.doi == "10.1000/abc"
    assert paper.authors == "Smith J; Jones A"
    assert paper.summary == "a trial of drug X in condition Y"
    assert paper.review_note == "a sound trial"


def test_the_last_decision_on_a_paper_is_the_one_that_counts(tmp_path):
    """A paper approved, sent back and rejected is one rejected paper."""
    log, store = a_recorded_run(tmp_path, [("approved", ""),
                                           ("rejected", "it is a review")])

    outcome = results.outcome_of("run-1", log, store)

    assert (outcome.approved, outcome.flagged, outcome.rejected) == (0, 0, 1)
    assert outcome.papers == [] and outcome.not_in_the_store == []
    assert outcome.left_out == [
        results.LeftOut("https://doi.org/10.1000/ABC", "rejected", "it is a review")]


def test_the_review_finished_at_the_last_decision_or_store(tmp_path):
    """The results page says when a review finished: the last store when there is
    one, the last decision before any, the start before either.
    """
    # A run reopened tomorrow says when it finished, not when it was reopened.
    log, store = a_recorded_run(tmp_path, [("approved", "")])
    decided_at = results.outcome_of("run-1", log, store).finished
    assert decided_at == run_log.runs(log)[0].entries[-1]["at"]

    run_log.note_a_store(1, "run-1", log)
    stored_at = run_log.runs(log)[0].entries[-1]["at"]
    assert stored_at >= decided_at
    assert results.outcome_of("run-1", log, store).finished == stored_at

    assert results.when_the_review_finished([{"event": "run started", "at": "t0"}]) == "t0"
    assert results.when_the_review_finished([{"event": "search"}]) == ""


def test_a_paper_the_store_has_let_go_is_named_rather_than_dropped(tmp_path):
    """The store keeps the last decision across runs, so a later rejection
    empties a row this run's log still says was approved."""
    log, store = a_recorded_run(tmp_path, [("approved", "")])
    storage.save_approved([judged(review="rejected", note="withdrawn")], store)

    outcome = results.outcome_of("run-1", log, store)

    assert outcome.approved == 1
    assert outcome.papers == []
    assert outcome.not_in_the_store == ["https://doi.org/10.1000/ABC"]


def test_a_run_with_some_approved_papers_in_the_store_names_the_others(tmp_path):
    """Two approvals and one row: the recap says one of two, and names the
    one the store does not hold without saying why (found in review on 21
    September: the page said all were saved, and the absent one withdrawn)."""
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    trial = judged()
    cohort = judged(a_paper("A cohort of drug X", doi="", pmid="222"))
    run_log.note_the_request(a_request(), "run-1", log)
    run_log.note_a_decision(trial, "approved", "", "run-1", log)
    run_log.note_a_decision(cohort, "approved", "", "run-1", log)
    storage.save_approved([trial], store)

    outcome = results.outcome_of("run-1", log, store)

    assert outcome.approved == 2
    assert [one.title for one in outcome.papers] == ["A trial of drug X"]
    assert outcome.not_in_the_store == ["222"]


def test_a_run_with_approvals_and_no_store_is_not_called_stored(tmp_path):
    """An approval is logged at the press and the store after it, so a run
    can hold approvals and no store; the results page must not call that
    review complete."""
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    run_log.note_the_request(a_request(), "run-1", log)
    run_log.note_a_decision(judged(), "approved", "", "run-1", log)

    before = results.outcome_of("run-1", log, store)
    assert before.stored is False
    assert before.approved == 1 and before.papers == []
    assert before.not_in_the_store == ["https://doi.org/10.1000/ABC"]
    assert [one.run for one in results.runs_with_approvals(log)] == ["run-1"], \
        "offered on the strength of the approval, as before"

    storage.save_approved([judged()], store)
    run_log.note_a_store(1, "run-1", log)

    after = results.outcome_of("run-1", log, store)
    assert after.stored is True
    assert after.not_in_the_store == []


def test_a_decision_or_a_send_back_after_the_store_makes_the_run_not_stored(tmp_path):
    """After Back, a further decision or a send-back with no second store
    leaves the kept set older than the counts, so the latest store no longer
    stands; the finish time follows the reviewer's last act either way
    (found in review on 22 September: any store read as complete)."""
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    cohort = judged(a_paper("A cohort of drug X", doi="", pmid="222"))
    run_log.note_the_request(a_request(), "run-1", log)
    run_log.note_a_decision(judged(), "approved", "", "run-1", log)
    storage.save_approved([judged()], store)
    run_log.note_a_store(1, "run-1", log)
    assert results.outcome_of("run-1", log, store).stored is True

    run_log.note_a_decision(cohort, "rejected", "a cohort", "run-1", log)
    decided_after = results.outcome_of("run-1", log, store)
    assert decided_after.stored is False
    assert decided_after.finished == run_log.runs(log)[0].entries[-1]["at"]

    run_log.note_a_store(1, "run-1", log)
    assert results.outcome_of("run-1", log, store).stored is True

    run_log.note_a_revision(cohort, "exclude", "run-1", log)
    sent_back_after = results.outcome_of("run-1", log, store)
    assert sent_back_after.stored is False
    assert sent_back_after.waiting == 1
    assert sent_back_after.finished == run_log.runs(log)[0].entries[-1]["at"]


def test_a_paper_sent_back_is_waiting_until_the_reviewer_decides_again(tmp_path):
    """The send-back makes the earlier decision history: until a new one
    arrives the paper is neither approved nor left out (FR-19). Read past,
    the "judged again" line left a rejected paper counted as rejected with
    its old note (found in review on 21 September)."""
    for first in ("rejected", "flagged"):
        log = tmp_path / f"{first}.jsonl"
        store = tmp_path / f"{first}.sqlite3"
        run_log.note_the_request(a_request(), "run-1", log)
        run_log.note_a_decision(judged(), first, "is it a trial?", "run-1", log)
        run_log.note_a_revision(judged(), "include", "run-1", log)

        waiting = results.outcome_of("run-1", log, store)
        assert (waiting.approved, waiting.flagged, waiting.rejected) == (0, 0, 0)
        assert waiting.waiting == 1
        assert waiting.left_out == [], "the old note does not stand for it"

        run_log.note_a_decision(judged(), "approved", "", "run-1", log)
        storage.save_approved([judged()], store)

        decided = results.outcome_of("run-1", log, store)
        assert (decided.approved, decided.waiting) == (1, 0)
        assert [one.title for one in decided.papers] == ["A trial of drug X"]


def test_the_stored_publication_status_travels_to_the_results(tmp_path):
    """A retraction the hand-over stored must reach the list, not only the
    review page (FR-09; found in review on 21 September)."""
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    retracted = judged(a_paper(publication_status="Retracted after publication"))
    run_log.note_the_request(a_request(), "run-1", log)
    run_log.note_a_decision(retracted, "approved", "", "run-1", log)
    storage.save_approved([retracted], store)

    [paper] = results.outcome_of("run-1", log, store).papers

    assert paper.publication_status == "Retracted after publication"


def test_a_paper_with_no_doi_is_matched_by_its_pubmed_identifier(tmp_path):
    log = tmp_path / "run_log.jsonl"
    store = tmp_path / "approved_records.sqlite3"
    paper = a_paper(doi="", pmid="12345678")
    run_log.note_the_request(a_request(), "run-1", log)
    run_log.note_a_decision(judged(paper), "approved", "", "run-1", log)
    storage.save_approved([judged(paper)], store)

    outcome = results.outcome_of("run-1", log, store)

    assert [one.pmid for one in outcome.papers] == ["12345678"]
    assert outcome.not_in_the_store == []


def test_a_run_the_log_does_not_hold_has_no_outcome(tmp_path):
    log, store = a_recorded_run(tmp_path)

    assert results.outcome_of("run-9", log, store) is None


def test_only_runs_with_an_approval_are_offered_newest_first(tmp_path):
    """A run with nothing approved has no list to show. Written as lines
    rather than through the log's functions, so the times are chosen."""
    log = tmp_path / "run_log.jsonl"
    lines = [
        {"run": "old", "at": "2026-09-16T09:00:00+00:00", "event": "run started",
         "question": "first"},
        {"run": "old", "at": "2026-09-16T09:01:00+00:00",
         "event": "reviewer decision", "paper": "1", "decision": "approved"},
        {"run": "empty", "at": "2026-09-16T10:00:00+00:00", "event": "run started",
         "question": "second"},
        {"run": "empty", "at": "2026-09-16T10:01:00+00:00",
         "event": "reviewer decision", "paper": "2", "decision": "rejected"},
        {"run": "new", "at": "2026-09-17T09:00:00+00:00", "event": "run started",
         "question": "third"},
        {"run": "new", "at": "2026-09-17T09:01:00+00:00",
         "event": "reviewer decision", "paper": "3", "decision": "approved"},
    ]
    log.write_text("".join(json.dumps(line) + "\n" for line in lines),
                   encoding="utf-8")

    assert [one.run for one in results.runs_with_approvals(log)] == ["new", "old"]


def test_a_search_line_says_what_the_log_did_not_record():
    lines = results.searches_made([
        {"event": "search", "source": "pubmed", "query": "drug X"},
        {"event": "search", "source": "semantic_scholar", "query": "drug Y",
         "result": "failed", "papers found": 0},
    ])

    assert lines == [
        "pubmed: drug X, result not recorded, papers found not recorded",
        "semantic_scholar: drug Y, failed, 0 papers found",
    ]


def test_every_row_reads_the_store_named_at_the_time_it_is_called(tmp_path, monkeypatch):
    """The screen names no store, so a test that moves the default file has
    to be moving the file this reads."""
    store = tmp_path / "approved_records.sqlite3"
    storage.save_approved([judged()], store)
    monkeypatch.setattr(storage, "DEFAULT_STORE", store)

    assert [row["title"] for row in storage.every_row()] == ["A trial of drug X"]


def test_reading_a_store_that_does_not_exist_does_not_create_one(tmp_path):
    """The live run starts from no store, and a look at an earlier run's
    outcome must not put an empty file where the absence is checked."""
    missing = tmp_path / "another.sqlite3"

    assert storage.every_row(missing) == []
    assert not missing.exists()
