"""Checks the script that measures the agents against the CLEF review
(TF-03, DS-01 to DS-04).
"""
# The script spends model requests and calls PubMed, so what is tested here is
# what it decides around those calls, with the calls themselves replaced: the
# draw, the checks on the labels and the prepared records, the caps, the
# refusal of demonstration mode, what a stopped retrieval still reports, what
# a repeat run refuses before spending, and how the figures and the hand-read
# sample are worked out. The command steps are driven through main with a
# stand-in curator or graph that counts how often it was called, so a refused
# step is shown to spend nothing rather than assumed to.
#
# What is not tested is a live step end to end: those are the runs given
# their days on 14 and 16 September, and they are made by hand.

import copy
import importlib.util
import json
import pathlib

import pytest

from research_agent.schemas import (
    Outcome,
    PaperRecord,
    ResearchRequest,
    ResearchState,
    ResultsBuffer,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
)
from research_agent.sources.pubmed import PubmedRecord

SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "measure_against_clef.py")


@pytest.fixture
def measuring():
    """The script as it stands on disk, loaded from its path."""
    # scripts/ is not a package, so it is loaded rather than imported, the way
    # the sample-drawing script is.
    spec = importlib.util.spec_from_file_location("measure_against_clef", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sandbox(measuring, monkeypatch, tmp_path):
    """The script pointed at a temporary labels file and private folder."""
    # The committed labels are fifty rows; the command tests need two, so the
    # stand-ins can answer for every paper without fifty of anything, and the
    # sample size the script checks the labels against is set to two with
    # them. Demonstration mode is set off here rather than left to the
    # settings file, because the spending steps refuse it.
    monkeypatch.setattr(measuring, "LABELS", tmp_path / "labels.csv")
    monkeypatch.setattr(measuring, "PRIVATE_DIR", tmp_path / "private")
    monkeypatch.setattr(measuring, "RECORDS", tmp_path / "private" / "records.json")
    monkeypatch.setattr(measuring, "FINGERPRINT", tmp_path / "private" / "records.sha256")
    monkeypatch.setattr(measuring, "SAMPLE_SIZE", 2)
    monkeypatch.setattr(measuring.settings, "demo_mode", False)
    measuring.write_labels([{"pmid": "1", "include": "include"},
                            {"pmid": "2", "include": "exclude"}], measuring.LABELS)
    instructions = tmp_path / "instructions.json"
    instructions.write_text(json.dumps({"inclusion_criteria": "trials",
                                        "exclusion_criteria": "reviews"}), encoding="utf-8")
    measuring.instructions = instructions
    return measuring


def a_paper(pmid, abstract="It worked."):
    if abstract:
        text_source = "abstract"
    else:
        text_source = "metadata"
    return PaperRecord(title=f"Paper {pmid}", pmid=pmid, abstract=abstract,
                       origins=[SearchOrigin(source="pubmed", query="")],
                       text_source=text_source)


def judged(pmid, decision, relevance=5, extract="It worked."):
    return ScreenedPaper(
        paper=a_paper(pmid),
        screening=Screening(paper_number=1, decision=decision, relevance=relevance,
                            reason="because", summary="it worked", extract=extract))


def prepared(sandbox, papers=None):
    """Saves records for the sandbox's two papers, as the fetch step would."""
    if papers is None:
        papers = [a_paper("1"), a_paper("2")]
    sandbox.save_the_records(papers)


class CuratorThatAnswers:
    """Stands in for the Curation Agent, counting its calls."""
    # Answers include for every paper unless told otherwise, so a test can see
    # both that it was called and what it was given.

    def __init__(self, answer=None):
        self.calls = 0
        self.states = []
        self.answer = answer

    def screen(self, state):
        self.calls += 1
        self.states.append(state)
        if self.answer is not None:
            return self.answer, state.budget
        return [judged(paper.pmid, "include") for paper in state.results.papers], \
            state.budget.with_one_model_request_used()


class PubmedThatReturns:
    """Stands in for the PubMed fetch in the test that tries several failed
    preparations in turn.
    """
    # Each failure case needs its own fixed reply. A function written inside
    # the loop reads the loop's variable when it is called, not when it is
    # written, which is right only for as long as the call stays in the same
    # turn of the loop, and the style checker refuses that shape outright
    # (B023). A small object holding its own records is the plain way round
    # it.

    def __init__(self, records):
        self.records = records

    def fetch(self, pmids):
        return self.records, []


def a_topic_file(pids):
    return "Topic: CD011571 \n\nTitle: A review \n\nQuery: \n(x)[tiab]\n\nPids: \n" + \
        "\n".join(f"    {pid} " for pid in pids) + "\n"


def a_qrels_file(topic, labels):
    return "\n".join(f"{topic}     0  {pid}     {label}  " for pid, label in labels)


# The draw.

def test_the_draw_keeps_every_relevant_paper_and_fills_to_fifty_in_identifier_order(measuring):
    """The recall denominator is fixed before the run, and the order says nothing."""
    pids = [str(100 + n) for n in range(60)]
    relevant = set(pids[:15])
    rows = measuring.draw_the_fifty(pids, relevant)

    assert len(rows) == measuring.SAMPLE_SIZE
    assert {row["pmid"] for row in rows if row["include"] == "include"} == relevant
    assert [row["pmid"] for row in rows] == sorted((row["pmid"] for row in rows), key=int)
    assert rows == measuring.draw_the_fifty(pids, relevant), "the same draw every time"


def test_the_draw_refuses_more_relevant_papers_than_places(measuring):
    """Dropping a relevant paper quietly would move the denominator nobody could see."""
    pids = [str(n) for n in range(60)]
    with pytest.raises(ValueError, match="do not fit"):
        measuring.draw_the_fifty(pids, set(pids[:51]))


def test_the_collection_files_are_read_as_the_script_expects(measuring):
    """The topic file lists identifiers after "Pids:"; the qrels file marks
    relevance in its fourth column, for the topic asked about only."""
    assert measuring.pids_in(a_topic_file(["12", "345"])) == ["12", "345"]
    qrels = a_qrels_file("CD011571", [("12", "1"), ("345", "0")]) + "\n" + \
        a_qrels_file("CD000001", [("999", "1")])
    assert measuring.relevant_in(qrels, "CD011571") == {"12"}


def test_the_committed_labels_are_the_fifty_the_proposal_describes(measuring):
    """The proposal fixed fifty with fifteen included; a changed file would
    change every figure measured against it without any figure saying so."""
    identifiers = measuring.read_identifiers(measuring.LABELS)
    included = measuring.read_included(measuring.LABELS)
    assert len(identifiers) == 50 and len(set(identifiers)) == 50
    assert len(included) == 15
    assert all(pmid.isdigit() for pmid in identifiers)
    assert measuring.why_the_labels_are_not_the_fifty(identifiers) == ""


def test_a_labels_file_that_is_not_the_fifty_is_refused_before_any_step_reads_it(
        sandbox, monkeypatch):
    """The checks on the records and the reference compare against the labels,
    so a short or repeated file would pass them with a matching cache; it is
    refused first, and neither PubMed nor the model is asked."""
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    fetched = []
    monkeypatch.setattr(sandbox.pubmed, "fetch", lambda pmids: fetched.append(pmids))
    screen = ["screen", "--cap", "5", "--instructions", str(sandbox.instructions)]

    for rows, papers in (
            ([{"pmid": "1", "include": "include"}], [a_paper("1")]),
            ([{"pmid": "1", "include": "include"}, {"pmid": "1", "include": "include"}],
             [a_paper("1"), a_paper("1")]),
            ([{"pmid": "1", "include": "include"}, {"pmid": "x", "include": "exclude"}],
             [a_paper("1"), a_paper("x")])):
        sandbox.write_labels(rows, sandbox.LABELS)
        prepared(sandbox, papers)
        assert sandbox.main(screen) == 1
        assert sandbox.main(["fetch"]) == 1
    assert curator.calls == 0 and fetched == []


# The preparation.

def test_a_record_not_returned_or_without_a_title_is_reported_and_a_title_only_record_is_kept(
        measuring, monkeypatch):
    """The cases are different: a record PubMed did not return, or returned
    with no title, cannot be screened; one with a title and no abstract can."""
    def fetch(pmids):
        return [PubmedRecord(pmid="1", title="One", abstract="Text.", text_source="abstract"),
                PubmedRecord(pmid="2", title="Two", abstract="", text_source="metadata"),
                PubmedRecord(pmid="4", title=" ", abstract="Text.", text_source="abstract")], []
    monkeypatch.setattr(measuring.pubmed, "fetch", fetch)

    papers, unusable, without_abstract, _ = measuring.fetch_the_fifty(["1", "2", "3", "4"])

    assert [paper.pmid for paper in papers] == ["1", "2"]
    assert unusable == ["3 was not returned", "4 came back with no title"]
    assert without_abstract == ["2"]
    assert papers[1].text_source == "metadata"
    assert papers[0].origins[0].query == "", "no search stands in front of these papers"


def test_a_failed_fetch_removes_the_earlier_records_rather_than_leaving_them_usable(
        sandbox, monkeypatch):
    """Otherwise a screening after a failed preparation would read the old cache
    as though the preparation had succeeded.
    """
    # Two ways to fail: a record not returned, and a record returned with no
    # title, which the shared format would refuse after the old cache had been
    # left in place.
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    one_missing = [PubmedRecord(pmid="1", title="One", abstract="Text.", text_source="abstract")]
    one_untitled = [PubmedRecord(pmid="1", title="One", abstract="Text.", text_source="abstract"),
                    PubmedRecord(pmid="2", title="", abstract="Text.", text_source="abstract")]

    for returned in (one_missing, one_untitled):
        prepared(sandbox)
        monkeypatch.setattr(sandbox.pubmed, "fetch", PubmedThatReturns(returned).fetch)
        assert sandbox.main(["fetch"]) == 1
        assert not sandbox.RECORDS.exists() and not sandbox.FINGERPRINT.exists()
        assert "not prepared" in sandbox.why_the_records_are_not_ready(["1", "2"])
        assert sandbox.main(["screen", "--cap", "5", "--instructions",
                             str(sandbox.instructions)]) == 1
    assert curator.calls == 0


def test_saved_records_that_are_not_the_fifty_are_refused_before_screening(
        sandbox, monkeypatch):
    """A cache from another draw, a short one, a repeated paper or an edited
    file would be screened as the fifty. Each is refused, and none costs a
    request."""
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    command = ["screen", "--cap", "5", "--instructions", str(sandbox.instructions)]

    assert sandbox.main(command) == 1, "nothing prepared"
    prepared(sandbox, [a_paper("2"), a_paper("1")])
    assert sandbox.main(command) == 1, "the wrong order"
    prepared(sandbox, [a_paper("1")])
    assert sandbox.main(command) == 1, "one short"
    prepared(sandbox, [a_paper("1"), a_paper("1")])
    assert sandbox.main(command) == 1, "a repeat"
    prepared(sandbox, [])
    assert sandbox.main(command) == 1, "empty"
    prepared(sandbox)
    sandbox.RECORDS.write_text("[]", encoding="utf-8")
    assert sandbox.main(command) == 1, "edited after the fingerprint was taken"
    assert curator.calls == 0


# The caps.

def test_a_cap_outside_a_step_bounds_is_refused_and_one_inside_is_not(measuring):
    """Below the baseline the planned sequence cannot be covered; above the
    ceiling a slip on the command line would spend a day's allowance."""
    assert "not started" in measuring.why_the_cap_is_refused(3, "retrieve")
    assert "at most 10" in measuring.why_the_cap_is_refused(11, "retrieve")
    assert "at most 18" in measuring.why_the_cap_is_refused(19, "screen")
    assert "at most 10" in measuring.why_the_cap_is_refused(11, "screen-again")
    for step in ("retrieve", "screen", "screen-again"):
        assert measuring.why_the_cap_is_refused(measuring.BASELINES[step], step) == ""
        assert measuring.why_the_cap_is_refused(measuring.CAPS[step], step) == ""
    assert measuring.BASELINES == {"retrieve": 4, "screen": 5, "screen-again": 5}
    assert measuring.CAPS == {"retrieve": 10, "screen": 18, "screen-again": 10}


def test_a_cap_over_the_ceiling_spends_nothing_through_the_command(sandbox, monkeypatch):
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    built = []
    monkeypatch.setattr(sandbox.graph, "build_graph",
                        lambda planner, curator: built.append(1))
    instructions = str(sandbox.instructions)

    assert sandbox.main(["retrieve", "--cap", "11", "--instructions", instructions]) == 1
    assert sandbox.main(["screen", "--cap", "19", "--instructions", instructions]) == 1
    assert sandbox.main(["screen", "--cap", "4", "--instructions", instructions]) == 1
    assert sandbox.main(["screen"]) == 1, "no cap and no instructions is refused too"
    assert curator.calls == 0 and built == []


def test_demonstration_mode_is_refused_by_every_step_that_would_spend(sandbox, monkeypatch):
    """In demonstration mode the application replays saved answers and counts
    each as a request, so a run would look complete and measure the replay."""
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    built = []
    monkeypatch.setattr(sandbox.graph, "build_graph",
                        lambda planner, curator: built.append(1))
    instructions = str(sandbox.instructions)
    assert sandbox.main(["screen", "--cap", "5", "--instructions", instructions]) == 0
    reference = next(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    assert json.loads(reference.read_text(encoding="utf-8"))["mode"] == "live"
    calls_while_live = curator.calls
    monkeypatch.setattr(sandbox.settings, "demo_mode", True)

    assert sandbox.main(["retrieve", "--cap", "10", "--instructions", instructions]) == 1
    assert sandbox.main(["screen", "--cap", "5", "--instructions", instructions]) == 1
    assert sandbox.main(["screen-again", "--cap", "5", "--instructions", instructions,
                         "--reference", str(reference)]) == 1
    assert curator.calls == calls_while_live and built == []


# The runs, with the agents replaced.

def test_the_screening_run_gives_the_curator_the_papers_under_the_cap_and_no_label(
        measuring, monkeypatch, tmp_path):
    """Which papers the review's authors included is read to score the
    answer, never to ask for it (ID-07)."""
    curator = CuratorThatAnswers()
    monkeypatch.setattr(measuring.curator, "screen", curator.screen)
    instructions = tmp_path / "instructions.json"
    instructions.write_text(json.dumps({"inclusion_criteria": "trials",
                                        "exclusion_criteria": "reviews"}), encoding="utf-8")
    request = measuring.request_from(instructions)

    answer, _, identifier = measuring.screen_the_fifty(request, [a_paper("1"), a_paper("2")],
                                                       cap=7)

    state = curator.states[0]
    assert state.request.question == measuring.TITLE
    assert state.request.inclusion_criteria == "trials"
    assert state.budget.model_request_limit == 7
    assert [paper.pmid for paper in state.results.papers] == ["1", "2"]
    assert state.results.ready is True
    dumped = state.model_dump_json()
    assert "include" not in dumped and "exclude" not in dumped, \
        "nothing in the state says which papers are labelled"
    assert len(answer) == 2 and identifier


def test_a_complete_screening_is_written_under_its_run_and_a_second_attempt_beside_it(
        sandbox, monkeypatch):
    """Two attempts are two files; a retry must not overwrite the evidence of
    the attempt before it."""
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    command = ["screen", "--cap", "5", "--instructions", str(sandbox.instructions)]

    assert sandbox.main(command) == 0
    assert sandbox.main(command) == 0

    written = sorted(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    assert len(written) == 2 and curator.calls == 2
    saved = json.loads(written[0].read_text(encoding="utf-8"))
    assert saved["complete"] is True
    assert saved["model"] == sandbox.settings.gemini_model
    assert saved["request"]["inclusion_criteria"] == "trials"
    assert saved["records_fingerprint"] == sandbox.FINGERPRINT.read_text().strip()
    assert saved["model_requests_used"] == 1
    assert saved["figures"]["precision"] == 0.5 and saved["figures"]["recall"] == 1


def test_a_screening_that_did_not_answer_for_every_paper_is_a_failed_attempt(
        sandbox, monkeypatch):
    """The application does not promise a list is one judgement a paper, so
    the script checks; a short list is recorded as failed, with what it spent
    and what it was given, and no figure is taken from it."""
    prepared(sandbox)
    curator = CuratorThatAnswers(answer=[judged("1", "include")])
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)

    assert sandbox.main(["screen", "--cap", "5", "--instructions",
                         str(sandbox.instructions)]) == 1

    failed = list(sandbox.PRIVATE_DIR.glob("failed_*.json"))
    assert len(failed) == 1 and not list(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    saved = json.loads(failed[0].read_text(encoding="utf-8"))
    assert saved["complete"] is False
    assert saved["model"] and saved["request"]["question"] == sandbox.TITLE
    assert saved["records_fingerprint"] and "cap" in saved and "model_requests_used" in saved


def test_the_repeat_run_refuses_a_reference_it_cannot_compare_with_before_spending(
        sandbox, monkeypatch):
    """A reference that does not match the repeat is refused before any
    request is spent.
    """
    # Changed records, criteria or model; a reference that is not a complete
    # live screening of the fifty, or does not name its run; a judgement of
    # the right paper that is not in the shared format: each is refused, and
    # none costs a request.
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    assert sandbox.main(["screen", "--cap", "5", "--instructions",
                         str(sandbox.instructions)]) == 0
    reference = next(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    calls_after_the_first = curator.calls

    def again(reference_path, instructions=None):
        if instructions is None:
            instructions = sandbox.instructions
        return sandbox.main(["screen-again", "--cap", "5", "--instructions",
                             str(instructions), "--reference", str(reference_path)])

    # The records changed, with a matching new fingerprint.
    prepared(sandbox, [a_paper("1", "Other text."), a_paper("2")])
    assert again(reference) == 1
    prepared(sandbox)
    # The criteria changed.
    other = sandbox.instructions.parent / "other.json"
    other.write_text(json.dumps({"inclusion_criteria": "something else",
                                 "exclusion_criteria": "reviews"}), encoding="utf-8")
    assert again(reference, other) == 1
    # The model changed.
    model_the_first_run_used = sandbox.settings.gemini_model
    monkeypatch.setattr(sandbox.settings, "gemini_model", "another-model")
    assert again(reference) == 1
    monkeypatch.setattr(sandbox.settings, "gemini_model", model_the_first_run_used)
    # A reference that is not a complete live screening of the fifty, or
    # whose judgements are the right papers but not the shared format.
    saved = json.loads(reference.read_text(encoding="utf-8"))
    score_out_of_range = copy.deepcopy(saved)
    score_out_of_range["judgements"][0]["screening"]["relevance"] = 11
    field_missing = copy.deepcopy(saved)
    del field_missing["judgements"][1]["screening"]["decision"]
    for broken in ("not json", json.dumps({"complete": False}),
                   json.dumps({**saved, "judgements": saved["judgements"][:1]}),
                   json.dumps({**saved, "judgements": saved["judgements"] * 2}),
                   json.dumps(score_out_of_range),
                   json.dumps(field_missing),
                   json.dumps({**saved, "run_id": ""}),
                   json.dumps({**saved, "mode": "demonstration"}),
                   json.dumps({key: value for key, value in saved.items() if key != "mode"})):
        bad = sandbox.PRIVATE_DIR / "bad.json"
        bad.write_text(broken, encoding="utf-8")
        assert again(bad) == 1, broken[:60]
    failed = sandbox.PRIVATE_DIR / "failed_x.json"
    failed.write_text(json.dumps({**saved, "complete": False}), encoding="utf-8")
    assert again(failed) == 1, "a failed attempt is not a reference"

    assert curator.calls == calls_after_the_first, "none of the refusals asked the model"


def test_the_repeat_run_with_a_valid_reference_counts_the_pairs_that_changed(
        sandbox, monkeypatch):
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    assert sandbox.main(["screen", "--cap", "5", "--instructions",
                         str(sandbox.instructions)]) == 0
    reference = next(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    changed = CuratorThatAnswers(answer=[judged("1", "exclude", 2), judged("2", "include", 5)])
    monkeypatch.setattr(sandbox.curator, "screen", changed.screen)

    assert sandbox.main(["screen-again", "--cap", "5", "--instructions",
                         str(sandbox.instructions), "--reference", str(reference)]) == 0

    saved = json.loads(next(sandbox.PRIVATE_DIR.glob("screening_again_*.json")).read_text())
    assert saved["reference"]["file"] == str(reference)
    assert saved["figures"]["differences"] == {
        "papers_compared": 2, "unmatched": [], "decision_differs": ["1"],
        "score_differs": ["1"]}


def test_the_retrieval_run_is_capped_and_kept_to_four_searches(measuring, monkeypatch):
    """graph.run takes no budget, so the state is built here with the cap on it."""
    seen = {}

    class GraphThatReturnsWhatItWasGiven:
        def invoke(self, state):
            seen["state"] = state
            return state
    monkeypatch.setattr(measuring.graph, "build_graph",
                        lambda planner, curator: GraphThatReturnsWhatItWasGiven())
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b",
                              results_per_search=measuring.RESULTS_PER_SEARCH)

    finished = measuring.retrieve(request, cap=10)

    assert seen["state"].budget.model_request_limit == 10
    assert seen["state"].step_limit == measuring.STEP_LIMIT == 4
    assert finished.request.results_per_search == 1000
    assert finished.run_id, "the run is named so its requests are filed under it"


def a_graph_that_ends_with(finished_state):
    class Graph:
        def invoke(self, state):
            return finished_state.model_copy(update={"run_id": state.run_id})
    return Graph()


def test_the_retrieval_report_is_written_even_when_the_private_folder_did_not_exist(
        sandbox, monkeypatch):
    """A spent run that could not write its report would be an allowance
    nobody could account for, so the folder is made before the run."""
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b")
    ended = ResearchState(request=request, results=ResultsBuffer(
        papers_before_deduplication=[a_paper("1")], ready=True,
        outcomes=[Outcome(subgoal=Subgoal(purpose="p", source="pubmed", query="q"),
                          result="finished", papers_found=1)]))
    monkeypatch.setattr(sandbox.graph, "build_graph",
                        lambda planner, curator: a_graph_that_ends_with(ended))
    assert not sandbox.PRIVATE_DIR.exists()

    assert sandbox.main(["retrieve", "--cap", "10", "--instructions",
                         str(sandbox.instructions)]) == 0
    assert sandbox.main(["retrieve", "--cap", "10", "--instructions",
                         str(sandbox.instructions)]) == 0

    reports = list(sandbox.PRIVATE_DIR.glob("retrieval_*.json"))
    assert len(reports) == 2, "two attempts, two reports"
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["figures"]["status"] == "completed"
    assert saved["figures"]["relevant_found_identifiers"] == ["1"]
    assert saved["mode"] == "live"


def test_a_returned_paper_with_no_pubmed_identifier_is_reported_not_dropped(
        sandbox, monkeypatch):
    """The planner may search Semantic Scholar, whose papers can come back with no
    PubMed identifier.
    """
    # Such a paper cannot be matched to the labels; the report says so and is
    # still written, rather than the run ending after its requests were spent
    # with nothing to show.
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b")
    from_elsewhere = PaperRecord(
        title="Found by the other service", paper_id="abc123", doi="10.1/x",
        origins=[SearchOrigin(source="semantic scholar", query="q")])
    ended = ResearchState(request=request, results=ResultsBuffer(
        papers_before_deduplication=[a_paper("1"), from_elsewhere], ready=True,
        outcomes=[Outcome(subgoal=Subgoal(purpose="p", source="semantic scholar", query="q"),
                          result="finished", papers_found=2)]))
    monkeypatch.setattr(sandbox.graph, "build_graph",
                        lambda planner, curator: a_graph_that_ends_with(ended))

    assert sandbox.main(["retrieve", "--cap", "10", "--instructions",
                         str(sandbox.instructions)]) == 0

    figures = json.loads(next(sandbox.PRIVATE_DIR.glob("retrieval_*.json")).read_text())["figures"]
    assert figures["papers_returned"] == 1 and figures["returned_identifiers"] == ["1"]
    assert figures["papers_returned_without_pmid"] == 1
    assert figures["returned_without_pmid"] == [
        {"doi": "10.1/x", "paper_id": "abc123", "source": "semantic scholar"}]


def test_a_run_that_stopped_with_only_a_paper_without_a_pubmed_identifier_is_partial_not_failed(
        sandbox, monkeypatch):
    """Failed means nothing came back. A paper without a PubMed identifier
    came back, so the run is partial, with its stop reason and the paper's
    identifiers kept in the report; the report is still written."""
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b")
    from_elsewhere = PaperRecord(
        title="Found by the other service", paper_id="abc123",
        origins=[SearchOrigin(source="semantic scholar", query="q")])
    stopped = ResearchState(request=request, stopped_because="the allowance is spent",
                            results=ResultsBuffer(papers=[from_elsewhere], outcomes=[]))
    monkeypatch.setattr(sandbox.graph, "build_graph",
                        lambda planner, curator: a_graph_that_ends_with(stopped))

    assert sandbox.main(["retrieve", "--cap", "10", "--instructions",
                         str(sandbox.instructions)]) == 1, "partial is not completed"

    figures = json.loads(next(sandbox.PRIVATE_DIR.glob("retrieval_*.json")).read_text())["figures"]
    assert figures["status"] == "partial"
    assert figures["stopped_because"] == "the allowance is spent"
    assert figures["papers_returned"] == 0 and figures["papers_returned_without_pmid"] == 1
    assert figures["returned_without_pmid"][0]["paper_id"] == "abc123"


# The figures.

def test_a_run_that_stopped_before_the_hand_over_still_counts_the_papers_it_returned(
        measuring):
    """The graph moves the returned papers at the hand-over; a run that
    stopped on a spent budget never moved them, and the figure would have
    read zero from the wrong field."""
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b")
    stopped = ResearchState(request=request, stopped_because="the allowance is spent",
                            results=ResultsBuffer(papers=[a_paper("1"), a_paper("2")],
                                                  outcomes=[]))

    figures = measuring.retrieval_figures(stopped, included={"1", "9"})

    assert figures["papers_returned"] == 2 and figures["relevant_found"] == 1
    assert figures["status"] == "partial" and figures["handed_over"] is False
    assert figures["stopped_because"] == "the allowance is spent"


def test_a_run_that_stopped_before_any_paper_came_back_is_a_failed_measurement(measuring):
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b")
    stopped = ResearchState(request=request, needs_human_attention=True,
                            attention_reason="the plan could not be produced")

    figures = measuring.retrieval_figures(stopped, included={"1"})

    assert figures["status"] == "failed" and figures["papers_returned"] == 0
    assert figures["attention_reason"] == "the plan could not be produced"


def test_retrieval_names_short_and_failed_searches_with_their_reasons(measuring):
    request = ResearchRequest(question="q", inclusion_criteria="a", exclusion_criteria="b",
                              results_per_search=2)
    outcomes = [
        Outcome(subgoal=Subgoal(purpose="p", source="pubmed", query="full"),
                result="finished", papers_found=2),
        Outcome(subgoal=Subgoal(purpose="p", source="pubmed", query="half"),
                result="partial", papers_found=1, reason="one group could not be fetched"),
        Outcome(subgoal=Subgoal(purpose="p", source="pubmed", query="broken"),
                result="failed", reason="service busy"),
    ]
    finished = ResearchState(request=request, results=ResultsBuffer(
        papers_before_deduplication=[a_paper("1"), a_paper("2"), a_paper("3")],
        outcomes=outcomes, ready=True))

    figures = measuring.retrieval_figures(finished, included={"1", "3", "9"})

    assert figures["relevant_found"] == 2 and figures["retrieval"] == pytest.approx(2 / 3)
    assert figures["status"] == "partial"
    assert "truncated" in figures["searches"][0]["note"]
    assert figures["searches"][1]["note"] == "partial: one group could not be fetched"
    assert figures["searches"][2]["note"] == "failed: service busy"


def test_precision_has_no_value_when_nothing_is_judged_include(measuring):
    """A denominator of nothing is said, not divided by."""
    figures = measuring.screening_figures(
        [judged("1", "exclude"), judged("2", "uncertain")], included={"1"})
    assert figures["precision"] is None
    assert figures["recall"] == 0
    assert figures["judged_uncertain"] == 1


def test_precision_and_recall_are_counted_against_the_labels(measuring):
    from research_agent.processing.validation import with_the_extracts_checked
    judgements = with_the_extracts_checked([
        judged("1", "include"), judged("2", "include"),
        judged("3", "exclude", extract="not in the paper"), judged("4", "uncertain")])

    figures = measuring.screening_figures(judgements, included={"1", "3", "4"})

    assert figures["judged_include"] == 2
    assert figures["judged_include_and_labelled_include"] == 1
    assert figures["precision"] == 0.5
    assert figures["recall"] == pytest.approx(1 / 3)
    assert figures["extracts_found_in_the_paper"] == 3


def test_a_comparison_reports_a_paper_with_no_partner_rather_than_skipping_it(measuring):
    first = [judged("1", "include", 8), judged("2", "exclude", 2)]
    second = [judged("1", "include", 6), judged("3", "include", 7)]
    assert measuring.differences_between(first, second) == {
        "papers_compared": 1, "unmatched": ["3"], "decision_differs": [],
        "score_differs": ["1"]}


# The hand-read sample.

def test_the_audit_sample_is_five_of_each_decision_when_both_have_enough(measuring):
    judgements = [judged(str(n), "include") for n in range(6)] + \
        [judged(str(n), "exclude") for n in range(10, 17)]
    chosen = measuring.audit_sample(judgements)
    includes = [pmid for pmid in chosen if int(pmid) < 6]
    assert len(chosen) == 10 and len(includes) == 5
    assert chosen == measuring.audit_sample(judgements), "the same ten every time"


def test_a_shortfall_is_filled_from_the_other_decision_and_then_from_uncertain(measuring):
    judgements = [judged("1", "include"), judged("2", "include"), judged("3", "include")] + \
        [judged(str(n), "exclude") for n in range(10, 14)] + \
        [judged(str(n), "uncertain") for n in range(20, 25)]
    chosen = measuring.audit_sample(judgements)
    assert len(chosen) == 10
    assert {"1", "2", "3"} <= set(chosen)
    assert {"10", "11", "12", "13"} <= set(chosen)
    assert sum(1 for pmid in chosen if int(pmid) >= 20) == 3


def test_fewer_than_ten_judgements_are_all_read(measuring):
    judgements = [judged("1", "include"), judged("2", "exclude")]
    assert sorted(measuring.audit_sample(judgements)) == ["1", "2"]


def test_the_audit_sample_is_read_from_a_complete_screening_only(sandbox, monkeypatch):
    prepared(sandbox)
    curator = CuratorThatAnswers()
    monkeypatch.setattr(sandbox.curator, "screen", curator.screen)
    assert sandbox.main(["screen", "--cap", "5", "--instructions",
                         str(sandbox.instructions)]) == 0
    reference = next(sandbox.PRIVATE_DIR.glob("screening_*.json"))
    failed = sandbox.PRIVATE_DIR / "failed_x.json"
    failed.write_text(json.dumps({**json.loads(reference.read_text()), "complete": False}),
                      encoding="utf-8")

    assert sandbox.main(["audit-sample", "--reference", str(failed)]) == 1
    assert sandbox.main(["audit-sample", "--reference", str(reference)]) == 0
    assert list(sandbox.PRIVATE_DIR.glob("audit_sample_*.json"))
