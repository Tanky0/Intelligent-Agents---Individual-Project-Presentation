"""The real Planning Agent driven by the real graph (IT-01, DC-01, DC-03)."""
# Everywhere else these two are tested apart. The planner's own tests call it
# directly, and the graph's tests give it a stand-in that answers from a script.
# Both are useful and neither shows that the two fit together: a change to what
# the graph passes, or to what the agent expects, would leave both sets of tests
# passing.
#
# So this replaces only the two edges the application does not own, the language
# model and the search services, and lets everything between them be the real
# thing. What it shows is that a run reaches the hand-over with the papers a
# search returned and the decisions the agent made. What it does not show is
# whether the model chooses well, because the model is not here.

from research_agent import graph
from research_agent.agents import asking, curator, planner
from research_agent.llm import OutOfBudget
from research_agent.schemas import (
    NextMove,
    ResearchRequest,
    Screening,
    ScreeningList,
    Subgoal,
    SubgoalList,
)
from research_agent.sources.pubmed import PubmedRecord


class ModelThatAnswersInOrder:
    """Stands in for the model, answering each request in turn."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.questions_asked = []

    def __call__(self, question, schema, budget=None):
        self.questions_asked.append(question)
        if budget is not None:
            budget = budget.with_one_model_request_used()
        return self.answers.pop(0), budget


class CuratorThatRecordsWhatItWasGiven:
    """Stands in for the Curation Agent, which is not built."""

    def __init__(self):
        self.papers_seen = 0
        self.buffer_was_ready = None

    def screen(self, state):
        self.papers_seen = len(state.results.papers)
        self.buffer_was_ready = state.results.ready
        return [], state.budget


def a_request():
    return ResearchRequest(
        question="Does drug X treat condition Y?",
        inclusion_criteria="trials reporting an outcome",
        exclusion_criteria="reviews and opinion pieces")


def test_the_real_planner_runs_a_whole_search_through_the_real_graph(monkeypatch):
    """One plan, two searches, two decisions, and a hand-over."""
    # The plan asks for two searches. The first is told to continue, the second
    # to finish. The planner and graph are real. The model, PubMed and the
    # not-yet-built Curation Agent are stand-ins.
    model = ModelThatAnswersInOrder([
        SubgoalList(subgoals=[
            Subgoal(purpose="find trials", source="pubmed", query="drug X trials"),
            Subgoal(purpose="find harms", source="pubmed", query="drug X harms"),
        ]),
        NextMove(move="continue", reason="worth looking at the second as well"),
        NextMove(move="finish", reason="enough has been found"),
    ])
    monkeypatch.setattr(asking.llm, "ask_for", model)
    monkeypatch.setattr(planner.pubmed, "search", lambda query, limit, **years: ["1"])
    monkeypatch.setattr(planner.pubmed, "fetch", lambda found: (
        [PubmedRecord(pmid="1", title="A trial of drug X", abstract="It worked.",
                      text_source="abstract")], []))

    curator = CuratorThatRecordsWhatItWasGiven()
    finished = graph.run(a_request(), planner, curator)

    # Both searches return the record with PubMed identifier 1, so the two
    # searches found the same paper. The hand-over combines them and keeps
    # both searches against it, which is what ID-14 holds the origins for.
    assert len(finished.results.papers) == 1, "the same paper found twice is one paper"
    assert [(o.source, o.query) for o in finished.results.papers[0].origins] == [
        ("pubmed", "drug X harms"), ("pubmed", "drug X trials")], \
        "and both searches are recorded against it"
    assert len(finished.results.papers_before_deduplication) == 2, \
        "what the searches returned is kept as it arrived"
    assert finished.budget.papers_sent_to_curation == 1, \
        "one place spent, not one for each search that found it"
    assert finished.results.ready is True, "the hand-over was signalled"
    assert curator.buffer_was_ready is True, "and curation waited for it"
    assert curator.papers_seen == 1, "the curator read what the planner wrote"

    assert [d.move for d in finished.decisions] == ["continue", "finish"]
    assert finished.decisions[0].subgoal.query == "drug X trials"
    assert finished.decisions[1].reason == "enough has been found"
    assert [o.result for o in finished.results.outcomes] == ["finished", "finished"]

    assert len(model.questions_asked) == 3, "one plan and two moves"
    assert "A trial of drug X" in model.questions_asked[1], \
        "the second question showed the paper the first search returned"


def test_a_service_failure_travels_from_the_source_to_the_run(monkeypatch):
    """A search that fails is reported and the run carries on to the end."""
    # This is the path FR-05 exists for, and it crosses every part: the source
    # reports a failure, the agent turns it into an outcome, the graph keeps it,
    # and the hand-over still happens.
    from research_agent.sources.http import FetchError

    problem = FetchError("pubmed", "https://example.test", 503, "service busy", 3)
    model = ModelThatAnswersInOrder([
        SubgoalList(subgoals=[Subgoal(purpose="find trials", source="pubmed",
                                      query="drug X trials")]),
        NextMove(move="finish", reason="nothing came back, so there is no more to do"),
    ])
    monkeypatch.setattr(asking.llm, "ask_for", model)
    monkeypatch.setattr(planner.pubmed, "search", lambda query, limit, **years: problem)
    monkeypatch.setattr(planner.pubmed, "fetch", lambda found: ([], [problem]))

    finished = graph.run(a_request(), planner, CuratorThatRecordsWhatItWasGiven())

    assert finished.results.outcomes[0].result == "failed"
    assert "service busy" in finished.results.outcomes[0].reason
    assert finished.results.ready is True, "the run still handed over what it had"
    assert finished.needs_human_attention is False, \
        "a service failing is reported, not a reason to stop the run"
    assert "Search status: failed" in model.questions_asked[1], \
        "and the agent was told the search failed rather than found nothing"


def test_both_real_agents_run_one_request_end_to_end(monkeypatch):
    """The planner and the curator in the same run, with the model and search
    service replaced.
    """
    # Everywhere else one of the two agents is a stand-in, so nothing had shown
    # that the real screening step fits what the graph hands it, or that what
    # screening spends reaches the state.
    model = ModelThatAnswersInOrder([
        SubgoalList(subgoals=[Subgoal(purpose="find trials", source="pubmed",
                                      query="drug X trials")]),
        NextMove(move="finish", reason="enough has been found"),
        ScreeningList(screenings=[Screening(
            paper_number=1, decision="include", relevance=8,
            reason="it reports an outcome for drug X",
            summary="a trial of drug X in condition Y",
            extract="It worked.")]),
    ])
    monkeypatch.setattr(asking.llm, "ask_for", model)
    monkeypatch.setattr(planner.pubmed, "search", lambda query, limit, **years: ["1"])
    monkeypatch.setattr(planner.pubmed, "fetch", lambda found: (
        [PubmedRecord(pmid="1", title="A trial of drug X", abstract="It worked.",
                      text_source="abstract")], []))

    finished = graph.run(a_request(), planner, curator)

    assert [judged.screening.decision for judged in finished.screened] == ["include"]
    assert finished.screened[0].paper.pmid == "1", \
        "the judgement is against the paper the search returned"
    assert finished.budget.model_requests_used == 3, \
        "one plan, one move and one batch of screening, all counted"
    assert finished.budget.papers_sent_to_curation == 1
    assert not finished.needs_human_attention
    assert finished.stopped_because == ""


def test_a_run_that_cannot_afford_to_screen_stops_without_asking_for_a_person(
        monkeypatch):
    """Running out at screening is the limit working, not a fault (ID-15)."""
    # The papers are still in the buffer and the reason is recorded. Reported as
    # needing a person, it would send somebody looking for a fault that is not
    # there.
    model = ModelThatAnswersInOrder([
        SubgoalList(subgoals=[Subgoal(purpose="find trials", source="pubmed",
                                      query="drug X trials")]),
        NextMove(move="finish", reason="enough has been found"),
        OutOfBudget("this run has used the 18 model requests it was allowed"),
    ])
    monkeypatch.setattr(asking.llm, "ask_for", model)
    monkeypatch.setattr(planner.pubmed, "search", lambda query, limit, **years: ["1"])
    monkeypatch.setattr(planner.pubmed, "fetch", lambda found: (
        [PubmedRecord(pmid="1", title="A trial of drug X", abstract="It worked.",
                      text_source="abstract")], []))

    finished = graph.run(a_request(), planner, curator)

    assert "18 model requests" in finished.stopped_because
    assert not finished.needs_human_attention, \
        "an allowance spent is not something a person has to look at"
    assert finished.attention_reason == ""
    assert finished.screened == [], "nothing was screened"
    assert finished.results.papers, "and what was found is still in the buffer"
