"""Tests for the wiring between the two agents (IT-01, IT-03)."""
# The loop is driven here with stand-in agents that follow a script instead of
# asking the model. Nothing calls a service and nothing calls Gemini, so the
# tests give the same answer every time and the daily model allowance is not
# spent on them.
#
# What is being tested is the wiring, not the agents. The stand-ins return
# answers already in the agreed formats, which is the promise the real agents
# will have to keep. That means these tests show the graph does the right thing
# with a correct answer and with a failure, and say nothing about whether the
# model gives good answers, which is what the functional tests are for.

import httpx
import pytest
import respx
from pydantic import ValidationError

from research_agent import graph
from research_agent.llm import OutOfBudget
from research_agent.processing import audit_log, run_log
from research_agent.schemas import (
    NextMove,
    Outcome,
    PaperRecord,
    ResearchRequest,
    ResearchState,
    RunBudget,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
    SubgoalList,
)
from research_agent.sources.crossref import CROSSREF_API, CrossrefRecord, PublicationStatus
from research_agent.sources.http import FetchError


def a_request():
    """The research request every test here starts from."""
    return ResearchRequest(
        question="Which studies evaluate drug X as a treatment for condition Y?",
        inclusion_criteria="original human research reporting an outcome",
        exclusion_criteria="reviews and opinion pieces",
    )


def a_subgoal(query="drug X AND condition Y"):
    return Subgoal(purpose="find trials of drug X", source="pubmed", query=query)


def a_paper(title="A trial of drug X"):
    return PaperRecord(title=title, origins=[SearchOrigin(source="pubmed")])


class StandInPlanner:
    """A Planning Agent that follows a script instead of asking the model."""
    # It is given the searches to plan and the moves to make, and works through
    # the moves in order. The last move is repeated if the graph asks for more,
    # so that a test does not have to count in advance how many times the loop
    # will come round.

    def __init__(self, subgoals, moves):
        self.subgoals = subgoals
        self.moves = list(moves)
        # What the graph asked for, in order, so a test can show the steps
        # happened in the order the design puts them in.
        self.asked_for = []
        self.papers_when_deciding = []
        self.batch_handed_over = []

    def plan_searches(self, state):
        self.asked_for.append("plan")
        return SubgoalList(subgoals=self.subgoals), state.budget

    def run_next_search(self, state):
        self.asked_for.append("search")
        searched = state.pending_searches[0]
        found = [a_paper(f"{searched.query} paper 0")]
        outcome = Outcome(subgoal=searched, result="finished",
                          papers_found=len(found))
        return found, outcome

    def decide_next_move(self, state, outcome, papers):
        self.asked_for.append("decide")
        # What the state held when the move was asked for, and what the search
        # itself returned. Kept so a test can show the results were already in
        # the state, and that the batch handed over is this search's own
        # (DC-03).
        self.papers_when_deciding.append(len(state.results.papers))
        self.batch_handed_over.append(len(papers))
        if len(self.moves) > 1:
            return self.moves.pop(0), state.budget
        return self.moves[0], state.budget


class StandInCurator:
    """A Curation Agent that judges every paper it is given the same way."""

    def __init__(self):
        # What the buffer looked like when screening was asked for. Kept so a
        # test can show curation did not start on a half-filled buffer.
        self.buffer_was_ready = None
        self.papers_seen = 0

    def screen(self, state):
        self.buffer_was_ready = state.results.ready
        self.papers_seen = len(state.results.papers)

        judged = []
        for number, paper in enumerate(state.results.papers, start=1):
            judged.append(ScreenedPaper(
                paper=paper,
                screening=Screening(
                    paper_number=number,
                    decision="include",
                    relevance=7,
                    reason="it reports an outcome for drug X",
                    summary="a trial of drug X in condition Y",
                    extract="drug X was given to 40 patients",
                ),
            ))
        return judged, state.budget


class FailedRequest:
    """Stands in for the ModelFailure that llm.py returns."""
    # A real ModelFailure is not built here, because the graph reads nothing from
    # it but the reason. What matters is that the answer is not the format the
    # graph expected and carries a sentence saying why.

    def __init__(self, reason):
        self.reason = reason


class AgentThatFails:
    """An agent whose every request comes back as a failure."""
    # It answers to everything both agents are asked for, so it can stand in for
    # either of them, and always gives back the same failure.

    def __init__(self, reason="the daily allowance is spent"):
        self.failure = FailedRequest(reason)

    def plan_searches(self, state):
        return self.failure, state.budget

    def run_next_search(self, state):
        return self.failure

    def decide_next_move(self, state, outcome, papers):
        return self.failure, state.budget

    def screen(self, state):
        return self.failure, state.budget


def test_the_planner_fills_the_buffer_and_the_curator_reads_it():
    """The hand-over the design draws between the two agents (IT-01, FR-17)."""
    # One agent writes papers into the buffer, ready is set, and only then does
    # the other agent read them. Curation starting on a half-filled buffer would
    # report on a search that was still running, so the state of ready when
    # screening was asked for is checked as well as the papers themselves.
    planner = StandInPlanner([a_subgoal("first"), a_subgoal("second")],
                             [NextMove(move="continue", reason="more to do")])
    curator = StandInCurator()

    finished = graph.run(a_request(), planner, curator)

    assert len(finished.results.papers) == 2, "one paper from each of the two searches"
    assert finished.results.ready is True, "the hand-over was signalled"
    assert curator.buffer_was_ready is True, "curation waited for the signal"
    assert curator.papers_seen == 2, "the curator read what the planner wrote"
    assert len(finished.screened) == 2, "and judged both of them"
    assert planner.asked_for == ["plan", "search", "decide", "search", "decide"], \
        "planned once, then searched and decided after each search (DC-01, DC-03)"


def test_the_next_move_is_decided_after_the_papers_are_in_the_state():
    """Observing comes before deciding, as Figure 2 draws it (DC-03)."""
    # Asked before the papers were added, the agent could see only the outcome,
    # which carries a count and a status rather than the papers themselves. A
    # move decided from a count is not a move decided from what the search
    # returned, and the reason-act-observe loop is the whole reason this project
    # is not a fixed list of steps.
    planner = StandInPlanner([a_subgoal("first"), a_subgoal("second")],
                             [NextMove(move="continue", reason="more to do")])

    graph.run(a_request(), planner, StandInCurator())

    assert planner.papers_when_deciding == [1, 2], \
        "each decision saw the paper the search had just returned"


def test_the_curator_is_given_what_came_back_out_of_the_json():
    """The hand-over between the agents is the JSON, not the record (FR-17)."""
    # The design calls the thing between the two agents a schema-validated JSON
    # buffer. Handing the Curation Agent the object the Planning Agent left in
    # the state would make that a description rather than a fact, so the buffer
    # is written out, read back and checked first. This shows the papers survived
    # that round trip, and that the file kept as evidence holds the same text.
    planner = StandInPlanner([a_subgoal()],
                             [NextMove(move="finish", reason="that is enough")])
    curator = StandInCurator()

    finished = graph.run(a_request(), planner, curator)

    assert curator.papers_seen == 1, "the curator was given the paper"
    written = list(graph.BUFFER_DIR.glob("buffer_*.json"))
    handed_over = finished.results.model_dump_json(indent=2)
    assert written[0].read_text(encoding="utf-8") == handed_over, \
        "the file holds the buffer that was handed over, not a second copy"


def test_a_search_that_returns_the_wrong_number_of_values_stops_the_run():
    """The papers and the outcome are expected as a pair (FR-05)."""
    # A tuple of any other length would be unpacked into the wrong names and stop
    # the run with a message about unpacking, which says nothing about the agent
    # that broke the agreement.

    class PlannerThatReturnsThreeThings(StandInPlanner):
        def run_next_search(self, state):
            searched = state.pending_searches[0]
            outcome = Outcome(subgoal=searched, result="finished", papers_found=0)
            return [], outcome, "something else"

    planner = PlannerThatReturnsThreeThings(
        [a_subgoal()], [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert "3 values" in finished.attention_reason


def test_the_buffer_is_written_out_when_the_hand_over_happens():
    """Figure 1 draws the buffer as a thing between the two agents (ID-10)."""
    # The code holds it inside the state of the run, so writing it out at the
    # hand-over is what produces the file the figure shows. conftest.py points
    # the folder at a temporary one, so this writes nothing into the evidence
    # kept for real runs.
    planner = StandInPlanner([a_subgoal()],
                             [NextMove(move="finish", reason="that is enough")])

    graph.run(a_request(), planner, StandInCurator())

    written = list(graph.BUFFER_DIR.glob("buffer_*.json"))
    assert len(written) == 1, "one buffer written for one run"
    assert "drug X AND condition Y paper 0" in written[0].read_text(encoding="utf-8"), \
        "the paper the search returned is in the file that was written"


def test_a_buffer_that_cannot_be_written_does_not_stop_the_run(monkeypatch):
    """A file that could not be saved must not throw away a finished run."""
    # The failure is collected instead, in the same way a reply that could not be
    # recorded is, so that the run report can say the buffer is missing rather
    # than the run simply ending (FR-05).
    # A folder cannot be created inside a file, so this makes the write fail
    # without depending on permissions, which differ between computers.
    blocked = graph.BUFFER_DIR.parent / "a file, not a folder"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("this is a file", encoding="utf-8")
    monkeypatch.setattr(graph, "BUFFER_DIR", blocked / "buffers")

    planner = StandInPlanner([a_subgoal()],
                             [NextMove(move="finish", reason="that is enough")])
    finished = graph.run(a_request(), planner, StandInCurator())

    assert len(finished.screened) == 1, "the run still finished"
    assert len(graph.buffer_failures) == 1, "and said the buffer was not written"

    # The log line names the run that lost the buffer. The run used to be
    # kept in one value for the whole process, so a second run starting
    # while the first was still writing would have taken the blame (R-19).
    from research_agent.processing import run_log
    failures = [line for line in run_log.read() if line["event"] == "failure"]
    assert len(failures) == 1
    assert failures[0]["what"] == "the buffer"
    assert failures[0]["run"] == finished.run_id


def test_a_lost_buffer_is_logged_under_its_own_run_when_two_runs_overlap(monkeypatch):
    """The run that lost the buffer is named, not the run that started last."""
    # The run used to be kept in one value for the whole process. Two runs at
    # once are two browser sessions, and the second to start replaced the
    # value, so a buffer the first run then failed to write was logged under
    # the second. The fault was found by reading and has not been seen in a
    # real run; this is the order that would show it. The first run is held
    # before its hand-over, which is where the buffer is written, until the
    # second run has begun, and then both fail to write. Each failure line has
    # to name its own run (R-19).
    #
    # Events with timeouts rather than sleeps, so the order is deliberate and
    # a run that never arrives fails the test rather than hanging it.
    import threading

    from research_agent.processing import run_log

    blocked = graph.BUFFER_DIR.parent / "a file, not a folder"
    blocked.parent.mkdir(parents=True, exist_ok=True)
    blocked.write_text("this is a file", encoding="utf-8")
    monkeypatch.setattr(graph, "BUFFER_DIR", blocked / "buffers")

    first_is_waiting = threading.Event()
    second_has_started = threading.Event()

    class FirstPlanner(StandInPlanner):
        def run_next_search(self, state):
            first_is_waiting.set()
            if not second_has_started.wait(timeout=5):
                raise RuntimeError("the second run never started")
            return super().run_next_search(state)

    class SecondPlanner(StandInPlanner):
        def plan_searches(self, state):
            second_has_started.set()
            return super().plan_searches(state)

    finished = {}

    def run_as(name, planner):
        finished[name] = graph.run(a_request(), planner, StandInCurator())

    a_move = [NextMove(move="finish", reason="that is enough")]
    first = threading.Thread(
        target=run_as, args=("first", FirstPlanner([a_subgoal()], a_move)))
    second = threading.Thread(
        target=run_as, args=("second", SecondPlanner([a_subgoal()], a_move)))
    first.start()
    assert first_is_waiting.wait(timeout=5), "the first run never reached its search"
    second.start()
    first.join(timeout=10)
    second.join(timeout=10)
    assert set(finished) == {"first", "second"}, "both runs finished"

    lost = [line for line in run_log.read()
            if line["event"] == "failure" and line["what"] == "the buffer"]
    assert len(lost) == 2, "each run lost its buffer"
    assert sorted(line["run"] for line in lost) == sorted(
        [finished["first"].run_id, finished["second"].run_id])


def test_a_judgement_that_is_not_a_judgement_is_refused_at_the_end_of_the_run():
    """The half of IT-03 the LangGraph trial could not show."""
    # LangGraph writes whatever a step hands back into the state without checking
    # it, which test_langgraph_spike.py measures. The graph can see that the
    # Curation Agent gave back a list, but not what is in the list, so anything
    # else in it would reach the reviewer as a judgement. Reading the whole
    # record at the end of the run is what refuses it (DC-05).

    class CuratorThatReturnsRubbish:
        def screen(self, state):
            return ["this is not a judgement"], state.budget

    planner = StandInPlanner([a_subgoal()],
                             [NextMove(move="finish", reason="that is enough")])

    with pytest.raises(ValidationError):
        graph.run(a_request(), planner, CuratorThatReturnsRubbish())


def test_a_record_corrupted_after_the_search_step_is_refused_at_the_hand_over():
    """The other half of IT-03: the whole-record check at the hand-over (DC-05)."""
    # This is the case test_langgraph_spike.py says it cannot show: a record
    # broken part way through, by changing something inside a list it already
    # holds. Here the Planning Agent, asked for its next move, empties the origins
    # of a paper the buffer already holds. Nothing before the hand-over refuses
    # that: the paper was built with its checks, so changing a list inside it
    # afterwards is not an assignment the format sees, and when LangGraph
    # builds the state for the next step it takes the buffer as it is rather
    # than reading it again. So a paper that cannot be traced to any search
    # reaches the hand-over, and reading the whole record there is what refuses
    # it, before the Curation Agent is given it as evidence.
    #
    # The refusal is required to name the record, "validation error for
    # ResearchState", because that is what reading the whole record says. With
    # that check removed from graph.py the same run is still refused, but later
    # and by accident, when deduplication rebuilds the paper and the format
    # refuses the copy ("validation error for PaperRecord"); that was tried
    # before this test was kept, and this test fails on it. A refusal that
    # depends on deduplication happening to rebuild records is not the boundary
    # the design promised.
    #
    # A corruption LangGraph does catch is worth recording beside it, so nobody
    # takes this check for the only one: a budget handed back by the planner
    # that was built without its checks and has overspent is refused when
    # LangGraph builds the state for the next step, before the hand-over, because
    # that value is read again rather than taken as it is.

    class PlannerThatBreaksAPaperItAlreadyReturned(StandInPlanner):
        def decide_next_move(self, state, outcome, papers):
            # The state given here shares its buffer with the one the graph
            # writes, so the change survives into the hand-over.
            state.results.papers[0].origins.clear()
            return NextMove(move="finish", reason="that is enough"), state.budget

    curator = StandInCurator()
    planner = PlannerThatBreaksAPaperItAlreadyReturned([a_subgoal()], [])

    with pytest.raises(ValidationError, match="validation error for ResearchState"):
        graph.run(a_request(), planner, curator)
    assert curator.buffer_was_ready is None, "the Curation Agent was never given the buffer"


def test_a_search_that_returns_something_that_is_not_a_paper_stops_the_run():
    """A bad record must not end the run with a stack trace (FR-05, DC-06)."""
    # The Planning Agent turns records from four services into one format, so a
    # record that does not fit is likelier to come from there than from anywhere
    # else. Left to raise, one bad record would lose every paper found before it.

    class PlannerThatReturnsRubbish(StandInPlanner):
        def run_next_search(self, state):
            searched = state.pending_searches[0]
            outcome = Outcome(subgoal=searched, result="finished", papers_found=1)
            return ["this is not a paper record"], outcome

    planner = PlannerThatReturnsRubbish(
        [a_subgoal()], [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert "did not match the agreed format" in finished.attention_reason
    assert finished.screened == [], "nothing was screened from a bad search"


def test_a_search_that_returns_no_papers_at_all_stops_the_run():
    """A pair of the right length can still hold the wrong thing (FR-05)."""
    # Nothing in place of the papers fails before the format sees it, so it is
    # caught together with a record of the wrong kind. Both mean the same thing
    # to whoever reads the run.

    class PlannerThatReturnsNothingForPapers(StandInPlanner):
        def run_next_search(self, state):
            searched = state.pending_searches[0]
            outcome = Outcome(subgoal=searched, result="finished", papers_found=0)
            return None, outcome

    planner = PlannerThatReturnsNothingForPapers(
        [a_subgoal()], [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert "did not match the agreed format" in finished.attention_reason


def test_a_search_that_returns_the_wrong_kind_of_outcome_stops_the_run():
    """The outcome is what the coverage report is built from (FR-05)."""
    # A search reporting on itself in some other shape would leave the report
    # with nothing to say about that subgoal.

    class PlannerThatReturnsNoOutcome(StandInPlanner):
        def run_next_search(self, state):
            return [a_paper()], "the search went fine"

    planner = PlannerThatReturnsNoOutcome(
        [a_subgoal()], [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert "did not match the agreed format" in finished.attention_reason


def test_an_agent_that_raises_is_not_quietly_turned_into_a_stopped_run():
    """Not everything should be caught, and this records which things are not."""
    # A request that could not be answered is an ordinary thing to report. An
    # agent raising is different: it is a fault in this application rather than
    # a service refusing, and catching it would hide the line that caused it
    # behind a sentence for the reviewer. The run ends loudly instead, which is
    # the same reasoning the hand-over check follows (DC-05).

    class PlannerThatBreaks(StandInPlanner):
        def plan_searches(self, state):
            raise RuntimeError("something in the agent went wrong")

    planner = PlannerThatBreaks([a_subgoal()],
                                [NextMove(move="finish", reason="unused")])

    with pytest.raises(RuntimeError):
        graph.run(a_request(), planner, StandInCurator())


def test_a_search_that_worked_is_kept_when_the_next_decision_fails():
    """A failure must not take the work before it (FR-05)."""
    # The search ran, the service answered, and the papers are real. If the
    # request that follows cannot be answered, the run has to stop, but stopping
    # is not a reason to throw away what was already retrieved. An earlier
    # version of this step returned only the failure, so a successful search
    # disappeared because the model call after it did not succeed.

    class PlannerThatCannotDecide(StandInPlanner):
        def decide_next_move(self, state, outcome, papers):
            return FailedRequest("the model could not be reached"), state.budget

    planner = PlannerThatCannotDecide(
        [a_subgoal()], [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert "could not be reached" in finished.attention_reason
    assert len(finished.results.papers) == 1, "the paper the search found is kept"
    assert len(finished.results.outcomes) == 1, "and what became of the subgoal"
    assert finished.steps_taken == 1, "and the step it used"


def test_no_move_is_asked_for_once_the_limit_is_reached():
    """A model request whose answer cannot be acted on is one wasted (OB-07)."""
    # The step that chooses where to go next reads the limit before it reads the
    # move, so at the limit the answer changes nothing. Asking anyway spends one
    # of the twenty requests a day, and a failed answer would then have to be
    # handled for no reason.
    planner = StandInPlanner([a_subgoal("first"), a_subgoal("second")],
                             [NextMove(move="continue", reason="more to do")])

    graph.build_graph(planner, StandInCurator()).invoke(
        ResearchState(request=a_request(), step_limit=1))

    assert planner.asked_for == ["plan", "search"], "no decision was asked for"


def test_every_move_is_written_down_including_one_that_dropped_a_subgoal():
    """The run report needs all of them, not the last one (FR-20)."""
    # An outcome records what a search did. It does not record what was decided
    # afterwards, so a subgoal the agent chose to drop leaves no trace there. And
    # the move kept for routing is overwritten by the next search, so a run of
    # several searches would end holding one reason and having lost the rest.
    planner = StandInPlanner(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="drop", reason="this line will not help"),
         NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert [decision.move for decision in finished.decisions] == ["drop", "finish"]
    assert finished.decisions[0].reason == "this line will not help", \
        "the reason for the dropped subgoal survived the search after it"
    assert finished.decisions[0].subgoal.query == "first", \
        "each decision says which search it was about"


def test_the_decision_is_given_only_the_papers_its_own_search_returned():
    """Two searches can use the same words, and joining them misleads the agent."""
    # The same query sent to both services, or a refinement that repeats one,
    # would otherwise hand the agent every paper matching those words while
    # telling it how many the current search returned. It would be judging one
    # search by another search's results.
    planner = StandInPlanner([a_subgoal("same words"), a_subgoal("same words")],
                             [NextMove(move="continue", reason="more to do")])

    graph.run(a_request(), planner, StandInCurator())

    assert planner.batch_handed_over == [1, 1],         "each decision saw one paper, its own"
    assert planner.papers_when_deciding == [1, 2], \
        "while the state still held everything found so far"


def test_a_stopped_run_does_not_keep_a_move_from_an_earlier_search():
    """The move kept for routing belongs to the search that has just run."""
    # It is cleared with each search and set again only when a new one is
    # decided. Without that, a run ending at the limit or on a failed request
    # would finish carrying the previous search's decision, and a reader would
    # take it for the decision about the last search. The lasting record is the
    # list of decisions; this field is only what the next step reads.

    class PlannerThatFailsTheSecondDecision(StandInPlanner):
        def decide_next_move(self, state, outcome, papers):
            self.asked_for.append("decide")
            if len(self.asked_for) > 3:
                return FailedRequest("the model could not be reached"), state.budget
            return (NextMove(move="continue", reason="the first search was fine"),
                    state.budget)

    planner = PlannerThatFailsTheSecondDecision(
        [a_subgoal("first"), a_subgoal("second")], [NextMove(
            move="continue", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.needs_human_attention is True
    assert finished.last_move is None, \
        "the first search's decision is not left standing for the second"
    assert len(finished.decisions) == 1, "the decision that was made is kept"
    assert finished.decisions[0].reason == "the first search was fine"


def test_a_refinement_searches_again_with_the_new_words():
    """Refining replaces the query and keeps what the search was for (FR-04)."""
    # The purpose and the service stay as they were, because refining means
    # asking the same question in different words rather than looking for
    # something else.
    planner = StandInPlanner(
        [a_subgoal("too broad")],
        [NextMove(move="refine", reason="too many unrelated results",
                  new_query="narrower words"),
         NextMove(move="finish", reason="that is better")])

    finished = graph.run(a_request(), planner, StandInCurator())

    searched = []
    for outcome in finished.results.outcomes:
        searched.append(outcome.subgoal.query)
    assert searched == ["too broad", "narrower words"], "the second search was the refined one"
    assert finished.results.outcomes[1].subgoal.purpose == "find trials of drug X"
    assert finished.results.outcomes[1].subgoal.source == "pubmed"


def test_a_plan_longer_than_the_limit_stops_and_shows_what_it_did_not_reach():
    """An agent with limited time has to stop and act on what it has (DC-06)."""
    # Six searches are planned and four steps are allowed, so two are never run.
    # They are left on the queue rather than thrown away, which is what lets the
    # coverage report say the run was cut short instead of it looking like a
    # search that simply found less (FR-05).
    planner = StandInPlanner([a_subgoal(f"search {number}") for number in range(6)],
                             [NextMove(move="continue", reason="more to do")])

    finished = ResearchState.model_validate(
        graph.build_graph(planner, StandInCurator()).invoke(
            ResearchState(request=a_request(), step_limit=4)))

    assert finished.steps_taken == 4, "the limit stopped it"
    assert finished.results.ready is True, "and it still handed over what it had"
    assert len(finished.pending_searches) == 2, \
        "the two it never reached are still there to be reported"


def test_refining_without_end_still_stops_at_the_limit():
    """Refining puts a search back on the queue, so it could go round for ever."""
    # The queue never empties here, and the agent never says finish, so nothing
    # but the limit can end this run. Bratman, Israel and Pollack (1988, p. 349)
    # give a plan the job of stopping an agent reconsidering without end, and the
    # limit is what does that when the agent will not.
    planner = StandInPlanner(
        [a_subgoal()],
        [NextMove(move="refine", reason="never satisfied", new_query="again")])

    finished = ResearchState.model_validate(
        graph.build_graph(planner, StandInCurator()).invoke(
            ResearchState(request=a_request(), step_limit=4)))

    assert finished.steps_taken == 4, "the limit stopped it"
    assert finished.results.ready is True, "and it still handed over what it had"


def test_the_limit_can_be_set_for_one_run_without_changing_the_default():
    """The interface has to be able to ask for a shorter run (OB-07)."""
    # A demonstration has twenty model requests for the whole day, so rehearsing
    # twice at the ordinary length may not be possible. Leaving the limit out
    # keeps whatever the record itself says, so the number stays written down in
    # one place.
    planner = StandInPlanner([a_subgoal(f"search {number}") for number in range(6)],
                             [NextMove(move="continue", reason="more to do")])

    shorter = graph.run(a_request(), planner, StandInCurator(), step_limit=2)

    assert shorter.steps_taken == 2, "it stopped at the limit it was given"
    assert shorter.step_limit == 2

    left_alone = graph.run(a_request(), StandInPlanner(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")]),
        StandInCurator())

    assert left_alone.step_limit == 12, "the record's own number when none is given"


def test_a_run_that_cannot_plan_stops_and_asks_for_a_person():
    """A model that will not answer must not look like a search that found
    nothing (DC-06).
    """
    # An empty plan and a failed plan produce the same empty buffer, and the
    # difference is the whole of what the reviewer needs to know.
    finished = graph.run(a_request(), AgentThatFails(), StandInCurator())

    assert finished.needs_human_attention is True
    assert "the daily allowance is spent" in finished.attention_reason
    assert finished.screened == [], "curation never ran"
    assert finished.results.ready is False, "and nothing was handed over"


def test_a_run_that_stops_for_a_person_writes_why_to_the_log():
    """The reason is in the state, which the screen shows, and in the log,
    which outlives the screen (FR-15).
    """
    # The live run of 17 September stopped on a refused model request and the
    # log ended at the search before it, so a reader of the log could not tell
    # whether the run had ended.
    finished = graph.run(a_request(), AgentThatFails(), StandInCurator())

    last = run_log.read()[-1]
    assert last["run"] == finished.run_id
    assert last["event"] == "failure"
    assert last["what"] == "the run needs a person"
    assert last["why"] == finished.attention_reason


def test_a_missing_saved_reply_says_so_rather_than_reporting_nothing():
    """In DEMO_MODE an answer that was never saved comes back as None."""
    # None carries no reason with it, so the graph has to supply one. Left as it
    # is, the run would stop with an empty sentence where the explanation should
    # be.

    class AgentWithNoSavedReply:
        def plan_searches(self, state):
            return None, state.budget

    finished = graph.run(a_request(), AgentWithNoSavedReply(), StandInCurator())

    assert finished.needs_human_attention is True
    assert "no saved answer" in finished.attention_reason


def test_screening_that_fails_stops_the_run_rather_than_storing_nothing():
    """A failure at the last step must not look like papers nobody included."""
    # The papers are already in the buffer by then, so a run that ended quietly
    # would show a full buffer and no judgements, which reads as a screening that
    # excluded everything (FR-05).
    planner = StandInPlanner([a_subgoal()],
                             [NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, AgentThatFails("the model refused"))

    assert finished.needs_human_attention is True
    assert "the model refused" in finished.attention_reason
    assert len(finished.results.papers) == 1, "what was found is still there"
    assert finished.screened == [], "and nothing was invented in place of a judgement"


@pytest.mark.parametrize(
    "move, expected",
    [("continue", "search"), ("drop", "search"), ("refine", "refine"),
     ("finish", "hand over")])
def test_each_of_the_four_moves_leads_somewhere(move, expected):
    """FR-04 asks for four moves, so each one needs somewhere to go."""
    # parametrize runs this once for each move, so a move with no destination is
    # reported against the move that has none rather than as one failure covering
    # all four.
    state = ResearchState(
        request=a_request(),
        pending_searches=[a_subgoal()],
        last_move=NextMove(move=move, reason="a reason", new_query="new words"),
    )

    assert graph.after_searching(state) == expected


def test_continuing_with_nothing_left_to_search_hands_over():
    """Continue means carry on, and there is nothing to carry on with here."""
    # Left to loop, the run would search an empty queue and fail on a list with
    # nothing in it.
    state = ResearchState(
        request=a_request(),
        last_move=NextMove(move="continue", reason="there may be more"),
    )

    assert graph.after_searching(state) == "hand over"


def test_what_a_run_spent_is_written_back_into_its_state():
    """The budget belongs to the run, not to the step that spent from it."""
    # Each step is handed the one the run is carrying and gives back what is
    # left. A step that kept its own would be a limit on that step, and the run
    # would spend the allowance once for every step it has.

    class PlannerThatSpends(StandInPlanner):
        def plan_searches(self, state):
            self.asked_for.append("plan")
            return (SubgoalList(subgoals=self.subgoals),
                    state.budget.with_one_model_request_used())

        def decide_next_move(self, state, outcome, papers):
            self.asked_for.append("decide")
            return (self.moves[0],
                    state.budget.with_one_model_request_used())

    planner = PlannerThatSpends([a_subgoal()],
                                [NextMove(move="finish", reason="enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.budget.model_requests_used == 2, \
        "the plan and the one decision each spent a request"
    assert not finished.needs_human_attention


def test_a_run_that_spends_its_allowance_stops_and_says_so():
    """Running out is the limit working, not a fault."""
    # The run keeps what it found and records why it stopped. Reported as
    # needing a person, it would send somebody looking for a fault that is not
    # there.

    class PlannerWithNothingLeft(StandInPlanner):
        def decide_next_move(self, state, outcome, papers):
            self.asked_for.append("decide")
            return (OutOfBudget("this run has used the 18 model requests it "
                                "was allowed"), state.budget)

    planner = PlannerWithNothingLeft([a_subgoal()],
                                     [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert "18 model requests" in finished.stopped_because
    assert not finished.needs_human_attention, \
        "an allowance spent is not something a person has to look at"
    assert finished.attention_reason == ""
    assert finished.results.papers, "what the search found is kept"

    # In the log under the words the coverage report uses for this kind of
    # stop, so a reader meets one name for it in both places.
    last = run_log.read()[-1]
    assert last["run"] == finished.run_id
    assert last["what"] == "the run stopped early"
    assert last["why"] == finished.stopped_because


def test_a_run_that_cannot_plan_within_its_allowance_stops_the_same_way():
    """The other place a run can run out, before any search has happened."""

    class PlannerWithNothingLeft(StandInPlanner):
        def plan_searches(self, state):
            self.asked_for.append("plan")
            return (OutOfBudget("this run has used the 18 model requests it "
                                "was allowed"), state.budget)

    planner = PlannerWithNothingLeft([a_subgoal()],
                                     [NextMove(move="finish", reason="unused")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert "18 model requests" in finished.stopped_because
    assert not finished.needs_human_attention
    assert planner.asked_for == ["plan"], "no search was run"


def test_a_run_says_whether_each_extract_was_found_in_its_paper(monkeypatch):
    """A run ends with the question answered rather than left open (SC-02)."""
    # Left as None, a check that never ran would look the same as one that ran
    # and found nothing, and the reviewer could not tell which paper had been
    # looked at.
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)

    class PlannerWhosePapersHaveAbstracts(StandInPlanner):
        def run_next_search(self, state):
            self.asked_for.append("search")
            searched = state.pending_searches[0]
            found = [PaperRecord(
                title=f"{searched.query} trial",
                abstract="Forty patients were given drug X and improved.",
                text_source="abstract",
                origins=[SearchOrigin(source="pubmed", query=searched.query)])]
            return found, Outcome(subgoal=searched, result="finished", papers_found=1)

    class CuratorQuotingOnePaperAndInventingForTheOther(StandInCurator):
        def screen(self, state):
            judged, budget = super().screen(state)
            judged[0].screening.extract = "Forty patients were given drug X"
            judged[1].screening.extract = "Every patient recovered within a week"
            return judged, budget

    planner = PlannerWhosePapersHaveAbstracts(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="continue", reason="more to do")])

    finished = graph.run(
        a_request(), planner, CuratorQuotingOnePaperAndInventingForTheOther())

    found = {item.screening.extract: item.extract_found for item in finished.screened}
    assert found == {
        "Forty patients were given drug X": True,
        "Every patient recovered within a week": False,
    }


def test_the_run_stores_the_screened_papers_in_rank_order(monkeypatch):
    """The reviewer reads what the run stored, so the run stores it ranked."""
    # Kept in the order the papers were handed over, the ranking would have to be
    # worked out again by whatever displayed it, and two things showing the same
    # run could show it differently (FR-11, ID-08).
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)

    class CuratorThatScoresThemDifferently(StandInCurator):
        def screen(self, state):
            judged, budget = super().screen(state)
            scores = [(3, "include"), (9, "exclude"), (9, "include")]
            for item, (relevance, decision) in zip(judged, scores):
                item.screening.relevance = relevance
                item.screening.decision = decision
            return judged, budget

    planner = StandInPlanner(
        [a_subgoal("first"), a_subgoal("second"), a_subgoal("third")],
        [NextMove(move="continue", reason="more to do")])

    finished = graph.run(a_request(), planner, CuratorThatScoresThemDifferently())

    assert [(item.screening.relevance, item.screening.decision)
            for item in finished.screened] == [
        (9, "include"), (9, "exclude"), (3, "include")]


class PlannerThatFindsSeveral(StandInPlanner):
    """A planner whose every search returns three papers with different titles."""

    def run_next_search(self, state):
        self.asked_for.append("search")
        searched = state.pending_searches[0]
        found = [a_paper(f"{searched.query} paper {number}") for number in range(3)]
        for paper in found:
            paper.origins[0].query = searched.query
        outcome = Outcome(subgoal=searched, result="finished",
                          papers_found=len(found))
        return found, outcome


def run_with_a_paper_allowance(planner, curator, places):
    """Runs one request with a smaller screening allowance than the default."""
    # `graph.run` does not take a budget, because a run is meant to start with
    # the allowance the project decided on. A test needs a smaller one so that
    # the limit can be reached without inventing fifty papers, so it builds the
    # graph itself and starts it from a state carrying the budget it wants.
    starting = ResearchState(
        request=a_request(), budget=RunBudget(papers_for_curation_limit=places))
    return ResearchState.model_validate(
        graph.build_graph(planner, curator).invoke(starting))


def test_the_hand_over_gives_curation_no_more_than_the_allowance(monkeypatch):
    """The screening allowance is spent where the papers are handed over (ID-15, ID-16)."""
    # Two searches return three papers each. With four places, curation is given
    # four and the other two are kept. Handing over all six would spend an
    # allowance the run did not have, and screening them in batches is what the
    # allowance exists to bound.
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)
    planner = PlannerThatFindsSeveral(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="continue", reason="more to do")])
    curator = StandInCurator()

    finished = run_with_a_paper_allowance(planner, curator, places=4)

    assert len(finished.results.papers) == 4, "four places, four papers"
    assert curator.papers_seen == 4, "and curation was given those four"
    assert len(finished.results.papers_beyond_the_screening_limit) == 2, \
        "the two with no place are kept rather than counted"
    assert len(finished.results.papers_before_deduplication) == 6, \
        "and everything the searches returned is kept as it arrived"
    assert finished.budget.papers_sent_to_curation == 4
    assert finished.budget.papers_it_can_still_take() == 0


def test_the_hand_over_shares_the_places_between_the_searches(monkeypatch):
    """Two places and two searches give one place each, not two to the first."""
    # Filling from the start of the deduplicated list would let the first search
    # take both, which is the outcome ID-16 exists to prevent.
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)
    planner = PlannerThatFindsSeveral(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="continue", reason="more to do")])

    finished = run_with_a_paper_allowance(planner, StandInCurator(), places=2)

    queries = sorted(paper.origins[0].query for paper in finished.results.papers)
    assert queries == ["first", "second"], "one place went to each search"


def test_a_paper_two_searches_found_costs_one_place_at_the_hand_over(monkeypatch):
    """The same paper twice is one paper, and one place (FR-10, ID-16)."""
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)

    class PlannerThatFindsTheSamePaperTwice(StandInPlanner):
        def run_next_search(self, state):
            self.asked_for.append("search")
            searched = state.pending_searches[0]
            found = [a_paper("the same trial of drug X")]
            found[0].origins[0].query = searched.query
            return found, Outcome(subgoal=searched, result="finished", papers_found=1)

    planner = PlannerThatFindsTheSamePaperTwice(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="continue", reason="more to do")])

    finished = run_with_a_paper_allowance(planner, StandInCurator(), places=50)

    assert len(finished.results.papers) == 1
    assert len(finished.results.papers[0].origins) == 2, "both searches are recorded"
    assert finished.budget.papers_sent_to_curation == 1, "one paper, one place"
    assert len(finished.results.papers_before_deduplication) == 2, \
        "the two records the searches returned are still there"


def test_an_excluded_paper_keeps_the_text_of_the_records_it_was_merged_from(monkeypatch):
    """A paper with no place keeps its own text, and so do its originals."""
    # Counting the papers the allowance had no room for would pass whatever the
    # records held, and a record emptied on the way through would look the same
    # as one that arrived that way. So this reads the text itself.
    #
    # Everything read here came back out of the JSON. The hand-over writes the
    # buffer out, reads it in again and returns what it read, so an assertion on
    # the state after the run is an assertion about what survived being written
    # and parsed, not about the object the step happened to build (FR-17).
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)

    def a_paper_with_an_abstract(title, abstract, query):
        # Built in one go rather than filled in afterwards, because the record
        # refuses an abstract while it still says it holds only metadata.
        return PaperRecord(
            title=title, abstract=abstract, text_source="abstract",
            origins=[SearchOrigin(source="pubmed", query=query)])

    class PlannerWhoseSecondSearchFindsOnePaperTwice(StandInPlanner):
        def run_next_search(self, state):
            self.asked_for.append("search")
            searched = state.pending_searches[0]
            if searched.query == "first":
                found = [a_paper_with_an_abstract(
                    "a trial nothing else found",
                    "the abstract of the paper with a place", "first")]
            else:
                # The same title twice, so deduplication combines the two. The
                # abstracts differ, which is what makes it possible to say
                # afterwards which record the combined text came from.
                found = [
                    a_paper_with_an_abstract(
                        "a trial seen twice", "the shorter account", "second"),
                    a_paper_with_an_abstract(
                        "a trial seen twice",
                        "the longer account, which says rather more", "second"),
                ]
            return found, Outcome(subgoal=searched, result="finished",
                                  papers_found=len(found))

    planner = PlannerWhoseSecondSearchFindsOnePaperTwice(
        [a_subgoal("first"), a_subgoal("second")],
        [NextMove(move="continue", reason="more to do")])

    finished = run_with_a_paper_allowance(planner, StandInCurator(), places=1)

    assert [paper.title for paper in finished.results.papers] == \
        ["a trial nothing else found"], "the one place went to the first search"

    excluded = finished.results.papers_beyond_the_screening_limit
    assert [paper.title for paper in excluded] == ["a trial seen twice"], \
        "the two records of the other paper were combined into one"
    assert excluded[0].abstract == "the longer account, which says rather more", \
        "and the combined record kept the longer of the two abstracts"
    assert [(origin.source, origin.query) for origin in excluded[0].origins] == \
        [("pubmed", "second"), ("pubmed", "second")], \
        "with both records' origins against it"

    originals = finished.results.papers_before_deduplication
    assert [paper.abstract for paper in originals] == [
        "the abstract of the paper with a place",
        "the shorter account",
        "the longer account, which says rather more",
    ], "and both records the combining chose between are kept unchanged"


class CuratorThatChangesItsMind:
    """A Curation Agent that answers a revision the way a reviewer hoped."""
    # It keeps what it was given, so a test can show which papers were sent and
    # what the reviewer had written on them when they arrived.

    def __init__(self, decision="exclude", extract="drug X was given to 40 patients"):
        self.decision = decision
        self.extract = extract
        self.sent_back = None

    def judge_again(self, state, sent_back):
        self.sent_back = list(sent_back)
        judged = []
        for number, item in enumerate(sent_back, start=1):
            judged.append(ScreenedPaper(
                paper=item.paper,
                screening=Screening(
                    paper_number=number,
                    decision=self.decision,
                    relevance=2,
                    reason=f"looked again because: {item.review_note}",
                    summary="a trial of drug X in condition Y",
                    extract=self.extract,
                ),
            ))
        return judged, state.budget


def a_judgement(title="A trial of drug X", review="waiting", note="",
                extract="drug X was given to 40 patients"):
    paper = PaperRecord(title=title, abstract="drug X was given to 40 patients",
                        text_source="abstract",
                        origins=[SearchOrigin(source="pubmed")])
    return ScreenedPaper(
        paper=paper,
        screening=Screening(paper_number=1, decision="include", relevance=7,
                            reason="it reports an outcome", summary="a trial",
                            extract=extract),
        review=review, review_note=note, extract_found=True)


def a_reviewed_run(*judgements, **changes):
    values = {"request": a_request(), "screened": list(judgements)}
    values.update(changes)
    return ResearchState(**values)


def test_only_the_flagged_and_rejected_papers_go_back():
    """A paper the reviewer approved has been decided, and asking about it
    again would spend a request of eighteen for nothing (FR-19, ID-15)."""
    curator = CuratorThatChangesItsMind()
    state = a_reviewed_run(
        a_judgement("kept", review="approved"),
        a_judgement("doubtful", review="flagged", note="is it a trial?"),
        a_judgement("thrown out", review="rejected", note="it is a review"))

    graph.judge_again(state, curator)

    assert [item.paper.title for item in curator.sent_back] == [
        "doubtful", "thrown out"]


def test_the_flagged_papers_can_go_back_on_their_own():
    """Told which answers go, the graph sends those and leaves the rest as
    they were, decision and all.
    """
    # Since 18 September the screen lets the reviewer send the flagged papers
    # back without the rejected ones: on the live run the reviewer sent one
    # flagged and forty rejected in the one press there was and read the
    # forty again.
    curator = CuratorThatChangesItsMind()
    state = a_reviewed_run(
        a_judgement("kept", review="approved"),
        a_judgement("doubtful", review="flagged", note="is it a trial?"),
        a_judgement("thrown out", review="rejected", note="it is a review"))

    assert [item.paper.title for item in graph.sent_back(state, ("flagged",))] == ["doubtful"]
    after = graph.judge_again(state, curator, ("flagged",))

    assert [item.paper.title for item in curator.sent_back] == ["doubtful"]
    assert [item.review for item in after.screened] == ["approved", "waiting", "rejected"]
    assert after.screened[2].screening.decision == "include", "not judged again"


class CuratorThatNotesTheRun(CuratorThatChangesItsMind):
    """Reads which run is in progress while it judges, as llm.py does when it
    saves a model answer, so a test can see what the saved answer would say."""

    def judge_again(self, state, sent_back):
        self.run_seen = run_log.run_in_progress()
        return super().judge_again(state, sent_back)


def test_the_second_judgement_is_made_inside_its_run():
    """A saved model answer names the run that produced it (ID-20), and the
    identifier is set for the length of run_research.
    """
    # The send-back is called from the screen after that, and on the live run
    # of 18 September its five answers were saved with no run (R-27). The graph
    # now sets it for the second judgement too, and puts back what was there
    # before.
    curator = CuratorThatNotesTheRun()
    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"),
        run_id="the run the reviewer read")

    graph.judge_again(state, curator)

    assert curator.run_seen == "the run the reviewer read"
    assert run_log.run_in_progress() == "", "the run should be cleared afterwards"


def test_a_paper_that_came_back_is_waiting_and_keeps_the_note():
    """FR-19 asks for a fresh decision before a revised record is stored, and
    a flag left in place would stand as an answer to a judgement nobody has
    read. The note is the only thing saying why the paper was judged twice."""
    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"))

    after = graph.judge_again(state, CuratorThatChangesItsMind())

    assert after.screened[0].review == "waiting"
    assert after.screened[0].review_note == "is it a trial?"
    assert after.screened[0].screening.decision == "exclude", "judged again"


def test_the_papers_that_did_not_go_back_are_left_where_they_were():
    """ID-08 orders a run once, at the end of it."""
    # Ranking again here would move papers the reviewer has already worked
    # through, and they would lose their place for a reason with nothing to do
    # with the paper in front of them. The cost is that a revised paper can sit
    # above one scoring higher.
    state = a_reviewed_run(
        a_judgement("kept", review="approved"),
        a_judgement("doubtful", review="flagged", note="is it a trial?"),
        a_judgement("also kept", review="approved"))

    after = graph.judge_again(state, CuratorThatChangesItsMind())

    assert [item.paper.title for item in after.screened] == [
        "kept", "doubtful", "also kept"]
    assert [item.review for item in after.screened] == [
        "approved", "waiting", "approved"]


def test_the_extract_of_a_fresh_judgement_is_looked_for_again():
    """A fresh judgement carries a fresh extract, and the answer recorded
    against the old one says nothing about it (SC-02)."""
    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"))

    after = graph.judge_again(
        state, CuratorThatChangesItsMind(extract="a sentence nobody wrote"))

    assert after.screened[0].extract_found is False


def test_a_run_with_nothing_sent_back_is_left_alone():
    state = a_reviewed_run(a_judgement("kept", review="approved"))

    after = graph.judge_again(state, CuratorThatChangesItsMind())

    assert after == state


def test_what_a_revision_spends_goes_back_into_the_run():
    """A revision nobody counted would let a run make more requests than it
    was allowed (ID-15)."""

    class CuratorThatSpends(CuratorThatChangesItsMind):
        def judge_again(self, state, sent_back):
            judged, _ = super().judge_again(state, sent_back)
            return judged, state.budget.with_one_model_request_used()

    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"),
        budget=RunBudget(model_requests_used=4))

    after = graph.judge_again(state, CuratorThatSpends())

    assert after.budget.model_requests_used == 5


def test_a_revision_with_nothing_left_to_spend_stops_rather_than_failing():
    """Running out is the limit working, as it is in the planning steps."""

    class CuratorWithNothingLeft:
        def judge_again(self, state, sent_back):
            return OutOfBudget("this run has used the 18 model requests it "
                               "was allowed"), state.budget

    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"))

    after = graph.judge_again(state, CuratorWithNothingLeft())

    assert "18 model requests" in after.stopped_because
    assert not after.needs_human_attention
    assert after.screened[0].review == "flagged", "the reviewer's answer is kept"


def test_a_revision_the_model_could_not_answer_asks_for_a_person():
    """A run that cannot be judged again needs somebody to look, the way a
    plan that cannot be made does (DC-06)."""

    class CuratorThatFails:
        def judge_again(self, state, sent_back):
            return FailedRequest("the model could not be reached"), state.budget

    state = a_reviewed_run(
        a_judgement("doubtful", review="flagged", note="is it a trial?"))

    after = graph.judge_again(state, CuratorThatFails())

    assert after.needs_human_attention
    assert "could not be judged again" in after.attention_reason


# The publication status check at the hand-over (FR-09, R-18). The source was
# built and tested in August and nothing in a run called it, so every paper
# reached the reviewer with no status. These tests replace the lookup with a
# stand-in that answers from a script, the way the agents are replaced above.
# The last two keep the real source and pretend only the reply, because what
# they check is the wiring from the hand-over to the source and the source's
# own once-per-DOI promise, which a stand-in could not show.


class PlannerThatFindsPapersWithDois(StandInPlanner):
    """Returns the same three papers for every search: one whose check will
    say retracted, one whose check will fail, and one with no DOI to ask with."""

    def run_next_search(self, state):
        self.asked_for.append("search")
        searched = state.pending_searches[0]
        found = [
            PaperRecord(title="A withdrawn trial of drug X", doi="10.1000/retracted",
                        origins=[SearchOrigin(source="pubmed", query=searched.query)]),
            PaperRecord(title="A trial nobody could check", doi="10.1000/unavailable",
                        origins=[SearchOrigin(source="pubmed", query=searched.query)]),
            PaperRecord(title="A trial with no DOI",
                        origins=[SearchOrigin(source="pubmed", query=searched.query)]),
        ]
        return found, Outcome(subgoal=searched, result="finished",
                              papers_found=len(found))


class StandInCrossref:
    """Answers a list of DOIs from a script and remembers what it was asked."""

    def __init__(self, answers):
        self.answers = answers
        self.asked_about = []

    def __call__(self, dois):
        self.asked_about.append(list(dois))
        return {doi: self.answers[doi] for doi in dois}


def a_scripted_crossref():
    return StandInCrossref({
        "10.1000/retracted": CrossrefRecord(
            "10.1000/retracted", "A withdrawn trial of drug X",
            PublicationStatus.RETRACTED, ["retraction"]),
        "10.1000/unavailable": FetchError(
            "crossref", f"{CROSSREF_API}/10.1000/unavailable", 503, "status 503", 3),
    })


def test_the_hand_over_asks_crossref_about_each_paper_with_a_doi(monkeypatch):
    """The status reaches the reviewer through the judgement, and a paper with
    no DOI is not asked about because there is nothing to ask with (FR-09)."""
    crossref = a_scripted_crossref()
    monkeypatch.setattr(graph, "fetch_statuses", crossref)
    planner = PlannerThatFindsPapersWithDois(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert crossref.asked_about == [["10.1000/retracted", "10.1000/unavailable"]], \
        "the two papers with a DOI were asked about, once each, and the third was not"
    by_title = {judged.paper.title: judged.paper.publication_status
                for judged in finished.screened}
    assert by_title == {
        "A withdrawn trial of drug X": "Retracted after publication",
        "A trial nobody could check": "",
        "A trial with no DOI": "",
    }


def test_a_status_check_that_failed_is_reported_and_does_not_stop_the_run(monkeypatch):
    """A failed check leaves the status blank, which is what a paper with no
    DOI carries too. The buffer, the run log and the coverage report are what
    say that this one was asked about and got no answer (FR-05, FR-15, R-18).
    """
    monkeypatch.setattr(graph, "fetch_statuses", a_scripted_crossref())
    planner = PlannerThatFindsPapersWithDois(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert len(finished.screened) == 3, "the run finished with every paper screened"
    assert finished.results.status_checks_failed == ["10.1000/unavailable: status 503"]

    failures = [line for line in run_log.read() if line["event"] == "failure"]
    assert len(failures) == 1
    assert failures[0]["what"] == "the publication status check for 10.1000/unavailable"
    assert failures[0]["why"] == "status 503"
    assert failures[0]["run"] == finished.run_id, "logged under the run that made it"

    report = audit_log.report_on(finished)
    assert report.status_checks_failed == ["10.1000/unavailable: status 503"]
    assert report.papers_with_no_doi_to_check == 1
    written = "\n".join(audit_log.as_lines(report))
    assert "  - 10.1000/unavailable: status 503" in written
    assert "not asked for, no DOI: 1" in written


def test_papers_beyond_the_allowance_are_not_asked_about(monkeypatch):
    """The lookups are bounded by the screening allowance, because only the
    papers curation is given are ever shown to anyone (ID-15, OB-07)."""
    monkeypatch.setattr(graph, "_write_the_buffer", lambda text: None)
    crossref = a_scripted_crossref()
    monkeypatch.setattr(graph, "fetch_statuses", crossref)
    planner = PlannerThatFindsPapersWithDois(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")])

    finished = run_with_a_paper_allowance(planner, StandInCurator(), places=1)

    assert len(finished.results.papers) == 1
    assert len(finished.results.papers_beyond_the_screening_limit) == 2
    asked = [doi for dois in crossref.asked_about for doi in dois]
    assert len(asked) <= 1, "at most the one paper given a place was asked about"
    for paper in finished.results.papers_beyond_the_screening_limit:
        assert paper.publication_status == "", "the papers with no place were not"


@respx.mock
def test_the_hand_over_reaches_the_real_crossref_source():
    """The three tests above replace the lookup. This one lets the real source
    answer a pretend reply, so the wiring from the hand-over to Crossref is
    shown and not only the graph's handling of an answer.
    """
    # respx refuses any request the test did not describe, so nothing here can
    # reach the real service.
    respx.get(f"{CROSSREF_API}/10.1000/retracted").mock(
        return_value=httpx.Response(200, json={"message": {
            "update-to": [{"type": "retraction"}]}}))

    class PlannerThatFindsOneRetractedPaper(StandInPlanner):
        def run_next_search(self, state):
            self.asked_for.append("search")
            searched = state.pending_searches[0]
            found = [PaperRecord(
                title="A withdrawn trial of drug X", doi="10.1000/retracted",
                origins=[SearchOrigin(source="pubmed", query=searched.query)])]
            return found, Outcome(subgoal=searched, result="finished", papers_found=1)

    planner = PlannerThatFindsOneRetractedPaper(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert finished.screened[0].paper.publication_status == "Retracted after publication"
    assert finished.results.status_checks_failed == []


@respx.mock
def test_two_records_with_one_doi_are_asked_about_once_and_both_keep_the_answer():
    """Deduplication keeps two records with one DOI apart when their PubMed
    identifiers disagree, so the hand-over sends that DOI twice.
    """
    # Before the repair the second lookup replaced the first answer, and a
    # retraction the run had found reached the reviewer as a blank on both
    # records (R-18).
    #
    # The second reply is a 404, which is not retried: if the DOI were asked
    # about twice, the second lookup would fail at once, both records would go
    # blank and the failure would be reported, and this test would say so. A 503
    # would be asked again and run the pretend replies out, so the test would
    # fail for the wrong reason.
    asked = respx.get(f"{CROSSREF_API}/10.1000/shared").mock(side_effect=[
        httpx.Response(200, json={"message": {"update-to": [{"type": "retraction"}]}}),
        httpx.Response(404, json={}),
    ])

    class PlannerThatFindsTwoRecordsWithOneDoi(StandInPlanner):
        def run_next_search(self, state):
            self.asked_for.append("search")
            searched = state.pending_searches[0]
            found = [
                PaperRecord(title="A withdrawn trial of drug X", doi="10.1000/shared",
                            pmid="111",
                            origins=[SearchOrigin(source="pubmed", query=searched.query)]),
                PaperRecord(title="A withdrawn trial of drug X", doi="10.1000/shared",
                            pmid="222",
                            origins=[SearchOrigin(source="pubmed", query=searched.query)]),
            ]
            return found, Outcome(subgoal=searched, result="finished", papers_found=2)

    planner = PlannerThatFindsTwoRecordsWithOneDoi(
        [a_subgoal()], [NextMove(move="finish", reason="that is enough")])

    finished = graph.run(a_request(), planner, StandInCurator())

    assert len(finished.results.papers) == 2, "the two records stay apart"
    for paper in finished.results.papers:
        assert paper.publication_status == "Retracted after publication"
    assert finished.results.status_checks_failed == []
    assert asked.call_count == 1, "and the DOI they share was asked about once"
