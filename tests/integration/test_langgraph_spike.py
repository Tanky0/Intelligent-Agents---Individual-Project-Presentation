"""Checks that LangGraph can do what the design's Figure 2 needs."""
# This began as a small trial written before any agent, to find out early whether
# LangGraph could carry the design at all. It was kept as a test rather than
# deleted, so that a later version of LangGraph that stops doing any of these
# things fails here instead of inside the planning agent, where the cause would
# be much harder to find.
#
# Six behaviours are checked. The first five are ones the design needs and a set
# of steps that simply runs one after another would not:
#
#   1. State described by a Pydantic class, which is the checked hand-over
#      between the two agents (DC-05).
#   2. A step that looks at what came back and chooses what to do next.
#   3. A step that can be given more than one destination to choose between,
#      which is what the four moves in FR-04 need.
#   4. A loop back to the planning step, which is the reason-act-observe cycle
#      the design asks for rather than a plan fixed before the run (DC-03).
#   5. That the record the agents will really use, with all its checking on,
#      works as that state and is not refused by it.
#
# The sixth is the opposite: something LangGraph does not do. It does not check
# what a step hands back against the format of the state, so the record can be
# left holding a value the format forbids. That is why graph.py reads the whole
# record again where it passes between the agents and again at the end of a run,
# and this is the measurement those two checks rest on.
#
# A test that claimed to check all four moves was removed on 1 September. It
# asserted that the graph had been built and that the move being tested was one
# of the four it had just been given, both of which are true whatever the graph
# does, so it could not fail. Reading the built graph instead showed why it was
# written that way: continue and drop lead to the same step, and LangGraph
# records one edge for the two, so four moves cannot be counted here at all. The
# four-way choice lives in graph.py, in the step that turns a move into a
# destination, and it is tested there against the state of a run.
#
# Nothing here calls a model or a service. The choices are made by fixed rules,
# so the test gives the same answer every time.

import pytest
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError

from research_agent.schemas import ResearchRequest, ResearchState


class TrialState(BaseModel):
    """Stands in for the real state of a run."""
    # The field names match the ones in schemas.py, so that this trial and the
    # thing it stands in for read alike. They hold the desires, intentions and
    # beliefs of a BDI agent under names saying what each one is, and the
    # mapping is written out where the real state is defined (DC-02).

    request: str = Field(description="the research question being answered")
    pending_searches: list[str] = Field(default_factory=list,
                                  description="the subgoals still to be done")
    results: list[str] = Field(default_factory=list,
                               description="what searching has returned so far")
    steps_taken: int = 0
    # An agent with limited time has to stop thinking and act, so the loop can
    # never run without end (DC-06).
    step_limit: int = 5


def plan(state):
    """The thinking step: breaks the question into subgoals."""
    if state.pending_searches:
        return {}

    subgoals = []
    for part in (1, 2, 3):
        subgoals.append(f"{state.request}::part{part}")
    return {"pending_searches": subgoals}


def act_and_observe(state):
    """The acting step: takes the next subgoal and records what came back."""
    remaining = list(state.pending_searches)
    current = remaining.pop(0)

    found = list(state.results)
    found.append(f"result-for:{current}")

    return {
        "pending_searches": remaining,
        "results": found,
        "steps_taken": state.steps_taken + 1,
    }


def decide_what_next(state):
    """Chooses the next move from what searching actually returned."""
    # This is what separates an agent that plans from a fixed list of steps. The
    # next move is decided by the results, not settled before the run began.
    if state.steps_taken >= state.step_limit:
        return "finish"
    if not state.pending_searches:
        return "finish"
    return "continue"


def build_graph():
    """Wires the two steps together with the choice between them."""
    graph = StateGraph(TrialState)
    graph.add_node("plan", plan)
    graph.add_node("act", act_and_observe)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges(
        "act",
        decide_what_next,
        {"continue": "act", "refine": "plan", "drop": "act", "finish": END},
    )
    return graph.compile()


def test_the_whole_shape_works_end_to_end():
    """Typed state carried through a loop that ends by its own decision."""
    finished = build_graph().invoke(TrialState(request="ivermectin for covid-19"))

    state = TrialState.model_validate(finished)
    assert state.pending_searches == [], "every subgoal should have been done"
    assert len(state.results) == 3, "one result recorded for each subgoal"
    assert state.steps_taken == 3


def test_the_loop_stops_even_if_it_is_never_told_to_finish():
    """A wrong decision must not leave the agent running for ever (DC-06)."""
    state = TrialState(request="anything", pending_searches=["a subgoal"] * 50,
                       step_limit=4)

    finished = TrialState.model_validate(build_graph().invoke(state))

    assert finished.steps_taken == 4


def test_the_real_state_of_a_run_works_as_graph_state():
    """The record the agents will actually use, driven through a real graph."""
    # Everything above uses the stand-in. This uses ResearchState from
    # schemas.py, because a format that refuses values it does not expect could
    # also refuse the updates a graph step hands back. Finding that out here
    # means a change to those rules fails against LangGraph rather than during
    # the first run of the planning agent.
    #
    # What this shows is that valid updates pass through LangGraph unrefused and
    # the record still checks out at the end. It does not show that every rule
    # holds at every step: the updates given here are valid ones. A record broken
    # part way through, by changing something inside a list it already holds,
    # would not be caught here. That test needs graph.py, where the whole record
    # is rebuilt and checked as it passes from one agent to the next.
    #
    # It is kept as a test rather than checked once by hand, because a check
    # nobody can repeat is not evidence.
    request = ResearchRequest(
        question="Which studies evaluate drug X as a treatment for condition Y?",
        inclusion_criteria="original human research reporting an outcome",
        exclusion_criteria="reviews and opinion pieces",
    )

    def take_a_step(state):
        return {"steps_taken": state.steps_taken + 1}

    def decide_whether_to_stop(state):
        if state.steps_taken >= state.step_limit:
            return "finish"
        return "again"

    graph = StateGraph(ResearchState)
    graph.add_node("step", take_a_step)
    graph.set_entry_point("step")
    graph.add_conditional_edges("step", decide_whether_to_stop,
                                {"again": "step", "finish": END})

    finished = graph.compile().invoke(ResearchState(request=request, step_limit=3))

    state = ResearchState.model_validate(finished)
    assert state.steps_taken == 3, "the limit stopped it"
    assert state.request.question == request.question, "the request survived"
    assert state.results.ready is False, "curation was never signalled"


def test_langgraph_does_not_check_what_a_step_hands_back():
    """Why the record is read again at the boundaries in graph.py (IT-03)."""
    # A step hands back only the fields it changed, and LangGraph writes them
    # into the state without checking them against the format. So a step can
    # leave the record holding a value its own format forbids, and nothing says
    # so. Here the run ends asking for a person with no reason written down,
    # which ResearchState refuses when it is built and allows when it is reached
    # a field at a time.
    #
    # This is measured rather than assumed, because the two checks in graph.py
    # rest on it. If a later version of LangGraph starts checking, this test
    # fails and says the checks may no longer be needed. If it keeps not
    # checking, this is the evidence that they are.
    request = ResearchRequest(
        question="Which studies evaluate drug X as a treatment for condition Y?",
        inclusion_criteria="original human research reporting an outcome",
        exclusion_criteria="reviews and opinion pieces",
    )

    def stop_without_saying_why(state):
        return {"needs_human_attention": True}

    graph = StateGraph(ResearchState)
    graph.add_node("stop", stop_without_saying_why)
    graph.set_entry_point("stop")
    graph.add_edge("stop", END)
    finished = graph.compile().invoke(ResearchState(request=request))

    assert finished["needs_human_attention"] is True
    assert finished["attention_reason"] == "", "LangGraph let the rule be broken"

    with pytest.raises(ValidationError):
        ResearchState.model_validate(finished)
