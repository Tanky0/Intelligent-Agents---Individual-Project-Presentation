"""Wires the two agents together as one run (DC-03)."""
# Figure 2 of the design shows a loop: the goal is broken into searches, a search
# is run, what came back is looked at, and the next move is decided from it. The
# buffer passes to the Curation Agent once the searches are done, or once the
# limit on the run is reached with searches still waiting, which is the case the
# coverage report has to be able to describe.
#
# This file builds that loop and makes no decision of its own: it reads what the
# agents wrote and moves to the step that answers it. What happens inside each
# step belongs to the two agents: the Planning Agent is in agents/planner.py and
# the Curation Agent in agents/curator.py.
#
# The last part of Figure 2 runs outside that loop. A paper the reviewer flagged
# or rejected goes back to the Curation Agent to be judged again, and `judge_again`
# below is where that happens. It is not a step in the graph because it starts
# from what a person decided, and the graph is finished by the time they see
# anything.
#
# The two parts of the design meet here. Breaking the goal into searches is the
# deliberative part, done once at the start, and deciding what to do after each
# search is the reactive part, done from what actually came back (DC-01).
# Wooldridge (2009) describes that division, and keeping them in separate steps
# is what makes it visible in the code rather than only in the report.
#
# The record is read against its own rules in two places: where it passes from
# one agent to the other, and again at the end of the run. Neither of those is
# something LangGraph does. A step hands back only the fields it changed, and
# those are written into the state without being checked against the format, so a
# step can leave the record holding a value the format forbids and nothing will
# say so. That was measured rather than assumed, and the test that measures it is
# in tests/integration/test_langgraph_spike.py. Checking at the hand-over means
# the Curation Agent is never given a record nothing has read (DC-05), and
# checking at the end means the same for the interface and for storage.
#
# The two agents are passed into build_graph instead of being imported here.
# This keeps the graph independent from their implementation and allows tests to
# replace them with simple stand-ins that do not call a model or the network. It
# also means they can be written after this file. What each agent has to provide
# is written out in build_graph.

import datetime

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from research_agent.config import REPO_ROOT
from research_agent.llm import OutOfBudget
from research_agent.processing import run_log
from research_agent.processing.dedup import deduplicate
from research_agent.processing.ranking import in_rank_order
from research_agent.processing.selection import share_out_places
from research_agent.processing.validation import with_the_extracts_checked
from research_agent.schemas import (
    NextMove,
    PlanningDecision,
    ResearchState,
    ResultsBuffer,
    Subgoal,
    SubgoalList,
)
from research_agent.sources.crossref import fetch_statuses
from research_agent.sources.http import FetchError

# Saves the buffer created when results are passed from one agent to the other.
# Figure 1 draws the buffer as a thing that sits between the two agents, while
# the code holds it as part of the state of the run, because a graph keeps one
# state. Saving it here gives a separate record of what was passed between the
# agents.
# Tests use a temporary folder instead, set in tests/conftest.py.
#
# Buffer files contain abstracts, which are other people's text, so they are
# not committed automatically. A buffer is only added to the repository when it is
# deliberately kept as evidence. This follows the same approach as saved
# replies (R-07).
BUFFER_DIR = REPO_ROOT / "evidence" / "sample_run"

# Keeps track of any buffer files that could not be saved.
# A save failure should not stop the run, but it should be recorded so a
# completed run does not appear to have produced evidence when no file was
# actually written.
buffer_failures = []


def _why_it_failed(answer):
    """Explains why an answer from an agent could not be used."""
    # A failed model request returns a ModelFailure with a reason.
    # None means demonstration mode was used but no saved reply was available.
    if answer is None:
        return "demonstration mode is on and no saved answer was found for this step"
    # A ModelFailure should always contain a reason. Using getattr here avoids
    # causing another error if an unexpected value reaches this function.
    reason = getattr(answer, "reason", "")
    if not reason:
        return (f"a step gave back a {type(answer).__name__}, which is neither "
                "an answer nor a reason")
    return reason


def _stop_because(reason):
    """Stops the run for a reason that is not a fault (OB-07)."""
    # Spending the model requests a run was allowed is the first of these. It is
    # the limit doing its job, and reporting it as needing a person would put an
    # ordinary stop in front of somebody as though something had broken. What the
    # run found is kept either way.
    return {"stopped_because": reason}


def _stop_and_ask_for_a_person(reason):
    """Stops the run and records why a person needs to take over (DC-06)."""
    # Several steps can reach this point, so the same handling is kept in one
    # place. If the agent cannot produce a usable answer after the allowed
    # repairs, the run stops instead of continuing to ask indefinitely.
    # Bratman, Israel and Pollack (1988) describe an agent with limited time as
    # one that has to stop and act rather than reconsider without end.
    return {"needs_human_attention": True, "attention_reason": reason}


def after_planning(state):
    """Chooses what happens after the Planning Agent creates the search plan."""
    # A plan cannot be empty because the schema already prevents that, so there
    # is no separate branch for an empty plan here. Adding one would be a branch
    # no test could reach, and this project has already been caught out once by a
    # check that could not fail.
    #
    # This routing is kept outside build_graph so it can be tested directly
    # without having to build the full graph.
    if state.needs_human_attention or state.stopped_because:
        return "stop"
    return "search"


def after_searching(state):
    """Chooses what happens after a search has finished."""
    # The Planning Agent can return four moves, but they do not all lead to
    # different places. Continue and drop both move on to the next search, or
    # hand over to the Curation Agent if there are no searches left. What
    # separates them is kept in the list of decisions rather than in the routing:
    # the outcome of a search records what the search did, not what was decided
    # afterwards, so a subgoal that was dropped would otherwise leave no trace.
    #
    # The step limit is checked before following the requested move so an agent
    # cannot keep the run going indefinitely by repeatedly choosing continue
    # (DC-06). The searches never reached are left in the state, so a run stopped
    # this way still shows what it did not do.
    #
    # A stop condition is checked first because a failed agent request may leave
    # no move to follow.
    if state.needs_human_attention:
        return "stop"
    # A run that has spent its allowance has nothing left to ask with, so it
    # cannot search again and cannot curate either. It stops where it is and
    # keeps what it found, rather than being sent on to a step that would ask
    # for a request it does not have.
    if state.stopped_because:
        return "stop"
    if state.steps_taken >= state.step_limit:
        return "hand over"

    move = state.last_move.move
    if move == "finish":
        return "hand over"
    if move == "refine":
        return "refine"
    if state.pending_searches:
        return "search"
    return "hand over"


def _write_the_buffer(as_json):
    """Writes the buffer to a JSON file and returns the file path."""
    # It is given the JSON the Curation Agent is handed rather than the record it
    # came from, so the file holds exactly what was passed between the agents and
    # not a second copy made separately.
    #
    # If the file cannot be written, the failure is recorded in buffer_failures
    # and None is returned. A file-writing problem should not discard a run that
    # has already completed its searches.
    # Use UTC so buffer files and saved replies can be matched without
    # depending on the computer's local time. Colons are left out because they
    # are not valid in Windows file names. Microseconds are kept so two fast
    # demonstration runs do not create the same file name.
    when = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = BUFFER_DIR / f"buffer_{when}.json"
    try:
        BUFFER_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(as_json, encoding="utf-8")
    except OSError as problem:
        buffer_failures.append(f"the buffer could not be written to {path}: {problem}")
        # Written to the log as well as collected. The list says a buffer was
        # lost; the log says which run lost it and when, which is what somebody
        # looking for a missing evidence file has to go on.
        #
        # The run is read from the log's own record of which run is in
        # progress, which run() sets for the length of the run. It used to be
        # kept in one value for the whole process, and two runs at once, which
        # two browser sessions are, would have had the second overwrite the
        # first, so a buffer the first run lost was logged against the second
        # (R-19). The log keeps the run per execution context for that reason.
        run_log.note_a_failure("the buffer",
                               f"could not be written to {path}: {problem}",
                               run_log.run_in_progress())
        return None
    return path


def _with_the_status_checked(papers):
    """Asks Crossref about each paper with a DOI, and says which checks failed."""
    # Returns the papers with the status written in, and one line for each
    # paper whose check failed. The source was built and tested in August and
    # nothing called it, so every paper reached the reviewer with no status,
    # which for a retracted paper is the silence the status check exists to
    # break (R-18).
    #
    # Asked here, at the hand-over, for the papers curation is given, rather
    # than as each search returns. A paper two searches found would be asked
    # about twice before the repeated records are combined, and the papers the
    # allowance has no place for are never shown to anyone, so asking about
    # them would be lookups nobody reads. Asking after the places are shared
    # out bounds the lookups at the screening allowance, and none of them is a
    # model request (OB-07). Two records can still carry one DOI here, because
    # deduplication keeps them apart when their PubMed identifiers disagree;
    # the source asks about that DOI once and the answer is given to both.
    #
    # A paper with no DOI is not asked about: Crossref is looked up by DOI and
    # there is nothing to ask with. A paper whose check failed is not given a
    # status either, and the line returned says so. Both carry a blank status,
    # which means no status was obtained, and the coverage report is what tells
    # the reader which of the two a paper is. Writing "none identified" for a
    # failed check was never an option: that sentence is an answer, and the
    # check gave none.
    #
    # A failed check is written to the run log as well as returned, the way a
    # lost buffer is, so the log says when it failed and the report says that it
    # did. A check that failed does not stop the run: the status is a
    # warning shown beside the paper, not something screening cannot do without.
    dois = []
    for paper in papers:
        if paper.doi:
            dois.append(paper.doi)
    answers = fetch_statuses(dois)

    checked = []
    failed = []
    for paper in papers:
        if not paper.doi:
            checked.append(paper)
            continue
        answer = answers[paper.doi]
        if isinstance(answer, FetchError):
            failed.append(f"{paper.doi}: {answer.reason}")
            run_log.note_a_failure(f"the publication status check for {paper.doi}",
                                   answer.reason, run_log.run_in_progress())
            checked.append(paper)
            continue
        checked.append(paper.model_copy(
            update={"publication_status": answer.status.value}))
    return checked, failed


def build_graph(planner, curator):
    """Builds the run, with the Planning Agent and Curation Agent given to it."""
    # The Planning Agent has to provide three things:
    #
    #     plan_searches(state)             the ordered searches to run
    #     run_next_search(state)           the papers found and what became of
    #                                      the subgoal, as a pair
    #     decide_next_move(state, outcome, papers)
    #                                      continue, refine, drop or finish, with
    #                                      the reason for it. The papers are the
    #                                      ones this search returned, handed over
    #                                      rather than looked up
    #
    # The Curation Agent has to provide one:
    #
    #     screen(state)                    a judgement for each paper
    #
    # Any of them may return a ModelFailure instead, or None in DEMO_MODE when no
    # reply has been saved for that step. Both stop the run and ask for a person.
    # The three that ask the model, plan_searches, decide_next_move and
    # screen, may also return an OutOfBudget, which stops the run without
    # asking for a person: the allowance running out is the limit working,
    # not a fault (OB-07). run_next_search asks no model and is not expected
    # to return one; anything it returns that is not the pair is treated as a
    # failure and asks for a person.
    #
    # Two parts of the behaviour are handled inside the agents rather than by the
    # graph itself.
    #
    # If an agent returns a badly formatted answer, Figure 2 allows three
    # repairs before a person is asked to take over (DC-06). By the time a
    # failure reaches this graph, those repairs have already been used.
    #
    # A failed search is handled differently. It is returned as a normal search
    # outcome with the reason for the failure, and the run continues with the
    # remaining searches. The coverage report then records what could not be
    # retrieved. Only a model request that cannot produce a usable answer
    # stops the whole run.

    def plan(state):
        """Breaks the research goal into the searches to run."""
        # This is the deliberative half of the design, and it happens once. The
        # searches decided here are what the agent then commits to, which is what
        # stops it reconsidering the whole plan after every result (DC-01).
        answer, budget = planner.plan_searches(state)
        # The budget is written back whatever the answer was. A request that
        # failed was still a request, and a run that lost the count of it would
        # keep asking with an allowance it had already spent.
        if isinstance(answer, OutOfBudget):
            return {"budget": budget, **_stop_because(answer.reason)}
        if not isinstance(answer, SubgoalList):
            return {"budget": budget, **_stop_and_ask_for_a_person(
                f"the searches could not be planned: {_why_it_failed(answer)}")}
        return {"pending_searches": answer.subgoals, "budget": budget}

    def search(state):
        """Runs the next search, records what came back, and decides what next."""
        # Acting and observing are one step on purpose. The move is decided from
        # what the search actually returned, so it has to be decided after the
        # search rather than before it, and it is written into the state so that
        # the choice of where to go next only has to read it (DC-03).
        found = planner.run_next_search(state)
        if not isinstance(found, tuple):
            return _stop_and_ask_for_a_person(
                f"a search could not be run: {_why_it_failed(found)}")
        # The papers and the outcome are expected as a pair. A tuple of any
        # other length would be unpacked into the wrong names on the next line
        # and stop the run with a message about unpacking rather than about
        # the agent that broke the agreement.
        if len(found) != 2:
            return _stop_and_ask_for_a_person(
                f"a search returned {len(found)} values instead of the papers "
                "and the outcome of the search")
        papers, outcome = found

        # Building the buffer validates every paper against the shared paper
        # format. If one record is invalid, the run is stopped cleanly instead
        # of raising an exception and losing all papers that were already
        # collected (DC-06).
        #
        # This is most likely to happen while records from the different search
        # services are being converted into the shared paper format.
        #
        # Three problems are caught together, and the message names all of
        # them rather than the first. A record of the wrong kind is refused by
        # the format, something that is not a list of records at all fails
        # before the format sees it, and an outcome of the wrong kind is
        # refused in the same breath. Naming only the papers would send
        # somebody looking at the papers when the outcome was the problem.
        try:
            buffer = ResultsBuffer(
                papers=list(state.results.papers) + list(papers),
                outcomes=list(state.results.outcomes) + [outcome],
                ready=state.results.ready,
            )
        except (TypeError, ValidationError) as problem:
            return _stop_and_ask_for_a_person(
                "what a search returned did not match the agreed format for the "
                f"papers and the outcome: {problem}")

        # The queue is what the agent has committed to, and working through it
        # in order is what stops the whole plan being reconsidered after every
        # result (DC-02). So the search that has just finished comes off the
        # front and the rest keep their order.
        #
        # No empty-queue check is needed here because the routes leading to
        # this step only call it when a pending search is available.
        remaining = list(state.pending_searches)
        remaining.pop(0)
        steps_taken = state.steps_taken + 1

        # What the search achieved, written down before anything else is asked.
        # Whatever happens next, these papers were retrieved and this subgoal
        # was dealt with, and a later failure must not take them with it. An
        # earlier version of this step returned only the failure, so a search
        # that worked was thrown away because the request after it did not.
        # Written as the search finishes rather than worked out at the end.
        # A search that failed, was refined and then worked leaves a finished
        # state behind, and only a line written at the time says the first one
        # happened.
        run_log.note_a_search(outcome, state.request, state.run_id)

        searched = {
            "pending_searches": remaining,
            "results": buffer,
            "steps_taken": steps_taken,
            # Cleared, and set again only if a new move is decided below. Left
            # alone, the move from the previous search would still be sitting
            # there, and a run that ended at the limit or on a failed request
            # would finish carrying a decision that was never made about the
            # search it appears to belong to. The lasting record is the list of
            # decisions; this field is only what the next step reads.
            "last_move": None,
        }

        # At the limit there is no move left to make, because the step that
        # chooses where to go next reads the limit before it reads the move.
        # Asking anyway would spend one of the twenty model requests a day on
        # an answer nothing can act on (OB-07).
        if steps_taken >= state.step_limit:
            return searched

        # Figure 2 writes what came back into the buffer before anything asks
        # what to do next, so the agent is shown the state with the new papers
        # already in it, and the papers this search returned are handed to it
        # directly. Working out which papers were new by looking at the buffer
        # would join two searches that happened to use the same words (DC-03).
        observed = state.model_copy(
            update={"results": buffer, "steps_taken": steps_taken})
        move, budget = planner.decide_next_move(observed, outcome, papers)
        searched["budget"] = budget
        # Running out here is not a failure. The searches that ran are kept and
        # the run stops with the reason, rather than asking for a person about
        # a limit working as it was meant to.
        if isinstance(move, OutOfBudget):
            return {**searched, **_stop_because(move.reason)}
        if not isinstance(move, NextMove):
            return {
                **searched,
                **_stop_and_ask_for_a_person(
                    f"the next move could not be decided: {_why_it_failed(move)}"),
            }

        return {
            **searched,
            "last_move": move,
            # Kept as well as last_move, because last_move is overwritten by
            # the next search and the run report needs all of them.
            "decisions": list(state.decisions) + [PlanningDecision(
                subgoal=outcome.subgoal, move=move.move, reason=move.reason)],
        }

    def refine(state):
        """Adds the refined search to the front of the queue."""
        # A refinement keeps the same purpose and source but changes the query.
        # It is placed first because it replaces the search that has just failed
        # to give a good enough result.
        #
        # Refined searches still count towards the normal step limit (DC-06).
        searched = state.results.outcomes[-1].subgoal
        replacement = Subgoal(
            purpose=searched.purpose,
            source=searched.source,
            query=state.last_move.new_query,
        )
        return {"pending_searches": [replacement] + list(state.pending_searches)}

    def hand_over(state):
        """Validates the complete record and signals that curation may start."""
        # The individual steps already build validated records, so this check
        # should normally pass. It is still needed because LangGraph itself does
        # not enforce the format of the state returned by each step.
        #
        # If the record is invalid, the run stops with an application error. This
        # is different from a search service failing: a failed search can be
        # reported and the run can continue, but an invalid shared record should
        # not be passed to the Curation Agent as if it were valid evidence
        # (DC-05).
        #
        # This is also where the repeated records are combined and the screening
        # allowance is spent. It happens here rather than
        # after each search because a paper two searches found is only repeated
        # once both have run, and because spending the allowance early would
        # commit places to the first search before the later ones had returned
        # anything to compete with.
        #
        # The publication status is asked for here as well, once the places are
        # shared out, so that only the papers curation is given are looked up
        # (R-18).
        ResearchState.model_validate(state.model_dump())

        retrieved = list(state.results.papers)
        combined = deduplicate(retrieved)
        # The allowance is asked what is left rather than being told fifty,
        # because a run that hands over more than once must not be given a
        # fresh fifty each time.
        for_curation, beyond_the_limit = share_out_places(
            combined, state.results.outcomes, state.budget.papers_it_can_still_take())
        for_curation, status_checks_failed = _with_the_status_checked(for_curation)

        finished = ResultsBuffer(
            papers=for_curation,
            outcomes=list(state.results.outcomes),
            ready=True,
            papers_before_deduplication=retrieved,
            papers_beyond_the_screening_limit=beyond_the_limit,
            status_checks_failed=status_checks_failed,
        )

        # The hand-over is the JSON, not the record that produced it. The
        # buffer is written out as JSON, read back and checked before curation
        # is given it, so what that agent works from is what came out of the
        # JSON rather than the object the Planning Agent happened to leave in
        # the state. The design calls the thing between the two agents a
        # schema-validated JSON buffer, and this is what makes that the actual
        # hand-over rather than a description of one (DC-05).
        #
        # The same text is saved to the file, so the evidence kept for a run is
        # the buffer that was passed and not a separate copy of it.
        as_json = finished.model_dump_json(indent=2)
        _write_the_buffer(as_json)
        # The budget is written back with the papers rather than left for
        # curation to work out, so what the state holds afterwards is what was
        # actually spent. A run whose places were spent here and whose budget
        # said otherwise would be a record of an allowance nothing was
        # keeping.
        return {
            "results": ResultsBuffer.model_validate_json(as_json),
            "budget": state.budget.with_papers_sent_to_curation(len(for_curation)),
        }

    def curate(state):
        """Screens the papers in the buffer against the review criteria."""
        # This step checks that the agent returned a list. The individual
        # judgements are validated later when the final run record is built, so
        # the same validation does not need to be repeated here.
        #
        # Screening asks the model, so it spends from the same allowance the
        # planning steps do, and hands the budget back the way they do.
        # A run that counted planning and not screening would report an
        # allowance it had not kept.
        #
        # The judgements are kept in the order the papers were handed over. They
        # are put in rank order at the end of the run, after the record has been
        # read back and checked, because ranking has to read the score inside
        # each judgement and this step has not established that there is one
        # (DC-05).
        judgements, budget = curator.screen(state)
        # Running out here is the limit working rather than a fault, as it is
        # in the planning steps. What was retrieved is still in the buffer, and
        # the reason is recorded.
        if isinstance(judgements, OutOfBudget):
            return {"budget": budget, **_stop_because(judgements.reason)}
        if not isinstance(judgements, list):
            return {"budget": budget, **_stop_and_ask_for_a_person(
                f"the papers could not be screened: {_why_it_failed(judgements)}")}
        return {"screened": judgements, "budget": budget}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan)
    graph.add_node("search", search)
    graph.add_node("refine", refine)
    graph.add_node("hand over", hand_over)
    graph.add_node("curate", curate)

    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan", after_planning, {"search": "search", "stop": END})
    graph.add_conditional_edges(
        "search", after_searching,
        {"search": "search", "refine": "refine", "hand over": "hand over",
         "stop": END})
    # Refining always goes back to searching. The new query is already at the
    # front of the queue, so the next search is the refined one.
    graph.add_edge("refine", "search")
    graph.add_edge("hand over", "curate")
    graph.add_edge("curate", END)
    return graph.compile()


def run(request, planner, curator, step_limit=None):
    """Runs one whole request and returns the state it ended in."""
    # LangGraph hands back a plain dictionary rather than the record it was
    # given, so it is read back into a ResearchState here. That is also the last
    # check on the record, and unlike the one at the hand-over it catches
    # something a run can really produce: the Curation Agent hands back a list of
    # judgements, and the graph can see that it is a list without seeing what is
    # in it. Anything else in that list is refused here rather than reaching the
    # reviewer.
    #
    # The maximum number of searches allowed in this run can be set here. A
    # shorter run makes fewer model requests, which matters when the free tier
    # allows twenty a day and a demonstration may need rehearsing more than once
    # (OB-07). It is a limit on the reasoning loop rather than a count of model
    # requests: planning, repairs and curation all make their own, so a run kept
    # short uses less of the allowance without being a measure of what is left.
    # If no limit is given here, use the default in schemas.py rather than
    # repeating the same number in two places.
    # Made here rather than worked out from the request, so that two runs of
    # the same question are two runs in the log.
    identifier = run_log.new_identifier()
    if step_limit is None:
        starting = ResearchState(request=request, run_id=identifier)
    else:
        starting = ResearchState(request=request, run_id=identifier,
                                 step_limit=step_limit)

    # The question and the criteria go down before anything is searched. The
    # decisions later in the log cannot be judged without them: a paper
    # excluded under one set of criteria is not excluded under another.
    run_log.note_the_request(request, starting.run_id)

    # The shared request handling records what each request carried, and it is
    # several calls below anything that knows which run is happening. It is
    # told here instead, for the length of the run and no longer, so a request
    # made by something else afterwards is not filed under this run.
    with run_log.while_running(starting.run_id):
        finished = build_graph(planner, curator).invoke(starting)
    checked = ResearchState.model_validate(finished)

    # Why a run stopped goes into the log as well as into the state. The
    # state reaches the screen, and the log outlives the screen: on 17
    # September the live run stopped on a refused model request, the reason
    # was in the coverage section, and the log's last line for the run was
    # the failed search before it, so a reader of the log alone could not
    # tell whether the run had ended or hung. Written once here,
    # from the finished state, rather than inside each step that can stop
    # the run, so there is one line to find however it stopped. The two
    # kinds of stop keep the words the coverage report uses for them.
    if checked.needs_human_attention:
        run_log.note_a_failure("the run needs a person", checked.attention_reason,
                               checked.run_id)
    elif checked.stopped_because:
        run_log.note_a_failure("the run stopped early", checked.stopped_because,
                               checked.run_id)

    # Ranked here rather than in the curation step, because ranking reads the
    # score inside each judgement and only the check above has established that
    # every judgement is one. Ranking first turned a list holding something
    # else into a message about a missing attribute, in place of the message
    # saying the record was refused (DC-05).
    #
    # The state that comes out of a run is what the reviewer is shown, so the
    # order it holds is the order that will be read. Leaving it unranked would
    # put the ranking in whatever displayed it, and two things showing the same
    # run could then show it differently.
    #
    # The extracts are looked for in the papers here as well, and for the same
    # reason: it reads the extract inside each judgement, which the check above
    # is what establishes there is. A paper whose extract could not be found is
    # marked and still shown, because whether to keep it is the reviewer's to
    # decide (DC-08).
    return checked.model_copy(update={
        "screened": in_rank_order(with_the_extracts_checked(checked.screened))})


# The two answers that send a paper back. Named once because the list of what
# goes back and the merge of what comes back have to agree, and a second copy
# of them is a second rule that could drift. Since 18 September the reviewer
# chooses on the screen whether the rejected papers go back with the flagged
# ones, so the two functions below take the answers that go; this pair is
# what they take when nobody chooses.
_SENT_BACK = ("flagged", "rejected")


def sent_back(state, answers_that_go=_SENT_BACK):
    """The papers the reviewer flagged or rejected, which is what goes back."""
    # Kept here rather than in the screen, because the graph is what decides
    # which papers reach the Curation Agent and the screen is what draws them.
    # A flag and a rejection are both sent: a flag says the paper needs another
    # look, and a rejection the reviewer wrote a reason for is a disagreement
    # the agent can answer. On the live run of 18 September
    # forty-one papers were sent back, one flagged and forty rejected, and
    # the second judgement of the forty was a long way round to a shorter
    # list, so the screen now offers the flagged alone as well as both, and
    # says which here.
    return [judged for judged in state.screened if judged.review in answers_that_go]


def judge_again(state, curator, answers_that_go=_SENT_BACK):
    """Hands the papers the reviewer sent back to the Curation Agent."""
    # This is the part of Figure 2 that was missing. Everything else in the loop
    # runs before a person sees anything; this one starts from what a person
    # decided, so it is called from the screen rather than being a step in the
    # graph.
    #
    # A fresh judgement leaves the paper waiting for a decision again, which is
    # what the design asks for: a revised record needs a new decision before it
    # is stored. Storage treats waiting and rejected alike, withdrawing both, so
    # this is not about what the store does with the row. It is about what the
    # record says: a flag left in place would stand as the reviewer's answer to
    # a judgement they have not read.
    #
    # The reviewer's note is left on the paper. Until they decide again it is the
    # only thing saying why this paper was judged twice, and their next decision
    # replaces it.
    #
    # The order is not worked out again. A run is ranked once, at the end of
    # it, and that is the order the reviewer starts reading. Ranking again here
    # would move papers they have already worked through, and somebody part way
    # down the list would lose their place for a reason that had nothing to do
    # with the paper in front of them. The cost is that a revised paper can then
    # sit above one that now scores higher, so the list a reviewer is working
    # through is the order the run finished in rather than the order the current
    # scores would give.
    #
    # The extracts of the new judgements are looked for in their papers, because
    # a fresh judgement carries a fresh extract and the answer recorded against
    # the old one says nothing about it (SC-02).
    #
    # Spending the run's allowance here is the same spending as the first
    # screening, so the budget goes back into the state. A revision nobody
    # counted would let a run make more requests than it was allowed.
    going_back = sent_back(state, answers_that_go)
    if not going_back:
        return state

    # Told which run this is, as run_research tells the graph, because the
    # identifier is set for the length of the run and this is called from the
    # screen after it has finished. Without it the model's answers to the
    # send-back were saved as nobody's: on the live run of 18 September the
    # five answers that changed the agent's mind on four papers carried no
    # run, while the run log's lines for the same judgements, which take the
    # run by hand, did (R-27).
    with run_log.while_running(state.run_id):
        judgements, budget = curator.judge_again(state, going_back)
    if isinstance(judgements, OutOfBudget):
        return state.model_copy(update={"budget": budget,
                                        **_stop_because(judgements.reason)})
    if not isinstance(judgements, list):
        return state.model_copy(update={"budget": budget, **_stop_and_ask_for_a_person(
            f"the papers sent back could not be judged again: "
            f"{_why_it_failed(judgements)}")})

    # One judgement for each paper sent, in the order they were sent, which is
    # what the Curation Agent checks before it hands anything back. So the
    # fresh ones are taken in turn as the old ones are passed: the first
    # paper that went back gets the first fresh judgement, and so on.
    fresh = with_the_extracts_checked(judgements)
    updated = []
    next_fresh = 0
    for judged in state.screened:
        if judged.review in answers_that_go:
            replacement = fresh[next_fresh].model_copy(
                update={"review_note": judged.review_note})
            updated.append(replacement)
            next_fresh += 1
        else:
            updated.append(judged)
    # Read back against its own rules, as at the hand-over and at the end of a
    # run. What comes back is built from a model answer, and this is where it
    # passes from the agent to the person again (DC-05).
    return ResearchState.model_validate(
        state.model_copy(update={"screened": updated, "budget": budget}).model_dump())
