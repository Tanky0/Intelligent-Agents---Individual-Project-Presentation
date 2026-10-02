"""Show the reviewer's screen with invented papers in a real browser, spending
nothing.
"""
# The interface checks need papers on the screen in known states: a complete
# run, a partial one, an empty one, a failed one, and a full set of fifty,
# each with the coverage warning that state produces. The six interface
# requirements and the acceptance baseline of 13 September ask for those
# states, and both are in `docs/testing.md`.
#
# A live run cannot be asked for those on demand. It spends the day's model
# allowance (OB-07).
#
# It also returns real abstracts that a screenshot could carry into the
# evidence folder, which is what the release boundary exists to stop (R-12).
#
# The functional tests already put invented papers through the real screen
# with AppTest; this puts them through a real browser, which is the part a
# person has to look at.
#
# Three things are replaced before the screen starts, and everything else is
# the application as it is. `graph.run` returns the chosen state instead of
# searching, so pressing "Start the run" shows it at once, whatever is typed
# in the form. `graph.judge_again` gives the papers a reviewer sends back a
# fresh judgement without a model, keeping the note as the real route does, so
# the revision journey can be walked. The store and the run log are files
# under a temporary folder, so nothing this session writes reaches the real
# store or the real log, which the baseline asks for. The stand-ins are put
# on the modules the screen calls through, which works because the screen
# calls them through their modules rather than by names bound at import time.
#
# Every paper is invented and says so in its title. The identifiers are nine
# digits beginning with 91, kept apart from the functional fixtures' 9000 and
# from any recording; whether PubMed has issued them is not checked or relied
# on. What keeps them out of the class of thing R-12 was about is that the
# text is invented and this docstring says so.
#
# Usage, from the repository root, with the project's interpreter:
#
#     python scripts/show_synthetic_review_states.py complete
#     python scripts/show_synthetic_review_states.py full --on-the-network
#
# The second form makes the server answer on every address of this computer,
# so a phone on the same network can open it; the addresses are printed.

import argparse
import pathlib
import socket
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent import graph
from research_agent.config import REPO_ROOT
from research_agent.processing import run_log, storage
from research_agent.processing.validation import with_the_extracts_checked
from research_agent.schemas import (
    NONE_IDENTIFIED,
    RETRACTED,
    Outcome,
    PaperRecord,
    ResearchState,
    ResultsBuffer,
    RunBudget,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
)

APP = REPO_ROOT / "research_agent" / "ui" / "app.py"

# The sentences the invented abstracts are built from. Each extract below is
# one of these copied word for word, so the extract check finds it, except
# where a paper is meant to show what a refused extract looks like.
SENTENCES = [
    "Forty patients with condition Y were given drug X for eight weeks.",
    "Symptoms improved in most of them by the end of the study.",
    "A cohort of one hundred people taking drug X was followed for a year.",
    "Two stopped treatment because of nausea in the first month.",
    "No new patients were studied and the findings are drawn from earlier reports.",
    "The dose was halved after the first week because of dizziness.",
]
NOT_IN_ANY_PAPER = "This sentence appears in no invented paper."

TOPICS = [
    "a randomised trial of drug X in condition Y",
    "a cohort study of drug X",
    "a narrative review of drug X",
    "adverse events of drug X in a small cohort",
    "drug X compared with usual care",
]


def invented_paper(number):
    """One invented paper, numbered so a reviewer can tell them apart."""
    # A few numbers carry a deliberate feature, so that the full set exercises
    # the layout and the report: 7 has a title long enough to wrap on a phone,
    # 8 an abstract long enough to make the page scroll, 9 no abstract at all,
    # 10 a retraction, and 12 no DOI. They are fixed numbers rather than random
    # ones so that two people looking at the same state see the same page.
    topic = TOPICS[number % len(TOPICS)]
    title = f"Invented paper {number}: {topic}"
    if number == 7:
        title = (f"Invented paper {number}: {topic}, with a title long enough "
                 "to wrap onto several lines on a narrow screen and to show "
                 "whether the identifier beside it stays readable")
    first = SENTENCES[number % len(SENTENCES)]
    second = SENTENCES[(number + 1) % len(SENTENCES)]
    abstract = f"Invented abstract {number}. {first} {second}"
    if number == 8:
        abstract = f"Invented abstract {number}. " + " ".join(SENTENCES * 8)
    text_source = "abstract"
    if number == 9:
        abstract = ""
        text_source = "metadata"
    status = NONE_IDENTIFIED
    if number == 10:
        status = RETRACTED
    doi = f"10.99999/invented.{number}"
    if number == 12:
        doi = ""
    return PaperRecord(
        title=title,
        abstract=abstract,
        text_source=text_source,
        authors=[f"Author {number} A", f"Author {number} B"],
        year=str(2015 + number % 10),
        journal="Journal of Invented Studies",
        publication_types=["Journal Article"],
        pmid=f"91{number:07d}",
        doi=doi,
        publication_status=status,
        origins=[SearchOrigin(source="pubmed", query="drug X condition Y")])


def judgement(paper, number, position):
    """The Curation Agent's judgement on one invented paper, in rank order."""
    # The score falls by one every five papers and the decision cycles include,
    # include, uncertain, uncertain, exclude inside each five, which is the
    # order ranking would put them in, so the list reads as a run's
    # output does. Paper 9 has no abstract and its extract is the prompt's own
    # placeholder, which is what the model returned for every such record on
    # the standard-dataset run; paper 11 is left never checked.
    decisions = ["include", "include", "uncertain", "uncertain", "exclude"]
    decision = decisions[(position - 1) % 5]
    relevance = max(0, 10 - (position - 1) // 5)
    extract = SENTENCES[number % len(SENTENCES)]
    if number == 9:
        extract = "(no abstract was retrieved)"
    if number == 13:
        extract = NOT_IN_ANY_PAPER
    screening = Screening(
        paper_number=position, decision=decision, relevance=relevance,
        reason=f"an invented reason for paper {number}, in one sentence",
        summary=f"an invented summary of paper {number}, built from its own wording",
        extract=extract)
    judged = ScreenedPaper(paper=paper, screening=screening)
    if number == 11:
        return judged
    return with_the_extracts_checked([judged])[0]


def finished(query, papers_found):
    return Outcome(subgoal=Subgoal(purpose=f"find {query}", source="pubmed",
                                   query=query),
                   result="finished", papers_found=papers_found)


def a_state(request, outcomes, papers, screened, **more):
    """A finished run in the shape `graph.run` returns one."""
    # The buffer is marked ready and keeps the papers before deduplication,
    # because that is what the hand-over leaves behind and what the coverage
    # report reads its counts from.
    budget = RunBudget(model_requests_used=2 + (len(screened) + 9) // 10,
                       papers_sent_to_curation=len(screened))
    values = {
        "request": request,
        "run_id": run_log.new_identifier(),
        "results": ResultsBuffer(outcomes=outcomes, papers=papers, ready=True,
                                 papers_before_deduplication=list(papers)),
        "screened": screened,
        "budget": budget,
        "steps_taken": len(outcomes),
        "step_limit": max(len(outcomes), 1),
    }
    values.update(more)
    return ResearchState(**values)


def complete_run(request):
    papers = [invented_paper(number) for number in range(1, 4)]
    screened = [judgement(paper, number, number)
                for number, paper in enumerate(papers, start=1)]
    return a_state(request, [finished("drug X condition Y", 3)], papers, screened)


def partial_run(request):
    """One search finished, one came back short, one failed, one never ran,
    and one publication-status check failed, so every kind of shortfall the
    report can name is on the page at once."""
    papers = [invented_paper(number) for number in range(1, 4)]
    screened = [judgement(paper, number, number)
                for number, paper in enumerate(papers, start=1)]
    outcomes = [
        finished("drug X condition Y", 2),
        Outcome(subgoal=Subgoal(purpose="find cohorts of drug X",
                                source="semantic scholar", query="drug X cohort"),
                result="partial", papers_found=1,
                reason="the service returned 100 of the 1000 asked for"),
        Outcome(subgoal=Subgoal(purpose="find reports of harm", source="pubmed",
                                query="drug X harms"),
                result="failed", papers_found=0,
                reason="PubMed answered 503 three times"),
    ]
    never_reached = [Subgoal(purpose="find drug X compared with usual care",
                             source="pubmed", query="drug X usual care")]
    state = a_state(request, outcomes, papers, screened,
                    pending_searches=never_reached,
                    stopped_because="the run reached its limit of 3 searches")
    state.results.status_checks_failed.append(
        "10.99999/invented.2: Crossref answered 500")
    return state


def empty_run(request):
    """Every search ran and none found anything."""
    return a_state(request, [finished("drug X condition Y", 0),
                             finished("drug X cohort", 0)], [], [])


def failed_run(request):
    """Every search failed and nothing was retrieved, so nothing was screened."""
    # Until 19 September this state also asked for a person, with an invented
    # reason saying every search had failed. A real run does not ask for one
    # here: the failed searches are on the report's own list, and screening an
    # empty buffer makes no request. The second session read the
    # invented line as one more sentence written for the developer and asked
    # for it to go (UX-06); what goes is the invention, not the record's line
    # for a run that did ask for a person.
    outcomes = [
        Outcome(subgoal=Subgoal(purpose="find trials of drug X", source="pubmed",
                                query="drug X condition Y"),
                result="failed", papers_found=0,
                reason="PubMed answered 503 three times"),
        Outcome(subgoal=Subgoal(purpose="find cohorts of drug X",
                                source="semantic scholar", query="drug X cohort"),
                result="failed", papers_found=0,
                reason="the connection was refused"),
    ]
    return a_state(request, outcomes, [], [])


def full_run(request):
    """Fifty papers, the most one run gives curation, with four more
    beyond the allowance and three repeated records combined away, so the
    report's counts are all non-trivial."""
    papers = [invented_paper(number) for number in range(1, 51)]
    screened = [judgement(paper, number, number)
                for number, paper in enumerate(papers, start=1)]
    beyond = [invented_paper(number) for number in range(51, 55)]
    repeated = [invented_paper(number) for number in range(1, 4)]
    state = a_state(request, [finished("drug X condition Y", 40),
                              finished("drug X cohort", 12),
                              finished("drug X harms", 5)],
                    papers, screened)
    state.results.papers_beyond_the_screening_limit.extend(beyond)
    state.results.papers_before_deduplication.extend(beyond + repeated)
    state.results.status_checks_failed.append(
        "10.99999/invented.5: Crossref answered 500")
    return state


STATES = {
    "complete": complete_run,
    "partial": partial_run,
    "empty": empty_run,
    "failed": failed_run,
    "full": full_run,
}

# Which state the next "Start the run" shows. Set once in main and read by
# the stand-in below, because the screen calls graph.run with the request
# and the agents and nothing else, so the choice has to be held here.
CHOSEN_STATE = "complete"


def run_without_searching(request, planner, curator, step_limit=None):
    """Stands in for `graph.run`: the chosen state, written to the log as a
    run would be, so the screen's log section has lines for it."""
    state = STATES[CHOSEN_STATE](request)
    run_log.note_the_request(request, state.run_id)
    for outcome in state.results.outcomes:
        run_log.note_a_search(outcome, request, state.run_id)
    return state


def judge_again_without_a_model(state, curator, answers_that_go=("flagged", "rejected")):
    """Stands in for `graph.judge_again`: the papers sent back get a fresh
    judgement that answers the note, come back waiting, and keep the note.
    """
    # The decision is turned round so the change is visible. One simulated
    # request is charged to exercise the screen's count; this is not a measure
    # of the real route, which batches ten papers and may ask again.
    # Which answers go is the screen's choice since 18 September, the flagged
    # alone or the flagged and the rejected, passed as the real route takes it.
    #
    # The fresh reason keeps the old one and adds that the paper was judged
    # again after the note. The review of the screen on 17 September found
    # the earlier stand-in, which replaced the reason with the note alone,
    # left the rating with nothing to rest on: the real route returns a
    # reason for the rating, and the screen shows the reason for that. The
    # two parts are separated by a full stop and the note is quoted, since
    # the second session of 19 September read them run together as
    # one sentence; on a real run the reason is the model's own sentence and
    # carries no note.
    going_back = graph.sent_back(state, answers_that_go)
    if not going_back:
        return state
    turned_round = {"include": "exclude", "exclude": "include",
                    "uncertain": "include"}
    updated = []
    for judged in state.screened:
        if judged not in going_back:
            updated.append(judged)
            continue
        fresh_screening = judged.screening.model_copy(update={
            "decision": turned_round[judged.screening.decision],
            "reason": (f"{judged.screening.reason}. Judged again after the "
                       f"reviewer wrote: \"{judged.review_note}\".")})
        fresh = judged.model_copy(update={"screening": fresh_screening,
                                          "review": "waiting"})
        updated.append(with_the_extracts_checked([fresh])[0])
    return state.model_copy(update={
        "screened": updated,
        "budget": state.budget.with_one_model_request_used()})


def keep_this_session_apart(folder):
    """Points the store and the run log at files under `folder`."""
    # The screen names neither file, and both modules look their file up at
    # the moment of use, so moving the two names moves everything that reads
    # or writes them; the functional tests move them the same way.
    #
    # The store is moved by its name and not by wrapping `save_approved`. The
    # wrapper this used to be moved the write alone: the results page read
    # the store by its name, found the real file empty, and said on
    # 17 September that a paper just approved was no longer stored.
    storage.DEFAULT_STORE = folder / "approved_records.sqlite3"
    run_log.DEFAULT_LOG = folder / "run_log.jsonl"


def addresses_of_this_computer():
    """Every address a phone on the same network could use, best effort."""
    try:
        return socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []


def main():
    global CHOSEN_STATE
    parser = argparse.ArgumentParser(
        description="Show the reviewer's screen with invented papers.")
    parser.add_argument("state", choices=sorted(STATES))
    parser.add_argument("--on-the-network", action="store_true",
                        help="answer on every address of this computer, so a "
                             "phone on the same network can open the screen")
    args = parser.parse_args()

    CHOSEN_STATE = args.state
    folder = pathlib.Path(tempfile.mkdtemp(prefix="synthetic-review-"))
    keep_this_session_apart(folder)
    graph.run = run_without_searching
    graph.judge_again = judge_again_without_a_model

    print(f"State: {args.state}. Type anything in the form and press Start "
          "the run; no service or model is called.")
    print(f"The store and the run log for this session are under {folder}.")
    options = {"browser.gatherUsageStats": False}
    if args.on_the_network:
        options["server.address"] = "0.0.0.0"
        for address in addresses_of_this_computer():
            print(f"On the phone, open http://{address}:8501")

    from streamlit.web import bootstrap
    bootstrap.load_config_options(flag_options=options)
    bootstrap.run(str(APP), False, [], options)


if __name__ == "__main__":
    main()
