"""What a review left: the approved papers of a run, read back."""
# The outcome of a run is the set of papers a reviewer approved, and until 17
# September the screen ended at a green line saying how many were stored,
# and the review of that day could not tell where the papers were. This module puts the outcome
# together for the screen that shows it, and reads it from the two records
# rather than from the session, so a run reviewed yesterday can be opened today
# and the page shows what is kept, not what a session remembers.
#
# Two records hold it, and neither holds all of it. The run log has the
# question, the criteria, the searches and every decision the reviewer made,
# paper by paper, but none of a paper's own words. The store has the
# approved papers with their titles, summaries and extracts, but knows no run:
# a paper is one row whatever run approved it, because a paper approved twice
# is one paper and the store keeps the last decision on it. So a run's
# outcome is put together here: which papers the log says the run approved,
# and what the store now holds for each.
#
# The two can disagree, and the disagreement is shown rather than hidden. A
# paper the log says was approved and the store does not hold is listed under
# its identifier as not in the store. Why it is not there is not said, because
# this run's log cannot tell: a later decision in another run, a store that
# failed, and a session closed after approving without storing all leave the
# same record. An earlier version said "a later decision withdrew it", which
# was wrong for two of the three (found in review on 21 September). Listing
# it keeps the recap honest: the counts are the run's, and the papers are
# what is kept now.
#
# The reviewer's last decision on a paper is the one that counts. A paper
# approved, sent back and then rejected has three lines in the log, and the
# run's outcome is the last of them, which is also what storage kept. A paper
# sent back and not yet decided again is waiting: the send-back is in the log
# as "judged again", and it makes the decision before it history, because the
# screen clears that decision and asks for a fresh one. Until 21
# September the recap read past that line and counted the old decision.
#
# The log names a paper by one identifier and does not say which kind it is, so
# it is matched against all three of the store's columns. A value that is one
# paper's PubMed identifier and another's Semantic Scholar identifier would
# match both, and the first is taken; that case has not been met and is not
# handled here, for the reason storage gives for its own version of it.

from dataclasses import dataclass

from research_agent.identifiers import normalise_doi
from research_agent.processing import run_log, storage


@dataclass(frozen=True)
class ApprovedPaper:
    """One approved paper as the store holds it, for the results screen."""
    # The fields the screen shows and nothing else: the description of the
    # paper, its identifiers for the link, the publication status, the
    # summary the reviewer approved and the note they wrote. The reason,
    # extract and relevance stay in the store; the results screen is the list
    # a person reads and uses, and the judgement was read on the review page.
    # The status travels because a retraction is the one thing about a paper
    # a reader of the list must not have to go back for; it was left
    # behind here until 22 September, so a stored retraction reached the
    # review page and not the results page or the file (found in review).

    identifier: str
    doi: str
    pmid: str
    paper_id: str
    title: str
    authors: str
    year: str
    journal: str
    publication_status: str
    summary: str
    review_note: str


@dataclass(frozen=True)
class LeftOut:
    """One paper the reviewer flagged or rejected, as the log names it."""
    # The identifier, the decision and the note, and nothing of the paper's
    # words: the log holds none, and the store holds only approved
    # papers. The screen adds the title when the run is still in the session.

    identifier: str
    decision: str
    note: str


@dataclass(frozen=True)
class ReviewOutcome:
    """What one run asked, searched, decided and now has stored."""
    # `searches` are lines already worded for a reader; the counts are of the
    # reviewer's last decisions; `papers` are the approved ones the store
    # holds, in the order the reviewer first decided them, which is the only
    # order the record has; `not_in_the_store` names, by identifier, the ones
    # the log says were approved and the store does not hold; `left_out`
    # are the flagged and rejected papers with the reviewer's notes, for the
    # reader who asks what was left out and why; `waiting` counts the papers
    # sent back and not decided again; `stored` says whether the run's latest
    # store stands, with no decision or send-back after it, because a run is
    # offered on the strength of an approval and an approval is written
    # before the store, so a run whose store failed or never happened has
    # approvals and nothing kept, and a run decided further after its store
    # has a kept set older than its counts; `finished` is when the review
    # ended, the time of the last store, decision or send-back.

    run: str
    started: str
    finished: str
    question: str
    include: str
    exclude: str
    searches: list
    approved: int
    flagged: int
    rejected: int
    waiting: int
    stored: bool
    papers: list
    not_in_the_store: list
    left_out: list


def last_decisions(entries):
    """Each paper's last decision and note in a run, by the log's identifier."""
    # The last rather than the first, because a paper sent back and decided
    # again has two lines, and the second is the decision that stands. A
    # dictionary keeps the order the papers were first decided in. The note
    # travels with the decision because the results page shows, for a paper
    # left out, why the reviewer left it out.
    #
    # The "judged again" line between the two makes the earlier decision
    # history: from that line until the next decision the paper is waiting,
    # which is what the review page shows for it. Read past, the line left a
    # rejected paper counted as rejected, with its old note, while the
    # reviewer had not yet said (found in review on 21 September).
    last = {}
    for entry in entries:
        paper = entry.get("paper", "no identifier")
        if entry.get("event") == "reviewer decision":
            last[paper] = (entry.get("decision", ""), entry.get("note", ""))
        if entry.get("event") == "judged again":
            last[paper] = ("waiting", "")
    return last


def the_papers_were_stored(entries):
    """Whether the run's latest store stands: nothing decided or sent back since."""
    # An approval is written to the log at the press, and the store comes
    # after it, so a run can hold approvals and no store: the store refused
    # the batch, or the session was closed first. And a store can be followed
    # by more: Back, another decision, a send-back, and no second store. Then
    # what is kept is the earlier set, not the decisions the page counts, so
    # the review is not complete either. Any store at all was read as
    # complete at first (found in review on 22 September). The results page
    # calls the review complete only when the last of these acts is a store.
    stored = False
    for entry in entries:
        if entry.get("event") == "approved papers stored":
            stored = True
        if entry.get("event") in ("reviewer decision", "judged again"):
            stored = False
    return stored


def when_the_review_finished(entries):
    """The time the review ended, as the log has it."""
    # The time of the last decision, send-back or store, whichever the log
    # holds last, because each is the reviewer's act on the run; before any,
    # the run's start. A run reopened tomorrow says when it finished, not
    # when it was reopened. The send-back counts since 22 September: a paper
    # sent back after the store is the reviewer's last act, and the page
    # would otherwise date the review before it.
    finished = ""
    for entry in entries:
        if "at" not in entry:
            continue
        if entry.get("event") == "run started" and not finished:
            finished = entry["at"]
        if entry.get("event") in ("reviewer decision", "judged again",
                                  "approved papers stored"):
            finished = entry["at"]
    return finished


def searches_made(entries):
    """One line for each search a run made: where, what, and what came back."""
    # Worded here rather than on the screen, because the fields are the log's
    # and a screen naming them would drift from the log the day a field was
    # renamed. What is not recorded is said to be not recorded, as `as_lines`
    # does, rather than shown as a blank a reader would take for an empty
    # query.
    lines = []
    for entry in entries:
        if entry.get("event") != "search":
            continue
        found = entry.get("papers found")
        if found is None:
            found = "papers found not recorded"
        elif found == 1:
            found = "1 paper found"
        else:
            found = f"{found} papers found"
        lines.append(f"{entry.get('source', 'source not recorded')}: "
                     f"{entry.get('query', 'query not recorded')}, "
                     f"{entry.get('result', 'result not recorded')}, {found}")
    return lines


def runs_with_approvals(log=None):
    """The runs whose outcome there is: those with at least one approval."""
    # Newest first, as `run_log.runs` gives them. A run with no approval has
    # nothing stored and no list to show, and offering it would open a page
    # that says so and nothing else.
    offered = []
    for recorded in run_log.runs(log):
        decisions = [choice for choice, note in last_decisions(recorded.entries).values()]
        if "approved" in decisions:
            offered.append(recorded)
    return offered


def _row_for(identifier, rows):
    """The stored row the log's identifier names, or None."""
    # The DOI is normalised the way storage normalised it when the row was
    # written: the log keeps the DOI as the source spelt it, and a paper found
    # under "https://doi.org/10.1/ABC" is stored under "10.1/abc".
    as_doi = normalise_doi(identifier)
    for row in rows:
        if row["doi"] and row["doi"] == as_doi:
            return row
        if row["pmid"] and row["pmid"] == identifier:
            return row
        if row["paper_id"] and row["paper_id"] == identifier:
            return row
    return None


def outcome_of(run, log=None, store=None):
    """The outcome of one run, or None when the log has no run by that name."""
    # Read from the log and the store each time it is asked for, not kept:
    # the store changes whenever any run stores, and a page showing yesterday's
    # list should show what is kept today.
    #
    # The counts are of the decisions the log holds. The panel on the review
    # page counts the session's answers instead, and the two agree except when
    # a decision could not be written to the log, which the review page says
    # at the moment it happens.
    recorded = None
    for one in run_log.runs(log):
        if one.run == run:
            recorded = one
            break
    if recorded is None:
        return None

    # The first "run started" line is the request; a run has one.
    asked = {}
    for entry in recorded.entries:
        if entry.get("event") == "run started":
            asked = entry
            break
    decisions = last_decisions(recorded.entries)
    counted = [choice for choice, note in decisions.values()]

    rows = storage.every_row(store)
    papers = []
    not_in_the_store = []
    left_out = []
    for identifier, (decision, note) in decisions.items():
        if decision in ("flagged", "rejected"):
            left_out.append(LeftOut(identifier, decision, note))
        if decision != "approved":
            continue
        row = _row_for(identifier, rows)
        if row is None:
            not_in_the_store.append(identifier)
            continue
        papers.append(ApprovedPaper(
            identifier=row["identifier"], doi=row["doi"], pmid=row["pmid"],
            paper_id=row["paper_id"], title=row["title"], authors=row["authors"],
            year=row["year"], journal=row["journal"],
            publication_status=row["publication_status"], summary=row["summary"],
            review_note=row["review_note"]))

    return ReviewOutcome(
        run=recorded.run,
        started=recorded.started,
        finished=when_the_review_finished(recorded.entries),
        question=recorded.question,
        include=asked.get("include", ""),
        exclude=asked.get("exclude", ""),
        searches=searches_made(recorded.entries),
        approved=counted.count("approved"),
        flagged=counted.count("flagged"),
        rejected=counted.count("rejected"),
        waiting=counted.count("waiting"),
        stored=the_papers_were_stored(recorded.entries),
        papers=papers,
        not_in_the_store=not_in_the_store,
        left_out=left_out)
