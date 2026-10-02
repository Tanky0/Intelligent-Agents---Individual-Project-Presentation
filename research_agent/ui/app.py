"""The screen the work is asked for and reviewed at."""
# A run starts here. The researcher writes the question and the two sets of
# criteria, and the screen builds the request the run is given. The criteria are
# theirs to write rather than the agent's to work out, because they are the part
# a systematic review has to state and defend afterwards.
#
# The decisions are plain functions and the screen calls them. Streamlit reruns
# the whole script on every click, so a decision written inside a button's branch
# is made again on the next rerun or not at all, and neither is easy to see. The
# functions below are what a test can hold still; the drawing is checked
# separately with Streamlit's own AppTest, which renders the script and presses
# its buttons without a browser.
#
# Approving, flagging and rejecting are three answers and not two. A flag says
# the paper needs another look rather than that it is wrong, and losing that
# distinction would make a reviewer choose between keeping a doubtful paper and
# throwing it out.
#
# A flag or a rejection has to carry a note. The Curation Agent is given it when
# the paper comes back for revision, and "rejected" with nothing written is a
# decision nobody can act on or disagree with. An approval needs no
# note: the judgement it approves is already written down.
#
# A paper the reviewer flagged or rejected can be sent back to the Curation
# Agent, which judges it again knowing what they wrote. It returns waiting for a
# decision, and the answer they gave before is cleared, because the design asks
# for a fresh decision before a revised record is stored. The old answer was made
# about a judgement that has been replaced, so left in place it would stand as
# the reviewer's decision on something they have not read.
#
# Decisions are kept in Streamlit's session state, filed under the paper's
# identifier. A rerun happens on every click, so anything held in an ordinary
# variable would be built again from the run and the reviewer would watch their
# answers disappear as they worked. They were filed by position once, and
# which_paper says why that stopped. They are cleared when the run on the
# screen is not the one they were made about, judged by its identifier and its
# papers together, because the same papers can come back from a second run of
# the same question with judgements nobody has read (R-20).
#
# The screen also shows what a run did, read back from the run log, on
# a page of its own reached from the panel. The coverage report describes the
# state a run finished in; the log is the order things happened in and the
# time each took, which is the part a reader needs when they want to know how
# the papers in front of them were found. Any run in the log can be chosen and
# not only the one on the screen, because the log outlives the session that
# wrote it.
#
# After a store the store accepts, the screen moves on to the outcome: the
# approved papers as the store holds them, under a recap of the run read from
# the log. It is read from the two records rather than from the
# session, so the outcome of an earlier run can be opened from the screen the
# form is on; and it offers the way back to the papers, whose decisions stand,
# and the way to a new search, without a reload.
#
# NCBI's disclaimer and copyright notice are shown as NCBI publishes them. The
# wording is kept in `docs/development.md` and repeated here, with a test
# that reads the document and fails if the two drift apart. The document the
# test reads has to be one a release carries, or the released clone fails a
# test on a file it does not hold. The copy is here because a screen should
# not read its legal wording out of a documentation file at the moment it
# draws it, where a missing or edited file would quietly remove the notice.
# Writing it from memory is what the record exists to prevent (OB-03).

import csv
import datetime
import io
from urllib.parse import quote

from pydantic import ValidationError

from research_agent import graph, llm
from research_agent.agents import curator, planner
from research_agent.processing import audit_log, results, run_log, storage
from research_agent.schemas import ResearchRequest, ResearchState, RunBudget

# Shown as NCBI publishes it, not paraphrased. The link goes with it because
# the notice covers material that is not in the public domain, including
# PubMed Central, and a reader should be able to reach the whole of it rather
# than the two paragraphs quoted here (OB-03).
NCBI_DISCLAIMER = (
    "For documents and software available from this server, the U.S. "
    "Government does not warrant or assume any legal liability or "
    "responsibility for the accuracy, completeness, or usefulness of any "
    "information, apparatus, product, or process disclosed."
)
NCBI_COPYRIGHT = (
    "Information that is created by or for the US government on this site is "
    "within the public domain. Public domain information on the National "
    "Library of Medicine (NLM) Web pages may be freely distributed and copied."
)
NCBI_NOTICE_URL = "https://www.ncbi.nlm.nih.gov/About/disclaimer.html"

# Where the screen keeps the reviewer's answers between reruns, the run they
# are drawn from, and what they were made about: the run's identifier and the
# list of its papers. Paper keys alone do not name an execution. A later run
# of the same question can return the same papers with new judgements, and
# while the keys were the whole check a second run of the same papers kept
# the first run's answers. In a controlled check, the second run inherited
# the earlier approval and note, and they were passed to the storage call
# with a judgement that had not been approved in that run (R-20). The
# identifier is on the state for the run log's sake and is read here for
# the same reason: a question does not name a run.
_DECISIONS = "decisions"
_RUN = "run"
_WHOSE = "decisions belong to"
# What to tell the reviewer on the next drawing, because a revision, and
# since the cards a decision too, redraws the page they would otherwise have
# been written on. The second holds the warnings for what the log could not
# write while the act was made, which have to travel with the redrawing: a
# page redrawn without them would tell the reviewer the act was recorded
# when it was not.
_CAME_BACK = "what came back"
_LOG_TROUBLE = "what the log could not write"
# What the agent had said about each paper that was sent back, by paper,
# kept for the rest of the run: the decision and the rating before the
# second judgement. A paper that came back stays on the page whatever its
# new rating, so the reviewer sees the fresh judgement they asked for; what
# changed is drawn from this too (the review's findings of 18 September).
_CAME_BACK_FROM = "what the agent had said about the papers sent back"
# Why the last press of the store button stored nothing, said at the top of
# the review page on the drawing after it, because under the button it was
# out of sight.
_STORE_FAILED = "why storing failed"
# Which page the session is on, and which run the results page is about.
# The review page is the default, so the first is absent until a store the
# store accepts, or until a reviewer asks for an earlier run's outcome.
_PAGE = "page"
_RESULTS_OF = "results of run"
# Where the record of runs was opened from, so its Back returns there: the
# review page or the form (None), or the results page.
_RECORD_FROM = "the record was opened from"
# Set by every move from one page to another, and read once by the drawing
# that follows. Streamlit keeps the page's scroll position across a rerun,
# so a page came into view wherever the last one had been scrolled to: the
# empty form at its foot after the results page (the review of the
# screen on 17 September), the papers on the fourth card after the form's "Start
# the run" button (picture 21 of the same evening). The second key
# counts the moves, so that each frame drawn differs from the last.
_TO_THE_TOP = "scroll to the top"
_MOVES = "moves to the top so far"
# Set by a first press of "Search" on the progress line while a run is on
# the screen, and read on the next drawing: a second press is the
# confirmation decided on 18 September, and any other press
# withdraws it, since the flag is taken off the session as soon as it is read.
_ASKED_TO_LEAVE_FOR_SEARCH = "asked to go back to Search"


class ReviewRefused(Exception):
    """Something the screen will not do: a decision, or a request to run."""
    # Raised rather than returned, for the same reason storage raises: the
    # caller asked for something to be done and it was not, and a return value
    # saying so is easy to draw past.


# What the three answers do, and what sending back brings, said once. In
# the session of 16 September, working through the review tasks, the
# reviewer could not tell what "flagged" meant or how it differed from "rejected", and the page
# explained neither (UX-02). Said once rather than under every set of
# buttons, because fifty copies of the same line would bury the papers it is
# there to help with. It sat above the first paper until 17 September, when
# it was moved to the side panel, above the panel's buttons: the
# panel stays in view at every paper, so the line is there when the reviewer
# is at paper forty and not only at paper one. The last entry is theirs
# too: the send-back button had nothing near it saying what comes back.
#
# Three headed entries rather than one sentence, since the reading
# of the afternoon of 17 September: one line holding every answer had to be
# read through to find the one wanted. The words are the ones given for
# it that evening, exactly as given, with
# British spelling: they know that a rejected paper is judged again only
# when Send back is pressed, and chose this wording anyway, so nothing is
# added to it and there is no Send back entry. A longer version with the
# code's facts in it ran to fourteen lines and pushed the panel's buttons
# below the fold of a 1179 by 731 window; each entry stays short, which the
# test holds it to, and whether the buttons are in view on that window
# is for a session to see.
#
# The sentences are held once, by the answer each is about, and the panel's
# headed entries are made from them: since 18 September the same sentence
# is also each answer button's hover text, as decided, and two
# copies of the sentences would be two places for them to drift apart.
WHAT_EACH_ANSWER_DOES = {
    "approved": "Use when the paper is relevant and should be kept.",
    "flagged": ("Use when you are unsure and want the system to provide a "
                "second judgement. Add a comment to guide the system's new "
                "assessment."),
    "rejected": ("Use when the paper does not match your query and should be "
                 "reassessed by the system. Add a comment explaining why it "
                 "should be excluded."),
}
WHAT_THE_ANSWERS_DO = (
    "**Approve:** " + WHAT_EACH_ANSWER_DOES["approved"],
    "**Flag:** " + WHAT_EACH_ANSWER_DOES["flagged"],
    "**Reject:** " + WHAT_EACH_ANSWER_DOES["rejected"],
)

# The line under every screen's title; why it says what it says is in main.
A_REVIEWER_DECIDES = ("A reviewer decides what is kept. Nothing is stored until "
                      "a reviewer approves it.")

# The three steps a review goes through, in order, as the progress line at
# the top of each screen names them: the words given on 18 September.
THE_STEPS = ("Search", "Assessment", "Result")


def decide(judged, choice, note=""):
    """Records what the reviewer decided about one paper."""
    # A new record rather than a changed one. The judgement is evidence of what
    # the model said and the decision is evidence of what a person made of it,
    # and editing the first in place would leave no way to tell them apart.
    #
    # An approval is refused when the paper's extract was not found in the
    # paper or was never looked for. The store refuses such a paper anyway,
    # because a stored claim has to be traceable to the paper's own words,
    # and it refuses the whole batch with it; until 17 September that was the
    # first the reviewer heard of it, at the store button, with the paper
    # named by its identifier among fifty. It was met that evening on
    # the invented papers, one of which is never checked on purpose. The
    # refusal belongs where the reviewer is (UX-02), at the paper, with what
    # to do instead: flag it, so the Curation Agent returns a new extract, or
    # reject it.
    if choice not in ("approved", "flagged", "rejected"):
        raise ReviewRefused(
            f"{choice!r} is not one of the three answers a reviewer gives")
    if choice in ("flagged", "rejected") and not note.strip():
        raise ReviewRefused(
            f"a paper that is {choice} has to say why: the note is what the "
            "Curation Agent is given when the paper comes back, and what "
            "anybody reading the review afterwards has to go on")
    if choice == "approved" and judged.extract_found is None:
        raise ReviewRefused(
            "this paper cannot be approved: its extract was never looked for "
            "in the paper, so a stored claim could not be traced back to it. "
            "Flag it for a second judgement, or reject it")
    if choice == "approved" and not judged.extract_found:
        raise ReviewRefused(
            "this paper cannot be approved: its extract was not found in the "
            "paper, so a stored claim could not be traced back to it. Flag it "
            "for a second judgement, or reject it")
    return judged.model_copy(update={"review": choice, "review_note": note})


def _why_the_request_was_refused(refusal):
    """The record's own reason, without the wording around it."""
    # The reason comes wrapped in the record's own words about where in the
    # record it happened, and a researcher needs the reason rather than the
    # place, so the wrapper is taken off.
    #
    # The reasons are joined because they arrive as a list, not because a blank
    # request produces more than one. The record stops at the first field it
    # finds blank, so a form with nothing filled in is refused for its question
    # alone and the researcher meets the next reason after they have answered
    # that one.
    return "; ".join(problem["msg"].removeprefix("Value error, ")
                     for problem in refusal.errors())


def what_was_asked_for(question, inclusion, exclusion, results_per_search=None):
    """Builds the request a run is given, or says what is wrong with it."""
    # The criteria are the researcher's to write, not the agent's to work out,
    # because they are the part a systematic review has to state and defend
    # afterwards. The record is what refuses a blank question or blank criteria,
    # and this passes that refusal on rather than repeating the rules: two
    # places deciding what a usable request is would drift, and the screen would
    # be the one that was wrong, because the record is what the run reads.
    #
    # The years a search may be bounded by are deliberately not offered. They
    # are meant for testing, and an ordinary run leaves them unset so that the
    # search matches the hand-screened sample the recall figure is measured
    # against. A box for them on the screen would invite a researcher to narrow
    # a run in a way that quietly makes that figure mean something else
    # (SC-01).
    #
    # The number of results asked of each search is offered, since 18
    # September, for a quick run that need not fill the fifty places
    # screening has; left alone it is the record's own fifty, this project's
    # default. That is not the setting the recall figure describes, since the
    # measurement of 19 September asked 1,000 of each search (SC-01). A
    # comment here credited the default with it until 22 September (found in
    # review). None means the default.
    try:
        if results_per_search is None:
            return ResearchRequest(question=question, inclusion_criteria=inclusion,
                                   exclusion_criteria=exclusion)
        return ResearchRequest(question=question, inclusion_criteria=inclusion,
                               exclusion_criteria=exclusion,
                               results_per_search=results_per_search)
    except ValidationError as refusal:
        raise ReviewRefused(_why_the_request_was_refused(refusal)) from refusal


def still_waiting(screened):
    """The papers nobody has decided about yet."""
    # Not used by the screen, which draws every paper whether or not it has an
    # answer, so that a reviewer can change one before the papers are stored.
    # It is kept for the reviewer who leaves the room half way through and
    # should come back to the papers they had not reached rather than to the
    # beginning; the screen does not offer that yet, and this is the part of it
    # a test can hold still.
    return [item for item in screened if item.review == "waiting"]


def what_to_warn_about(report):
    """The lines the screen must show about coverage, or nothing."""
    # The warning is built from the coverage report rather than written here, so
    # a reader sees the same words on screen as in the run's own record. Two
    # renderings would drift, and the one nobody looked at would be the one that
    # was wrong. The screen draws them as two groups side by side
    # (`_draw_the_coverage`); the words are the same, and this is what says
    # whether there is anything to draw.
    #
    # Nothing is returned when the coverage was complete and nothing failed
    # quietly. A warning shown every time is a warning nobody reads.
    #
    # A run whose failures nobody noted is warned about rather than passed. Not
    # knowing and nothing being wrong are different, and the reviewer is the
    # person who should be told which they have. Asking only whether something
    # went wrong got this the other way round, because an unknown answer is not
    # a yes.
    if report.everything_was_searched and report.anything_went_wrong_quietly is False:
        return []
    # Called through the module because the run log renders its own lines under
    # the same name, and a reader of this file should be able to see which of
    # the two is meant without looking at the imports.
    return audit_log.as_lines(report)


def what_to_do_next(report, papers_found):
    """One sentence saying what a person can do about a run's shortfall."""
    # The coverage report says what was missed, in the run's own record, and
    # the review of the partial and failed states on 16 September
    # could say what was missed and not what to do about it (UX-06). The
    # record cannot carry this sentence: it is advice to a reader, not a fact
    # about the run, and the log keeps facts. So it is written here, on the
    # screen, under the record's lines.
    #
    # The sentence must not promise what a rerun cannot do. A search that
    # failed at a service may answer later and may not, so it says "later" and
    # not "again and it will work". "Failed" rather than "refused", since the
    # second session of 19 September: a service that timed out did not
    # refuse anything, and the report's own list calls these searches failed.
    #
    # Whether there are papers is decided first and on its own. Until 22
    # September it was asked only inside the branch for a short search, so a
    # run whose every search finished, found nothing and had a write failure
    # was told "the papers can be reviewed" over an empty page (found in
    # review). And "not known" is kept apart from "listed above": a report
    # with no failure snapshot cannot say that anything failed quietly, only
    # that nobody recorded whether it did.
    if papers_found:
        first = "What to do next: the papers here can be reviewed now."
    else:
        first = "What to do next: nothing here needs a decision."
    if not report.everything_was_searched:
        if papers_found:
            second = ("To reach what was missed, run the same question again "
                      "later, or with fewer searches.")
        else:
            second = ("Run the same question again later; a service that "
                      "failed may answer then.")
    elif report.anything_went_wrong_quietly is None:
        second = ("Whether anything failed quietly is not known: the run's "
                  "failure lists were not recorded with it.")
    else:
        second = "What failed quietly is listed above and did not stop the run."
    return f"{first} {second}"


def why_there_is_nothing_to_review(report):
    """The sentence under the headline when a run left no paper to review."""
    # Three runs end with nothing to review and they are not the same run. One
    # searched and found nothing. One retrieved papers and stopped before it
    # screened them, which is what the live run of 17 September did: three
    # searches, 150 records, then a refused model request, and the page said
    # "No paper was found" over 150 records that had been found. One stopped
    # before it retrieved anything at all, which is what the second attempt
    # that morning did when the plan itself was refused. A reviewer reading
    # the first sentence over the second or third run is told the searches
    # came back empty when they did not, and would run the same question
    # again looking for papers that were already there.
    #
    # The count is the coverage report's, so the sentence and the report
    # cannot disagree about how many were retrieved. Screening a paper always
    # leaves a judgement, so papers retrieved with none screened means the run
    # stopped before screening, whatever else the report says.
    #
    # A run that retrieved nothing "failed before it retrieved anything" when
    # a search failed or the run stopped, whether or not it asked for a
    # person: the second session of 19 September chose those
    # words, and a run whose every search failed is what the launcher's failed
    # state shows. The words are the session's, as is "No paper found" for the
    # run that searched everything and found nothing.
    if report.papers_retrieved:
        return (f"{report.papers_retrieved} papers were retrieved and none was "
                "screened: the run stopped before screening. The reason is "
                "under 'Run details'. Nothing can be reviewed until the "
                "question is run again.")
    if report.searches_failed or report.stopped_because or report.needs_a_person:
        return ("The run failed before it retrieved anything. Details are "
                "under 'Run details'.")
    return "No paper found."


def headline(report, papers_found):
    """The words in the box at the top of the Assessment screen."""
    # A reviewer reads this before anything else, so it says what is wrong
    # rather than making them work it out from the report below.
    #
    # A short search and a fault in the run are different things and are said
    # differently. An earlier version told a run whose searches all finished
    # that it "did not search everything it planned", because it asked only
    # whether there was anything to warn about. A reviewer sent to look for a
    # missing search that does not exist is a reviewer who stops believing the
    # warning.
    #
    # When the run left nothing to review, the sentence saying why is part of
    # the headline rather than a plain line under its box. The second
    # session of 19 September read the coloured box and the line under it as
    # two statements, and one box carrying both was chosen (UX-06); the colour
    # is the run's, green for a search that completed and found nothing,
    # yellow for one that failed before it found anything. "Search completed"
    # and "Run details" are the words given in that session, in place of
    # "every planned search ran and finished" and "what this run did not
    # manage to search", which read as written for the developer.
    if not report.everything_was_searched:
        said = "This run did not search everything it planned."
        if papers_found:
            return (f"{said} What is missing is under 'Run details', and the "
                    "papers here are what it did find.")
    elif report.anything_went_wrong_quietly:
        said = ("Search completed, but something failed without stopping the "
                "run. What it was is under 'Run details'.")
    elif report.anything_went_wrong_quietly is None:
        said = ("Search completed. Whether anything failed quietly was not "
                "recorded, so this run cannot say it was clean.")
    else:
        said = "Search completed."
    if papers_found:
        return said
    return f"{said} {why_there_is_nothing_to_review(report)}"


def named_identifier(doi, pmid, paper_id):
    """The paper's identifier as its kind, its text and its address."""
    # Returns the kind ("DOI", "PubMed", "Semantic Scholar", or "" when the
    # record has none), the text to draw after it, linked where it can be,
    # and the address of the paper's page, "" when the project links to
    # none, for the button that opens it. In the session of 16 September,
    # asked to find a paper's identifier, the reviewer could not tell which
    # part of the line was one: a DOI after a journal name reads as more citation (UX-05).
    # The kind says where to look it up, which is what an identifier is for.
    # Since the second review of 17 September the DOI and the PubMed
    # identifier are links as well, to the DOI resolver and to the PubMed
    # page, so the reviewer can open the paper from the line that names it.
    # Markdown's own link, which streamlit.write draws. The identifier is
    # percent-encoded in the address and left as it is in the text, because a
    # DOI may hold parentheses and a bracket, which end a Markdown link early,
    # and the resolver reads the encoded form. The Semantic Scholar identifier
    # stays a word: the two forms were decided on.
    #
    # The kind comes back apart from the text because the two pages draw it
    # differently: the review page as a bold label on its own line, the
    # results page inside the citation line. Takes the three values rather
    # than a paper, because the results page draws it from a stored row,
    # which is not a PaperRecord.
    if doi:
        address = f"https://doi.org/{quote(doi)}"
        return "DOI", f"[{doi}]({address})", address
    if pmid:
        address = f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid)}/"
        return "PubMed", f"[{pmid}]({address})", address
    if paper_id:
        return "Semantic Scholar", paper_id, ""
    return "", "", ""


def _identifier(paper):
    """The one identifier a paper is keyed by in the session, or "" if none."""
    # The first that is not empty, in the order storage files them: a DOI names
    # one paper and nothing else, so it comes before the PubMed identifier, and
    # that before Semantic Scholar's. The citation line chooses in the same
    # order and adds the kind as a word; this stays bare because it is the key
    # the reviewer's answers are filed under, and a key with a word in it
    # would stop matching the answers already filed.
    if paper.doi:
        return paper.doi
    if paper.pmid:
        return paper.pmid
    if paper.paper_id:
        return paper.paper_id
    return ""


def what_the_agent_would_do(decision):
    """The agent's decision as a plain sentence."""
    # The decision word used to be dropped into "the agent would ... it",
    # which reads as English for include and exclude and not for the third
    # decision: the screen of 16 September showed "the agent would
    # uncertain it" on every paper the model was unsure about. Three plain
    # sentences chosen by if and else rather than a lookup table, so each
    # sentence is read where it is written. The decision is the agent's own
    # field and is not read off the rating: a paper rated 9 can still be one
    # the agent would exclude. Its own function since 19 September, because
    # the card says what the agent said before a send-back as well as now.
    if decision == "include":
        return "the agent would include it"
    if decision == "exclude":
        return "the agent would exclude it"
    return "the agent is not sure about it"


def _judgement_lines(judged):
    """One paper as the reviewer reads it, before they decide."""
    # The extract is shown with the summary, because the summary is built from
    # the paper's wording and the extract is what lets the reviewer check that
    # without opening the paper.
    #
    # Whether the extract was found is shown as well, and shown as three
    # answers. A paper whose extract was never looked for is not the same as one
    # whose extract is not in the paper, and the reviewer is the person who has
    # to tell them apart (SC-02).
    #
    # The publication status is shown because a retracted paper reads exactly
    # like a sound one, and a reviewer should weigh a retraction rather than
    # discover it afterwards. It is shown on every paper, "not
    # checked" when the record carries none, since the review of the
    # screen on 17 September: left out, a paper nobody checked read like a paper
    # with nothing to report.
    #
    # A block of labelled lines under the title, one fact each, since that
    # review: the citation was one line, and the author, the journal and
    # the identifier were put on lines of their own, the labels in
    # bold, the journal in italics. The order is the one given, and the reason is
    # always drawn: it is what the rating rests on.
    if judged.extract_found is None:
        found = "never looked for"
    elif judged.extract_found:
        found = "found in the paper"
    else:
        found = "not found in the paper"

    paper = judged.paper
    kind, text, _address = named_identifier(paper.doi, paper.pmid, paper.paper_id)

    lines = [paper.title]
    lines.append(f"**Author:** {first_author_of(paper.authors)} "
                 f"({paper.year or 'year not given'})")
    lines.append(f"**Journal:** *{paper.journal or 'not given'}*")
    if kind:
        lines.append(f"**{kind}:** {text}")
    else:
        lines.append("**Identifier:** none")
    lines.append(f"**Publication status:** "
                 f"{paper.publication_status or 'not checked'}")
    lines.append(f"**Relevance:** rated {judged.screening.relevance} of 10; "
                 f"{what_the_agent_would_do(judged.screening.decision)}")
    lines.append(f"**Reason:** {judged.screening.reason}")
    lines.append(f"**Summary:** {judged.screening.summary}")
    lines.append(f"**Extract from the paper ({found}):**")
    # The extract is the paper's own words and the one line the reviewer
    # reads to check the rest, so it is drawn as a quotation and every other
    # line stays plain (UX-05). On the screen of 16 September the
    # seven lines of a paper were one weight, and the extract did not stand
    # out from the agent's reason above it. A quotation is the ordinary mark
    # for words that are somebody else's, which is what the extract is; a
    # box or a colour would mark it as important instead, and the page keeps
    # one colour for one meaning. Markdown's own quotation, so no style rule
    # is added to the page for it. It comes last and after the summary,
    # which it is the sentence in support of.
    #
    # A blank line between paragraphs can end a Markdown quotation (R-29).
    # The record allows several paragraphs, so each line needs its quotation
    # mark, including the blank separator. Otherwise the second paragraph
    # looks like the agent's words rather than part of the supporting extract.
    # The source text itself stays unchanged.
    quoted = []
    for line in judged.screening.extract.split("\n"):
        quoted.append(f"> {line}")
    lines.append("\n".join(quoted))
    return lines


def which_paper(judged):
    """What a reviewer's answer is filed under."""
    # The paper's own identifier, or its title when it has none. Filing answers
    # by their position in the list was enough while one run was on the screen
    # and wrong as soon as a second was loaded: the answers stayed where they
    # were and the new run's first paper arrived already approved by somebody
    # who had never seen it.
    key = _identifier(judged.paper)
    if not key:
        key = judged.paper.title
    return key


def what_storing_now_would_do(state, screened):
    """One line beside the store button saying what storing now leaves out."""
    # The session of 16 September asked whether the papers could be stored
    # while some had no decision, and the page did not say. They can: storage keeps
    # the approved papers and leaves a waiting paper waiting, neither stored
    # nor counted as rejected. Said here so a reviewer who presses Store with
    # papers undecided does so knowing it, and one who wanted them all decided
    # first is told how many are left (UX-02).
    #
    # The wording is the one chosen in the review of the screen on 17 September:
    # it names the approved list the results page shows, which is what an
    # undecided paper is left out of, rather than the store. That wording
    # drops the clause saying an undecided paper is not counted as rejected;
    # the count line above, which keeps waiting and rejected apart, is where
    # that fact stays visible.
    answers = state.get(_DECISIONS, {})
    waiting = 0
    for judged in screened:
        if which_paper(judged) not in answers:
            waiting += 1
    if waiting == 0:
        return "Every paper has a decision."
    if waiting == 1:
        return ("1 paper currently has no decision. Proceeding now will exclude "
                "it from your approved list for consultation. Consider "
                "reviewing it first.")
    return (f"{waiting} papers currently have no decision. Proceeding now will "
            "exclude them from your approved list for consultation. Consider "
            "reviewing them first.")


def the_answer_in_its_colour(choice):
    """The reviewer's answer as a coloured word: green, amber or red."""
    # The second review of 17 September chose the current decision
    # on each paper as a coloured word, so the eye finds it without reading
    # (UX-05). The word carries the meaning and the colour goes with it, never
    # alone: the baseline says colour is never the only sign, and a reader
    # who cannot tell green from red still reads "approved".
    #
    # Streamlit's own coloured text, ":green[approved]", rather than a style
    # rule, so the page keeps its one rule for the top bar. The word is drawn
    # as text and not as a box, because a box on this page means something
    # happened to the run, not what a reviewer said.
    #
    # Amber for flagged, Streamlit's "yellow", since the review of the
    # screen on 17 September: amber was chosen, and the measurement backs it.
    # Read from the installed bundle's palettes, the yellow text is #926c05 on
    # the light page, 4.81 to 1, and #ffffc2 on the dark, 18.3 to 1; the
    # orange used before it is #e2660c on the light page, 3.42 to 1, under
    # the 4.5 to 1 the baseline asks of text (UX-01). Green and red clear it
    # on both pages, 4.91 and 5.28 on the light, 11.62 and 6.85 on the dark.
    if choice == "approved":
        return ":green[approved]"
    if choice == "flagged":
        return ":yellow[flagged]"
    return ":red[rejected]"


def how_many_decided(state, screened):
    """One line saying how many papers have an answer, which, and how many wait."""
    # In a set of fifty, on 16 September, the reviewer could not tell which
    # papers had been dealt with and which had not, and a reviewer who cannot
    # see that is one who either stores too early or reads the list twice
    # (UX-03). The count in the panel says where the work stands.
    #
    # Since the second review of 17 September the line says how the
    # decided split: approved, flagged and rejected. Decided against waiting
    # says how far the review has got; the split says what storing now would
    # keep and what sending back would send, which is what the two buttons
    # under the line do.
    #
    # Counted from the answers in the session and not from the papers' own
    # review field, because the field is written only when the papers are
    # stored or sent back; until then the session is the only record of what
    # the reviewer has said.
    answers = state.get(_DECISIONS, {})
    approved = 0
    flagged = 0
    rejected = 0
    for judged in screened:
        answer = answers.get(which_paper(judged))
        if answer is None:
            continue
        choice = answer[0]
        if choice == "approved":
            approved += 1
        elif choice == "flagged":
            flagged += 1
        else:
            rejected += 1
    decided = approved + flagged + rejected
    waiting = len(screened) - decided
    return (f"{decided} of {len(screened)} papers decided, {waiting} waiting: "
            f"{approved} approved, {flagged} flagged, {rejected} rejected.")


def shown_for_review(state, screened):
    """The papers the Assessment page draws: those the agent rated above 5
    of 10, and any paper the reviewer has already answered.
    """
    # The rule set on 18 September, after the live run put all fifty papers
    # in front of the reviewer and forty-one back again after the send-back: the
    # reviewer's hour goes to the papers the agent found likely. The number
    # is `SHOWN_FROM_RELEVANCE` in the coverage report's module, which counts
    # the papers kept back for the report, so the page and the report cannot
    # disagree. The run screens at most fifty papers, so the page
    # never shows more than fifty, and it shows exactly as many as are above
    # the line, however few. A paper the reviewer has answered, or sent back
    # and had judged again, stays shown whatever its rating: an answer never
    # disappears from under them, and a fresh judgement they asked for is
    # always put in front of them, even one that rated the paper lower.
    #
    # Nothing is dropped. A paper kept back stays in the run and in the
    # record, is counted on the page and in the coverage report, and is a
    # paper waiting for a decision as far as the store is concerned, which
    # means it is never stored.
    answers = state.get(_DECISIONS, {})
    came_back = state.get(_CAME_BACK_FROM, {})
    shown = []
    for judged in screened:
        key = which_paper(judged)
        likely = judged.screening.relevance >= audit_log.SHOWN_FROM_RELEVANCE
        if likely or key in answers or key in came_back:
            shown.append(judged)
    return shown


def decided_so_far(state, screened):
    """The run's papers with the reviewer's answers so far written onto them."""
    # Kept apart from the drawing because storage is given this, and what is
    # stored should be worked out by something a test can call rather than by
    # the order buttons happened to be pressed.
    answers = state.get(_DECISIONS, {})
    decided = []
    for judged in screened:
        key = which_paper(judged)
        if key in answers:
            choice, note = answers[key]
            decided.append(decide(judged, choice, note))
        else:
            decided.append(judged)
    return decided


# Streamlit's own link blues, read from the palettes in the installed 1.62
# bundle: blue80 on the light page, 5.49 to 1 against white, and blue60 on
# the dark page, 6.58 to 1 against its background. Streamlit itself draws a
# link in the appearance's blue text colour, blue90 and blue60; the light
# one here is a shade lighter, the conventional link blue that was named,
# and still clears the 4.5 to 1 asked of text (UX-01).
_LINK_BLUE_ON_THE_LIGHT_PAGE = "#0068c9"
_LINK_BLUE_ON_THE_DARK_PAGE = "#3d9df3"


def _link_blue(streamlit):
    """The link blue for the appearance in use, the light one when unknown."""
    # Streamlit tells the script the appearance it inferred from the page's
    # background (`st.context.theme.type`), and says itself that the answer
    # can lag on the first drawing of a session and just after the reader
    # changes the appearance in the settings menu; the next drawing corrects
    # it. A style rule cannot ask the appearance, which is why the script
    # does.
    if streamlit.context.theme.type == "dark":
        return _LINK_BLUE_ON_THE_DARK_PAGE
    return _LINK_BLUE_ON_THE_LIGHT_PAGE


def main():
    """Draws the screen."""
    # The finished run is read from the session rather than started here. A
    # screen that ran a search when it was drawn would spend model requests
    # every time somebody pressed a button, and the free tier allows twenty a
    # day (OB-07). Whatever loads a run puts it in the session under `run`, as
    # the state and its coverage report.
    import streamlit

    session = streamlit.session_state
    session.setdefault(_DECISIONS, {})
    finished = session.get(_RUN)

    # Answers belong to the run they were made about. Loading a second run
    # while the first run's answers were still in the session submitted them
    # as though they had been made about the new papers. The papers that came
    # back from a revision keep the run's identifier, so the answers on the
    # papers that did not go back survive that, as they are meant to.
    if finished is None:
        whose = None
    else:
        state = finished[0]
        papers = [which_paper(judged) for judged in state.screened]
        whose = (state.run_id, papers)
    if session.get(_WHOSE) != whose:
        session[_DECISIONS] = {}
        session.pop(_CAME_BACK_FROM, None)
        session[_WHOSE] = whose

    # The bar Streamlit draws across the top has no edge on a page of the
    # same colour, so content slid under something the reader could not
    # see: the session of 16 September found the title and a
    # focused box half hidden under it, in two browsers (UX-05). Streamlit
    # has no setting for the bar's edge, so this was the one style rule
    # written into the page: a lower border in a grey that shows on both
    # appearances. The edge was chosen over hiding the bar's controls,
    # which a setting could have done.
    #
    # The second rule is the results page's box: the mock-up of 17
    # September has it pale green, Streamlit's coloured boxes hold text
    # only, and a second rule was chosen over a plain border the same
    # evening. The green is Streamlit's own green70,
    # #21c354, at the same low opacity Streamlit gives its success box, so
    # it tints the light page and the dark page each in their own way and
    # the text keeps its contrast on both: 11.3 to 1 on the light page's
    # tint (#e4f8ea) and 15.3 to 1 on the dark page's (#10261e), against
    # 12.5 and 18.1 on the bare pages, measured with the palettes' own body
    # text colours (`box_contrast.py`, session scratchpad). Streamlit gives
    # a container with a key the class "st-key-" plus the key, which is what
    # the rule is aimed at.
    #
    # The third rule is the panel's two actions, "Display Approved" and the
    # send-back button: a white face, the panel's violet #440099 for the
    # lettering and a three-pixel border in the lighter violet #ad73e5, the
    # colours and widths given on the evening of 17 September. A button
    # filled in the accent had read as a warning while the accent was
    # red, and the two actions were wanted apart from the panel's
    # other buttons once the panel was violet; a first border in the
    # panel's own violet could not be seen against it, which is why the
    # border is the lighter one (picture 20). The figures given, checked:
    # the lettering 12.31 to 1 on the face, the border 3.73 against the
    # panel and 3.3 against the face, above the 3 to 1 asked of a control's
    # boundary. The lettering is the accent set in .streamlit/config.toml,
    # and a test keeps the two the same. Streamlit gives a button with a
    # key the same "st-key-" class as a container, which is how the two
    # are found.
    #
    # The fourth is the panel's help sentences. Streamlit draws a caption
    # at six tenths of the text's opacity, which on the grey panel was the
    # grey chosen; on the violet white was chosen, so in
    # the panel a caption is drawn at full opacity.
    #
    # The fifth hides the control that folds the panel away, decided
    # the same evening: the panel holds the count and the two
    # actions, and folded away it takes them with it. Only where the panel
    # sits beside the page, though. Read from the installed bundle, the
    # panel lies over the page instead on a window 768 pixels wide or
    # narrower, Streamlit's "md" breakpoint (not the 640 at which columns
    # stack), and there a reviewer on a phone has to be able to close it
    # to reach the papers; so the rule applies from 769 pixels up, and the
    # control that opens a folded panel is left alone everywhere.
    #
    # The sixth is the spacing round the panel's two lines. Streamlit gives
    # a rule 2em above and below, a block each (picture 21 of the 17th),
    # and the block holding it ends 1rem early to cancel a paragraph's
    # margin, which a rule has none of. A quarter rem above and one and a
    # quarter below, less that rem, leaves the line 12 pixels from the text
    # on each side, with the panel's half-rem gap.
    #
    # The seventh draws the record's line as a link, blue and underlined,
    # decided after the live run of 18 September: as a plain
    # button it did not read as a way to a page. It stays a button, since a
    # link would reload the page and lose the run in the session; the rule
    # gives it a link's look. The blue is Streamlit's own link blue for the
    # appearance in use, read from the palettes in the installed bundle:
    # no one blue clears 4.5 to 1 on both pages (blue70 is 3.9 on the light
    # and 4.84 on the dark), so the appearance is asked for, and the light
    # blue is used when it is not yet known.
    streamlit.markdown(
        '<style>header[data-testid="stHeader"] '
        '{ border-bottom: 1px solid rgba(128, 128, 128, 0.4); } '
        '.st-key-saved-papers { background-color: rgba(33, 195, 84, 0.12); '
        'border-color: rgba(33, 195, 84, 0.4); } '
        '.st-key-store button, .st-key-send-back button, '
        '.st-key-send-back-all button '
        '{ background-color: #ffffff; color: #440099; '
        'border: 3px solid #ad73e5; } '
        'section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] '
        '{ opacity: 1; } '
        '@media (min-width: 769px) { [data-testid="stSidebarCollapseButton"] '
        '{ display: none; } } '
        'section[data-testid="stSidebar"] hr { margin: 0.25rem 0 1.25rem; } '
        f'.st-key-run-record button {{ color: {_link_blue(streamlit)}; '
        'text-decoration: underline; }'
        '</style>',
        unsafe_allow_html=True)
    # Every page opens at its top, whichever page it follows: the frame is
    # drawn before any page, on the one drawing after a move.
    if session.pop(_TO_THE_TOP, False):
        _scroll_to_the_top(streamlit, session)

    # The results page is asked for by the store button and by the list of
    # earlier runs, and drawn instead of whatever else the session holds.
    # Asked before the form, because a reviewer opening yesterday's outcome
    # has no run loaded, and before the papers, because the one who has just
    # stored has.
    # The record of runs is a page of its own, reached from the panel of
    # every other page; the run it opens on is the one on the screen, or
    # the one whose outcome was being read.
    if session.get(_PAGE) == "record":
        if session.get(_RECORD_FROM) == "results":
            opened_on = session.get(_RESULTS_OF)
        elif finished is None:
            opened_on = None
        else:
            opened_on = finished[0].run_id
        _draw_the_record(streamlit, session, opened_on)
        _draw_the_notice(streamlit)
        return

    # Each of the three screens is titled for the step it is, under the
    # progress line that names all three: decided on 18 September,
    # in place of one title on every screen. The person stays in the
    # sentence under the title. The mock-up of 16 September read
    # "Nothing is stored automatically", which is true and leaves out who
    # does store: the gate is a reviewer's decision, and the line
    # under the title is where a reader learns that.
    if session.get(_PAGE) == "results":
        _draw_the_progress_line(streamlit, session, "Result", finished)
        streamlit.title("Result")
        streamlit.caption(A_REVIEWER_DECIDES)
        _draw_the_results(streamlit, session, finished)
        _draw_the_notice(streamlit)
        _offer_the_record(streamlit, session)
        return

    if finished is None:
        _draw_the_progress_line(streamlit, session, "Search", finished)
        streamlit.title("Search")
        streamlit.caption(A_REVIEWER_DECIDES)
        _draw_the_form(streamlit)
        _draw_the_stored_runs(streamlit, session)
        _draw_the_notice(streamlit)
        # Offered here as well, because the log outlives the session that
        # wrote it. Somebody opening the screen the morning after a run has
        # no run loaded and is exactly the person who wants to read what it
        # did.
        _offer_the_record(streamlit, session)
        return

    _draw_the_progress_line(streamlit, session, "Assessment", finished)
    streamlit.title("Assessment")
    streamlit.caption(A_REVIEWER_DECIDES)
    state, report = finished
    papers_found = bool(state.screened)
    if what_to_warn_about(report):
        streamlit.warning(headline(report, papers_found))
        with streamlit.expander("Run details"):
            _draw_the_coverage(streamlit, report, papers_found)
    else:
        streamlit.success(headline(report, papers_found))

    # A run whose searches all finished and found nothing used to show the
    # green headline and a store button with no paper between them, and the
    # review of the screen on 16 September could not tell that nothing had
    # been found or why. That is the case the design called dangerous: a
    # tidy page over a search that came back empty. So it is
    # said in words, in the headline's own box since the second
    # session of 19 September, and the store button is not offered, because
    # there is nothing it could store; the panel holds the way on instead.
    # The record's warning, the notice and the way to the record still
    # follow: a person asking why nothing came is the person who reads the
    # record.
    if not papers_found:
        _draw_the_panel_with_nothing_to_review(streamlit, session)
        _warn_if_the_record_is_missing(streamlit, state.run_id)
        _draw_the_notice(streamlit)
        _offer_the_record(streamlit, session)
        return

    # Taken off the session as it is drawn, so it is said once about the
    # revision it belongs to rather than on every rerun afterwards.
    came_back = session.pop(_CAME_BACK, None)
    if came_back:
        streamlit.success(came_back)
    for warning in session.pop(_LOG_TROUBLE, []):
        streamlit.warning(warning)
    store_failed = session.pop(_STORE_FAILED, None)
    if store_failed:
        streamlit.error(store_failed)

    # The papers rated 5 or below are kept back from the page and counted,
    # in the words given for it (shown_for_review says why). A run whose every
    # paper was rated 5 or below has nothing to put in front of the
    # reviewer, and says so the way a run that found nothing does, with the
    # same panel: there is nothing the review's buttons could act on, and
    # the way on is still owed.
    shown = shown_for_review(session, state.screened)
    kept_back = len(state.screened) - len(shown)
    if not shown:
        streamlit.write(f"The agent rated all {kept_back} papers 5 or below, so "
                        "none is shown for review. They stay in the run record.")
        _draw_the_panel_with_nothing_to_review(streamlit, session)
        _warn_if_the_record_is_missing(streamlit, state.run_id)
        _draw_the_notice(streamlit)
        _offer_the_record(streamlit, session)
        return
    if kept_back == 1:
        streamlit.write("1 paper rated 5 or below, not shown.")
    elif kept_back:
        streamlit.write(f"{kept_back} papers rated 5 or below, not shown.")

    # One bordered card per paper, one per row, full width. The
    # review of the screen on 17 September chose cards in a grid, two or three
    # per row, and the mock-up of the results page gave the card
    # its shape; two columns were built, and the review of that
    # evening (picture 12) chose one card per row on every
    # window: a review card carries the reason, the summary, the extract,
    # the note box and three buttons, and half a window folded its lines
    # and its badges. So the rows follow one another, and a phone and the
    # widest window draw the same page. The number is the paper's place in
    # the run, kept as the key of its widgets even when earlier papers are
    # kept back, so a decision stays with its paper across redrawings.
    # The number a person reads on the card counts the cards shown, from 1,
    # so that two people can name a paper to each other (decided
    # after the live run of 18 September); it is not the paper's place in
    # the run, which the widget keys keep and which skips the papers kept
    # back.
    shown_as = 0
    for number, judged in enumerate(state.screened):
        if judged not in shown:
            continue
        shown_as += 1
        with streamlit.container(border=True):
            _draw_a_review_card(streamlit, session, state, number, judged, shown_as)

    _draw_the_panel(streamlit, session, state, shown, came_back)
    _warn_if_the_record_is_missing(streamlit, state.run_id)
    _draw_the_notice(streamlit)
    # At the foot of this page as well, since the second session of
    # 19 September: the line was on the form, on the results page and on
    # the Assessment page with nothing to review, and not on the one with
    # the papers, where the session looked for it (B11). The record's Back
    # already returns here, since the form and this page share the same
    # "no page" key.
    _offer_the_record(streamlit, session)


def the_answer_as_a_badge(choice):
    """The reviewer's answer as a small badge beside the title."""
    # The same three colours as the coloured word, green, amber and red, and
    # the word inside the badge, so colour is never the only sign (UX-01).
    # Streamlit's own badge, ":green-badge[approved]", so no style rule.
    if choice == "approved":
        return ":green-badge[approved]"
    if choice == "flagged":
        return ":yellow-badge[flagged]"
    return ":red-badge[rejected]"


def _draw_a_review_card(streamlit, session, state, number, judged, shown_as):
    """One paper as a bordered card: what it is, what the agent made of it,
    and the reviewer's three answers at the foot.
    """
    # The card's number, then the title in bold with the relevance as a
    # small badge and, once the
    # reviewer has answered, their answer as a badge beside it; a grey line
    # with the first author, the year, the journal in italics and the
    # identifier as a link, the same line as the results page's card; then
    # the publication status, the relevance sentence, the reason, the
    # summary and the extract as `_judgement_lines` gives them; the three
    # answers side by side; the note box under them; and the answer written
    # out with its note. The card is the mock-up of 17 September,
    # with the decision buttons where the results page has its "Open
    # paper page". The note box moved under the answers on 18 September, so
    # that the hover text on an answer, which opens upwards, never covers it.
    paper = judged.paper
    kind, text, _address = named_identifier(paper.doi, paper.pmid, paper.paper_id)
    now = judged.screening
    # What the agent had said before the reviewer sent the paper back,
    # against what it says now. On the live run of 18 September the agent
    # changed its mind on four papers of forty-one and the page said
    # nothing of it, so the reviewer could not tell whether the second
    # judgement had been worth asking for. A line under the citation said
    # "from ... to ..." in red from that evening; the second
    # session, on the 19th, found the change still did not catch the eye,
    # because the card's own badge and relevance line showed the new
    # values in the ordinary style. So the changed values themselves are
    # marked: the badge and the relevance line in red with the old value
    # in brackets beside the new, and one word, "unchanged", where nothing
    # changed. The new reason is the reason line below, since the fresh
    # judgement replaced the old one on the card.
    came_back_from = session.get(_CAME_BACK_FROM, {}).get(which_paper(judged))
    changed = (came_back_from is not None
               and came_back_from != (now.decision, now.relevance))
    badge = f":gray-badge[rated {now.relevance} of 10]"
    if changed:
        before_decision, before_relevance = came_back_from
        if before_relevance == now.relevance:
            badge = f":red-badge[rated {now.relevance} of 10]"
        else:
            badge = (f":red-badge[rated {now.relevance} of 10 "
                     f"(was {before_relevance})]")
    badges = badge
    decided = session[_DECISIONS].get(which_paper(judged))
    if decided is not None:
        badges = f"{badges} {the_answer_as_a_badge(decided[0])}"
    streamlit.markdown(f"**{shown_as}. {paper.title}** {badges}")
    if kind:
        identifier = f"{kind}: {text}"
    else:
        identifier = "no identifier"
    streamlit.caption(f"{first_author_of(paper.authors)} "
                      f"({paper.year or 'year not given'}) · "
                      f"*{paper.journal or 'journal not given'}* · {identifier}")
    if came_back_from is not None and not changed:
        streamlit.write("**Judged again:** unchanged")
    elif changed:
        streamlit.write(f":red[**Judged again:** from: {before_decision} "
                        f"{before_relevance} to: {now.decision} {now.relevance}]")
    # From the publication status on, the card draws the judgement's own
    # lines; the title and the citation above are drawn the card's way, so
    # the first four are taken and not written. Each line is named as it is
    # taken, because the relevance line is the one a second judgement can
    # change, and a reader should not have to count `_judgement_lines` to
    # know which line is being marked (found in review on 21 September).
    (_title, _author, _journal, _identifier, status_line, relevance_line,
     reason_line, summary_line, extract_heading, extract) = _judgement_lines(judged)
    if changed:
        relevance_line = (f":red[{relevance_line} (was rated {before_relevance} "
                          f"of 10; {what_the_agent_would_do(before_decision)})]")
    streamlit.write(status_line)
    streamlit.write(relevance_line)
    streamlit.write(reason_line)
    streamlit.write(summary_line)
    streamlit.write(extract_heading)
    streamlit.write(extract)

    # The note is read from the session before its box is drawn, because
    # the three answers now sit above the box: the hover text on an answer
    # opens upwards, where Streamlit puts a button's tooltip (the
    # placement is the tooltip's own and the script cannot set it), and
    # on the live run of 18 September it covered the note box that sat
    # above the buttons. Above the answers there is only the paper's own
    # text, which a tooltip may cover for a moment. Streamlit keeps a
    # widget's value under its key from one drawing to the next, so the
    # note the reviewer typed is here whether its box has been drawn yet
    # or not.
    note_key = f"note {number}"
    note = session.get(note_key, "")
    # The three answers side by side rather than one above the other. They
    # are one choice, so they are drawn as one thing (UX-05). Whether that
    # makes a reviewer faster or less likely to press the wrong one is not
    # claimed here: this changes the arrangement. The time a paper takes is
    # what the task-time measurement records, and how easy the screen is to
    # use is what the ease rating records.
    #
    # A form was the other way to group them, and is not used yet. With
    # `enter_to_submit` left at its default, a reviewer typing the reason
    # for a flag would decide the paper by pressing Enter, which is the
    # mis-click the decision rule exists to prevent (UX-02). That can be
    # turned off, so it is not a reason to refuse a form. What holds it
    # back is that a form
    # changes when the note reaches the run: without one it arrives when
    # the reviewer presses Enter or leaves the field, and with one it
    # arrives only when they press an answer. Whether that helps somebody
    # working down a list is what has not been measured.
    answers = streamlit.columns(3)
    decided_just_now = False
    for column, choice in zip(answers, ("approved", "flagged", "rejected"),
                              strict=True):
        with column:
            # The help is the panel's sentence for the answer, shown when
            # the pointer rests on the button, decided on 18
            # September: a reviewer at paper forty has the panel in view
            # but their eyes on the card. On a phone there is no pointer to
            # rest, and the panel's text stays the reference.
            pressed = streamlit.button(choice, key=f"{choice} {number}",
                                       help=WHAT_EACH_ANSWER_DOES[choice])
        if pressed:
            wrote_before = len(run_log.writing_failures)
            # Asked before the answer is written down, not when the papers
            # are stored. A flag recorded without its note failed later,
            # at the moment the reviewer pressed store, by which time they
            # had moved on and the paper it belonged to was off the
            # screen. What they had said before stays as it was.
            try:
                decide(judged, choice, note)
            except ReviewRefused as refusal:
                streamlit.error(str(refusal))
            else:
                session[_DECISIONS][which_paper(judged)] = (choice, note)
                # Written when the reviewer decides rather than when the
                # papers are stored, because a decision they later changed
                # is part of what happened and storage keeps only the last
                # one.
                run_log.note_a_decision(judged, choice, note, state.run_id)
                # What the log could not write travels with the redrawing
                # below, or the warning would go with the page it was
                # written on.
                session[_LOG_TROUBLE] = _what_the_log_could_not_write(
                    "That decision was recorded here", wrote_before)
                decided_just_now = True

    # The box keeps what was typed across the reruns a click causes and
    # across a send-back, because the note travels with the paper and is
    # what the Curation Agent was given. In the session of 16
    # September the reason was still in the box after a send-back, and
    # whether it was meant to be could not be told. It is, and the label
    # now says so once the box holds something; clearing the box was the
    # other way, and would have thrown away the reason a reviewer wants in
    # front of them when they decide the paper again.
    if note:
        note_label = "Your note, kept until you decide again"
    else:
        note_label = "Why, if you flag or reject it"
    streamlit.text_input(note_label, key=note_key)

    # Drawn again from the top after a decision, because the badge beside
    # the title was drawn before the button was read: Streamlit builds the
    # page as it goes, and without this the answer would appear beside the
    # title one click late. Asked for here, after the note box, and not
    # inside the button's branch above it: Streamlit forgets the value of
    # a box that a drawing did not reach, and a redrawing asked for before
    # the box is drawn would throw the reviewer's note away.
    if decided_just_now:
        streamlit.rerun()

    # "You said" in bold, so the eye finds the current answer among seven
    # lines of the same weight. Changing a decision on 16 September, the
    # reviewer could not see afterwards that it had changed: the line
    # is the current answer and nothing else on the page says which it
    # is (UX-05). Bold and the answer's colour are the whole of it;
    # recording which earlier answer it replaced would need the session
    # to keep a history, and a reviewer wants the answer that stands,
    # not the ones that do not. The badge beside the title says the same
    # since the cards; this line is where the note is read.
    if decided is not None:
        choice, note = decided
        answer = the_answer_in_its_colour(choice)
        if note:
            streamlit.write(f"**You said:** {answer}, because {note}")
        else:
            streamlit.write(f"**You said:** {answer}")


def _draw_the_panel(streamlit, session, state, shown, came_back):
    """The side panel: the count, the help, the two buttons that act on the
    whole run, and the way to the record.
    """
    # The count and the storing sentence are about the papers shown, since
    # 18 September: a paper kept back as rated 5 or below is neither decided
    # nor waiting on the page, and a count that included it would send the
    # reviewer looking for a paper that is not there. Storing and sending
    # back act on the whole run, where a kept-back paper is one waiting for a
    # decision, which the store leaves out and the send-back does not send.
    #
    # The panel stays in view while the papers scroll. The session of 16
    # September found the store and send-back buttons at the bottom of
    # fifty papers and the confirmation at the top. A fixed panel for the
    # counts and buttons was point 6 in the ordered changes (UX-05). The
    # count was above the papers until then, where it was right
    # and then scrolled away; one copy in the panel is in view at every
    # paper, so the copy at the top is gone rather than kept as a second
    # place for one fact.
    #
    # Drawn after the cards, so the count and the sentence beside the store
    # button include the answer the reviewer has just given: Streamlit builds
    # the page as it goes, and anything written before the cards shows the
    # answer before last. On a narrow screen Streamlit folds the panel away
    # behind its own control at the top left, which is the trade for a
    # panel that stays put on a wide one.
    # Everything in the panel sits in one container with half of Streamlit's
    # usual gap between elements. With the usual gap, the two lines and the
    # help, the panel outgrew the session's 731-pixel window and "Display
    # Approved" sat below its fold (picture 21 of 17 September). The
    # container is Streamlit's own way of setting the gap; the lines' own
    # spacing is set by a style rule beside the page's others, since the
    # divider has no setting for it.
    with streamlit.sidebar, streamlit.container(gap="xsmall"):
        streamlit.subheader("Your decisions")
        streamlit.write(how_many_decided(session, shown))
        # Once the first paper is decided, the panel says where the review
        # ends: it was decided on 17 September, so a reviewer whose
        # first press was on a card is told what to press last. A sentence
        # rather than scrolling the panel, which Streamlit cannot do from
        # here, and which the tighter panel makes unnecessary: the button is
        # in view without it.
        if session.get(_DECISIONS):
            # In bold, as decided on the morning of the 18th (picture
            # 23): it is the one instruction in the panel. In a box
            # as well, since the second session of 19 September, where
            # the bold line did not stand out on the violet: Streamlit's
            # info box, which the panel's theme draws as the confirmation
            # boxes are drawn, white on the lighter violet (6.73 to 1, set
            # in .streamlit/config.toml). The pale box was the first of the
            # two treatments offered, over capitals, and a box
            # already measured is one fewer colour on the page.
            streamlit.info("**When you are done, press Display Approved below.**")
        # Two thin lines, drawn where picture 20 of 17 September
        # draws them: one under the count, one under the help, so
        # the count, the help and the storing sentence read as three parts.
        # Streamlit's divider, in the panel's own border colour (white, set
        # in .streamlit/config.toml), rather than a style rule.
        streamlit.divider()
        # Above the panel's two buttons, where it was placed on 17
        # September, and nowhere on the page: the page copy above the first
        # paper scrolled away with it. One caption per entry, so each answer
        # starts its own line under its bold head. In italics, as decided
        # on the morning of the 18th, to set the help apart from the
        # panel's other text; the underscores are markdown's italics, put
        # round the sentence here so the sentence itself stays the words
        # given exactly.
        for entry in WHAT_THE_ANSWERS_DO:
            streamlit.caption(f"_{entry}_")
        streamlit.divider()

        # Said a second time here, beside the send-back button. The
        # confirmation used to appear at the top only, out of sight of the
        # reviewer who had just pressed the button, who read nothing and
        # pressed again. The top copy stays for the reviewer who starts from
        # the headline.
        if came_back:
            streamlit.success(came_back)

        _draw_the_hand_back(streamlit, session, state)

        # Beside the button rather than at the top, because it is about what
        # this button will do.
        streamlit.write(what_storing_now_would_do(session, shown))
        # The label is the one chosen in the review of the screen on 17 September,
        # and names what the reviewer sees next: the approved papers, on the
        # results page. The press still stores them first, and the page it
        # opens says so in its first line. The key stays "store", because
        # the key is what the tests and the session know the button by.
        #
        # A plain button, drawn white with violet lettering by the page's
        # style rule on its key, like the send-back button above it. It was
        # Streamlit's primary type for an hour on the evening of 17
        # September, which filled it in the accent; the accent's red read
        # as "forbidden" in that evening's reading, and a filled button of
        # any colour was not what the mock-up drew, so the type went back and the rule does
        # the work.
        if streamlit.button("Display Approved", key="store"):
            try:
                # Called through the module rather than by a name imported
                # here, because Streamlit runs this file as its own script: a
                # name bound at import time would belong to that copy, and a
                # test replacing storage would be replacing something the
                # screen never calls.
                kept = storage.save_approved(decided_so_far(session, state.screened))
            except storage.RefusedByStorage as refusal:
                _say_the_store_failed(streamlit, session, state, str(refusal))
            except Exception as problem:  # noqa: BLE001
                # Every fault, not only the store's own refusal. Anything
                # drawn under this button lands at the foot of the panel,
                # below the fold of a 731-pixel window, and on 17 September
                # a press that seemed to do nothing was reported with no
                # red box in sight: whatever it was, nobody could see it.
                # The fault is written at the top of the page instead, and
                # to the log, with its own words.
                _say_the_store_failed(streamlit, session, state,
                                      f"{type(problem).__name__}: {problem}")
            else:
                # Written to the log so the results page can say when the
                # review finished, and the record when the papers were kept.
                run_log.note_a_store(kept, state.run_id)
                # The page moves on by itself to the outcome. A green line
                # counting the stored papers used to be the end of the
                # review, in the panel beside the button, and the review of the
                # screen on 17 September could not tell where the papers were.
                # The results page, drawn on the next run of the script, is
                # the answer, its heading carrying the count.
                session[_PAGE] = "results"
                session[_RESULTS_OF] = state.run_id
                session[_TO_THE_TOP] = True
                streamlit.rerun()


def _say_the_store_failed(streamlit, session, state, why):
    """Puts a failed store at the top of the page and in the log."""
    # A refusal at the end of a review is a fault in the run, and the reviewer
    # is the only person who will ever see it on the screen. It used to be
    # drawn under the button, which is the foot of the panel, below the fold
    # of the session's window: on 17 September a press that seemed to do nothing
    # was reported with nothing red in sight. So the message is carried to the
    # next drawing and written at the top of the page, where the reviewer
    # looks, and the page is drawn again to put it there.
    run_log.note_a_failure("storing the approved papers", why, state.run_id)
    session[_STORE_FAILED] = f"The approved papers were not stored: {why}"
    streamlit.rerun()


def _draw_the_coverage(streamlit, report, papers_found):
    """Draws the coverage report as two headed groups, side by side."""
    # The mock-up of 16 September put the counts and the shortfalls in
    # two columns under two headings, and the one list the screen drew, even
    # with the same two headings inside it, still read as a list on
    # 17 September (UX-06). So the two groups the record keeps apart are drawn
    # apart: the counts on the left, what fell short on the right, and the
    # sentence saying what to do next under both, last, so a reader who opens
    # the section reads what happened before being told what to do.
    #
    # Streamlit's own columns rather than the mock-up's table. A table does
    # not reflow: at 320 pixels it scrolls sideways. Columns do, without a
    # style rule: below Streamlit's own breakpoint of 640 pixels each column
    # is given the whole width, so the two groups stack one under the other
    # (checked in the installed 1.62 bundle, where the column's width rule
    # carries that media query). The headings are drawn in bold rather than as
    # subheaders, because a subheader inside a closed section reads as a
    # section of its own and the two groups are one report.
    #
    # The words are the record's, from the two functions the record's own
    # one-column form is built from, so the screen and the run's record cannot
    # say different things.
    did, needs = streamlit.columns(2)
    with did:
        streamlit.markdown("**What the run did**")
        for line in audit_log.what_the_run_did(report):
            streamlit.write(line)
    with needs:
        streamlit.markdown("**What needs attention**")
        for line in audit_log.what_needs_attention(report):
            streamlit.write(line)
    streamlit.write(what_to_do_next(report, papers_found))


def _draw_the_hand_back(streamlit, session, state):
    """Offers to send the flagged and rejected papers back to be judged again."""
    # This is the path Figure 2 draws from the reviewer to the Curation Agent.
    # Until now a flag was a note nobody acted on: the paper stayed on
    # the screen with the judgement the reviewer disagreed with.
    #
    # Nothing is offered when there is nothing to send. The button would do
    # nothing, and a control that does nothing teaches a reviewer to distrust
    # the others.
    #
    # The decisions on the papers that went back are cleared, because the design
    # asks for a fresh decision before a revised record is stored. Waiting and
    # rejected are treated alike by storage, which withdraws both, so this is not
    # about what the store does: it is about what the record says. An answer left
    # in place would stand as the reviewer's decision on a judgement they have
    # not read.
    #
    # The run on the screen is replaced by the one that comes back. The coverage
    # report beside it is not rebuilt: it describes the run as it finished, and
    # the failure lists it was built from are the run's own, so a report made
    # here would carry whatever had happened since.
    # The reviewer's answers are in the session rather than on the run, so the
    # run is written up with them first. Asking the run itself which papers
    # were flagged would find none, because nothing has been decided on it.
    decided = state.model_copy(update={
        "screened": decided_so_far(session, state.screened)})
    flagged = graph.sent_back(decided, ("flagged",))
    rejected = graph.sent_back(decided, ("rejected",))
    if not flagged and not rejected:
        return

    # Two buttons since 18 September, the flagged alone first: on the live
    # run one flagged and forty rejected papers went back in one
    # press, because that was the one press there was, and the forty were read
    # again. The flagged are the likely case, a paper the reviewer wants
    # looked at again; the rejected go with them only when asked. Each
    # button names its count, and a button that would send nothing is not
    # drawn.
    answers_that_go = None
    if flagged and streamlit.button(
            f"Send the {len(flagged)} flagged papers back to be judged again",
            key="send back"):
        answers_that_go = ("flagged",)
    if rejected:
        if flagged:
            label = (f"Send the {len(flagged)} flagged and {len(rejected)} rejected "
                     "papers back to be judged again")
        else:
            label = f"Send the {len(rejected)} rejected papers back to be judged again"
        if streamlit.button(label, key="send back all"):
            answers_that_go = ("flagged", "rejected")
    if answers_that_go is None:
        return
    going_back = graph.sent_back(decided, answers_that_go)

    wrote_before = len(run_log.writing_failures)

    def the_second_judgement():
        # Called through the module for the reason storage is: Streamlit runs
        # this file as its own script, so a name bound at import time belongs
        # to that copy.
        return graph.judge_again(decided, curator, answers_that_go)

    judged_again = _run_saying_what_it_waits_for(
        streamlit, "Judging them again. This makes a model request.",
        the_second_judgement)

    if judged_again.needs_human_attention or judged_again.stopped_because:
        why = judged_again.attention_reason
        if not why:
            why = judged_again.stopped_because
        streamlit.error(why)
        run_log.note_a_failure("judging the papers again", why, state.run_id)
        # A revision that could not be made leaves the reviewer where they
        # were, with the papers and the answers they had. What it does not
        # leave is the allowance: a request that failed was still a request,
        # and a run that kept the old count would offer the same button again
        # and spend from an allowance it had already used.
        session[_RUN] = (state.model_copy(update={"budget": judged_again.budget}),
                         session[_RUN][1])
        # This path does not redraw, so the warning is written here.
        for warning in _what_the_log_could_not_write("That failure was recorded here",
                                                     wrote_before):
            streamlit.warning(warning)
        return

    _note_what_came_back(going_back, judged_again, state.run_id)
    came_back_from = session.get(_CAME_BACK_FROM, {})
    for was in going_back:
        session[_DECISIONS].pop(which_paper(was), None)
        came_back_from[which_paper(was)] = (was.screening.decision,
                                            was.screening.relevance)
    session[_CAME_BACK_FROM] = came_back_from
    # Said on the next drawing rather than this one. Drawing the page again is
    # what puts the fresh judgements in front of the reviewer, and anything
    # written here goes with the page it was written on. The lines the log
    # could not write travel with it, or a reviewer would be told the revision
    # was recorded when it was not.
    session[_CAME_BACK] = (f"{len(going_back)} papers were judged again. Each "
                           "is waiting for your decision.")
    session[_LOG_TROUBLE] = _what_the_log_could_not_write("That revision was made",
                                                          wrote_before)
    session[_RUN] = (judged_again, session[_RUN][1])
    # To the top like every other move, since the finding of 18
    # September: the send-back button is at the foot of the panel, and the
    # page came back wherever it had been scrolled to, with the line saying
    # the papers were judged again out of sight above.
    session[_TO_THE_TOP] = True
    streamlit.rerun()


def _note_what_came_back(went_back, judged_again, run):
    """Writes down that the judgements changed."""
    # One line for each paper, naming it by its identifier and saying what the
    # agent said before and what it says now. The decisions either side of it
    # are already in the log, so without this a reader sees a paper rejected and
    # then approved with nothing in between saying that the judgement itself had
    # changed.
    fresh = {which_paper(judged): judged for judged in judged_again.screened}
    for was in went_back:
        now = fresh.get(which_paper(was))
        if now is not None:
            run_log.note_a_revision(now, was.screening.decision, run)


def _what_the_log_could_not_write(what_was_done, before):
    """The warnings to show when a decision or a revision was not logged."""
    # The decision itself is kept: losing what somebody decided because the
    # record of it could not be written would be the wrong way round. But a
    # reviewer who is not told carries on believing the review is being written
    # down, and the coverage report they were shown was built before they
    # started, so it cannot say this. The sentences are made here and
    # shown on the next drawing, because both paths redraw the page after the
    # act, and a warning written before the redrawing goes with the page it
    # was written on.
    warnings = []
    for failure in run_log.writing_failures[before:]:
        warnings.append(f"{what_was_done} but not written to the run log: "
                        f"{failure}")
    return warnings


def _run_saying_what_it_waits_for(streamlit, label, work):
    """Runs the work under Streamlit's status box, whose label the model
    interface changes while it waits for an overloaded service, and hands
    back what the work returns.
    """
    # Until 18 September the work ran under a spinner with fixed words, and
    # watching the live run's second attempt, nobody could tell a
    # run waiting a minute for the service from one that would never come
    # back. Streamlit's status box is its spinner with a label that can be
    # changed from the script while it runs, so the model interface is given
    # a way to write into it: "The model is busy: asking again in 2 minutes
    # 30 seconds". The way in is taken away again whatever happens, because
    # it is a name in the model interface and would otherwise outlive the
    # box it writes to.
    with streamlit.status(label) as box:
        def say(line):
            box.update(label=line)

        llm.say_while_waiting = say
        try:
            return work()
        finally:
            llm.say_while_waiting = None


# The free tier's daily allowance, by model, as read on the account's own
# pages: the design's model from the rate-limit page on 28 August 2026, the
# Lite model from the usage page on 18 September. Google does not
# publish the figures in its documentation, so a model not read here gets
# no figure rather than a guess. Until 22 September the caption said "twenty
# a day" whatever the model, which was the 3.6 Flash figure shown over a run
# on the Lite model (asked for on 20 September).
_DAILY_ALLOWANCE_READ = {
    "gemini-3.6-flash": "twenty a day for this model (read on 28 August 2026)",
    "gemini-3.5-flash-lite": "500 a day for this model (read on 18 September 2026)",
}


def what_a_run_costs(model):
    """The caption under the form: what a run spends, on the model named."""
    # The model is the one the settings name, which is the one the model
    # interface asks, so the caption and the run cannot name different
    # models. The run's own cap is true whatever the model; the tier's figure
    # is given only where it was read, and its absence is said (OB-07).
    cap = RunBudget.model_fields["model_request_limit"].default
    first = ("Starting a run makes model requests as it plans and screens, "
             f"on {model}; a run makes at most {cap}.")
    allowance = _DAILY_ALLOWANCE_READ.get(model)
    if allowance is None:
        return (f"{first} The free tier's daily allowance for this model has "
                "not been read.")
    return f"{first} The free tier allows {allowance}."


def _draw_the_form(streamlit):
    """Takes the question and the criteria, and starts a run from them."""
    # Drawn only while no run is loaded. Starting a second run would replace the
    # papers the reviewer is working through, and their answers would be cleared
    # as belonging to a different run, so the button that would do it is not
    # put in front of them while they are deciding.
    #
    # What a run costs is written next to the button, by `what_a_run_costs`. A
    # run makes model requests as it plans, refines and screens, and the free
    # tier's daily allowance is small, so a reviewer who starts one to see what
    # happens has spent part of what a rehearsal needs (OB-07).
    #
    # How many searches and refinements a run may make is offered here, which is
    # what makes a shorter run possible without editing the code. The run's
    # two internal allowances are not offered, and they are not offered for two
    # different reasons. Raising the eighteen model requests spends the two the
    # day's allowance keeps back for a second attempt, and above twenty a run
    # would ask for more than the free tier gives. Raising the fifty papers
    # cannot make a run cost more than eighteen either, because `llm.py` refuses
    # the request that would be the nineteenth and hands back `OutOfBudget`
    # instead of sending it. What more papers change is how much of what the
    # searching left over is spent on screening, since curation runs after the
    # searching has finished: a longer screening reaches the limit and stops
    # with papers unjudged. Neither is a number to move while a run is being
    # asked for.
    streamlit.info("No run has been loaded yet.")
    streamlit.subheader("Start a run")
    question = streamlit.text_area("The research question", key="question")
    inclusion = streamlit.text_area("What a paper must be to be kept",
                                    key="inclusion")
    exclusion = streamlit.text_area("What rules a paper out", key="exclusion")
    # Taken from the record rather than written here again, so that the number
    # a run actually uses and the number offered cannot drift apart. The two
    # names come from the second session of 19 September, in
    # place of "Searches and refinements this run may make" and "Results
    # asked of each search", which read as written for the developer. The
    # first is as given. The second was given as "Maximum papers
    # reviewed", which is not what the number is: it is how many results
    # each search asks its service for, and at most fifty papers are
    # screened whatever it says, none when a run retrieves nothing; so the
    # nearest true plain name, offered as the alternative, is used instead.
    steps = streamlit.number_input(
        "Maximum search attempts", min_value=1,
        value=ResearchState.model_fields["step_limit"].default, key="steps")
    # Offered since 18 September, as decided, for a quick run; the
    # default is the record's fifty, this project's own default and not
    # the setting the recall was measured at, which was 1,000, so
    # the form left alone does not change what an ordinary run is. The
    # least is one screening batch, and the most is the number the
    # recall and standard-dataset measurements ask of a search; a run that asks a
    # thousand of many searches meets the run's own ceiling on NCBI
    # attempts, which the coverage report says rather than the run failing
    # (OB-04).
    results_per_search = streamlit.number_input(
        "Maximum results per search", min_value=10, max_value=1000,
        value=ResearchRequest.model_fields["results_per_search"].default,
        key="results per search")
    streamlit.caption(what_a_run_costs(llm.settings.gemini_model))

    if not streamlit.button("Start the run", key="start"):
        return
    try:
        request = what_was_asked_for(question, inclusion, exclusion,
                                     int(results_per_search))
    except ReviewRefused as refusal:
        streamlit.error(str(refusal))
        return

    # Read before the run starts and again when it ends, because the three
    # failure lists outlive a run: a report built from everything on them
    # would hand this run whatever an earlier one had left there.
    before = audit_log.failures_so_far()

    def the_run():
        # Called through the module for the reason storage is: Streamlit
        # runs this file as its own script, so a name bound at import time
        # belongs to that copy and a test replacing the run would replace
        # something the screen never calls.
        return graph.run(request, planner, curator, step_limit=int(steps))

    try:
        state = _run_saying_what_it_waits_for(
            streamlit, "Searching and screening.", the_run)
    except ValidationError as refusal:
        # The run reads its whole record again at the hand-over and at the end,
        # and lets the refusal raise so that an invalid shared record is never
        # passed on as evidence (DC-05). That decision is the run's and is left
        # alone. What is decided here is what the reviewer sees and what the
        # log keeps: left uncaught, Streamlit showed its own stack trace, and
        # nothing wrote the end of the run to the log, so its request and
        # searches were there with no line saying why they stopped.
        # Handled the way a refused store is, an error on the page and a line
        # in the log. The run's identifier is made inside the run and does not
        # survive the raise, so the line is filed as no run's. The log groups
        # its lines by identifier, because two runs' lines can interleave, so
        # this line is shown under "not recorded" and cannot be tied to its
        # run by where it sits: the reason is kept, the link to the execution
        # is not, and the log's record says so. The record's own reasons are
        # shown and logged rather than the whole error, which quotes the
        # record and so may quote a paper, and the log holds none of a
        # paper's words.
        why = ("the run was stopped because its record failed its own checks: "
               + _why_the_request_was_refused(refusal))
        streamlit.error(why)
        run_log.note_a_failure("running the request", why)
        return
    streamlit.session_state[_RUN] = (
        state, audit_log.report_on(state, audit_log.failures_during(before)))
    # Drawn again from the top, so the reviewer gets the papers rather than
    # the form they have just used with the papers underneath it; and moved
    # to the top, because the form was scrolled to its button when it was
    # pressed and the papers came into view at the fourth card (picture
    # 21 of 17 September).
    streamlit.session_state[_TO_THE_TOP] = True
    streamlit.rerun()


def _draw_the_stored_runs(streamlit, session):
    """Offers the outcome of an earlier run, read from the records."""
    # Drawn on the screen a person meets with no run loaded, which is the
    # screen the morning after a review. The stored set outlives the session
    # that made it, and until 17 September the only way to it was a SQLite
    # reader. Only runs with an approval are offered: a run with none has no
    # list to show.
    #
    # The selector is the log's, so a run is named by the same heading here
    # and in the record of runs below.
    offered = {one.run: one for one in results.runs_with_approvals()}
    if not offered:
        return
    streamlit.subheader("Approved papers of an earlier run")
    chosen = streamlit.selectbox(
        "Which run", list(offered),
        format_func=lambda name: run_log.heading(offered[name]),
        key="which stored run")
    if streamlit.button("Show its approved papers", key="show stored run"):
        session[_PAGE] = "results"
        session[_RESULTS_OF] = chosen
        session[_TO_THE_TOP] = True
        streamlit.rerun()


def _draw_the_results(streamlit, session, finished):
    """The outcome of a review: what was asked, what is kept, and the way out."""
    # This was decided on 17 September. The list of
    # approved papers is what a person uses after a review, and the screen
    # ended at a line counting them; the line is now the top of this page.
    #
    # Read from the run log and the store, not from the session, so the page
    # is the same for the run just stored and for one reviewed yesterday and
    # chosen from the list of earlier runs. Nothing here asks a model or a
    # service: the words are the store's, the counts are the log's, and the
    # identifier is a link the browser follows (OB-07).
    #
    # Two ways out, both buttons, because a reviewer told to reload the page
    # is one who did not know it was safe to. Back returns to the review page
    # of the run in the session with the decisions as they were, and pressing
    # Store there again updates the run's set, because storage keys rows by
    # the paper's identifier rather than adding rows. That button is offered
    # only when the run is in the session: an earlier run reopened from the
    # records has no review page to go back to, and a control that does
    # nothing teaches a reviewer to distrust the others. New search clears the
    # run and the decisions and draws the empty form; the run just stored
    # stays in the records and can be reopened from the form's screen.
    #
    # The two buttons are in the side panel, since the review of the
    # screen on 17 September, for the reason the review page's buttons are: the
    # panel stays in view while the list scrolls, and a way out at the foot
    # of forty papers is one the reader has to go looking for.
    #
    # The coverage counts are shown from the report when the run is on the
    # screen. The log does not hold them for an earlier run, and the page
    # says so rather than showing the searches as though they were the whole
    # account.
    run = session.get(_RESULTS_OF)
    on_screen = finished is not None and finished[0].run_id == run

    outcome = results.outcome_of(run)
    if outcome is None:
        streamlit.warning(
            f"Nothing in the run log belongs to run {run or 'with no identifier'}, "
            "so its outcome cannot be shown. Either the run was loaded from "
            "somewhere else, or its lines could not be written.")
    elif on_screen:
        _draw_the_outcome(streamlit, outcome, finished[0], finished[1])
    else:
        _draw_the_outcome(streamlit, outcome, None, None)

    with streamlit.sidebar:
        streamlit.subheader("What next")
        if on_screen and streamlit.button("Back to the paper selection",
                                          key="back to the papers"):
            session.pop(_PAGE, None)
            session[_TO_THE_TOP] = True
            streamlit.rerun()
        _offer_a_new_search(streamlit, session)


def _offer_a_new_search(streamlit, session):
    """The button that clears the run and draws the empty form."""
    # One function because two panels draw it: the results page's, and since
    # 19 September the panel of a review page with no paper on it. The key
    # stays "new search", the name the tests and the session know it by.
    if streamlit.button("Start a new search", key="new search"):
        _forget_the_run(session)
        session[_TO_THE_TOP] = True
        streamlit.rerun()


def _draw_the_panel_with_nothing_to_review(streamlit, session):
    """The side panel when the run left no paper on the page: the way on."""
    # The review page's panel is the count, the help and the two buttons that
    # act on the papers, and a run with no paper on the page drew none of it,
    # since there was nothing for its buttons to act on. The second
    # session of 19 September found the failed state with no panel at all and
    # no way on but the browser, and decided that a reviewer is never left
    # without one: any state with no papers shows the panel with at least
    # "Start a new search" (UX-05). So it is this and nothing else: the
    # heading the results page's panel has for the same button, and the
    # button. No "Display Approved", because the 16th's finding was a store
    # button with nothing to store.
    with streamlit.sidebar:
        streamlit.subheader("What next")
        _offer_a_new_search(streamlit, session)


def _forget_the_run(session):
    """Drops the run on the screen and everything that hangs off it, so the
    next drawing is the Search page with an empty form.
    """
    # Held in one place because two controls do it, "Start a new search" and,
    # since 18 September, the Search step of the progress line, and the two
    # must clear the same keys.
    for key in (_RUN, _DECISIONS, _WHOSE, _PAGE, _RESULTS_OF, _CAME_BACK_FROM):
        session.pop(key, None)


def when_the_review_finished(stamp):
    """"17 Sep 2026, 16:40" in this computer's time, from the log's stamp."""
    # The log writes UTC so that two computers' logs can be read side by
    # side; a person reading the results page wants the clock on the wall.
    # A stamp the log does not have is said to be not recorded rather than
    # shown as a blank. One that cannot be read is shown as it was written:
    # it is still the record's value, and a reader can see what was there,
    # where "not recorded" would say the log held nothing (the comment said
    # that until 22 September; the code never did).
    if not stamp:
        return "time not recorded"
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    local = moment.astimezone()
    return f"{local.day} {local:%b %Y, %H:%M}"


def first_author_of(names):
    """The first author, and "et al." when there are others."""
    # As the project's own references cite, rather than every name: a trial
    # can carry forty authors, and the line is there to tell the papers
    # apart.
    if not names:
        return "author not given"
    if len(names) == 1:
        return names[0]
    return f"{names[0]} et al."


def the_list_as_a_file(papers):
    """The approved papers as one CSV text, for the download button."""
    # Title, authors, year, journal, the publication status, the link, the
    # summary and the reviewer's note: what the results page shows, in a
    # file the reviewer keeps. The extract and the agent's reason stay in
    # the store; the list is what a person takes away, and the file is
    # written on this computer by the browser, with no service asked
    # (OB-07). CSV rather than Markdown because a spreadsheet opens it and a
    # reference manager imports it. The status column is written as the
    # store holds it, empty where no check was made, so a file can be
    # filtered on it; the card's "not checked" is a reading for the eye.
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["title", "authors", "year", "journal", "publication status",
                     "link", "summary", "your note"])
    for paper in papers:
        kind, _text, address = named_identifier(paper.doi, paper.pmid, paper.paper_id)
        if not address and kind:
            address = f"{kind} {paper.paper_id}"
        writer.writerow([paper.title, paper.authors, paper.year, paper.journal,
                         paper.publication_status, address, paper.summary,
                         paper.review_note])
    return out.getvalue()


def _draw_the_outcome(streamlit, outcome, state, report):
    """Draws one run's outcome as the mock-up of 17 September has it."""
    # The heading carries the count; one grey line names the question and
    # when the review finished; a box says the approved papers are saved and
    # offers the list as a file; one line gives the three counts and a
    # switch that unfolds the flagged and rejected papers under the approved
    # ones; then one card per approved paper, full width, with the title,
    # a green badge, the citation in grey with the identifier as a link, the
    # summary, the reviewer's note, and a button to the paper's page.
    #
    # The mock-up's box is pale green with the button inside it; Streamlit's
    # coloured boxes hold text only, so the box is a bordered container the
    # page's second style rule tints green, chosen on the evening
    # of 17 September over a plain border. "Copy the links" is not offered:
    # Streamlit has no way to write to the clipboard without a script, and
    # the page carries one script already, for scrolling.
    #
    # The approved papers the store does not hold are named by identifier, so
    # the count of approvals and the cards under it cannot quietly disagree,
    # and the box says whether all, some or none of them are in the store. It
    # said "all" whenever one was, and called each absent one withdrawn, which
    # a failed store or a session closed after approving also produces
    # (found in review on 21 September). What was asked and searched stays on
    # the page, folded, for the reader who wants the recap decided on before
    # the mock-up.
    #
    # "Review complete" only when the run's latest store stands: a run is
    # offered on the strength of an approval, and an approval is written
    # before the store, so a run whose store was refused or never pressed has
    # approvals and nothing kept, and calling that complete would send a
    # reader looking for a list that does not exist; a decision or a
    # send-back after the store leaves a kept set older than the counts, and
    # is not complete either.
    if outcome.stored:
        headline = "Review complete"
    else:
        headline = "Review not stored"
    if outcome.approved == 1:
        streamlit.header(f"{headline}: 1 paper approved")
    else:
        streamlit.header(f"{headline}: {outcome.approved} papers approved")
    streamlit.caption(f'Screening of "{outcome.question or "question not recorded"}", '
                      f"finished {when_the_review_finished(outcome.finished)}")

    # The key names the box for the page's second style rule, which tints
    # it green; the button sits centred beside the two lines, as the
    # mock-up has it.
    with streamlit.container(border=True, key="saved-papers"):
        words, action = streamlit.columns([3, 1], vertical_alignment="center")
        with words:
            if outcome.papers and not outcome.not_in_the_store:
                streamlit.markdown("**All approved papers are saved**")
            elif outcome.papers:
                streamlit.markdown(f"**{len(outcome.papers)} of {outcome.approved} "
                                   "approved papers are in the store**")
            else:
                streamlit.markdown("**No approved paper is in the store**")
            # In a person's terms, as decided: where the list is
            # and how to get back to it. The file's path is on the record
            # of runs, for whoever checks the files.
            streamlit.write("They are on this computer and are shown again "
                            "from the start screen under \"Approved papers of "
                            "an earlier run\". Nothing is sent to anyone.")
        with action:
            if outcome.papers:
                streamlit.download_button(
                    "Download the list", data=the_list_as_a_file(outcome.papers),
                    file_name=f"approved-papers-{outcome.run}.csv",
                    mime="text/csv", key="download the list")

    counts, unfold = streamlit.columns([3, 1])
    with counts:
        # A paper sent back and not decided again is neither approved nor
        # left out, and a recap that left it out of every count would not add
        # up to the papers reviewed; the word is the review page's own.
        line = (f"**{outcome.approved}** approved · "
                f"**{outcome.flagged}** flagged · "
                f"**{outcome.rejected}** rejected")
        if outcome.waiting:
            line += f" · **{outcome.waiting}** waiting"
        streamlit.markdown(line)
    show_the_rest = False
    with unfold:
        if outcome.left_out:
            show_the_rest = streamlit.toggle("View flagged and rejected",
                                             key="view flagged and rejected")

    for paper in outcome.papers:
        _draw_an_approved_card(streamlit, paper)
    for identifier in outcome.not_in_the_store:
        # No cause is given, because this run's record cannot tell a later
        # decision in another run from a store that failed or was never made.
        streamlit.write(f"- {identifier}: approved in this run and not in "
                        "the current store.")

    if show_the_rest:
        # The titles come from the session when the run is still in it.
        # The log names a paper by its identifier and holds none of its
        # words, and the store holds only approved papers, so a run
        # reopened tomorrow shows the identifier alone.
        titles = {}
        if state is not None:
            for judged in state.screened:
                titles[_identifier(judged.paper)] = judged.paper.title
        for left in outcome.left_out:
            _draw_a_left_out_card(streamlit, left, titles.get(left.identifier, ""))

    with streamlit.expander("What was asked and searched"):
        streamlit.write(f"**A paper must be:** {outcome.include or 'not recorded'}")
        streamlit.write(f"**A paper is ruled out if:** "
                        f"{outcome.exclude or 'not recorded'}")
        streamlit.write(f"**Run:** {outcome.run}, started "
                        f"{outcome.started or 'time not recorded'}")
        if outcome.searches:
            for line in outcome.searches:
                streamlit.write(f"- {line}")
        else:
            streamlit.write("No search is recorded for this run.")
        if report is not None:
            for line in audit_log.what_the_run_did(report):
                streamlit.write(line)
        else:
            streamlit.write("The counts of papers retrieved, combined and "
                            "screened are kept with the run while it is on "
                            "the screen, and not in the record of earlier runs.")


def _draw_an_approved_card(streamlit, paper):
    """One approved paper as a bordered card, full width (UX-05)."""
    # Title in bold with the green badge beside it; a grey line with the
    # first author, the year, the journal in italics and the identifier as a
    # link; the publication status; the summary; the reviewer's note; and on
    # the right a button to the paper's page at the resolver or PubMed. A
    # card rather than a run of labelled lines, from the mock-up: the list
    # is what a person reads and uses, and a card is read as one thing. A
    # Semantic Scholar identifier has no page the project links to, so its
    # card has no button and its grey line says so.
    #
    # The status is on every card, "not checked" where the record carries
    # none, for the reason the review page shows it on every paper: a
    # retracted paper reads exactly like a sound one, and the list is the
    # page a person uses after the review. Until 22 September the
    # card and the file left it out, so a retraction stored with a paper
    # was shown on the review page and nowhere after it (found in review).
    kind, text, address = named_identifier(paper.doi, paper.pmid, paper.paper_id)
    authors = [name.strip() for name in paper.authors.split(";") if name.strip()]
    with streamlit.container(border=True):
        words, action = streamlit.columns([5, 1])
        with words:
            streamlit.markdown(f"**{paper.title}** :green-badge[approved]")
            if kind:
                identifier = f"{kind}: {text}"
            else:
                identifier = "no identifier"
            streamlit.caption(f"{first_author_of(authors)} "
                              f"({paper.year or 'year not given'}) · "
                              f"*{paper.journal or 'journal not given'}* · "
                              f"{identifier}")
            streamlit.write(f"**Publication status:** "
                            f"{paper.publication_status or 'not checked'}")
            streamlit.write(paper.summary)
            if paper.review_note:
                streamlit.write(f"**Your note:** {paper.review_note}")
        with action:
            # "Open paper page", not "Open full text": the address is the
            # DOI resolver's or the PubMed record's page, which may hold an
            # abstract and a link on, and nothing here fetched or checked the
            # full text (the label promised it until 22 September).
            if address:
                streamlit.link_button("Open paper page", address)


def _draw_a_left_out_card(streamlit, left, title):
    """One flagged or rejected paper, with its badge and the reviewer's note."""
    # Shown only when the reviewer unfolds them: they are not the outcome,
    # and a person asking what was left out and why wants the decision and
    # the note beside each. Nothing here touches the store.
    if left.decision == "flagged":
        badge = ":yellow-badge[flagged]"
    else:
        badge = ":red-badge[rejected]"
    with streamlit.container(border=True):
        if title:
            streamlit.markdown(f"**{title}** {badge}")
            streamlit.caption(left.identifier)
        else:
            streamlit.markdown(f"**{left.identifier}** {badge}")
        if left.note:
            streamlit.write(f"**Your note:** {left.note}")


def _warn_if_the_record_is_missing(streamlit, run):
    """Warns when the run on screen has no lines in the log."""
    # A run whose lines never reached the log is a hole in the record, and the
    # person at the screen is the only one who will notice it. The coverage
    # report says a line could not be written; it cannot say that none was.
    # Said on the page the reviewer is at, not on the record's own page: a
    # warning on a page somebody has to go to is a warning nobody sees, and
    # this one is the worst hole in the record. An empty log with a run on the
    # screen is that same hole and the worst case of it.
    if run and run not in {one.run for one in run_log.runs()}:
        streamlit.warning(
            "Nothing in the run log belongs to the run on screen. Either it "
            "was loaded from somewhere else, or its lines could not be "
            "written.")


def _draw_the_progress_line(streamlit, session, current, finished):
    """"Search -> Assessment -> Result" across the top of the page, the step
    the person is on in bold and in the violet, the other two pressable.
    """
    # It was decided on 18 September so that a person sees the three
    # activities in order and where they are among them (UX-05). Drawn at the
    # top of the page body rather than in Streamlit's own bar across the top:
    # the script has no way to write into that bar, and moving page content
    # into it by a style rule is the kind of trick this project avoids. The
    # current step is text, since pressing where one already is does nothing;
    # the others are Streamlit's plain text-like buttons. The violet is the
    # theme's violet text colour, set in .streamlit/config.toml for each
    # appearance, so the step is #440099 on the light page and the lighter
    # #ad73e5 on the dark, for the same reason as the accent.
    #
    # Three rules on the presses, as given on 18 September, each a warning that blocks
    # nothing else. Assessment or Result with no search done: a search is
    # needed first. Result while the run's papers have not been stored:
    # press Display Approved first; stored means Display Approved has been
    # pressed for the run on the screen, which is what the session keeps in
    # the results key. Search from Assessment or Result: it resets the run,
    # as "Start a new search" does, so a first press only says so and a
    # second press within the next drawing confirms it; the approved papers
    # already stored stay in the store, and the warning says that too. An
    # earlier run's papers reopened from the form are not a run and go back
    # at the first press. Every move opens its page at the top like the
    # others.
    asked_to_leave = session.pop(_ASKED_TO_LEAVE_FOR_SEARCH, False)
    pressed = None
    with streamlit.container(horizontal=True, vertical_alignment="center"):
        for step in THE_STEPS:
            if step != THE_STEPS[0]:
                # Streamlit's own arrow icon, drawn by markdown, rather than
                # the two characters the rule was written with, which a
                # browser sets as a hyphen and a bracket.
                streamlit.markdown(":material/arrow_forward:")
            if step == current:
                streamlit.markdown(f":violet[**{step}**]")
            elif streamlit.button(step, key=f"step {step}", type="tertiary"):
                pressed = step
    if pressed is None:
        return

    if pressed in ("Assessment", "Result") and finished is None:
        streamlit.warning("A search needs to be done first.")
        return
    if pressed == "Result" and session.get(_RESULTS_OF) != finished[0].run_id:
        streamlit.warning("Press Display Approved first.")
        return
    # Asked only while there is a run to lose: an earlier run's approved
    # papers, reopened from the form, are read from disk and lose nothing.
    if pressed == "Search" and finished is not None and not asked_to_leave:
        session[_ASKED_TO_LEAVE_FOR_SEARCH] = True
        streamlit.warning(
            "Going back to Search resets the current search; the approved "
            "papers already saved stay in the store. Press Search again to "
            "confirm.")
        return

    if pressed == "Search":
        _forget_the_run(session)
    elif pressed == "Assessment":
        session.pop(_PAGE, None)
    else:
        session[_PAGE] = "results"
    session[_TO_THE_TOP] = True
    streamlit.rerun()


def _offer_the_record(streamlit, session):
    """The one discreet way to the record of runs: a line at the very foot."""
    # The record is for anyone checking what was done, which a reviewer
    # deciding on papers rarely is. It sat on the review page as a closed
    # section from 16 September, and the review of the screen on 17
    # September found even that technical and in the way, so it became a
    # page of its own reached from a "Run record" button in the panel. That
    # button confused the same evening's reading, beside the launcher's
    # "Start the run": two buttons whose names both say "run". So the button
    # is out of every panel, and the record is reached from one place only,
    # a link-like line at the very bottom of the form, of the results page
    # and, since the second session of 19 September, of the review
    # page, worded for what it is. The page it was reached from is remembered
    # so that Back returns there.
    if streamlit.button("Record of this run, for checking what was done",
                        key="run record", type="tertiary"):
        session[_RECORD_FROM] = session.get(_PAGE)
        session[_PAGE] = "record"
        session[_TO_THE_TOP] = True
        streamlit.rerun()


def _draw_the_record(streamlit, session, run):
    """The record of every run on this computer, from the lines each wrote."""
    # Shown by the screen rather than by a separate program. The person
    # checking what was searched is the person who has the screen open, and a
    # second program would be one they had to be told about and one more
    # thing to keep working.
    #
    # Any run in the log can be chosen, not only the one on the screen. The
    # log outlives the session that wrote it, so somebody coming back to a run
    # made yesterday would otherwise have nothing to read. The run on the
    # screen, or the one whose outcome was being read, is offered first.
    #
    # The two files are named here and nowhere a reviewer reads: the
    # review of 17 September found a path meaningless on the results page, and the
    # person checking the files is the person who opens this page.
    streamlit.subheader("Run record")
    streamlit.caption(
        "The record of every run on this computer, for checking what was "
        "done. The approved papers are kept in data/approved_records.sqlite3 "
        "and this record in evidence/run_log.jsonl, both inside the project "
        "folder.")

    with streamlit.sidebar:
        streamlit.subheader("What next")
        if streamlit.button("Back", key="back from the record"):
            came_from = session.pop(_RECORD_FROM, None)
            if came_from is None:
                session.pop(_PAGE, None)
            else:
                session[_PAGE] = came_from
            session[_TO_THE_TOP] = True
            streamlit.rerun()

    named = {one.run: one for one in run_log.runs()}
    if not named:
        streamlit.write("The run log is empty.")
        return

    names = list(named)
    if run in named:
        shown_first = names.index(run)
    else:
        shown_first = 0
    chosen = streamlit.selectbox(
        "Which run to show", names, index=shown_first,
        format_func=lambda name: run_log.heading(named[name]),
        key="which run to show")
    for line in run_log.as_lines(named[chosen].entries):
        streamlit.write(line)


# The one script the page carries. Streamlit offers no way to move the
# page from Python: it keeps the scroll position across a rerun, and the
# only scrolling its own code does is to a heading named in the address.
# So the page is moved by a line of script in an embedded frame, which
# Streamlit gives the rights to reach the page around it (its own words
# for `st.iframe`: an HTML string is embedded in a frame that allows
# script and same-origin access to the app; and read from the installed
# bundle, the part of the page that scrolls is the section marked stMain).
# The frame is drawn once, on the rerun after a move between pages, and
# not again, so the script runs once. One pixel high, because Streamlit
# refuses zero.
SCROLL_TO_THE_TOP = (
    '<script>window.parent.document.querySelector(\'[data-testid="stMain"]\')'
    '.scrollTo(0, 0);</script>')


def _scroll_to_the_top(streamlit, session):
    """Moves the page to its top, on the rerun after a move between pages."""
    # The empty form is shorter than the results page it replaces, and with
    # the scroll position kept the reader landed at its foot, below the
    # field they were meant to type in; the papers came into view at the
    # fourth card after the form's button, which sits at the form's foot.
    # So every move draws this frame first, and each frame carries its own
    # number: Streamlit redraws an element only where it differs from the one
    # drawn at the same place before, and two moves in a row would otherwise
    # leave the first frame standing with its script already spent. Whether
    # the script does what it says in a browser is not something AppTest can
    # show; the tests read that the frame is drawn once per move and carries
    # the script, and the next look at the screen in a browser is what
    # shows it working.
    moves = session.get(_MOVES, 0) + 1
    session[_MOVES] = moves
    streamlit.iframe(f"{SCROLL_TO_THE_TOP}<!-- move {moves} -->", height=1)


def _draw_the_notice(streamlit):
    """Puts NCBI's notice on every screen the application draws (OB-03)."""
    # Drawn from one place rather than at the end of each branch, because a
    # branch added later would otherwise be a screen with no notice on it, and
    # the obligation is about what a user sees rather than about what the
    # ordinary path draws.
    streamlit.subheader("NCBI disclaimer and copyright notice")
    streamlit.write(NCBI_DISCLAIMER)
    streamlit.write(NCBI_COPYRIGHT)
    streamlit.write(NCBI_NOTICE_URL)


# Streamlit runs this file as a script, so this is where the screen starts.
if __name__ == "__main__":
    main()
