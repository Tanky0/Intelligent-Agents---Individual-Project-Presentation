"""The reviewer's decisions, and the screen that puts them in front of them."""
# Two kinds of test. The decisions are called directly: the three answers, what
# a flag or a rejection has to carry, what comes back for revision, and the
# wording NCBI requires.
#
# The screen is run with Streamlit's own AppTest, which renders the script and
# presses its buttons in this process. It needs no browser, so there is no
# reason to leave the drawing unchecked, and what it catches is the kind of
# thing reading the code does not: a control that is drawn but wired to nothing,
# or an answer that disappears on the next rerun (FR-13, FR-16, OB-03).

import csv
import datetime
import io
import re
from pathlib import Path

import pytest

from research_agent.config import REPO_ROOT
from research_agent.processing import run_log
from research_agent.processing.audit_log import report_on
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
from research_agent.ui.app import (
    NCBI_COPYRIGHT,
    NCBI_DISCLAIMER,
    NCBI_NOTICE_URL,
    SCROLL_TO_THE_TOP,
    WHAT_THE_ANSWERS_DO,
    ReviewRefused,
    _judgement_lines,
    decide,
    decided_so_far,
    first_author_of,
    headline,
    how_many_decided,
    named_identifier,
    still_waiting,
    the_answer_in_its_colour,
    the_list_as_a_file,
    what_to_do_next,
    what_to_warn_about,
    what_was_asked_for,
    when_the_review_finished,
    which_paper,
    why_there_is_nothing_to_review,
)


def a_paper(title="A trial of drug X"):
    return PaperRecord(
        title=title,
        abstract="Forty patients were given drug X.",
        text_source="abstract",
        origins=[SearchOrigin(source="pubmed", query="drug X")])


def judged(title="A trial of drug X", review="waiting", extract_found=True,
           relevance=8):
    return ScreenedPaper(
        paper=a_paper(title),
        screening=Screening(paper_number=1, decision="include", relevance=relevance,
                            reason="it reports an outcome for drug X",
                            summary="a trial of drug X in condition Y",
                            extract="Forty patients were given drug X"),
        review=review,
        extract_found=extract_found)


def a_state(**changes):
    values = {
        "request": ResearchRequest(question="Does drug X treat condition Y?",
                                   inclusion_criteria="trials",
                                   exclusion_criteria="reviews"),
        "results": ResultsBuffer(
            outcomes=[Outcome(subgoal=Subgoal(purpose="find trials",
                                              source="pubmed",
                                              query="drug X trials"),
                              result="finished", papers_found=1)],
            papers=[a_paper()], ready=True),
    }
    values.update(changes)
    return ResearchState(**values)


def test_a_reviewer_has_three_answers_and_approving_needs_no_note():
    """A flag says the paper needs another look, not that it is wrong."""
    # Without it a reviewer chooses between keeping a doubtful paper and
    # throwing it out.
    assert decide(judged(), "approved").review == "approved"
    assert decide(judged(), "flagged", "not sure it is a trial").review == "flagged"
    assert decide(judged(), "rejected", "it is a review").review == "rejected"


@pytest.mark.parametrize("choice", ["flagged", "rejected"])
@pytest.mark.parametrize("note", ["", "   ", "\n"])
def test_a_flag_or_a_rejection_has_to_say_why(choice, note):
    """The note is what the Curation Agent is given when the paper comes back."""
    # Empty, the decision is one nobody can act on or disagree with, and a note
    # of nothing but spaces is empty.
    with pytest.raises(ReviewRefused, match="has to say why"):
        decide(judged(), choice, note)


@pytest.mark.parametrize("found, words", [
    (None, "was never looked for in the paper"),
    (False, "was not found in the paper"),
])
def test_approving_a_paper_whose_extract_is_not_traceable_is_refused_at_the_paper(
        found, words):
    """The store refuses such a paper, and the whole batch with it (SC-02)."""
    # Until 17 September the reviewer learnt that at the store button, with the
    # paper named by its identifier among fifty: the review of the screen on 17 September
    # met it on the invented papers, one of which is never checked on purpose.
    # The refusal belongs where the reviewer is (UX-02), with what to do
    # instead.
    with pytest.raises(ReviewRefused, match=words) as refusal:
        decide(judged(extract_found=found), "approved")
    assert "Flag it for a second judgement, or reject it" in str(refusal.value)
    # The other two answers stay open: they are how the paper is dealt with.
    assert decide(judged(extract_found=found), "flagged", "no extract").review == "flagged"
    assert decide(judged(extract_found=found), "rejected", "no extract").review == "rejected"


def test_the_refused_approval_is_said_at_the_paper_and_nothing_is_recorded():
    """On the screen, the press is refused where the paper is, the reason
    under its buttons, and no decision is filed for it."""
    screen = a_screen(a_state(screened=[judged(extract_found=None)]))

    screen.button(key="approved 0").click().run()

    assert "was never looked for in the paper" in screen.error[0].value
    assert screen.session_state["decisions"] == {}


def test_an_answer_that_is_not_one_of_the_three_is_refused():
    """Waiting is a state a paper is in, not a decision somebody made."""
    with pytest.raises(ReviewRefused, match="not one of the three answers"):
        decide(judged(), "waiting")


def test_the_judgement_is_not_edited_when_a_decision_is_recorded():
    """The judgement is what the model said and the decision is what a person
    made of it, and editing the first in place would lose the difference."""
    original = judged()

    decide(original, "rejected", "it is a review")

    assert original.review == "waiting"
    assert original.review_note == ""


def test_the_papers_that_go_back_are_the_flagged_and_the_rejected():
    """A flag says the paper needs another look, and a rejection the reviewer
    wrote a reason for is a disagreement the agent can answer (FR-19)."""
    from research_agent import graph

    state = a_state(screened=[
        decide(judged("kept"), "approved"),
        decide(judged("doubtful"), "flagged", "is it a trial?"),
        decide(judged("thrown out"), "rejected", "it is a review")])

    going_back = graph.sent_back(state)

    assert [item.paper.title for item in going_back] == ["doubtful", "thrown out"]
    assert going_back[0].review_note == "is it a trial?", "the note travels with it"


def test_the_papers_nobody_has_decided_about_are_the_ones_left():
    """A reviewer who stops half way comes back to what they had not reached."""
    screened = [decide(judged("decided"), "approved"), judged("not yet")]

    assert [item.paper.title for item in still_waiting(screened)] == ["not yet"]


def test_a_complete_run_shows_no_coverage_warning():
    """A warning shown every time is a warning nobody reads."""
    report = report_on(a_state(), ((), (), ()))

    assert what_to_warn_about(report) == []


def test_a_short_run_warns_in_the_words_of_the_run_s_own_record():
    """Two renderings would drift, and the one nobody looked at would be the
    one that was wrong."""
    short = a_state(pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])

    warning = what_to_warn_about(report_on(short, ((), (), ())))

    assert "Searches never reached:" in "\n".join(warning)
    assert "  - pubmed: drug X harms" in warning


def test_a_run_whose_failures_are_unknown_still_warns():
    """Not knowing is not the same as nothing being wrong, and the reviewer
    is the person who should be told which they have."""
    warning = what_to_warn_about(report_on(a_state()))

    assert "not known" in "\n".join(warning)


def test_a_short_run_tells_the_reader_what_to_do_next():
    """The record says what was missed; read cold on 16 September, it did not say
    what to do about it (UX-06).
    """
    # A run with papers says they can be reviewed now; a run with none says
    # nothing here needs a decision. Neither promises that a rerun will reach
    # what was missed.
    short = a_state(pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])
    report = report_on(short, ((), (), ()))

    with_papers = what_to_do_next(report, papers_found=True)
    without = what_to_do_next(report, papers_found=False)

    assert with_papers.startswith("What to do next:")
    assert "can be reviewed now" in with_papers
    assert "nothing here needs a decision" in without
    assert "later" in with_papers and "later" in without


def test_an_empty_completed_run_is_not_told_its_papers_can_be_reviewed():
    """Every search finished, nothing was screened, and a buffer write
    failed: the advice said the papers could be reviewed over an empty page
    (found in review on 21 September). Whether there are papers is decided
    on its own, and a report with no failure snapshot says "not known"
    rather than "listed above"."""
    finished = a_state()
    a_write_failed = ((), ("the buffer could not be written",), ())

    empty_with_a_failure = what_to_do_next(report_on(finished, a_write_failed),
                                           papers_found=False)
    assert "nothing here needs a decision" in empty_with_a_failure
    assert "listed above" in empty_with_a_failure
    assert "can be reviewed" not in empty_with_a_failure

    empty_unknown = what_to_do_next(report_on(finished), papers_found=False)
    assert "nothing here needs a decision" in empty_unknown
    assert "not known" in empty_unknown
    assert "listed above" not in empty_unknown

    papers_unknown = what_to_do_next(report_on(finished), papers_found=True)
    assert "can be reviewed now" in papers_unknown
    assert "not known" in papers_unknown

    papers_with_a_failure = what_to_do_next(report_on(finished, a_write_failed),
                                            papers_found=True)
    assert "can be reviewed now" in papers_with_a_failure
    assert "listed above" in papers_with_a_failure


def test_the_next_step_is_drawn_under_the_report_on_the_screen():
    """Last inside the coverage section, so a reader who opens it reads what
    happened before being told what to do."""
    short = a_state(screened=[judged()],
                    pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])
    screen = a_screen(short, a_report(short))

    lines_in_the_section = [item.value for item in screen.expander[0].markdown]
    assert lines_in_the_section[-1].startswith("What to do next:")
    assert "Searches never reached:" in lines_in_the_section


def test_the_coverage_is_two_headed_groups_side_by_side():
    """The mock-up of 17 September put the counts and the shortfalls in two
    columns, and the one list with two headings still read as a list at the
    screen that day (UX-06).
    """
    # The counts are in the left column under their heading, what fell short in
    # the right under its own, in the record's words, and the next step is
    # under both. Columns rather than the mock-up's table because a table does
    # not reflow at 320; Streamlit stacks columns below 640 pixels, which a
    # person checks at the screen, not this test.
    short = a_state(screened=[judged()],
                    pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])
    screen = a_screen(short, a_report(short))

    section = screen.expander[0]
    did, needs = [column for column in section.columns]
    assert [item.value for item in did.markdown][:3] == [
        "**What the run did**", "Planned searches: 2", "Finished searches: 1"]
    # The invented paper has no DOI, so the report has that to say as well;
    # Streamlit strips the indent from the list line as it draws it.
    assert [item.value for item in needs.markdown] == [
        "**What needs attention**",
        "Publication status not asked for, no DOI: 1",
        "Searches never reached:", "- pubmed: drug X harms"]
    assert did.weight == needs.weight
    outside_the_columns = [item.value for item in section.markdown
                           if item.value.startswith("What to do next:")]
    assert len(outside_the_columns) == 1
    assert not any(item.value.startswith("What to do next:")
                   for column in (did, needs) for item in column.markdown)


def test_the_reviewer_is_shown_whether_the_extract_was_found():
    """Never looked for and not found are different, and the reviewer is the
    person who has to tell them apart (SC-02)."""
    assert "**Extract from the paper (found in the paper):**" in \
        _judgement_lines(judged())
    assert "**Extract from the paper (not found in the paper):**" in \
        _judgement_lines(judged(extract_found=False))
    assert "**Extract from the paper (never looked for):**" in \
        _judgement_lines(judged(extract_found=None))


def test_the_reviewer_is_shown_the_summary_and_the_extract_together():
    """The extract is what lets a reviewer check the summary without opening
    the paper (FR-12, FR-16)."""
    shown = "\n".join(_judgement_lines(judged()))

    assert "a trial of drug X in condition Y" in shown
    assert "Forty patients were given drug X" in shown
    assert "**Relevance:** rated 8 of 10; the agent would include it" in shown


def test_the_ncbi_wording_is_the_wording_the_record_holds():
    """NCBI asks that its notice be evident, as published (OB-03)."""
    # The wording is kept in `docs/development.md` and repeated in the
    # application. A screen should not read its legal wording out of a
    # documentation file while it draws, where a missing or edited file would
    # quietly remove the notice. This test is what stops the two drifting apart,
    # and it fails if either changes without the other.
    #
    # It reads the development page rather than the boundaries document it read
    # until 23 September, because that document is kept back from the release
    # and this test travels with one: a released clone would have run it against
    # a file it does not carry.
    recorded = Path(REPO_ROOT / "docs" / "development.md").read_text(
        encoding="utf-8")
    quoted = " ".join(
        line.lstrip("> ").strip()
        for line in recorded.splitlines() if line.strip().startswith(">"))
    quoted = re.sub(r"\s+", " ", quoted)

    assert NCBI_DISCLAIMER in quoted
    assert NCBI_COPYRIGHT in quoted
    assert NCBI_NOTICE_URL in recorded


def test_the_notice_carries_its_link():
    """The published page covers material that is not in the public domain,
    including PubMed Central, so a reader should be able to reach all of it."""
    assert NCBI_NOTICE_URL.startswith("https://www.ncbi.nlm.nih.gov/")


def a_report(state=None, failures=((), (), ())):
    """A coverage report. `failures=None` is a run nobody took a snapshot of."""
    return report_on(state or a_state(), failures)


def a_screen(state=None, report=None):
    """The application, rendered from its own file, with a run loaded."""
    # Run from the file rather than from the function, because Streamlit runs a
    # script and a function lifted out of its module loses everything the module
    # holds. This is what a reviewer's browser would run.
    from streamlit.testing.v1 import AppTest

    screen = AppTest.from_file(str(REPO_ROOT / "research_agent" / "ui" / "app.py"))
    screen.session_state["run"] = (state or a_state(screened=[judged()]),
                                   report or a_report())
    return screen.run()


def a_screen_with_no_run():
    """The screen as somebody meets it before there is anything to review."""
    # Worth testing on its own, because it is the only state that has to work
    # for a person who has not run anything yet: the form that asks for the
    # question is here, and so is the log of runs made before this session.
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(
        str(REPO_ROOT / "research_agent" / "ui" / "app.py")).run()


def written_on(screen):
    """Everything the screen put on the page, as one block."""
    drawn = [item.value for item in screen.markdown]
    drawn += [item.value for item in screen.subheader]
    drawn += [item.value for item in screen.header]
    drawn += [item.value for item in screen.caption]
    return "\n".join(drawn)


def test_the_screen_shows_the_ncbi_notice(monkeypatch):
    """NCBI asks that it be evident to users, which is a claim about the
    screen and not about the source (OB-03)."""
    screen = a_screen()

    shown = written_on(screen)
    assert NCBI_DISCLAIMER in shown
    assert NCBI_COPYRIGHT in shown
    assert NCBI_NOTICE_URL in shown


def test_the_screen_shows_each_paper_with_its_citation_and_judgement():
    """A title alone reads well and cannot be followed up."""
    paper = a_paper("A trial of drug X").model_copy(update={
        "authors": ["Smith J"], "year": "2024", "journal": "Journal of Trials",
        "pmid": "12345678", "publication_status": "Retracted after publication"})
    state = a_state(screened=[judged().model_copy(update={"paper": paper})])

    shown = written_on(a_screen(state))

    # Numbered from 1 among the cards shown, since the finding of 18
    # September that a card carried no number a person could refer to.
    assert "**1. A trial of drug X** :gray-badge[rated 8 of 10]" in shown
    assert ("Smith J (2024) · *Journal of Trials* · PubMed: "
            "[12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)") in shown
    assert "**Publication status:** Retracted after publication" in shown
    assert "**Extract from the paper (found in the paper):**" in shown


def test_the_top_bar_is_given_an_edge():
    """Streamlit's bar sits over the page in the page's own colour, and the
    session of 16 September saw the title and a focused box vanish under it
    (UX-05).
    """
    # The one style rule written into the page draws the bar's lower edge; this
    # checks the rule is on the page and names the bar.
    shown = written_on(a_screen())

    assert 'header[data-testid="stHeader"]' in shown
    assert "border-bottom" in shown


def test_an_uncertain_judgement_reads_as_a_sentence():
    """The decision word was once written straight into the sentence, and the
    third decision is not a verb: the screen of 16 September read
    "the agent would uncertain it" (UX-CHK-05)."""
    unsure_screening = judged().screening.model_copy(update={"decision": "uncertain"})
    unsure = judged().model_copy(update={"screening": unsure_screening})
    state = a_state(screened=[unsure])

    shown = written_on(a_screen(state))

    assert "the agent is not sure about it" in shown
    assert "would uncertain" not in shown


def test_the_extract_is_the_one_line_drawn_as_a_quotation():
    """The extract is the paper's words and what the reviewer reads to check the
    rest; on the screen of 16 September it was one weight with the agent's
    reason above it (UX-05).
    """
    # It is drawn as a Markdown quotation under its label, and no other line of
    # the paper is.
    drawn = [item.value for item in a_screen(a_state(screened=[judged()])).markdown]

    quoted = [line for line in drawn if line.startswith("> ")]
    assert quoted == ["> Forty patients were given drug X"]
    label = drawn.index("**Extract from the paper (found in the paper):**")
    assert drawn[label + 1] == "> Forty patients were given drug X"


def test_an_extract_of_two_paragraphs_is_quoted_whole():
    """Both paragraphs must remain visibly part of the same extract (R-29)."""
    # Every line carries the mark, the blank one between the paragraphs too,
    # and the marked lines are drawn as one element, so the reader sees one
    # quotation and not two with the agent's words between.
    two_paragraphs = "Forty patients were given drug X.\n\nTen improved."
    screening = judged().screening.model_copy(update={"extract": two_paragraphs})
    paper = judged().model_copy(update={"screening": screening})

    drawn = [item.value for item in a_screen(a_state(screened=[paper])).markdown]

    quoted = [line for line in drawn if line.startswith("> ")]
    assert quoted == ["> Forty patients were given drug X.\n> \n> Ten improved."]
    unquoted = [line for line in drawn if not line.startswith("> ")]
    assert not any("Ten improved" in line for line in unquoted)


def test_a_short_run_warns_at_the_top_with_the_report_underneath():
    """The headline is what a reviewer reads first; the report is there for
    somebody who wants the detail."""
    short = a_state(screened=[judged()],
                    pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])
    screen = a_screen(short, a_report(short))

    assert screen.warning, "the reviewer is told before they read the papers"
    assert screen.warning[0].value == (
        "This run did not search everything it planned. What is missing is "
        "under 'Run details', and the papers here are what it did find.")
    assert screen.expander, "and the report itself is underneath"
    assert screen.expander[0].label == "Run details"


def test_a_complete_run_says_so_rather_than_warning():
    screen = a_screen()

    assert not screen.warning
    assert screen.success[0].value == "Search completed."


def test_a_run_that_found_nothing_says_so_and_offers_no_store():
    """Every search ran and none found a paper."""
    # The screen used to show the green headline and a store button with
    # nothing between them, and a reader could not tell that nothing had been
    # found (FR-16, UX-06). Since the second session of 19 September the two
    # sentences share the green box, in the words given at that session.
    empty = a_state(screened=[],
                    results=ResultsBuffer(
                        outcomes=[Outcome(subgoal=Subgoal(purpose="find trials",
                                                          source="pubmed",
                                                          query="drug X trials"),
                                          result="finished", papers_found=0)],
                        papers=[], ready=True))
    screen = a_screen(empty, a_report(empty))

    assert screen.success[0].value == "Search completed. No paper found."
    assert "No paper found" not in written_on(screen), "said once, in the box"
    keys_of_buttons_drawn = [button.key for button in screen.button]
    assert "store" not in keys_of_buttons_drawn, "nothing to store"
    assert "new search" in keys_of_buttons_drawn, "but a way on"


def a_run_that_stopped_before_screening(papers):
    """A run with these papers retrieved, no hand-over and no screening,
    stopped by a refused model request: the live run of 17 September."""
    outcome = Outcome(subgoal=Subgoal(purpose="find trials", source="pubmed",
                                      query="drug X trials"),
                      result="finished", papers_found=len(papers))
    return a_state(screened=[],
                   results=ResultsBuffer(outcomes=[outcome], papers=papers,
                                         ready=False),
                   needs_human_attention=True,
                   attention_reason="the next move could not be decided: 503")


def test_papers_retrieved_and_none_screened_are_not_called_not_found():
    """The live run of 17 September retrieved 150 records over three searches and
    stopped on a refused model request, and the page said "No paper was found"
    over them.
    """
    # The sentence now says how many were retrieved, that the run stopped
    # before screening, and where the reason is (FR-05, FR-16).
    papers = [a_paper("A trial of drug X"), a_paper("Another trial of drug X")]
    stopped = a_run_that_stopped_before_screening(papers)
    screen = a_screen(stopped, a_report(stopped))

    assert screen.warning, "the reason is under the warning's section"
    assert screen.warning[0].value == (
        "This run did not search everything it planned. 2 papers were "
        "retrieved and none was screened: the run stopped before screening. "
        "The reason is under 'Run details'. Nothing can be reviewed until the "
        "question is run again.")
    assert "No paper found" not in written_on(screen)
    assert "store" not in [button.key for button in screen.button]


def test_a_run_that_stopped_before_retrieving_anything_says_so():
    """The second attempt of 17 September was refused at the plan itself: nothing
    was searched, so "no paper found" would be an answer to a search that never
    happened.
    """
    # The words are the second session's, of 19 September, and go in the yellow
    # box with the headline.
    stopped = a_run_that_stopped_before_screening([])
    screen = a_screen(stopped, a_report(stopped))

    assert why_there_is_nothing_to_review(a_report(stopped)) == (
        "The run failed before it retrieved anything. Details are under "
        "'Run details'.")
    assert screen.warning[0].value == (
        "This run did not search everything it planned. The run failed before "
        "it retrieved anything. Details are under 'Run details'.")
    assert "The run needs a person: the next move could not be decided: 503" \
        in written_on(screen), "the reason itself is in the section"


def test_a_run_whose_searches_all_failed_says_it_failed_without_asking_for_a_person():
    """The launcher's failed state until 19 September: two searches failed,
    nothing retrieved.
    """
    # A real run does not ask for a person here, and the second session of 19
    # September found the invented "needs a person" line misleading, so it went; the
    # sentence saying the run failed does not depend on it.
    failed = a_state(screened=[],
                     results=ResultsBuffer(
                         outcomes=[Outcome(subgoal=Subgoal(purpose="find trials",
                                                           source="pubmed",
                                                           query="drug X trials"),
                                           result="failed", papers_found=0,
                                           reason="PubMed answered 503 three times")],
                         papers=[], ready=True))
    screen = a_screen(failed, a_report(failed))

    assert screen.warning[0].value == (
        "This run did not search everything it planned. The run failed before "
        "it retrieved anything. Details are under 'Run details'.")
    assert "The run needs a person" not in written_on(screen)
    assert "a service that failed may answer then" in written_on(screen)


def test_a_finished_run_with_no_paper_still_says_none_was_found():
    empty = a_state(screened=[], results=ResultsBuffer(outcomes=[], papers=[],
                                                      ready=True))

    assert why_there_is_nothing_to_review(a_report(empty)) == "No paper found."


def test_a_decision_survives_the_rerun_a_click_causes():
    """Streamlit runs the whole script again on every click."""
    # Held in an ordinary variable, the answer would be built again from the run
    # and the reviewer would watch their decisions disappear as they worked.
    screen = a_screen()

    screen.button(key="approved 0").click().run()

    assert screen.session_state["decisions"][which_paper(judged())][0] == "approved"


def test_storing_asks_storage_and_reports_what_it_refused(monkeypatch):
    """A refusal belongs on the screen, not in a log nobody reads: at the top of
    the page, where the reviewer looks, and in the log.
    """
    # Under the button it was at the foot of the panel, below the fold of a
    # 731-pixel window, and a press that seemed to do nothing was reported on
    # 17 September with nothing red in sight.
    from research_agent.processing import storage

    def refuse(screened, store=None):
        raise storage.RefusedByStorage("the extract was not found in the paper")

    monkeypatch.setattr(storage, "save_approved", refuse)
    screen = a_screen(a_state(run_id="run-1", screened=[judged()]))
    screen.button(key="approved 0").click().run()

    screen = screen.button(key="store").click().run()

    assert "was not found in the paper" in screen.main.error[0].value
    kinds = [type(element).__name__ for element in screen.main]
    # The first bordered block is the first card; the progress line above
    # the title is a block too, without a border.
    cards = [index for index, element in enumerate(screen.main)
             if type(element).__name__ == "Block"
             and element.proto.flex_container.border]
    assert kinds.index("Error") < cards[0], "at the top, before the first card"
    assert not list(screen.sidebar.error)
    failures = [line for line in run_log.read() if line["event"] == "failure"]
    assert [line["what"] for line in failures] == ["storing the approved papers"]


def test_any_fault_while_storing_is_said_at_the_top_and_logged(monkeypatch):
    """Not only the store's own refusal."""
    # A fault of any other kind, raised under the button, was drawn at the foot
    # of the panel where nobody saw it; every fault is now carried to the top
    # of the page with its own words, and written to the log.
    from research_agent.processing import storage

    def break_down(screened, store=None):
        raise OSError("the disk is not there")

    monkeypatch.setattr(storage, "save_approved", break_down)
    screen = a_screen(a_state(run_id="run-1", screened=[judged()]))
    screen.button(key="approved 0").click().run()

    screen = screen.button(key="store").click().run()

    assert screen.main.error[0].value == (
        "The approved papers were not stored: OSError: the disk is not there")
    assert not list(screen.exception), "not left to Streamlit to draw where it fell"
    assert "page" not in screen.session_state, "the review page stays"
    failures = [line for line in run_log.read() if line["event"] == "failure"]
    assert failures[0]["why"] == "OSError: the disk is not there"


def test_storing_is_given_the_answers_the_reviewer_made(monkeypatch):
    """What is stored should be worked out from the decisions rather than
    from the order buttons happened to be pressed."""
    from research_agent.processing import storage

    given = []

    def remember(screened, store=None):
        given.append(screened)
        return sum(1 for item in screened if item.review == "approved")

    monkeypatch.setattr(storage, "save_approved", remember)
    screen = a_screen(a_state(run_id="run-1", screened=[judged()]))
    screen.button(key="approved 0").click().run()

    screen.button(key="store").click().run()

    assert [item.review for item in given[0]] == ["approved"]
    # The store is written to the log with its count and time, so the
    # results page can say when the review finished.
    stored = [line for line in run_log.read() if line["event"] == "approved papers stored"]
    assert [line["papers stored"] for line in stored] == [1]
    assert stored[0]["run"] == "run-1"


def test_the_finish_time_is_the_clock_on_the_wall():
    """The log writes UTC; a person reads local time. A stamp the log lacks
    or one that cannot be read is said to be not recorded or shown as it
    is, never as a blank."""
    local = datetime.datetime.fromisoformat("2026-09-17T14:40:00+00:00").astimezone()
    assert when_the_review_finished("2026-09-17T14:40:00+00:00") == (
        f"{local.day} {local:%b %Y, %H:%M}")
    assert when_the_review_finished("") == "time not recorded"
    assert when_the_review_finished("not a time") == "not a time"


def test_the_list_downloads_as_one_csv_with_the_link():
    """The mock-up of 17 September has a "Download the list" button:
    one file with the title, authors, year, journal, the publication status,
    the link, the summary and the note; made by the browser on this
    computer, no service asked. The status column is the store's value,
    empty where no check was made (FR-09; added 22 September)."""
    from research_agent.processing.results import ApprovedPaper

    papers = [ApprovedPaper(identifier="12345678", doi="", pmid="12345678",
                            paper_id="", title="A trial of drug X",
                            authors="Smith J; Jones A", year="2024",
                            journal="Journal of Trials",
                            publication_status="Retracted after publication",
                            summary="a trial", review_note="sound"),
              ApprovedPaper(identifier="abc", doi="", pmid="", paper_id="abc",
                            title="A cohort", authors="Brown K", year="2020",
                            journal="", publication_status="", summary="a cohort",
                            review_note="")]

    rows = list(csv.reader(io.StringIO(the_list_as_a_file(papers))))

    assert rows[0] == ["title", "authors", "year", "journal", "publication status",
                       "link", "summary", "your note"]
    assert rows[1] == ["A trial of drug X", "Smith J; Jones A", "2024",
                       "Journal of Trials", "Retracted after publication",
                       "https://pubmed.ncbi.nlm.nih.gov/12345678/", "a trial", "sound"]
    assert rows[2] == ["A cohort", "Brown K", "2020", "", "", "Semantic Scholar abc",
                       "a cohort", ""]


def test_a_stored_retraction_is_shown_on_the_results_card():
    """The status the hand-over stored reaches the list a person uses after
    the review, not only the review page (FR-09; found in review on 21
    September: a stored retraction did not survive into the results)."""
    from research_agent.processing import run_log

    retracted = a_paper_the_store_can_keep().model_copy(update={
        "paper": a_paper_the_store_can_keep().paper.model_copy(
            update={"publication_status": "Retracted after publication"})})
    state = a_state(run_id="run-1", screened=[retracted])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()

    screen = screen.button(key="store").click().run()

    shown = written_on(screen)
    assert "**Publication status:** Retracted after publication" in shown
    assert "**A trial of drug X** :green-badge[approved]" in shown


def a_paper_the_store_can_keep():
    """A judged paper with an identifier, which is what storage insists on."""
    paper = a_paper().model_copy(update={
        "authors": ["Smith J", "Jones A"], "year": "2024",
        "journal": "Journal of Trials", "pmid": "12345678"})
    return judged().model_copy(update={"paper": paper})


def a_reviewed_and_stored_screen():
    """A run with one approved paper, stored: the screen is on the outcome."""
    # The store call is the real one. Every test has the store's name pointed
    # at a temporary file, and the write and the results page's read both
    # follow that name, so what the page shows is what storage kept.
    from research_agent.processing import run_log

    # Named, as a run the graph made is: the log files lines under the
    # identifier, and the outcome is read back by it.
    state = a_state(run_id="run-1", screened=[a_paper_the_store_can_keep()])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    return screen.button(key="store").click().run()


def test_storing_moves_the_screen_to_the_outcome_as_the_mock_up_has_it():
    """Once the papers are stored the page moves by itself to the approved
    papers, in the shape of the mock-up of 17 September.
    """
    # Asked for on 17 September, the mock-up given that evening
    # (11-results-page-mockup.png): the heading with the count, a grey line
    # with the question and the finish time, a box saying the papers are saved
    # with the list to download, the counts line, one bordered card per
    # approved paper with the badge, the grey citation with the identifier as
    # a link, the summary, the note and a button to the paper's page; no
    # review controls in the way.
    screen = a_reviewed_and_stored_screen()

    assert [item.value for item in screen.header] == ["Review complete: 1 paper approved"]
    shown = written_on(screen)
    assert 'Screening of "Does drug X treat condition Y?", finished ' in shown
    assert "**All approved papers are saved**" in shown
    # The box is named for the page's second style rule, which tints it
    # green as the mock-up has it (chosen on the evening of 17 September,
    # `13-results-page-as-built-box-to-change.png`); the rule
    # itself is in the page's one style block.
    boxes = [block for block in screen.main.children.values()
             if type(block).__name__ == "Block" and block.proto.id.endswith("saved-papers")]
    assert len(boxes) == 1 and boxes[0].proto.flex_container.border
    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    assert ".st-key-saved-papers { background-color: rgba(33, 195, 84, 0.12);" in style
    # Where the list is, in a person's terms; the file's path is on the
    # record of runs and nowhere a reviewer reads (the review of the screen on
    # 17 September).
    assert ("They are on this computer and are shown again from the start "
            "screen under \"Approved papers of an earlier run\".") in shown
    assert "approved_records.sqlite3" not in shown
    assert "**1** approved · **0** flagged · **0** rejected" in shown
    assert "**A trial of drug X** :green-badge[approved]" in shown
    assert ("Smith J et al. (2024) · *Journal of Trials* · PubMed: "
            "[12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)") in shown
    assert "a trial of drug X in condition Y" in shown
    assert "trials" in shown and "reviews" in shown, "the recap is on the page, folded"
    downloads = {item.key: item.label for item in screen.get("download_button")}
    assert downloads == {"download the list": "Download the list"}
    assert all(item.proto.url.endswith(".csv") for item in screen.get("download_button"))
    assert [(element.proto.label, element.proto.url)
            for element in screen.get("link_button")] == [
        ("Open paper page", "https://pubmed.ncbi.nlm.nih.gov/12345678/")]
    keys = {button.key for button in screen.button}
    assert {"back to the papers", "new search"} <= keys
    assert "store" not in keys and "approved 0" not in keys
    # In the panel, where they stay in view while the list scrolls, as the
    # review of the screen on 17 September decided. The record's line is the
    # one thing on the page's foot, after the notice, and nothing of it is
    # in the panel (their message of the same evening).
    assert {button.key for button in screen.sidebar.button} == {
        "back to the papers", "new search"}
    # The first two are the progress line's steps behind this one (18
    # September); the record's line is the page's own one button.
    assert [(button.key, button.label) for button in screen.main.button] == [
        ("step Search", "Search"), ("step Assessment", "Assessment"),
        ("run record", "Record of this run, for checking what was done")]
    assert [type(element).__name__ for element in screen.main][-2:] == [
        "Markdown", "Button"], "the record's line is the last thing on the page"
    # The file's path is on the record's page and nowhere a reviewer reads.
    assert any("approved_records.sqlite3" in item.value
               for item in on_the_record_page(screen).caption)


def test_the_store_button_is_labelled_for_what_the_reviewer_sees_next():
    """The button reads "Display Approved", after the page it opens; its key
    stays "store".
    """
    # Named so in the review of the screen on 17 September. The key is what the tests
    # and the session know the button by, and the press still stores first, as
    # the results page's first line says.
    screen = a_screen()

    labels = {button.key: button.label for button in screen.sidebar.button}
    assert labels["store"] == "Display Approved"


def test_the_panels_two_actions_are_white_with_violet_lettering():
    """The panel's two actions, "Display Approved" and the send-back button,
    are white with violet lettering and a violet border, apart from the
    panel's other buttons.
    """
    # Asked for on the evening of 17 September: a white face, the panel's
    # violet #440099 for the lettering and a three-pixel border in the lighter
    # #ad73e5 (picture 20: a border in the panel's own violet could not be
    # seen), not a button filled in the accent, whose red had read as a
    # warning. Both stay Streamlit's plain type and the page's style rule,
    # aimed at their keys, does the drawing; the answer buttons on every card
    # are left as they were. The panel's help sentences lose the caption's
    # faded opacity, and are white as decided.
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?").run()
    screen.button(key="flagged 0").click().run()

    types = {button.key: button.proto.type for button in screen.button}
    assert types["store"] == "secondary" and types["send back"] == "secondary"
    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    # The second send-back button of 18 September, the one that sends the
    # rejected papers too, takes the same rule by its own key.
    assert (".st-key-store button, .st-key-send-back button, "
            ".st-key-send-back-all button "
            "{ background-color: #ffffff; color: #440099; "
            "border: 3px solid #ad73e5; }") in style
    assert ('section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] '
            "{ opacity: 1; }") in style


def test_each_answer_button_carries_its_sentence_as_hover_text():
    """Asked for on 18 September: the panel's three sentences on the buttons
    themselves, shown when the pointer rests on one: a reviewer at paper forty
    has the panel in view but their eyes on the card.
    """
    # The sentence is held once and the panel's headed entry is made from it,
    # so the two cannot drift apart. On a phone there is no pointer, and the
    # panel's text stays the reference.
    from research_agent.ui.app import WHAT_EACH_ANSWER_DOES

    screen = a_screen()

    helps = {button.key: button.proto.help for button in screen.main.button
             if button.key.endswith(" 0")}
    assert helps == {
        "approved 0": "Use when the paper is relevant and should be kept.",
        "flagged 0": WHAT_EACH_ANSWER_DOES["flagged"],
        "rejected 0": WHAT_EACH_ANSWER_DOES["rejected"]}
    assert WHAT_THE_ANSWERS_DO[0] == "**Approve:** " + helps["approved 0"]


def in_drawing_order(block):
    """Every element under a block, depth first, in the order it was drawn."""
    found = []
    for child in block.children.values():
        if hasattr(child, "children"):
            found.extend(in_drawing_order(child))
        else:
            found.append(child)
    return found


def test_the_answers_sit_above_the_note_box_so_the_hover_text_never_covers_it():
    """On the live run of 18 September the hover text on an answer was found
    opening upwards over the note box that sat above the buttons.
    """
    # Streamlit places a button's tooltip itself and the script cannot move it,
    # so the answers are drawn first and the box under them, where the tooltip
    # cannot reach. The note is read from the session before the box is drawn,
    # and the redrawing after a decision is asked for after the box, because
    # Streamlit forgets the value of a box a drawing did not reach: a note
    # typed and then decided on must still be in the box afterwards.
    screen = a_screen()

    card = [block for block in screen.main.children.values()
            if type(block).__name__ == "Block" and block.proto.flex_container.border][0]
    keys = [getattr(element, "key", None) for element in in_drawing_order(card)]
    assert keys.index("rejected 0") < keys.index("note 0")
    assert keys.index("approved 0") < keys.index("note 0")

    screen.text_input(key="note 0").set_value("not sure it is a trial")
    screen = screen.button(key="flagged 0").click().run()
    assert screen.text_input(key="note 0").value == "not sure it is a trial"
    assert screen.text_input(key="note 0").label == "Your note, kept until you decide again"
    assert "because not sure it is a trial" in written_on(screen)


def test_papers_rated_five_or_below_are_kept_back_from_the_page_and_counted(monkeypatch):
    """The Assessment page shows the papers the agent rated above 5 of 10 and
    counts the rest in one line, "N papers rated 5 or below, not shown".
    """
    # The rule set on 18 September, after the live run put fifty papers in
    # front of the reviewer. The kept-back papers stay in the run and in the
    # coverage report's counts; the panel's count and its storing sentence are
    # about the papers shown; and storing keeps a kept-back paper out, as a
    # paper with no decision (FR-13).
    from research_agent.processing import audit_log, storage

    three = a_state(screened=[judged("A trial of drug X", relevance=8),
                              judged("A cohort of drug X", relevance=6),
                              judged("A review of drug X", relevance=3)])
    screen = a_screen(three)

    shown = written_on(screen)
    # Numbered among the cards shown, so the second card is 2 even though
    # a kept-back paper would have been the run's third.
    assert "**1. A trial of drug X**" in shown and "**2. A cohort of drug X**" in shown
    assert "A review of drug X**" not in shown
    assert "1 paper rated 5 or below, not shown." in shown
    assert {button.key for button in screen.main.button if button.key.startswith("approved")} == {
        "approved 0", "approved 1"}
    assert ("0 of 2 papers decided, 2 waiting: 0 approved, 0 flagged, "
            "0 rejected.") in [item.value for item in screen.sidebar.markdown]
    assert any("2 papers currently have no decision" in item.value
               for item in screen.sidebar.markdown)

    report = audit_log.report_on(three)
    assert report.papers_screened == 3
    assert report.papers_rated_five_or_below == 1
    assert "Papers rated 5 or below, not shown: 1" in audit_log.what_the_run_did(report)
    assert "Papers rated 5 or below" not in " ".join(
        audit_log.what_the_run_did(audit_log.report_on(a_state(screened=[judged()]))))

    stored = {}
    monkeypatch.setattr(storage, "save_approved",
                        lambda screened: stored.update(
                            {judged.paper.title: judged.review for judged in screened}) or 0)
    screen.button(key="approved 0").click().run()
    screen.button(key="store").click().run()
    assert stored == {"A trial of drug X": "approved", "A cohort of drug X": "waiting",
                      "A review of drug X": "waiting"}


def test_a_run_whose_every_paper_was_rated_five_or_below_says_so():
    """Nothing to put in front of the reviewer is said in words, as a run that
    found nothing is, and the review's panel is not drawn: its buttons would
    have nothing to act on.
    """
    # The panel with the way on is (since the second session of 19
    # September), and the record's line stays, since the papers are in the
    # record.
    screen = a_screen(a_state(screened=[judged("A review of drug X", relevance=2),
                                        judged("A letter on drug X", relevance=0)]))

    assert ("The agent rated all 2 papers 5 or below, so none is shown for "
            "review. They stay in the run record.") in written_on(screen)
    assert "A review of drug X**" not in written_on(screen)
    assert {button.key for button in screen.button} == {
        "step Search", "step Result", "run record", "new search"}
    assert {button.key for button in screen.sidebar.button} == {"new search"}


def test_a_run_with_nothing_to_review_still_offers_a_new_search():
    """The second session of 19 September found the failed state with no panel at
    all, so the reviewer had no way on but the browser.
    """
    # Every state with no paper on the page draws the panel with "Start a new
    # search", under the heading the results page's panel gives the same
    # button, and nothing else: no "Display Approved" with nothing to store.
    # The press clears the run and draws the empty form, as it does from the
    # results page.
    failed = a_state(screened=[],
                     results=ResultsBuffer(
                         outcomes=[Outcome(subgoal=Subgoal(purpose="find trials",
                                                           source="pubmed",
                                                           query="drug X trials"),
                                           result="failed", papers_found=0,
                                           reason="PubMed answered 503 three times")],
                         papers=[], ready=True))
    screen = a_screen(failed, a_report(failed))

    assert [item.value for item in screen.sidebar.subheader] == ["What next"]
    assert [(button.key, button.label) for button in screen.sidebar.button] == [
        ("new search", "Start a new search")]
    assert "store" not in {button.key for button in screen.button}

    screen = screen.button(key="new search").click().run()

    assert "run" not in screen.session_state
    assert screen.text_area(key="question").value == ""
    assert screen.button(key="start") is not None


def test_a_paper_sent_back_stays_shown_whatever_its_new_rating(monkeypatch):
    """A fresh judgement the reviewer asked for is always put in front of them,
    even one that rated the paper below the line: a paper that vanished after
    the send-back would hide the very change they sent it back to see.
    """
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X", relevance=2)]}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    screen = screen.button(key="send back").click().run()

    assert "**1. A trial of drug X** :red-badge[rated 2 of 10 (was 8)]" in written_on(screen)
    assert "not shown" not in written_on(screen)
    # The paper has no identifier, so it is filed under its title; what the
    # agent had said is kept for the card to show the change (finding d).
    assert screen.session_state["what the agent had said about the papers sent back"] == {
        "A trial of drug X": ("include", 8)}


def test_a_card_judged_again_says_what_the_agent_changed(monkeypatch):
    """Each re-judged card says, in red, what the agent had said and what it
    says now, decision and rating, or "unchanged" in one word.
    """
    # On the live run of 18 September the agent changed its mind on four
    # papers of forty-one sent back and the page said nothing of it, so the
    # reviewer could not tell whether the second judgement had been worth
    # asking for (finding d). The new reason is the card's reason line. A
    # paper that did not go back carries no such line.
    from research_agent import graph

    def judged_again(state, curator, answers_that_go=None):
        fresh = []
        for paper in state.screened:
            if paper.paper.title == "A trial of drug X":
                fresh.append(judged("A trial of drug X").model_copy(update={
                    "screening": paper.screening.model_copy(update={
                        "decision": "exclude", "relevance": 0,
                        "reason": "on a second look it is a review"})}))
            elif paper.paper.title == "A cohort of drug X":
                fresh.append(judged("A cohort of drug X"))
            else:
                fresh.append(paper)
        return state.model_copy(update={"screened": fresh})

    monkeypatch.setattr(graph, "judge_again", judged_again)
    screen = a_screen(a_state(screened=[judged("A trial of drug X"),
                                        judged("A cohort of drug X"),
                                        judged("A study of drug X")]))
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    screen.text_input(key="note 1").set_value("is it a cohort?")
    screen.button(key="flagged 1").click().run()
    screen = screen.button(key="send back").click().run()

    shown = written_on(screen)
    assert ":red[**Judged again:** from: include 8 to: exclude 0]" in shown
    assert "**Reason:** on a second look it is a review" in shown
    assert shown.count("**Judged again:** unchanged") == 1
    assert shown.count("Judged again") == 2, "the paper that did not go back says nothing"
    assert shown.index("**1. A trial of drug X**") < shown.index("from: include 8") \
        < shown.index("**2. A cohort of drug X**")
    # The second session of 19 September: the from-to line was
    # there and the change still did not catch the eye, because the badge
    # and the relevance line showed the new values in the ordinary style.
    # The changed values themselves are marked, the old ones in brackets.
    assert "**1. A trial of drug X** :red-badge[rated 0 of 10 (was 8)]" in shown
    assert (":red[**Relevance:** rated 0 of 10; the agent would exclude it "
            "(was rated 8 of 10; the agent would include it)]") in shown
    assert "**2. A cohort of drug X** :gray-badge[rated 8 of 10]" in shown
    assert "**Relevance:** rated 8 of 10; the agent would include it" in shown


def test_a_second_judgement_that_changes_the_decision_alone_is_marked_too(monkeypatch):
    """A decision turned round with the rating kept: the badge goes red without
    a bracket, and the relevance line carries the old decision.
    """
    # That is what the launcher's stand-in does, and what the live run's four
    # changes were not; the badge has no bracket since its number did not
    # change.
    from research_agent import graph

    def judged_again(state, curator, answers_that_go=None):
        fresh = []
        for paper in state.screened:
            if paper.paper.title == "A trial of drug X":
                fresh.append(judged("A trial of drug X").model_copy(update={
                    "screening": paper.screening.model_copy(update={
                        "decision": "exclude"})}))
            else:
                fresh.append(paper)
        return state.model_copy(update={"screened": fresh})

    monkeypatch.setattr(graph, "judge_again", judged_again)
    screen = a_screen(a_state(screened=[judged("A trial of drug X")]))
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    screen = screen.button(key="send back").click().run()

    shown = written_on(screen)
    assert "**1. A trial of drug X** :red-badge[rated 8 of 10]" in shown
    assert (":red[**Relevance:** rated 8 of 10; the agent would exclude it "
            "(was rated 8 of 10; the agent would include it)]") in shown


def test_the_send_back_offers_the_flagged_alone_and_with_the_rejected(monkeypatch):
    """Finding f of 18 September: one press sent one flagged and forty rejected
    papers back together.
    """
    # Two buttons now, each with its count, the flagged alone first as the
    # likely case; the graph is told which answers go, and the rejected keep
    # their decision when only the flagged are sent. With no flagged paper the
    # one button offered is the rejected ones', and with nothing to send there
    # is none.
    from research_agent import graph

    told = []

    def judged_again(state, curator, answers_that_go=None):
        told.append(answers_that_go)
        return state.model_copy(update={"screened": [
            judged("A trial of drug X"), judged("A cohort of drug X"),
            judged("A review of drug X")]})

    monkeypatch.setattr(graph, "judge_again", judged_again)
    screen = a_screen(a_state(screened=[judged("A trial of drug X"),
                                        judged("A cohort of drug X"),
                                        judged("A review of drug X")]))
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    screen.text_input(key="note 1").set_value("a cohort, not a trial")
    screen.button(key="rejected 1").click().run()
    screen.text_input(key="note 2").set_value("a review")
    screen = screen.button(key="rejected 2").click().run()

    labels = {button.key: button.label for button in screen.sidebar.button}
    assert labels["send back"] == "Send the 1 flagged papers back to be judged again"
    assert labels["send back all"] == (
        "Send the 1 flagged and 2 rejected papers back to be judged again")
    assert list(labels) == ["send back", "send back all", "store"], "the flagged alone first"

    screen = screen.button(key="send back").click().run()
    assert told == [("flagged",)]
    assert set(screen.session_state["decisions"]) == {"A cohort of drug X", "A review of drug X"}, \
        "the rejected keep their decision"
    labels = {button.key: button.label for button in screen.sidebar.button}
    assert "send back" not in labels
    assert labels["send back all"] == "Send the 2 rejected papers back to be judged again"

    screen = screen.button(key="send back all").click().run()
    assert told == [("flagged",), ("flagged", "rejected")]
    assert screen.session_state["decisions"] == {}
    assert not [button for button in screen.sidebar.button if button.key.startswith("send back")]


def the_progress_line_on(screen):
    """The steps across the top of the page, in order, each as what it is
    drawn as: the current one as violet text, the others as buttons, with
    the arrows between them; and the container they sit in."""
    line = [block for block in screen.main.children.values()
            if type(block).__name__ == "Block"][0]
    drawn = []
    for element in line.children.values():
        if element.type == "button":
            drawn.append(("button", element.label, element.key, element.proto.type))
        else:
            drawn.append(("text", element.value))
    return line, drawn


def test_each_screen_is_named_for_its_step_under_the_progress_line():
    """The three screens are Search, Assessment and Result, each titled so,
    under a line reading "Search -> Assessment -> Result" (UX-05).
    """
    # Asked for on 18 September: the step the person is on in bold violet and
    # the other two pressable, so a person sees the three activities in order
    # and where they are among them. The record of runs is not one of the
    # three and carries no line. The one title that used to head every screen
    # is gone.
    from research_agent.ui.app import A_REVIEWER_DECIDES

    arrow = ("text", ":material/arrow_forward:")
    search = ("button", "Search", "step Search", "tertiary")
    assessment = ("button", "Assessment", "step Assessment", "tertiary")
    result = ("button", "Result", "step Result", "tertiary")

    screen = a_screen_with_no_run()
    assert [item.value for item in screen.title] == ["Search"]
    assert screen.main.caption[0].value == A_REVIEWER_DECIDES
    line, drawn = the_progress_line_on(screen)
    assert drawn == [("text", ":violet[**Search**]"), arrow, assessment, arrow, result]
    assert line.proto.flex_container.direction == line.proto.flex_container.HORIZONTAL
    assert line.proto.flex_container.align == line.proto.flex_container.ALIGN_CENTER

    screen = a_screen()
    assert [item.value for item in screen.title] == ["Assessment"]
    assert the_progress_line_on(screen)[1] == [
        search, arrow, ("text", ":violet[**Assessment**]"), arrow, result]

    screen = a_reviewed_and_stored_screen()
    assert [item.value for item in screen.title] == ["Result"]
    assert the_progress_line_on(screen)[1] == [
        search, arrow, assessment, arrow, ("text", ":violet[**Result**]")]

    screen = screen.button(key="run record").click().run()
    assert [item.value for item in screen.title] == []
    assert not [button for button in screen.button if button.key.startswith("step ")]
    assert "Research agent" not in written_on(screen)


def test_a_step_past_the_search_is_refused_until_a_search_is_done():
    """The first of the three rules of 18 September: Assessment or Result before any search is
    done gets "A search needs to be done first." and nothing moves."""
    screen = a_screen_with_no_run()

    for step in ("Assessment", "Result"):
        screen = screen.button(key=f"step {step}").click().run()
        assert [item.value for item in screen.warning] == [
            "A search needs to be done first."]
        assert [item.value for item in screen.title] == ["Search"]
        assert frames_on(screen) == []


def test_the_result_step_is_refused_until_the_papers_are_stored():
    """The second rule: Result before Display Approved has been pressed for the
    run gets "Press Display Approved first."
    """
    # Once the papers are stored, Assessment and Result move between the two
    # pages, each opening at its top like every other move.
    screen = a_screen()

    screen = screen.button(key="step Result").click().run()
    assert [item.value for item in screen.warning] == ["Press Display Approved first."]
    assert [item.value for item in screen.title] == ["Assessment"]
    assert frames_on(screen) == []

    screen = a_reviewed_and_stored_screen()
    screen = screen.button(key="step Assessment").click().run()
    assert [item.value for item in screen.title] == ["Assessment"]
    assert len(frames_on(screen)) == 1
    screen = screen.button(key="step Result").click().run()
    assert [item.value for item in screen.title] == ["Result"]
    assert [item.value for item in screen.warning] == []
    assert len(frames_on(screen)) == 1


def test_going_back_to_search_asks_once_and_then_resets_the_run():
    """The third rule: Search from Assessment or Result resets the current search,
    so the first press only says so and a second press confirms; the approved
    papers already stored stay in the store.
    """
    # The asking lapses with any other drawing in between, so a press made
    # later is asked again rather than taken as the confirmation of a question
    # the person may no longer have in view.
    screen = a_reviewed_and_stored_screen()
    stored = screen.session_state["run"][0].run_id

    screen = screen.button(key="step Search").click().run()
    assert [item.value for item in screen.warning] == [
        ("Going back to Search resets the current search; the approved papers "
         "already saved stay in the store. Press Search again to confirm.")]
    assert [item.value for item in screen.title] == ["Result"]

    screen = screen.run()
    screen = screen.button(key="step Search").click().run()
    assert len(screen.warning) == 1, "asked again after a drawing in between"
    assert [item.value for item in screen.title] == ["Result"]

    screen = screen.button(key="step Search").click().run()
    assert [item.value for item in screen.title] == ["Search"]
    assert "run" not in screen.session_state
    assert len(frames_on(screen)) == 1
    assert screen.selectbox(key="which stored run").value == stored

    # The stored run reopened from the form is read from disk, not a run
    # on the screen, so there is nothing to lose and nothing to ask.
    screen = screen.button(key="show stored run").click().run()
    assert [item.value for item in screen.title] == ["Result"]
    screen = screen.button(key="step Search").click().run()
    assert [item.value for item in screen.title] == ["Search"]
    assert [item.value for item in screen.warning] == []


def test_the_panel_is_divided_under_the_count_and_under_the_help():
    """Two thin lines where picture 20 of 17 September drew them: one under the
    count, one under the help, so the count, the help and the storing sentence
    read as three parts.
    """
    # Streamlit's own divider, which the panel's theme draws white; nothing is
    # drawn between the help and the send-back button, which belongs with the
    # actions below the line.
    screen = a_screen()

    panel = [(type(element).__name__, getattr(element, "value", ""))
             for element in screen.sidebar]
    kinds = [kind for kind, _ in panel]
    assert kinds.count("Divider") == 2
    first_line = kinds.index("Divider")
    assert panel[first_line - 1][0] == "Markdown", "the count is above the first line"
    assert "papers decided" in panel[first_line - 1][1]
    assert panel[first_line + 1] == ("Caption", f"_{WHAT_THE_ANSWERS_DO[0]}_")
    second_line = kinds.index("Divider", first_line + 1)
    assert panel[second_line - 1] == ("Caption", f"_{WHAT_THE_ANSWERS_DO[-1]}_")
    assert "currently has no decision" in panel[second_line + 1][1]


def test_the_panel_is_tight_enough_to_fit_a_short_window():
    """With Streamlit's usual gap between elements and its usual space round a
    rule, the panel outgrew the 731-pixel window of the session and "Display
    Approved" sat below the fold (picture 21 of 17 September).
    """
    # The panel's elements sit in one container at half the usual gap,
    # Streamlit's own setting, and the page's style rule takes the rule's own
    # space down; whether the button is in view at 1179 by 731 is seen at the
    # screen.
    screen = a_screen()

    blocks = [element for element in screen.sidebar
              if type(element).__name__ == "Block"]
    assert any("gap_size: XSMALL" in str(block.proto) for block in blocks)
    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    assert 'section[data-testid="stSidebar"] hr { margin: 0.25rem 0 1.25rem; }' in style


def test_the_panel_says_what_to_press_last_once_a_paper_is_decided():
    """Asked for on 17 September (picture 21): a reviewer whose first press was on
    a card is shown what to do next.
    """
    # Streamlit cannot scroll the panel from the script, so the panel says it
    # in a sentence under the count, from the first decision on, and the
    # tighter panel keeps the button in view. In bold, at their ask of the next
    # morning (picture 23), as the one instruction in the panel; and in a box,
    # Streamlit's info box drawn as the panel's confirmation boxes are, since
    # their second session of 19 September found the bold line did not stand
    # out on the violet.
    prompt = "**When you are done, press Display Approved below.**"
    screen = a_screen()
    assert [item.value for item in screen.sidebar.info] == []

    screen.button(key="approved 0").click().run()

    assert [item.value for item in screen.sidebar.info] == [prompt]
    assert prompt not in [item.value for item in screen.sidebar.markdown], \
        "in the box and not as a plain line as well"
    in_order = [element for element in screen.sidebar
                if type(element).__name__ in ("Markdown", "Info")]
    said_at = [type(element).__name__ for element in in_order].index("Info")
    assert "papers decided" in in_order[said_at - 1].value or \
        "paper decided" in in_order[said_at - 1].value, "right under the count"


def test_the_panel_cannot_be_folded_away_where_it_sits_beside_the_page():
    """Picture 18 of 17 September circled the control that folds the panel away,
    which takes the count and the two actions with it.
    """
    # The rule hides it only from 769 pixels up: Streamlit lays the panel over
    # the page on a window 768 wide or narrower, its "md" breakpoint read from
    # the installed bundle, and a reviewer on a phone has to be able to close
    # it. The control that opens a folded panel is not touched.
    screen = a_screen()

    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    assert ('@media (min-width: 769px) { [data-testid="stSidebarCollapseButton"] '
            "{ display: none; } }") in style
    assert "stExpandSidebarButton" not in style


def test_the_read_back_follows_the_store_the_write_went_to(monkeypatch, tmp_path):
    """The fault found on 17 September: a paper approved after Back and stored
    again was listed as withdrawn, because the launcher had moved the write to
    its own folder and the read-back still looked at the real file.
    """
    # Both now follow the store's one name, so moving it moves both; a second
    # store after Back updates the same file and the page lists both papers.
    from research_agent.processing import run_log, storage

    monkeypatch.setattr(storage, "DEFAULT_STORE",
                        tmp_path / "elsewhere" / "records.sqlite3")
    second = a_paper_the_store_can_keep().model_copy(update={
        "paper": a_paper("A cohort of drug X").model_copy(update={"pmid": "222"})})
    state = a_state(run_id="run-1",
                    screened=[a_paper_the_store_can_keep(), second])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()
    assert "Review complete: 1 paper approved" in written_on(screen)

    screen = screen.button(key="back to the papers").click().run()
    screen.button(key="approved 1").click().run()
    screen = screen.button(key="store").click().run()

    shown = written_on(screen)
    assert "Review complete: 2 papers approved" in shown
    assert "**2** approved · **0** flagged · **0** rejected" in shown
    assert "not in the current store" not in shown
    assert (tmp_path / "elsewhere" / "records.sqlite3").exists()


def test_a_run_with_some_approved_papers_in_the_store_says_which():
    """Two approved and stored, one then taken out of the store by a later
    decision: the box says one of two, the absent one is named, and no cause
    is claimed (found in review on 21 September: the page said all were saved
    and the absent one withdrawn)."""
    from research_agent.processing import run_log, storage

    second = a_paper_the_store_can_keep().model_copy(update={
        "paper": a_paper("A cohort of drug X").model_copy(update={"pmid": "222"})})
    state = a_state(run_id="run-1",
                    screened=[a_paper_the_store_can_keep(), second])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    screen.button(key="approved 1").click().run()
    screen = screen.button(key="store").click().run()
    assert "**All approved papers are saved**" in written_on(screen)

    # A later decision, as another run would write it: the store keeps the
    # last decision on a paper.
    storage.save_approved([second.model_copy(update={"review": "rejected",
                                                     "review_note": "a cohort"})])
    screen = screen.run()

    shown = written_on(screen)
    assert "**1 of 2 approved papers are in the store**" in shown
    assert "- 222: approved in this run and not in the current store." in shown
    assert "withdrew" not in shown
    assert "**A trial of drug X** :green-badge[approved]" in shown


def test_a_run_approved_and_never_stored_is_not_called_complete():
    """An approval is logged at the press and the store after it, so the
    record of runs offers a run whose store never happened; its page says so
    rather than "Review complete" (FR-14)."""
    from research_agent.processing import run_log

    state = a_state(run_id="run-1", screened=[a_paper_the_store_can_keep()])
    run_log.note_the_request(state.request, state.run_id)
    run_log.note_a_decision(state.screened[0], "approved", "", state.run_id)
    screen = a_screen_with_no_run()

    screen = screen.button(key="show stored run").click().run()

    assert [item.value for item in screen.header] == ["Review not stored: 1 paper approved"]
    shown = written_on(screen)
    assert "**No approved paper is in the store**" in shown
    assert "- 12345678: approved in this run and not in the current store." in shown
    assert not list(screen.get("download_button"))


def test_a_run_decided_further_after_its_store_is_not_called_complete():
    """Store, Back, one more decision and no second store: the kept set is
    older than the counts, and the page reopened from the record says
    "Review not stored" (found in review on 22 September)."""
    from research_agent.processing import run_log

    second = a_paper_the_store_can_keep().model_copy(update={
        "paper": a_paper("A cohort of drug X").model_copy(update={"pmid": "222"})})
    state = a_state(run_id="run-1",
                    screened=[a_paper_the_store_can_keep(), second])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()
    assert [item.value for item in screen.header] == ["Review complete: 1 paper approved"]

    screen = screen.button(key="back to the papers").click().run()
    screen.text_input(key="note 1").set_value("a cohort")
    screen.button(key="rejected 1").click().run()
    # Reopened from the record on a fresh screen, as the morning after.
    screen = a_screen_with_no_run()
    screen = screen.button(key="show stored run").click().run()

    assert [item.value for item in screen.header] == ["Review not stored: 1 paper approved"]
    assert "**1** approved · **0** flagged · **1** rejected" in written_on(screen)


def test_a_paper_sent_back_and_not_decided_again_is_counted_as_waiting(monkeypatch):
    """The results page agrees with the review page: a paper sent back and
    not yet decided again is waiting, not still rejected with its old note
    (FR-19; found in review on 21 September)."""
    from research_agent import graph
    from research_agent.processing import run_log

    def judged_again(state, curator, answers_that_go=None):
        fresh = []
        for one in state.screened:
            if one.review == "rejected":
                fresh.append(one.model_copy(update={"review": "waiting"}))
            else:
                fresh.append(one)
        return state.model_copy(update={"screened": fresh})

    monkeypatch.setattr(graph, "judge_again", judged_again)
    second = a_paper_the_store_can_keep().model_copy(update={
        "paper": a_paper("A cohort of drug X").model_copy(update={"pmid": "222"})})
    state = a_state(run_id="run-1",
                    screened=[a_paper_the_store_can_keep(), second])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    screen.text_input(key="note 1").set_value("it is a cohort")
    screen.button(key="rejected 1").click().run()
    screen.button(key="send back all").click().run()
    assert "**You said:** :red[rejected]" not in written_on(screen), \
        "the review page shows the paper waiting"

    screen = screen.button(key="store").click().run()

    shown = written_on(screen)
    assert "**1** approved · **0** flagged · **0** rejected · **1** waiting" in shown
    assert "it is a cohort" not in shown, "the old note does not stand for it"


def test_the_flagged_and_rejected_papers_unfold_under_the_approved_ones():
    """The counts line has a switch, "View flagged and rejected", that unfolds
    those papers below the approved ones, each with its decision badge and the
    reviewer's note, so the person can see what was left out and why.
    """
    # Nothing changes in the store. The titles come from the session; a run
    # reopened tomorrow shows the identifiers alone, since the log holds no
    # paper's words and the store holds only approved papers.
    from research_agent.processing import run_log

    trial = a_paper_the_store_can_keep()
    cohort = a_paper("A cohort of drug X").model_copy(update={"pmid": "222"})
    review = a_paper("A review of drug X").model_copy(update={"pmid": "333"})
    state = a_state(run_id="run-1", screened=[
        trial, judged().model_copy(update={"paper": cohort}),
        judged().model_copy(update={"paper": review})])
    run_log.note_the_request(state.request, state.run_id)
    screen = a_screen(state)
    screen.button(key="approved 0").click().run()
    screen.text_input(key="note 1").set_value("not sure it is a trial")
    screen.button(key="flagged 1").click().run()
    screen.text_input(key="note 2").set_value("it is a review")
    screen.button(key="rejected 2").click().run()
    screen = screen.button(key="store").click().run()

    shown = written_on(screen)
    assert "**1** approved · **1** flagged · **1** rejected" in shown
    assert "A cohort of drug X" not in shown, "folded until asked for"
    screen = screen.toggle(key="view flagged and rejected").set_value(True).run()

    shown = written_on(screen)
    assert "**A cohort of drug X** :yellow-badge[flagged]" in shown
    assert "**Your note:** not sure it is a trial" in shown
    assert "**A review of drug X** :red-badge[rejected]" in shown
    assert "**Your note:** it is a review" in shown
    assert shown.index("**A trial of drug X**") < shown.index("**A cohort of drug X**")

    # Reopened from the records, with no run in the session: the
    # identifier stands for the title.
    screen = screen.button(key="new search").click().run()
    screen = screen.button(key="show stored run").click().run()
    screen = screen.toggle(key="view flagged and rejected").set_value(True).run()
    shown = written_on(screen)
    assert "**222** :yellow-badge[flagged]" in shown
    assert "A cohort of drug X" not in shown


def test_back_returns_to_the_papers_with_the_decisions_as_they_were():
    """The review page of the same run, the approval still on it, and Store
    offered again: pressing it updates the run's set, because storage keys
    rows by identifier (decided on 17 September)."""
    screen = a_reviewed_and_stored_screen()

    screen = screen.button(key="back to the papers").click().run()

    assert "**You said:** :green[approved]" in written_on(screen)
    keys = {button.key for button in screen.button}
    assert "store" in keys and "back to the papers" not in keys
    screen = screen.button(key="store").click().run()
    assert [item.value for item in screen.header] == ["Review complete: 1 paper approved"]


def test_a_new_search_clears_the_run_and_draws_the_empty_form():
    """No reload: the run and the decisions go, the form comes back empty,
    and the run just stored stays in the records to be opened again."""
    screen = a_reviewed_and_stored_screen()

    screen = screen.button(key="new search").click().run()

    assert "run" not in screen.session_state
    assert screen.session_state["decisions"] == {}
    assert screen.text_area(key="question").value == ""
    assert screen.button(key="start") is not None
    assert screen.selectbox(key="which stored run") is not None


def frames_on(screen):
    """The embedded frames on the page, by the script each carries."""
    found = []
    for element in screen.main:
        if getattr(element, "type", "") == "iframe":
            found.append(element.proto.srcdoc)
    return found


def test_a_new_search_moves_the_page_to_the_top_once():
    """Streamlit keeps the scroll position across a rerun, so the empty form came
    into view at its foot after the results page; the review of the screen on
    17 September decided on the top (UX-05).
    """
    # The page is moved by the one script it carries, in an embedded frame
    # drawn on that rerun and not on the next, so it runs once. Whether it
    # moves the page in a browser is seen at the screen; this reads that the
    # frame is there once.
    screen = a_reviewed_and_stored_screen()
    screen = screen.run()
    assert frames_on(screen) == []

    screen = screen.button(key="new search").click().run()
    assert len(frames_on(screen)) == 1
    assert frames_on(screen)[0].startswith(SCROLL_TO_THE_TOP)
    assert 'querySelector(\'[data-testid="stMain"]\').scrollTo(0, 0)' in SCROLL_TO_THE_TOP

    screen = screen.run()
    assert frames_on(screen) == []


def test_every_move_between_pages_opens_the_next_page_at_its_top(monkeypatch):
    """Picture 21 of 17 September: after "Start the run" the papers came into view
    at the fourth card, because the form had been scrolled to its button.
    """
    # The same frame that moves the page after a new search is drawn after
    # every move: the run's start, the store, Back, the record and its Back, an
    # earlier run opened. Each frame carries its own number, because Streamlit
    # leaves an element alone when the next drawing puts the same one in the
    # same place, and two moves in a row would otherwise leave the first frame
    # standing with its script already run.
    from research_agent import graph

    monkeypatch.setattr(graph, "run", lambda request, planner, curator, step_limit=None:
                        a_state(run_id="run-1", screened=[a_paper_the_store_can_keep()]))
    screen = a_screen_with_no_run()
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")

    frames = []
    screen = screen.button(key="start").click().run()
    frames += frames_on(screen)
    screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()
    frames += frames_on(screen)
    screen = screen.button(key="run record").click().run()
    frames += frames_on(screen)
    screen = screen.button(key="back from the record").click().run()
    frames += frames_on(screen)
    screen = screen.button(key="back to the papers").click().run()
    frames += frames_on(screen)

    assert len(frames) == 5, "one frame per move, none on the reruns between"
    assert all(frame.startswith(SCROLL_TO_THE_TOP) for frame in frames)
    assert len(set(frames)) == 5, "no two frames alike"


def test_the_send_back_opens_the_page_at_its_top_too(monkeypatch):
    """The send-back draws the same scroll frame as every other move, once,
    on the drawing after it.
    """
    # The one move that lacked the frame: on the live run of 18 September the
    # page came back after the send-back wherever it had been scrolled to,
    # with the line saying the papers were judged again out of sight above
    # (finding c of 18 September).
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X")]}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen = screen.button(key="flagged 0").click().run()
    assert frames_on(screen) == []

    screen = screen.button(key="send back").click().run()
    assert len(frames_on(screen)) == 1
    assert frames_on(screen)[0].startswith(SCROLL_TO_THE_TOP)
    screen = screen.run()
    assert frames_on(screen) == []


def test_an_earlier_run_can_be_opened_from_the_form_screen():
    """Reopened from the records, the outcome is the same page, without
    Back: there is no review page in the session to go back to."""
    stored = a_reviewed_and_stored_screen()
    run = stored.session_state["run"][0].run_id

    screen = a_screen_with_no_run()
    assert screen.selectbox(key="which stored run").value == run
    screen = screen.button(key="show stored run").click().run()

    shown = written_on(screen)
    assert "A trial of drug X" in shown
    assert "Review complete: 1 paper approved" in shown
    assert "not in the record of earlier runs" in shown
    keys = {button.key for button in screen.button}
    assert "new search" in keys and "back to the papers" not in keys


def test_no_earlier_run_is_offered_when_none_has_an_approval():
    screen = a_screen_with_no_run()

    assert "Approved papers of an earlier run" not in written_on(screen)


def test_the_outcome_of_a_run_the_log_does_not_hold_is_said_to_be_missing():
    from streamlit.testing.v1 import AppTest

    screen = AppTest.from_file(str(REPO_ROOT / "research_agent" / "ui" / "app.py"))
    screen.session_state["page"] = "results"
    screen.session_state["results of run"] = "nowhere"
    screen = screen.run()

    assert "Nothing in the run log belongs to run nowhere" in screen.warning[0].value
    assert {button.key for button in screen.sidebar.button} == {"new search"}
    assert {button.key for button in screen.main.button} == {
        "step Search", "step Assessment", "run record"}


def test_the_decisions_written_onto_the_run_are_the_ones_recorded():
    """Called directly as well, because the screen is not the only thing that
    will ask for them."""
    screened = [judged("first"), judged("second")]
    answers = {"decisions": {which_paper(judged("second")): ("rejected",
                                                             "it is a review")}}

    decided = decided_so_far(answers, screened)

    assert [item.review for item in decided] == ["waiting", "rejected"]
    assert decided[1].review_note == "it is a review"


def test_a_paper_with_nothing_to_cite_says_so_rather_than_looking_complete():
    """A citation with empty pieces reads as though the record were poor
    rather than as though the fields were missing, on the review page's
    block and on the results page's card alike."""
    bare = PaperRecord(title="A trial", text_source="metadata",
                       origins=[SearchOrigin(source="pubmed", query="q")])
    block = _judgement_lines(judged().model_copy(update={"paper": bare}))
    assert "**Author:** author not given (year not given)" in block
    assert "**Journal:** *not given*" in block
    assert "**Identifier:** none" in block
    assert first_author_of([]) == "author not given"
    assert first_author_of(["Smith J"]) == "Smith J"
    assert first_author_of(["Smith J", "Jones A"]) == "Smith J et al."


def test_the_identifier_says_what_kind_it_is():
    """Asked on 16 September to find a paper's identifier, the participant could
    not tell which part of the line was one (UX-05).
    """
    # The kind names where to look it up; the DOI comes first because it names
    # one paper and nothing else; the address is the page a button opens, and
    # Semantic Scholar has none.
    assert named_identifier("10.99999/invented.1", "12345678", "")[0] == "DOI"
    assert named_identifier("", "12345678", "")[0] == "PubMed"
    assert named_identifier("", "", "abc123") == ("Semantic Scholar", "abc123", "")
    assert named_identifier("", "", "") == ("", "", "")


def test_the_doi_and_the_pubmed_identifier_are_links_and_nothing_else_is():
    """The DOI links to the resolver, the PubMed identifier to its page, and
    a record with neither gets no link (UX-05).
    """
    # The second review of 17 September chose the identifier as a link, so
    # the reviewer can open the paper from the line that names it. The address
    # is percent-encoded and the text is not, because a DOI may hold
    # parentheses, which would end a Markdown link early.
    assert named_identifier("10.1016/S0140-6736(20)30183-5", "", "") == (
        "DOI",
        ("[10.1016/S0140-6736(20)30183-5]"
         "(https://doi.org/10.1016/S0140-6736%2820%2930183-5)"),
        "https://doi.org/10.1016/S0140-6736%2820%2930183-5")
    assert named_identifier("", "12345678", "") == (
        "PubMed", "[12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)",
        "https://pubmed.ncbi.nlm.nih.gov/12345678/")
    assert "](" not in named_identifier("", "", "abc123")[1]
    with_doi = a_paper().model_copy(update={"doi": "10.1016/S0140-6736(20)30183-5"})
    assert ("**DOI:** [10.1016/S0140-6736(20)30183-5]"
            "(https://doi.org/10.1016/S0140-6736%2820%2930183-5)") in \
        _judgement_lines(judged().model_copy(update={"paper": with_doi}))


def test_each_paper_is_a_block_of_labelled_lines_in_the_owners_order():
    """A paper's facts are labelled lines under the title, the labels bold and
    the journal in italics, in the order the review of the screen on 17
    September decided (UX-05).
    """
    # The order: author, journal, identifier, publication status ("not
    # checked" when the record carries none, which used to be left out),
    # relevance as "rated n of 10; the agent would ...", reason, summary, and
    # the extract last as the sentence supporting the summary. The decision is
    # the agent's own field, not read off the rating.
    paper = a_paper().model_copy(update={
        "authors": ["Smith J", "Jones A", "Brown K"], "year": "2024",
        "journal": "Journal of Trials", "pmid": "12345678"})
    excluded_screening = judged().screening.model_copy(update={
        "decision": "exclude", "relevance": 9})
    block = _judgement_lines(judged().model_copy(update={
        "paper": paper, "screening": excluded_screening}))

    assert block == [
        "A trial of drug X",
        "**Author:** Smith J et al. (2024)",
        "**Journal:** *Journal of Trials*",
        "**PubMed:** [12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)",
        "**Publication status:** not checked",
        "**Relevance:** rated 9 of 10; the agent would exclude it",
        "**Reason:** it reports an outcome for drug X",
        "**Summary:** a trial of drug X in condition Y",
        "**Extract from the paper (found in the paper):**",
        "> Forty patients were given drug X",
    ]
    one_author = paper.model_copy(update={
        "authors": ["Smith J"], "publication_status": "Retracted after publication"})
    block = _judgement_lines(judged().model_copy(update={"paper": one_author}))
    assert "**Author:** Smith J (2024)" in block
    assert "**Publication status:** Retracted after publication" in block


def test_the_headline_says_which_run_the_reviewer_has():
    short = a_state(pending_searches=[Subgoal(purpose="find harms",
                                              source="pubmed",
                                              query="drug X harms")])

    assert headline(a_report(), True) == "Search completed."
    assert "did not search everything it planned" in headline(a_report(short), True)


def test_a_flag_with_no_note_is_refused_on_the_spot():
    """The refusal belongs where the reviewer is, not at the end."""
    # Recorded without its note, the flag failed when they pressed store, by
    # which time they had moved on and the paper was off the screen.
    screen = a_screen()

    screen.button(key="flagged 0").click().run()

    assert "has to say why" in screen.error[0].value
    assert screen.session_state["decisions"] == {}, "nothing was written down"


def test_a_refused_flag_leaves_the_earlier_answer_alone():
    """A reviewer who approves a paper and then mis-flags it should not lose
    the answer they made properly."""
    screen = a_screen()
    screen.button(key="approved 0").click().run()

    screen.button(key="flagged 0").click().run()

    assert screen.session_state["decisions"][
        which_paper(judged())] == ("approved", "")


def test_a_flag_with_a_note_is_recorded():
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?").run()

    screen.button(key="flagged 0").click().run()

    assert not screen.error
    assert screen.session_state["decisions"][
        which_paper(judged())] == ("flagged", "is it a trial?")


def test_the_screen_shows_the_answer_the_reviewer_already_gave():
    """Without it a reviewer working down a long list cannot tell which papers
    they have dealt with."""
    screen = a_screen()

    screen.button(key="approved 0").click().run()

    assert "**You said:** :green[approved]" in written_on(screen), \
        "in bold, so the current answer is found among lines of one weight"


def test_the_answer_is_a_coloured_word_and_the_word_is_still_there():
    """The second review of 17 September chose the current decision as a
    coloured word, green, amber or red (UX-05).
    """
    # The word carries the meaning and the colour goes with it, never alone
    # (UX-01): each is Streamlit's coloured text around the answer itself,
    # drawn as a line of the paper and not as one of the page's coloured boxes,
    # which mean that something happened to the run. Amber is Streamlit's
    # "yellow", chosen in the review of the screen on 17 September; its light text,
    # #926c05, clears 4.5 to 1 on white where the orange used before, #e2660c,
    # did not.
    assert the_answer_in_its_colour("approved") == ":green[approved]"
    assert the_answer_in_its_colour("flagged") == ":yellow[flagged]"
    assert the_answer_in_its_colour("rejected") == ":red[rejected]"

    screen = a_screen()
    screen.text_input(key="note 0").set_value("it is a review")
    screen.button(key="rejected 0").click().run()

    assert "**You said:** :red[rejected], because it is a review" in \
        [item.value for item in screen.main.markdown]
    boxes = list(screen.success) + list(screen.warning) + list(screen.error)
    assert not any("You said" in box.value for box in boxes)


def test_the_screen_counts_the_decided_and_the_waiting_papers():
    """In fifty papers the participant could not tell, on 16 September, which had
    been dealt with (UX-03, UX-05).
    """
    # The count sits in the side panel and is written after every button has
    # been read, so it includes the answer just given.
    two = a_state(screened=[judged("A trial of drug X"), judged("A cohort of drug X")])
    screen = a_screen(two)

    assert "0 of 2 papers decided, 2 waiting:" in written_on(screen)

    screen.button(key="approved 1").click().run()

    assert "1 of 2 papers decided, 1 waiting:" in written_on(screen)


def test_the_count_says_how_the_decided_papers_split():
    """Decided against waiting says how far the review has got; the second look of
    17 September chose the split as well, approved, flagged and rejected,
    which is what the store and send-back buttons under the line act on.
    """
    # Counted from the session's answers, so a changed answer moves between the
    # counts rather than being counted twice.
    three = a_state(screened=[judged("first"), judged("second"), judged("third")])
    answers = {"decisions": {which_paper(judged("first")): ("approved", ""),
                             which_paper(judged("second")): ("rejected", "a review")}}

    assert how_many_decided(answers, three.screened) == (
        "2 of 3 papers decided, 1 waiting: 1 approved, 0 flagged, 1 rejected.")

    answers["decisions"][which_paper(judged("first"))] = ("flagged", "not sure")
    assert how_many_decided(answers, three.screened) == (
        "2 of 3 papers decided, 1 waiting: 0 approved, 1 flagged, 1 rejected.")


def test_the_count_and_the_two_buttons_are_in_the_side_panel_and_nowhere_else():
    """The session of 16 September found the store and send-back buttons at the
    bottom of fifty papers, and a fixed panel holding the counts and
    the buttons was decided on (UX-05, point 6 of the order chosen).
    """
    # The panel stays in view while the papers scroll, so the count is there
    # once and the copy that used to sit above the papers is gone.
    two = a_state(screened=[judged("A trial of drug X"), judged("A cohort of drug X")])
    screen = a_screen(two)
    screen.button(key="flagged 0").click().run()
    screen = screen.text_input(key="note 0").set_value("not sure it is a trial")\
        .run()
    screen.button(key="flagged 0").click().run()

    in_the_panel = [item.value for item in screen.sidebar.markdown]
    assert ("1 of 2 papers decided, 1 waiting: 0 approved, 1 flagged, "
            "0 rejected.") in in_the_panel
    assert "Your decisions" in [item.value for item in screen.sidebar.subheader]
    assert {button.key for button in screen.sidebar.button} == {"store", "send back"}

    on_the_page = [item.value for item in screen.main.markdown]
    assert not any("papers decided" in line for line in on_the_page)
    # The record's line is the page's own one button besides the cards'
    # and the progress line's, at the foot (the second session of 19
    # September looked for it here and found it on Result alone).
    assert {button.key for button in screen.main.button} == {
        "step Search", "step Result",
        "approved 0", "flagged 0", "rejected 0",
        "approved 1", "flagged 1", "rejected 1", "run record"}


def test_the_record_is_reached_from_the_foot_of_the_review_page_and_back_returns():
    """The record's line is the last thing on the review page with papers too,
    and Back from the record returns to the papers with the decisions as they
    were.
    """
    # The line "Record of this run, for checking what was done" was on the
    # form, on the results page and on the review page with nothing to
    # review, and not on the review page with papers, where the second session
    # of 19 September looked for it (B11).
    screen = a_screen()
    screen.button(key="approved 0").click().run()

    assert [type(element).__name__ for element in screen.main][-2:] == [
        "Markdown", "Button"], "the notice, then the record's line, last"
    assert screen.main.button[-1].key == "run record"
    screen = screen.button(key="run record").click().run()
    assert "Run record" in [item.value for item in screen.subheader]

    screen = screen.button(key="back from the record").click().run()

    assert "Run record" not in [item.value for item in screen.subheader]
    assert "**You said:** :green[approved]" in written_on(screen)
    assert "store" in {button.key for button in screen.sidebar.button}


def test_the_screen_says_what_the_three_answers_do_once_in_the_panel():
    """The participant could not tell what "flagged" meant or how it differed from
    "rejected", and the page said neither (UX-02).
    """
    # Drawn once, in the side panel since the second review of 17 September: the
    # panel is in view at every paper, where a line above the first paper was
    # not. Three headed entries, Approve, Flag and Reject, in the words given
    # on the evening of 17 September exactly, no Send back entry: the wording
    # was chosen with the code's reading known. Each is held short; whether the
    # buttons are in view on a 1179 by 731 window is seen at the screen.
    screen = a_screen(a_state(screened=[judged("A trial of drug X"),
                                        judged("A cohort of drug X")]))

    in_the_panel = [item.value for item in screen.sidebar.caption]
    on_the_page = [item.value for item in screen.main.caption]
    assert WHAT_THE_ANSWERS_DO == (
        "**Approve:** Use when the paper is relevant and should be kept.",
        ("**Flag:** Use when you are unsure and want the system to provide a "
         "second judgement. Add a comment to guide the system's new assessment."),
        ("**Reject:** Use when the paper does not match your query and should be "
         "reassessed by the system. Add a comment explaining why it should be "
         "excluded."))
    # Each drawn in italics, as decided on the morning of 18 September,
    # with the sentence itself untouched between the underscores.
    for entry in WHAT_THE_ANSWERS_DO:
        assert len(entry) <= 150, entry
        assert in_the_panel.count(f"_{entry}_") == 1, "once, in the panel"
        assert entry not in on_the_page
    assert not any("Send back" in entry for entry in in_the_panel)
    # Above the panel's buttons, not under them: the panel is read top down.
    panel = [(type(element).__name__, getattr(element, "value", ""))
             for element in screen.sidebar]
    at_the_last_entry = panel.index(("Caption", f"_{WHAT_THE_ANSWERS_DO[-1]}_"))
    assert at_the_last_entry < [kind for kind, _ in panel].index("Button")


def test_the_store_button_says_what_storing_now_leaves_out():
    """The session of 16 September asked whether papers could be stored with some
    undecided, and the page did not say (UX-02).
    """
    # Storage leaves a waiting paper waiting, neither stored nor rejected, and
    # the line beside the button says so, in the words chosen in the review
    # of the screen on 17 September, with the count as it stands after the last
    # press.
    two = a_state(screened=[judged("A trial of drug X"), judged("A cohort of drug X")])
    screen = a_screen(two)

    assert ("2 papers currently have no decision. Proceeding now will exclude "
            "them from your approved list for consultation. Consider reviewing "
            "them first.") in written_on(screen)

    screen.button(key="approved 0").click().run()
    assert ("1 paper currently has no decision. Proceeding now will exclude it "
            "from your approved list for consultation. Consider reviewing it "
            "first.") in written_on(screen)

    screen.button(key="approved 1").click().run()
    assert "Every paper has a decision." in written_on(screen)


def test_loading_another_run_does_not_carry_the_answers_over():
    """The answers belong to the papers they were made about."""
    # Kept by position, the first paper of a second run arrived already approved
    # by somebody who had never seen it.
    screen = a_screen()
    screen.button(key="approved 0").click().run()

    screen.session_state["run"] = (a_state(screened=[judged("a different paper")]),
                                   a_report())
    screen.run()

    assert screen.session_state["decisions"] == {}
    assert "You said:" not in written_on(screen)


def test_another_run_of_the_same_papers_does_not_inherit_the_answers(monkeypatch):
    """Paper keys alone do not name an execution (R-20)."""
    # A later run of the same question can return the same papers with new
    # judgements. The answers were told apart by the list of papers they were
    # made about, so a second run of the same papers kept the first run's
    # answers. In a controlled check, the second run inherited the earlier
    # approval and note, and pressing store passed them to storage with a
    # judgement that had not been approved in that run. The run's identifier
    # is part of the check now.
    from research_agent.processing import storage

    given = []

    def remember(screened, store=None):
        given.append(screened)
        return 0

    monkeypatch.setattr(storage, "save_approved", remember)
    screen = a_screen(a_state(run_id="first", screened=[judged()]))
    screen.text_input(key="note 0").set_value("Approval from the first run")
    screen.button(key="approved 0").click().run()
    assert screen.session_state["decisions"], "the first run's answer was made"

    screen.session_state["run"] = (a_state(run_id="second", screened=[judged()]),
                                   a_report())
    screen.run()
    screen.button(key="store").click().run()

    assert screen.session_state["decisions"] == {}
    assert "You said:" not in written_on(screen)
    assert [item.review for item in given[0]] == ["waiting"]
    assert given[0][0].review_note == ""


def test_a_complete_search_with_unknown_failures_is_not_called_incomplete():
    """A reviewer sent to look for a missing search that does not exist is a
    reviewer who stops believing the warning."""
    screen = a_screen(report=a_report(failures=None))

    assert screen.warning[0].value == (
        "Search completed. Whether anything failed quietly was not recorded, "
        "so this run cannot say it was clean.")


def test_a_complete_search_with_a_quiet_failure_says_which_it_was():
    from research_agent import graph
    from research_agent.processing.audit_log import failures_during, failures_so_far

    before = failures_so_far()
    graph.buffer_failures.append("the buffer could not be written")
    screen = a_screen(report=a_report(failures=failures_during(before)))

    assert screen.warning[0].value == (
        "Search completed, but something failed without stopping the run. "
        "What it was is under 'Run details'.")


def test_a_decision_that_could_not_be_logged_warns_the_reviewer_at_once(
        tmp_path, monkeypatch):
    """The coverage report they were shown was built before they started."""
    # So it cannot say a decision went unlogged, and a reviewer not told carries
    # on believing the review is being written down (FR-15).
    from research_agent.processing import run_log

    # A folder where the file should be, so the write fails while the reviewer
    # is at the screen rather than before or after.
    in_the_way = tmp_path / "run_log.jsonl"
    in_the_way.mkdir()
    monkeypatch.setattr(run_log, "DEFAULT_LOG", in_the_way)
    screen = a_screen()

    screen.button(key="approved 0").click().run()

    assert any("not written to the run log" in warning.value
               for warning in screen.warning)
    assert screen.session_state["decisions"], "the decision itself is kept"


def a_logged_run(identifier="abc123", question="Does drug X treat condition Y?"):
    """One run written into the log the screen reads."""
    # Written through the log's own functions rather than by hand, so what the
    # screen shows is what a run would actually have left behind. The log is
    # redirected to a temporary file for every test by `tests/conftest.py`.
    from research_agent.processing import run_log

    run_log.note_the_request(
        ResearchRequest(question=question, inclusion_criteria="trials",
                        exclusion_criteria="reviews"), identifier)
    return identifier


def on_the_record_page(screen):
    """The record of runs, opened as the line at the foot of the form, of the
    results page or, since 19 September, of the review page opens it.
    """
    # Set the way the line does rather than pressed, so the tests that read the
    # record page do not depend on where the line sits.
    screen.session_state["page"] = "record"
    return screen.run()


def test_the_screen_shows_the_lines_the_run_wrote():
    """The coverage report says what state a run finished in."""
    # The log is the order things happened in and the time each took, and until
    # now nothing showed it (FR-15). It is on the record's own page, opened
    # from the panel, with the run on screen offered first.
    a_logged_run()
    state = a_state(screened=[judged()], run_id="abc123")

    screen = on_the_record_page(a_screen(state))

    shown = written_on(screen)
    assert "Run record" in shown
    assert "run started" in shown
    assert "Does drug X treat condition Y?" in shown
    assert screen.selectbox(key="which run to show").value == "abc123"


def test_a_run_with_no_lines_in_the_log_says_the_record_is_missing():
    """A run whose lines never reached the log is a hole in the record."""
    # The person at the screen is the only one who will notice: the coverage
    # report can say a line could not be written, not that none was.
    a_logged_run("an earlier run")
    state = a_state(screened=[judged()], run_id="the run on screen")

    screen = a_screen(state)

    assert any("Nothing in the run log belongs to the run on screen"
               in warning.value for warning in screen.warning)


def test_an_earlier_run_can_be_read_with_no_run_loaded():
    """The log outlives the session that wrote it."""
    # Somebody opening the screen the morning after a run has no run loaded and
    # is exactly the person who wants to read what it did.
    a_logged_run("yesterday's run")
    screen = a_screen_with_no_run()
    assert "No run has been loaded yet" in screen.info[0].value
    # The line at the very foot of the form's page, after the notice, and
    # nothing in the panel: a "Run record" button in the panel was found
    # confusing beside the launcher's "Start the run" on 17 September.
    assert [type(element).__name__ for element in screen.main][-2:] == [
        "Markdown", "Button"]
    assert screen.main.button[-1].key == "run record"
    assert not list(screen.sidebar.button)

    screen = screen.button(key="run record").click().run()

    assert "yesterday's run" in written_on(screen) + "".join(
        option for box in screen.selectbox for option in box.options)


def test_the_records_line_is_drawn_as_a_link_on_both_pages(monkeypatch):
    """Finding g of 18 September: the line "Record of this run, for checking what
    was done" did not read as a link.
    """
    # It stays a button, since a link would reload the page and lose the run in
    # the session, and the page's style rule draws it blue and underlined, on
    # the form's page and on the results page alike. The blue is the
    # appearance's: no one blue clears 4.5 to 1 on both pages, so the script
    # asks Streamlit which appearance is in use, and takes the light blue when
    # it is not told.
    from research_agent.ui import app
    from tests.unit.test_theme import DARK_PAGE, LIGHT_PAGE, contrast

    rule = ".st-key-run-record button { color: #0068c9; text-decoration: underline; }"
    screen = a_screen_with_no_run()
    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    assert rule in style
    screen = a_reviewed_and_stored_screen()
    style = [item.value for item in screen.markdown if item.value.startswith("<style>")][0]
    assert rule in style
    assert screen.main.button[-1].key == "run record"

    assert contrast(app._LINK_BLUE_ON_THE_LIGHT_PAGE, LIGHT_PAGE) >= 4.5
    assert contrast(app._LINK_BLUE_ON_THE_DARK_PAGE, DARK_PAGE) >= 4.5

    class DarkAppearance:
        class theme:
            type = "dark"

    class LightAppearance:
        class theme:
            type = "light"

    class UnknownAppearance:
        class theme:
            type = None

    class Streamlit:
        context = DarkAppearance
    assert app._link_blue(Streamlit) == "#3d9df3"
    Streamlit.context = LightAppearance
    assert app._link_blue(Streamlit) == "#0068c9"
    Streamlit.context = UnknownAppearance
    assert app._link_blue(Streamlit) == "#0068c9"


def test_an_empty_log_says_so_rather_than_showing_nothing():
    """A page with nothing on it reads as a page that is broken."""
    screen = on_the_record_page(a_screen())

    assert "The run log is empty." in written_on(screen)


def test_the_run_record_is_a_page_of_its_own_off_the_review_page():
    """The run record is a page of its own, reached from one line at the foot
    of each page and from no panel, and Back returns to the page it was
    opened from (UX-05).
    """
    # The participant could not tell what "What the runs on this computer
    # did" was for, and found it in the way of the papers; as a closed section
    # on the review page it was still technical and in the way (the fourth
    # reading, 17 September), and as a "Run record" button in the panel it
    # confused beside "Start the run" (the same evening). The review page's
    # line dates from the second session of 19 September, which looked for it
    # there. The page carries the run selector, the lines and the two files
    # named in its caption; the missing-record warning stays on the review
    # page, where it is seen.
    a_logged_run("an earlier run")
    state = a_state(screened=[judged()], run_id="the run on screen")
    screen = a_screen(state)

    assert not [section for section in screen.expander
                if section.label.startswith("The record of every run")]
    assert not list(screen.selectbox)
    assert [button.key for button in screen.main.button
            if button.key == "run record"] == ["run record"]
    assert "run record" not in {button.key for button in screen.sidebar.button}
    assert any("Nothing in the run log belongs to the run on screen"
               in warning.value for warning in screen.warning), \
        "the warning is on the review page"

    screen = on_the_record_page(screen)
    assert "Run record" in [item.value for item in screen.subheader]
    assert screen.selectbox(key="which run to show").value == "an earlier run"
    assert any("approved_records.sqlite3" in item.value and "run_log.jsonl" in item.value
               for item in screen.caption)
    assert not list(screen.main.button), "the record page has no review controls"
    assert {button.key for button in screen.sidebar.button} == {"back from the record"}

    screen = screen.button(key="back from the record").click().run()
    assert "A trial of drug X" in written_on(screen)
    assert "Run record" not in [item.value for item in screen.subheader]


def test_the_three_answers_are_side_by_side():
    """They are one choice, so they are drawn as one thing (UX-05)."""
    # What this checks is the arrangement. Whether a reviewer is faster is
    # UX-03's measurement and how the screen is to use is UX-04's; neither is
    # claimed here.
    screen = a_screen()

    answers = [column for column in screen.columns
               if any(getattr(child, "key", "") in
                      ("approved 0", "flagged 0", "rejected 0")
                      for child in column.children.values())]
    assert len(answers) == 3, "the three answers are not in three columns"
    assert {column.weight for column in answers} == {1 / 3}, "of equal width"


def test_each_paper_is_a_bordered_card_one_per_row():
    """Each paper is one bordered card, one per row, with the title, the
    badges, the grey line, the text and the decision buttons at its foot
    (UX-05).
    """
    # The review of the screen on 17 September chose cards
    # (`10-card-layout-example.png` and `11-results-page-mockup.png`); two
    # columns were built, and the reading of them that evening
    # (`12-review-cards-two-columns-as-built.png`) chose one card per row
    # on every window. A line between papers used to do what the border does
    # now, so there is no divider any more.
    screen = a_screen(a_state(screened=[judged("the first"), judged("second"),
                                        judged("third")]))

    assert not [column for column in screen.columns if column.weight == 0.5], \
        "no grid of columns around the cards"
    cards = [block for block in screen.main.children.values()
             if type(block).__name__ == "Block"
             and block.proto.HasField("flex_container")
             and block.proto.flex_container.border]
    assert [card.markdown[0].value for card in cards] == [
        "**1. the first** :gray-badge[rated 8 of 10]",
        "**2. second** :gray-badge[rated 8 of 10]",
        "**3. third** :gray-badge[rated 8 of 10]"], \
        "one card per paper, in order, numbered from 1 (the finding of 18 September)"
    for number, card in enumerate(cards):
        assert {button.key for button in card.button} == {
            f"approved {number}", f"flagged {number}", f"rejected {number}"}, \
            "the paper's own buttons are inside its card"
        assert [box.key for box in card.text_input] == [f"note {number}"]
    assert not list(screen.main.divider)

    screen.button(key="approved 1").click().run()
    assert "**2. second** :gray-badge[rated 8 of 10] :green-badge[approved]" in \
        written_on(screen), "the answer becomes a badge beside the title"


def test_the_cost_caption_names_the_model_and_the_allowance_read_for_it(monkeypatch):
    """The caption said "twenty a day" whatever the model, the 3.6 Flash
    figure, over a run on the Lite model whose allowance is 500 (TC-01).
    Asked for on 20 September: true for the model the settings name."""
    from research_agent import llm
    from research_agent.ui.app import what_a_run_costs

    lite = what_a_run_costs("gemini-3.5-flash-lite")
    assert lite.startswith("Starting a run makes model requests as it plans and "
                           "screens, on gemini-3.5-flash-lite; a run makes at most 18.")
    assert "500 a day for this model (read on 18 September 2026)" in lite
    assert "twenty a day" in what_a_run_costs("gemini-3.6-flash")
    unread = what_a_run_costs("gemini-3.7-flash")
    assert "on gemini-3.7-flash; a run makes at most 18." in unread
    assert "has not been read" in unread and "a day for this model" not in unread

    # Through the drawn form as well, because the direct calls above would
    # not catch the form reading a different setting from the one the model
    # interface asks; the caption has to name the model the run will use.
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-3.5-flash-lite")
    screen = a_screen_with_no_run()
    captions = [item.value for item in screen.main.caption]
    assert what_a_run_costs("gemini-3.5-flash-lite") in captions


def test_nothing_is_offered_to_send_back_when_nothing_was_flagged():
    """A model request asking about no papers is one the run does not get
    back, and the free tier gives twenty a day."""
    screen = a_screen()

    assert not [button for button in screen.button if button.key == "send back"]


def test_sending_papers_back_puts_the_fresh_judgements_on_the_screen(monkeypatch):
    """Until this, a flag was a note nobody acted on: the paper stayed on the
    screen with the judgement the reviewer disagreed with (FR-19)."""
    from research_agent import graph

    sent = {}

    def judged_afresh(state, curator, answers_that_go=None):
        sent["went back"] = [item.paper.title for item in graph.sent_back(state)]
        fresh = judged("A trial of drug X").model_copy(update={
            "screening": state.screened[0].screening.model_copy(
                update={"decision": "exclude", "reason": "it is a review"})})
        return state.model_copy(update={"screened": [fresh]})

    monkeypatch.setattr(graph, "judge_again", judged_afresh)
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()

    assert sent["went back"] == ["A trial of drug X"]
    assert "it is a review" in written_on(screen), "the fresh judgement is drawn"


def test_a_paper_that_came_back_needs_a_fresh_decision(monkeypatch):
    """FR-19 asks for a fresh decision before a revised record is stored."""
    # An answer left in place would stand as the reviewer's decision on a
    # judgement they have not read.
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X")]}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    assert screen.session_state["decisions"], "the flag was recorded"

    screen.button(key="send back").click().run()

    assert screen.session_state["decisions"] == {}


def test_a_kept_note_is_labelled_as_kept(monkeypatch):
    """The reason typed stays in the box after a send-back, on purpose, and the
    participant could not tell on 16 September whether it was meant to (point
    10 of the review).
    """
    # The label says so once the box holds a note, and says what to type while
    # it is empty; the note itself is not cleared.
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X")]}))
    screen = a_screen()
    assert screen.text_input(key="note 0").label == "Why, if you flag or reject it"

    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()
    screen.button(key="send back").click().run()

    box = screen.text_input(key="note 0")
    assert box.value == "is it a trial?", "the note is kept, not cleared"
    assert box.label == "Your note, kept until you decide again"


def test_the_reviewer_is_told_the_papers_came_back(monkeypatch):
    """The page they were told on is replaced by the one holding the fresh
    judgements, so the message has to survive that drawing."""
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X")]}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()

    told = [shown.value for shown in screen.success
            if "1 papers were judged again" in shown.value]
    assert len(told) == 2, ("once at the top for the reviewer who starts from "
                            "the headline, once beside the send-back button "
                            "the reviewer has just pressed")


def test_a_revision_is_written_to_the_run_log(monkeypatch):
    """A reader would otherwise see a paper rejected and then approved with
    nothing in between saying the judgement had changed (FR-15)."""
    from research_agent import graph
    from research_agent.processing import run_log

    def now_excluded(state, curator, answers_that_go=None):
        fresh = state.screened[0].model_copy(update={
            "screening": state.screened[0].screening.model_copy(
                update={"decision": "exclude"}), "review": "waiting"})
        return state.model_copy(update={"screened": [fresh]})

    monkeypatch.setattr(graph, "judge_again", now_excluded)
    screen = a_screen(a_state(screened=[judged()], run_id="abc123"))
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()

    written = [line for line in run_log.read() if line["event"] == "judged again"]
    assert len(written) == 1
    assert written[0]["the agent had said"] == "include"
    assert written[0]["the agent now says"] == "exclude"
    assert written[0]["run"] == "abc123"


def test_a_revision_that_failed_still_counts_what_it_spent(monkeypatch):
    """A request that failed was still a request."""
    # Keeping the count from before it would let the same allowance be spent
    # twice: the reviewer presses the button again and the run hands out the
    # requests it had already used (ID-15).
    from research_agent import graph
    from research_agent.schemas import RunBudget

    given = []

    def fails_after_spending(state, curator, answers_that_go=None):
        given.append(state.budget.model_requests_used)
        spent = state.budget.with_one_model_request_used()
        return state.model_copy(update={
            "budget": spent, "needs_human_attention": True,
            "attention_reason": "the papers sent back could not be judged again"})

    monkeypatch.setattr(graph, "judge_again", fails_after_spending)
    screen = a_screen(a_state(screened=[judged()],
                              budget=RunBudget(model_requests_used=16)))
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()
    screen.button(key="send back").click().run()

    assert given == [16, 17], "the second attempt started from what the first spent"
    assert screen.session_state["run"][0].budget.model_requests_used == 18
    assert screen.session_state["decisions"], "the reviewer's answers are kept"


def test_a_revision_the_log_could_not_record_warns_the_reviewer(tmp_path,
                                                                monkeypatch):
    """The page the warning would have been written on is replaced by the one
    holding the fresh judgements, so it has to travel with the message that
    says the revision was made (FR-15)."""
    from research_agent import graph
    from research_agent.processing import run_log

    # A folder where the file should be, so the write fails while the reviewer
    # is at the screen rather than before or after.
    in_the_way = tmp_path / "run_log.jsonl"
    in_the_way.mkdir()
    monkeypatch.setattr(run_log, "DEFAULT_LOG", in_the_way)
    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(
                            update={"screened": [judged("A trial of drug X")]}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()

    assert any("not written to the run log" in warning.value
               for warning in screen.warning)
    assert any("judged again" in shown.value for shown in screen.success), \
        "and the revision itself is still reported"


def test_a_revision_that_could_not_be_paid_for_says_so(monkeypatch):
    """Spending the allowance is the limit working rather than a fault, and a
    reviewer told nothing would think the papers had been judged again."""
    from research_agent import graph

    monkeypatch.setattr(graph, "judge_again",
                        lambda state, curator, answers=None: state.model_copy(update={
                            "stopped_because": "this run has used the 18 model "
                                               "requests it was allowed"}))
    screen = a_screen()
    screen.text_input(key="note 0").set_value("is it a trial?")
    screen.button(key="flagged 0").click().run()

    screen.button(key="send back").click().run()

    assert any("18 model requests" in shown.value for shown in screen.error)


def test_the_question_and_both_criteria_make_the_request_a_run_is_given():
    """The criteria are the researcher's to write, not the agent's to work
    out: they are the part a review has to state and defend afterwards."""
    asked = what_was_asked_for("Does drug X treat condition Y?",
                               "trials reporting an outcome",
                               "reviews and opinion pieces")

    assert asked.question == "Does drug X treat condition Y?"
    assert asked.inclusion_criteria == "trials reporting an outcome"
    assert asked.exclusion_criteria == "reviews and opinion pieces"


@pytest.mark.parametrize("question, inclusion, exclusion", [
    ("", "trials", "reviews"),
    ("   ", "trials", "reviews"),
    ("Does drug X treat condition Y?", "", "reviews"),
    ("Does drug X treat condition Y?", "trials", ""),
])
def test_a_request_that_cannot_be_acted_on_is_refused_before_the_run(
        question, inclusion, exclusion):
    """A blank question reaches the Planning Agent as a goal, and a run with
    no criteria screens papers against nothing.
    """
    # Either produces a run that looks ordinary and finds the wrong thing, so it
    # is refused where the researcher is rather than inside the run.
    with pytest.raises(ReviewRefused):
        what_was_asked_for(question, inclusion, exclusion)


def test_the_refusal_says_what_was_wrong_in_the_record_s_own_words():
    """A researcher needs the reason, not where in the record it happened."""
    with pytest.raises(ReviewRefused) as refused:
        what_was_asked_for("", "trials", "reviews")

    assert "the research question cannot be empty" in str(refused.value)


def test_starting_a_run_from_the_screen_puts_it_in_front_of_the_reviewer(
        monkeypatch):
    """The screen is where the work is asked for as well as reviewed (FR-01)."""
    # The run itself is replaced here. A real one makes model requests, and the
    # free tier allows twenty a day, so a suite that started runs would spend
    # what a rehearsal needs (OB-07).
    from research_agent import graph

    asked = {}

    def instead_of_running(request, planner, curator, step_limit=None):
        asked["request"] = request
        asked["step_limit"] = step_limit
        return a_state(screened=[judged()])

    monkeypatch.setattr(graph, "run", instead_of_running)
    screen = a_screen_with_no_run()
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")

    screen.button(key="start").click().run()

    assert asked["request"].question == "Does drug X treat condition Y?"
    assert asked["step_limit"] == 12, "the record's own default, not a second copy"
    assert asked["request"].results_per_search == 50, "the record's own default"
    assert "A trial of drug X" in written_on(screen), "the papers are drawn"


def test_the_form_offers_the_number_of_results_asked_of_each_search(monkeypatch):
    """Asked for on 18 September: the number beside the step limit, for a quick
    run; the default stays the record's fifty, this project's own default and
    not the setting the recall was measured at (1,000, SC-01), so the form
    left alone does not change what an ordinary run is.
    """
    # A lower number reaches the request the run is given. The two names are
    # the plain ones given in the second session of 19 September; the second is
    # the nearest true one, since the number is the results each search asks
    # for and not the papers reviewed.
    from research_agent import graph

    asked = {}

    def instead_of_running(request, planner, curator, step_limit=None):
        asked["request"] = request
        return a_state(screened=[judged()])

    monkeypatch.setattr(graph, "run", instead_of_running)
    screen = a_screen_with_no_run()
    assert screen.number_input(key="steps").label == "Maximum search attempts"
    box = screen.number_input(key="results per search")
    assert box.label == "Maximum results per search"
    assert box.value == 50
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")
    screen.number_input(key="results per search").set_value(20)

    screen.button(key="start").click().run()

    assert asked["request"].results_per_search == 20


def test_a_run_is_not_started_from_a_request_the_record_refuses(monkeypatch):
    """Nothing is spent finding out that the question was blank."""
    from research_agent import graph

    def should_not_be_called(request, planner, curator, step_limit=None):
        raise AssertionError("a refused request must not reach the run")

    monkeypatch.setattr(graph, "run", should_not_be_called)
    screen = a_screen_with_no_run()
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")

    screen.button(key="start").click().run()

    assert any("question cannot be empty" in shown.value
               for shown in screen.error)


def test_a_run_refused_by_its_own_record_is_shown_and_logged_not_crashed(monkeypatch):
    """The run lets the refusal raise on purpose (DC-05); the screen is where
    it is shown and written down (FR-15).
    """
    # Before this, Streamlit showed its own stack trace and the log held the
    # request and its searches with no line saying why they stopped. The
    # refusal here is a real one, made by giving the record a run that asks for
    # a person without saying why, which is the rule the LangGraph trial showed
    # can be broken part way through.
    from pydantic import ValidationError

    from research_agent import graph
    from research_agent.processing import run_log

    try:
        ResearchState.model_validate({"request": a_state().request.model_dump(),
                                      "needs_human_attention": True})
    except ValidationError as real_refusal:
        refusal = real_refusal

    def instead_of_running(request, planner, curator, step_limit=None):
        raise refusal

    monkeypatch.setattr(graph, "run", instead_of_running)
    screen = a_screen_with_no_run()
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")

    screen.button(key="start").click().run()

    assert not screen.exception, "the screen handled it rather than crashing"
    assert any("failed its own checks" in shown.value and "has to say why" in shown.value
               for shown in screen.error)
    assert [box for box in screen.text_area if box.key == "question"], \
        "no run was loaded, so the form is still there"
    logged = [line for line in run_log.read()
              if line["event"] == "failure" and line["what"] == "running the request"]
    assert len(logged) == 1 and "has to say why" in logged[0]["why"]
    assert logged[0]["run"] == "not recorded", "the identifier did not survive the raise"


def test_the_run_is_watched_from_a_box_the_model_interface_writes_into(monkeypatch):
    """The run is drawn under Streamlit's status box, and the model interface
    is given the box's label to write into while the run lasts.
    """
    # Watching the live run's second attempt on 18 September, nobody could
    # tell a run waiting for the service from one that would never come back,
    # and the ask was for "the model is busy, asking again in three minutes"
    # in place of the spinner. The label is written while the run lasts, and
    # nothing after: a run that stops keeps the last line said on its box,
    # which is how the line is read here (the refusal path is the one that
    # leaves the page as it was); a run that ends another way draws the page
    # again. The panel's send-back runs the same way.
    from pydantic import ValidationError

    from research_agent import graph, llm

    try:
        ResearchState.model_validate({"request": a_state().request.model_dump(),
                                      "needs_human_attention": True})
    except ValidationError as real_refusal:
        refusal = real_refusal
    given_a_way_in = []

    def instead_of_running(request, planner, curator, step_limit=None):
        given_a_way_in.append(llm.say_while_waiting is not None)
        llm.say_while_waiting("The model is busy: asking again in 2 minutes "
                              "30 seconds (2 of 4 asks made).")
        raise refusal

    monkeypatch.setattr(graph, "run", instead_of_running)
    assert llm.say_while_waiting is None, "nobody listens before a run"
    screen = a_screen_with_no_run()
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")
    screen.button(key="start").click().run()

    assert given_a_way_in == [True]
    assert llm.say_while_waiting is None, "and nobody after"
    assert [box.label for box in screen.status] == [
        "The model is busy: asking again in 2 minutes 30 seconds (2 of 4 asks made)."]
    assert not screen.exception

    def instead_of_judging_again(decided, curator, answers_that_go=None):
        given_a_way_in.append(llm.say_while_waiting is not None)
        return a_state(screened=[judged()])

    monkeypatch.setattr(graph, "judge_again", instead_of_judging_again)
    screen = a_screen()
    screen.button(key="flagged 0").click().run()
    screen.text_input(key="note 0").set_value("not sure").run()
    screen.button(key="flagged 0").click().run()
    screen.button(key="send back").click().run()
    assert given_a_way_in == [True, True]
    assert llm.say_while_waiting is None


def test_the_form_is_not_drawn_while_a_run_is_being_reviewed():
    """Starting another run would replace the papers being worked through,
    and the answers already given would be cleared as another run's."""
    screen = a_screen()

    assert not [box for box in screen.text_area if box.key == "question"]


def test_an_empty_log_with_a_run_on_screen_is_still_the_missing_record():
    """A run that wrote nothing at all is the worst case of a run whose lines
    are missing, not a quieter one."""
    state = a_state(screened=[judged()], run_id="the run on screen")

    screen = a_screen(state)

    assert any("Nothing in the run log belongs to the run on screen"
               in warning.value for warning in screen.warning)
    assert "The run log is empty." in written_on(on_the_record_page(screen))
