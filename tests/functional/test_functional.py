"""The functional tests FT-01 to FT-09, as docs/testing.md defines them."""
# Each one drives the application the way a person meets it: the screen's form
# starts a run, the real graph runs the real Planning and Curation Agents, the
# real PubMed source reads a reply, and the reviewer's buttons decide what is
# stored. The two edges the application does not own are replaced with the
# synthetic replies in synthetic_replies.py, and conftest.py says what else
# is replaced beside them and why, so what these show is that the system
# behaves the same way twice on replies this project wrote. Whether a live
# service or the model behaves that way is separate evidence, from the live
# run, and is recorded as such in docs/testing.md.
#
# The screen is run from its own file with Streamlit's AppTest, as the unit
# tests of the screen do, because Streamlit runs a script and a function lifted
# out of its module loses what the module holds.

import sqlite3

import httpx
import pytest

from research_agent.config import REPO_ROOT
from research_agent.processing import run_log, storage
from research_agent.sources.pubmed import EUTILS
from tests.functional import synthetic_replies as replies


def a_screen_with_no_run():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(
        str(REPO_ROOT / "research_agent" / "ui" / "app.py")).run()


def a_run_started_from_the_form(steps=None):
    screen = a_screen_with_no_run()
    screen.text_area(key="question").set_value("Does drug X treat condition Y?")
    screen.text_area(key="inclusion").set_value(
        "studies of drug X in condition Y reporting an outcome")
    screen.text_area(key="exclusion").set_value("reviews and opinion pieces")
    if steps is not None:
        screen.number_input(key="steps").set_value(steps)
    return screen.button(key="start").click().run()


def the_state_on(screen):
    return screen.session_state["run"][0]


def written_on(screen):
    """The text of seven kinds of element, joined, for assertions that read for
    words.
    """
    # AppTest keeps each kind in its own list, so this says whether something
    # was drawn and not where on the page it sits; a test that needs the
    # position reads the element's own list, or the drawing code.
    drawn = [item.value for item in screen.markdown]
    drawn += [item.value for item in screen.subheader]
    drawn += [item.value for item in screen.header]
    drawn += [item.value for item in screen.caption]
    drawn += [item.value for item in screen.warning]
    drawn += [item.value for item in screen.success]
    drawn += [item.value for item in screen.error]
    return "\n".join(drawn)


def decide_on_the_screen(screen, number, choice, note=""):
    if note:
        screen.text_input(key=f"note {number}").set_value(note)
    return screen.button(key=f"{choice} {number}").click().run()


def rows_stored_in(store):
    with sqlite3.connect(store) as connection:
        return connection.execute(
            "SELECT identifier, pmid, title, extract, decision FROM approved_papers "
            "ORDER BY identifier").fetchall()


# FT-01 to FT-05: the run, as started from the form.

def test_ft01_a_question_comes_back_as_ranked_cited_summaries_with_a_coverage_warning(
        replayed_model, synthetic_pubmed):
    """FT-01 (FR-01, FR-16): the question and criteria are accepted, the papers
    come back ranked with a citation, a summary and an extract each, and a
    run that could not search everything says so.
    """
    # Two searches are planned; the second is refused by the service, so the
    # coverage is partial and the headline is a warning. That the headline is
    # drawn above the papers is read from the drawing code (main in
    # ui/app.py draws it before the loop over the papers); this test checks
    # that it is drawn as a warning, not where it sits.
    replayed_model(replies.PLAN_TRIAL_AND_HARMS, replies.MOVE_CONTINUE,
                   replies.MOVE_FINISH, replies.SCREENING_THREE_PAPERS)

    screen = a_run_started_from_the_form()

    state = the_state_on(screen)
    assert [judged.paper.pmid for judged in state.screened] == [
        "900000001", "900000003", "900000002"], "ranked by relevance, 9 then 6 then 2"
    assert [judged.screening.relevance for judged in state.screened] == [9, 6, 2]
    shown = written_on(screen)
    # Since 18 September the paper rated 2 is kept back from the page and
    # counted (the rule of 18 September: papers rated 5 or below are not shown); it
    # stays in the run, which the state above shows.
    assert "1 paper rated 5 or below, not shown." in shown
    review = replies.PAPERS["900000002"]
    assert f"{review['title']}**" not in shown
    for card, judged in enumerate(state.screened[:2], start=1):
        paper = replies.PAPERS[judged.paper.pmid]
        last_name, initials = paper["author"]
        assert (f"**{card}. {paper['title']}** "
                f":gray-badge[rated {judged.screening.relevance} of 10]") in shown, \
            "the card's number, then the title with the relevance badge"
        assert f"{last_name} {initials} ({paper['year']}) · *{paper['journal']}* · " \
            f"PubMed: [{judged.paper.pmid}]" \
            f"(https://pubmed.ncbi.nlm.nih.gov/{judged.paper.pmid}/)" in shown, \
            "cited by author, year, journal and a labelled identifier that links"
        assert f"**Summary:** {judged.screening.summary}" in shown
        assert (f"**Extract from the paper (found in the paper):**\n"
                f"> {judged.screening.extract}") in shown, \
            "the extract is drawn as a quotation under its label"
    assert any("This run did not search everything it planned" in warning.value
               for warning in screen.warning), "the headline is drawn as a warning"
    assert [outcome.result for outcome in state.results.outcomes] == ["finished", "failed"]


def test_ft02_a_search_that_returns_too_little_is_refined_rather_than_accepted(
        replayed_model, synthetic_pubmed):
    """FT-02 (FR-04): the first search finds nothing, the agent refines it, and
    the refined search is what the papers come from."""
    replayed_model(replies.PLAN_ONE_BROAD_SEARCH, replies.MOVE_REFINE,
                   replies.MOVE_FINISH, replies.SCREENING_THREE_PAPERS)

    screen = a_run_started_from_the_form()

    state = the_state_on(screen)
    assert [decision.move for decision in state.decisions] == ["refine", "finish"]
    assert [outcome.subgoal.query for outcome in state.results.outcomes] == [
        "drug X condition Y", "drug X randomised"]
    assert [outcome.papers_found for outcome in state.results.outcomes] == [0, 3]
    assert len(state.screened) == 3, "the papers are the refined search's"
    assert "Search completed." in [box.value for box in screen.success]


def test_ft03_a_run_that_could_not_fetch_everything_says_so_instead_of_appearing_complete(
        replayed_model, synthetic_pubmed):
    """FT-03 (FR-05): the search names three papers and PubMed returns two, so
    the search is partial, the run says which paper was missing, and the
    screen does not call the coverage complete."""
    replayed_model(replies.PLAN_TRIAL_AND_COHORT, replies.MOVE_FINISH,
                   replies.SCREENING_TRIAL_AND_COHORT)

    screen = a_run_started_from_the_form()

    state = the_state_on(screen)
    outcome = state.results.outcomes[0]
    assert outcome.result == "partial"
    assert "900000009" in outcome.reason, "the missing paper is named"
    assert len(state.screened) == 2, "the two that came back were screened"
    shown = written_on(screen)
    assert "This run did not search everything it planned" in shown
    assert "Search completed" not in shown


def test_ft04_a_service_that_never_answers_ends_the_run_with_a_reported_failure(
        replayed_model, synthetic_pubmed):
    """FT-04 (FR-18): every attempt at PubMed times out; the run ends with the
    search recorded as failed and the reviewer told, not with a stack trace."""
    synthetic_pubmed.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(
        side_effect=httpx.ConnectTimeout("the service never answered"))
    replayed_model(replies.PLAN_ONE_BROAD_SEARCH, replies.MOVE_FINISH)

    screen = a_run_started_from_the_form()

    assert not screen.exception, "the run ended, it did not crash"
    state = the_state_on(screen)
    assert state.results.outcomes[0].result == "failed"
    assert "never answered" in state.results.outcomes[0].reason
    assert state.results.ready is True, "the run still reached its end"
    assert state.screened == []
    assert screen.warning[0].value == (
        "This run did not search everything it planned. The run failed before "
        "it retrieved anything. Details are under 'Run details'.")


def test_ft05_a_plan_the_model_cannot_produce_stops_the_run_and_asks_for_a_person(
        replayed_model, synthetic_pubmed):
    """FT-05 (DC-06): the model's answer is refused as not fitting the format four
    times, the first attempt and the three repairs Figure 2 allows, and the run
    stops asking for a person rather than searching on a plan it never had.
    """
    # The stand-in returns the failure llm.ask_for produces after a badly
    # formatted answer, marked worth asking again; it does not exercise that
    # detection itself. What is exercised is the real agent's repair loop and
    # the graph's and screen's handling of the failure that comes out of it.
    model = replayed_model(*[replies.AN_ANSWER_THAT_DOES_NOT_FIT] * 4)

    screen = a_run_started_from_the_form()

    state = the_state_on(screen)
    assert state.needs_human_attention is True
    assert "could not be planned" in state.attention_reason
    assert len(model.questions_asked) == 4, "one attempt and three repairs, no more"
    assert state.budget.model_requests_used == 4, "every attempt was counted"
    assert state.results.outcomes == [], "nothing was searched"
    assert not synthetic_pubmed.calls, "PubMed was never asked"
    assert "The run failed before it retrieved anything" in screen.warning[0].value
    assert "The run needs a person: the searches could not be planned" \
        in written_on(screen), "the reason is under 'Run details'"


# FT-06 to FT-09: the reviewer, on the papers a run brought back.

def a_reviewed_run(replayed_model):
    replayed_model(replies.PLAN_TRIAL_AND_HARMS, replies.MOVE_CONTINUE,
                   replies.MOVE_FINISH, replies.SCREENING_THREE_PAPERS)
    return a_run_started_from_the_form()


def test_ft06_the_reviewer_decides_and_nothing_is_stored_without_a_decision(
        replayed_model, synthetic_pubmed, temporary_store):
    """FT-06 (FR-13, DC-08): approve one, reject one with a note, leave one
    undecided; only the approved paper is stored, and a flag without a note is
    refused where the reviewer is.
    """
    # Since 18 September the paper rated 2 is kept back from the page, so it is
    # the one left undecided, and the one rejected is the cohort rated 6.
    screen = a_reviewed_run(replayed_model)

    screen = decide_on_the_screen(screen, 0, "approved")
    screen = decide_on_the_screen(screen, 1, "flagged")
    assert any("has to say why" in shown.value for shown in screen.error), \
        "a flag with no note is refused on the spot"
    screen = decide_on_the_screen(screen, 1, "rejected", "harms, not the outcome")
    screen = screen.button(key="store").click().run()

    assert "Review complete: 1 paper approved" in written_on(screen)
    stored = rows_stored_in(temporary_store)
    assert [row[1] for row in stored] == ["900000001"], \
        "the approved paper, and neither the rejected one nor the undecided one"


def test_ft07_an_approved_record_is_stored_with_its_extract_and_identifier(
        replayed_model, synthetic_pubmed, temporary_store):
    """FT-07 (FR-14): the stored row carries the paper's identifier and the
    extract the reviewer approved, word for word."""
    screen = a_reviewed_run(replayed_model)
    screen = decide_on_the_screen(screen, 0, "approved")
    screen = screen.button(key="store").click().run()

    stored = rows_stored_in(temporary_store)
    assert len(stored) == 1
    identifier, pmid, title, extract, decision = stored[0]
    assert identifier == "900000001" and pmid == "900000001"
    assert title == "A randomised trial of drug X in condition Y"
    assert extract == "Symptoms improved in most of them"
    assert decision == "include"


def test_ft08_the_store_and_the_reviewer_decision_log_agree(
        replayed_model, synthetic_pubmed, temporary_store):
    """FT-08 (FR-14, SC-04): what the store holds is what the log says the
    reviewer approved, no more and no less."""
    screen = a_reviewed_run(replayed_model)
    screen = decide_on_the_screen(screen, 0, "approved")
    screen = decide_on_the_screen(screen, 1, "approved")
    # The third paper, rated 2, is kept back from the page since 18
    # September and stays waiting; the reconciliation below counts it as
    # what it is, a paper not approved.
    screen = screen.button(key="store").click().run()

    state = the_state_on(screen)
    approved_in_the_log = set()
    for line in run_log.read():
        if line["event"] != "reviewer decision":
            continue
        if line["decision"] == "approved" and line["run"] == state.run_id:
            approved_in_the_log.add(line["paper"])
    assert approved_in_the_log == {"900000001", "900000003"}
    assert storage.stored_identifiers(temporary_store) == approved_in_the_log
    from research_agent.ui.app import decided_so_far
    answers = {"decisions": screen.session_state["decisions"]}
    decided = decided_so_far(answers, state.screened)
    assert storage.what_does_not_reconcile(decided, temporary_store) == (set(), set())


def test_ft09_a_flagged_record_comes_back_revised_and_needs_a_new_decision(
        replayed_model, synthetic_pubmed, temporary_store):
    """FT-09 (FR-19): a flagged paper goes back to the Curation Agent, comes
    back with a fresh judgement and no decision on it, is not stored until the
    reviewer decides again, and is stored once they do."""
    replayed_model(replies.PLAN_TRIAL_AND_HARMS, replies.MOVE_CONTINUE,
                   replies.MOVE_FINISH, replies.SCREENING_THREE_PAPERS,
                   replies.SCREENING_AGAIN_THE_COHORT)
    screen = a_run_started_from_the_form()
    screen = decide_on_the_screen(screen, 0, "approved")
    screen = decide_on_the_screen(screen, 1, "flagged", "harms, not the outcome asked about")

    screen = screen.button(key="send back").click().run()

    state = the_state_on(screen)
    cohort = state.screened[1]
    assert cohort.paper.pmid == "900000003"
    assert cohort.screening.decision == "exclude", "the fresh judgement replaced the old"
    assert cohort.review == "waiting", "and it needs a new decision"
    assert cohort.review_note == "harms, not the outcome asked about", \
        "the reviewer's note stays until they decide again"
    assert "900000003" not in screen.session_state["decisions"], \
        "the flag was cleared from the reviewer's answers"
    assert "the agent would exclude it" in written_on(screen)
    revised = [line for line in run_log.read() if line["event"] == "judged again"]
    assert len(revised) == 1 and revised[0]["paper"] == "900000003"

    screen = screen.button(key="store").click().run()
    assert [row[1] for row in rows_stored_in(temporary_store)] == ["900000001"], \
        "the revised paper is not stored without a new decision"

    # Storing moves the screen to the outcome, so the reviewer goes back to
    # the papers to decide the revised one, and stores again from there.
    screen = screen.button(key="back to the papers").click().run()
    screen = decide_on_the_screen(screen, 1, "approved")
    screen = screen.button(key="store").click().run()
    assert [row[1] for row in rows_stored_in(temporary_store)] == ["900000001", "900000003"]
    assert "Review complete: 2 papers approved" in written_on(screen), \
        "the second store updates the run's set on the outcome page"


def test_the_fixtures_keep_their_conventions():
    """The conventions the fixture module sets for itself hold: nine-digit
    identifiers beginning with 9, a journal name that says invented, and an
    identifier used for a missing record that no paper carries.
    """
    # This checks shape, not provenance. Where the papers came from is recorded
    # in the module's docstring, and nothing here asks PubMed
    # what the identifiers belong to (R-12).
    for pmid, paper in replies.PAPERS.items():
        assert pmid.startswith("9") and len(pmid) == 9
        assert "Invented" in paper["journal"]
    assert "900000009" not in replies.PAPERS
    with pytest.raises(KeyError):
        replies._article_xml("900000009")
