"""The invented review states a person looks at in a browser (UX-CHK-01 to
UX-CHK-06).
"""
# What is checked here is that each state is a run the application would
# accept, that it produces the coverage headline it is meant to show, that the
# full set carries the features the layout checks need, and that the stand-in
# for the revision route behaves as the real one does about the note and the
# decision. Whether the page looks right is what the person is for.

import importlib.util

from research_agent.config import REPO_ROOT
from research_agent.processing.audit_log import report_on
from research_agent.schemas import ResearchRequest
from research_agent.ui.app import decide, headline

SCRIPT = REPO_ROOT / "scripts" / "show_synthetic_review_states.py"


def the_script():
    spec = importlib.util.spec_from_file_location("show_synthetic_review_states",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_request():
    return ResearchRequest(question="Does drug X help in condition Y?",
                           inclusion_criteria="trials", exclusion_criteria="reviews")


def test_each_state_is_a_run_the_application_accepts_with_its_own_headline():
    """A state the record refuses would never reach the screen, and a state
    with the wrong headline would have a person checking the wrong warning."""
    script = the_script()

    complete = script.complete_run(a_request())
    partial = script.partial_run(a_request())
    empty = script.empty_run(a_request())
    failed = script.failed_run(a_request())
    full = script.full_run(a_request())

    assert headline(report_on(complete, ((), (), ())), True) == "Search completed."
    assert "did not search everything" in headline(report_on(partial, ((), (), ())), True)
    assert headline(report_on(empty, ((), (), ())), False) == \
        "Search completed. No paper found."
    assert empty.screened == []
    assert headline(report_on(failed, ((), (), ())), False) == (
        "This run did not search everything it planned. The run failed before "
        "it retrieved anything. Details are under 'Run details'.")
    assert not failed.needs_human_attention, "a real run does not ask for one here"
    assert len(full.screened) == 50


def test_the_full_set_carries_the_features_the_layout_checks_need():
    """Each feature of the full set is a line of the acceptance baseline or
    of the coverage report.
    """
    # One long title, one long abstract, one paper with no abstract whose
    # extract is the prompt's placeholder, one retraction, one paper with no
    # DOI, one extract never checked and one not found.
    script = the_script()
    full = script.full_run(a_request())
    by_number = {judged.paper.pmid: judged for judged in full.screened}

    assert len(by_number["910000007"].paper.title) > 150
    assert len(by_number["910000008"].paper.abstract) > 2000
    assert by_number["910000009"].paper.text_source == "metadata"
    assert by_number["910000009"].extract_found is False
    assert by_number["910000010"].paper.publication_status == "Retracted after publication"
    assert by_number["910000012"].paper.doi == ""
    assert by_number["910000011"].extract_found is None
    assert by_number["910000013"].extract_found is False
    report = report_on(full, ((), (), ()))
    assert report.papers_beyond_the_allowance == 4
    assert report.papers_retrieved == 57
    assert report.status_checks_failed


def test_every_invented_paper_says_so_and_cannot_be_mistaken_for_a_recording():
    """The titles say invented and the identifiers are nine digits beginning
    with 91, the shape this project's recordings and fixtures never use."""
    script = the_script()

    for judged in script.full_run(a_request()).screened:
        assert judged.paper.title.startswith("Invented paper ")
        assert len(judged.paper.pmid) == 9
        assert judged.paper.pmid.startswith("91")
        assert judged.screening.decision in ("include", "exclude", "uncertain")


def test_the_revision_stand_in_keeps_the_note_and_clears_the_decision():
    """The real route leaves the note on the paper and brings it back waiting
    (FR-19); a stand-in that did otherwise would have a person walking a
    journey the application does not offer."""
    script = the_script()
    state = script.complete_run(a_request())
    first, second, third = state.screened
    state = state.model_copy(update={"screened": [
        decide(first, "approved"),
        decide(second, "flagged", "is this really a trial?"),
        third]})

    again = script.judge_again_without_a_model(state, curator=None)

    kept, revised, untouched = again.screened
    assert kept.review == "approved"
    assert revised.review == "waiting"
    assert revised.review_note == "is this really a trial?"
    assert revised.screening.decision != second.screening.decision
    assert "is this really a trial?" in revised.screening.reason
    # The old reason stays in front of the note, so the rating still has a
    # reason under it, as decided in the review of the screen on 17 September.
    assert revised.screening.reason.startswith(second.screening.reason)
    assert untouched == third
    assert again.budget.model_requests_used == state.budget.model_requests_used + 1


def test_nothing_sent_back_leaves_the_run_as_it_was():
    script = the_script()
    state = script.complete_run(a_request())

    assert script.judge_again_without_a_model(state, curator=None) is state


def test_the_results_page_reads_the_store_the_launcher_writes_to(monkeypatch, tmp_path):
    """The fault reproduced on 17 September: on the launcher, a paper approved and
    stored was listed as "no longer in the store".
    """
    # The launcher moved the store's write into its own folder and left the
    # name the results page reads by pointing at the real file, so the read-
    # back found nothing. The write and the read have to follow the same name,
    # whatever the launcher does with it (FR-14).
    from streamlit.testing.v1 import AppTest

    from research_agent import graph
    from research_agent.processing import run_log, storage

    script = the_script()
    script.CHOSEN_STATE = "complete"
    monkeypatch.setattr(graph, "run", script.run_without_searching)
    # Set through monkeypatch first, so what the launcher assigns is put
    # back when the test ends; the folder is not pytest's own, because the
    # fixture that keeps tests off the real store already points there and
    # the fault was that the two names differed.
    monkeypatch.setattr(storage, "DEFAULT_STORE", storage.DEFAULT_STORE)
    monkeypatch.setattr(storage, "save_approved", storage.save_approved)
    monkeypatch.setattr(run_log, "DEFAULT_LOG", run_log.DEFAULT_LOG)
    script.keep_this_session_apart(tmp_path / "session")

    screen = AppTest.from_file(str(script.APP)).run()
    screen.text_area(key="question").set_value("Does drug X help in condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")
    screen = screen.button(key="start").click().run()
    screen = screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()

    shown = "\n".join(item.value for item in screen.markdown)
    shown += "\n".join(item.value for item in screen.header)
    assert "Review complete: 1 paper approved" in shown
    assert "no longer in the store" not in shown
    assert (tmp_path / "session" / "approved_records.sqlite3").exists()


def the_launcher_on_the_real_screen(monkeypatch, tmp_path, state="full"):
    """The real screen with the launcher's stand-ins and its own folder, as
    the launcher shortcut runs it, through AppTest."""
    from streamlit.testing.v1 import AppTest

    from research_agent import graph
    from research_agent.processing import run_log, storage

    script = the_script()
    script.CHOSEN_STATE = state
    monkeypatch.setattr(graph, "run", script.run_without_searching)
    monkeypatch.setattr(graph, "judge_again", script.judge_again_without_a_model)
    monkeypatch.setattr(storage, "DEFAULT_STORE", storage.DEFAULT_STORE)
    monkeypatch.setattr(run_log, "DEFAULT_LOG", run_log.DEFAULT_LOG)
    script.keep_this_session_apart(tmp_path / "session")
    return AppTest.from_file(str(script.APP)).run()


def a_run_started(screen):
    screen.text_area(key="question").set_value("Does drug X help in condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")
    return screen.button(key="start").click().run()


def headed(screen):
    return [item.value for item in screen.header]


def test_the_whole_flow_on_the_invented_papers(monkeypatch, tmp_path):
    """The whole flow on the launcher's full state, every page reached read by
    its heading and no exception left anywhere.
    """
    # The flow asked to be kept as a test on the evening of 17 September, as
    # the shortcut runs it: the form, Start the run, approve one paper,
    # Display Approved (the results page), Back, approve three more, one of
    # them the paper whose extract is never checked (the sequence pressed at
    # the screen that evening, `15-fault-store-refused-after-back.png`: the
    # store refused the whole batch for it, and the reviewer heard of it only
    # then), Display Approved again (the count updated, not doubled), Start a
    # new search, Start the run, approve, Display Approved (the results page
    # reached again).
    screen = the_launcher_on_the_real_screen(monkeypatch, tmp_path)
    assert screen.button(key="start"), "the form"

    screen = a_run_started(screen)
    assert screen.button(key="approved 0"), "the papers"
    screen = screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()
    assert headed(screen) == ["Review complete: 1 paper approved"]
    assert not list(screen.exception) and not list(screen.error)

    screen = screen.button(key="back to the papers").click().run()
    assert screen.button(key="approved 1"), "the papers again, decisions kept"
    screen = screen.button(key="approved 1").click().run()
    screen = screen.button(key="approved 2").click().run()
    # Paper 11 is the one the launcher leaves never checked; its position
    # in the ranked list is read from the run rather than assumed.
    run = screen.session_state["run"][0]
    never_checked = [number for number, judged in enumerate(run.screened)
                     if judged.extract_found is None]
    assert len(never_checked) == 1
    screen = screen.button(key=f"approved {never_checked[0]}").click().run()
    assert "was never looked for in the paper" in screen.error[0].value, \
        "refused at the paper, with the reason, not at the store"
    screen = screen.button(key="store").click().run()
    assert headed(screen) == ["Review complete: 3 papers approved"]
    assert not list(screen.exception) and not list(screen.error)

    screen = screen.button(key="new search").click().run()
    assert screen.button(key="start"), "the empty form"
    screen = a_run_started(screen)
    screen = screen.button(key="approved 0").click().run()
    screen = screen.button(key="store").click().run()
    assert headed(screen) == ["Review complete: 1 paper approved"]
    assert not list(screen.exception) and not list(screen.error)


def test_pressing_start_on_the_real_screen_shows_the_chosen_state(monkeypatch, tmp_path):
    """The launcher's whole idea: the stand-in is put on the module the screen
    calls through, so the real form leads to the invented papers with no
    service called.
    """
    # Run here with AppTest, as the browser would run it, with the log moved to
    # a file of this test's own.
    from streamlit.testing.v1 import AppTest

    from research_agent import graph
    from research_agent.processing import run_log

    script = the_script()
    script.CHOSEN_STATE = "partial"
    monkeypatch.setattr(graph, "run", script.run_without_searching)
    monkeypatch.setattr(run_log, "DEFAULT_LOG", tmp_path / "run_log.jsonl")

    screen = AppTest.from_file(str(script.APP)).run()
    screen.text_area(key="question").set_value("Does drug X help in condition Y?")
    screen.text_area(key="inclusion").set_value("trials")
    screen.text_area(key="exclusion").set_value("reviews")
    screen = screen.button(key="start").click().run()

    drawn = [item.value for item in screen.markdown]
    assert any(line.startswith("**1. Invented paper 1: a cohort study of drug X**")
               for line in drawn)
    assert any("did not search everything" in warning.value
               for warning in screen.warning)
    assert (tmp_path / "run_log.jsonl").exists()
