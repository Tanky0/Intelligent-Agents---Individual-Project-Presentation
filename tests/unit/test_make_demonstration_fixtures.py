"""The replies the public offline demonstration replays, made and replayed
(R-12).
"""
# What is checked: that the script leaves a set of replies with a manifest
# recording a fixture origin for each, that every file passes the replay check
# and the export check, that nothing in them came from a service or a paper,
# and that a run in demonstration mode with nothing but those files answers
# the invented question through the real screen with no network and no model.

import importlib.util
import json
import shutil

import pytest

from research_agent.config import REPO_ROOT, settings
from research_agent.sources import saved_replies
from tests.functional import synthetic_replies

SCRIPT = REPO_ROOT / "scripts" / "make_demonstration_fixtures.py"


def the_script():
    spec = importlib.util.spec_from_file_location("make_demonstration_fixtures",
                                                  SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixtures_made(tmp_path, monkeypatch):
    """The script run once into a folder of this test's own."""
    # The script sets the recording folder, the mode and the stand-in model on
    # the modules it imports, so the settings it changes are put back for the
    # tests after this one by changing them through monkeypatch first.
    from research_agent.agents import asking
    from research_agent.processing import run_log
    from research_agent.sources import http, pubmed

    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(settings, "record_replies", True)
    monkeypatch.setattr(asking.llm, "ask_for", asking.llm.ask_for)
    monkeypatch.setattr(pubmed, "_limiter", pubmed._limiter)
    monkeypatch.setattr(http, "FIRST_WAIT_SECONDS", http.FIRST_WAIT_SECONDS)
    monkeypatch.setattr(saved_replies, "CAPTURED_DIR", tmp_path / "captured")
    monkeypatch.setattr(run_log, "DEFAULT_LOG", tmp_path / "fixture_run_log.jsonl")

    script = the_script()
    folder = tmp_path / "fixtures"
    folder.mkdir()
    state = script.run_once_and_record(folder)
    manifest = script.manifest_for(folder, script.what_it_stands_in_for)
    return script, folder, manifest, state


def test_the_run_leaves_a_reply_for_every_edge_it_crossed(fixtures_made):
    """Two PubMed replies (a search and a fetch) and three model answers (the
    plan, the decision to finish, the screening)."""
    _, folder, manifest, state = fixtures_made

    files = saved_replies.files_in(folder)
    model_answers = [path for path in files if "question" in saved_replies._read(path)]
    assert len(files) == 5
    assert len(model_answers) == 3
    assert len(state.screened) == 3
    assert set(manifest["files"]) == {path.name for path in files}


def test_every_fixture_passes_the_replay_and_export_checks(fixtures_made):
    """A fixture is only useful if the demonstration will replay it and the
    export will carry it, and both boundaries read the same origin record."""
    _, folder, manifest, _ = fixtures_made

    for path in saved_replies.files_in(folder):
        entry = manifest["files"][path.name]
        assert saved_replies.why_it_cannot_be_replayed(entry, path) == []
        assert saved_replies.why_it_cannot_be_exported(entry, path) == []


def test_no_fixture_carries_an_address_a_key_or_a_real_identifier(fixtures_made):
    """Nothing in the set came from a service, so nothing in it can name a
    person or a key; and every paper is one of the three invented ones."""
    _, folder, _, _ = fixtures_made

    for path in saved_replies.files_in(folder):
        text = path.read_text(encoding="utf-8")
        assert "@" not in text
        assert saved_replies.what_should_not_be_published(json.loads(text)) == []
        for identifier in synthetic_replies.PAPERS:
            assert identifier.startswith("9")


def test_the_demonstration_replays_the_fixtures_with_no_network_and_no_model(
        fixtures_made, monkeypatch):
    """The whole point: the files, copied to the approved folder with their
    manifest, are what a run in demonstration mode answers from.
    """
    # The real screen starts the real graph; the model is not called (a call
    # would fail the test) and no request leaves the machine.
    from streamlit.testing.v1 import AppTest

    script, folder, manifest, _ = fixtures_made

    approved = saved_replies.APPROVED_DIR
    shutil.copytree(folder, approved)
    saved_replies.MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(settings, "demo_mode", True)

    def a_model_that_must_not_be_asked(question, schema, budget=None):
        raise AssertionError("the demonstration asked the model")

    from research_agent import llm
    monkeypatch.setattr(llm, "_connection", a_model_that_must_not_be_asked)

    screen = AppTest.from_file(str(REPO_ROOT / "research_agent" / "ui" / "app.py")).run()
    screen.text_area(key="question").set_value(script.QUESTION)
    screen.text_area(key="inclusion").set_value(script.INCLUSION)
    screen.text_area(key="exclusion").set_value(script.EXCLUSION)
    screen = screen.button(key="start").click().run()

    replayed = screen.session_state["run"][0]
    assert [judged.paper.pmid for judged in replayed.screened] == [
        "900000001", "900000003", "900000002"]
    # The title heads its card in bold, since the cards of 17 September,
    # numbered from 1 since the finding of 18 September.
    drawn = [item.value for item in screen.markdown]
    assert any(line.startswith(f"**1. {synthetic_replies.PAPERS['900000001']['title']}**")
               for line in drawn)
    assert any(box.value == "Search completed."
               for box in screen.success), "every request answered from a file"
