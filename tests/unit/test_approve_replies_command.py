"""Checks the command that publishes replies, and the one that reviews them again."""
# The checks it applies live in saved_replies.py and are tested there. What is
# tested here is the part only the command does: deciding what the person asked
# for, refusing to act when the set is not what the manifest says it is, and
# asking before recording that somebody reviewed something.
#
# That last one is why this file exists. Reapproving writes down that a person
# looked at each published reply, which is the only part of the record no code
# can produce. A command that recorded it without asking, or that recorded it
# while the folder held a file nobody approved, would be writing a false
# statement in somebody else's name.
#
# The command is loaded from its path because scripts/ is not a package. It is
# the same file the person runs, not a copy of its logic.

import importlib.util
import json
import pathlib

import pytest

from research_agent.sources import saved_replies

COMMAND_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / "scripts" / "approve_replies.py")


def _load_the_command():
    """Imports the command as it stands on disk."""
    spec = importlib.util.spec_from_file_location("approve_replies", COMMAND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def command(tmp_path, monkeypatch):
    """The command, pointed at folders of its own and a known commit."""
    # git is not asked where the repository stands. A test that ran git would be
    # testing this machine's checkout rather than the command, and would answer
    # differently on somebody else's.
    module = _load_the_command()
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"
    monkeypatch.setattr(module, "approval_commit",
                        lambda: {"commit": "a" * 40,
                                 "working_tree": "nothing uncommitted"})
    return module


def _a_reply_that_needs_no_basis():
    """A Crossref answer holding one record and no abstract."""
    return {
        "captured_at": "2026-09-01T10:00:00+00:00",
        "method": "GET",
        "url": "https://api.crossref.org/works?query=anything",
        "asked": {"query": "anything"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"message": {"items": [{"title": ["a paper"]}]}}),
    }


def _publish_one(command, name="a_reply.json"):
    """Puts one approved reply in the folder, through the real approval path."""
    captured = saved_replies.CAPTURED_DIR / name
    captured.parent.mkdir(parents=True, exist_ok=True)
    captured.write_text(json.dumps(_a_reply_that_needs_no_basis()),
                        encoding="utf-8")

    manifest = saved_replies.read_manifest()
    assert saved_replies.approve_reply(captured, manifest,
                                       command.approval_commit()) == []
    assert saved_replies.write_manifest(manifest)
    return saved_replies.APPROVED_DIR / name


def test_reapproving_records_a_second_approval_and_keeps_the_first(command,
                                                                   monkeypatch):
    """The ordinary case: a person confirms they have reviewed what is there."""
    _publish_one(command)
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    manifest = saved_replies.read_manifest()
    assert command.reapprove_what_is_published(manifest) == 0

    history = (saved_replies.read_manifest()["files"]["a_reply.json"]
               ["approval_history"])
    assert len(history) == 2, "the review should be added to what was recorded"
    assert history[1]["repository_head_at_review"] == "a" * 40
    assert history[1]["publication_basis"]["kind"] == "no recognised article-text field"


def test_saying_anything_but_yes_records_nothing(command, monkeypatch):
    """The confirmation is the whole point of the command."""
    # Everything else it does could be done by a script. What it is really for is
    # a person saying they have looked, so anything other than saying so has to
    # leave the record alone.
    _publish_one(command)
    monkeypatch.setattr("builtins.input", lambda _: "I suppose so")

    manifest = saved_replies.read_manifest()
    assert command.reapprove_what_is_published(manifest) == 0

    history = (saved_replies.read_manifest()["files"]["a_reply.json"]
               ["approval_history"])
    assert len(history) == 1, "a review nobody confirmed was recorded"


def test_a_published_file_nobody_approved_stops_the_whole_command(command,
                                                                  monkeypatch):
    """A review of a set that is not what the manifest describes is not a review."""
    # An earlier version listed the strays and carried on, so the record would
    # have said a person reviewed the published replies while the folder held one
    # that was never approved and that nobody was shown.
    _publish_one(command)
    stray = saved_replies.APPROVED_DIR / "nobody_approved_this.json"
    stray.write_text(json.dumps(_a_reply_that_needs_no_basis()), encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    manifest = saved_replies.read_manifest()
    assert command.reapprove_what_is_published(manifest) == 1

    history = (saved_replies.read_manifest()["files"]["a_reply.json"]
               ["approval_history"])
    assert len(history) == 1, "something was recorded despite the stray file"


def test_a_manifest_entry_whose_file_has_gone_stops_the_command(command,
                                                                monkeypatch):
    """The other half of the same disagreement, and it was not checked at all."""
    published = _publish_one(command)
    published.unlink()
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    assert command.reapprove_what_is_published(saved_replies.read_manifest()) == 1


def test_nothing_is_recorded_when_the_commit_cannot_be_established(command,
                                                                   monkeypatch):
    """An approval that cannot say what was reviewed is not a record."""
    _publish_one(command)
    monkeypatch.setattr(command, "approval_commit", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    manifest = saved_replies.read_manifest()
    assert command.reapprove_what_is_published(manifest) == 1

    history = (saved_replies.read_manifest()["files"]["a_reply.json"]
               ["approval_history"])
    assert len(history) == 1


def test_one_basis_is_not_spread_across_several_replies(command):
    """A basis names the reply it was written for, and is not filled in."""
    # An earlier version supplied the name when it was missing, so one licence
    # description, written after reading one paper, was rebound to every file the
    # command was given. The convenience quietly undid the rule it was attached
    # to, which is why the name has to be written by the person.
    one = pathlib.Path("one.json")
    two = pathlib.Path("two.json")
    basis = {"kind": "open licence", "licence": "CC0 1.0"}

    assert command._basis_for(basis, one) is basis, "left as it was written"
    assert "applies_to" not in command._basis_for(basis, two)

    for_each = {"one.json": {"kind": "open licence", "applies_to": "one.json"},
                "two.json": {"kind": "open licence", "applies_to": "two.json"}}
    assert command._basis_for(for_each, one)["applies_to"] == "one.json"
    assert command._basis_for(for_each, two)["applies_to"] == "two.json"
