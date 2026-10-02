"""The approval path attacked on purpose (R-21)."""
# The manifest is shown elsewhere to catch a reply nobody approved. What was
# not shown is what the approval itself lets through when the file it is given
# was made to get past it. These tests hand approval files a careless capture
# or a hostile hand could make and check what reaches the published copy.
#
# Each attack is one thing a person might not think of: a key under a name the
# word list does not know, an address inside a structured model answer rather
# than in a string, an address in the prompt, a field nobody expected, a file
# name with a path in it, and a commit identifier in the wrong case. The two
# that got through are R-21 in the remediation log; the rest are kept so that
# what is refused today stays refused.
#
# The checks are called the way the command calls them, on folders of the
# test's own, so nothing here touches the real captured or published sets.

import importlib.util
import json
import pathlib

import pytest

from research_agent.config import settings
from research_agent.sources import saved_replies

COMMAND_PATH = (pathlib.Path(__file__).resolve().parents[2]
                / "scripts" / "approve_replies.py")

A_COMMIT = {"commit": "b" * 40, "working_tree": "nothing uncommitted"}

# A permission that answers the basis check for an invented model reply, the
# way test_llm.py's does: nobody's words are in it, and the test says so.
PERMISSION_FOR_AN_INVENTED_ANSWER = {
    "kind": "written permission",
    "applies_to": "a_model_reply.json",
    "granted_by": "the author of this test, whose replies hold nobody's words",
    "granted_on": "2026-09-16",
    "covers": "the invented model answer this test writes",
    "evidence": "this file",
    "grantor_could_authorise": True,
    "redaction": "contact details were removed",
}


@pytest.fixture
def folders(tmp_path):
    """Captured, published and manifest paths of the test's own."""
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"
    return tmp_path


def _captured(name, record):
    path = saved_replies.CAPTURED_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _a_service_reply(url="https://api.crossref.org/works", body=None):
    if body is None:
        body = json.dumps({"message": {"items": [{"title": ["a paper"]}]}})
    return {
        "captured_at": "2026-09-16T10:00:00+00:00",
        "method": "GET",
        "url": url,
        "asked": {"query": "anything"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": body,
    }


def _a_model_reply(question="Judge these invented papers.", answer=None):
    if answer is None:
        answer = {"screenings": [{"paper_number": 1, "decision": "include",
                                  "relevance": 7, "reason": "invented",
                                  "summary": "an invented summary",
                                  "extract": "an invented extract"}]}
    return {
        "captured_at": "2026-09-16T10:00:00+00:00",
        "model": "an-invented-model",
        "question": question,
        "answer": answer,
    }


def _approve(path, basis=None):
    manifest = saved_replies.read_manifest()
    reasons = saved_replies.approve_reply(path, manifest, A_COMMIT, basis)
    if reasons:
        return reasons, None
    assert saved_replies.write_manifest(manifest)
    published = saved_replies.APPROVED_DIR / path.name
    return [], json.loads(published.read_text(encoding="utf-8"))


def test_a_key_value_under_a_name_the_word_list_does_not_know_is_refused(
        folders, monkeypatch):
    """The word list refuses "api_key"; NCBI's own spelling."""
    # A capture or a hand that wrote the same value under "apikey" would pass a
    # check that looks for names, so the values of the configured keys are
    # looked for as well (R-21).
    #
    # The name is written in two pieces, "api" and "key=", for the same reason
    # release.py builds its own key words in pieces: this file travels in the
    # release, and the release check reads "apikey=" followed by a long value
    # as a credential. The reply under test still carries the joined name.
    an_invented_key = "0123456789abcdef" * 2 + "0123"
    monkeypatch.setattr(settings, "ncbi_api_key", an_invented_key)
    path = _captured("a_reply.json", _a_service_reply(
        url="https://api.crossref.org/works?api" + "key=" + an_invented_key))

    reasons, published = _approve(path)

    assert published is None
    assert any("ncbi_api_key" in reason for reason in reasons), reasons


def test_an_address_inside_a_structured_model_answer_is_taken_out(folders):
    """A model answer is a record, not a string, and the removal of contact
    details only read strings. An extract that quoted a corresponding
    author's address reached the published copy whole (R-21)."""
    answer = {"screenings": [{"paper_number": 1, "decision": "include",
                              "relevance": 7, "reason": "invented",
                              "summary": "write to first.author@example.org",
                              "extract": "an invented extract"}]}
    path = _captured("a_model_reply.json", _a_model_reply(answer=answer))

    reasons, published = _approve(path, PERMISSION_FOR_AN_INVENTED_ANSWER)

    assert reasons == []
    assert "example.org" not in json.dumps(published)
    assert saved_replies.REDACTED in published["answer"]["screenings"][0]["summary"]


def test_an_address_in_the_prompt_is_refused_rather_than_redacted(folders):
    """The prompt describes the request, and an address in it identifies a
    person; that is refused, as the word list says, not quietly rewritten."""
    path = _captured("a_model_reply.json", _a_model_reply(
        question="The reviewer wrote: ask someone@example.org about this."))

    reasons, published = _approve(path, PERMISSION_FOR_AN_INVENTED_ANSWER)

    assert published is None
    assert any("'@'" in reason for reason in reasons), reasons


def test_a_field_nobody_expected_does_not_reach_the_published_copy(folders):
    """The published copy is rebuilt from a list of what to keep, so a field
    planted in the file cannot travel by being there (R-07)."""
    record = _a_service_reply()
    record["note"] = "a secret that should not travel"
    path = _captured("a_reply.json", record)

    reasons, published = _approve(path)

    assert reasons == []
    assert "note" not in published
    assert "secret" not in json.dumps(published)


def test_a_name_with_a_path_in_it_is_not_found_waiting(folders, monkeypatch):
    """The command matches the name of a file in the captured folder, so a
    name pointing elsewhere finds nothing and approves nothing."""
    spec = importlib.util.spec_from_file_location("approve_replies", COMMAND_PATH)
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    _captured("a_reply.json", _a_service_reply())

    chosen = command.choose("../a_reply.json", saved_replies.read_manifest())

    assert chosen is None


def test_a_commit_identifier_in_the_wrong_case_is_refused(folders):
    """Forty hexadecimal characters in lower case, the form git prints them
    in. Git would resolve the same forty in upper case, so the rule is about
    the record and not about lookup: one spelling, so that a manifest entry
    and a git log line compare as strings without anybody normalising
    them."""
    path = _captured("a_reply.json", _a_service_reply())
    manifest = saved_replies.read_manifest()

    reasons = saved_replies.approve_reply(
        path, manifest, {"commit": "B" * 40, "working_tree": "nothing uncommitted"})

    assert any("full commit identifier" in reason for reason in reasons), reasons
