"""Tests for saving and replaying service replies."""
# The most important of these is the first one. A private key travelling in the
# request parameters must never reach a saved file, because the repository is
# public and one recorded run would publish it.
#
# Each test writes into a folder that pytest creates and deletes for it, so no
# test can disturb the replies saved for a demonstration.

import json

import httpx
import pytest

from research_agent.sources import saved_replies


# A fixture is preparation that pytest runs for a test. autouse=True means it
# runs for every test in this file without each one asking for it.
@pytest.fixture(autouse=True)
def temporary_saved_replies_folder(tmp_path, monkeypatch):
    """Points the saved replies at a folder that only lasts for one test."""
    # tmp_path is an empty folder pytest makes and deletes again. monkeypatch
    # replaces a value for one test and puts the original back afterwards.
    # Without this, running the tests would overwrite the real saved replies.
    monkeypatch.setattr(saved_replies, "CAPTURED_DIR", tmp_path / "captured")
    monkeypatch.setattr(saved_replies, "APPROVED_DIR", tmp_path / "approved")


# A Crossref answer holding one record with no abstract in it. The gate reads
# every reply for the text of a paper before it may be published, so a body
# shaped like nothing in particular could not be approved at all, and the
# fixtures have to stand in for what a service actually sends.
A_REPLY_WITH_NO_ARTICLE_TEXT = json.dumps(
    {"message": {"items": [{"title": ["a paper"], "DOI": "10.0000/example"}]}})


FOUND_ONE_PAPER = '{"message": {"items": [{"title": ["the first paper"]}]}}'
FOUND_ANOTHER_PAPER = '{"message": {"items": [{"title": ["the second paper"]}]}}'


def _reply(body=A_REPLY_WITH_NO_ARTICLE_TEXT, status=200, headers=None):
    """Builds a pretend reply, so no service is called."""
    return httpx.Response(status, content=body.encode("utf-8"),
                          headers=headers or {})


# An approval has to say which commit was being reviewed and why what it holds
# may be published. Both are supplied here rather than left out, because a test
# that approved without them would be exercising a path the command no longer
# has. They are invented, and say so: these replies are invented too.
_REVIEWED_AT = {"commit": "0" * 40, "working_tree": "invented for a test"}

_WHY_THESE_MAY_BE_PUBLISHED = {
    "kind": "written permission",
    "granted_by": "the author of this test, whose replies hold nobody's words",
    "granted_on": "when the test was written",
    "covers": "the invented replies this test file writes",
    "evidence": "this file",
    "grantor_could_authorise": True,
    "redaction": "nothing was taken out; there was nothing in it to take",
}


def _approve_what_was_captured():
    """Approves what was captured, the way the approval command does."""
    # It calls the same checks and writes the same manifest, so a test is not
    # exercising a path the application does not have. Copying the files across
    # by hand would have been simpler and would have proved less.
    manifest = saved_replies.read_manifest()
    for path in saved_replies.files_in(saved_replies.CAPTURED_DIR):
        # A basis is written for one reply, so it names the one it is for.
        basis = {**_WHY_THESE_MAY_BE_PUBLISHED, "applies_to": path.name}
        refused = saved_replies.approve_reply(path, manifest, _REVIEWED_AT, basis)
        assert not refused, f"{path.name} was refused: {refused}"
    assert saved_replies.write_manifest(manifest), "the manifest did not write"


def test_a_model_reply_saved_inside_a_run_carries_it():
    saved_replies.save_model_reply("model-a", "a question", "an answer",
                                   run="run one")

    assert _the_one_captured_record()["run"] == "run one"


def test_a_model_reply_saved_outside_any_run_has_no_run_field():
    """The run field is left out rather than written blank."""
    # A reply captured outside a run and one captured before the field
    # existed both have no run recorded, the record cannot say which of the
    # two it was, and the check over the approved set says so in those words
    # for both.
    saved_replies.save_model_reply("model-a", "a question", "an answer")

    assert "run" not in _the_one_captured_record()


def test_the_run_survives_approval_and_is_read_back_by_replay():
    """The approval rebuild keeps only the fields on its list (R-07). A run
    dropped there would be captured and then lost, and a check for a mixed
    replay set would find no run on anything (ID-20)."""
    saved_replies.save_model_reply("model-a", "a question", "an answer",
                                   run="run one")
    _approve_what_was_captured()

    replayed = saved_replies.load_model_reply("model-a", "a question")
    assert replayed["run"] == "run one"


def test_the_run_is_screened_with_the_other_describing_fields():
    """A field left off DESCRIBING_FIELDS reaches the published folder
    unchecked, which is the arrangement the list exists to refuse."""
    record = {"captured_at": "today", "model": "model-a", "question": "q",
              "answer": "a", "run": "someone@example.test"}

    assert "@" in saved_replies.what_should_not_be_published(record)


def test_the_run_of_each_model_reply_is_listed(tmp_path):
    """None for a record that has no run, and service replies left out,
    because nothing writes a run into them."""
    folder = tmp_path / "replies"
    folder.mkdir()
    _write_record(folder / "with-a-run.json", {
        "captured_at": "today", "model": "model-a", "question": "q",
        "answer": "a", "run": "run one"})
    _write_record(folder / "without.json", {
        "captured_at": "today", "model": "model-a", "question": "q2",
        "answer": "a"})
    _write_record(folder / "service.json", {
        "captured_at": "today", "method": "GET", "url": "https://example.test/",
        "asked": {}, "status": 200, "headers": {}, "body": "{}"})

    assert saved_replies.run_of_each_model_reply(folder) == {
        "with-a-run.json": "run one", "without.json": None}


def test_the_approved_folder_is_the_one_read_when_none_is_named():
    saved_replies.save_model_reply("model-a", "a question", "an answer",
                                   run="run one")
    _approve_what_was_captured()

    assert list(saved_replies.run_of_each_model_reply().values()) == ["run one"]


def test_a_folder_that_does_not_exist_lists_no_model_reply(tmp_path):
    assert saved_replies.run_of_each_model_reply(tmp_path / "missing") == {}


def _the_one_captured_record():
    [path] = saved_replies.files_in(saved_replies.CAPTURED_DIR)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_record(path, record):
    path.write_text(json.dumps(record), encoding="utf-8")


def test_nothing_identifying_the_caller_is_ever_written_to_disk():
    """The safety-critical case, and the reason for a list of what to keep."""
    # These files go into a public repository. An earlier version removed a list
    # of things it thought were secret, and let through everything nobody had
    # thought of. Only values describing what was asked are written now.
    params = {
        "db": "pubmed", "term": "ivermectin",       # what was asked
        "api_key": "SECRET-KEY-VALUE",              # who was asking
        "access_token": "ANOTHER-SECRET",
        # Written in two pieces so the file does not hold a whole
        # authorization header. This file is published with the rest of the
        # tests, and the release check looks for exactly this shape.
        "authorization": "Bear" + "er THIRD-SECRET",
        "password": "FOURTH-SECRET",
        "email": "someone@example.com",
        "tool": "uoeo-research-agent",
    }

    saved_replies.save("GET", "https://api.crossref.org/search", params, None, _reply())

    written = saved_replies.files_in(saved_replies.CAPTURED_DIR)[0].read_text(
        encoding="utf-8")

    for secret in ("SECRET-KEY-VALUE", "ANOTHER-SECRET", "THIRD-SECRET",
                   "FOURTH-SECRET", "someone@example.com"):
        assert secret not in written, f"{secret} reached the file"

    assert "ivermectin" in written, "what was asked is still recorded"


def test_no_session_identifier_or_cookie_is_written_to_disk():
    """NCBI sets a session identifier on every reply. It says who asked, not
    what came back, so it has no business in a file others can read."""
    reply = _reply(headers={"content-type": "application/json"})
    # Invented values. A real session identifier must not appear here either,
    # for the same reason the test exists. The cookie is written in two pieces
    # so the file does not hold the session identifier's name next to a value:
    # this file is published with the rest of the tests, and the release check
    # looks for exactly that shape.
    for name, value in (("ncbi-sid", "INVENTED_SESSION_VALUE"),
                        ("ncbi-phid", "INVENTED_PAGE_VALUE"),
                        ("set-cookie",
                         "ncbi" + "_sid=INVENTED_SESSION_VALUE; domain=.example")):
        reply.headers[name] = value

    saved_replies.save("GET", "https://api.crossref.org/thing", {}, None, reply)
    written = saved_replies.files_in(saved_replies.CAPTURED_DIR)[0].read_text(
        encoding="utf-8")

    for leaked in ("ncbi-sid", "ncbi-phid", "set-cookie", "INVENTED_SESSION_VALUE"):
        assert leaked not in written, f"{leaked} reached the file"


def test_a_reply_that_cannot_be_written_is_reported_not_raised():
    """The request already succeeded. Losing a real result because the evidence
    of it could not be filed away would be the wrong way round."""
    blocked = saved_replies.CAPTURED_DIR.parent / "blocked"
    blocked.write_text("this is a file, not a folder", encoding="utf-8")
    saved_replies.CAPTURED_DIR = blocked / "replies"

    assert saved_replies.save("GET", "https://api.crossref.org/x", {}, None, _reply()) is False


def test_a_damaged_saved_file_is_treated_as_missing():
    """A half-written file must report that nothing was saved, rather than
    stopping the run with a complaint about the shape of a file."""
    saved_replies.save("GET", "https://api.crossref.org/x", {"term": "a"}, None, _reply())
    _approve_what_was_captured()
    damaged = saved_replies.files_in(saved_replies.APPROVED_DIR)[0]
    damaged.write_text("{ this is not finished", encoding="utf-8")

    assert saved_replies.load("GET", "https://api.crossref.org/x", {"term": "a"}, None) is None


def test_every_saved_reply_records_when_it_was_captured():
    """A saved reply can be months old. The date makes that visible."""
    saved_replies.save("GET", "https://api.crossref.org/x", {"term": "a"}, None, _reply())

    _approve_what_was_captured()
    saved = saved_replies.load("GET", "https://api.crossref.org/x", {"term": "a"}, None)

    assert saved["captured_at"].startswith("20")


def test_the_same_request_finds_the_reply_saved_for_it():
    params = {"term": "ivermectin", "retmax": "3"}

    saved_replies.save("GET", "https://api.crossref.org/search", params, None,
                    _reply(FOUND_ONE_PAPER))
    _approve_what_was_captured()
    loaded = saved_replies.load("GET", "https://api.crossref.org/search", params, None)

    assert loaded["body"] == FOUND_ONE_PAPER
    assert loaded["status"] == 200


def test_a_changed_key_still_finds_the_same_reply():
    """The key is left out of the file name, so rotating it changes nothing."""
    before = {"term": "ivermectin", "api_key": "OLD"}
    after = {"term": "ivermectin", "api_key": "NEW"}

    saved_replies.save("GET", "https://api.crossref.org/search", before, None, _reply())

    _approve_what_was_captured()

    assert saved_replies.load("GET", "https://api.crossref.org/search", after, None)


def test_a_different_request_does_not_find_the_wrong_reply():
    """A demonstration replaying the wrong paper would be worse than failing."""
    saved_replies.save("GET", "https://api.crossref.org/search", {"term": "one"}, None,
                    _reply(FOUND_ANOTHER_PAPER))

    _approve_what_was_captured()

    assert saved_replies.load("GET", "https://api.crossref.org/search",
                           {"term": "two"}, None) is None


def test_asking_for_more_results_is_a_different_request():
    """A value that changes the reply has to change the file name (R-05)."""
    # Two searches for the same words, one asking for one paper and one asking
    # for a hundred, are not the same request. If both were filed under one name,
    # a demonstration could replay the short reply in place of the long one and
    # appear to have searched properly.
    search = "https://api.crossref.org/paper/search"

    saved_replies.save("GET", search, {"query": "same", "limit": "1",
                                       "fields": "title"}, None, _reply())
    saved_replies.save("GET", search, {"query": "same", "limit": "100",
                                       "fields": "title,abstract"}, None, _reply())

    assert saved_replies.count() == 2, "the two requests must be filed apart"


def test_each_span_of_years_is_a_different_request():
    """The years decide which papers come back, so they decide the file name."""
    # Semantic Scholar takes the whole range as one "year" value. Until ID-02 it
    # was left out of the identity, so a search with no dates and two searches
    # for different decades were all filed under one name and overwrote each
    # other. Checking the years of what comes back cannot recover a paper that is
    # absent from the wrong reply, so the dates have to be part of the identity
    # and not only of the search (R-05).
    search = "https://api.semanticscholar.org/graph/v1/paper/search"

    saved_replies.save("GET", search, {"query": "same"}, None, _reply())
    saved_replies.save("GET", search, {"query": "same", "year": "2020-2024"},
                       None, _reply())
    saved_replies.save("GET", search, {"query": "same", "year": "1990-1999"},
                       None, _reply())

    assert saved_replies.count() == 3, "no dates and each range are three requests"


def test_the_date_the_bounds_are_read_against_is_part_of_the_identity():
    """PubMed reads mindate and maxdate against whichever date datetype names."""
    # pdat is the date the paper was published and edat the date the record was
    # added, so the same bounds return different papers. datetype was not kept
    # until ID-02, so the two were filed under one name.
    search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    asked = {"db": "pubmed", "term": "same", "mindate": "2020", "maxdate": "2024"}

    saved_replies.save("GET", search, dict(asked, datetype="pdat"), None, _reply())
    saved_replies.save("GET", search, dict(asked, datetype="edat"), None, _reply())

    assert saved_replies.count() == 2, "pdat and edat are different requests"


def test_a_dated_search_reads_back_its_own_reply():
    """Filing them apart is not enough: each has to find the right one again."""
    # This is the failure the identity change exists to prevent. Without the
    # year in the name, the second save overwrote the first and both searches
    # replayed the same papers.
    search = "https://api.crossref.org/search"
    recent = {"term": "same", "year": "2020-2024"}
    older = {"term": "same", "year": "1990-1999"}

    saved_replies.save("GET", search, recent, None, _reply(FOUND_ONE_PAPER))
    saved_replies.save("GET", search, older, None, _reply(FOUND_ANOTHER_PAPER))
    _approve_what_was_captured()

    assert saved_replies.load("GET", search, recent, None)["body"] == FOUND_ONE_PAPER
    assert saved_replies.load("GET", search, older, None)["body"] == FOUND_ANOTHER_PAPER


def test_a_request_with_no_saved_reply_returns_nothing():
    assert saved_replies.load("GET", "https://api.crossref.org/missing", {}, None) is None


def test_only_the_headers_replay_needs_are_kept():
    """The text is saved unpacked, so a header saying it is still packed would
    tell the code to unpack it twice and reading the reply would fail. The rate
    limit figures are kept, because a replayed run should behave like a real one.
    """
    reply = _reply(headers={"x-rate-limit-limit": "3"})
    reply.headers["content-encoding"] = "gzip"

    saved_replies.save("GET", "https://api.crossref.org/thing", {}, None, reply)
    _approve_what_was_captured()
    loaded = saved_replies.load("GET", "https://api.crossref.org/thing", {}, None)

    assert "content-encoding" not in loaded["headers"]
    assert loaded["headers"]["x-rate-limit-limit"] == "3", "useful headers are kept"


def test_counting_replies():
    assert saved_replies.count() == 0

    saved_replies.save("GET", "https://api.crossref.org/a", {}, None, _reply())
    saved_replies.save("GET", "https://api.crossref.org/b", {}, None, _reply())

    assert saved_replies.count() == 2


def test_a_captured_reply_is_not_replayed_until_it_is_approved():
    """Capture is automatic, publication is not (R-07)."""
    # A reply that has been captured and not looked at must not turn up in a
    # demonstration. Nothing stops it being captured, and nothing replays it.
    saved_replies.save("GET", "https://api.crossref.org/x", {"term": "a"}, None, _reply())

    assert saved_replies.count() == 1, "it was captured"
    assert saved_replies.load("GET", "https://api.crossref.org/x",
                              {"term": "a"}, None) is None, "and not replayed"


def test_a_reply_changed_after_it_was_approved_is_not_replayed():
    """The fingerprint is checked before replaying, not only by the tests."""
    # A file edited by hand after approval, or dropped in with the right name,
    # would otherwise be played back during a demonstration and only reported
    # the next time the tests ran.
    saved_replies.save("GET", "https://api.crossref.org/x", {"term": "a"}, None, _reply())
    _approve_what_was_captured()

    approved = saved_replies.files_in(saved_replies.APPROVED_DIR)[0]
    approved.write_text('{"captured_at": "2026-01-01T00:00:00+00:00", "method": '
                        '"GET", "url": "https://api.crossref.org/x", "asked": {}, '
                        '"status": 200, "headers": {}, "body": "something else"}',
                        encoding="utf-8")

    assert saved_replies.load("GET", "https://api.crossref.org/x",
                              {"term": "a"}, None) is None


def test_a_reply_that_is_not_a_record_is_refused_rather_than_breaking():
    """A file holding a list, not a record, must not stop the approval run."""
    saved_replies.CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
    odd = saved_replies.CAPTURED_DIR / "not_a_record.json"
    odd.write_text('["this is a list"]', encoding="utf-8")

    reasons = saved_replies.why_it_cannot_be_approved(odd)

    assert reasons, "it has to be refused"
    assert "replaying" in reasons[0]


def test_a_saved_reply_that_is_not_text_is_read_as_unavailable(tmp_path):
    """A file that cannot be read is treated as one that is not there."""
    # That is what the function says it does. Reading a file raises
    # UnicodeDecodeError when its bytes are not text, and that was not among the
    # things it answered for, so replay stopped with a traceback rather than
    # reporting that nothing was saved for the request.
    path = tmp_path / "a_reply.json"
    path.write_bytes(b'{"body": "\xff\xfe not utf-8"}')

    assert saved_replies._read(path) is None


# A fixture is a reply this project wrote so the public demonstration can run
# with no network and publish only text this project wrote. It is recorded
# differently from a captured reply: instead of a decision about publishing
# somebody else's words, it carries an account of where it came from.
def _a_fixture_on_disk(what_it_holds=None):
    """Writes one fixture file and returns it with a manifest entry for it."""
    # Built here rather than through the approval command because that command
    # promotes what a service sent. A fixture was never captured from anything,
    # so there is nothing for it to promote.
    saved_replies.APPROVED_DIR.mkdir(parents=True, exist_ok=True)
    # Checked against None rather than written as "what_it_holds or ...",
    # because an empty record is a valid thing for a test to hand in and "or"
    # would have replaced it with the good one without saying so.
    if what_it_holds is None:
        record = {
            "captured_at": "2026-09-09T00:00:00+00:00",
            "model": "a model",
            "question": "a question",
            "answer": "an answer",
        }
    else:
        record = what_it_holds
    path = saved_replies.APPROVED_DIR / "a_fixture.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    here = saved_replies.fingerprint(path)

    entry = {
        "sha256": here,
        "approval_history": [{
            "approved_at": "2026-09-09T00:00:00+00:00",
            "approved_sha256": here,
            "repository_head_at_review": "0" * 40,
            "working_tree_at_review": "invented for a test",
            "fixture_origin": {
                "written_by": "the author of this test",
                "written_on": "2026-09-09",
                "what_it_stands_in_for": "an answer from the model",
                "copied_from_a_paper": False,
                "applies_to_sha256": here,
            },
        }],
    }
    return path, entry


def test_a_fixture_this_project_wrote_can_be_replayed():
    """The demonstration cannot run unless its own fixtures replay."""
    # This is the regression. The export boundary learned about fixtures and the
    # replay check did not, so a fixture passed the question about publishing it
    # and failed the question about replaying it, giving a reason about
    # publication that was not what was wrong. Nothing caught it because no
    # fixture existed yet (R-16).
    path, entry = _a_fixture_on_disk()

    assert saved_replies.why_it_cannot_be_replayed(entry, path) == []


def test_replaying_and_exporting_agree_about_a_fixture():
    """One file, two boundaries, and they must not disagree about it."""
    # A file that may be published and may not be replayed is not a state this
    # project has any use for, and it is what the gap produced.
    path, entry = _a_fixture_on_disk()

    assert saved_replies.why_it_cannot_be_exported(entry, path) == []
    assert saved_replies.why_it_cannot_be_replayed(entry, path) == []


def test_a_fixture_recorded_about_other_bytes_is_not_replayed():
    """Recording where a file came from is about a file, not about a name."""
    # Replay has to ask the same question the export boundary asks, or a fixture
    # could be edited after it was written down and still be replayed.
    path, entry = _a_fixture_on_disk()
    entry["approval_history"][-1]["fixture_origin"]["applies_to_sha256"] = "0" * 64

    reasons = saved_replies.why_it_cannot_be_replayed(entry, path)

    assert reasons, "it has to be refused"
    assert "a different version of the file" in " ".join(reasons)


def test_a_fixture_that_holds_nothing_replayable_is_refused():
    """A file in the replay folder is not a reply because it parses."""
    # The demonstration this exists for would fail on it, so the check that says
    # a fixture is a fixture reads what is in it rather than only its record.
    path, entry = _a_fixture_on_disk({"not": "a reply of any shape"})

    reasons = saved_replies.why_it_cannot_be_replayed(entry, path)

    assert reasons, "it has to be refused"
    assert "replaying it would need" in " ".join(reasons)
