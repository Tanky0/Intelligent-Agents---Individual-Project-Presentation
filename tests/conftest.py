"""Preparation applied to every test. Keeps tests away from the saved replies."""
# The name of this file cannot be changed. pytest looks for a file called exactly
# conftest.py and reads it automatically; anything else is ignored, and ignored
# silently. Renaming it once during development removed the protection below
# while every test still passed, and nine invented replies were written into the
# folder kept for demonstrations before it was noticed.
#
# What the protection is for: the application saves every reply it receives, so
# that whatever was run last can be run again with no network. Tests must never
# take part in that. Their replies are invented, and an invented reply saved here
# could later be played back during a demonstration in place of a real one. A
# made-up paper appearing in a demonstration would be worse than the
# demonstration failing.
#
# Rather than relying on each test to be careful, both halves are switched off
# here for all of them. Saving is turned off, and both folders are pointed at
# temporary ones that pytest deletes afterwards, so even a test that turned
# saving back on could not reach the real replies.
#
# There are two folders to redirect. Replies are captured into one and approved
# into the other, and a test must not be able to write to either: one would put
# invented replies where a demonstration reads from, and the other would leave
# them waiting to be approved by somebody who assumed they were real.

import pytest

from research_agent import graph
from research_agent.config import settings
from research_agent.processing import run_log, storage
from research_agent.sources import http, saved_replies


# autouse=True means pytest applies this to every test in the project without
# any test having to ask for it.
@pytest.fixture(autouse=True)
def never_touch_the_saved_replies(tmp_path, monkeypatch):
    """Keeps every test away from the replies kept for a demonstration."""
    monkeypatch.setattr(settings, "record_replies", False)
    monkeypatch.setattr(saved_replies, "CAPTURED_DIR", tmp_path / "captured")
    monkeypatch.setattr(saved_replies, "APPROVED_DIR", tmp_path / "approved")
    monkeypatch.setattr(saved_replies, "MANIFEST_PATH", tmp_path / "manifest.json")
    # The list of recordings that failed is kept between calls, so it is
    # emptied for each test rather than carrying one test's failure into
    # the next.
    monkeypatch.setattr(saved_replies, "recording_failures", [])
    monkeypatch.setattr(saved_replies, "promotion_failures", [])


@pytest.fixture(autouse=True)
def never_write_buffers_into_the_evidence_folder(tmp_path, monkeypatch):
    """Keeps buffer files created by tests separate from real run evidence."""
    # The application writes a buffer when the agents hand work over (ID-10).
    # Tests use invented papers, so their buffers are redirected to a temporary
    # folder instead of being written into the evidence folder.
    #
    # It is applied to every test rather than to the ones that remember.
    monkeypatch.setattr(graph, "BUFFER_DIR", tmp_path / "buffers")
    monkeypatch.setattr(graph, "buffer_failures", [])


@pytest.fixture(autouse=True)
def never_write_a_run_log_into_the_evidence_folder(tmp_path, monkeypatch):
    """Keeps the lines tests write out of a real run's log (FR-15)."""
    # The log is appended to, so a test run would leave its invented searches
    # and decisions among a real run's, and nothing in the file would say which
    # were which. Applied to every test rather than to the ones that remember,
    # for the same reason the buffer fixture is.
    monkeypatch.setattr(run_log, "DEFAULT_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(run_log, "writing_failures", [])
    # Every test starts with no run in progress, so a test that set one and
    # then failed cannot leave the next one writing lines under a run it knows
    # nothing about. Set the way the application sets it, rather than by
    # replacing the value, so the two cannot drift apart.
    with run_log.while_running(""):
        yield


@pytest.fixture(autouse=True)
def never_read_the_real_store(tmp_path, monkeypatch):
    """Keeps every test's storing and reading away from the real store."""
    # The screen names no store: it writes through `storage.save_approved`
    # and the results page reads through `storage.every_row`, and every
    # function in `storage.py` looks the file up by its name at the moment of
    # use. So one name moved here moves both, for every test, and a screen
    # test that stores nothing cannot show, or create, the real file.
    monkeypatch.setattr(storage, "DEFAULT_STORE", tmp_path / "approved_records.sqlite3")


@pytest.fixture(autouse=True)
def never_carry_an_ncbi_count_between_tests(monkeypatch):
    """Starts every test with no NCBI attempts counted (OB-04)."""
    # The count of attempts a run has made at NCBI lives for the whole process,
    # keyed by run. Most tests run under no run at all, so without this the
    # hundred-and-first pretend attempt across the suite would be refused by a
    # threshold meant for one real run.
    monkeypatch.setattr(http, "ncbi_attempts_by_run", {})
