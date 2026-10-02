"""Tests for the shared HTTP code (UT-22, FR-18)."""
# Pretend responses are used so the tests give the same answer every time and do
# not depend on any service being available.

import json

import httpx
import respx

from research_agent.config import settings
from research_agent.processing import run_log
from research_agent.sources import http, saved_replies
from research_agent.sources.http import (
    FetchError,
    RateLimiter,
    _follow_reported_limit,
    get_json,
    limiter_for,
    request,
)


def test_a_service_is_given_one_limiter_and_keeps_it():
    """Asking twice for the same service must give back the same limiter."""
    # A limiter made for each call starts with nothing to wait for, so two calls
    # one after another would go out together, which is what a rate limit
    # forbids.
    first = limiter_for("a service used only by this test", 2.0)
    second = limiter_for("a service used only by this test", 2.0)

    assert first is second


def test_two_services_do_not_share_a_limiter():
    """Each service counts its own requests, so waiting for one is not waiting
    for the other."""
    one = limiter_for("the first test service", 2.0)
    other = limiter_for("the second test service", 2.0)

    assert one is not other


def test_a_reported_limit_replaces_the_configured_one():
    """Crossref reports 3 requests per 1 second, so the gap becomes a third."""
    limiter = RateLimiter(10.0)

    _follow_reported_limit(limiter, {"x-rate-limit-limit": "3",
                                     "x-rate-limit-interval": "1s"})

    assert limiter.requests_per_second == 3.0
    assert limiter.min_gap == 1 / 3


def test_a_longer_period_is_converted_to_requests_per_second():
    """50 requests per 10 seconds is 5 per second, not 50."""
    limiter = RateLimiter(1.0)

    _follow_reported_limit(limiter, {"x-rate-limit-limit": "50",
                                     "x-rate-limit-interval": "10s"})

    assert limiter.requests_per_second == 5.0


def test_a_service_that_reports_nothing_keeps_its_configured_rate():
    """NCBI sends no such headers, so its own limit must be left alone."""
    limiter = RateLimiter(10.0)

    _follow_reported_limit(limiter, {})

    assert limiter.requests_per_second == 10.0


def test_an_unreadable_limit_is_ignored():
    """A malformed header must not stop a run."""
    limiter = RateLimiter(3.0)

    _follow_reported_limit(limiter, {"x-rate-limit-limit": "lots",
                                     "x-rate-limit-interval": "1s"})

    assert limiter.requests_per_second == 3.0


@respx.mock
def test_the_limit_is_picked_up_from_a_real_response():
    """The whole path: a response carrying the headers updates the limiter."""
    respx.get("https://example.test/thing").mock(
        return_value=httpx.Response(
            200, json={"ok": True},
            headers={"X-Rate-Limit-Limit": "2", "X-Rate-Limit-Interval": "1s"}))

    limiter = RateLimiter(100.0)
    with httpx.Client() as client:
        get_json(client, "https://example.test/thing", source="test", limiter=limiter)

    assert limiter.requests_per_second == 2.0


def test_each_wait_is_twice_the_one_before(monkeypatch):
    """Semantic Scholar requires this of applications requesting a key, and
    every service used here asks callers to back off. A claim made on an
    application form should be one the code plainly honours."""
    waits = []
    monkeypatch.setattr("research_agent.sources.http.time.sleep", waits.append)

    with respx.mock:
        respx.get("https://example.test/busy").mock(
            return_value=httpx.Response(503))
        get_json(httpx.Client(), "https://example.test/busy", source="test")

    # Three attempts means two waits: after the first failure and the second.
    assert waits == [1.0, 2.0]


@respx.mock
def test_a_successful_reply_that_is_not_readable_json_is_reported():
    """A status of 200 is not a promise that the body can be read."""
    # Letting it raise would end a run over one bad reply and lose the sources
    # that did answer, so it is returned as a failure the coverage report can
    # count (FR-05).
    respx.get("https://example.test/thing").mock(
        return_value=httpx.Response(200, text="this is not json at all"))

    result = get_json(httpx.Client(), "https://example.test/thing", source="test")

    assert isinstance(result, FetchError)
    assert result.status == 200
    assert "not readable JSON" in result.reason


def attempts_written():
    return [line for line in run_log.read() if line["event"] == "request attempt"]


@respx.mock
def test_what_a_request_carried_is_written_to_the_run_log():
    """The log held the bounds a request asked for and not what went out."""
    # Both services take the years in their own wording, so a reader repeating a
    # search from the log was repeating the request rather than the search
    # (FR-15).
    respx.get("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    get_json(httpx.Client(), "https://example.test/search", source="pubmed",
             params={"db": "pubmed", "term": "drug X", "retmax": "50",
                     "mindate": "2020", "maxdate": "2024"})

    written = attempts_written()
    assert len(written) == 1
    assert written[0]["source"] == "pubmed"
    assert written[0]["method"] == "GET"
    assert written[0]["url"] == "https://example.test/search"
    assert written[0]["values sent beside it"] == [
        ["db", "pubmed"], ["term", "drug X"], ["retmax", "50"],
        ["mindate", "2020"], ["maxdate", "2024"]]


@respx.mock
def test_no_key_or_contact_address_reaches_the_run_log():
    """The keys and the contact address travel with these requests, and the log is
    kept in `evidence/`.
    """
    # This is the same allowed list the saved replies use, so the two cannot
    # disagree about what may be written down (ID-05, ID-13).
    respx.get("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    get_json(httpx.Client(), "https://example.test/search", source="pubmed",
             params={"term": "drug X", "api_key": "a secret",
                     "tool": "research-agent", "email": "somebody@example.test"})

    written = attempts_written()
    assert written[0]["values sent beside it"] == [["term", "drug X"]]
    assert "a secret" not in json.dumps(written)
    assert "somebody@example.test" not in json.dumps(written)


@respx.mock
def test_a_credential_written_into_the_address_does_not_reach_the_log():
    """An address can carry a key as well as a request can."""
    # Writing the address down whole was the first attempt, and a key in its
    # query and a password in front of its host both survived into the log. The
    # allowed list only ever saw the values passed beside the request.
    respx.get("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    get_json(httpx.Client(),
             "https://someone:a%20password@example.test/search"
             "?term=drug+X&api_key=a+secret",
             source="pubmed")

    written = attempts_written()
    assert written[0]["url"] == "https://example.test/search"
    assert written[0]["values in the address"] == [["term", "drug X"]]
    assert "a secret" not in json.dumps(written)
    assert "a password" not in json.dumps(written)
    assert "someone" not in json.dumps(written)


@respx.mock
def test_a_value_sent_twice_is_written_down_twice():
    """Merging the two lost one of them."""
    # A value beside the request and a different value of the same name in the
    # body both went to the service, and the log kept only the body's. A reader
    # would have seen a search that was never made.
    respx.post("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    request(httpx.Client(), "POST", "https://example.test/search",
            source="pubmed", params={"term": "in the address"},
            data={"term": "in the body"})

    written = attempts_written()
    assert written[0]["method"] == "POST"
    assert written[0]["values sent beside it"] == [["term", "in the address"]]
    assert written[0]["values sent in the body"] == [["term", "in the body"]]


def test_a_request_that_never_answered_is_still_recorded(monkeypatch):
    """Written before the request is handed over, so one that timed out is
    still recorded as an attempt. A run that tried three times and recorded
    none of them would understate what it did."""
    monkeypatch.setattr("research_agent.sources.http.time.sleep", lambda seconds: None)

    with respx.mock:
        respx.get("https://example.test/silent").mock(
            side_effect=httpx.TimeoutException("no answer"))
        get_json(httpx.Client(), "https://example.test/silent", source="pubmed",
                 params={"term": "drug X"})

    assert [line["attempt"] for line in attempts_written()] == [1, 2, 3]


def test_a_replayed_reply_records_no_attempt(monkeypatch):
    """Demonstration mode asks nothing of a service, so there is no attempt."""
    # A line saying otherwise would overstate what the run asked of the services
    # whose limits this project keeps to (OB-01, OB-06).
    monkeypatch.setattr(settings, "demo_mode", True)
    monkeypatch.setattr(saved_replies, "load", lambda *asked: {
        "status": 200, "body": '{"ok": true}',
        "headers": {"content-type": "application/json"}})

    class NoConnection:
        """A connection that fails the test if anything tries to use it."""

        def request(self, *asked, **named):
            raise AssertionError("a replayed reply must send nothing")

    answer = get_json(NoConnection(), "https://example.test/thing",
                      source="pubmed", params={"term": "drug X"})

    assert answer == {"ok": True}, "the saved reply came back"
    assert attempts_written() == []
    assert http.ncbi_attempts_by_run == {}, "and nothing was counted against NCBI"


# The hundred-attempt threshold for one run (OB-04). A hundred is this
# project's own number for a large job, which NCBI asks be run off-peak. The
# threshold is brought down to a few in these tests, because the point is
# where the count is kept and what happens at the line, not the number.


@respx.mock
def test_the_attempt_past_the_ncbi_threshold_is_refused_and_not_made(monkeypatch):
    """A run that reaches the threshold has the rest refused and reported."""
    # Nothing checks the window NCBI asks for, so a run that carried on past
    # the threshold outside it would have breached the guidance with nothing
    # saying so. A refused attempt comes back as a failure the coverage report
    # can count (FR-05).
    monkeypatch.setattr(http, "NCBI_REQUESTS_PER_RUN", 3)
    asked = respx.get("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    with run_log.while_running("run one"):
        for _ in range(3):
            get_json(httpx.Client(), "https://example.test/search", source="pubmed")
        refused = get_json(httpx.Client(), "https://example.test/search",
                           source="pubmed")

    assert asked.call_count == 3, "the three within the threshold went out"
    assert isinstance(refused, FetchError)
    assert "OB-04" in refused.reason
    assert refused.attempts == 0, "the refused request was never attempted"
    assert len(attempts_written()) == 3, "and no attempt line was written for it"


def test_retries_count_towards_the_ncbi_threshold(monkeypatch):
    """Every attempt is counted, not only the first for each request."""
    # Counting only first attempts would let a busy service triple what a run
    # asked of NCBI while the count said it was under the line.
    monkeypatch.setattr(http, "NCBI_REQUESTS_PER_RUN", 3)
    monkeypatch.setattr("research_agent.sources.http.time.sleep", lambda seconds: None)

    with respx.mock:
        busy = respx.get("https://example.test/busy").mock(
            return_value=httpx.Response(503))
        with run_log.while_running("run one"):
            get_json(httpx.Client(), "https://example.test/busy", source="pubmed")
            refused = get_json(httpx.Client(), "https://example.test/busy",
                               source="pubmed")

    assert busy.call_count == 3, "three attempts at the one request"
    assert isinstance(refused, FetchError)
    assert "OB-04" in refused.reason


def test_an_attempt_that_never_left_the_computer_still_counts(monkeypatch):
    """The count is of attempts, not of requests NCBI received."""
    # An attempt is counted before the connection is asked, and a connection
    # can fail before anything leaves the computer. Counting it anyway is the
    # cautious side: the count can only run ahead of what NCBI saw, never
    # behind. With the threshold at two, the third attempt at one request is
    # refused part way through its retries, and the refusal says two attempts
    # were counted, not that two requests were delivered.
    monkeypatch.setattr(http, "NCBI_REQUESTS_PER_RUN", 2)
    monkeypatch.setattr("research_agent.sources.http.time.sleep", lambda seconds: None)

    class NoRoute:
        """A connection that fails before sending, as a cut cable would."""

        def request(self, *asked, **named):
            raise httpx.ConnectError("the network is unreachable")

    with run_log.while_running("run one"):
        answer = get_json(NoRoute(), "https://example.test/search", source="pubmed")

    assert isinstance(answer, FetchError)
    assert answer.attempts == 2, "two attempts were made and counted"
    assert "OB-04" in answer.reason, "the third was refused by the threshold"
    assert "counted 2 attempts" in answer.reason
    assert "sent" not in answer.reason, "nothing claims a delivery"
    assert http.ncbi_attempts_by_run == {"run one": 2}


@respx.mock
def test_each_run_has_its_own_ncbi_count(monkeypatch):
    """Two browser sessions are two runs in one process. One run's requests
    must not spend the other's threshold, which is the same reason the run
    log keeps the run per execution context (R-19)."""
    monkeypatch.setattr(http, "NCBI_REQUESTS_PER_RUN", 1)
    asked = respx.get("https://example.test/search").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    with run_log.while_running("run one"):
        first = get_json(httpx.Client(), "https://example.test/search", source="pubmed")
    with run_log.while_running("run two"):
        second = get_json(httpx.Client(), "https://example.test/search", source="pubmed")

    assert first == {"ok": True}
    assert second == {"ok": True}
    assert asked.call_count == 2
    assert http.ncbi_attempts_by_run == {"run one": 1, "run two": 1}


@respx.mock
def test_other_services_are_not_counted_against_the_ncbi_threshold(monkeypatch):
    """The threshold is one service's guidance, not a limit on the run."""
    monkeypatch.setattr(http, "NCBI_REQUESTS_PER_RUN", 1)
    asked = respx.get("https://example.test/works").mock(
        return_value=httpx.Response(200, json={"ok": True}))

    with run_log.while_running("run one"):
        for _ in range(3):
            get_json(httpx.Client(), "https://example.test/works", source="crossref")

    assert asked.call_count == 3
    assert http.ncbi_attempts_by_run == {}
