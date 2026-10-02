"""Tests for looking up a paper's retraction status (FR-09, UT-11)."""
# These tests use pretend replies rather than calling Crossref. Two reasons. The
# tests then give the same answer every time, so a failure means this code
# changed rather than that a service was busy. And a retraction notice cannot be
# arranged on demand, so the cases that matter most could not be tested at all
# against the real service.
#
# respx is the library that supplies the pretend replies. A test marked with
# @respx.mock has its network calls caught by respx instead of leaving the
# computer, and any request the test did not describe is refused rather than
# quietly sent.

from typing import get_args

import httpx
import pytest
import respx

from research_agent import schemas
from research_agent.sources.crossref import (
    CROSSREF_API,
    PublicationStatus,
    fetch_status,
    fetch_statuses,
    normalise_doi,
)
from research_agent.sources.http import FetchError, RateLimiter

# A very high rate means no real waiting during the tests. The gap between
# requests is tested on its own in test_http.py.
FAST = RateLimiter(1000.0)


# parametrize runs the same test once for each pair below, reporting each as a
# separate result. Writing four almost identical tests would hide which form of
# DOI failed when one of them did.
@pytest.mark.parametrize(
    "written,expected",
    [
        ("10.1016/J.IJSU.2021.105906", "10.1016/j.ijsu.2021.105906"),
        ("https://doi.org/10.1038/S41467-025-62058-5", "10.1038/s41467-025-62058-5"),
        ("doi:10.1145/191246.191322", "10.1145/191246.191322"),
        ("  10.1000/XYZ  ", "10.1000/xyz"),
    ],
)
def test_the_same_doi_written_four_ways_gives_one_answer(written, expected):
    """Duplicate papers are found by comparing DOIs as plain text (FR-10)."""
    assert normalise_doi(written) == expected


def _pretend_reply(body, status=200):
    """Arranges the reply Crossref will appear to send."""
    # url__startswith means the pretend reply is used for any address beginning
    # with the Crossref one, because each test asks about a different DOI and the
    # full address is therefore different every time.
    return respx.get(url__startswith=CROSSREF_API).mock(
        return_value=httpx.Response(status, json=body)
    )


@respx.mock
def test_a_retracted_paper_is_reported_as_retracted():
    _pretend_reply({"message": {"title": ["A withdrawn study"],
                                "update-to": [{"type": "retraction"}]}})

    record = fetch_status("10.1000/retracted", limiter=FAST)

    assert record.status is PublicationStatus.RETRACTED


@respx.mock
def test_a_paper_under_question_is_not_reported_as_retracted():
    """An expression of concern is a doubt raised, not a retraction."""
    _pretend_reply({"message": {"title": ["A questioned study"],
                                "update-to": [{"type": "expression_of_concern"}]}})

    record = fetch_status("10.1000/concern", limiter=FAST)

    assert record.status is PublicationStatus.CONCERNS_RAISED


@respx.mock
def test_no_notice_is_reported_as_none_identified_never_as_not_retracted():
    """The most important test in this file."""
    # Crossref only knows about a retraction if the publisher sent one in, so no
    # notice does not prove a paper is sound. Reporting it as "not retracted"
    # would turn missing information into a claim about safety. The last check
    # makes sure the code has no way of saying that even by accident.
    _pretend_reply({"message": {"title": ["An ordinary study"]}})

    record = fetch_status("10.1000/ordinary", limiter=FAST)

    assert record.status is PublicationStatus.NONE_IDENTIFIED
    assert record.status.value == "No retraction or formal concern identified"

    # Each of the three responses must describe what was checked without
    # making any claim about the paper's quality.
    for status in PublicationStatus:
        assert "not retracted" not in status.value.lower()
        assert "no concerns" not in status.value.lower()


@respx.mock
def test_a_missing_paper_is_reported_at_once_without_retrying():
    """A 404 gives the same answer next time, so retrying wastes the allowance."""
    pretend = _pretend_reply({}, status=404)

    result = fetch_status("10.1000/missing", limiter=FAST)

    assert isinstance(result, FetchError)
    assert result.status == 404
    assert pretend.call_count == 1


@respx.mock
def test_a_service_problem_is_retried_three_times_then_reported():
    """A 503 may work on a later attempt, but not for ever (DC-06)."""
    pretend = _pretend_reply({}, status=503)

    result = fetch_status("10.1000/unavailable", limiter=FAST)

    assert isinstance(result, FetchError)
    assert pretend.call_count == 3
    assert result.attempts == 3


@respx.mock
def test_a_doi_with_an_awkward_character_asks_for_the_whole_doi():
    """A DOI suffix may hold characters that mean something else in a web address."""
    # Crossref names the number sign among them. Sent as it stands, everything
    # after it is treated as part of the page rather than part of the DOI, so the
    # request asks about something other than the paper wanted. What comes back
    # might be a refusal, or it might be an answer about a different paper, and
    # the second is the one nothing would notice (FR-09).
    asked = respx.get("https://api.crossref.org/works/10.1000/abc%23def").mock(
        return_value=httpx.Response(200, json={"message": {
            "DOI": "10.1000/abc#def", "title": ["A paper with an awkward DOI"]}}))

    result = fetch_status("10.1000/abc#def", limiter=FAST)

    assert asked.called, "the whole DOI was asked for, with the character encoded"
    assert result.doi == "10.1000/abc#def"


# Replies that parse as JSON and that this code cannot read. The first two are
# not records at all; then records with no message record; then message
# records whose notices or title are the wrong kind of value. What Crossref
# sends is not claimed here; what the code reads is.
@pytest.mark.parametrize("body", [
    [],
    "a string",
    {},
    {"message": []},
    {"message": "not a record"},
    {"message": {"update-to": 1}},
    {"message": {"update-to": ["not a record"]}},
    {"message": {"update-to": [{}]}},
    {"message": {"update-to": [{"type": None}]}},
    {"message": {"update-to": [{"type": ""}]}},
    {"message": {"update-to": [{"type": "   "}]}},
    {"message": {"update-to": [{"type": 3}]}},
    {"message": {"title": "a string, whose first character would be read"}},
    {"message": {"title": [1]}},
])
@respx.mock
def test_a_status_reply_of_the_wrong_shape_is_reported_not_raised(body):
    """A reply that parses is not yet a reply this can read."""
    # Before this, some of these raised inside the source, and the rest were
    # read quietly, which was worse. An empty record was read as a paper with
    # no notices found, so was a notice with no kind, and a title given as text
    # would have been read as its first character only. Status is a warning
    # shown to the reviewer, not something a run cannot do without, so a raise
    # was the wrong size of failure and a silent misreading was the wrong kind
    # (FR-05, FR-09).
    _pretend_reply(body)

    result = fetch_status("10.1000/odd", limiter=FAST)

    assert isinstance(result, FetchError), f"got {result!r}"
    assert "put together as expected" in result.reason


@respx.mock
def test_a_message_with_no_notices_and_no_title_is_an_ordinary_answer():
    """A message with no notice list and no title is an ordinary answer, not
    a wrong shape.
    """
    # Absent is allowed for these two because of what the code does with the
    # absence and not because of what Crossref is known to send: a message
    # with no notice list is a paper with no notices, which most papers are,
    # and one with no title is a paper whose title is not known.
    _pretend_reply({"message": {"DOI": "10.1000/plain"}})

    record = fetch_status("10.1000/plain", limiter=FAST)

    assert record.status is PublicationStatus.NONE_IDENTIFIED
    assert record.title is None


def test_the_three_sentences_are_the_same_in_every_place_they_are_kept():
    """The wording is kept together in the schema, and it still appears twice
    there, as named constants and again inside the Literal, repeated so the
    annotation is valid for static type checking.
    """
    # This is what holds those two to each other, and the enum this source
    # produces to both, so that a sentence edited in one place and not another
    # is a failing test rather than a status the record refuses in a live run
    # (ID-19).
    named = {schemas.NOT_CHECKED, schemas.RETRACTED, schemas.CONCERNS_RAISED,
             schemas.NONE_IDENTIFIED}
    written_out = set(get_args(schemas.PublicationStatus))
    produced_here = {status.value for status in PublicationStatus}

    assert named == written_out
    assert produced_here == named - {schemas.NOT_CHECKED}


@respx.mock
def test_a_list_of_dois_gives_an_answer_for_each_with_the_failures_kept():
    """The hand-over asks about every paper it gives curation in one go (R-18)."""
    # A failure is handed back beside the records rather than dropped, because
    # the caller reports a check that failed. Dropped, a paper whose check
    # failed would look exactly like one Crossref was never asked about.
    respx.get(f"{CROSSREF_API}/10.1000/retracted").mock(
        return_value=httpx.Response(200, json={"message": {
            "update-to": [{"type": "retraction"}]}}))
    respx.get(f"{CROSSREF_API}/10.1000/missing").mock(
        return_value=httpx.Response(404, json={}))

    answers = fetch_statuses(["10.1000/retracted", "10.1000/missing"], limiter=FAST)

    assert set(answers) == {"10.1000/retracted", "10.1000/missing"}
    assert answers["10.1000/retracted"].status is PublicationStatus.RETRACTED
    assert isinstance(answers["10.1000/missing"], FetchError)
    assert answers["10.1000/missing"].status == 404


@respx.mock
def test_a_doi_given_twice_is_asked_about_once_and_keeps_its_first_answer():
    """Deduplication keeps two records with one DOI apart when their PubMed
    identifiers disagree, so the hand-over can send the same DOI twice.
    """
    # Asked twice, a second lookup that failed replaced the retraction the
    # first had found, and both records went to the reviewer blank (R-18).
    #
    # The second reply is a 404, which is not retried, so a second lookup gets
    # a failure at once. A 503 would be asked again and run the pretend replies
    # out, and the test would then fail for the wrong reason.
    asked = respx.get(f"{CROSSREF_API}/10.1000/twice").mock(side_effect=[
        httpx.Response(200, json={"message": {"update-to": [{"type": "retraction"}]}}),
        httpx.Response(404, json={}),
    ])

    answers = fetch_statuses(["10.1000/twice", "10.1000/twice"], limiter=FAST)

    assert list(answers) == ["10.1000/twice"]
    assert answers["10.1000/twice"].status is PublicationStatus.RETRACTED
    assert asked.call_count == 1


@respx.mock
def test_a_list_with_no_dois_asks_nothing():
    """A run whose papers have no DOI must not touch the network to find out."""
    # respx refuses any request a test did not describe, so a request here
    # would fail the test rather than quietly leave the computer.
    assert fetch_statuses([], limiter=FAST) == {}
