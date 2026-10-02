"""Tests for retrieving papers from Semantic Scholar (FR-07, UT-09)."""
# These use pretend replies rather than calling the service. A test that called
# the real service would spend part of the allowance the key permits, and would
# fail whenever the service was busy rather than when the code was wrong.
#
# The identifiers below are invented, so that a saved reply mentioning one can be
# recognised as having come from a test.

import httpx
import pytest
import respx

from research_agent.config import settings
from research_agent.sources import semantic_scholar
from research_agent.sources.http import FetchError, RateLimiter
from research_agent.sources.semantic_scholar import SEARCH_API, search

# A very high rate means no real waiting during the tests.
FAST = RateLimiter(1000.0)

FULL_REPLY = {
    "total": 1262162,
    "data": [{
        "paperId": "invented0000000000000000000000000000cafe",
        "title": "Ivermectin for COVID-19: a randomised trial",
        "abstract": "Patients received ivermectin for five days.",
        "year": 2022,
        "venue": "Journal of Tests",
        "publicationTypes": ["JournalArticle", "Randomized Controlled Trial"],
        "authors": [{"name": "A B Smith"}, {"name": "C Jones"}],
        "externalIds": {"DOI": "10.1000/invented", "PubMed": "9999990",
                        "CorpusId": 1},
    }]
}

# The service returns a paper with no abstract by giving the field as null,
# rather than by leaving it out.
NO_ABSTRACT_REPLY = {
    "total": 1,
    "data": [{
        "paperId": "invented0000000000000000000000000000beef",
        "title": "A letter with no abstract",
        "abstract": None,
        "year": None,
        "venue": None,
        "publicationTypes": None,
        "authors": None,
        "externalIds": None,
    }]
}

NOTHING_FOUND_REPLY = {"total": 0, "data": []}


def _pretend_reply(body, status=200):
    return respx.get(url__startswith=SEARCH_API).mock(
        return_value=httpx.Response(status, json=body))


@respx.mock
def test_every_field_needed_for_screening_is_read():
    _pretend_reply(FULL_REPLY)

    papers = search("ivermectin covid-19", limiter=FAST)

    paper = papers[0]
    assert paper.title == "Ivermectin for COVID-19: a randomised trial"
    assert paper.abstract == "Patients received ivermectin for five days."
    assert paper.year == "2022"
    assert paper.journal == "Journal of Tests"
    assert paper.authors == ["A B Smith", "C Jones"]
    assert paper.publication_types == ["JournalArticle",
                                       "Randomized Controlled Trial"]


@respx.mock
def test_the_identifiers_that_find_the_same_paper_elsewhere_are_kept():
    """The DOI and PubMed identifier are how a paper returned by both this
    service and PubMed is recognised as one paper rather than two (FR-10)."""
    _pretend_reply(FULL_REPLY)

    paper = search("ivermectin covid-19", limiter=FAST)[0]

    assert paper.doi == "10.1000/invented"
    assert paper.pmid == "9999990"


@respx.mock
def test_a_paper_with_no_abstract_is_marked_as_metadata_only():
    """The Curation Agent must not screen a title as though it were an
    abstract, so each paper records which it has (DC-07)."""
    _pretend_reply(NO_ABSTRACT_REPLY)

    paper = search("anything", limiter=FAST)[0]

    assert paper.abstract == ""
    assert paper.text_source == "metadata"


@respx.mock
def test_missing_fields_do_not_stop_a_paper_being_read():
    """This service gives a field as null rather than leaving it out, and it
    does so for most fields on a sparse record. Reading those directly would
    fail rather than returning the paper it does have."""
    _pretend_reply(NO_ABSTRACT_REPLY)

    paper = search("anything", limiter=FAST)[0]

    assert paper.title == "A letter with no abstract"
    assert paper.authors == []
    assert paper.year == ""
    assert paper.doi == ""
    assert paper.publication_types == []


@respx.mock
def test_a_search_that_finds_nothing_returns_an_empty_list():
    """Finding nothing is a real answer, not a failure."""
    _pretend_reply(NOTHING_FOUND_REPLY)

    assert search("nonsense query", limiter=FAST) == []


@respx.mock
def test_being_asked_to_wait_is_retried_then_reported():
    """The shared pool refuses a request when it is busy. That may work on a
    later attempt, but not for ever (DC-06)."""
    pretend = _pretend_reply({}, status=429)

    result = search("ivermectin", limiter=FAST)

    assert isinstance(result, FetchError)
    assert pretend.call_count == 3
    assert result.status == 429


@respx.mock
def test_the_source_is_recorded_on_every_paper():
    """A record has to say where it came from, so that a reviewer can tell
    which service reported it."""
    _pretend_reply(FULL_REPLY)

    assert search("ivermectin", limiter=FAST)[0].source == "semantic scholar"


@respx.mock
def test_the_key_is_sent_in_the_header_the_service_asks_for(monkeypatch):
    """Semantic Scholar reads the key from x-api-key and no other header."""
    # Sent under any other name it would be ignored, and the requests would go to
    # the shared pool with nothing saying so. The check looks at the request that
    # actually went out rather than at how it was built.
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "invented-key")
    _pretend_reply(FULL_REPLY)

    search("ivermectin", limiter=FAST)

    sent = respx.calls.last.request
    assert sent.headers["x-api-key"] == "invented-key"


@respx.mock
def test_no_key_header_is_sent_when_there_is_no_key(monkeypatch):
    """The searches still work without a key, so an empty one is not sent."""
    monkeypatch.setattr(settings, "semantic_scholar_api_key", "")
    _pretend_reply(FULL_REPLY)

    search("ivermectin", limiter=FAST)

    sent = respx.calls.last.request
    assert "x-api-key" not in sent.headers


def test_requests_are_kept_below_one_a_second():
    """The key allows one a second and the terms ask for less than that."""
    # The figure is pinned here so that raising it later has to be a deliberate
    # change to a test rather than a quiet edit to a number.
    assert semantic_scholar._REQUESTS_PER_SECOND < 1.0
    assert semantic_scholar._LIMITER.min_gap > 1.0


@respx.mock
def test_searches_one_after_another_share_the_same_limiter():
    """Two searches in a row must not both go out at once."""
    # The limit counts every request from this account together. A limiter made
    # inside each search would start with nothing to wait for, so the second
    # search would leave immediately after the first. Sharing one means the
    # second waits.
    _pretend_reply(FULL_REPLY)
    semantic_scholar._LIMITER.last_call = 0.0

    search("ivermectin covid-19")

    assert semantic_scholar._LIMITER.last_call > 0.0


@respx.mock
def test_a_search_never_asks_for_more_results_than_the_service_allows():
    """The service states that one search may ask for at most a hundred."""
    # The shared setting in a research request allows far more, because it is a
    # figure for the project rather than a promise from any one service. Sent as
    # it stands the request would be refused, so it is brought down here, where
    # the rule is known, rather than relying on every caller to remember it.
    asked = respx.get(SEARCH_API).mock(
        return_value=httpx.Response(200, json={"data": []}))

    search("anything", limit=5000, limiter=FAST)

    assert asked.calls[0].request.url.params["limit"] == "100"


# The years a request may carry, and the one value this service takes for them.
# Written out here rather than worked out, so that a change to the wording is a
# change this test notices rather than follows.
YEARS_AS_THIS_SERVICE_WANTS_THEM = [
    ({"from_year": 2020, "to_year": 2024}, "2020-2024"),
    ({"from_year": 2020}, "2020-"),
    ({"to_year": 2024}, "-2024"),
    ({"from_year": 2020, "to_year": 2020}, "2020-2020"),
]


def _remembering_what_was_sent(sent):
    """A pretend reply that records the values the request carried."""
    def remember(request):
        sent.update(dict(request.url.params))
        return httpx.Response(200, json={"data": []})
    return remember


@pytest.mark.parametrize("years, expected", YEARS_AS_THIS_SERVICE_WANTS_THEM)
@respx.mock
def test_a_search_sends_the_year_range_as_one_value(years, expected):
    """This service takes a range as one piece of text."""
    # Either end may be left empty for an open range, which is why a search
    # naming only a first year sends "2020-" rather than sending nothing.
    sent = {}
    respx.get(SEARCH_API).mock(side_effect=_remembering_what_was_sent(sent))

    search("anything", limit=5, limiter=FAST, **years)

    assert sent.get("year") == expected


@respx.mock
def test_a_search_with_no_years_sends_no_year():
    """The ordinary search is unchanged."""
    sent = {}
    respx.get(SEARCH_API).mock(side_effect=_remembering_what_was_sent(sent))

    search("anything", limit=5, limiter=FAST)

    assert "year" not in sent


# Replies that parse as JSON and that this code cannot read. The first two are
# not records at all; then records with no list of papers, including one that
# names a total and carries no list; then papers whose fields are the wrong
# kind of value, down to the nested authors and identifiers. What the service
# sends is not claimed here; what the code reads is.
@pytest.mark.parametrize("body", [
    [],
    "a string",
    {"total": 2},
    {"data": None},
    {"data": "not a list"},
    {"data": {"paperId": "x"}},
    {"data": ["not a record"]},
    {"data": [{"title": []}]},
    {"data": [{"abstract": 1}]},
    {"data": [{"year": []}]},
    {"data": [{"publicationTypes": "Review"}]},
    {"data": [{"publicationTypes": [1]}]},
    {"data": [{"authors": "a string"}]},
    {"data": [{"authors": [1]}]},
    {"data": [{"authors": [{"name": 1}]}]},
    {"data": [{"externalIds": []}]},
    {"data": [{"externalIds": {"DOI": 1}}]},
])
@respx.mock
def test_a_search_reply_of_the_wrong_shape_is_reported_not_raised(body):
    """A reply that parses is not yet a reply this can read."""
    # Before this, some of these raised inside the source, which would have
    # ended the run over one search and lost the sources that had answered.
    # Others passed through the record this file builds, which checks nothing,
    # and failed later in the planner: a title that is a non-empty list raises
    # where the planner strips it. A title that is an empty list was blanked
    # here, in _read_record, by the same "or" that turns an absent title into
    # an empty one, and the planner then counted the paper as untitled and
    # passed it over, quietly. And a record with a total and no list was read
    # as nothing found (FR-05).
    _pretend_reply(body)

    result = search("anything", limit=5, limiter=FAST)

    assert isinstance(result, FetchError), f"got {result!r}"
    assert "put together as expected" in result.reason


@respx.mock
def test_an_empty_list_of_papers_is_nothing_found():
    """An empty list is how nothing found is written, and it stays an
    ordinary empty answer. A missing list is the case above, and the two
    must not be read the same way."""
    _pretend_reply({"total": 0, "offset": 0, "data": []})

    assert search("anything", limit=5, limiter=FAST) == []


@respx.mock
def test_a_paper_with_its_optional_fields_left_out_is_still_read():
    """Absent is allowed everywhere inside a paper, because the record reads an
    absent field as empty and an empty field hides nothing from the reviewer.
    """
    # Only the wrong kind of value is refused. This says what the code does
    # with an absence, not what the service is known to send.
    _pretend_reply({"data": [{"paperId": "abc"}]})

    records = search("anything", limit=5, limiter=FAST)

    assert len(records) == 1
    assert records[0].paper_id == "abc"
    assert records[0].title == ""
