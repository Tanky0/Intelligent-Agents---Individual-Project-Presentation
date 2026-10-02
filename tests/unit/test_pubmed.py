"""Tests for getting papers from PubMed (FR-06, UT-08)."""
# These use pretend replies rather than calling NCBI, so the tests give the same
# answer every time and a busy service cannot look like a fault in this code. The
# replies below are cut-down copies of real PubMed replies, keeping the parts
# this code has to read.
#
# respx is the library that supplies the pretend replies. A test marked with
# @respx.mock has its network calls caught by respx instead of leaving the
# computer, and any request the test did not describe is refused rather than
# quietly sent.

import httpx
import pytest
import respx

from research_agent.config import settings
from research_agent.sources import http, pubmed
from research_agent.sources.http import FetchError, RateLimiter
from research_agent.sources.pubmed import BATCH_SIZE, EUTILS, fetch, search

# A very high rate means no real waiting during the tests.
FAST = RateLimiter(1000.0)

# A paper whose abstract is split into labelled parts, which is the usual shape.
PAPER_WITH_ABSTRACT = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal><Title>Journal of Tests</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Ivermectin in <i>vitro</i> and beyond</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">First part.</AbstractText>
          <AbstractText Label="RESULTS">Second part.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Smith</LastName><Initials>AB</Initials></Author>
          <Author><LastName>Jones</LastName><Initials>C</Initials></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
    <PubmedData><ArticleIdList>
      <ArticleId IdType="doi">10.1000/TEST.123</ArticleId>
    </ArticleIdList></PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""

PAPER_WITHOUT_ABSTRACT = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle><MedlineCitation>
    <PMID>99999999</PMID>
    <Article><ArticleTitle>A letter with no abstract</ArticleTitle></Article>
  </MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


@respx.mock
def test_a_search_returns_the_paper_identifiers():
    respx.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={
            "esearchresult": {"count": "2", "idlist": ["111", "222"]}}))

    assert search("ivermectin", limiter=FAST) == ["111", "222"]


@respx.mock
def test_a_search_that_finds_nothing_returns_an_empty_list():
    """Finding nothing is a real answer, not a failure."""
    respx.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(
        return_value=httpx.Response(200, json={
            "esearchresult": {"count": "0", "idlist": []}}))

    assert search("nonsense query", limiter=FAST) == []


@respx.mock
def test_every_field_needed_for_screening_is_read():
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    records, errors = fetch(["12345678"], limiter=FAST)

    assert errors == []
    record = records[0]
    assert record.pmid == "12345678"
    assert record.year == "2021"
    assert record.journal == "Journal of Tests"
    assert record.authors == ["Smith AB", "Jones C"]
    assert record.doi == "10.1000/TEST.123"
    assert record.publication_types == ["Randomized Controlled Trial"]


@respx.mock
def test_a_title_containing_formatting_is_read_in_full():
    """Titles carry formatting tags. Reading only the plain text of the first
    part would drop everything after the first tag, without any error."""
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    records, _ = fetch(["12345678"], limiter=FAST)

    assert records[0].title == "Ivermectin in vitro and beyond"


@respx.mock
def test_an_abstract_split_into_sections_is_joined_back_together():
    """PubMed splits a structured abstract into separate labelled parts.
    Reading only the first would leave most of the abstract behind."""
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    records, _ = fetch(["12345678"], limiter=FAST)

    assert records[0].abstract == "First part. Second part."
    assert records[0].text_source == "abstract"


@respx.mock
def test_a_paper_with_no_abstract_is_marked_as_metadata_only():
    """The Curation Agent must not screen a title as though it were an
    abstract, so each paper records which it has (DC-07)."""
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITHOUT_ABSTRACT))

    records, _ = fetch(["99999999"], limiter=FAST)

    assert records[0].abstract == ""
    assert records[0].text_source == "metadata"


@respx.mock
def test_the_identifiers_are_sent_in_the_body_not_the_address():
    """A long list of identifiers in the web address made NCBI send back a
    cut-off reply, so the list travels in the body instead (R-02)."""
    route = respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    fetch(["12345678"], limiter=FAST)

    assert route.called
    assert route.calls[0].request.method == "POST"


@respx.mock
def test_one_failed_group_does_not_lose_the_papers_that_worked():
    """A result that is quietly smaller than it should be is the failure the
    design was most concerned about, so the loss is reported (FR-05)."""
    responses = [
        httpx.Response(200, text=PAPER_WITH_ABSTRACT),
        httpx.Response(404),
    ]
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        side_effect=responses)

    # One paper more than a group holds, so there are two groups. Taking the
    # size from the code means this still tests two groups if it changes. The
    # paper the first reply returns is asked for by name, because a paper that
    # was not asked for is now left out.
    too_many = ["12345678"]
    for number in range(BATCH_SIZE):
        too_many.append(str(number))

    records, errors = fetch(too_many, limiter=FAST)

    assert len(records) == 1, "the paper that came back is still returned"

    # Two separate losses, and both have to be reported: the group the service
    # refused, and the papers missing from the group it answered (R-04).
    statuses = []
    for failure in errors:
        assert isinstance(failure, FetchError)
        statuses.append(failure.status)

    assert 404 in statuses, "the group that failed is reported"
    assert 200 in statuses, "the papers missing from the group that worked too"


@respx.mock
def test_papers_missing_from_a_reply_are_reported_not_lost():
    """A reply can look complete and still be short of papers (R-04)."""
    # NCBI does not say that anything is missing: the papers simply are not
    # there. Without comparing what was asked for against what came back, three
    # papers would go in and one would come out with nothing reported, and the
    # recall figure would fall for a reason nobody could see.
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    records, errors = fetch(["12345678", "22222222", "33333333"], limiter=FAST)

    assert len(records) == 1, "the paper that did come back is still returned"
    assert len(errors) == 1, "the two that did not are reported"
    assert "22222222" in errors[0].reason
    assert "33333333" in errors[0].reason


@respx.mock
def test_a_paper_nobody_asked_for_is_not_kept():
    """A paper with no query behind it cannot be explained to a reviewer."""
    # It would sit in the results with nothing saying which search found it, so
    # it is left out and the oddity is reported instead (FR-15).
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=PAPER_WITH_ABSTRACT))

    records, errors = fetch(["99999999"], limiter=FAST)

    assert records == [], "the paper that was not asked for is not kept"
    reasons = " ".join(failure.reason for failure in errors)
    assert "not asked for" in reasons
    assert "12345678" in reasons


@respx.mock
def test_the_same_paper_twice_in_one_reply_is_counted_once():
    """Counting it twice would say more was found than there is."""
    twice = PAPER_WITH_ABSTRACT.replace(
        "</PubmedArticleSet>",
        PAPER_WITH_ABSTRACT.split("<PubmedArticleSet>")[1])
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text=twice))

    records, errors = fetch(["12345678"], limiter=FAST)

    assert len(records) == 1, "the paper is kept once"
    reasons = " ".join(failure.reason for failure in errors)
    assert "more than once" in reasons


@respx.mock
def test_a_reply_that_is_not_readable_is_reported_not_fatal():
    """A damaged reply must not stop the batches that worked."""
    respx.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(
        return_value=httpx.Response(200, text="<PubmedArticleSet><broken"))

    records, errors = fetch(["12345678"], limiter=FAST)

    assert records == []
    assert len(errors) == 1
    assert "could not be read" in errors[0].reason


def test_every_pubmed_call_waits_on_the_one_ncbi_limiter():
    """NCBI counts every request against one allowance (OB-01)."""
    # The limiter is looked up by service, so two calls get the same one; a
    # limiter made for each call would let them go out together.
    assert pubmed._limiter() is pubmed._limiter()
    assert pubmed._limiter() is http.limiter_for("ncbi", settings.ncbi_rate_limit)


def test_a_failed_search_passed_to_fetch_is_reported_not_crashed():
    """search returns either identifiers or a failure, and passing the failure
    on is an easy mistake. It is reported as a failure rather than raising
    something that points at the wrong cause."""
    failure = FetchError("pubmed", "https://example.test", 500, "status 500", 3)

    records, errors = fetch(failure)

    assert records == []
    assert errors == [failure]


# The years a request may carry, and the values NCBI wants for them. Written
# out here rather than worked out, so that a change to the wording is a change
# this test notices rather than follows.
YEARS_AS_NCBI_WANTS_THEM = [
    ({"from_year": 2020, "to_year": 2024},
     {"datetype": "pdat", "mindate": "2020", "maxdate": "2024"}),
    ({"from_year": 2020}, {"datetype": "pdat", "mindate": "2020",
                           "maxdate": "3000"}),
    ({"to_year": 2024}, {"datetype": "pdat", "mindate": "1500",
                         "maxdate": "2024"}),
    ({}, {}),
]


@pytest.mark.parametrize("years, expected", YEARS_AS_NCBI_WANTS_THEM)
@respx.mock
def test_a_search_sends_the_years_the_way_ncbi_asks_for_them(years, expected):
    """NCBI wants the two dates together and wants to be told which date."""
    # Sent on their own they are ignored. Without datetype the service filters on
    # the day it added the record rather than the day the paper was published, so
    # a paper from 1998 catalogued last year would match a search for work since
    # 2020.
    #
    # A search naming only one year still sends both, with the other end open,
    # because NCBI ignores one without the other.
    sent = {}

    def remember(request):
        sent.update(dict(request.url.params))
        return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})

    respx.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(side_effect=remember)

    search("anything", limit=5, limiter=FAST, **years)

    for name, value in expected.items():
        assert sent.get(name) == value, f"{name} was sent as {sent.get(name)!r}"
    if not expected:
        for name in ("datetype", "mindate", "maxdate"):
            assert name not in sent, f"{name} was sent with no years asked for"


def _pretend_search_reply(body):
    return respx.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(
        return_value=httpx.Response(200, json=body))


# Replies that parse as JSON and that this code cannot read. The first two are
# not records at all; then records put together differently from the one the
# code reads, including a result that gives a count and carries no list. The
# last is the one that would have failed furthest from its cause: fetch joins
# the identifiers into one request, and a number cannot be joined. What NCBI
# sends is not claimed here; what the code reads is.
@pytest.mark.parametrize("body", [
    [],
    "a string",
    {},
    {"esearchresult": []},
    {"esearchresult": {"count": "2"}},
    {"esearchresult": {"idlist": None}},
    {"esearchresult": {"idlist": "1"}},
    {"esearchresult": {"idlist": [1]}},
])
@respx.mock
def test_a_search_reply_of_the_wrong_shape_is_reported_not_raised(body):
    """A reply that parses is not yet a reply this can read."""
    # Before this, a reply that was not a record, or had no result record, or
    # whose result record was not one, raised inside the source, which would
    # have ended the run over one search and lost the sources that had
    # answered. The rest did not raise here: a result with the list left out
    # was read as nothing found, even when the same record said how many there
    # were, and a list that was null, text or held a number was handed back
    # from search unchanged. A null then failed in fetch where the identifiers
    # are counted, and a number where they are joined. Text joined without
    # complaint, one character at a time, so a search could have requested
    # identifiers nobody chose (FR-05).
    _pretend_search_reply(body)

    result = search("anything", limit=5, limiter=FAST)

    assert isinstance(result, FetchError), f"got {result!r}"
    assert "put together as expected" in result.reason


@respx.mock
def test_an_empty_identifier_list_is_nothing_found():
    """An empty list is how nothing found is written, and it stays an
    ordinary empty answer. A missing list is the case above, and the two
    must not be read the same way."""
    _pretend_search_reply({"esearchresult": {"count": "0", "idlist": []}})

    assert search("anything", limit=5, limiter=FAST) == []
