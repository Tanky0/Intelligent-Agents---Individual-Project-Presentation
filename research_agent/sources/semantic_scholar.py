"""Retrieves paper records from Semantic Scholar."""
# The searches used here work without an account, but this project holds a key
# and sends it as the x-api-key header when the settings have one. A key takes
# requests out of the pool shared by every caller without one, where a busy
# moment returns a refusal to wait for (R-03). The shared request handling
# already treats such a refusal as temporary and tries again after a pause, so
# nothing further is needed here.
#
# The key allows one request a second, counted across the whole service rather
# than for each kind of search, and the terms ask callers to keep below that
# figure rather than sit on it. Two things follow. The gap between requests is
# set a little wider than a second, and one limiter is shared by everything in
# this file. A limiter made afresh for each search would have nothing to wait
# for, so two searches one after another would go out together, which is what
# the limit forbids.
#
# This service matches a query far more loosely than PubMed does. The same words
# that return several hundred papers from PubMed return over a million here,
# because a paper counts as matching if the words appear anywhere in it. The
# count reported by a search is therefore not a measure of how much evidence
# exists, and the planning agent should read it as a starting point rather than
# as a total.
#
# Each record carries whatever identifiers Semantic Scholar holds for the paper,
# including its PubMed identifier and DOI where it has them. Those are what allow
# the same paper to be recognised when it also comes back from PubMed.
#
# Records are described the same way as those from PubMed, including whether the
# searchable text holds an abstract or only reference details, so that the
# Curation Agent can treat them alike (DC-07).

from dataclasses import dataclass, field

import httpx

from research_agent.config import settings
from research_agent.sources.http import FetchError, get_json, limiter_for

SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Semantic Scholar returns only the fields it is asked for. These are the ones
# the Curation Agent and deduplication need.
_FIELDS = "title,abstract,year,authors,externalIds,venue,publicationTypes"

# What this service allows one search to ask for. It is its number rather than
# this project's, taken from the interface description it publishes: the
# relevance search used here states that the limit "Must be <= 100". The same
# description says a query can reach no more than a thousand results however it
# is paged, and that anything larger needs the separate bulk search. Neither
# paging nor bulk search is built, so a hundred is as far as one search goes.
#
# The shared setting in ResearchRequest allows a far larger figure, because it
# is a project-wide ceiling that has to suit PubMed as well. Sending it here as
# it stands would have the request refused, so search below brings it down.
MOST_IN_ONE_REQUEST = 100

# The key allows one request a second and the terms ask for a rate below that,
# so this asks for slightly fewer. It leaves a gap of a quarter of a second
# more than the limit, which covers the difference between the moment a request
# is sent and the moment the service counts it. That is still much slower than
# the rate used for NCBI.
_REQUESTS_PER_SECOND = 0.8

# One limiter for the whole service, rather than one made inside each search.
# The limit counts every request from this account together, so searches run
# one after another have to share the same gap. See the note at the top of the
# file, and limiter_for in http.py.
_LIMITER = limiter_for("semantic scholar", _REQUESTS_PER_SECOND)


@dataclass
class SemanticScholarRecord:
    paper_id: str
    title: str
    abstract: str
    authors: list = field(default_factory=list)
    year: str = ""
    journal: str = ""
    publication_types: list = field(default_factory=list)
    doi: str = ""
    pmid: str = ""
    # Use "abstract" when abstract text is available; otherwise use "metadata".
    text_source: str = "metadata"
    source: str = "semantic scholar"


def _new_connection():
    """Opens a connection that names this application to Semantic Scholar."""
    headers = {"User-Agent": settings.user_agent("semantic scholar")}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key
    return httpx.Client(headers=headers, timeout=40.0)


def _read_authors(paper):
    """Returns the author names, in the order the service lists them."""
    names = []
    for author in paper.get("authors") or []:
        name = author.get("name")
        if name:
            names.append(name)
    return names


def _why_the_search_reply_is_not_readable(body):
    """Returns what is wrong with the shape of a search reply, or None."""
    # A reply that parsed is not yet a reply this can read. The list of papers
    # has to be there: a record that names a total and carries no list is
    # exactly the reply that must not be read as nothing found, and an empty
    # list is how nothing found is written.
    #
    # Inside each paper only the fields this file reads are checked, and each
    # may be absent or null. That is allowed because of what the record does
    # with an absence, which is to read it as empty, and an empty field hides
    # nothing from the reviewer. It is not a claim about what the service sends.
    #
    # The record these are read into is a plain dataclass that checks nothing,
    # so a wrong kind of value would travel on to the planner. A title that is
    # a non-empty list raises there, where the planner strips it, with a
    # message about stripping rather than about the reply. A title that is an
    # empty list is blanked here, in _read_record, by the same "or" that turns
    # an absent title into an empty one; the planner then counts the paper as
    # untitled and passes it over, which is the quiet kind of wrong.
    if not isinstance(body, dict):
        return "not a record"
    papers = body.get("data")
    if not isinstance(papers, list):
        return "no list of papers"
    for paper in papers:
        if not isinstance(paper, dict):
            return "a paper that is not a record"
        for name in ("paperId", "title", "abstract", "venue"):
            value = paper.get(name)
            if value is not None and not isinstance(value, str):
                return f"a {name} that is not text"
        year = paper.get("year")
        if year is not None and not isinstance(year, (int, str)):
            return "a year that is neither a number nor text"
        kinds = paper.get("publicationTypes")
        if kinds is not None:
            if not isinstance(kinds, list):
                return "publication types that are not a list"
            for kind in kinds:
                if not isinstance(kind, str):
                    return "a publication type that is not text"
        authors = paper.get("authors")
        if authors is not None:
            if not isinstance(authors, list):
                return "authors that are not a list"
            for author in authors:
                if not isinstance(author, dict):
                    return "an author that is not a record"
                name = author.get("name")
                if name is not None and not isinstance(name, str):
                    return "an author name that is not text"
        identifiers = paper.get("externalIds")
        if identifiers is not None:
            if not isinstance(identifiers, dict):
                return "external identifiers that are not a record"
            for name in ("DOI", "PubMed"):
                value = identifiers.get(name)
                if value is not None and not isinstance(value, str):
                    return f"a {name} that is not text"
    return None


def _read_record(paper):
    """Turns one paper from the reply into a record."""
    abstract = paper.get("abstract") or ""

    if abstract:
        text_source = "abstract"
    else:
        text_source = "metadata"

    identifiers = paper.get("externalIds") or {}

    return SemanticScholarRecord(
        paper_id=paper.get("paperId") or "",
        title=paper.get("title") or "",
        abstract=abstract,
        authors=_read_authors(paper),
        year=str(paper.get("year") or ""),
        journal=paper.get("venue") or "",
        publication_types=paper.get("publicationTypes") or [],
        doi=identifiers.get("DOI") or "",
        pmid=identifiers.get("PubMed") or "",
        text_source=text_source,
    )


def _the_years_asked_for(from_year, to_year):
    """The year value this service wants, or nothing when none were asked for."""
    # It takes a range as one piece of text, with either end left empty for an
    # open one: "2016-2020" for a range, "2010-" for anything since, "-2015" for
    # anything up to. That is this service's wording, which is why it is written
    # here rather than shared with the PubMed source.
    if from_year is None and to_year is None:
        return {}
    first = "" if from_year is None else str(from_year)
    last = "" if to_year is None else str(to_year)
    return {"year": f"{first}-{last}"}


def search(query, limit=50, connection=None, limiter=None,
           from_year=None, to_year=None):
    """Returns the papers matching a search, or a description of what went wrong."""
    # Unlike the PubMed source, one call returns the papers themselves rather
    # than identifiers to fetch afterwards, because this service returns the
    # details with the search.
    #
    # Sending the years is not the same as knowing they were applied. Sending a
    # filter does not by itself prove it was applied correctly: a service may
    # reject it, ignore it, or read it differently. So the caller checks the
    # years of what comes back.
    opened_own_connection = connection is None
    if connection is None:
        connection = _new_connection()
    if limiter is None:
        limiter = _LIMITER

    try:
        # Brought down here rather than trusted from the caller, because the
        # limit is this service's rule and this is the file that knows it. The
        # Planning Agent lowers it too, so that it can say in the run that the
        # number asked for was reduced; this is what stops any other caller
        # having its request refused.
        asked_for = min(limit, MOST_IN_ONE_REQUEST)

        body = get_json(
            connection,
            SEARCH_API,
            source="semantic scholar",
            params={"query": query, "limit": str(asked_for),
                    "fields": _FIELDS,
                    **_the_years_asked_for(from_year, to_year)},
            limiter=limiter,
        )
        if isinstance(body, FetchError):
            return body

        # Reported the way an unreadable reply is, rather than read into and
        # raised on. A raise here would end the run over one search and lose
        # the sources that had answered.
        problem = _why_the_search_reply_is_not_readable(body)
        if problem is not None:
            reason = f"the reply was not put together as expected: {problem}"
            return FetchError("semantic scholar", SEARCH_API, 200, reason, 1)

        records = []
        for paper in body["data"]:
            records.append(_read_record(paper))
        return records
    finally:
        # Close the connection only if it was opened here. A supplied
        # connection may still be in use elsewhere.
        if opened_own_connection:
            connection.close()
