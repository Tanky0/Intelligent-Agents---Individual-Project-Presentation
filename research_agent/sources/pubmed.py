"""Retrieves paper records from PubMed."""
# Retrieval uses two PubMed API calls. The first searches PubMed and returns
# paper identifiers. The second submits those identifiers in the request body
# and retrieves the corresponding metadata and available abstracts.
#
# Submitting the identifiers in the request body avoids the incomplete responses
# observed when long identifier lists were placed in the URL (R-02).
#
# Each record states whether its searchable text contains an abstract or only
# bibliographic metadata. The Curation Agent will use this to avoid screening a
# record that has only a title as though it had read an abstract. Because even an
# abstract provides only part of the evidence contained in a full paper, the
# agent works in what Wooldridge (2009) terms an inaccessible environment, unable
# to obtain everything a fully informed decision would need. Recording the
# available text type makes that limitation explicit (DC-07).

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

from research_agent.config import settings
from research_agent.schemas import EARLIEST_YEAR, LATEST_YEAR
from research_agent.sources.http import FetchError, get_json, limiter_for, post_text

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Long identifier lists are sent in batches. NCBI asks that a request carrying
# more than about two hundred identifiers be sent as POST, which these are, so
# a hundred is comfortably inside what the service expects. It was ten while
# the truncated replies were being dealt with, and raising it cuts the requests
# needed for eight hundred papers from eighty-three to nine. That is what keeps
# a recall measurement below the hundred attempts in one run that this project
# treats as a large job, which NCBI asks be run off-peak; the shared request
# handling refuses the rest of a run that reaches that number (OB-04).
BATCH_SIZE = 100


@dataclass
class PubmedRecord:
    pmid: str
    title: str
    abstract: str
    authors: list = field(default_factory=list)
    year: str = ""
    journal: str = ""
    publication_types: list = field(default_factory=list)
    doi: str = ""
    # Use "abstract" when abstract text is available; otherwise use "metadata".
    text_source: str = "metadata"
    source: str = "pubmed"


# The ends of a range when only one year was asked for. NCBI requires mindate
# and maxdate together, so a search limited at one end supplies this project's
# supported endpoint for the other. They are not a claim about what exists;
# they are the same two years ResearchRequest will accept, taken from there so
# that the request and the search cannot disagree about the range.


def _the_years_asked_for(from_year, to_year):
    """The date values NCBI wants, or nothing when no years were asked for."""
    # datetype=pdat makes the intended date field explicit, which is the
    # publication date rather than any other date NCBI holds for a record.
    if from_year is None and to_year is None:
        return {}
    return {
        "datetype": "pdat",
        "mindate": str(from_year if from_year is not None else EARLIEST_YEAR),
        "maxdate": str(to_year if to_year is not None else LATEST_YEAR),
    }


def _request_values(extra):
    """Adds the details NCBI wants on every request to the ones given."""
    # NCBI asks each caller to name itself and give a contact address, and the
    # key raises the number of requests allowed each second.
    values = {
        "api_key": settings.ncbi_api_key,
        "email": settings.ncbi_email,
        "tool": settings.ncbi_tool,
    }
    for name, value in extra.items():
        values[name] = value
    return values


def _new_connection():
    """Opens a connection that names this application to NCBI."""
    return httpx.Client(headers={"User-Agent": settings.user_agent("pubmed")},
                        timeout=60.0)


def _limiter():
    """Returns the limiter shared by everything that talks to NCBI."""
    # The search and the fetch are one service with one allowance, so both ask
    # for the same limiter rather than keeping one each. The rate is read here
    # rather than fixed above, because it depends on whether a key is set.
    return limiter_for("ncbi", settings.ncbi_rate_limit)


def _text_of(element):
    """Return all text from an element, including text inside nested tags."""
    # Titles and abstracts may contain markup such as <i> and <sup>. Using .text
    # would return only the text before the first nested tag and silently omit
    # the remaining content.
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _read_abstract(article):
    """Return the complete abstract by joining its labelled sections."""
    # PubMed may store structured abstracts as separate AbstractText elements,
    # such as Background, Methods and Results. Reading only the first element
    # would lose the remaining sections of the abstract.
    parts = []
    for tag in article.findall(".//AbstractText"):
        text = _text_of(tag)
        if text:
            parts.append(text)
    return " ".join(parts)


def _read_authors(article):
    names = []
    for author in article.findall(".//Author"):
        surname = author.findtext("LastName", "")
        initials = author.findtext("Initials", "")
        if surname:
            names.append(f"{surname} {initials}".strip())
    return names


def _read_doi(article):
    for identifier in article.findall(".//ArticleId"):
        if identifier.get("IdType") == "doi":
            return (identifier.text or "").strip()
    return ""


def _read_publication_types(article):
    """Returns the kinds of paper this is, such as Review or Clinical Trial."""
    kinds = []
    for tag in article.findall(".//PublicationType"):
        kinds.append(tag.text or "")
    return kinds


def _read_article(article):
    """Turns one paper from the reply into a record."""
    abstract = _read_abstract(article)

    if abstract:
        text_source = "abstract"
    else:
        text_source = "metadata"

    return PubmedRecord(
        pmid=article.findtext(".//PMID", ""),
        title=_text_of(article.find(".//ArticleTitle")),
        abstract=abstract,
        authors=_read_authors(article),
        year=article.findtext(".//PubDate/Year", ""),
        journal=article.findtext(".//Journal/Title", ""),
        publication_types=_read_publication_types(article),
        doi=_read_doi(article),
        text_source=text_source,
    )


def _why_the_search_reply_is_not_readable(body):
    """Returns what is wrong with the shape of a search reply, or None."""
    # A reply that parsed is not yet a reply this can read. The identifier list
    # has to be there: a result record that gives a count and carries no list
    # is exactly the reply that must not be read as nothing found, and an empty
    # list is how nothing found is written.
    #
    # Each identifier is checked for being text because fetch joins them into
    # one request, and a number would fail there, further from the reply that
    # caused it and with a message about joining rather than about the reply.
    if not isinstance(body, dict):
        return "not a record"
    result = body.get("esearchresult")
    if not isinstance(result, dict):
        return "no esearchresult record"
    identifiers = result.get("idlist")
    if not isinstance(identifiers, list):
        return "no list of identifiers"
    for identifier in identifiers:
        if not isinstance(identifier, str):
            return "an identifier that is not text"
    return None


def search(query, limit=50, connection=None, limiter=None,
           from_year=None, to_year=None):
    """Returns the paper identifiers matching a search, or a description of
    what went wrong. Check which of the two came back before using it, or pass
    it straight to fetch, which reports the failure rather than breaking.
    """
    # The years are turned into this service's own date values here, because
    # each service words them differently and this is the file that knows how
    # NCBI does it. Sending a filter does not by itself prove it was applied
    # correctly, so the caller checks the years of what comes back.
    opened_own_connection = connection is None
    if connection is None:
        connection = _new_connection()
    if limiter is None:
        limiter = _limiter()

    try:
        body = get_json(
            connection,
            f"{EUTILS}/esearch.fcgi",
            source="pubmed",
            params=_request_values({"db": "pubmed", "term": query,
                                    "retmax": str(limit), "retmode": "json",
                                    **_the_years_asked_for(from_year, to_year)}),
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
            return FetchError("pubmed", f"{EUTILS}/esearch.fcgi", 200,
                              reason, 1)
        return body["esearchresult"]["idlist"]
    finally:
        # Close the connection only if it was opened here. A supplied
        # connection may still be in use elsewhere.
        if opened_own_connection:
            connection.close()


def fetch(pmids, connection=None, limiter=None):
    """Return the records retrieved, and any batches that failed."""
    # Separating the two keeps a failure to its own group of papers, and leaves
    # it in a form that can be counted and reported.
    # search returns either a list of identifiers or a description of a
    # failure. Passing the failure straight on is an easy mistake, so it is
    # turned into an empty result and a reported error rather than a confusing
    # complaint about the wrong kind of value.
    if isinstance(pmids, FetchError):
        return [], [pmids]

    opened_own_connection = connection is None
    if connection is None:
        connection = _new_connection()
    if limiter is None:
        limiter = _limiter()

    records = []
    errors = []

    try:
        for start in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[start:start + BATCH_SIZE]
            text = post_text(
                connection,
                f"{EUTILS}/efetch.fcgi",
                source="pubmed",
                data=_request_values({"db": "pubmed", "id": ",".join(batch),
                                      "retmode": "xml"}),
                limiter=limiter,
            )
            if isinstance(text, FetchError):
                errors.append(text)
                continue

            try:
                reply = ET.fromstring(text)
            except ET.ParseError as problem:
                # A reply that is not readable XML is a failed batch, not a
                # batch of no papers. Letting it stop the run would lose the
                # batches that did work.
                errors.append(FetchError("pubmed", f"{EUTILS}/efetch.fcgi", 200,
                                         f"the reply could not be read: {problem}",
                                         1))
                continue

            asked_for = set(batch)
            returned = set()
            unexpected = []
            repeated = []

            for article in reply.findall(".//PubmedArticle"):
                record = _read_article(article)

                # A paper nobody asked for is not kept. It would enter the
                # buffer with no query behind it, so a reviewer asking why it
                # is in front of them could not be answered.
                if record.pmid not in asked_for:
                    # Counted once however many times it arrives, so the number
                    # reported is papers rather than occurrences.
                    if record.pmid not in unexpected:
                        unexpected.append(record.pmid)
                    continue

                # The same paper twice in one reply is kept once. Counting it
                # twice would make the coverage report say more was found than
                # there is.
                if record.pmid in returned:
                    if record.pmid not in repeated:
                        repeated.append(record.pmid)
                    continue

                returned.add(record.pmid)
                records.append(record)

            if unexpected:
                errors.append(FetchError(
                    "pubmed", f"{EUTILS}/efetch.fcgi", 200,
                    f"the reply held {len(unexpected)} papers that were not asked "
                    f"for, and they were not kept: {', '.join(unexpected)}", 1))

            if repeated:
                errors.append(FetchError(
                    "pubmed", f"{EUTILS}/efetch.fcgi", 200,
                    f"the reply held {len(repeated)} papers more than once, and "
                    f"each was kept once: {', '.join(repeated)}", 1))

            # A reply can be complete-looking and still hold fewer papers than
            # were asked for. Nothing in it says so, and the papers simply do
            # not appear. The identifiers asked for are therefore compared with
            # the ones that came back, because a search that quietly returns
            # less than it found is the failure this design cares most about,
            # and it would lower the recall figure without anything explaining
            # why (R-04).
            missing = []
            for pmid in batch:
                if pmid not in returned:
                    missing.append(pmid)

            if missing:
                errors.append(FetchError(
                    "pubmed", f"{EUTILS}/efetch.fcgi", 200,
                    f"{len(missing)} of {len(batch)} papers asked for were not "
                    f"in the reply: {', '.join(missing)}", 1))
    finally:
        # Close the connection only if it was opened here. A supplied
        # connection may still be in use elsewhere.
        if opened_own_connection:
            connection.close()

    return records, errors
