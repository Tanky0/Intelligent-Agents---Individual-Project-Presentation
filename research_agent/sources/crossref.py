"""Looks up a paper's DOI and checks whether it has been retracted."""
# Crossref needs no account and no key. Instead, requests that include a contact
# email address are handled by a separate queue that is served more reliably.
# That address is added to every request by config.user_agent.
#
# Requests start at the rate Crossref publishes. Crossref also returns its
# current limit with every reply, and the shared request handling reads that
# figure and adjusts the spacing, so a limit that changes is followed rather than
# one written into this file that would quietly become wrong.
#
# The publication status has three possible values rather than two. Crossref only
# knows about a retraction if the publisher has sent one in, so finding no notice
# does not prove that a paper is sound. A true or false answer would force a
# missing notice to be reported as "not retracted", which would turn absent
# information into a claim about the paper. That is the same hidden failure the
# design set out to prevent, the one Page et al. (2021) describe (DC-07).

from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote

import httpx

from research_agent import schemas
from research_agent.config import settings
from research_agent.identifiers import normalise_doi
from research_agent.sources.http import FetchError, get_json, limiter_for

CROSSREF_API = "https://api.crossref.org/works"

# The number of requests per second published by Crossref.
_REQUESTS_PER_SECOND = 3.0

class PublicationStatus(Enum):
    """The three possible answers, for the reason given at the top of the file."""
    # Each answer is written so that it explains itself wherever it appears. The
    # last one says what was looked for and not found, rather than making a claim
    # about the paper.
    #
    # The wording is taken from schemas.py rather than written here, because the
    # paper record only accepts these exact sentences. Written twice, one copy
    # could be edited and the other not, and a status this source produced would
    # then be refused by the record it was meant for.

    RETRACTED = schemas.RETRACTED
    CONCERNS_RAISED = schemas.CONCERNS_RAISED
    NONE_IDENTIFIED = schemas.NONE_IDENTIFIED


@dataclass
class CrossrefRecord:
    doi: str
    title: str | None
    status: PublicationStatus
    update_notices: list


def _why_the_status_reply_is_not_readable(body):
    """Returns what is wrong with the shape of a status reply, or None."""
    # A reply that parsed is not yet a reply this can read. Only the fields this
    # file reads are checked: the message record, the notices in it, and the
    # title.
    #
    # Two things may be absent, and the reason is what the code does with the
    # absence, not what Crossref is known to send. A message with no notice
    # list is read as a paper with no notices, which is what most papers are,
    # and _classify already words that as none identified rather than as not
    # retracted. A message with no title is read as a paper whose title is not
    # known. Neither absence hides anything the reviewer would have been told.
    #
    # A notice that is present has to say what kind it is, as non-blank text.
    # The two ways it can fail to were read differently before and both were
    # wrong. A notice with its kind missing or blank was read as no notice at
    # all, quietly, and nothing can say what such a notice meant, only that it
    # was there and was not read. A notice whose kind was not text raised in
    # _classify, which lowers it. Both are refused here so that _classify does
    # not have to know about either.
    #
    # A title that is text rather than a list is refused rather than read,
    # because reading it would take only its first character as the title.
    # That is the silent kind of wrong, which nothing downstream would notice.
    if not isinstance(body, dict):
        return "not a record"
    message = body.get("message")
    if not isinstance(message, dict):
        return "no message record"
    notices = message.get("update-to")
    if notices is not None:
        if not isinstance(notices, list):
            return "notices that are not a list"
        for notice in notices:
            if not isinstance(notice, dict):
                return "a notice that is not a record"
            kind = notice.get("type")
            if not isinstance(kind, str) or not kind.strip():
                return "a notice with no kind as text"
    titles = message.get("title")
    if titles is not None:
        if not isinstance(titles, list):
            return "a title that is not a list"
        for title in titles:
            if not isinstance(title, str):
                return "a title that is not text"
    return None


def _classify(message):
    """Return the publication status indicated by Crossref notices."""
    notices = []
    for update in message.get("update-to") or []:
        # Crossref usually returns a list in this field, but it may be
        # missing or contain an unexpected value.
        if isinstance(update, dict):
            notices.append(update.get("type", "").lower())

    # A paper carrying both a concern and a retraction has been retracted. Two
    # loops ensure that every notice is checked first for evidence that the
    # paper was retracted after publication. With one loop, the function could
    # return a concern before finding a later notice showing that it was
    # retracted.
    for notice in notices:
        if "retraction" in notice:
            return PublicationStatus.RETRACTED, notices

    for notice in notices:
        if "concern" in notice:
            return PublicationStatus.CONCERNS_RAISED, notices

    return PublicationStatus.NONE_IDENTIFIED, notices


def _open_connection():
    """A connection to Crossref that identifies this application on every request."""
    # Made in one place because two callers open one: a lookup of a single DOI
    # and the lookup of a whole list at the hand-over. Written twice, one copy
    # could lose the contact address and its requests would go to the queue
    # Crossref serves less reliably, with nothing saying why they had slowed.
    return httpx.Client(
        headers={"User-Agent": settings.user_agent("crossref")}, timeout=30.0
    )


def fetch_status(doi, connection=None, limiter=None):
    """Returns the record for one DOI, or a description of what went wrong."""
    # Tests can provide a simulated Crossref connection, allowing the function
    # to run without network access. If no connection is provided, the function
    # opens one.
    clean_doi = normalise_doi(doi)

    opened_own_connection = connection is None
    if connection is None:
        connection = _open_connection()
    if limiter is None:
        limiter = limiter_for("crossref", _REQUESTS_PER_SECOND)

    try:
        # A DOI suffix may contain characters that mean something else in a
        # web address, and Crossref names the number sign among them. They are
        # encoded so the address asks for the whole DOI rather than stopping at
        # one of those characters. The slash between the prefix and the suffix
        # is left alone, because it is part of the address.
        asked_for = quote(clean_doi, safe="/")
        body = get_json(connection, f"{CROSSREF_API}/{asked_for}", source="crossref",
                        limiter=limiter)
        if isinstance(body, FetchError):
            return body

        # Reported the way an unreadable reply is, rather than read into and
        # raised on. Status is a warning shown to the reviewer, not something
        # a run cannot do without, so a raise here would be the wrong size of
        # failure.
        problem = _why_the_status_reply_is_not_readable(body)
        if problem is not None:
            reason = f"the reply was not put together as expected: {problem}"
            return FetchError("crossref", f"{CROSSREF_API}/{asked_for}", 200,
                              reason, 1)
        message = body["message"]
        status, notices = _classify(message)

        titles = message.get("title")
        if titles:
            title = titles[0]
        else:
            title = None

        return CrossrefRecord(clean_doi, title, status, notices)
    finally:
        # Close the connection only if it was opened here. A supplied
        # connection may still be in use elsewhere.
        if opened_own_connection:
            connection.close()


def fetch_statuses(dois, connection=None, limiter=None):
    """The answer for each DOI in the list, by DOI: a record, or what went wrong."""
    # Written for the hand-over, which asks about every paper curation is given
    # (R-18). Up to fifty papers, so one connection is opened for the lot rather
    # than one for each: fetch_status opens and closes its own when it is given
    # none, and every connection opened starts again with the set-up a request
    # needs before anything is sent, which a connection kept open reuses from
    # one lookup to the next. That is reuse, not a promise of one physical
    # connection for the whole list: the connection underneath can be reopened.
    #
    # A DOI that appears twice in the list is asked about once. Deduplication
    # keeps two records with one DOI apart when their PubMed identifiers
    # disagree, so the hand-over can send the same DOI twice, and asking twice
    # replaced the first answer with the second: a retraction found by the first
    # lookup was erased when the second failed, and both records went to the
    # reviewer blank. The answer found once is given to every record carrying
    # the DOI; the records themselves stay apart.
    #
    # A failure is kept in the answer beside the records rather than dropped,
    # because the caller reports a check that failed. A paper whose check failed
    # and a paper Crossref was never asked about both carry no status, and only
    # the failure being handed back lets the report tell the two apart.
    #
    # The limiter is the one every Crossref request shares, so the lookups keep
    # to the published rate however many there are. That is what bounds the
    # cost of this call: at most one lookup for each paper, three attempts each
    # at most, spaced by the limiter, and none of them a model request (OB-07).
    #
    # A list with nothing in it opens no connection. A run whose papers have no
    # DOI has nothing to ask and must not touch the network to find that out.
    if not dois:
        return {}

    opened_own_connection = connection is None
    if connection is None:
        connection = _open_connection()
    if limiter is None:
        limiter = limiter_for("crossref", _REQUESTS_PER_SECOND)

    answers = {}
    try:
        for doi in dois:
            if doi in answers:
                continue
            answers[doi] = fetch_status(doi, connection, limiter)
    finally:
        if opened_own_connection:
            connection.close()
    return answers
