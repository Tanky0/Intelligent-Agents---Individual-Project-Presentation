"""Saves the records a reviewer approved, and refuses the rest (SC-02)."""
# The store is a SQLite file, as the design names and the service and data
# boundaries in `docs/development.md` record. It sits in `data/` inside the
# project folder and git is told to ignore it, so it stays on the computer
# running the system without being in the repository. Nothing leaves the
# machine.
#
# Two different refusals live here, and they are different on purpose.
#
# A record the reviewer has not approved is simply not saved. Waiting, flagged
# and rejected are all answers, and none of them is approval. There is nothing
# wrong with such a record; the store is for the ones a person decided to keep.
#
# A record the reviewer approved whose extract is not in its paper is refused,
# and that refusal is reported rather than passed over. Approving is the
# reviewer's to do, and storing something unverifiable is not: a stored claim is
# meant to be traceable back to the sentence it rests on, and a claim whose
# sentence is not in the paper cannot be. A reviewer can approve a paper without
# having checked every quotation in it, so this is the check that does not
# depend on their attention (SC-02).
#
# That check is made here rather than read off the record. `extract_found` says
# what was true when the run ended, and a judgement can be edited afterwards: an
# earlier version of this file trusted the flag, and saved a claim whose extract
# had been replaced by a sentence from nowhere. The flag is still refused when
# it says the extract was not found or was never looked for, because a record
# that failed the check or never had one should not reach the store whatever it
# now says.
#
# What is stored is the identifier, the description of the paper, the judgement,
# and the summary and extract the claim rests on. Those last two are the paper's
# own words, and they are kept because a claim that cannot be traced back to a
# sentence is what the check on every accepted claim exists to prevent (SC-02).
# What is left out is the whole abstract: the boundaries in
# `docs/development.md` send it to the buffer, and the store keeps a bounded
# supporting extract, so keeping the field itself would put the text of papers
# into a file that is copied and inspected.
#
# A paper is stored under one identifier and remembers the others. A DOI names
# one paper and nothing else, so it is preferred, then the PubMed identifier,
# then Semantic Scholar's. This order is the store's own decision, recorded in
# the traceability matrix. The order deduplication works in is about something
# else: it says what deduplication may match two records on, which is a DOI, a
# PubMed identifier, an exact title and a similar title, and a title is not
# something a stored claim can be traced back by.
#
# The other identifiers are kept because a later run can learn one the first did
# not know. Storing only the chosen one made a paper approved as a PubMed record
# and later approved with a DOI into two rows, which is one paper counted twice
# in the reconciliation and two claims where there was one. Keeping them also
# means a sparser record does not blank what the store already knew.
#
# Each identifier is matched against its own kind. Two identifiers of different
# kinds are not comparable, so a value that appears as one paper's PubMed
# identifier and another's Semantic Scholar identifier is two papers and not
# one. What is not handled is two papers whose preferred identifiers are spelt
# identically across kinds: the store would refuse the second on its primary
# key rather than merge them, which is a safe failure and not a graceful one.
#
# A decision can be withdrawn. A paper the reviewer has since rejected, flagged
# or left waiting is removed from the store when that decision is saved, so the
# store holds what they last decided rather than everything they ever approved.
# Leaving it would make the store a record of approvals rather than of the
# review, and the reconciliation would then be right to report a disagreement
# that was really the reviewer changing their mind.

import sqlite3
from contextlib import closing

from research_agent.config import REPO_ROOT
from research_agent.identifiers import normalise_doi
from research_agent.processing.validation import is_in_the_paper

# The store lives beside the run rather than in a folder somebody has to be
# told about. It is inside the project folder and ignored by git, and a public
# export is built from an allowlist that does not name it (R-12).
DEFAULT_STORE = REPO_ROOT / "data" / "approved_records.sqlite3"

# One row for each approved paper. The identifier is what a claim is traced
# back by, so it is the key rather than a number this file would have to
# invent. The other two are kept alongside so a paper approved twice under
# different identifiers is recognised as the same paper.
_TABLE = """
CREATE TABLE IF NOT EXISTS approved_papers (
    identifier TEXT PRIMARY KEY,
    doi TEXT NOT NULL,
    pmid TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT NOT NULL,
    year TEXT NOT NULL,
    journal TEXT NOT NULL,
    publication_status TEXT NOT NULL,
    decision TEXT NOT NULL,
    relevance INTEGER NOT NULL,
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    extract TEXT NOT NULL,
    review_note TEXT NOT NULL
)
"""


class RefusedByStorage(Exception):
    """An approved record that storage will not keep."""
    # Raised rather than returned. The caller asked for something to be saved
    # and it was not, and a return value saying so is easy to write code that
    # ignores. This is the last check before a claim becomes part of the working
    # record, so it is the wrong place to be quiet (SC-02).


def _identifiers_of(paper):
    """The paper's identifiers, in the order a claim should be traced by."""
    # A DOI first, because it names one paper and nothing else. The PubMed
    # identifier next, then Semantic Scholar's own. The DOI is normalised the
    # same way deduplication normalises it, so the same paper does not reach the
    # store under two spellings of one DOI.
    return [value for value in (normalise_doi(paper.doi).strip(),
                                paper.pmid.strip(),
                                paper.paper_id.strip()) if value]


def _naming(paper):
    """How a refusal names the paper it is about."""
    # By identifier rather than by title. A refusal is written to the run log as
    # well as shown on the screen, and the log deliberately holds none of a
    # paper's own words. The identifier is what a reader follows anyway.
    identifiers = _identifiers_of(paper)
    if identifiers:
        return repr(identifiers[0])
    return "a paper with no identifier"


def _preferred(known):
    """The one identifier a claim is traced back by."""
    # The first that is not empty, in the order the module docstring gives: a
    # DOI names one paper and nothing else, so it comes before the PubMed
    # identifier, and that before Semantic Scholar's.
    for column in ("doi", "pmid", "paper_id"):
        if known[column]:
            return known[column]
    return ""


def _identifiers_after_merging(found, paper):
    """What the stored row should know once this record is added to it."""
    # A record that carries only a PubMed identifier is not saying the paper has
    # no DOI, so a stored DOI is kept rather than blanked. Overwriting the whole
    # row with what the newest record happened to carry lost identifiers the
    # store already had, and a paper approved again from a sparser record came
    # back as a second row.
    #
    # A record whose identifier of some kind disagrees with the stored one is
    # refused rather than allowed to overwrite it. Two different DOIs are two
    # different papers, or one of the two records is wrong, and neither is
    # something a store should decide by taking the newer value.
    known = {"doi": "", "pmid": "", "paper_id": ""}
    if found is not None:
        known = {"doi": found[1], "pmid": found[2], "paper_id": found[3]}

    for column, value in _identifiers_by_kind(paper).items():
        if known[column] and known[column] != value:
            # A stored value can only be here when a row was found, so the
            # row's own identifier is what names the paper.
            raise RefusedByStorage(
                f"the paper stored as {found[0]!r} carries {column} {value!r}, "
                f"and the store holds {known[column]!r} for that paper. Two "
                "identifiers of one kind are two papers or one mistake, and "
                "storage is not the place to choose between them")
        known[column] = value
    return known


def _the_store(store):
    """The file to use: the one named, or the default looked up now."""
    # Looked up when called rather than fixed as a default argument, the way
    # the run log looks up its file. A default argument is fixed when the file
    # is read, so a launcher or a test that moves the store by replacing
    # DEFAULT_STORE would move only the functions written this way and leave
    # the others on the real file. That happened on 17 September: the launcher
    # for the invented papers wrapped `save_approved` to write into its own
    # folder and left DEFAULT_STORE alone, `every_row` looked the default up
    # and read the real store, and the results page listed a paper just
    # approved as withdrawn. One name for the file, read at the moment of use
    # by every function here, is what stops the write and the read parting.
    if store is None:
        return DEFAULT_STORE
    return store


def _connection(store):
    store.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(store)
    connection.execute(_TABLE)
    return connection


def _row_already_holding(connection, paper):
    """The stored row for this paper, if there is one."""
    # Each identifier is matched against its own kind only. Matching any value
    # against any column would make a DOI that happened to read like a PubMed
    # identifier join two different papers, and identifiers of different kinds
    # are not comparable in the first place.
    #
    # Every identifier is looked up, not only the first that matches. Stopping
    # at the first left a record carrying a DOI already stored and a PubMed
    # identifier belonging to a different stored paper: it updated the DOI row
    # and gave two rows the same PubMed identifier, which is the state the
    # conflict check exists to prevent. Matching two rows means the records
    # disagree about which papers are which, and that is refused here rather
    # than resolved by choosing one.
    rows = {}
    for column, value in _identifiers_by_kind(paper).items():
        # The column name is put into the statement rather than passed as a
        # value, because a placeholder can carry a value and not the name of a
        # column. It is safe here because the three names come from this file
        # and never from a record.
        for found in connection.execute(
                f"SELECT identifier, doi, pmid, paper_id FROM approved_papers "
                f"WHERE {column} = ?",
                (value,)).fetchall():
            rows[found[0]] = found

    if len(rows) > 1:
        raise RefusedByStorage(
            f"a paper being saved matches {len(rows)} stored papers, "
            f"{sorted(rows)}. Its identifiers say those are one paper and the "
            "store says they are several, and storage is not the place to "
            "decide which is right")
    if not rows:
        return None
    return list(rows.values())[0]


def _identifiers_by_kind(paper):
    """The paper's identifiers, by the column each belongs in."""
    # Empty ones are left out. A record that does not carry a DOI is not saying
    # the paper has none, so it must not be treated as a paper whose DOI is the
    # empty string, or every such record would match every other.
    values = {
        "doi": normalise_doi(paper.doi).strip(),
        "pmid": paper.pmid.strip(),
        "paper_id": paper.paper_id.strip(),
    }
    return {column: value for column, value in values.items() if value}


def _checked(screened):
    """The approved records, refusing any that could not be traced or verified."""
    keeping = []
    for item in screened:
        if item.review != "approved":
            continue
        if item.extract_found is None:
            raise RefusedByStorage(
                f"the extract for {_naming(item.paper)} was never looked for in "
                "the paper, and a check that did not run is not a check that "
                "passed")
        if not item.extract_found:
            raise RefusedByStorage(
                f"the extract for {_naming(item.paper)} was not found in the "
                "paper, so the claim cannot be traced back to it")
        if not is_in_the_paper(item.screening.extract, item.paper):
            raise RefusedByStorage(
                f"the extract now on {_naming(item.paper)} is not in the paper, "
                "whatever the record of the earlier check says")
        if not _identifiers_of(item.paper):
            raise RefusedByStorage(
                "a paper the reviewer approved carries no DOI, PubMed identifier or "
                "Semantic Scholar identifier, so a stored claim could not be "
                "traced back to it")
        keeping.append(item)
    return keeping


def save_approved(screened, store=None):
    """Saves every approved record, withdraws the rest, and returns how many."""
    # Refuses the whole batch rather than saving part of it. A batch half saved
    # would leave the reviewer's decisions and the store disagreeing, with
    # nothing saying where the disagreement starts, and the reconciliation is
    # the check that the two match (SC-04).
    keeping = _checked(screened)
    withdrawing = [item for item in screened if item.review != "approved"]

    # The connection is its own second context here, which is what makes the
    # whole batch one write: either every row lands or none does, and the
    # store never holds part of a reviewer's decisions.
    with closing(_connection(_the_store(store))) as connection, connection:
        for item in withdrawing:
            found = _row_already_holding(connection, item.paper)
            if found is not None:
                # Checked before deleting, not only before writing. A record
                # naming a different DOI with the same PubMed identifier
                # matched the stored row and deleted it, so a disagreement
                # between two records became a reviewer's decision being
                # thrown away.
                _identifiers_after_merging(found, item.paper)
                connection.execute(
                    "DELETE FROM approved_papers WHERE identifier = ?", (found[0],))

        for item in keeping:
            paper = item.paper
            found = _row_already_holding(connection, paper)
            known = _identifiers_after_merging(found, paper)
            values = (
                _preferred(known),
                known["doi"],
                known["pmid"],
                known["paper_id"],
                paper.title,
                "; ".join(paper.authors),
                paper.year,
                paper.journal,
                paper.publication_status,
                item.screening.decision,
                item.screening.relevance,
                item.screening.reason,
                item.screening.summary,
                item.screening.extract,
                item.review_note,
            )
            if found is None:
                connection.execute(
                    "INSERT INTO approved_papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values)
            else:
                connection.execute(
                    """
                    UPDATE approved_papers SET
                        identifier=?, doi=?, pmid=?, paper_id=?, title=?,
                        authors=?, year=?, journal=?, publication_status=?,
                        decision=?, relevance=?, reason=?, summary=?,
                        extract=?, review_note=?
                    WHERE identifier=?
                    """,
                    (*values, found[0]))
    return len(keeping)


def stored_identifiers(store=None):
    """Every identifier the store holds a paper under."""
    # This is what the reconciliation compares with the reviewer's decisions
    # (SC-04). Reading it back from the file rather than remembering what was
    # written is the point: a count kept in the application would agree with
    # itself whatever the file held.
    with closing(_connection(_the_store(store))) as connection:
        return {row[0] for row in connection.execute(
            "SELECT identifier FROM approved_papers")}


def every_row(store=None):
    """Every stored paper, each as a dictionary keyed by the table's columns."""
    # Read for the results screen, which shows the approved papers again after
    # the session that approved them has gone (`processing/results.py`). The
    # reading sits here so that the table's columns are named in one file.
    store = _the_store(store)
    # A store that does not exist holds nothing, and reading is not a reason
    # to make one: the live run starts from no store at all, and a look at an
    # earlier run's outcome must not put an empty file where the run will
    # look for the absence of one.
    if not store.exists():
        return []
    with closing(_connection(store)) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT * FROM approved_papers")]


def what_does_not_reconcile(screened, store=None):
    """The approved papers missing from the store, and the extra ones in it."""
    # Returns the two sets, in that order. The reconciliation asks for a full
    # match between approved decisions and stored records, and a single true or
    # false would say the two disagree without saying how, which is the part
    # somebody correcting it needs (SC-04).
    #
    # It compares which papers are there, not what is written about them. A row
    # whose judgement or extract was changed after it was stored has the same
    # identifier and is reported as agreeing. Catching that needs the stored
    # values read back and compared, which is worth building when there is a
    # reviewed run to compare against.
    #
    # The comparison is against everything the store holds, not against one
    # run's share of it. Approving papers in a second run therefore reports the
    # first run's papers as extra, which is right when a store is meant to hold
    # one review and misleading when it is meant to accumulate. Which of those
    # it is has not been settled: the screen writes to the store when a
    # reviewer saves their decisions (`ui/app.py`), but no second run has yet
    # been reviewed against a store the first one filled.
    approved = set()
    for item in screened:
        if item.review == "approved":
            identifiers = _identifiers_of(item.paper)
            if identifiers:
                approved.add(identifiers[0])
    held = stored_identifiers(store)
    return approved - held, held - approved
