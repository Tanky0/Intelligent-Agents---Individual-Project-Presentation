"""Combines repeated search records before screening."""
# Identifier matches are considered before title matches across the whole input.
# The original members remain available until grouping is finished: comparing
# only a merged record could hide a conflicting identifier behind a missing one.
# Sorting records before grouping makes ambiguous choices repeatable when the
# services return the same records in a different order.
#
# The result is a new list. The caller's records are left intact in the private
# buffer, because selecting one value cannot preserve every conflicting account
# of a paper. Origins survive, but that is not a claim that every field survives.
# This module neither screens papers nor imposes the screening allowance.

import json
import re

from rapidfuzz import fuzz

from research_agent.identifiers import normalise_doi
from research_agent.schemas import (
    CONCERNS_RAISED,
    NONE_IDENTIFIED,
    NOT_CHECKED,
    RETRACTED,
    PaperRecord,
)

# A choice, not a measured accuracy result, and the real records have not
# tested it: on the two runs read on 18 September (91 records from the
# standard-dataset retrieval of the 15th, 100 from the live run), every
# record carried a DOI or a PubMed identifier, so the identifier passes
# decided every pair and the title passes merged nothing at any value from
# 80 to 100, which the deduplication decision in `docs/development.md` records.
# The number is kept rather than moved on a guess. Requiring both word order
# and sorted-word similarity avoids treating reversed statements as identical
# merely because they contain the same words; matching year and first author
# provide further support, and
# the same reading found seventeen pairs of identical or near-identical
# titles that were different papers by their identifiers, every one with a
# different year or first author as well, which is what those guards are
# for. Uncertain pairs remain separate for the reviewer.
TITLE_SIMILARITY_THRESHOLD = 95

# Most serious first. The wording is read from the schema rather than written
# here, so that the order and the record cannot disagree about what a status
# is called.
_STATUS_ORDER = (RETRACTED, CONCERNS_RAISED, NONE_IDENTIFIED)


def _normalise_title(value):
    return " ".join(re.sub(r"[\W_]+", " ", value.casefold()).split())


def _record_key(paper):
    values = paper.model_dump(mode="json")
    values["origins"] = sorted(values["origins"], key=lambda item: (item["source"], item["query"]))
    return json.dumps(values, sort_keys=True, ensure_ascii=True)


def _identifiers_agree(members):
    for field in ("doi", "pmid"):
        values = {getattr(paper, field) for paper in members if getattr(paper, field)}
        if len(values) > 1:
            return False
    return True


def _title_metadata_conflicts(left, right):
    if left.year.strip() and right.year.strip() and left.year.strip() != right.year.strip():
        return True
    left_author = _normalise_title(left.authors[0]) if left.authors else ""
    right_author = _normalise_title(right.authors[0]) if right.authors else ""
    return bool(left_author and right_author and left_author != right_author)


def _titles_match(left, right, fuzzy):
    first = _normalise_title(left.title)
    second = _normalise_title(right.title)
    if not first or not second or _title_metadata_conflicts(left, right):
        return False
    if not fuzzy:
        return first == second
    year = left.year.strip()
    if not (len(year) == 4 and year.isascii() and year.isdigit() and year == right.year.strip()):
        return False
    if not left.authors or not right.authors:
        return False
    if not _normalise_title(left.authors[0]) or not _normalise_title(right.authors[0]):
        return False
    return (
        fuzz.token_sort_ratio(first, second) >= TITLE_SIMILARITY_THRESHOLD
        and fuzz.ratio(first, second) >= TITLE_SIMILARITY_THRESHOLD
    )


def _pair_matches(left, right, stage):
    if stage in ("doi", "pmid"):
        value = getattr(left, stage)
        return bool(value and value == getattr(right, stage))
    return _titles_match(left, right, fuzzy=stage == "similar title")


def _choose_field(members, field):
    candidates = []
    for paper in members:
        value = getattr(paper, field)
        if value and (not isinstance(value, str) or value.strip()):
            candidates.append(paper)
    if not candidates:
        return [] if field == "authors" else ""

    # The order is a repeatable choice rather than a judgement about the
    # papers: the value from a member that carries a DOI, then the longest,
    # then the first in wording, then the whole record. It exists so the merged
    # record reads the same whichever order the services returned the members
    # in. A list of authors is turned into text first so that its length and
    # its wording are counted the way a title's are, one rule for every field.
    def preference(paper):
        value = getattr(paper, field)
        if isinstance(value, list):
            text = json.dumps(value, ensure_ascii=True)
        else:
            text = value
        return (not bool(paper.doi), -len(text), text, _record_key(paper))

    return getattr(min(candidates, key=preference), field)


def _body_preference(paper):
    # Full text outranks an abstract, and an abstract outranks nothing, so the
    # member with the most of the paper supplies the body. Length, wording,
    # the licence and the whole record settle ties in that order, so the same
    # members give the same owner however the services ordered them.
    if paper.full_text:
        rank = 2
        body = paper.full_text
    elif paper.abstract:
        rank = 1
        body = paper.abstract
    else:
        rank = 0
        body = ""
    return (-rank, -len(body), body, paper.full_text_licence, _record_key(paper))


def _publication_status(members):
    """The most serious status any of the merged records carries."""
    # A retraction outranks a concern, and any answer from Crossref outranks
    # silence, so the merged record says the worst that was found and never
    # turns "not checked" into "nothing found".
    #
    # An earlier version also kept any wording it did not recognise, because
    # the field was free text and a notice nobody had listed could arrive in
    # it. The field now holds only the three answers and the empty string,
    # checked when the record is built, so an unrecognised wording cannot reach
    # here and there is nothing left to keep.
    statuses = {paper.publication_status for paper in members}
    for status in _STATUS_ORDER:
        if status in statuses:
            return status
    return NOT_CHECKED


def _merge(members):
    body_owner = min(members, key=_body_preference)
    values = {
        field: _choose_field(members, field)
        for field in ("title", "authors", "year", "journal", "pmid", "doi", "paper_id")
    }
    values["origins"] = sorted(
        [origin.model_dump() for paper in members for origin in paper.origins],
        key=lambda origin: (origin["source"], origin["query"]),
    )
    values["publication_types"] = sorted(
        {kind for paper in members for kind in paper.publication_types}
    )
    # The body owner's abstract is kept so the two texts stay together when
    # that member supplies both. If it has no abstract, whether it holds full
    # text alone or metadata alone, one is taken from another member, and a
    # group with no abstract in any member stays without one.
    if body_owner.abstract:
        values["abstract"] = body_owner.abstract
    else:
        values["abstract"] = _choose_field(members, "abstract")
    # A longer body never borrows another record's licence. Equal bodies use
    # the licence wording as a repeatable tie-break, not as a legal judgement.
    values["full_text"] = body_owner.full_text
    if body_owner.full_text:
        values["full_text_licence"] = body_owner.full_text_licence
    else:
        values["full_text_licence"] = ""
    if values["full_text"]:
        values["text_source"] = "full text"
    elif values["abstract"]:
        values["text_source"] = "abstract"
    else:
        values["text_source"] = "metadata"
    values["publication_status"] = _publication_status(members)
    return PaperRecord.model_validate(values)


def deduplicate(papers):
    """Keeps identifier evidence ahead of weaker title evidence across a run."""
    # Candidate pairs have a fixed order and groups retain their original
    # members. An identifier-free bridge can join the first compatible group
    # in that order, but cannot then bring a conflicting group into it. This is
    # a repeatable choice under ambiguity, not proof that the bridge belongs.
    records = []
    for paper in papers:
        values = paper.model_dump()
        values["doi"] = normalise_doi(paper.doi).strip()
        values["pmid"] = paper.pmid.strip()
        records.append(PaperRecord.model_validate(values))
    records.sort(key=_record_key)
    groups = [[paper] for paper in records]
    owners = list(range(len(records)))
    for stage in ("doi", "pmid", "exact title", "similar title"):
        for first, left in enumerate(records):
            for second in range(first + 1, len(records)):
                left_group, right_group = owners[first], owners[second]
                if left_group == right_group or not _pair_matches(left, records[second], stage):
                    continue
                members = groups[left_group] + groups[right_group]
                if not _identifiers_agree(members):
                    continue
                if stage in ("exact title", "similar title") and any(
                    _title_metadata_conflicts(a, b)
                    for a in groups[left_group] for b in groups[right_group]
                ):
                    continue
                keep, remove = sorted((left_group, right_group))
                groups[keep] = members
                groups[remove] = []
                owners = [keep if owner == remove else owner for owner in owners]
    return sorted((_merge(group) for group in groups if group), key=_record_key)
