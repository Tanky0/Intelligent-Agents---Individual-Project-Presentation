"""What reaches the store and what is refused (FR-14, SC-02, SC-04)."""
# Every test writes to a file pytest gives it, so none of them touches the store
# a real run would use.
#
# The refusals are the point. A store that quietly kept a claim whose extract
# was never found would look exactly like one that checked, and the reviewer
# reading it later could not tell which they had.

import pytest

from research_agent.processing.storage import (
    RefusedByStorage,
    save_approved,
    stored_identifiers,
    what_does_not_reconcile,
)
from research_agent.schemas import PaperRecord, ScreenedPaper, Screening, SearchOrigin


def a_paper(title="A trial of drug X", **changes):
    values = {
        "title": title,
        "pmid": "12345678",
        "authors": ["Smith J", "Jones A"],
        "year": "2024",
        "journal": "Journal of Trials",
        "abstract": "Forty patients were given drug X.",
        "text_source": "abstract",
        "origins": [SearchOrigin(source="pubmed", query="drug X")],
    }
    values.update(changes)
    return PaperRecord(**values)


def judged(paper=None, review="approved", extract_found=True, **changes):
    values = {
        "paper": paper or a_paper(),
        "screening": Screening(
            paper_number=1,
            decision="include",
            relevance=8,
            reason="it reports an outcome for drug X",
            summary="a trial of drug X in condition Y",
            extract="Forty patients were given drug X",
        ),
        "review": review,
        "extract_found": extract_found,
    }
    values.update(changes)
    return ScreenedPaper(**values)


def test_an_approved_record_is_saved(tmp_path):
    store = tmp_path / "records.sqlite3"

    assert save_approved([judged()], store) == 1
    assert stored_identifiers(store) == {"12345678"}


@pytest.mark.parametrize("review", ["waiting", "flagged", "rejected"])
def test_only_approved_records_reach_the_store(tmp_path, review):
    """Waiting, flagged and rejected are all answers, and none is approval."""
    store = tmp_path / "records.sqlite3"

    assert save_approved([judged(review=review)], store) == 0
    assert stored_identifiers(store) == set()


def test_an_approved_record_whose_extract_was_not_found_is_refused(tmp_path):
    """Approving is the reviewer's to do; storing the unverifiable is not."""
    # A reviewer can approve a paper without having checked every quotation in
    # it, so this is the check that does not depend on their attention (SC-02).
    store = tmp_path / "records.sqlite3"

    with pytest.raises(RefusedByStorage, match="was not found in the paper"):
        save_approved([judged(extract_found=False)], store)

    assert stored_identifiers(store) == set()


def test_an_approved_record_whose_extract_was_never_looked_for_is_refused(tmp_path):
    """Never looked for is not the same as looked for and found."""
    # Treating the first as approval would let a run that skipped the check
    # store everything it had.
    store = tmp_path / "records.sqlite3"

    with pytest.raises(RefusedByStorage, match="never looked for"):
        save_approved([judged(extract_found=None)], store)


def test_a_paper_with_no_identifier_at_all_is_refused(tmp_path):
    """A stored claim is meant to be traceable back to its paper."""
    store = tmp_path / "records.sqlite3"
    nameless = a_paper(pmid="", doi="", paper_id="")

    with pytest.raises(RefusedByStorage, match="carries no DOI"):
        save_approved([judged(paper=nameless)], store)


def test_the_doi_is_preferred_to_the_other_identifiers(tmp_path):
    """A DOI names one paper and nothing else, so it is the one to trace by."""
    store = tmp_path / "records.sqlite3"
    both = a_paper(doi="10.1000/abc", pmid="12345678")

    save_approved([judged(paper=both)], store)

    assert stored_identifiers(store) == {"10.1000/abc"}


def test_one_refusal_stops_the_whole_batch(tmp_path):
    """A batch half saved leaves the decisions and the store disagreeing."""
    # Nothing would say where the disagreement starts, and SC-04 is the check
    # that the two match.
    store = tmp_path / "records.sqlite3"
    good = judged(paper=a_paper("the sound one", pmid="1"))
    bad = judged(paper=a_paper("the unverifiable one", pmid="2"),
                 extract_found=False)

    with pytest.raises(RefusedByStorage):
        save_approved([good, bad], store)

    assert stored_identifiers(store) == set(), "not even the sound one"


def test_approving_the_same_paper_twice_keeps_one_record(tmp_path):
    """A reviewer may approve, change their mind, and approve again."""
    # The store keeps what they last decided rather than a row for each time
    # they pressed the button.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(review="approved")], store)

    again = judged(review="approved")
    again.screening.reason = "on second thoughts, it is the better trial"
    save_approved([again], store)

    assert stored_identifiers(store) == {"12345678"}


def test_an_extract_changed_after_the_check_ran_is_refused(tmp_path):
    """extract_found says what was true when the run ended."""
    # A judgement can be edited afterwards, and an earlier version of this file
    # trusted the flag and saved a claim whose extract had been replaced by a
    # sentence from nowhere. The paper is read again here rather than the flag
    # being believed.
    store = tmp_path / "records.sqlite3"
    edited = judged()
    edited.screening.extract = "Every patient recovered within a week"

    with pytest.raises(RefusedByStorage, match="whatever the record of the earlier"):
        save_approved([edited], store)

    assert stored_identifiers(store) == set()


def test_learning_a_doi_for_a_stored_paper_keeps_one_record(tmp_path):
    """A later run can know an identifier the first one did not."""
    # Storing only the chosen identifier made a paper approved as a PubMed
    # record, and later approved with a DOI, into two rows: one paper counted
    # twice in the reconciliation, and two claims where there was one.
    store = tmp_path / "records.sqlite3"
    save_approved([judged()], store)

    with_a_doi = judged(paper=a_paper(doi="10.1000/abc", pmid="12345678"))
    save_approved([with_a_doi], store)

    assert stored_identifiers(store) == {"10.1000/abc"}, \
        "one row, now traced by the identifier that names one paper"


def test_the_same_doi_written_two_ways_is_one_paper(tmp_path):
    """The DOI is normalised the way deduplication normalises it."""
    # Otherwise the same paper reaches the store under two spellings of one
    # identifier, which is the thing the identifier exists to prevent.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper(doi="https://doi.org/10.1000/ABC", pmid=""))],
                  store)
    save_approved([judged(paper=a_paper(doi="10.1000/abc", pmid=""))], store)

    assert stored_identifiers(store) == {"10.1000/abc"}


@pytest.mark.parametrize("withdrawn", ["rejected", "flagged", "waiting"])
def test_a_decision_the_reviewer_took_back_removes_the_stored_record(tmp_path, withdrawn):
    """The store holds what the reviewer last decided, not every approval."""
    # Left in, the store would be a record of approvals rather than of the
    # review, and the reconciliation would report a disagreement that was really
    # somebody changing their mind.
    store = tmp_path / "records.sqlite3"
    save_approved([judged()], store)

    save_approved([judged(review=withdrawn)], store)

    assert stored_identifiers(store) == set()


def test_a_sparser_record_does_not_lose_the_identifiers_already_known(tmp_path):
    """A record with no DOI is not saying the paper has none."""
    # Overwriting the row with whatever the newest record carried blanked the
    # stored DOI, so approving the same paper again from a PubMed-only record
    # came back as a second row.
    store = tmp_path / "records.sqlite3"
    for paper in [a_paper(pmid="111", doi=""),
                  a_paper(pmid="111", doi="10.1000/a"),
                  a_paper(pmid="", doi="10.1000/a"),
                  a_paper(pmid="111", doi="")]:
        save_approved([judged(paper=paper)], store)

    assert stored_identifiers(store) == {"10.1000/a"}, "one paper, one row"


def test_a_conflicting_identifier_is_refused_rather_than_overwriting(tmp_path):
    """Two different DOIs are two papers, or one of the records is wrong."""
    # Neither is something a store should settle by taking the newer value.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper(pmid="111", doi="10.1000/a"))], store)

    with pytest.raises(RefusedByStorage, match="two papers or one mistake"):
        save_approved([judged(paper=a_paper(pmid="111", doi="10.1000/other"))], store)

    assert stored_identifiers(store) == {"10.1000/a"}, "what was there is untouched"


def test_a_withdrawal_naming_a_different_paper_does_not_delete_the_stored_one(tmp_path):
    """The conflict check belongs before a delete as much as before a write."""
    # A rejection carrying a different DOI with the same PubMed identifier
    # matched the stored row and removed it, so a disagreement between two
    # records became a reviewer's approval being thrown away.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper(doi="10.1000/a", pmid="123"))], store)

    with pytest.raises(RefusedByStorage, match="two papers or one mistake"):
        save_approved([judged(paper=a_paper(doi="10.1000/b", pmid="123"),
                              review="rejected")], store)

    assert stored_identifiers(store) == {"10.1000/a"}, "the approval is still there"


def test_a_record_matching_two_stored_papers_is_refused(tmp_path):
    """Stopping at the first identifier that matched let two rows share one."""
    # A record carrying a DOI already stored and a PubMed identifier belonging
    # to a different stored paper updated the DOI row, and the store ended with
    # two rows holding the same PubMed identifier. The records and the store
    # disagree about which papers are which, and storage is not the place to
    # decide.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper("the first", doi="10.1000/a", pmid=""))],
                  store)
    save_approved([judged(paper=a_paper("the second", doi="10.1000/b",
                                        pmid="123"))], store)

    with pytest.raises(RefusedByStorage, match="matches 2 stored papers"):
        save_approved([judged(paper=a_paper("both at once", doi="10.1000/a",
                                            pmid="123"))], store)

    assert stored_identifiers(store) == {"10.1000/a", "10.1000/b"}


def test_an_identifier_is_matched_only_against_its_own_kind(tmp_path):
    """One paper's PubMed identifier is not another's Semantic Scholar one."""
    # Matching any value against any column would join two different papers
    # because two identifiers of different kinds happened to be spelt the same,
    # and identifiers of different kinds are not comparable in the first place.
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper("the first", pmid="111", doi=""))], store)
    save_approved([judged(paper=a_paper(
        "the second", pmid="", doi="10.1000/b", paper_id="111"))], store)

    assert stored_identifiers(store) == {"111", "10.1000/b"}, "two papers, two rows"


def test_a_write_that_fails_part_way_leaves_the_store_as_it_was(tmp_path, monkeypatch):
    """The batch is one write, so a failure in the middle undoes the start."""
    # Without that, a reviewer's session could end with some of their decisions
    # stored and the rest not, and nothing saying where the line fell.
    import sqlite3

    from research_agent.processing import storage as storage_module

    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper("stored first", pmid="1"))], store)

    class FailsOnTheSecondInsert(sqlite3.Connection):
        """A connection that stops working part way through a batch."""
        # The real failures this stands for are a full disk and a file that goes
        # away underneath the run. Neither can be arranged inside a test, and
        # what is being checked is what the store looks like afterwards rather
        # than the cause.

        inserts = 0

        def execute(self, statement, *arguments):
            if statement.strip().startswith("INSERT"):
                FailsOnTheSecondInsert.inserts += 1
                if FailsOnTheSecondInsert.inserts == 2:
                    raise sqlite3.OperationalError("the disk went away")
            return super().execute(statement, *arguments)

    def breaking_connection(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, factory=FailsOnTheSecondInsert)
        connection.execute(storage_module._TABLE)
        return connection

    monkeypatch.setattr(storage_module, "_connection", breaking_connection)

    with pytest.raises(sqlite3.OperationalError):
        save_approved([judged(paper=a_paper("second", pmid="2")),
                       judged(paper=a_paper("third", pmid="3"))], store)

    monkeypatch.undo()
    assert stored_identifiers(store) == {"1"}, \
        "the paper stored before is still there and neither new one landed"


def test_the_abstract_and_the_full_text_are_not_stored(tmp_path):
    """The retrieved text stays in the private runtime buffer (ID-10, ID-13)."""
    # A store that is copied or inspected should not carry the text of papers
    # with it.
    store = tmp_path / "records.sqlite3"
    with_text = a_paper(
        abstract="Forty patients were given drug X and most improved.",
        full_text="The whole paper, at length.", text_source="full text",
        full_text_licence="CC BY")

    save_approved([judged(paper=with_text)], store)

    written = store.read_bytes()
    assert b"and most improved" not in written
    assert b"The whole paper, at length." not in written
    assert b"Forty patients were given drug X" in written, "but the extract is kept"


def test_an_empty_batch_leaves_a_usable_store(tmp_path):
    """A run that approved nothing is not a failure, and the next run should
    not have to create the file itself."""
    store = tmp_path / "records.sqlite3"

    assert save_approved([], store) == 0
    assert stored_identifiers(store) == set()


def test_the_store_and_the_decisions_agree_after_saving(tmp_path):
    store = tmp_path / "records.sqlite3"
    screened = [judged(paper=a_paper("first", pmid="1")),
                judged(paper=a_paper("second", pmid="2"), review="rejected")]

    save_approved(screened, store)

    assert what_does_not_reconcile(screened, store) == (set(), set())


def test_reconciliation_names_what_is_missing_and_what_is_extra(tmp_path):
    """A single true or false would say the two disagree without saying how,
    which is the part somebody correcting it needs (SC-04)."""
    store = tmp_path / "records.sqlite3"
    save_approved([judged(paper=a_paper("stored earlier", pmid="1"))], store)

    now_approved = [judged(paper=a_paper("approved since", pmid="2"))]

    missing, extra = what_does_not_reconcile(now_approved, store)

    assert missing == {"2"}
    assert extra == {"1"}
