"""Checks the hand-screened sample as it is prepared for release."""
# The sample is the benchmark the recall figure in SC-01 is measured against, so
# two different things have to stay true of it. It must keep the labels a person
# wrote, because a benchmark that quietly changes measures nothing. And it must
# not carry the abstracts those labels were written from, because an abstract is
# somebody else's text and nothing has established a right to publish one
# (R-12).
#
# These checks run against the files prepared for public release alone. The originals with the
# abstracts in them are kept privately and are not on this machine for anybody
# else, so a check needing them could only ever run here. The one-time comparison
# against those originals was made when the files were reduced and its result is
# recorded in the remediation log; this is what holds afterwards.

import csv
import hashlib
import json
import pathlib

import pytest

SAMPLE = (pathlib.Path(__file__).resolve().parents[2]
          / "data" / "hand_screened_sample")

# The columns these files hold, and the only ones. pmid says which
# paper, include says what was decided, and rationale says why in the
# screener's own words. Everything else the draw returned, including the
# abstract, the title and the journal, is kept privately: the identifier is
# enough to find the paper again at its source.
COLUMNS = ["pmid", "include", "rationale"]

# What each sample decided, counted when the labels were written and before the
# agent was ever run against them. A benchmark whose answers can drift is not
# one, so the numbers are written here rather than recomputed from whatever the
# file happens to say.
SAMPLES = {"screening_labels.csv": 3, "screening_labels_check.csv": 5}


def rows_in(name):
    with (SAMPLE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_both_samples_are_where_this_test_expects_them():
    """Guards against the checks below reading nothing."""
    # A wrong path finds no rows, and every check that walks the rows then passes
    # without looking at anything.
    for name in SAMPLES:
        assert (SAMPLE / name).exists(), f"{SAMPLE / name} is not there"
        assert rows_in(name), f"{name} holds no rows"


@pytest.mark.parametrize("name", list(SAMPLES))
def test_a_sample_holds_only_the_columns_agreed_for_release(name):
    """The abstract column is the one this is really about."""
    # Eighty-seven of the hundred rows carried a full abstract until 3 September.
    # They were removed, and this fails if any column outside the agreed three
    # comes back, whether it holds an abstract, a title or anything else nobody
    # decided could be published.
    rows = rows_in(name)

    assert list(rows[0]) == COLUMNS, f"{name} holds {list(rows[0])}"
    for row in rows:
        assert set(row) == set(COLUMNS)


@pytest.mark.parametrize("name, included", SAMPLES.items())
def test_a_sample_still_holds_the_decisions_it_was_drawn_for(name, included):
    """Fifty rows, and the same number included as when they were labelled."""
    # The recall figure is measured against these decisions. A row lost, added or
    # changed would move that figure without anybody deciding to move it.
    rows = rows_in(name)

    assert len(rows) == 50, f"{name} holds {len(rows)} rows"
    assert sum(1 for row in rows if row["include"] == "include") == included
    assert all(row["include"] in ("include", "exclude") for row in rows)
    assert all(row["rationale"].strip() for row in rows), \
        "every decision carries a reason"


def test_the_hundred_identifiers_are_all_there_and_none_is_repeated():
    """The two samples together, because the second was drawn to avoid the
    first. A paper in both would be screened twice and counted twice."""
    identifiers = [row["pmid"] for name in SAMPLES for row in rows_in(name)]

    assert len(identifiers) == 100
    assert len(set(identifiers)) == 100, "a paper appears in both samples"
    assert all(pmid.isdigit() for pmid in identifiers)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_every_rationale_stays_in_the_short_form_it_was_written_in(name):
    """The rationales are short notes, and this keeps them short notes."""
    # It does not prove that no rationale copies an abstract. A length says
    # nothing about where words came from, and this project has twice thrown out
    # a rule that pretended otherwise. What established that nothing was copied
    # was comparing each rationale with the abstract it was written from, once,
    # when the files were reduced: the longest run of words shared by any pair
    # was two. That comparison is recorded in R-12 and cannot be repeated here,
    # because the abstracts are deliberately no longer available.
    #
    # What this keeps is the format. The longest rationale is forty-two
    # characters, and a limit well above that leaves room to write one properly
    # while refusing a paragraph.
    for row in rows_in(name):
        assert len(row["rationale"]) <= 200, \
            f"{row['pmid']} has a rationale of {len(row['rationale'])} characters"


# The two files as they were reduced, each taken as its identifiers, decisions
# and rationales in the order they appear. The checks above count rows and
# decisions, and every one of them would still pass if a paper, a decision or a
# reason were swapped for a different one. This is what makes the benchmark the
# same benchmark: a fingerprint of what it actually says.
#
# If a change is deliberate, screen the paper, change the file, and change the
# fingerprint in the same commit, so the record says who moved it and why. That
# is why the fingerprint is written here rather than worked out each time.
AS_REDUCED = {
    "screening_labels.csv":
        "64639fc20fc4c2c9a35e79b0738339329e2f4fbf41c30c148faff00afc53e624",
    "screening_labels_check.csv":
        "ffaa59b25c13911ab3899e774a67c60f7d72e512a917f97d7b341e4a94fd7fb5",
}


@pytest.mark.parametrize("name, expected", AS_REDUCED.items())
def test_a_sample_still_says_what_it_said_when_it_was_reduced(name, expected):
    """The benchmark cannot drift without somebody saying so."""
    # The recall figure in SC-01 is measured against these decisions. A row
    # quietly edited would move that figure while every count above stayed the
    # same, and nothing else here would notice.
    #
    # The fingerprint is of the parsed rows rather than of the bytes, so saving
    # the file again with different line endings does not fail it, while changing
    # a paper, a decision or a reason does.
    rows = rows_in(name)
    canonical = json.dumps(
        [[row["pmid"], row["include"], row["rationale"]] for row in rows],
        ensure_ascii=False, sort_keys=True)

    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == expected, (
        f"{name} no longer says what it said when it was reduced. If that is "
        f"deliberate, update the fingerprint in this file in the same commit")
