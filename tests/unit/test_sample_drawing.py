"""Checks the guards on the script that draws the hand-screened sample."""
# Drawing a sample is a one-off, and the two ways it can go quietly wrong both
# produce a file that looks finished. A check sample drawn without leaving out
# the first sample overlaps it, so a paper is screened twice and counted twice in
# the recall figure. A draw that finds fewer papers than the sample size returns
# a shorter sample, and the only sign is a row count nobody reads.
#
# Both used to happen silently. The first began when the draw with the abstracts
# in it moved to a folder the repository ignores: the list of papers to leave out
# was read from that file, so on a fresh checkout it was empty and nothing said
# so (R-12).
#
# Nothing here calls PubMed. The parts that would are not exercised; what is
# tested is the two decisions made before any request.

import csv
import importlib.util
import pathlib

import pytest

SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "get_abstracts_to_screen.py")


@pytest.fixture
def drawing():
    """The script as it stands on disk, loaded from its path."""
    # scripts/ is not a package, so it is loaded rather than imported. It is the
    # same file somebody runs, not a copy of what it does.
    spec = importlib.util.spec_from_file_location("get_abstracts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_labels(path, identifiers):
    """Writes a labels file holding these identifiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "include", "rationale"])
        writer.writeheader()
        for pmid in identifiers:
            writer.writerow({"pmid": pmid, "include": "exclude",
                             "rationale": "a reason"})


def test_the_real_labels_are_readable_as_the_first_sample(drawing):
    """The check that follows only means something if the real file passes it."""
    assert drawing.why_the_first_sample_cannot_be_read(
        drawing.SAMPLES["first"]["labels"]) is None


def test_a_missing_first_sample_stops_the_check_sample(drawing, tmp_path):
    """The case the move created."""
    # The list of papers to leave out used to come from the draw, which is not in
    # the repository. Absent, it read as nothing to leave out, and the check
    # sample would have been drawn from the whole pool.
    wrong = drawing.why_the_first_sample_cannot_be_read(tmp_path / "not_there.csv")

    assert wrong and "not there" in wrong


@pytest.mark.parametrize("identifiers, why", [
    ([str(n) for n in range(49)], "forty-nine identifiers, not fifty"),
    ([str(n) for n in range(51)], "fifty-one identifiers, not fifty"),
    (["1"] * 50, "the same identifier fifty times"),
    ([str(n) for n in range(49)] + ["not-a-pmid"], "something that is not an identifier"),
])
def test_a_first_sample_that_is_not_fifty_papers_stops_the_draw(drawing, tmp_path,
                                                                identifiers, why):
    """The list left out has to be the whole list."""
    # A short or repeated list leaves papers in the pool that were already
    # screened, and the two samples overlap by however many were missing. Each of
    # these would have gone unnoticed, because the file exists and can be read.
    labels = tmp_path / "screening_labels.csv"
    write_labels(labels, identifiers)

    assert drawing.why_the_first_sample_cannot_be_read(labels), why


def test_a_readable_first_sample_of_fifty_is_accepted(drawing, tmp_path):
    """A guard that refused everything would stop the sample being drawn."""
    labels = tmp_path / "screening_labels.csv"
    write_labels(labels, [str(n) for n in range(1000, 1050)])

    assert drawing.why_the_first_sample_cannot_be_read(labels) is None
    assert len(drawing.papers_already_drawn(labels)) == 50


def test_the_sample_size_is_the_size_the_benchmark_is_defined_against(drawing):
    """Fifty, and the recall figure in SC-01 is stated against fifty."""
    assert drawing.SAMPLE_SIZE == 50


# Everything above asks the helpers questions. The tests below run the command
# itself, because that is where the two silent failures lived and where a
# helper being correct proves nothing: the checks have to be reached, in the
# right order, and stop the run when they fail.
#
# PubMed is replaced. What is being tested is what the command decides, not
# what the service returns, and a test that made a real request would answer
# differently on a different day.


def prepare(drawing, tmp_path, monkeypatch, which, available=200,
            first_labels=None):
    """Points the command at temporary files and records what it calls."""
    called = {"searched": 0, "fetched": None, "written": None}

    drawn = tmp_path / "drawn"
    drawing.SAMPLES["first"] = {
        "seed": 42, "file": drawn / "first.csv",
        "labels": first_labels if first_labels is not None
        else tmp_path / "screening_labels.csv"}
    drawing.SAMPLES["check"] = {
        "seed": 43, "file": drawn / "check.csv",
        "labels": tmp_path / "screening_labels_check.csv"}

    def search(query, limit):
        called["searched"] += 1
        return [str(n) for n in range(1, available + 1)]

    def fetch(identifiers):
        called["fetched"] = list(identifiers)
        return [], []

    def write_spreadsheet(path, rows):
        called["written"] = path

    monkeypatch.setattr(drawing, "search", search)
    monkeypatch.setattr(drawing, "fetch", fetch)
    monkeypatch.setattr(drawing, "write_spreadsheet", write_spreadsheet)
    monkeypatch.setattr(drawing.settings, "ncbi_api_key", "not-a-real-key",
                        raising=False)
    monkeypatch.setattr(drawing.sys, "argv", ["get_abstracts_to_screen.py", which])
    return called


def test_a_check_draw_with_no_first_sample_stops_before_asking_pubmed(
        drawing, tmp_path, monkeypatch):
    """The list of papers to leave out is missing, so nothing is drawn."""
    # Nothing is asked of PubMed either. A request spent on a draw that is going
    # to be refused is a request wasted against a service that asks callers to
    # keep their volume down.
    called = prepare(drawing, tmp_path, monkeypatch, "check",
                     first_labels=tmp_path / "not_there.csv")

    assert drawing.main() == 1
    assert called["searched"] == 0, "PubMed was asked before the check failed"
    assert called["fetched"] is None
    assert called["written"] is None


def test_a_check_draw_with_a_short_first_sample_stops(drawing, tmp_path,
                                                      monkeypatch):
    """Forty-nine identifiers is not the first sample, and the papers missing
    from it are the ones that would appear in both samples."""
    labels = tmp_path / "screening_labels.csv"
    write_labels(labels, [str(n) for n in range(49)])
    called = prepare(drawing, tmp_path, monkeypatch, "check", first_labels=labels)

    assert drawing.main() == 1
    assert called["searched"] == 0


def test_a_draw_that_finds_fewer_than_fifty_papers_writes_nothing(
        drawing, tmp_path, monkeypatch):
    """The silent one. It used to take whatever was there."""
    # A sample of forty-nine under the name of a sample of fifty measures
    # something other than what SC-01 says it measures, and the only sign was a
    # row count nobody read. Nothing is fetched and nothing is written.
    called = prepare(drawing, tmp_path, monkeypatch, "first", available=49)

    assert drawing.main() == 1
    assert called["searched"] == 1, "the search happens, then the count refuses"
    assert called["fetched"] is None, "papers were fetched for a short sample"
    assert called["written"] is None, "a short sample was written"


def test_a_check_draw_takes_fifty_papers_none_of_them_already_screened(
        drawing, tmp_path, monkeypatch):
    """The ordinary case, and the one that says the exclusion really happens."""
    labels = tmp_path / "screening_labels.csv"
    already = [str(n) for n in range(1, 51)]
    write_labels(labels, already)
    called = prepare(drawing, tmp_path, monkeypatch, "check", available=200,
                     first_labels=labels)

    assert drawing.main() == 0
    assert len(called["fetched"]) == 50
    assert not set(called["fetched"]) & set(already), \
        "the check sample includes a paper from the first sample"
    assert called["written"] is not None
