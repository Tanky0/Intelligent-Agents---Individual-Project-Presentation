"""Uses invented records to expose lost evidence and false merges (FR-10)."""
# Permutation checks cover ambiguous groups as well as merged fields. Checking
# only the final sort order would miss a bridge joining a different paper when
# the services returned their results in a different order.

import subprocess
import sys
from itertools import permutations

import pytest
from pydantic import ValidationError

from research_agent.identifiers import normalise_doi
from research_agent.processing import dedup
from research_agent.schemas import (
    CONCERNS_RAISED,
    NONE_IDENTIFIED,
    NOT_CHECKED,
    RETRACTED,
    PaperRecord,
    SearchOrigin,
)


def paper(title="Synthetic study of library search", **values):
    values.setdefault("origins", [SearchOrigin(source="pubmed", query="synthetic search")])
    return PaperRecord(title=title, **values)


def across_permutations(records):
    expected = None
    result = None
    for ordering in permutations(records):
        result = dedup.deduplicate(ordering)
        actual = [item.model_dump() for item in result]
        if expected is None:
            expected = actual
        assert actual == expected
    return result


def test_empty_input_has_no_papers():
    assert dedup.deduplicate([]) == []


def test_identifier_import_does_not_load_http_or_replace_ssl_context():
    script = (
        "import ssl, sys; before = ssl.SSLContext; "
        "from research_agent.identifiers import normalise_doi; "
        "assert normalise_doi('doi:10.1000/A') == '10.1000/a'; "
        "assert 'research_agent.sources.http' not in sys.modules; "
        "assert ssl.SSLContext is before"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["10.1000/ABC", "https://doi.org/10.1000/ABC", "doi:10.1000/abc"])
def test_doi_forms_merge_even_when_titles_differ(value):
    merged = across_permutations([
        paper("First title", doi=value), paper("Other", doi="10.1000/abc")])
    assert len(merged) == 1
    assert merged[0].doi == normalise_doi(value)


def test_matching_pmid_merges_without_doi():
    merged = across_permutations([paper("First title", pmid=" 123 "), paper("Other", pmid="123")])
    assert len(merged) == 1
    assert merged[0].pmid == "123"


def test_exact_title_normalises_case_spacing_and_punctuation_without_identifiers():
    merged = across_permutations([paper("A synthetic-study"), paper(" a SYNTHETIC study! ")])
    assert len(merged) == 1


def test_empty_identifiers_do_not_merge_unrelated_titles():
    assert len(across_permutations([paper("Library search"), paper("Ocean temperature")])) == 2


def test_punctuation_only_titles_are_not_matching_keys():
    assert len(across_permutations([paper("..."), paper("!!!")])) == 2


def test_near_identical_title_merges_with_supporting_metadata():
    records = [
        paper("Synthetic evaluation of library retrieval methods",
              year="2024", authors=["A Example"]),
        paper("Synthetic evaluation of library retrieval method",
              year="2024", authors=["A Example"]),
    ]
    assert len(across_permutations(records)) == 1


@pytest.mark.parametrize("score,count", [(94, 2), (94.999, 2), (95, 1), (95.001, 1)])
def test_fuzzy_boundary_is_inclusive(monkeypatch, score, count):
    monkeypatch.setattr(dedup.fuzz, "token_sort_ratio", lambda *args: score)
    monkeypatch.setattr(dedup.fuzz, "ratio", lambda *args: score)
    records = [paper("First title", year="2024", authors=["A Example"]),
               paper("Second title", year="2024", authors=["A Example"])]
    assert len(across_permutations(records)) == count


def test_reversing_a_statement_does_not_merge_even_with_matching_metadata():
    records = [paper("A inhibits B", year="2024", authors=["A Example"]),
               paper("B inhibits A", year="2024", authors=["A Example"])]
    assert len(across_permutations(records)) == 2


@pytest.mark.parametrize("metadata", [
    {}, {"year": "2024"}, {"year": "2024", "authors": [" "]},
    {"year": "2023", "authors": ["A Example"]},
    {"year": "2024", "authors": ["B Example"]},
])
def test_fuzzy_match_needs_matching_year_and_first_author(metadata):
    records = [
        paper("Synthetic evaluation of library retrieval methods",
              year="2024", authors=["A Example"]),
        paper("Synthetic evaluation of library retrieval method", **metadata),
    ]
    assert len(across_permutations(records)) == 2


@pytest.mark.parametrize("identifiers", [
    ({"doi": "10.1000/a"}, {"doi": "10.1000/b"}),
    ({"pmid": "1"}, {"pmid": "2"}),
    ({"doi": "10.1000/a", "pmid": "1"}, {"doi": "10.1000/a", "pmid": "2"}),
    ({"doi": "10.1000/a", "pmid": "1"}, {"doi": "10.1000/b", "pmid": "1"}),
])
def test_identifier_conflicts_block_even_exact_titles(identifiers):
    assert len(across_permutations([paper(**values) for values in identifiers])) == 2


def test_origins_are_all_kept_and_sorted():
    records = [paper(origins=[SearchOrigin(source="semantic scholar", query="z")]),
               paper(origins=[SearchOrigin(source="pubmed", query="b"),
                              SearchOrigin(source="pubmed", query="a")])]
    merged = across_permutations(records)
    assert [(origin.source, origin.query) for origin in merged[0].origins] == [
        ("pubmed", "a"), ("pubmed", "b"), ("semantic scholar", "z")]


def test_group_conflict_cannot_be_hidden_by_an_identifier_free_bridge():
    records = [paper(doi="10.1000/a", pmid="1"), paper(), paper(doi="10.1000/b", pmid="2")]
    merged = across_permutations(records)
    assert len(merged) == 2
    assert {(item.doi, item.pmid) for item in merged} == {("10.1000/a", "1"), ("10.1000/b", "2")}
    assert sorted(len(item.origins) for item in merged) == [1, 2]


def test_stronger_identifier_match_is_considered_before_title_bridge():
    records = [paper("Common title", doi="10.1000/a"),
               paper("Common title", pmid="2"),
               paper("Different title", doi="10.1000/b", pmid="2")]
    merged = across_permutations(records)
    assert len(merged) == 2
    assert next(item for item in merged if item.pmid == "2").doi == "10.1000/b"


def test_missing_identifier_can_join_a_record_with_a_pmid_by_title():
    merged = across_permutations([paper(), paper(pmid="123")])
    assert len(merged) == 1
    assert merged[0].pmid == "123"


def test_full_text_keeps_its_own_licence_and_complementary_abstract():
    records = [paper(abstract="Invented abstract.", text_source="abstract", doi="10.1000/a"),
               paper(full_text="Invented body.", full_text_licence="https://example.org/licence-b",
                     text_source="full text")]
    merged = across_permutations(records)[0]
    assert merged.full_text == "Invented body."
    assert merged.full_text_licence == "https://example.org/licence-b"
    assert merged.abstract == "Invented abstract."
    assert merged.text_source == "full text"


@pytest.mark.parametrize("first,second,expected", [
    ("Short", "Longer body", "Longer body"), ("Beta", "Alfa", "Alfa"),
])
def test_equal_kind_of_text_uses_length_then_wording(first, second, expected):
    records = [paper(full_text=first, full_text_licence="licence-a", text_source="full text"),
               paper(full_text=second, full_text_licence="licence-b", text_source="full text")]
    merged = across_permutations(records)[0]
    assert merged.full_text == expected
    assert merged.full_text_licence == "licence-b"


def test_identical_bodies_with_different_licences_have_a_stable_owner():
    records = [paper(full_text="Invented body", full_text_licence=licence, text_source="full text")
               for licence in ("licence-b", "licence-a")]
    assert across_permutations(records)[0].full_text_licence == "licence-a"


def test_field_choice_prefers_doi_then_length_then_wording_and_fills_gaps():
    records = [paper(doi="10.1000/a", journal="Short", authors=["A Example"]),
               paper(journal="Longer journal", paper_id="synthetic-id", authors=["A Example"])]
    merged = across_permutations(records)[0]
    assert merged.journal == "Short"
    assert merged.paper_id == "synthetic-id"
    for journals, expected in [(('Short', 'Longer'), 'Longer'), (('Beta', 'Alfa'), 'Alfa')]:
        merged = across_permutations([paper(journal=value) for value in journals])
        assert merged[0].journal == expected


def test_whitespace_is_not_preferred_over_a_supplied_field():
    merged = across_permutations([paper(doi="10.1000/a", journal=" "), paper(journal="Journal")])
    assert merged[0].journal == "Journal"


@pytest.mark.parametrize("statuses,expected", [
    ([NONE_IDENTIFIED, CONCERNS_RAISED], CONCERNS_RAISED),
    ([CONCERNS_RAISED, RETRACTED], RETRACTED),
    ([NOT_CHECKED, NONE_IDENTIFIED], NONE_IDENTIFIED),
    ([NOT_CHECKED, NOT_CHECKED], NOT_CHECKED),
])
def test_the_most_serious_publication_status_survives(statuses, expected):
    """The merged record says the worst that was found."""
    # The third case is the one that matters most and is easy to get backwards:
    # a record Crossref was never asked about, merged with one it answered for,
    # keeps the answer. Silence must not outrank an answer, and an answer must
    # not be read as silence (ID-19).
    records = [paper(publication_status=status) for status in statuses]
    assert across_permutations(records)[0].publication_status == expected


def test_an_unrecognised_status_wording_is_refused_by_the_record():
    """An earlier merge kept any wording it did not recognise, because the field
    was free text and a notice nobody had listed could arrive in it.
    """
    # Now the record refuses it before deduplication is reached, which is the
    # stronger guard, and this pins that the weaker one was not needed.
    with pytest.raises(ValidationError):
        paper(publication_status="Unrecognised notice")


def test_records_and_their_nested_values_are_not_changed():
    records = [paper(doi="doi:10.1000/A"), paper(doi="10.1000/a")]
    before = [item.model_dump() for item in records]
    result = dedup.deduplicate(records)
    result[0].origins[0].query = "changed output"
    assert [item.model_dump() for item in records] == before
