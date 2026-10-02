"""Checks that the screening allowance is shared out rather than filled (ID-16)."""
# The failure this guards against is quiet: taking the first fifty papers of a
# deduplicated list gives a full buffer and a tidy summary, and nothing in the
# result says that one search supplied all of them.

from research_agent.processing.selection import share_out_places
from research_agent.schemas import Outcome, PaperRecord, SearchOrigin, Subgoal


def paper(title, *searches):
    return PaperRecord(
        title=title,
        origins=[SearchOrigin(source=source, query=query) for source, query in searches],
    )


def outcome(source, query, papers_found=1):
    return Outcome(
        subgoal=Subgoal(purpose="find papers about the question", source=source, query=query),
        result="finished",
        papers_found=papers_found,
    )


def titles(papers):
    return [item.title for item in papers]


def test_no_places_left_chooses_nothing_and_keeps_every_paper():
    papers = [paper("first", ("pubmed", "a")), paper("second", ("pubmed", "a"))]

    chosen, left_over = share_out_places(papers, [outcome("pubmed", "a")], 0)

    assert titles(chosen) == []
    assert titles(left_over) == ["first", "second"]


def test_one_large_search_cannot_take_every_place_from_a_smaller_one():
    papers = [
        paper("big one", ("pubmed", "a")),
        paper("big two", ("pubmed", "a")),
        paper("big three", ("pubmed", "a")),
        paper("small one", ("semantic scholar", "b")),
    ]
    outcomes = [outcome("pubmed", "a", 3), outcome("semantic scholar", "b", 1)]

    chosen, left_over = share_out_places(papers, outcomes, 2)

    assert titles(chosen) == ["big one", "small one"]
    assert titles(left_over) == ["big two", "big three"]


def test_a_paper_two_searches_found_takes_one_place_and_is_passed_over():
    shared = paper("found twice", ("pubmed", "a"), ("semantic scholar", "b"))
    papers = [shared, paper("only pubmed", ("pubmed", "a")),
              paper("only semantic scholar", ("semantic scholar", "b"))]
    outcomes = [outcome("pubmed", "a", 2), outcome("semantic scholar", "b", 2)]

    chosen, left_over = share_out_places(papers, outcomes, 3)

    assert titles(chosen) == ["found twice", "only semantic scholar", "only pubmed"]
    assert left_over == []


def test_a_search_recorded_twice_does_not_get_two_turns():
    papers = [
        paper("repeated one", ("pubmed", "a")),
        paper("repeated two", ("pubmed", "a")),
        paper("other", ("semantic scholar", "b")),
    ]
    outcomes = [outcome("pubmed", "a"), outcome("pubmed", "a"), outcome("semantic scholar", "b")]

    chosen, _ = share_out_places(papers, outcomes, 2)

    assert titles(chosen) == ["repeated one", "other"]


def test_a_search_that_returned_nothing_does_not_hold_up_the_others():
    papers = [paper("one", ("pubmed", "a")), paper("two", ("pubmed", "a"))]
    outcomes = [outcome("semantic scholar", "empty", 0), outcome("pubmed", "a", 2)]

    chosen, left_over = share_out_places(papers, outcomes, 2)

    assert titles(chosen) == ["one", "two"]
    assert left_over == []


def test_papers_no_search_claims_are_taken_last():
    papers = [
        paper("no search behind it", ("pubmed", "")),
        paper("from a search", ("pubmed", "a")),
    ]

    chosen, left_over = share_out_places(papers, [outcome("pubmed", "a")], 1)

    assert titles(chosen) == ["from a search"]
    assert titles(left_over) == ["no search behind it"]


def test_papers_no_search_claims_are_still_available_when_places_remain():
    papers = [
        paper("no search behind it", ("pubmed", "")),
        paper("from a search", ("pubmed", "a")),
    ]

    chosen, left_over = share_out_places(papers, [outcome("pubmed", "a")], 5)

    assert titles(chosen) == ["from a search", "no search behind it"]
    assert left_over == []


def test_more_places_than_papers_ends_rather_than_going_round_for_ever():
    papers = [paper("only one", ("pubmed", "a"))]

    chosen, left_over = share_out_places(papers, [outcome("pubmed", "a")], 50)

    assert titles(chosen) == ["only one"]
    assert left_over == []


def test_no_outcomes_still_hands_over_what_was_found():
    papers = [paper("first", ("pubmed", "a")), paper("second", ("pubmed", "b"))]

    chosen, left_over = share_out_places(papers, [], 1)

    assert titles(chosen) == ["first"]
    assert titles(left_over) == ["second"]


def test_the_papers_that_were_not_chosen_are_returned_rather_than_counted():
    papers = [paper(f"paper {number}", ("pubmed", "a")) for number in range(5)]

    chosen, left_over = share_out_places(papers, [outcome("pubmed", "a", 5)], 2)

    assert len(chosen) == 2
    assert titles(left_over) == ["paper 2", "paper 3", "paper 4"]
    assert titles(chosen) + titles(left_over) == titles(papers)
