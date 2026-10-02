"""The order the reviewer reads the screened papers in (FR-11, ID-08)."""
# Fifty papers over eleven scores means most of them tie with something, so the
# tie-break is not an edge case. Most of what is below is about equal scores;
# the first two check the ordinary case, that nothing to rank is not a failure
# and that the highest score comes first.

from research_agent.processing.ranking import in_rank_order
from research_agent.schemas import PaperRecord, ScreenedPaper, Screening, SearchOrigin


def judged(title, relevance, decision="include"):
    return ScreenedPaper(
        paper=PaperRecord(
            title=title,
            origins=[SearchOrigin(source="pubmed", query="drug X")],
            abstract="Forty patients were given drug X.",
            text_source="abstract",
        ),
        screening=Screening(
            paper_number=1,
            decision=decision,
            relevance=relevance,
            reason="it reports an outcome for drug X",
            summary="a trial of drug X in condition Y",
            extract="Forty patients were given drug X",
        ),
    )


def titles(ranked):
    return [item.paper.title for item in ranked]


def test_nothing_to_rank_is_not_a_failure():
    assert in_rank_order([]) == []


def test_the_most_relevant_paper_comes_first():
    given = [judged("middling", 5), judged("best", 9), judged("worst", 1)]

    assert titles(in_rank_order(given)) == ["best", "middling", "worst"]


def test_papers_sharing_a_score_are_ordered_by_the_decision():
    """Include before uncertain before exclude, for a reviewer reading down."""
    given = [
        judged("thrown out", 7, "exclude"),
        judged("not sure", 7, "uncertain"),
        judged("keep", 7, "include"),
    ]

    assert titles(in_rank_order(given)) == ["keep", "not sure", "thrown out"]


def test_a_high_scoring_exclusion_is_still_shown_near_the_top():
    """The agent scoring a paper 9 and excluding it is a disagreement."""
    # It belongs where the reviewer will see it. It goes below every paper
    # scored 9 that would be included or was uncertain, and above everything
    # scored 8 or less, rather than at the bottom of the list.
    given = [
        judged("kept but dull", 3, "include"),
        judged("excluded but relevant", 9, "exclude"),
        judged("kept and relevant", 9, "include"),
    ]

    assert titles(in_rank_order(given)) == [
        "kept and relevant", "excluded but relevant", "kept but dull"]


def test_papers_tying_on_both_keep_the_order_they_were_handed_over_in():
    """The last tie is settled by leaving the papers where they were."""
    # That order comes from the hand-over: the allowance is shared out a paper at
    # a time across the searches, with deduplication's order inside each search's
    # queue. It is repeatable but arbitrary. What matters is that asking twice
    # gives the same answer, so the reviewer is not shown a different ranking
    # each time the same run is read.
    given = [judged(f"paper {number}", 6) for number in range(5)]

    assert titles(in_rank_order(given)) == [f"paper {number}" for number in range(5)]


def test_ranking_the_same_papers_twice_gives_the_same_order():
    given = [
        judged("a", 7, "uncertain"), judged("b", 9), judged("c", 7),
        judged("d", 9, "exclude"), judged("e", 7),
    ]

    assert titles(in_rank_order(given)) == titles(in_rank_order(in_rank_order(given)))


def test_the_papers_that_were_given_are_left_as_they_were():
    """The order they arrived in settles the last ties, so it has to survive."""
    # Sorting the caller's own list would spend that order the first time it was
    # used, and the second ranking would be built on the first one's output.
    given = [judged("second", 9), judged("first", 10)]

    in_rank_order(given)

    assert titles(given) == ["second", "first"]
