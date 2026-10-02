"""Puts the screened papers in the order the reviewer reads them."""
# Relevance is a whole number from 0 to 10, so eleven scores are shared between
# however many papers a run screened. Fifty papers over eleven scores means most
# of them tie with something, and the ties are what this file is mostly about.
#
# The order is settled in three steps.
#
# The score first, highest first, because that is what ranking by relevance
# means.
#
# Then the decision, include before uncertain before exclude. Sorting on the
# score alone would leave papers scored alike in the order they were handed
# over, which is an order nothing about screening chose: a paper the agent would
# exclude could sit above one it would include with the same score. A reviewer
# works down the list and stops when the useful papers run out, so among papers
# scored alike the ones the agent would include come first. An excluded paper
# scored 9 is a disagreement worth seeing, and it still sits above every paper
# scored 8, below the papers scored 9 that would be included or were uncertain.
#
# Then the order the papers were handed over in, which is left as it is rather
# than replaced. The alternative was to sort what remains by something in the
# record, the title being the obvious one, and that is worse: it reads like a
# judgement about the papers and is not one. Keeping the hand-over order also
# means the ranked list and the buffer agree wherever the ranking has nothing to
# say about two papers.
#
# That hand-over order is worth knowing. The screening allowance is shared out a
# paper at a time across the searches, so the papers arrive interleaved by
# search, with deduplication's own order inside each search's queue.
# Deduplication sorts on a representation of the whole record, so the order is
# repeatable but arbitrary. Nothing here should be read as a claim that a paper
# earlier in the list is better than one later on the same score and decision.
#
# Ranking is done in code rather than by asking the model. The model scores each
# paper while it screens; putting the scored papers in order needs no judgement,
# and a model request spent on it would come out of an allowance of twenty a day
# (OB-07).

# Include before uncertain before exclude, as the reason above sets out. The
# three are written here rather than sorted by their own words, because
# alphabetical order would put exclude first and that is the opposite of what
# a reviewer wants at the top of the list.
_DECISION_ORDER = ("include", "uncertain", "exclude")


def in_rank_order(judged):
    """Returns the screened papers in the order the reviewer should see them."""
    # A new list. The order papers were handed over in is what settles the last
    # ties, so the caller's list has to stay as it was for that order to still
    # mean something the next time this is asked.
    return sorted(
        judged,
        # Only the two decided keys are in here. The third step of the order is
        # to leave papers tying on both of these where they were, so writing
        # their positions into the key would be saying the same thing twice.
        key=lambda item: (-item.screening.relevance,
                          _DECISION_ORDER.index(item.screening.decision)),
    )
