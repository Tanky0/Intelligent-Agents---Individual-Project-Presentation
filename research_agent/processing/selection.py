"""Shares the screening allowance out across the searches that ran."""
# The screening allowance lets one run give fifty papers to curation, and the
# limit is on what curation is given, not on what searching finds. It does not
# say which fifty. Taking the first fifty of the deduplicated list would answer
# that badly: that list is ordered by a representation of the whole record,
# which is repeatable but arbitrary, so one search could fill every place and
# make every later search pointless. That is the outcome the limit exists to
# avoid.
#
# So the places are shared out a paper at a time instead. One queue is made for
# each search, in the order the outcomes record them, and the queues are
# visited in turn until the places run out or every queue is empty. This shares
# the available places out. It does not promise every search a place, and
# cannot: with fewer places than searches the later queues are never reached.
#
# A paper found by several searches stands in several queues and still costs
# one place. Whichever queue reaches it first takes it, and the others pass
# over it to their next paper rather than losing their turn. Counting it once
# is what makes fifty a count of papers rather than of retrievals.
#
# The order the outcomes give is used rather than the order papers arrived,
# because the outcomes are what records which searches ran, including one that
# returned nothing. A search recorded twice does not get two turns.
#
# The papers that were not selected are returned as well, rather than counted
# and dropped. A reviewer told that twelve papers were not screened, with
# nothing to look at, cannot check the claim.


def _searches_in_the_order_they_ran(outcomes):
    searches = []
    for outcome in outcomes:
        search = (outcome.subgoal.source, outcome.subgoal.query)
        if search not in searches:
            searches.append(search)
    return searches


def _queues_of_papers(papers, searches):
    """One queue of positions for each search, and one for the rest."""
    # Positions rather than papers, because two papers can hold equal values and
    # the run has to be able to tell which of them has already been taken.
    #
    # A paper reaches the last queue only when no search that ran claims it.
    # Such a paper carries an origin with an empty query, which is how a record
    # fetched by identifier rather than found by a search is marked
    # (`schemas.py`). Today the only records made that way are the
    # standard-dataset measurement's fifty, and `scripts/measure_against_clef.py`
    # hands them to the Curation Agent directly without passing through here
    # (DS-02); the recall measurement against the hand-screened sample decides
    # from identifiers and hands nothing to curation. So this queue is built
    # and tested for a route no run uses yet. It stays because a paper no
    # search claims must still be given a place rather than dropped in
    # silence, and such records are not a search and are not given a turn as
    # though they were one.
    queues = [[] for _ in searches]
    unclaimed = []
    for position, paper in enumerate(papers):
        origins = {(origin.source, origin.query) for origin in paper.origins}
        claimed = False
        for queue, search in zip(queues, searches):
            if search in origins:
                queue.append(position)
                claimed = True
        if not claimed:
            unclaimed.append(position)
    return queues + [unclaimed]


def share_out_places(papers, outcomes, places):
    """Chooses which papers curation is given, and returns the rest with them."""
    # Returns the chosen papers and the ones left over, in that order.
    #
    # The chosen papers come out in the order the places were given: the first
    # paper of each search, then the second of each, and so on. That is the
    # order the sharing out produced, so it is what shows how the places went,
    # and it is not the order the papers arrived in.
    #
    # The ones left over keep the order they arrived in, because nothing chose
    # between them and a new order would suggest something had.
    if places <= 0:
        return [], list(papers)

    queues = _queues_of_papers(papers, _searches_in_the_order_they_ran(outcomes))
    chosen = []
    taken = set()

    while len(chosen) < places:
        # A round that takes nothing has no papers left to take, and going
        # round again would not change that. Without this the loop would not
        # end when the papers run out before the places do.
        took_one = False
        for queue in queues:
            # A paper another queue already took is passed over, so the
            # search does not lose its turn to a paper it shares. The queues
            # are this function's own lists, so taking from the front of
            # them is the plainest way to keep each one's place.
            while queue and queue[0] in taken:
                queue.pop(0)
            if not queue:
                continue
            position = queue.pop(0)
            taken.add(position)
            chosen.append(position)
            took_one = True
            if len(chosen) >= places:
                break
        if not took_one:
            break

    return (
        [papers[position] for position in chosen],
        [paper for position, paper in enumerate(papers) if position not in taken],
    )
