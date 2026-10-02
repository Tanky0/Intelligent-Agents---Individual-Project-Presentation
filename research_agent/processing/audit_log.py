"""Reads what a run did and reports what it did not do."""
# The design names partial retrieval as the failure that matters most here. A
# run that searched four of its five subgoals still hands back tidy summaries,
# and nothing in them says the fifth was never run. Every piece of that has been
# collected as the run worked: outcomes say how each search ended, the queue
# keeps the searches never reached, and failures that did not stop the run were
# put on lists. Some of that is read elsewhere for its own purposes, the queue
# by the graph deciding where to go next and the outcomes by the sharing out of
# the screening allowance, but nothing reads it to say what the run missed.
#
# So this reports the gaps rather than the results. A reader who wants to know
# what was found reads the papers; a reader who wants to know what the figure
# rests on reads this.
#
# Three kinds of gap are kept apart, because they call for different things. A
# search that failed may work on another day. A search that was never reached
# was a decision, either the loop limit or the run stopping, and running it
# would need another run rather than another attempt. A search that returned
# part of what it should have is the quiet one: it looks like a search that
# worked, and only the outcome says otherwise.
#
# The three failure lists are read here as well, and none of them stopped the
# run when it happened. A reply that could not be recorded means a demonstration
# will be missing that reply. A buffer that could not be written means the run
# produced no evidence file even though it finished. A line that could not be
# written to the run log means the record of what happened has a hole in it.
# All three are the kind of thing that is noticed weeks later, when somebody
# looks for the file.
#
# No paper text goes into the report. It carries counts, the queries that were
# run and the reasons things went wrong, so it can be shown on screen and kept
# with a run without carrying abstracts into places the boundary keeps them out
# of, as the service and data boundaries in `docs/development.md` record.
#
# This does not write the run log the proposal asks for. That one records
# queries, filters, timestamps, sources and the reviewer's decisions as they
# happen, and is a different thing from reading a finished run.

from dataclasses import dataclass, field

from research_agent import graph
from research_agent.processing import run_log
from research_agent.sources import saved_replies

# The lowest relevance, of ten, at which a screened paper is put in front of
# the reviewer. The rule set on 18 September, after the live run put all
# fifty papers on the page and forty-one of them back again after the
# send-back: the reviewer's hour goes to the papers the agent found likely,
# and a paper rated 5 or below is not drawn. It is not dropped: it stays in
# the run, is counted here and on the page as "rated 5 or below, not shown",
# and the store treats it as a paper waiting for a decision, which is never
# stored. The number lives here, where the count is made for the report, and
# the screen reads it from here, so the page and the report cannot disagree
# about which papers are meant.
SHOWN_FROM_RELEVANCE = 6


@dataclass(frozen=True)
class CoverageReport:
    """What a run searched, what it did not, and what went wrong."""
    # A record rather than a block of text, because a reader is not the only
    # thing that needs it. The interface has to decide whether to show a warning
    # when the coverage is short, and that is a question with an answer
    # rather than a paragraph to be read. The screen asks it before it draws
    # anything about the run (`ui/app.py`).

    searches_planned: int = 0
    searches_finished: int = 0
    searches_partial: list[str] = field(default_factory=list)
    searches_failed: list[str] = field(default_factory=list)
    searches_dropped: list[str] = field(default_factory=list)
    searches_never_reached: list[str] = field(default_factory=list)
    papers_retrieved: int = 0
    # None until the hand-over has combined the repeated records. A run that
    # stopped before it has papers and no answer to this question, and a
    # number here would be read as one. A hand-over that combined nothing
    # answers nought, which is not the same thing.
    papers_after_combining: int | None = None
    papers_screened: int = 0
    # Screened and judged, and not put on the page: the count the reviewer
    # is given so that the papers they see and the papers the run holds are
    # never confused for the same number (see SHOWN_FROM_RELEVANCE).
    papers_rated_five_or_below: int = 0
    papers_beyond_the_allowance: int = 0
    # Kept apart because they are different states. A judgement whose extract
    # was looked for and not found is one whose words are not in the paper,
    # which is a reason to go and look; whether the paper supports the summary
    # some other way is the reviewer's judgement and not this count. One never
    # looked for is a check that did not run, and a reader told the second
    # when it was the first goes looking at a paper that is not the problem.
    extracts_not_found: int = 0
    extracts_never_checked: int = 0
    # Kept apart for the same reason. A paper with no DOI could not be asked
    # about, and a paper whose check failed was asked about and got no answer;
    # both are shown to the reviewer with no status, and only this report says
    # which is which (R-18). The failed checks name the DOI and what
    # went wrong, so a reader can ask again by hand.
    papers_with_no_doi_to_check: int = 0
    status_checks_failed: list[str] = field(default_factory=list)
    stopped_because: str = ""
    needs_a_person: str = ""
    # None when nobody took a snapshot as the run ended, which is not the same
    # as nothing having failed. The lists outlive a run, so a report built
    # without one cannot tell whose failures it would be reading.
    replies_not_recorded: list[str] | None = None
    buffers_not_written: list[str] | None = None
    lines_not_logged: list[str] | None = None

    @property
    def everything_was_searched(self):
        """True when every search that was planned ran and finished."""
        # This is what the interface asks before deciding whether to warn. It is
        # deliberately strict: a partial search counts as short coverage, because
        # a search that returned some of its records is not the search that was
        # planned, and the reader is the one who should judge whether the
        # difference matters.
        #
        # A run that stopped early or that needs a person is short coverage as
        # well, whatever its outcomes say. A run whose planning failed has no
        # outcomes at all, and answering yes to a run that never searched is the
        # worst way for this to be wrong.
        return not (self.searches_partial or self.searches_failed
                    or self.searches_dropped or self.searches_never_reached
                    or self.stopped_because or self.needs_a_person)

    @property
    def anything_went_wrong_quietly(self):
        """True when something failed without stopping the run."""
        # Kept apart from the coverage question because the answer is acted on
        # differently: short coverage is reported to the reader as a limit of
        # the search, while a reply that was not recorded is a fault in the run
        # that somebody has to go and look at.
        #
        # None rather than false when nobody took a snapshot as the run ended.
        # Answering false would say the run was clean on the strength of not
        # having looked.
        #
        # A status check that failed is a yes whether or not a snapshot was
        # taken. It is carried in the buffer rather than on a process-wide list,
        # so the report always knows about it, and a reviewer shown a paper with
        # no status should be told that a check was made and failed (R-18).
        if self.status_checks_failed:
            return True
        if self.replies_not_recorded is None:
            return None
        return bool(self.replies_not_recorded or self.buffers_not_written
                    or self.lines_not_logged)


def _describe(outcome):
    """One line naming the search and what happened to it."""
    # The query is included because a reader deciding whether the gap matters
    # needs to know what was not searched for, and a count of failures does not
    # tell them that.
    line = f"{outcome.subgoal.source}: {outcome.subgoal.query}"
    if outcome.reason:
        line += f" ({outcome.reason})"
    return line


def failures_so_far():
    """How much is on each failure list at this moment."""
    # All three live for as long as the process does, because the code that
    # writes to them is called from several places and has no run to hand. A
    # caller notes this before a run begins and passes it to `failures_during`
    # when the run ends.
    return (len(saved_replies.recording_failures), len(graph.buffer_failures),
            len(run_log.writing_failures))


def failures_during(before):
    """What appeared on the three lists between then and now."""
    # Taken when the run ends rather than when the report is built. A mark on
    # its own has no far end: reporting an earlier run after a later one had
    # finished would hand it the later run's failures as well, because
    # everything after the mark had grown. This is read once, while it is still
    # the run's own, and the report is given what it says.
    replies_before, buffers_before, lines_before = before
    return (tuple(saved_replies.recording_failures[replies_before:]),
            tuple(graph.buffer_failures[buffers_before:]),
            tuple(run_log.writing_failures[lines_before:]))


def report_on(state, failures=None):
    """Builds the coverage report for a finished run."""
    # The searches never reached are taken from the queue rather than worked out
    # by subtracting what ran from what was planned. A refinement adds a search
    # to the queue part way through, so the two numbers do not agree, and the
    # queue is the one that says what is actually left undone.
    #
    # `failures` is what `failures_during` returned when the run ended. Without
    # it the report says the answer is not available rather than saying nothing
    # failed. Silence and no failures read the same on the page, and a reader
    # who cannot tell them apart will take the reassuring one.
    outcomes = list(state.results.outcomes)
    buffer = state.results

    # None, not an empty list, for a list nobody supplied: an empty list would
    # say nothing failed, and the report has to be able to say it does not
    # know. That holds for the whole snapshot and for each list inside it,
    # because a caller may know about one kind of failure and not another.
    # Each list is copied so the report keeps what it was given as it was
    # given, whatever the caller does with the snapshot afterwards.
    replies_not_recorded = None
    buffers_not_written = None
    lines_not_logged = None
    if failures is not None:
        replies, buffers, lines = failures
        if replies is not None:
            replies_not_recorded = list(replies)
        if buffers is not None:
            buffers_not_written = list(buffers)
        if lines is not None:
            lines_not_logged = list(lines)

    # Everything the searches returned, which the hand-over keeps. Before the
    # hand-over there is no such list, and the buffer itself is still
    # everything that was retrieved.
    if buffer.papers_before_deduplication:
        papers_retrieved = len(buffer.papers_before_deduplication)
    else:
        papers_retrieved = len(buffer.papers)

    combined = None
    # The hand-over is what combines the repeated records, and its own signal
    # is what says it happened. Asking whether any original records were kept
    # gets it wrong for a run that reached the hand-over and had nothing to
    # combine, where nought is the answer rather than the question being
    # unanswerable.
    if buffer.ready:
        combined = len(buffer.papers) + len(buffer.papers_beyond_the_screening_limit)

    # Counted only once the hand-over has happened, because that is where the
    # asking happens. Before it, the papers have not been combined or given
    # places, and a count of them would describe a set nobody was asked about.
    papers_with_no_doi_to_check = 0
    if buffer.ready:
        for paper in buffer.papers:
            if not paper.doi:
                papers_with_no_doi_to_check += 1

    return CoverageReport(
        searches_planned=len(outcomes) + len(state.pending_searches),
        searches_finished=sum(1 for one in outcomes if one.result == "finished"),
        searches_partial=[_describe(one) for one in outcomes
                          if one.result == "partial"],
        searches_failed=[_describe(one) for one in outcomes
                         if one.result == "failed"],
        searches_dropped=[_describe(one) for one in outcomes
                          if one.result == "dropped"],
        searches_never_reached=[f"{one.source}: {one.query}"
                                for one in state.pending_searches],
        papers_retrieved=papers_retrieved,
        papers_after_combining=combined,
        papers_screened=len(state.screened),
        papers_rated_five_or_below=sum(
            1 for judged in state.screened
            if judged.screening.relevance < SHOWN_FROM_RELEVANCE),
        papers_beyond_the_allowance=len(buffer.papers_beyond_the_screening_limit),
        extracts_not_found=sum(1 for judged in state.screened
                               if judged.extract_found is False),
        extracts_never_checked=sum(1 for judged in state.screened
                                   if judged.extract_found is None),
        papers_with_no_doi_to_check=papers_with_no_doi_to_check,
        status_checks_failed=list(buffer.status_checks_failed),
        stopped_because=state.stopped_because,
        needs_a_person=state.attention_reason,
        replies_not_recorded=replies_not_recorded,
        buffers_not_written=buffers_not_written,
        lines_not_logged=lines_not_logged,
    )


def what_the_run_did(report):
    """The counts that are true of every run, one line each."""
    # Each line is a label and a number, as short as it can be while saying
    # what was counted. The mock-up of 16 September wrote the counts
    # that way, and the sentences that stood here before ("Papers after
    # combining repeated records") read as prose beside it. The words are the
    # project's own where the mock-up's would say something else: a search
    # "finished" is the outcome's word, and repeated records are combined
    # rather than removed, because the identifiers of the record dropped are
    # kept on the one that stays.
    lines = [
        f"Planned searches: {report.searches_planned}",
        f"Finished searches: {report.searches_finished}",
        f"Papers retrieved: {report.papers_retrieved}",
    ]
    if report.papers_after_combining is None:
        lines.append("Repeated records were not combined: the run did not reach "
                     "the hand-over")
    else:
        lines.append("Papers after combining repeated records: "
                     f"{report.papers_after_combining}")
    lines.append(f"Papers screened: {report.papers_screened}")
    # Said only when there is one, so a run that put every paper in front of
    # the reviewer does not carry a line about papers it kept back.
    if report.papers_rated_five_or_below:
        lines.append("Papers rated 5 or below, not shown: "
                     f"{report.papers_rated_five_or_below}")
    return lines


def what_needs_attention(report):
    """Everything that fell short, one line each; nothing when nothing did."""
    # The count lines are as short as the mock-up's, with the two places where
    # the mock-up's wording was not what happens written plainly instead: a
    # paper with no DOI was not asked about, which is not "could not be
    # verified", and the papers not screened had no places left in the run's
    # allowance, which "capacity limits" only gestures at.
    #
    # The line saying the failure lists were never read is here because it is
    # a shortfall in the record, not a count of what the run did: a reader
    # who cannot be told whether anything failed quietly is a reader who has
    # something to look into.
    attention = []
    if report.papers_beyond_the_allowance:
        attention.append(
            f"Papers retrieved but not screened, no places left: "
            f"{report.papers_beyond_the_allowance}")
    if report.extracts_not_found:
        attention.append(
            f"Extracts not found in the paper: {report.extracts_not_found}")
    if report.extracts_never_checked:
        attention.append(
            f"Extracts never checked: {report.extracts_never_checked}")
    if report.papers_with_no_doi_to_check:
        attention.append(
            f"Publication status not asked for, no DOI: "
            f"{report.papers_with_no_doi_to_check}")

    for heading, entries in (
        ("Searches that returned part of what they should have",
         report.searches_partial),
        ("Searches that failed", report.searches_failed),
        ("Searches the agent dropped", report.searches_dropped),
        ("Searches never reached", report.searches_never_reached),
        ("Publication status checks that failed", report.status_checks_failed),
        ("Replies that could not be recorded", report.replies_not_recorded or []),
        ("Buffers that could not be written", report.buffers_not_written or []),
        ("Lines that could not be written to the run log",
         report.lines_not_logged or []),
    ):
        if entries:
            attention.append(f"{heading}:")
            attention.extend(f"  - {entry}" for entry in entries)

    if report.stopped_because:
        attention.append(f"The run stopped early: {report.stopped_because}")
    if report.needs_a_person:
        attention.append(f"The run needs a person: {report.needs_a_person}")
    if report.anything_went_wrong_quietly is None:
        attention.append("Whether anything failed quietly is not known: nothing "
                         "noted what the run itself put on the failure lists.")
    return attention


def as_lines(report):
    """The report as plain lines, for the console and the run record."""
    # Written here rather than in the interface so that what a reader sees on
    # screen and what is kept with the run are the same words. Two renderings
    # would drift, and the one nobody looked at would be the one that was wrong.
    # The screen draws the two groups side by side under the same two headings
    # (`ui/app.py`), from the two functions above; this is the one-column form.
    #
    # A run with nothing to report says so. An empty report is easy to read as a
    # report that was never built.
    #
    # The lines come in two groups under two headings: what the run did, the
    # counts that are true of every run, and what needs attention, everything
    # that fell short. They were one list, and the review of the screen on 16 September
    # could not tell the figures that were fine from the ones that were not
    # (UX-06). The second heading is left out when nothing fell short,
    # so a clean run does not carry an empty warning.
    lines = ["What the run did:"]
    lines.extend(what_the_run_did(report))

    attention = what_needs_attention(report)
    if attention:
        lines.append("What needs attention:")
        lines.extend(attention)

    if report.everything_was_searched and report.anything_went_wrong_quietly is False:
        lines.append("Every planned search ran and finished, and nothing failed "
                     "quietly.")
    return lines
