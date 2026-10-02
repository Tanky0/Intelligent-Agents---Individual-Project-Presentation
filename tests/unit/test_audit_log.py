"""What the coverage report says about a run that did not do everything (FR-05)."""
# The failure the design names is the quiet one: a run that searched four of its
# five subgoals still hands back tidy summaries. So most of what is below builds
# a run with a gap in it and checks the gap is named, not counted away.

from research_agent import graph
from research_agent.processing.audit_log import (
    as_lines,
    failures_during,
    failures_so_far,
    report_on,
    what_needs_attention,
    what_the_run_did,
)
from research_agent.processing.validation import with_the_extracts_checked
from research_agent.schemas import (
    Outcome,
    PaperRecord,
    ResearchRequest,
    ResearchState,
    ResultsBuffer,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
)
from research_agent.sources import saved_replies


def as_text(report):
    """The report as one block, so a test can look for a line in it."""
    return "\n".join(as_lines(report))


def nothing_failed():
    """A snapshot for a run during which neither list grew."""
    # Taken and closed straight away, which is what a run that failed at nothing
    # would leave behind.
    return failures_during(failures_so_far())


def a_subgoal(query="drug X trials", source="pubmed"):
    return Subgoal(purpose="find trials of drug X", source=source, query=query)


def an_outcome(query="drug X trials", result="finished", reason="", found=1):
    return Outcome(subgoal=a_subgoal(query), result=result,
                   papers_found=found, reason=reason)


def a_paper(title="A trial of drug X"):
    return PaperRecord(
        title=title,
        abstract="Forty patients were given drug X.",
        text_source="abstract",
        origins=[SearchOrigin(source="pubmed", query="drug X trials")])


def judged(paper=None, extract="Forty patients were given drug X"):
    return ScreenedPaper(
        paper=paper or a_paper(),
        screening=Screening(paper_number=1, decision="include", relevance=7,
                            reason="it reports an outcome", summary="a trial",
                            extract=extract))


def a_state(**changes):
    values = {
        "request": ResearchRequest(
            question="Does drug X treat condition Y?",
            inclusion_criteria="trials reporting an outcome",
            exclusion_criteria="reviews"),
        "results": ResultsBuffer(outcomes=[an_outcome()], papers=[a_paper()],
                                 ready=True),
    }
    values.update(changes)
    return ResearchState(**values)


def test_a_run_that_did_everything_says_so():
    report = report_on(a_state(), nothing_failed())

    assert report.everything_was_searched is True
    assert report.anything_went_wrong_quietly is False
    assert "Every planned search ran and finished" in "\n".join(as_lines(report))


def test_the_lines_come_under_two_headings_and_the_second_only_when_needed():
    """The counts that are true of every run and the shortfalls were one list, and
    read cold on 16 September it did not show which figures were fine from the
    ones that were not (UX-06).
    """
    # A clean run carries no empty "needs attention" heading.
    # A paper with a DOI, in the buffer and screened with its extract found:
    # nothing about it needs attention. Without the DOI the report counts one
    # paper whose status could not be asked for, which is a shortfall and
    # belongs under the second heading; the count reads the buffer's papers.
    with_doi = a_paper().model_copy(update={"doi": "10.99999/invented.1"})
    buffer = ResultsBuffer(outcomes=[an_outcome()], papers=[with_doi], ready=True)
    screened = with_the_extracts_checked([judged(with_doi)])
    clean = as_lines(report_on(a_state(results=buffer, screened=screened),
                               nothing_failed()))
    short = as_lines(report_on(a_state(results=buffer, screened=screened,
                                       pending_searches=[a_subgoal("drug X harms")]),
                               nothing_failed()))

    assert clean[0] == "What the run did:"
    assert "What needs attention:" not in clean
    assert short[0] == "What the run did:"
    assert "What needs attention:" in short
    counts_end = short.index("What needs attention:")
    assert "Searches never reached:" in short[counts_end:], \
        "a shortfall sits under the second heading, not among the counts"
    assert all(":" in line for line in short[1:counts_end]), \
        "and only counts sit under the first"


def test_the_two_groups_are_the_one_column_form_taken_apart():
    """Every line of the record's one-column form is one of the two groups'
    lines or a heading over them (UX-06).
    """
    # The screen draws the two groups side by side from the two functions,
    # and the record keeps the one-column form; the second review of 17
    # September chose the mock-up's shape, and a second wording for the
    # screen would be a second thing to keep true.
    with_doi = a_paper().model_copy(update={"doi": "10.99999/invented.1"})
    buffer = ResultsBuffer(outcomes=[an_outcome()], papers=[with_doi], ready=True)
    screened = with_the_extracts_checked([judged(with_doi)])
    clean = report_on(a_state(results=buffer, screened=screened), nothing_failed())
    short = report_on(a_state(results=buffer, screened=screened,
                              pending_searches=[a_subgoal("drug X harms")]),
                      nothing_failed())

    assert what_the_run_did(clean) == [
        "Planned searches: 1", "Finished searches: 1", "Papers retrieved: 1",
        "Papers after combining repeated records: 1", "Papers screened: 1"]
    assert what_needs_attention(clean) == []
    assert what_needs_attention(short) == [
        "Searches never reached:", "  - pubmed: drug X harms"]
    assert as_lines(short) == (["What the run did:"] + what_the_run_did(short)
                               + ["What needs attention:"]
                               + what_needs_attention(short))


def test_the_failure_lists_never_read_is_a_shortfall_not_a_count():
    """A reader who cannot be told whether anything failed quietly has
    something to look into, so the line sits with what needs attention."""
    unknown = report_on(a_state())

    assert "Whether anything failed quietly is not known" in \
        what_needs_attention(unknown)[-1]
    assert not any("not known" in line for line in what_the_run_did(unknown))


def test_a_search_never_reached_is_named_and_not_counted_away():
    """The queue is what says which searches were left undone."""
    # Subtracting what ran from what was planned would not agree with it,
    # because a refinement adds a search to the queue part way through.
    report = report_on(a_state(pending_searches=[a_subgoal("drug X harms")]))

    assert report.searches_never_reached == ["pubmed: drug X harms"]
    assert report.searches_planned == 2
    assert report.everything_was_searched is False
    assert "Searches never reached:" in "\n".join(as_lines(report))


def test_a_partial_search_counts_as_short_coverage():
    """A search that returned some of its records looks like one that worked."""
    # Only the outcome says otherwise, which is why it is reported rather than
    # added to the finished count.
    buffer = ResultsBuffer(
        outcomes=[an_outcome(result="partial",
                             reason="3 records had no title and were left out")],
        papers=[a_paper()], ready=True)

    report = report_on(a_state(results=buffer))

    assert report.searches_finished == 0
    assert report.searches_partial == [
        "pubmed: drug X trials (3 records had no title and were left out)"]
    assert report.everything_was_searched is False


def test_a_failed_search_carries_the_reason_and_the_query():
    """A count of failures does not tell a reader what was not searched for."""
    buffer = ResultsBuffer(
        outcomes=[an_outcome(result="failed", reason="pubmed: service busy",
                             found=0)],
        ready=True)

    report = report_on(a_state(results=buffer))

    assert report.searches_failed == ["pubmed: drug X trials (pubmed: service busy)"]


def test_a_dropped_search_is_reported_separately_from_a_failed_one():
    """Dropping was the agent's decision; failing was not."""
    # A reader deciding whether to run the search again needs to know which.
    buffer = ResultsBuffer(
        outcomes=[an_outcome(result="dropped", reason="the same ground as the "
                             "first search", found=0)],
        ready=True)

    report = report_on(a_state(results=buffer))

    assert report.searches_dropped and not report.searches_failed


def test_the_papers_the_allowance_had_no_room_for_are_reported():
    """A summary built from the screened papers reads as though the rest were
    never found."""
    buffer = ResultsBuffer(
        outcomes=[an_outcome()],
        papers=[a_paper("screened")],
        papers_before_deduplication=[a_paper("one"), a_paper("two"),
                                     a_paper("three")],
        papers_beyond_the_screening_limit=[a_paper("no place for this one")],
        ready=True)

    report = report_on(a_state(results=buffer, screened=[judged()]))

    assert report.papers_retrieved == 3
    assert report.papers_after_combining == 2
    assert report.papers_screened == 1
    assert report.papers_beyond_the_allowance == 1
    assert "Papers retrieved but not screened, no places left: 1" in "\n".join(
        as_lines(report))


def test_a_judgement_whose_extract_was_not_found_is_counted():
    """SC-02 is about claims that can be traced back, so the count belongs
    with the coverage rather than being left to storage to discover."""
    invented = judged(extract="Every patient recovered within a week")

    report = report_on(a_state(screened=with_the_extracts_checked([invented])))

    assert report.extracts_not_found == 1


def test_a_judgement_never_checked_is_counted_apart_from_one_not_found():
    """The two are different states and lead somewhere different."""
    # An extract looked for and not found is one whose words are not in the
    # paper, which is a reason to go and look. One never looked for is a check
    # that did not run, and a reader told the second when it was the first goes
    # looking at a paper that is not the problem.
    checked = with_the_extracts_checked(
        [judged(extract="Every patient recovered within a week")])

    report = report_on(a_state(screened=checked + [judged()]))

    assert report.extracts_not_found == 1
    assert report.extracts_never_checked == 1
    written = "\n".join(as_lines(report))
    assert "Extracts not found in the paper: 1" in written
    assert "Extracts never checked: 1" in written


def test_a_reply_that_could_not_be_recorded_is_reported():
    """Nothing stopped when it happened, so nothing would say so later."""
    # A demonstration missing that reply is found weeks afterwards, when
    # somebody replays the run.
    before = failures_so_far()
    saved_replies.recording_failures.append(
        "GET https://example.test could not be recorded")

    report = report_on(a_state(), failures_during(before))

    assert report.anything_went_wrong_quietly is True
    assert report.replies_not_recorded == [
        "GET https://example.test could not be recorded"]
    assert "Replies that could not be recorded:" in "\n".join(as_lines(report))


def test_a_buffer_that_could_not_be_written_is_reported():
    """A finished run with no evidence file looks like a finished run."""
    before = failures_so_far()
    graph.buffer_failures.append(
        "the buffer could not be written to evidence/x.json: no space left")

    report = report_on(a_state(), failures_during(before))

    assert report.anything_went_wrong_quietly is True
    assert "Buffers that could not be written:" in "\n".join(as_lines(report))


def test_a_run_that_stopped_early_says_why():
    report = report_on(a_state(
        stopped_because="this run has used the 18 model requests it was allowed"))

    assert "The run stopped early: this run has used the 18 model requests" in \
        "\n".join(as_lines(report))


def test_a_run_that_needs_a_person_says_why():
    report = report_on(a_state(
        needs_human_attention=True,
        attention_reason="the searches could not be planned"))

    assert report.needs_a_person == "the searches could not be planned"
    assert "The run needs a person:" in "\n".join(as_lines(report))


def test_a_run_with_a_gap_does_not_claim_everything_ran():
    """The reassuring line is the one most likely to be read and least likely
    to be checked, so it appears only when there is nothing to report."""
    report = report_on(a_state(pending_searches=[a_subgoal("drug X harms")]))

    assert "Every planned search ran and finished" not in "\n".join(as_lines(report))


def test_the_report_carries_no_paper_text():
    """It is shown on screen and kept with the run, and the boundary keeps
    abstracts out of both, as the boundaries in `docs/development.md` record."""
    buffer = ResultsBuffer(outcomes=[an_outcome()], papers=[a_paper()],
                           ready=True)

    written = "\n".join(as_lines(report_on(a_state(results=buffer,
                                                   screened=[judged()]))))

    assert "Forty patients were given drug X" not in written


def test_an_earlier_run_s_failure_is_not_reported_as_this_run_s():
    """Both lists outlive a run, so each run's own share is taken as it ends."""
    # Without that, the second run in one process would be blamed for the
    # first's, and somebody would go looking for a fault in the wrong place.
    saved_replies.recording_failures.append("something from an earlier run")

    report = report_on(a_state(), nothing_failed())

    assert report.replies_not_recorded == []
    assert report.anything_went_wrong_quietly is False


def test_a_later_run_s_failure_is_not_added_to_an_earlier_report():
    """The snapshot is closed when the run ends, not when the report is built."""
    # A mark on its own has no far end, so reporting the first run after the
    # second had finished handed it the second run's failures too.
    before = failures_so_far()
    quiet_run = failures_during(before)
    saved_replies.recording_failures.append("this belongs to a later run")

    report = report_on(a_state(), quiet_run)

    assert report.replies_not_recorded == []
    assert report.anything_went_wrong_quietly is False


def test_a_report_with_no_snapshot_says_it_does_not_know():
    """Silence and no failures read the same on the page."""
    # A reader who cannot tell them apart will take the reassuring one, so the
    # report says the answer is not available instead of saying nothing failed.
    report = report_on(a_state())

    assert report.anything_went_wrong_quietly is None
    assert report.replies_not_recorded is None
    written = as_text(report)
    assert "not known" in written
    assert "nothing failed" not in written


def test_a_snapshot_missing_one_list_leaves_that_answer_unknown():
    """A caller may know about two kinds of failure and not the third."""
    # The one it does not know about stays None rather than becoming an empty
    # list, for the same reason the whole snapshot does: an empty list says
    # nothing failed, and not knowing is a different answer. The check is here
    # so a plainer rewrite of the copying cannot drop it, as one did.
    report = report_on(a_state(), (None, ("a buffer",), ()))

    assert report.replies_not_recorded is None
    assert report.buffers_not_written == ["a buffer"]
    assert report.lines_not_logged == []


def test_a_hand_over_that_combined_nothing_answers_nought():
    """Nought is an answer and no answer is not."""
    # Asking whether any original records were kept got this wrong for a run
    # that reached the hand-over with nothing to combine.
    empty = ResultsBuffer(outcomes=[an_outcome(found=0)], ready=True)

    report = report_on(a_state(results=empty), nothing_failed())

    assert report.papers_after_combining == 0
    assert "Papers after combining repeated records: 0" in as_text(report)


def test_a_run_that_stopped_before_the_hand_over_does_not_claim_a_combined_count():
    """Nothing was combined, so there is no answer rather than an answer of
    however many papers the buffer happened to hold."""
    buffer = ResultsBuffer(outcomes=[an_outcome()],
                           papers=[a_paper("one"), a_paper("two")])

    report = report_on(a_state(results=buffer))

    assert report.papers_retrieved == 2
    assert report.papers_after_combining is None
    assert "Repeated records were not combined" in as_text(report)


def test_a_run_that_needs_a_person_is_not_told_everything_ran():
    """A run whose planning failed has no outcomes at all."""
    # Answering yes to a run that never searched is the worst way for this to be
    # wrong, and it was what the earlier version did.
    never_searched = a_state(
        results=ResultsBuffer(),
        needs_human_attention=True,
        attention_reason="the searches could not be planned")

    report = report_on(never_searched)

    assert report.everything_was_searched is False
    assert "Every planned search ran and finished" not in as_text(report)


def test_a_run_that_stopped_early_is_not_told_everything_ran():
    report = report_on(a_state(
        stopped_because="this run has used the 18 model requests it was allowed"))

    assert report.everything_was_searched is False
    assert "Every planned search ran and finished" not in as_text(report)


def test_a_status_check_that_failed_is_named_and_counts_as_a_quiet_failure():
    """A paper whose check failed is shown with no status, exactly like a
    paper with no DOI, so the report is what tells the two apart (R-18).
    """
    # The failed check makes the run one that failed quietly even though no
    # snapshot of the failure lists was taken: it is carried in the buffer, so
    # the report knows about it whether or not anybody noted the lists.
    buffer = ResultsBuffer(
        outcomes=[an_outcome()], papers=[a_paper()], ready=True,
        status_checks_failed=["10.1000/unavailable: status 503"])

    report = report_on(a_state(results=buffer))

    assert report.status_checks_failed == ["10.1000/unavailable: status 503"]
    assert report.anything_went_wrong_quietly is True
    written = as_text(report)
    assert "Publication status checks that failed:" in written
    assert "  - 10.1000/unavailable: status 503" in written
    assert "Every planned search ran and finished" not in written


def test_papers_with_no_doi_are_counted_only_once_the_hand_over_has_happened():
    """Before the hand-over nothing has been asked about, so a count of papers
    with no DOI would describe a set nobody looked up."""
    not_yet = ResultsBuffer(outcomes=[an_outcome()], papers=[a_paper()],
                            ready=False)
    handed_over = ResultsBuffer(outcomes=[an_outcome()], papers=[a_paper()],
                                ready=True)

    before = report_on(a_state(results=not_yet))
    after = report_on(a_state(results=handed_over))

    assert before.papers_with_no_doi_to_check == 0
    assert "no DOI" not in as_text(before)
    assert after.papers_with_no_doi_to_check == 1
    assert "Publication status not asked for, no DOI: 1" in as_text(after)
