"""Checks the record formats the two agents pass between them."""
# Two things are being checked here. The first is that a buffer written out as
# JSON text and read back holds the same values, because that hand-over is the
# whole of the agreement between the two agents (UT-21, FR-17, DC-05). The
# second is that a value of the wrong shape is refused rather than converted,
# which is what makes the check worth doing at all (UT-STRICT-01, TC-04).
#
# Nothing here calls a model or a service.

import json

import pytest
from pydantic import ValidationError

from research_agent.schemas import (
    LONGEST_RATIONALE,
    NextMove,
    Outcome,
    PaperRecord,
    PlanningDecision,
    ResearchRequest,
    ResearchState,
    ResultsBuffer,
    RunBudget,
    ScreenedPaper,
    Screening,
    ScreeningList,
    SearchOrigin,
    Subgoal,
    SubgoalList,
)

# The four functions below build the sample records the tests work with. They
# are functions rather than fixed values written once at the top of the file,
# because a test that changes a record must not change it for the tests that
# run afterwards. Each call builds a new one.


def a_request():
    """A request of the kind the interface will build."""
    return ResearchRequest(
        question="Which studies evaluate drug X as a treatment for condition Y?",
        inclusion_criteria="original human research reporting an outcome",
        exclusion_criteria="reviews, opinion pieces, non-human research",
    )


def a_subgoal():
    return Subgoal(purpose="find trials of the drug",
                   source="pubmed",
                   query="drug x AND condition y")


def a_paper():
    return PaperRecord(title="A trial of drug X",
                       origins=[SearchOrigin(source="pubmed", query="drug x")],
                       abstract="Patients given drug X recovered sooner.",
                       pmid="12345678", text_source="abstract")


def from_pubmed():
    """Where a paper came from, when the test does not care which search."""
    return [SearchOrigin(source="pubmed", query="a search")]


def a_screening():
    return Screening(paper_number=1, decision="include", relevance=8,
                     reason="a trial of the drug in patients",
                     summary="Patients given drug X recovered sooner.",
                     extract="Patients given drug X recovered sooner.")


def test_a_buffer_written_as_json_and_read_back_is_unchanged():
    """The hand-over between the two agents (UT-21, FR-17, DC-05)."""
    # The buffer is written as JSON text and read back as a checked record, so a
    # field lost or altered in either direction shows up here rather than as a
    # missing paper further along.
    buffer = ResultsBuffer(
        papers=[a_paper()],
        outcomes=[Outcome(subgoal=a_subgoal(), result="finished", papers_found=1)],
        ready=True,
    )

    read_back = ResultsBuffer.model_validate_json(buffer.model_dump_json())

    assert read_back == buffer
    assert read_back.papers[0].pmid == "12345678"
    assert read_back.outcomes[0].subgoal.query == "drug x AND condition y"


def test_a_whole_run_written_as_json_and_read_back_is_unchanged():
    """The same check for the state a run carries, which is what gets kept."""
    state = ResearchState(request=a_request(), pending_searches=[a_subgoal()],
                          results=ResultsBuffer(papers=[a_paper()]))

    read_back = ResearchState.model_validate_json(state.model_dump_json())

    assert read_back == state


def test_a_score_sent_as_text_is_refused():
    """A relevance score of "8" is not quietly turned into 8 (UT-STRICT-01)."""
    # A model that answers with text where a number was asked for has not
    # answered the question properly, and converting it would hide that.
    with pytest.raises(ValidationError):
        Screening(paper_number=1, decision="include", relevance="8",
                  reason="r", summary="s", extract="e")


def test_a_field_the_format_does_not_have_is_refused():
    """A field the model invented is refused rather than carried along."""
    with pytest.raises(ValidationError):
        Screening(paper_number=1, decision="include", relevance=8,
                  reason="r", summary="s", extract="e", confidence=0.9)


def test_a_score_outside_the_scale_is_refused():
    """Relevance runs from 0 to 10, so 11 is not a score."""
    with pytest.raises(ValidationError):
        Screening(paper_number=1, decision="include", relevance=11,
                  reason="r", summary="s", extract="e")


@pytest.mark.parametrize("move", ["continue", "refine", "drop", "finish"])
def test_each_of_the_four_moves_is_accepted(move):
    """FR-04 names four moves, so all four have to be usable."""
    # parametrize runs this once for each, so a move that stopped being accepted
    # is reported by name rather than as one failure covering all four.
    # A refinement has to name the search to run instead, so all four are given
    # one here. The other three do not need it and ignore it.
    assert NextMove(move=move, reason="because",
                    new_query="a different search").move == move


def test_a_move_that_is_not_one_of_the_four_is_refused():
    """Anything else is a model answer that cannot be acted on."""
    with pytest.raises(ValidationError):
        NextMove(move="try again later", reason="because")


def test_a_paper_has_to_say_which_service_it_came_from():
    """DC-07 asks every record to name its source, so it cannot be left out."""
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper with no service named", origins=[])


def test_a_paper_says_it_has_only_metadata_until_told_otherwise():
    """The cautious answer is the starting one (DC-07)."""
    # A record that has not been given any text should not look as though an
    # abstract was read, because the Curation Agent decides how much weight to
    # give a record from this field.
    assert PaperRecord(title="A paper", origins=from_pubmed()).text_source == "metadata"


def test_a_paper_with_no_abstract_and_no_search_behind_it_is_still_a_paper():
    """Two cases that both have to work."""
    # One relevant paper in the hand-screened sample has no abstract at all, so a
    # record without one has to be able to reach the reviewer. The recall
    # measurement also feeds the fifty labelled records straight to the Curation
    # Agent, without a search in front of them, so a record that names no query
    # has to be valid as well (SC-01).
    paper = PaperRecord(title="A paper with no abstract",
                        origins=[SearchOrigin(source="pubmed")],
                        pmid="33994641")

    assert paper.abstract == ""
    assert paper.origins[0].query == "", "no search stands behind it"


def test_two_new_papers_do_not_share_one_list_of_authors():
    """Adding an author to one record must not add it to every record."""
    # A list used as a default value on a record did exactly that earlier in this
    # project, and nothing reported it.
    first = PaperRecord(title="First paper", origins=from_pubmed())
    second = PaperRecord(title="Second paper", origins=from_pubmed())

    first.authors.append("Smith A")

    assert second.authors == []


def test_a_new_buffer_is_not_ready():
    """Curation waits for the signal, so it must start switched off (FR-17)."""
    assert ResultsBuffer().ready is False


def test_a_paper_just_screened_has_not_had_its_extract_checked():
    """Not checked and checked-but-not-found are different answers (SC-02)."""
    # None says the check has not run. False would say it ran and the extract was
    # not in the paper, which is a claim about the paper rather than about the
    # check.
    screened = ScreenedPaper(paper=a_paper(), screening=a_screening())

    assert screened.extract_found is None
    assert screened.review == "waiting"


def test_a_judgement_has_to_say_which_paper_it_is_about():
    """Papers are screened several at a time, so the link cannot be left out."""
    # Without it the only way to match a judgement to a paper is the order they
    # come back in, and the model may not return them in the order it was given.
    with pytest.raises(ValidationError):
        Screening(decision="include", relevance=8, reason="r", summary="s",
                  extract="e")


def test_a_batch_of_judgements_keeps_each_one_with_its_paper():
    """One request covers several papers, and each answer names its own."""
    batch = ScreeningList(screenings=[
        Screening(paper_number=1, decision="include", relevance=9,
                  reason="a trial", summary="s", extract="e"),
        Screening(paper_number=2, decision="exclude", relevance=1,
                  reason="a review", summary="s", extract="e"),
    ])

    numbers = []
    for screening in batch.screenings:
        numbers.append(screening.paper_number)

    assert numbers == [1, 2]
    assert batch.screenings[1].decision == "exclude"


def test_papers_are_numbered_from_one():
    """The papers are numbered as they are listed for the model, starting at
    one, so a zero is a number that was never given out."""
    with pytest.raises(ValidationError):
        Screening(paper_number=0, decision="include", relevance=8, reason="r",
                  summary="s", extract="e")


def test_a_plan_keeps_the_order_the_model_gave():
    """The plan is an ordered queue, so the order has to survive (FR-02)."""
    first = Subgoal(purpose="trials", source="pubmed", query="one")
    second = Subgoal(purpose="safety", source="semantic scholar", query="two")

    plan = SubgoalList(subgoals=[first, second])

    queries = []
    for subgoal in plan.subgoals:
        queries.append(subgoal.query)

    assert queries == ["one", "two"]


def test_a_failed_subgoal_can_be_recorded_with_what_went_wrong():
    """A search that failed has to be reportable, not silently absent (FR-05)."""
    outcome = Outcome(subgoal=a_subgoal(), result="failed",
                      reason="pubmed did not answer after three attempts")

    assert outcome.papers_found == 0
    assert "three attempts" in outcome.reason


def test_a_request_with_no_dates_set_asks_for_no_date_limit():
    """The ordinary run searches without a date limit, matching the sample."""
    # The recall figure is measured against records drawn with no date
    # restriction, so a default that quietly narrowed the search would make the
    # two measure different things (ID-02, SC-01).
    request = a_request()

    assert request.from_year is None
    assert request.to_year is None


def test_a_request_with_no_question_is_refused():
    """An empty goal would reach the Planning Agent looking like a real one."""
    with pytest.raises(ValidationError):
        ResearchRequest(question="   ", inclusion_criteria="something",
                        exclusion_criteria="something else")


def test_a_year_range_that_holds_nothing_is_refused():
    """A first year after a last year asks for a range with no years in it."""
    # The search would run, find nothing, and look like a topic with no papers.
    with pytest.raises(ValidationError):
        ResearchRequest(question="a question", inclusion_criteria="in",
                        exclusion_criteria="out", from_year=2030, to_year=2020)


def test_asking_for_no_results_at_all_is_refused():
    """A search asking for nothing is not a search."""
    with pytest.raises(ValidationError):
        ResearchRequest(question="a question", inclusion_criteria="in",
                        exclusion_criteria="out", results_per_search=0)


def test_a_refinement_without_a_new_search_is_refused():
    """Refining without saying what to run instead repeats the same search."""
    # The agent would get the same results and decide to refine again, until the
    # step limit stopped it, with nothing reporting the circle.
    with pytest.raises(ValidationError):
        NextMove(move="refine", reason="too few results")


def test_a_subgoal_that_failed_has_to_say_why():
    """The coverage report exists to say what the search did not reach."""
    with pytest.raises(ValidationError):
        Outcome(subgoal=a_subgoal(), result="failed")


def test_a_count_of_papers_cannot_be_negative():
    """A negative count would make the coverage report add up to nonsense."""
    with pytest.raises(ValidationError):
        Outcome(subgoal=a_subgoal(), result="finished", papers_found=-1)


def test_a_paper_cannot_claim_text_it_does_not_hold():
    """A record saying full text was read, holding none, would be trusted
    further than the evidence behind it (DC-07)."""
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper", origins=from_pubmed(), text_source="full text")


def test_full_text_without_its_licence_is_refused():
    """Full text is only used under a licence permitting it, so the licence
    has to travel with the text (EX-04)."""
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper", origins=from_pubmed(),
                    text_source="full text", full_text="the whole paper")


def test_a_judgement_with_no_extract_is_refused():
    """The extract is what makes a made-up claim findable (SC-03)."""
    with pytest.raises(ValidationError):
        Screening(paper_number=1, decision="include", relevance=8,
                  reason="a trial", summary="a summary", extract="  ")


def test_one_paper_cannot_be_judged_twice_in_a_batch():
    """Two answers for one paper means another paper has none, and the wrong
    summary would sit against a paper with nothing showing it (ID-12)."""
    twice = [
        Screening(paper_number=3, decision="include", relevance=9,
                  reason="r", summary="s", extract="e"),
        Screening(paper_number=3, decision="exclude", relevance=0,
                  reason="r", summary="s", extract="e"),
    ]

    with pytest.raises(ValidationError):
        ScreeningList(screenings=twice)


def test_a_run_that_stops_for_a_person_has_to_say_why():
    """An unexplained stop leaves the reviewer to read the whole run."""
    with pytest.raises(ValidationError):
        ResearchState(request=a_request(), needs_human_attention=True)


def test_a_run_cannot_start_with_no_steps_allowed():
    """A limit of zero stops the run before it does anything, which would look
    like a run that searched and found nothing."""
    with pytest.raises(ValidationError):
        ResearchState(request=a_request(), step_limit=0)


def test_a_record_cannot_be_broken_after_it_is_built():
    """Checking on creation alone leaves the record open afterwards."""
    # A step count set to text later would pass every check made when the record
    # was built, and fail somewhere else entirely.
    state = ResearchState(request=a_request())

    with pytest.raises(ValidationError):
        state.steps_taken = "many"


@pytest.mark.parametrize("missing", ["question", "inclusion_criteria",
                                     "exclusion_criteria"])
def test_a_request_missing_any_of_its_three_answers_is_refused(missing):
    """All three are needed, so each is checked rather than only the first."""
    values = {"question": "a question", "inclusion_criteria": "in",
              "exclusion_criteria": "out"}
    values[missing] = "   "

    with pytest.raises(ValidationError):
        ResearchRequest(**values)


def test_asking_for_more_results_than_a_search_returns_is_refused():
    """NCBI returns at most ten thousand records for one search (OB-01)."""
    with pytest.raises(ValidationError):
        ResearchRequest(question="a question", inclusion_criteria="in",
                        exclusion_criteria="out", results_per_search=10001)


@pytest.mark.parametrize("result", ["failed", "dropped"])
def test_a_subgoal_that_did_not_finish_has_to_say_why(result):
    """Dropped and failed both need a reason, so both are checked."""
    with pytest.raises(ValidationError):
        Outcome(subgoal=a_subgoal(), result=result)


@pytest.mark.parametrize("blank", ["reason", "summary", "extract"])
def test_a_judgement_missing_any_of_its_written_parts_is_refused(blank):
    """Each of the three is checked, not just the extract."""
    values = {"paper_number": 1, "decision": "include", "relevance": 8,
              "reason": "a trial", "summary": "a summary", "extract": "a sentence"}
    values[blank] = "  "

    with pytest.raises(ValidationError):
        Screening(**values)


def test_a_record_claiming_an_abstract_it_does_not_hold_is_refused():
    """The Curation Agent weights its judgement by this field (DC-07)."""
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper", origins=from_pubmed(), text_source="abstract")


def test_a_record_holding_full_text_cannot_say_it_has_only_metadata():
    """The opposite mistake: the curator would read less than is there."""
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper", origins=from_pubmed(),
                    text_source="metadata", full_text="the whole paper",
                    full_text_licence="a licence")


def test_a_record_holding_an_abstract_cannot_say_it_has_only_metadata():
    """Understating what is there is as misleading as overstating it."""
    # The curator weights its judgement by this field, so a record that has an
    # abstract and says otherwise would have its judgement trusted less than the
    # evidence behind it deserves.
    with pytest.raises(ValidationError):
        PaperRecord(title="A paper", origins=from_pubmed(),
                    text_source="metadata", abstract="there is one here")


def test_a_paper_with_no_title_is_refused():
    """The title is the one piece of text every record carries."""
    with pytest.raises(ValidationError):
        PaperRecord(title="   ", origins=from_pubmed())


@pytest.mark.parametrize("blank", ["purpose", "query"])
def test_a_subgoal_missing_its_purpose_or_its_query_is_refused(blank):
    """A search with no query is not a search, and one with no purpose cannot
    be reported in the coverage report."""
    values = {"purpose": "find trials", "source": "pubmed", "query": "drug x"}
    values[blank] = "  "

    with pytest.raises(ValidationError):
        Subgoal(**values)


def test_a_plan_with_no_searches_in_it_is_refused():
    """An empty plan would run, search nothing, and end with an empty buffer."""
    # That looks exactly like a question with no literature behind it. A model
    # that cannot produce a plan should reach the repair attempts and then a
    # person instead (DC-06).
    with pytest.raises(ValidationError):
        SubgoalList(subgoals=[])


def test_a_next_move_with_no_reason_is_refused():
    """The design requires a written reason for every planning decision, and
    the run report is built from them (FR-20)."""
    with pytest.raises(ValidationError):
        NextMove(move="drop", reason="  ")


def test_a_rationale_longer_than_a_sentence_is_refused_without_being_kept():
    """The model is asked for one sentence, and a request is not a check: a
    paragraph of working out would have been stored word for word, which FR-20
    says is not kept.
    """
    # The bound is 300 characters, chosen against the thirteen real answers of
    # 17 and 18 September, 123 to 195 characters each (R-14). A refusal's
    # message travels into the run's record, so it says how long the text was
    # and not what it said.
    a_sentence = "x" * LONGEST_RATIONALE
    a_paragraph = "First I weighed the results against the criteria. " * 8

    assert NextMove(move="continue", reason=a_sentence).reason == a_sentence
    with pytest.raises(ValidationError) as refused:
        NextMove(move="continue", reason=a_sentence + "x")
    assert "at most 300 characters" in str(refused.value)

    with pytest.raises(ValidationError) as refused:
        PlanningDecision(subgoal=a_subgoal(), move="continue", reason=a_paragraph)
    assert "this one is 400" in str(refused.value)
    assert "weighed the results against" not in str(refused.value)


def test_an_empty_batch_of_judgements_is_refused():
    """Nothing was screened, so there is nothing to hand to the reviewer."""
    with pytest.raises(ValidationError):
        ScreeningList(screenings=[])


def test_a_reason_to_stop_without_being_stopped_is_refused():
    """Otherwise the reason is written down and never shown to anyone."""
    with pytest.raises(ValidationError):
        ResearchState(request=a_request(), attention_reason="something happened")


def test_a_run_past_its_own_limit_is_refused():
    """If the count can pass the limit, the limit did not stop it (DC-06)."""
    with pytest.raises(ValidationError):
        ResearchState(request=a_request(), steps_taken=13, step_limit=12)


def test_a_new_run_has_a_limit_on_how_long_it_can_go_on():
    """An agent with limited time has to stop (DC-06)."""
    state = ResearchState(request=a_request())

    assert state.steps_taken == 0
    assert state.step_limit > 0


def test_a_new_run_is_not_waiting_for_a_person():
    """Nothing has gone wrong yet, so nothing should be asking for help."""
    state = ResearchState(request=a_request())

    assert state.needs_human_attention is False
    assert state.attention_reason == ""


def test_a_run_can_record_that_it_needs_a_person():
    """Figure 2 stops and asks for a person once the repairs are used up."""
    # A plan that could not be produced has to be told apart from a search that
    # ran and found nothing, so the run says which of the two happened (DC-06).
    state = ResearchState(
        request=a_request(),
        needs_human_attention=True,
        attention_reason="the plan did not fit the format after three attempts",
    )

    assert state.pending_searches == []
    assert state.needs_human_attention is True
    assert "three attempts" in state.attention_reason


# The record of what the Planning Agent decided, and where a run keeps them.

def test_a_decision_with_no_reason_is_refused():
    """A decision the run report cannot explain is worse than none (FR-20)."""
    # It reaches the report looking like a choice somebody made, with nothing
    # saying why, and a reader has no way to tell it apart from one that was
    # reasoned.
    with pytest.raises(ValidationError):
        PlanningDecision(subgoal=a_subgoal(), move="drop", reason="   ")


def test_a_decision_has_to_be_one_of_the_four_moves():
    """The four are what the graph can act on. A fifth would route nowhere."""
    with pytest.raises(ValidationError):
        PlanningDecision(subgoal=a_subgoal(), move="think about it",
                         reason="a reason")


def test_the_decisions_of_a_run_survive_being_written_out_and_read_back():
    """The buffer passes between the agents as JSON, so the decisions have to
    come back in the order they were made and with their subgoals attached.
    """
    # Order matters because the report reads them as the story of the run. A set
    # that came back shuffled would still be complete and would still be wrong.
    state = ResearchState(
        request=ResearchRequest(question="Does drug X treat condition Y?",
                                inclusion_criteria="trials",
                                exclusion_criteria="reviews"),
        decisions=[
            PlanningDecision(subgoal=a_subgoal(), move="refine",
                             reason="too many unrelated results"),
            PlanningDecision(subgoal=a_subgoal(), move="finish",
                             reason="that is enough"),
        ])

    read_back = ResearchState.model_validate_json(state.model_dump_json())

    assert [d.move for d in read_back.decisions] == ["refine", "finish"]
    assert read_back.decisions[0].reason == "too many unrelated results"
    assert read_back.decisions[0].subgoal.query == a_subgoal().query


def test_a_stored_decision_that_is_not_a_decision_is_refused():
    """A run read back from JSON is only as good as what is checked on the way
    in. Something else in the list would reach the report as a decision."""
    state = ResearchState(
        request=ResearchRequest(question="Does drug X treat condition Y?",
                                inclusion_criteria="trials",
                                exclusion_criteria="reviews"))
    written = json.loads(state.model_dump_json())
    written["decisions"] = ["not a decision at all"]

    with pytest.raises(ValidationError):
        ResearchState.model_validate(written)


@pytest.mark.parametrize("years", [
    {"from_year": 2020},
    {"to_year": 2024},
    {"from_year": 2020, "to_year": 2024},
    {"from_year": 2020, "to_year": 2020},
])
def test_a_year_limit_is_accepted_now_that_the_searches_send_it(years):
    """These were refused while nothing sent them to a search."""
    # Accepting a limit and then searching every year anyway is the worst
    # answer, because the researcher is given papers they asked not to see with
    # nothing saying the years were ignored. Both sources send them now, each in
    # its own wording, and the planner checks the years of what comes back
    # (ID-02).
    request = ResearchRequest(question="Does drug X treat condition Y?",
                              inclusion_criteria="trials",
                              exclusion_criteria="reviews", **years)

    assert request.from_year == years.get("from_year")
    assert request.to_year == years.get("to_year")


@pytest.mark.parametrize("years", [
    {"from_year": 1500}, {"to_year": 1500},
    {"from_year": 3000}, {"to_year": 3000},
], ids=["earliest first", "earliest last", "latest first", "latest last"])
def test_the_years_this_project_supports_are_accepted(years):
    """The two ends themselves, which a rule about a range often refuses."""
    request = ResearchRequest(question="Does drug X treat condition Y?",
                              inclusion_criteria="trials",
                              exclusion_criteria="reviews", **years)

    assert request.from_year == years.get("from_year")
    assert request.to_year == years.get("to_year")


@pytest.mark.parametrize("years", [
    {"from_year": 1499}, {"to_year": 1499},
    {"from_year": 3001}, {"to_year": 3001},
], ids=["before the earliest", "before the earliest, last",
        "after the latest", "after the latest, first"])
def test_a_year_outside_what_this_project_supports_is_refused(years):
    """A year outside the range became a range holding nothing."""
    # NCBI wants a first and a last date together, so the PubMed source fills an
    # open end with the same two years. A last year of 1400 asked for everything
    # from 1500 to 1400, and a first year of 3500 asked for 3500 to 3000. Neither
    # was refused, and neither could have found anything.
    with pytest.raises(ValidationError):
        ResearchRequest(question="Does drug X treat condition Y?",
                        inclusion_criteria="trials",
                        exclusion_criteria="reviews", **years)


def test_a_request_without_year_limits_is_still_accepted():
    """A request that names no years is the ordinary one."""
    request = ResearchRequest(question="Does drug X treat condition Y?",
                              inclusion_criteria="trials",
                              exclusion_criteria="reviews")

    assert request.from_year is None
    assert request.to_year is None


def test_a_partial_search_has_to_say_what_it_missed():
    """Partial is the status; the reason is what was missed. One without the
    other tells a reader that something went wrong and not what (FR-05)."""
    with pytest.raises(ValidationError):
        Outcome(subgoal=a_subgoal(), result="partial", papers_found=3)


def test_a_new_budget_has_spent_nothing():
    """The figures are this project's choices, checked rather than assumed."""
    budget = RunBudget()

    assert budget.model_request_limit == 18, \
        "eighteen leaves two of the day's twenty unspent"
    assert budget.papers_for_curation_limit == 50
    assert budget.model_requests_used == 0
    assert budget.papers_sent_to_curation == 0
    assert budget.has_a_model_request_left()


def test_spending_a_request_gives_back_a_new_budget():
    """Nothing is changed in place."""
    # The steps of the graph pass this between them, and changing a field on a
    # record that travels that way is how a value gets past its own rules.
    budget = RunBudget()

    spent = budget.with_one_model_request_used()

    assert spent.model_requests_used == 1
    assert budget.model_requests_used == 0, "the one it was made from is unchanged"
    assert spent is not budget


def test_a_budget_with_nothing_left_says_so_before_it_is_spent():
    """Asked before the request is sent, not after."""
    # A limit checked afterwards reports a request that has already gone, which
    # is a record of the allowance being used rather than a limit on it.
    budget = RunBudget(model_request_limit=2, model_requests_used=2)

    assert not budget.has_a_model_request_left()


def test_spending_a_request_the_run_did_not_have_is_refused():
    """It cannot be put right afterwards; the request is gone."""
    # What it can do is refuse to exist, so the step that overspent fails where
    # it happened rather than leaving a record saying the run kept to a limit it
    # did not.
    budget = RunBudget(model_request_limit=1, model_requests_used=1)

    with pytest.raises(ValidationError):
        budget.with_one_model_request_used()


def test_a_budget_that_has_already_overspent_cannot_be_built():
    """A record read back from JSON is only as good as what is checked on the
    way in, and this one says how much of an allowance is left."""
    with pytest.raises(ValidationError):
        RunBudget(model_request_limit=5, model_requests_used=6)
    with pytest.raises(ValidationError):
        RunBudget(papers_for_curation_limit=50, papers_sent_to_curation=51)


def test_papers_handed_to_curation_are_counted_the_same_way():
    """The paper limit is on what curation is given, not on what searching
    finds. A first search filling the buffer must not make every later search
    pointless."""
    budget = RunBudget(papers_for_curation_limit=50)

    after = budget.with_papers_sent_to_curation(20)

    assert after.papers_sent_to_curation == 20
    assert after.papers_it_can_still_take() == 30
    assert budget.papers_sent_to_curation == 0


def test_a_run_carries_a_budget_without_being_given_one():
    """Every run has one, so nothing has to remember to supply it."""
    state = ResearchState(
        request=ResearchRequest(question="Does drug X treat condition Y?",
                                inclusion_criteria="trials",
                                exclusion_criteria="reviews"))

    assert state.budget.model_request_limit == 18
    assert state.stopped_because == ""


def test_a_run_that_stopped_for_an_ordinary_reason_is_not_asking_for_a_person():
    """Spending the allowance is the limit working, not a fault."""
    # A run reporting it as needing a person would put an ordinary stop in front
    # of somebody as though something had broken.
    state = ResearchState(
        request=ResearchRequest(question="Does drug X treat condition Y?",
                                inclusion_criteria="trials",
                                exclusion_criteria="reviews"),
        stopped_because="this run has used the 18 model requests it was allowed")

    assert not state.needs_human_attention
    assert state.attention_reason == ""
