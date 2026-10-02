"""Holds the record formats the two agents pass between them (DC-05)."""
# The two agents never call each other. The Planning and Retrieval Agent writes
# papers into a buffer, and the Curation Agent reads them from it. Both sides
# agree on the formats in this file, and every record is checked against its
# format before it is passed on.
#
# A checked buffer is used instead of an agent communication language such as
# KQML. KQML is built for separate agents that hold their own knowledge and have
# to negotiate about it (Finin et al., 1994). Both agents here belong to one
# application, have fixed roles and share these formats, so there is nothing to
# negotiate and a checked hand-over is enough (DC-05).
#
# The state of a run holds the beliefs, desires and intentions a BDI agent is
# described as having (Rao and Georgeff, 1995), under names that say what each
# part is rather than which term it answers to. The mapping is set out where the
# state is defined. This is not a claim to be a BDI interpreter: there is no plan
# library and none of the reasoning cycle that model sets out (DC-02).
#
# Every paper says which service it came from and how much of it was available to
# read. The agent never sees a whole study, so it works in what Wooldridge (2009)
# calls an inaccessible environment, and recording how much was available keeps
# that limitation in front of the Curation Agent and the reviewer (DC-07).
#
# These are Pydantic classes rather than the plain records the sources use,
# because two kinds of value arrive from outside the application and cannot be
# assumed to be the right shape: answers from the language model, and papers from
# the search services.
#
# Nothing here imports another part of the application, so any part can use it.
# Turning a record from one of the search sources into the format below is left
# to the agent that retrieves it, which is where both formats are already in
# view.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _is_blank(text):
    """True when a value is empty or only spaces."""
    # Used because a field that is required but holds a space passes an ordinary
    # check for emptiness while carrying nothing a person could read.
    return not text or not text.strip()

# The publication years this project supports on a request. They are not a
# claim about what exists: they are the ends of the range this project will
# ask a service for, and the PubMed source uses the same two to fill an open
# end, because NCBI wants a first and a last date together.
#
# Written here rather than only in that source, so that a request naming a year
# outside them is refused where the request is read, rather than being turned
# into a range that holds nothing further down.
EARLIEST_YEAR = 1500
LATEST_YEAR = 3000

# The two services that are searched for papers. Crossref is not here because
# it is not searched: it supplies the publication status of a paper already
# found.
SearchableSource = Literal["pubmed", "semantic scholar"]

# How much of a paper was available to read. "metadata" means the title and
# publication details only, with no abstract at all (DC-07). "full text" is
# kept as a value a record may carry; no source fills it since the PubMed
# Central source was removed on 19 September, because no run needed it, so
# every record a run makes reads "metadata" or "abstract".
TextSource = Literal["metadata", "abstract", "full text"]

# A paper's publication status: the three ways this project classifies what
# Crossref says about a paper, and the empty string when no status was
# obtained. The sentences are this project's, not Crossref's; the
# source reads Crossref's notices and sorts them into these three. Held as
# fixed values rather than free text so that a retracted paper cannot be
# described in two different ways, and so that nothing reading the record has
# to interpret prose to find a retraction.
#
# The empty string is kept apart from "no retraction or formal concern
# identified" on purpose. The first is silence: no status was obtained,
# because the paper has no DOI to ask with, or the check failed, or the record
# has not reached the hand-over, which is where a run asks (R-18). The second
# is an answer a reviewer could rely on. Merging them would turn silence into
# reassurance.
#
# The wording is kept together here, beside the field it is written into. It
# appears twice in this block, as named constants and again inside the
# Literal. The strings are repeated so the annotation is valid for static
# type checking, which reads Literal parameters as written; at run time Python
# would accept the names. A test holds the two to each other. Everywhere
# else, sources/crossref.py and processing/dedup.py read the names from here
# rather than writing the sentences again, which is what stops them drifting
# apart.
NOT_CHECKED = ""
RETRACTED = "Retracted after publication"
CONCERNS_RAISED = "Formal concerns raised after publication"
NONE_IDENTIFIED = "No retraction or formal concern identified"
PublicationStatus = Literal[
    "",
    "Retracted after publication",
    "Formal concerns raised after publication",
    "No retraction or formal concern identified",
]

# The four moves the Planning Agent can make after seeing what a search
# returned.
Move = Literal["continue", "refine", "drop", "finish"]

# What the Curation Agent recommends for one paper. "uncertain" is one of the
# three because a first pass forced to answer include or exclude on a paper it
# cannot judge will guess, and a guess is harder for the reviewer to notice
# than an admission.
ScreeningChoice = Literal["include", "exclude", "uncertain"]

# What the reviewer decided. "waiting" is the state before anyone has looked,
# so a paper nobody has seen cannot be mistaken for one that was rejected.
ReviewChoice = Literal["waiting", "approved", "flagged", "rejected"]


class Checked(BaseModel):
    """The checking every format in this file shares."""
    # Strict checking refuses a value of the wrong type instead of converting it,
    # so a relevance score returned as the text "8" is rejected rather than
    # quietly turned into the number 8. Forbidding extra fields refuses a record
    # carrying a field the format does not have, which is how a field the model
    # invented is caught.
    #
    # The conversion that describes a format to Gemini leaves the rule about
    # extra fields out, so that rule is applied here, when the answer arrives,
    # rather than by the model.
    #
    # Checking on assignment closes part of a further hole: without it, a record
    # built correctly could have a field set to anything at all a line later.
    #
    # It does not close all of it, and the rest is worth knowing. Adding an item
    # to a list the record already holds is not an assignment, so it is not
    # checked: a search that is not a Subgoal can be appended to the list of
    # them. And an assignment refused because of a rule spanning two fields
    # leaves the refused value in place, so the record can be left in a state its
    # own rules reject. Both are reasons to check a whole record again where it
    # passes from one agent to the next, rather than trusting that it stayed as
    # it was built.

    model_config = ConfigDict(strict=True, extra="forbid", validate_assignment=True)


class ResearchRequest(Checked):
    """What the researcher asked for, and the rules papers are judged by."""
    # The criteria are written by the researcher rather than worked out by the
    # agent, because they are the part a systematic review has to be able to
    # state and defend afterwards.

    question: str
    inclusion_criteria: str
    exclusion_criteria: str
    # Limits meant for testing, left unset in an ordinary run. None means no
    # restriction, so the search matches the hand-screened sample that the
    # recall figure is measured against (SC-01).
    #
    # Both are sent to the searches now, each service in its own wording, and
    # the years of what comes back are checked against them. A request giving
    # either was refused until that was built, because taking a year from a
    # researcher and then searching every year would return papers they asked
    # not to see with nothing saying so.
    #
    # The two ends are the years this project supports, and they are the same
    # ones the PubMed source uses to fill an open end of a range. Left
    # unbounded, a last year of 1400 asked NCBI for everything from 1500 to
    # 1400, which is a range holding nothing, and a first year of 3500 asked
    # for 3500 to 3000, which is the same thing the other way round. Neither
    # was refused and neither would have found anything.
    from_year: int | None = Field(default=None, ge=EARLIEST_YEAR,
                                  le=LATEST_YEAR)
    to_year: int | None = Field(default=None, ge=EARLIEST_YEAR, le=LATEST_YEAR)
    # At least one result. The upper figure is a ceiling this project sets,
    # not a promise that any service will return that many: it is what NCBI
    # allows for one search, and the same number also drives Semantic Scholar,
    # which has its own limits and its own way of returning results in pages.
    # Each source keeps to its own rules. This only refuses a number no service
    # could sensibly be asked for (OB-01).
    results_per_search: int = Field(default=50, ge=1, le=10000)

    @model_validator(mode="after")
    def _check_the_request_can_be_acted_on(self):
        """Refuses a request that is valid in shape but impossible to act on."""
        # An empty question would reach the Planning Agent as a goal, and a first
        # year after a last year would ask for a range that holds nothing. Both
        # would produce a run that looks ordinary and finds nothing, with no
        # reason a reader could see.
        if _is_blank(self.question):
            raise ValueError("the research question cannot be empty")
        if _is_blank(self.inclusion_criteria):
            raise ValueError("the inclusion criteria cannot be empty")
        if _is_blank(self.exclusion_criteria):
            raise ValueError("the exclusion criteria cannot be empty")
        if (self.from_year is not None and self.to_year is not None
                and self.from_year > self.to_year):
            raise ValueError(
                f"the first year {self.from_year} is after the last year "
                f"{self.to_year}, so the range holds nothing")
        return self


class Subgoal(Checked):
    """One search the Planning Agent has decided to run."""
    # The language model fills all three fields. What the search then returned is
    # kept in Outcome rather than here, so the model is never asked to report on
    # its own results.

    purpose: str = Field(description="what this search is looking for, in one sentence")
    source: SearchableSource = Field(description="which service to search")
    query: str = Field(description="the search text to send to that service")

    @model_validator(mode="after")
    def _check_the_search_can_be_run_and_explained(self):
        """A subgoal with no query is not a search, and one with no purpose
        cannot be reported.
        """
        # An empty query would be sent to the service, which would either refuse
        # it or answer with everything. The purpose is what the coverage report
        # uses to say what this search was for.
        if _is_blank(self.query):
            raise ValueError("a subgoal has to say what to search for")
        if _is_blank(self.purpose):
            raise ValueError("a subgoal has to say what it is looking for")
        return self


class SubgoalList(Checked):
    """A whole plan: the searches to run, in the order to run them."""
    # The model is asked for a record holding a list rather than a bare list,
    # because a record is what a structured answer has to be. The order is the
    # order they are run in, and the ones further down may be changed by what the
    # earlier ones find.

    subgoals: list[Subgoal]

    @model_validator(mode="after")
    def _check_the_plan_has_something_in_it(self):
        """An empty plan is a failure to plan, not a plan with nothing to do."""
        # Left to pass, it would run through the loop, search nothing, and end
        # with a buffer no papers were ever written into. That looks exactly like
        # a question with no literature behind it. A model that cannot produce a
        # plan should reach the repair attempts and then a person instead
        # (DC-06).
        if not self.subgoals:
            raise ValueError("a plan with no searches in it is not a plan")
        return self


# The most a planning rationale may hold, in characters. The model is asked
# for one sentence, and a request is not a check: several paragraphs of
# working out would have been kept word for word, which is what the design
# says is not stored (R-14). Chosen on 18 September against the thirteen real
# answers read by then, from two models over six runs: every one a single
# sentence of 123 to 195 characters. Three hundred is room for a sentence
# half as long again as the longest of them. It is a limit on length and
# nothing more: a short paragraph of two or three sentences would pass it,
# and whether an answer is one sentence or working out is read from the
# answers themselves, not from the bound. A tighter bound would refuse
# an honest sentence, and a refusal is repaired by asking again, which spends
# one of a run's eighteen requests.
#
# Checked in the validators below rather than declared on the field, so the
# format sent to the model is the one that has answered on real runs: a
# declared length would be written into the JSON schema the request carries,
# and whether the service accepts that keyword could not be tried without a
# request. The prompt is not changed to name the number either, because a
# saved answer is filed under its question and a changed question would
# leave every approved demonstration answer unfound (R-12).
LONGEST_RATIONALE = 300


def _check_a_rationale(reason):
    """Refuses a missing rationale and one longer than a sentence should be."""
    # One function for both records that carry a rationale, so the two cannot
    # come to disagree about what one is. The message names the length and the
    # bound and not the text: a refusal is read back into the run's record, and
    # the text is what the bound exists to keep out (R-14).
    # The design requires a written reason for every planning decision, and
    # an empty one would leave the run report saying a subgoal was dropped
    # with nothing about why.
    if _is_blank(reason):
        raise ValueError("a planning decision has to give its reason")
    if len(reason) > LONGEST_RATIONALE:
        raise ValueError(
            f"a planning rationale is one sentence, at most {LONGEST_RATIONALE} "
            f"characters; this one is {len(reason)}")


class NextMove(Checked):
    """What to do after seeing what a search returned."""
    # This is the observing step of the reason-act-observe cycle. The next move
    # is decided by the results, not settled before the run began (DC-03).
    #
    # The reason is asked to be one sentence, kept so that a reader of the run
    # report can see why a subgoal was refined or dropped, and is refused above
    # LONGEST_RATIONALE characters. The model is asked for its decision and that
    # sentence, and not for the steps it took to reach them, which the design
    # deliberately does not store. The bound is a limit on length: a
    # short paragraph would pass it, and it does not show that what passes is
    # one sentence rather than reasoning; the thirteen real answers read on 17
    # and 18 September are what shows that so far (R-14).

    move: Move = Field(description="continue, refine, drop or finish")
    reason: str = Field(description="one sentence saying why this move was chosen")
    new_query: str = Field(
        default="",
        description="the replacement search text, needed only when refining",
    )

    @model_validator(mode="after")
    def _check_a_refinement_says_what_to_search_instead(self):
        """Refining without a new query is not a refinement."""
        # The agent would run the same search again, get the same results, and
        # decide to refine again, until the step limit stopped it. Nothing would
        # report that the run had gone in a circle.
        if self.move == "refine" and _is_blank(self.new_query):
            raise ValueError("a refinement has to give the search to run instead")
        _check_a_rationale(self.reason)
        return self


class PlanningDecision(Checked):
    """One move the Planning Agent made, kept for the run report."""
    # The state keeps only the most recent move, because that is all the step
    # choosing where to go next needs. That is not enough to report with: a run
    # of six searches would end holding one reason and having lost five, and a
    # subgoal the agent decided to drop would leave no trace at all, because the
    # outcome of a search records what the search did rather than what was
    # decided afterwards.
    #
    # So each decision is written down with the subgoal it was about. The reason
    # is the sentence the agent gave, bounded as NextMove bounds it, and not how
    # it arrived at it, which the design deliberately does not store.

    subgoal: Subgoal
    move: Move
    reason: str = Field(description="one sentence saying why this move was chosen")

    @model_validator(mode="after")
    def _check_the_decision_says_why(self):
        """A decision with no reason is one the run report cannot explain."""
        _check_a_rationale(self.reason)
        return self


class Outcome(Checked):
    """What became of one subgoal, including one that failed."""
    # A search that returned some of what it should have is recorded as partial
    # rather than finished. The difference is not a detail of wording: whatever
    # builds the coverage report, and whatever decides what to do next, has to
    # tell a complete search from a short one, and reading a sentence to work
    # that out is what a status is for.
    #
    # Failures are recorded rather than dropped. A run that searched four of its
    # five subgoals and still returned tidy summaries reads as complete, and the
    # design names that as the failure that matters most here. A search that is
    # written up has to say how many records each source returned, so one that
    # failed has to appear rather than simply be absent (Rethlefsen et al.,
    # 2021). The coverage report is built from these.

    subgoal: Subgoal
    result: Literal["finished", "partial", "dropped", "failed"]
    # A count of papers cannot be less than none of them. A negative number
    # would make the coverage report add up to something impossible.
    papers_found: int = Field(default=0, ge=0)
    # Why it was dropped, or what went wrong. A failure carries the description
    # the request handling returns after its last attempt.
    reason: str = ""

    @model_validator(mode="after")
    def _check_a_subgoal_that_did_not_finish_says_why(self):
        """A failure with no reason is a failure nobody can act on."""
        # The coverage report exists to tell a reader what the search did not
        # reach. "One subgoal failed" without saying what happened tells them
        # only that something went wrong somewhere.
        if self.result in ("failed", "partial", "dropped") and _is_blank(self.reason):
            raise ValueError(f"a subgoal recorded as {self.result} has to say why")
        return self


class SearchOrigin(Checked):
    """One search that found a paper: which service, and what was asked."""
    # A paper can be found by more than one search, and by both services. Keeping
    # a list of these rather than one service and one query means deduplication
    # can join two records without having to choose which half of where they came
    # from to forget. It is written before deduplication exists because this is
    # the format deduplication will be handed, and adding it afterwards would
    # mean changing the format and everything reading it at the same time.

    source: SearchableSource = Field(description="the service that returned it")
    # Left empty when a paper did not come from a search at all. The
    # standard-dataset measurement, `scripts/measure_against_clef.py`, fetches
    # its fifty records by identifier and hands them to the Curation Agent
    # with no search in front of them, and those records still have a service
    # behind them (DS-02). The recall measurement against the hand-screened
    # sample decides from the identifiers a search returned and hands nothing
    # to curation.
    query: str = ""


class PaperRecord(Checked):
    """One paper, in the single format both agents use (DC-07)."""
    # Each search source returns a record of its own shape. Those are turned into
    # this one as they are written into the buffer, so that the Curation Agent
    # reads one format instead of three.
    #
    # Only the title and where it came from are required. A paper with no
    # abstract and no identifier still has to reach the reviewer to be seen: one relevant
    # paper in the hand-screened sample has no abstract at all, and dropping such
    # a paper would decide relevance on what the record happens to carry rather
    # than on what the study did. Checking that an accepted paper has an
    # identifier that resolves belongs with the other checks, in
    # processing/validation.py (SC-02).

    title: str
    # Every search that found this paper. At least one, because a record with
    # no service behind it cannot be traced to anything (DC-07).
    origins: list[SearchOrigin]
    abstract: str = ""
    # Empty in every record a run makes: no source retrieves full text since
    # the PubMed Central source was removed on 19 September. The field stays
    # so the record's format does not change, and the licence stays with it
    # because only openly licensed full text may be held at all.
    full_text: str = ""
    full_text_licence: str = ""
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    journal: str = ""
    publication_types: list[str] = Field(default_factory=list)
    pmid: str = ""
    doi: str = ""
    # Semantic Scholar gives every paper an identifier of its own, and for some
    # papers it is the only identifier there is.
    paper_id: str = ""
    # One of the three ways this project classifies what Crossref says, or the
    # empty string when no status was obtained: the paper has no DOI, or its
    # check failed, or it has not reached the hand-over, where the asking
    # happens (R-18). Anything else is refused when the record is built, so a
    # retracted paper says so in one wording wherever it is shown, and silence
    # is never mistaken for an answer.
    publication_status: PublicationStatus = NOT_CHECKED
    text_source: TextSource = "metadata"

    @model_validator(mode="after")
    def _check_the_record_has_the_text_it_claims(self):
        """A record must not say it holds more than it does (DC-07)."""
        # The Curation Agent decides how much weight to give a paper from this
        # field. A record saying full text was read, with no full text in it,
        # would have its judgement trusted further than the evidence behind it.
        if not self.origins:
            raise ValueError(
                "a paper has to say which search found it, or it cannot be "
                "traced back to anything (DC-07)")
        if _is_blank(self.title):
            raise ValueError(
                "a paper has to have a title, because it is the one piece of "
                "text every record carries and the reviewer reads first")
        if self.text_source == "abstract" and _is_blank(self.abstract):
            raise ValueError("a record saying it has an abstract has to hold one")
        if self.text_source == "metadata" and not _is_blank(self.full_text):
            raise ValueError(
                "a record holding full text must not say it has only metadata, "
                "or the Curation Agent will read less of it than is there")
        if self.text_source == "metadata" and not _is_blank(self.abstract):
            raise ValueError(
                "a record holding an abstract must not say it has only "
                "metadata, for the same reason")
        if self.text_source == "full text":
            if _is_blank(self.full_text):
                raise ValueError("a record saying it has full text has to hold it")
            if _is_blank(self.full_text_licence):
                raise ValueError(
                    "full text is only used under a licence that permits it, so "
                    "the licence has to be recorded with the text (EX-04)")
        return self


class Screening(Checked):
    """The Curation Agent's judgement on one paper."""
    # The summary is extractive: it is built from the paper's own wording rather
    # than written afresh, and the extract is the sentence it rests on, copied
    # from the paper. A claim that is not in the paper can then be found by
    # looking for it, instead of being merely unlikely (SC-03).
    #
    # The identifier is not asked for. It is already known from the record, and
    # asking the model for it would be inviting it to supply one.
    #
    # Relevance is a whole number from 0 to 10. A finer scale would suggest the
    # model can tell 71 from 73 out of a hundred, which it cannot. Papers sharing
    # a score are expected, so ranking them needs a settled way of breaking a
    # tie.

    # Links this judgement to the correct paper in the request. The number is
    # used because the model may return its answers in a different order.
    paper_number: int = Field(
        ge=1, description="the number of the paper this judgement is for, as it was given"
    )
    decision: ScreeningChoice = Field(description="include, exclude or uncertain")
    relevance: int = Field(
        ge=0, le=10, description="0 for irrelevant, 10 for directly relevant"
    )
    reason: str = Field(description="one sentence for the reviewer, in plain words")
    summary: str = Field(description="a short summary built from the paper's own wording")
    extract: str = Field(
        description="the sentence from the paper that supports the summary, copied word for word"
    )

    @model_validator(mode="after")
    def _check_the_judgement_says_something(self):
        """A judgement with nothing written in it is worse than none at all."""
        # It reaches the reviewer looking like a decision that was made, and
        # there is nothing to check the extract against or to disagree with.
        if _is_blank(self.reason):
            raise ValueError("a judgement has to give a reason")
        if _is_blank(self.summary):
            raise ValueError("a judgement has to give a summary")
        if _is_blank(self.extract):
            raise ValueError(
                "a judgement has to give the extract its summary rests on, "
                "because that is what makes a made-up claim findable (SC-03)")
        return self


class ScreeningList(Checked):
    """The judgements returned for a batch of papers in one request."""
    # Papers are screened several at a time because the free tier allows twenty
    # model requests a day. One request for each paper would spend a whole day's
    # allowance on a single run, and a run of forty papers could not be made at
    # all (OB-07).

    screenings: list[Screening]

    @model_validator(mode="after")
    def _check_no_paper_is_judged_twice(self):
        """Two judgements for one paper means another paper has none."""
        # The wrong summary would then sit against a paper, and the reviewer
        # would have no way of telling. What this cannot check is the other half:
        # whether every paper sent was answered at all. That needs the batch that
        # was sent, which only the Curation Agent has, so it is checked there.
        if not self.screenings:
            raise ValueError("a batch of judgements cannot be empty")

        seen = []
        for screening in self.screenings:
            if screening.paper_number in seen:
                raise ValueError(
                    f"paper {screening.paper_number} was judged more than once")
            seen.append(screening.paper_number)
        return self


class ScreenedPaper(Checked):
    """A paper, what the Curation Agent made of it, and what the reviewer said."""
    # The three are kept together because the reviewer decides on the paper and
    # the judgement as one thing, and because storage saves the pair.

    paper: PaperRecord
    screening: Screening
    # None means the extract has not been looked for yet. Starting at false
    # would make a check that never ran look like a check that ran and found
    # nothing, and those two need different handling (SC-02).
    extract_found: bool | None = None
    review: ReviewChoice = "waiting"
    # What the reviewer wrote when flagging or rejecting. The Curation Agent is
    # given it when the paper comes back for revision.
    review_note: str = ""


class ResultsBuffer(Checked):
    """What the Planning Agent hands to the Curation Agent (DC-05)."""
    # Papers are added as each search returns, and ready is set when no more
    # searching will happen. Curation waits for that signal, so it cannot start
    # on a half-filled buffer and report on a search that is still running.
    #
    # Ready does not mean the plan was finished. A run stopped by its own limit
    # hands over what it has, with the searches it never reached still on the
    # queue, because that is the case the coverage report has to be able to
    # describe (DC-06). Anything reading ready as a promise of complete
    # coverage would be reading it wrongly: what was and was not searched is in
    # the outcomes and in the searches still waiting.
    #
    # While searching is going on, papers holds everything the searches
    # returned. At the hand-over it is narrowed to the papers curation is given:
    # the repeated records are combined, and the screening allowance decides how
    # many of what is left are passed on. The two lists below
    # keep what that narrowing took out, so the buffer says what was retrieved
    # as well as what was screened.
    #
    # Both are empty until the hand-over, because until then nothing has been
    # taken out. They hold whole records rather than counts. A reviewer told
    # that twelve papers were not screened, with nothing to look at, cannot
    # check the claim, and a summary built only from what was screened would
    # read as though the rest had never been found.
    #
    # This buffer stays out of any public release whatever it holds, because it
    # carries the abstracts and full text a run retrieved.

    papers: list[PaperRecord] = Field(default_factory=list)
    outcomes: list[Outcome] = Field(default_factory=list)
    ready: bool = False
    # Everything the searches returned, kept as it arrived, before repeated
    # records were combined. Deduplication has to choose one value where two
    # records disagree, and choosing is not the same as the other value being
    # wrong, so the records it chose between are kept rather than described.
    papers_before_deduplication: list[PaperRecord] = Field(default_factory=list)
    # The combined papers the screening allowance had no place for. They were
    # retrieved and deduplicated like the rest, and the only thing that
    # happened to them is that the run ran out of places.
    papers_beyond_the_screening_limit: list[PaperRecord] = Field(default_factory=list)
    # One line for each paper curation was given whose publication status
    # check failed, naming the DOI and what went wrong. A failed check leaves
    # the status blank, which is what a paper with no DOI carries as well, so
    # without this list the coverage report could not tell a check that failed
    # from a paper that could not be asked about (R-18). Empty
    # until the hand-over, which is where the asking happens.
    status_checks_failed: list[str] = Field(default_factory=list)


class RunBudget(Checked):
    """What one run may spend, and what it has spent so far (OB-07)."""
    # The free allowance for the model is twenty requests a day. A run that spent
    # them all would leave nothing for a second attempt, or for the demonstration
    # it is being built for, and the limit on the loop does not bound this: the
    # loop counts searches, while planning is one request, each move after a
    # search is another, and every repair is another again.
    #
    # Eighteen leaves two of the day's allowance unspent. Fifty papers is what
    # the curation step may be given, and it is a limit on that step rather than
    # on searching, so a first search filling the buffer does not make every
    # later search pointless.
    #
    # Neither number is a finding. They are this project's choices, written here
    # so that a run keeping to them can be checked rather than assumed.
    #
    # Nothing here is changed in place. A step that spends a request builds a new
    # budget and hands it back, and the new one is checked the way every other
    # record is. Changing a field on a record that steps of a graph pass between
    # them is how a value gets past its own rules, which this project has already
    # written down once.

    model_request_limit: int = Field(default=18, ge=1)
    model_requests_used: int = Field(default=0, ge=0)
    papers_for_curation_limit: int = Field(default=50, ge=1)
    papers_sent_to_curation: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_nothing_was_spent_that_was_not_allowed(self):
        """A budget that has overspent is a budget nothing was checking."""
        # It cannot be corrected after the fact: the requests are gone. What it
        # can do is refuse to exist, so the step that spent them fails where it
        # happened rather than leaving a record saying the run kept to a limit it
        # did not.
        if self.model_requests_used > self.model_request_limit:
            raise ValueError(
                f"{self.model_requests_used} model requests have been used "
                f"against a limit of {self.model_request_limit}")
        if self.papers_sent_to_curation > self.papers_for_curation_limit:
            raise ValueError(
                f"{self.papers_sent_to_curation} papers have been sent for "
                f"curation against a limit of "
                f"{self.papers_for_curation_limit}")
        return self

    def has_a_model_request_left(self):
        """True when the run may ask the model once more."""
        # Asked before the request is sent, not after. A limit checked afterwards
        # reports a request that has already been spent, which is a record of the
        # allowance going rather than a limit on it.
        return self.model_requests_used < self.model_request_limit

    def with_one_model_request_used(self):
        """The same budget with one more model request spent."""
        # A new record rather than a changed one. It is checked on the way in
        # like any other, so spending a request the run did not have raises here
        # instead of being written down as though it were allowed.
        return RunBudget(
            model_request_limit=self.model_request_limit,
            model_requests_used=self.model_requests_used + 1,
            papers_for_curation_limit=self.papers_for_curation_limit,
            papers_sent_to_curation=self.papers_sent_to_curation,
        )

    def papers_it_can_still_take(self):
        """How many more papers curation may be given."""
        return self.papers_for_curation_limit - self.papers_sent_to_curation

    def with_papers_sent_to_curation(self, how_many):
        """The same budget with that many more papers handed to curation."""
        return RunBudget(
            model_request_limit=self.model_request_limit,
            model_requests_used=self.model_requests_used,
            papers_for_curation_limit=self.papers_for_curation_limit,
            papers_sent_to_curation=self.papers_sent_to_curation + how_many,
        )


class ResearchState(Checked):
    """Everything one run carries from step to step (DC-02)."""
    # The first three parts are the beliefs, desires and intentions a BDI agent
    # is described as having (Rao and Georgeff, 1995), under names that say what
    # each one holds:
    #
    #     request           the objective, which is the desire
    #     pending_searches  the searches committed to, which are the intentions
    #     results           what searching has returned, which are the beliefs
    #
    # They are named this way round on purpose. The design calls for a planning
    # state that is BDI-inspired rather than a BDI interpreter, so the test of it
    # is whether the state behaves that way, not whether the fields carry the
    # vocabulary. Committing to a queue of searches and working through it is
    # what stops the agent reconsidering the whole plan at every step, which is
    # the job Bratman, Israel and Pollack (1988, p. 349) give to a plan in an
    # agent with limited time. That behaviour is the claim; the names only have
    # to be readable.
    #
    # One record holds the whole run, including what the Curation Agent adds,
    # because both agents are steps in one graph and a graph keeps one state.

    request: ResearchRequest
    # Names this execution, not this question. The run log is shared by every
    # run on the machine, and a decision recorded an hour later has to be
    # tied to the run it was made in rather than to the question, which two
    # runs can share. Empty until a run starts, because a state built
    # for a test is not an execution.
    run_id: str = ""
    pending_searches: list[Subgoal] = Field(default_factory=list)
    results: ResultsBuffer = Field(default_factory=ResultsBuffer)
    screened: list[ScreenedPaper] = Field(default_factory=list)
    # Every move the Planning Agent made, in the order it made them. This is
    # what the run report is written from. It is separate from last_move
    # because the two are for different things: one is the record, the other is
    # what the next step reads.
    decisions: list[PlanningDecision] = Field(default_factory=list)
    # The decision made after the most recent search.
    # It is stored in the state because the routing step can only use
    # information available in the state. None means that no search has been
    # completed yet (DC-03).
    last_move: NextMove | None = None
    # Set to true when the model still cannot produce a correctly formatted
    # plan after the three repair attempts allowed. The run then stops and asks
    # for a person, which is what Figure 2 requires once the repairs are used
    # up. Without it, having no subgoals could be mistaken for a successful
    # search that found no papers (DC-06).
    needs_human_attention: bool = False
    # What went wrong, in one sentence, so that whoever picks the run up does
    # not have to read the whole of it to find out.
    attention_reason: str = ""
    # What one run may spend, and what it has spent (OB-07). Carried in the
    # state because the steps of the graph have nothing else in common, and
    # a limit each step kept for itself would be a limit on each step.
    budget: RunBudget = Field(default_factory=RunBudget)
    # Why a run stopped for a reason that is not a fault. Spending the model
    # requests the run was allowed is the first of these: it is the limit doing
    # what it is for, and reporting it as needing a person would put an
    # ordinary stop in front of somebody as though something had gone wrong.
    # Empty when the run has not stopped that way.
    stopped_because: str = ""
    steps_taken: int = Field(default=0, ge=0)
    # An agent with limited time has to stop and act on what it has, so the
    # loop cannot run without end (DC-06). Twelve allows a plan of about six
    # searches with a few refinements among them. The number matters less than
    # there being one, but it cannot be none: a limit of zero would stop the
    # run before it had done anything, and look like a run that found nothing.
    step_limit: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _check_a_run_that_stopped_says_why(self):
        """A run asking for a person has to say what it is asking about."""
        # Otherwise the interface shows a stop with no explanation, and whoever
        # picks it up has to read the whole run to find out what happened.
        if self.needs_human_attention and _is_blank(self.attention_reason):
            raise ValueError(
                "a run that stops and asks for a person has to say why")
        if not self.needs_human_attention and not _is_blank(self.attention_reason):
            raise ValueError(
                "a run carrying a reason to stop has to be marked as stopped, "
                "or the reason will be written down and never shown")
        if self.steps_taken > self.step_limit:
            raise ValueError(
                f"the run has taken {self.steps_taken} steps against a limit of "
                f"{self.step_limit}, so the limit did not stop it (DC-06)")
        return self
