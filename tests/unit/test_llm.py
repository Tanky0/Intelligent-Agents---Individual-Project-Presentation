"""Tests for the one place the application talks to the model (TC-02)."""
# None of these call the real model. Two reasons: the free allowance is counted
# per day and would be spent by running the tests, and a model gives a slightly
# different answer each time, so a test that called it could pass or fail for
# reasons that have nothing to do with this code.

import json
import time

import httpx
import pytest
import respx
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, field_validator

from research_agent import llm
from research_agent.processing import run_log
from research_agent.schemas import NextMove, RunBudget, ScreeningList
from research_agent.sources import saved_replies
from research_agent.sources.http import RateLimiter


# A fixture is preparation that pytest runs for a test. autouse=True means it
# runs for every test in this file without each one asking for it.
@pytest.fixture(autouse=True)
def temporary_saved_replies_folder(tmp_path, monkeypatch):
    """Saves answers into a folder that only lasts for one test."""
    # tmp_path is an empty folder pytest makes and deletes again. monkeypatch
    # replaces a value for one test and puts the original back afterwards.
    # Together they stop a test from overwriting the answers saved for a
    # demonstration.
    monkeypatch.setattr(saved_replies, "CAPTURED_DIR", tmp_path / "captured")
    monkeypatch.setattr(saved_replies, "APPROVED_DIR", tmp_path / "approved")


@pytest.fixture(autouse=True)
def do_not_wait_between_requests(monkeypatch):
    """Removes the wait the free allowance needs, for the tests only."""
    # The model is limited to five requests a minute, so the application leaves
    # twelve seconds between them. A test that made two calls would sit waiting
    # for that gap while making no request at all.
    monkeypatch.setattr(llm, "_LIMITER", RateLimiter(1000.0))


# An approval has to say which commit was being reviewed and why what it holds
# may be published. Both are supplied here rather than left out, because a test
# that approved without them would be exercising a path the command no longer
# has. They are invented, and say so: these replies are invented too.
_REVIEWED_AT = {"commit": "0" * 40, "working_tree": "invented for a test"}

_WHY_THESE_MAY_BE_PUBLISHED = {
    "kind": "written permission",
    "granted_by": "the author of this test, whose replies hold nobody's words",
    "granted_on": "when the test was written",
    "covers": "the invented model answers this test file writes",
    "evidence": "this file",
    "grantor_could_authorise": True,
    "redaction": "nothing was taken out; there was nothing in it to take",
}


def _approve_what_was_captured():
    """Approves what was captured, the way the approval command does."""
    # It calls the same checks and writes the same manifest, so a test is not
    # exercising a path the application does not have. Copying the files across
    # by hand would have been simpler and would have proved less.
    manifest = saved_replies.read_manifest()
    for path in saved_replies.files_in(saved_replies.CAPTURED_DIR):
        # A basis is written for one reply, so it names the one it is for.
        basis = {**_WHY_THESE_MAY_BE_PUBLISHED, "applies_to": path.name}
        refused = saved_replies.approve_reply(path, manifest, _REVIEWED_AT, basis)
        assert not refused, f"{path.name} was refused: {refused}"
    assert saved_replies.write_manifest(manifest), "the manifest did not write"


def test_a_saved_answer_is_replayed_without_calling_the_model(monkeypatch):
    """The point of DEMO_MODE: the same run repeats without spending anything."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    saved_replies.save_model_reply(llm.settings.gemini_model, "what is two plus two", "four")
    _approve_what_was_captured()

    assert llm.ask("what is two plus two") == "four"


def test_an_unsaved_question_returns_nothing_rather_than_guessing(monkeypatch):
    """An agent continuing without an answer would look complete but not be."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)

    assert llm.ask("a question never asked before") is None


def test_the_same_question_and_model_find_the_same_answer():
    saved_replies.save_model_reply("model-a", "a question", "an answer")
    _approve_what_was_captured()

    assert saved_replies.load_model_reply("model-a", "a question")["answer"] == "an answer"


def test_the_same_question_to_a_different_model_is_a_different_answer():
    """Replaying one model's answer as another's would misreport what was used."""
    saved_replies.save_model_reply("model-a", "a question", "an answer")
    _approve_what_was_captured()

    assert saved_replies.load_model_reply("model-b", "a question") is None


def test_changing_one_word_of_a_question_no_longer_matches():
    """A saved answer must belong to the exact question it was given for."""
    saved_replies.save_model_reply("model-a", "screen this abstract", "include")
    _approve_what_was_captured()

    assert saved_replies.load_model_reply("model-a", "screen that abstract") is None


def test_an_answer_is_saved_only_when_asked_for(monkeypatch):
    """Saving on every call would fill the folder during ordinary running."""
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    monkeypatch.setattr(llm, "_connection", lambda: _PretendModel("an answer"))

    llm.ask("a question")

    assert saved_replies.count() == 0

    monkeypatch.setattr(llm.settings, "record_replies", True)
    llm.ask("a question")

    assert saved_replies.count() == 1


def test_a_structured_response_is_checked_before_it_is_returned(monkeypatch):
    """A response that does not fit the schema must not reach an agent (FR-17)."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)

    saved_under = "list two things\n[schema:TwoThings]"
    saved_replies.save_model_reply(llm.settings.gemini_model, saved_under,
                                '{"first": "a", "second": "b"}')
    _approve_what_was_captured()

    answer, _ = llm.ask_for("list two things", TwoThings)

    assert answer.first == "a"
    assert answer.second == "b"


def test_a_saved_response_that_no_longer_fits_the_schema_is_refused(monkeypatch):
    """A schema can change after a response was saved. The old one must not pass."""
    # It is refused in the same way a live request that could not be answered is,
    # by returning a failure rather than raising. The two ways of running the
    # application then fail alike, and the one that would otherwise end in a
    # stack trace is the one a demonstration uses.
    monkeypatch.setattr(llm.settings, "demo_mode", True)

    saved_under = "list two things\n[schema:TwoThings]"
    saved_replies.save_model_reply(llm.settings.gemini_model, saved_under,
                                '{"first": "a"}')
    _approve_what_was_captured()

    answer, _ = llm.ask_for("list two things", TwoThings)

    assert isinstance(answer, llm.ModelFailure)
    assert "ValidationError" in answer.reason, "the kind of problem is kept"


def test_a_refused_request_is_reported_rather_than_stopping_the_run(monkeypatch):
    """The daily allowance runs out, and waiting will not bring it back."""
    # A run that stopped with an error at that point would lose every paper it
    # had already screened. The failure is returned instead, so the run can
    # report what it could not do (FR-05).
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    monkeypatch.setattr(llm, "_connection", lambda: _RefusingModel())

    answer = llm.ask("a question")

    assert isinstance(answer, llm.ModelFailure)
    assert "allowance" in answer.reason
    assert "RuntimeError" in answer.reason, "the kind of problem is kept too"


def test_a_refused_structured_request_is_reported_in_the_same_way(monkeypatch):
    """Both ways of asking have to fail alike, or a caller has to handle two."""
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    monkeypatch.setattr(llm, "_connection", lambda: _RefusingModel())

    answer, _ = llm.ask_for("list two things", TwoThings)

    assert isinstance(answer, llm.ModelFailure)


def test_nothing_is_saved_when_a_request_was_refused(monkeypatch):
    """A failure is not an answer, so it must not be replayed as one."""
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", True)
    monkeypatch.setattr(llm, "_connection", lambda: _RefusingModel())

    llm.ask("a question")

    assert saved_replies.count() == 0


def test_an_answer_saved_inside_a_run_carries_that_run(monkeypatch):
    """The file name cannot say which run an answer came from: it is a hash of the
    model and the question, so a later run overwrites the earlier answer to the
    same question.
    """
    # The capture time cannot say it either, because a time names a day and not
    # an execution (ID-20).
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", True)
    monkeypatch.setattr(llm, "_connection", lambda: _PretendModel("an answer"))

    with run_log.while_running("run one"):
        llm.ask("a question")

    assert _the_one_captured_record()["run"] == "run one"


def test_a_structured_answer_saved_inside_a_run_carries_it_too(monkeypatch):
    """An answer is saved from two places, and a check that every replayed
    reply came from one run would pass on one place and miss the other."""
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", True)
    monkeypatch.setattr(llm, "_connection", lambda: _PretendStructuredModel(
        TwoThings(first="a", second="b")))

    with run_log.while_running("run one"):
        llm.ask_for("list two things", TwoThings)

    assert _the_one_captured_record()["run"] == "run one"


def test_an_answer_saved_outside_any_run_has_no_run(monkeypatch):
    """Left out rather than written blank, so an answer captured outside a run
    reads the same as one captured before the field existed.
    """
    # No run was recorded for either, the record cannot say which of the two it
    # was, and the check over the approved set says so in those words.
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", True)
    monkeypatch.setattr(llm, "_connection", lambda: _PretendModel("an answer"))

    llm.ask("a question")

    assert "run" not in _the_one_captured_record()


def _the_one_captured_record():
    [path] = saved_replies.files_in(saved_replies.CAPTURED_DIR)
    return json.loads(path.read_text(encoding="utf-8"))


class TwoThings(BaseModel):
    first: str
    second: str


class _RefusingModel:
    """Stands in for a model that refuses, as it does once the allowance goes."""

    def invoke(self, question):
        raise RuntimeError("the daily allowance for this project is spent")

    def with_structured_output(self, schema):
        return self


class _PretendModel:
    """Stands in for the model library so no request is made."""

    def __init__(self, answer):
        self._answer = answer

    def invoke(self, question):
        return _PretendReply(self._answer)


class _PretendReply:
    def __init__(self, text):
        self.text = text


class _PretendStructuredModel:
    """Stands in for the model library when an answer in a schema is asked for."""

    def __init__(self, answer):
        self._answer = answer

    def with_structured_output(self, schema):
        return self

    def invoke(self, question):
        return self._answer


class _ModelAnsweringWithText:
    """Stands in for the model library answering a structured request with a
    given text, read by the library's own parser, so a refusal raises what the
    real connection raises, with the same error chained underneath it."""

    def __init__(self, text):
        self._text = text
        self._parser = None

    def with_structured_output(self, schema):
        self._parser = PydanticOutputParser(pydantic_object=schema)
        return self

    def invoke(self, question):
        return self._parser.parse_result([Generation(text=self._text)])


def test_a_refused_rationale_is_reported_without_the_text_that_was_refused(monkeypatch):
    """A refused rationale must not reach the log through its failure (R-28)."""
    # The length check refuses this paragraph; it does not detect paragraphs.
    # The parser and its chained error both contain input that must stay out
    # of the reason eventually written into the state and run log.
    paragraph = "First I weighed the results against the criteria. " * 8
    monkeypatch.setattr(llm, "_connection", lambda: _ModelAnsweringWithText(
        json.dumps({"move": "continue", "reason": paragraph})))

    answer, _ = llm.ask_for("what next", NextMove, RunBudget())

    assert isinstance(answer, llm.ModelFailure)
    assert answer.worth_asking_again, "a repair can still put it right"
    assert "weighed the results" not in answer.reason
    assert answer.reason == (
        "OutputParserException: the answer was not in the format asked for")


def test_an_answer_that_is_not_json_is_reported_without_its_text(monkeypatch):
    """The same message quotes an answer that could not be read at all, which
    is where a model's working out would arrive if it wrote prose instead of
    the record asked for."""
    monkeypatch.setattr(llm, "_connection", lambda: _ModelAnsweringWithText(
        "Let me think this through first: {"))

    answer, _ = llm.ask_for("what next", NextMove, RunBudget())

    assert isinstance(answer, llm.ModelFailure)
    assert answer.worth_asking_again
    assert "think this through" not in answer.reason
    assert answer.reason == (
        "OutputParserException: the answer was not in the format asked for")


def test_a_short_refused_value_is_not_kept_either(monkeypatch):
    """The validation error's own message quotes each refused value cut to
    about fifty characters, so a short one would have been kept whole."""
    # A long-value test can miss a retained short value, so this answer is
    # short enough for the library's diagnostic to quote it whole (R-28).
    monkeypatch.setattr(llm, "_connection", lambda: _ModelAnsweringWithText(
        json.dumps({"move": "wander", "reason": "because it seemed best"})))

    answer, _ = llm.ask_for("what next", NextMove, RunBudget())

    assert isinstance(answer, llm.ModelFailure)
    assert "wander" not in answer.reason
    assert "seemed best" not in answer.reason
    assert answer.reason == (
        "OutputParserException: the answer was not in the format asked for")


def test_a_batch_refused_for_one_field_carries_no_extract(monkeypatch):
    """A batch of judgements may hold ten extracts, and the refusal of one field
    would have carried every value the error quoted into the run log."""
    # One wrong field must not bring either paper's extract into the reason.
    # The second judgement is out of range; the first remains valid (R-28).
    batch = {"screenings": [
        {"paper_number": 1, "decision": "include", "relevance": 8,
         "reason": "an invented reason", "summary": "an invented summary",
         "extract": "an invented sentence from a paper"},
        {"paper_number": 2, "decision": "exclude", "relevance": 11,
         "reason": "another invented reason", "summary": "another invented summary",
         "extract": "another invented sentence from a paper"},
    ]}
    monkeypatch.setattr(llm, "_connection", lambda: _ModelAnsweringWithText(
        json.dumps(batch)))

    answer, _ = llm.ask_for("screen these", ScreeningList, RunBudget())

    assert isinstance(answer, llm.ModelFailure)
    assert "invented" not in answer.reason
    assert answer.reason == (
        "OutputParserException: the answer was not in the format asked for")


def test_a_saved_answer_refused_directly_keeps_none_of_its_text(monkeypatch):
    """In demonstration mode the format check raises its error directly, not
    through the parser, and that path has to keep the text out too."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    saved_under = "what next\n[schema:NextMove]"
    saved_replies.save_model_reply(llm.settings.gemini_model, saved_under,
                                json.dumps({"move": "invented refused move",
                                            "reason": "invented private rationale"}))
    _approve_what_was_captured()

    answer, _ = llm.ask_for("what next", NextMove)

    assert isinstance(answer, llm.ModelFailure)
    assert "invented" not in answer.reason
    assert answer.reason == (
        "ValidationError: the answer was not in the format asked for")


def test_a_validator_message_cannot_carry_the_refused_value(monkeypatch):
    """Removing only the error's input field still leaves this leak (R-28)."""
    # A validator may repeat input in its own message. Using the real parser
    # here tests the chained error too, without making a model request.
    class RefusedAnswer(BaseModel):
        reason: str

        @field_validator("reason")
        @classmethod
        def refuse(cls, value):
            raise ValueError(f"not accepted: {value}")

    monkeypatch.setattr(llm, "_connection", lambda: _ModelAnsweringWithText(
        json.dumps({"reason": "invented private extract"})))

    answer, _ = llm.ask_for("what next", RefusedAnswer, RunBudget())

    assert isinstance(answer, llm.ModelFailure)
    assert answer.worth_asking_again
    assert "invented" not in answer.reason
    assert "not accepted" not in answer.reason
    assert answer.reason == (
        "OutputParserException: the answer was not in the format asked for")


def test_a_saved_reply_holding_no_answer_is_reported_rather_than_raising(monkeypatch):
    """A saved reply is a file, and a file can be damaged (FR-05)."""
    # Reading the answer straight out of it would stop a run with a message about
    # a missing key, when what happened is that a recording cannot be replayed.
    # That is the same kind of problem as an answer that no longer fits its
    # schema, and it is reported the same way.
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    monkeypatch.setattr(saved_replies, "load_model_reply",
                        lambda model, question: {"captured_at": "yesterday"})

    answer = llm.ask("a question")

    assert isinstance(answer, llm.ModelFailure)
    assert "no answer to replay" in answer.reason


def test_a_saved_answer_that_is_not_text_is_reported_in_the_same_way(monkeypatch):
    """An answer left as a number would be read as text a step later."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    monkeypatch.setattr(saved_replies, "load_model_reply",
                        lambda model, question: {"answer": 42})

    answer, _ = llm.ask_for("list two things", TwoThings)

    assert isinstance(answer, llm.ModelFailure)
    assert "no answer to replay" in answer.reason


def test_a_saved_answer_that_is_blank_is_reported_rather_than_replayed(monkeypatch):
    """An answer of nothing but spaces is not an answer."""
    # Replayed as one, it would reach an agent as an empty reply and be acted on
    # as though the model had said something. That is the quietest of the three
    # ways a damaged recording can fail, so it is worth closing (FR-05).
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    monkeypatch.setattr(saved_replies, "load_model_reply",
                        lambda model, question: {"answer": "   "})

    answer = llm.ask("a question")

    assert isinstance(answer, llm.ModelFailure)
    assert "no answer to replay" in answer.reason


def test_a_live_answer_that_is_not_the_format_asked_for_is_reported(monkeypatch):
    """A request that did not raise is not an answer in the right format."""
    # Left unchecked, something else coming back would travel on and be described
    # to the agent as demonstration mode having no saved answer, which is a
    # different problem with a different fix. Nothing is recorded either, because
    # what came back is not an answer to replay.
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", True)
    monkeypatch.setattr(llm, "_connection", lambda: _ModelReturningNothing())

    answer, _ = llm.ask_for("list two things", TwoThings)

    assert isinstance(answer, llm.ModelFailure)
    assert "NoneType" in answer.reason
    assert "TwoThings" in answer.reason
    assert saved_replies.count() == 0, "nothing was recorded"


class _ModelReturningNothing:
    """A model whose structured request succeeds but answers with nothing."""

    def with_structured_output(self, schema):
        return self

    def invoke(self, question):
        return None


def test_a_run_with_nothing_left_is_refused_before_a_request_is_sent(monkeypatch):
    """The refusal comes back instead of the request going out."""
    # Nothing here went wrong. The run spent what it was given, so the answer
    # says so rather than looking like a service that could not be reached, and
    # it does not suggest trying again: another attempt would need a request the
    # run does not have.
    #
    # Demonstration mode is on and nothing is saved, so if this ever did reach
    # the reading of a saved answer the test would see None instead.
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    spent = RunBudget(model_request_limit=2, model_requests_used=2)

    answer, budget = llm.ask_for("list two things", TwoThings, spent)

    assert isinstance(answer, llm.OutOfBudget)
    assert "2 model requests" in answer.reason
    assert not answer.worth_asking_again
    assert budget.model_requests_used == 2, "nothing further was spent"


def test_a_replayed_answer_spends_a_request_like_a_live_one(monkeypatch):
    """A demonstration whose live equivalent would run out is not one."""
    # Counting only the requests that leave the computer would let a saved run
    # of thirty answers look like something the allowance could have paid for.
    monkeypatch.setattr(llm.settings, "demo_mode", True)
    saved_under = "list two things\n[schema:TwoThings]"
    saved_replies.save_model_reply(llm.settings.gemini_model, saved_under,
                                   '{"first": "a", "second": "b"}')
    _approve_what_was_captured()

    answer, budget = llm.ask_for("list two things", TwoThings, RunBudget())

    assert answer.first == "a"
    assert budget.model_requests_used == 1


def test_a_request_with_no_saved_answer_still_spends_one(monkeypatch):
    """It was still a request the run decided to make."""
    # Counting only the ones that came back with something would let a run that
    # found nothing saved keep asking without limit.
    monkeypatch.setattr(llm.settings, "demo_mode", True)

    answer, budget = llm.ask_for("nothing was saved for this", TwoThings,
                                 RunBudget())

    assert answer is None
    assert budget.model_requests_used == 1


def test_a_caller_outside_a_run_is_not_counted(monkeypatch):
    """Nothing has to invent a budget to ask a question."""
    monkeypatch.setattr(llm.settings, "demo_mode", True)

    answer, budget = llm.ask_for("nothing was saved for this", TwoThings)

    assert answer is None
    assert budget is None


# How many HTTP attempts one invocation makes (R-13, OB-07). These tests let
# the real model library build its request and answer it with a pretend
# refusal, so what is counted is what would leave this computer. Replacing the
# library's own call would hide the behaviour being measured. respx refuses
# any request it was not told about, so nothing reaches Google, and the key
# is invented.

GOOGLE = "https://generativelanguage.googleapis.com/"


def _a_refusal(status):
    """Arranges for every request to Google to be refused with this status."""
    return respx.post(url__startswith=GOOGLE).mock(
        return_value=httpx.Response(status, json={"error": {"message": "refused"}}))


def _a_fresh_connection(monkeypatch):
    """Makes the next call build the connection the way llm.py builds it."""
    # The connection is kept for the whole process, so a test has to clear it,
    # or it would measure whatever an earlier test left behind.
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    monkeypatch.setattr(llm.settings, "gemini_api_key", "invented-key-for-a-test")
    monkeypatch.setattr(llm, "_chat", None)


@respx.mock
def test_the_library_left_to_itself_asks_again_inside_one_invocation(monkeypatch):
    """The measurement R-13 asked for before any fix was made."""
    # Built without the retry setting, the library asks Google again on its own
    # when a request is refused with a status it treats as passing, and those
    # attempts go through neither this project's count nor its wait. The number
    # is the library's and may change with its version; what matters is that
    # it is more than one. Its waits between attempts are replaced so the test
    # takes no time; the attempts are still made.
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    refused = _a_refusal(503)
    connection = ChatGoogleGenerativeAI(model="gemini-3.6-flash",
                                        google_api_key="invented-key-for-a-test")

    with pytest.raises(Exception, match="503"):
        connection.invoke("a question")

    assert refused.call_count > 1, "one invocation, several attempts"


@respx.mock
def test_one_invocation_is_one_attempt_and_a_refusal_is_reported(monkeypatch):
    """The fix: the connection llm.py builds never asks again on its own."""
    # So the count and the wait around each invocation cover every attempt that
    # leaves this computer, which is what the free tier's limits rest on. The
    # refusal comes back as a failure naming the status, and it is not worth
    # asking again through the repair loop (R-13). The plain path asks again
    # for nothing, a 503 included: the second attempt belongs to a run, which
    # has a count to spend it from, and the tests at the end of this file
    # measure it there.
    _a_fresh_connection(monkeypatch)
    refused = _a_refusal(503)

    answer = llm.ask("a question")

    assert refused.call_count == 1
    assert isinstance(answer, llm.ModelFailure)
    assert "503" in answer.reason
    assert not answer.worth_asking_again


@respx.mock
def test_a_structured_request_makes_one_attempt_and_spends_one(monkeypatch):
    """The same through the structured path, with the budget: one attempt at
    Google and one request spent, so what the run counts is what was sent."""
    _a_fresh_connection(monkeypatch)
    refused = _a_refusal(429)

    answer, budget = llm.ask_for("list two things", TwoThings, RunBudget())

    assert refused.call_count == 1
    assert budget.model_requests_used == 1
    assert isinstance(answer, llm.ModelFailure)
    assert "429" in answer.reason
    assert not answer.worth_asking_again


# Asking again after a 503 (R-13, revisited 17 September, the waits
# lengthened on the 18th). The service answering that it is overloaded ended
# five standard-dataset attempts on 16 September and the live run on the
# 17th, each on one request; on the morning of the 18th the live run was
# refused three times inside the thirty and ninety seconds of the 17th. The
# run's path now waits one, three and five minutes and asks at most three
# times more, and every attempt is counted and spaced as a first one is.
# Measured at the HTTP boundary like the tests above, so what is counted is
# what would leave this computer.


def _no_real_waiting(monkeypatch):
    """Records the waits ask_for asks for instead of sleeping through them."""
    waited = []

    def record_instead_of_sleeping(seconds):
        # The limiter's spacing goes through the same sleep, in fractions of
        # a second under the test fixture; only the waits for the service
        # are of interest here.
        if seconds >= 1:
            waited.append(seconds)

    monkeypatch.setattr(llm.time, "sleep", record_instead_of_sleeping)
    return waited


def _what_the_screen_was_told(monkeypatch):
    """Stands in for the screen's status box: every line said is kept."""
    told = []
    monkeypatch.setattr(llm, "say_while_waiting", told.append)
    return told


@respx.mock
def test_an_overloaded_service_is_asked_again_after_a_wait_and_every_attempt_is_counted(
        monkeypatch):
    """Four attempts at most, one, three and five minutes apart, each spent from
    the run's count before it is sent; the failure says how many were made, so
    a reader of the report does not go looking for a retry to add.
    """
    # The waits are slept in slices of half a minute, so the screen's Stop
    # control has a point to take effect at and the line on the screen counts
    # down; the total waited is the nine minutes.
    _a_fresh_connection(monkeypatch)
    waited = _no_real_waiting(monkeypatch)
    refused = _a_refusal(503)

    answer, budget = llm.ask_for("list two things", TwoThings, RunBudget())

    assert refused.call_count == 4
    assert sum(waited) == 60 + 180 + 300
    assert max(waited) == 30
    assert budget.model_requests_used == 4
    assert isinstance(answer, llm.ModelFailure)
    assert "503" in answer.reason
    assert "asked 4 times" in answer.reason
    assert not answer.worth_asking_again


@respx.mock
def test_the_screen_is_told_how_long_each_wait_has_left(monkeypatch):
    """Before every slice of a wait the screen is told the time left and the
    count of asks, so the line reads as a countdown.
    """
    # Watching the live run's second attempt on 18 September, nobody could
    # tell a run waiting for the service from one that would never come back,
    # and the ask was for "the model is busy, asking again in three minutes"
    # in place of the mute spinner. The screen is told when the model is
    # asked again, and, when it answers, that the run goes on. Nothing is
    # said when no screen has asked to be told.
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    _no_real_waiting(monkeypatch)
    told = _what_the_screen_was_told(monkeypatch)
    model = _OverloadedOnceThenAnswering(TwoThings(first="a", second="b"))
    monkeypatch.setattr(llm, "_connection", lambda: model)

    answer, _ = llm.ask_for("list two things", TwoThings, RunBudget())

    assert answer == TwoThings(first="a", second="b")
    assert told == [
        "The model is busy: asking again in 1 minute (1 of 4 asks made).",
        "The model is busy: asking again in 30 seconds (1 of 4 asks made).",
        "Asking the model again (ask 2 of 4).",
        "The model answered; the run goes on."]
    assert llm._in_words(300) == "5 minutes"
    assert llm._in_words(150) == "2 minutes 30 seconds"

    monkeypatch.setattr(llm, "say_while_waiting", None)
    model = _OverloadedOnceThenAnswering(TwoThings(first="a", second="b"))
    monkeypatch.setattr(llm, "_connection", lambda: model)
    answer, _ = llm.ask_for("list two things", TwoThings, RunBudget())
    assert answer == TwoThings(first="a", second="b"), "waited in silence"


@respx.mock
def test_a_run_with_one_request_left_is_not_asked_again_for_the_service(monkeypatch):
    """The wait and the second attempt are worth nothing to a run that cannot
    pay for the attempt: it stops on the first refusal with its count intact,
    rather than sleeping and then being refused by its own budget."""
    _a_fresh_connection(monkeypatch)
    waited = _no_real_waiting(monkeypatch)
    refused = _a_refusal(503)

    answer, budget = llm.ask_for("list two things", TwoThings,
                                 RunBudget(model_request_limit=1))

    assert refused.call_count == 1
    assert waited == []
    assert budget.model_requests_used == 1
    assert isinstance(answer, llm.ModelFailure)
    assert "asked" not in answer.reason


@respx.mock
def test_every_request_has_a_time_limit_and_one_that_runs_out_is_asked_again(
        monkeypatch):
    """The live run's second attempt on 18 September sent its planning request and
    waited seventeen minutes for an answer that never came, with nothing on the
    screen or in the log, until the process was killed by hand (R-25).
    """
    # The connection now gives up after two minutes on any one operation; a
    # request that runs out of time is given the same waits as a 503, asked
    # again, and reported as a timeout in words a reader can act on. The
    # library raises its HTTP client's timeout error, which is what is stood
    # in for here, so the path from the connection to the report is the real
    # one.
    _a_fresh_connection(monkeypatch)
    waited = _no_real_waiting(monkeypatch)
    never_answered = respx.post(url__startswith=GOOGLE).mock(
        side_effect=httpx.ReadTimeout("no answer"))

    assert llm._connection().timeout == 120
    answer, budget = llm.ask_for("list two things", TwoThings, RunBudget())

    assert never_answered.call_count == 4
    assert sum(waited) == 60 + 180 + 300
    assert budget.model_requests_used == 4
    assert isinstance(answer, llm.ModelFailure)
    assert ("no answer in the 120 seconds allowed for one operation on the "
            "connection (ReadTimeout)") in answer.reason
    assert "asked 4 times" in answer.reason
    assert not answer.worth_asking_again


class _OverloadedOnceThenAnswering:
    """Stands in for the model library when the service recovers: the first
    attempt is refused with the status the real library carries on a 503, and
    the second is answered."""

    class _Overloaded(Exception):
        code = 503

    def __init__(self, answer):
        self._answer = answer
        self.attempts = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, question):
        self.attempts += 1
        if self.attempts == 1:
            raise self._Overloaded("503 UNAVAILABLE")
        return self._answer


def test_a_service_that_recovers_is_answered_on_the_second_attempt(monkeypatch):
    """The point of waiting: the answer comes back as if the first attempt
    had worked, and the run's count shows both attempts."""
    monkeypatch.setattr(llm.settings, "demo_mode", False)
    monkeypatch.setattr(llm.settings, "record_replies", False)
    waited = _no_real_waiting(monkeypatch)
    model = _OverloadedOnceThenAnswering(TwoThings(first="a", second="b"))
    monkeypatch.setattr(llm, "_connection", lambda: model)

    answer, budget = llm.ask_for("list two things", TwoThings, RunBudget())

    assert model.attempts == 2
    assert waited == [30, 30], "one minute, in two slices"
    assert budget.model_requests_used == 2
    assert answer == TwoThings(first="a", second="b")
