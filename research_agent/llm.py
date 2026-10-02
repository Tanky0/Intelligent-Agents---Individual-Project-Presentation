"""Provides the application's single interface to the language model."""
# Every model request passes through this file, and no other part of the
# application imports the Gemini library. This keeps provider-specific code
# separate from the agents that use it.
#
# The system uses Gemini 3.6 Flash through the free tier, as specified in the
# design (Google DeepMind, 2026b). The provider controls which models are
# available to each account and may change this access without notice. The model
# is specified in only one file, so if it becomes unavailable or is no longer
# included in the free tier, it can be replaced by changing only that file.
#
# The module supports plain and structured requests. A plain request returns
# text. A structured request checks the response against the required schema.
# Responses that do not match the schema are rejected here before an agent can
# treat them as valid.
#
# Model responses are recorded and replayed in the same way as responses from the
# paper sources. Because the free allowance is calculated daily, repeated
# demonstration rehearsals could use it up before the final demonstration.
# DEMO_MODE replays recorded responses, allowing the same run to be repeated
# without making additional model requests.
#
# The free tier allows five requests a minute and twenty a day, far fewer than
# the paper services allow. The daily figure is the one that shapes the design:
# screening one paper at a time would spend a whole day's allowance on a single
# run, which is why papers are screened in batches. Requests are also
# spaced out here, using the same limiter the paper sources use rather than a
# second one written for this file. Nothing is spaced out in DEMO_MODE,
# because a replayed response makes no request (OB-07).
#
# One counted invocation here is at most one HTTP attempt at Google: a replayed
# answer sends none. The library's own retrying is turned off where the
# connection is made, because those attempts passed through neither the count
# nor the wait (R-13). The one case in which this file asks again itself is a
# 503, the service saying it is overloaded: at most three times more, after
# waits of one, three and five minutes, and each further attempt is counted
# and spaced like a first one. Decided on 17 September, after that answer had
# ended six runs in two days, with waits of thirty and ninety seconds;
# lengthened on the 18th, when the live run was refused three times inside
# those two minutes.
#
# A request that gets no answer within two minutes is treated the same way,
# since 18 September, when one held a run open for seventeen minutes (R-25).
# The two minutes are the library's limit on each
# operation of the connection, not a deadline on the whole request: an answer
# arriving slowly in pieces can hold a request longer, and no total has been
# measured. While it waits, this file tells the screen how long is left, if
# the screen has asked to be told, so that a person watching can tell a wait
# from a run that is stuck.
#
# Two things about that allowance matter when planning a demonstration. It
# belongs to the project rather than to the key, so anything else using the same
# project spends from the same five a minute. And the daily part of it returns at
# midnight Pacific time, not at midnight here.

import json
import time
from dataclasses import dataclass

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from research_agent import certificates
from research_agent.config import settings
from research_agent.processing import run_log
from research_agent.sources import saved_replies
from research_agent.sources.http import limiter_for

# Must run before the model library opens a connection. See certificates.py.
certificates.install()

# Keep one shared model connection here so every agent can use it. Otherwise,
# the connection would have to be passed through several functions simply to
# reach the function that sends the model request.
_chat = None

# Five requests a minute is what the free tier allows for this model. Asking
# for slightly fewer leaves a gap of twelve and a half seconds between one
# request and the next, so at most five fall inside any one minute and a
# sixth never does, even when one of them is slow to leave. Five in a minute
# is the limit met, not passed: the service's rate-limit page, read on 17
# September, showed a peak of five of five for this model, which is this gap doing what
# it is for. The comment used to say five could not fall inside a minute,
# which the arithmetic does not support: five requests span four gaps, fifty
# seconds.
_REQUESTS_PER_MINUTE = 4.8

# One limiter for the whole file, because the allowance belongs to the account
# rather than to any one request. A limiter made inside each call would have
# nothing to wait for and every request would go out at once.
_LIMITER = limiter_for("gemini", _REQUESTS_PER_MINUTE / 60.0)


@dataclass
class OutOfBudget:
    """A model request the run was not allowed to make (OB-07)."""
    # Its own answer rather than a ModelFailure, because it is not a failure.
    # Nothing went wrong: the run spent the requests it was given, and the limit
    # did what it is for. Reported as a failure it would reach the graph as
    # something needing a person, and whoever picked it up would look for a fault
    # that is not there.
    #
    # It carries no suggestion of trying again. Another attempt would need a
    # request the run does not have, which is the whole point.

    reason: str
    worth_asking_again: bool = False


@dataclass
class ModelFailure:
    """A model request that did not succeed, described so it can be reported."""
    # Waiting does not help with every refusal. The daily part of the allowance,
    # once spent, refuses every further request until it resets, and no amount of
    # trying again inside the same day will change that. A run that stopped with
    # an error would also lose the papers it had already screened. So a failure
    # is returned in the same way the paper sources return theirs, and the run
    # reports what it could not do rather than appearing to have done it.
    #
    # Whether the answer is worth trying again is recorded here rather than left
    # to be worked out from the wording of the reason. The decision is based on
    # the type of failure: formatting and validation errors may be fixed by
    # another attempt, while other failures are returned without spending another
    # model request (OB-07).
    #
    # Unknown failures are not repeated by default. That is the cautious way
    # round, because a request spent on a problem another attempt cannot fix is
    # a request the rest of the run no longer has.

    model: str
    reason: str
    worth_asking_again: bool = False


# These errors mean the model returned an answer, but it did not match the
# required format. Another attempt may be able to correct it. Error types are
# checked directly instead of relying on the wording of an error message.
_ANSWERS_WORTH_ASKING_AGAIN = (OutputParserException, ValidationError)

# How long to wait, in seconds, before asking again when the service has said
# it is overloaded, and so how many more times it is asked: three. A 503 is
# the one refusal that is about the service and not about this project: the
# message with it says demand is high and to try again later. A 429 says
# the rate or the allowance is spent, and the message does not say which: a
# minute's limit would clear in a minute where a day's would not, and this
# file cannot tell the two apart, so neither is asked again and the refusal
# is reported (OB-07). When the library's own retrying was turned off, the
# choice was not to retry at all, because nothing could tell a 503 from the
# rest; the status can. The waits are long because a
# 503 that clears in a second would not have ended a run: on 16 and 17
# September the answer lasted minutes, and ended five standard-dataset
# attempts and the live run. They were thirty and ninety seconds on the
# 17th, and on the morning of the 18th the live run was refused three times
# inside those two minutes, so one, three and five minutes were chosen:
# nine minutes of waiting at most on one request. Not more attempts and not
# longer waits, because each attempt is a counted request the rest of the
# run no longer has; whether a fifth attempt would be answered is not
# known and was not tried.
_WAITS_WHEN_THE_SERVICE_IS_OVERLOADED = (60, 180, 300)

# A wait is slept in slices of this many seconds, with a word to the screen
# before each. Two reasons. The screen's Stop control takes effect at the
# next thing the script puts on the page (read in Streamlit's script runner,
# where every message to the page is a point at which a stop request is
# honoured), so a wait slept in one piece could not be stopped until it had
# ended. And the line on the screen counts down, so a person can see the
# wait moving.
_SECONDS_PER_SLICE_OF_A_WAIT = 30

# Where a line about a wait goes while the screen is running a run. The
# screen sets this to a function of one line before it starts a run and
# takes it away after: "the model is busy, asking again in three
# minutes" was decided on 18 September in place of a spinner that looks the
# same whether the run is waiting or stuck. Held here rather than passed
# down, because the wait is several calls below the screen and every
# function between would carry an argument it has no use for. Nothing is
# said when nobody is listening: a run from a test or a script waits in
# silence.
say_while_waiting = None

# How long one request may wait for its answer, in seconds. On 18 September
# the live run's second attempt sent its planning request and nothing came
# back: the connection stayed open for seventeen minutes with no answer, no
# refusal and no line in the log, until the process was killed by
# hand (R-25). Two minutes is longer than the one answer timed so far, the
# plan in 45 seconds that same morning, with room for a screening batch of
# ten abstracts, and shorter than a person waits before deciding the
# screen is stuck. The library hands the number to its HTTP client, which
# applies it to each operation on the connection, connecting, each read and
# write, and the wait for a connection from its pool, and not to the
# request as a whole; a request whose answer
# arrives slowly in pieces can last longer than two minutes, and no total
# was measured. A request that runs out of time is asked again after the
# same waits as a 503, then reported.
_SECONDS_TO_WAIT_FOR_AN_ANSWER = 120


def _the_service_was_overloaded(problem):
    """True when the service refused with 503, or gave no answer in time."""
    # Read from the status the library carries rather than from the words of
    # the message, which are Google's and may change. The status is what the
    # refusal is: a 503 is the service overloaded, a 429 the allowance or the
    # rate, a 400 something wrong with the request, and only the first is worth
    # a wait. A request that ran out of time is the library's own timeout
    # error, and is given the same wait, because asking again after a pause
    # is the one thing this file can do about it and costs what a 503's
    # retry costs. It is not read as the service being overloaded: a
    # timeout says nothing about why no answer came, and the report calls
    # it a timeout.
    if isinstance(problem, httpx.TimeoutException):
        return True
    return getattr(problem, "code", None) == 503


def _describe(problem):
    """Turns whatever went wrong into one line that can be read in a report."""
    # Other faults keep their kind and message, because a mistake in this code
    # would otherwise be indistinguishable from the service refusing.
    # A request that ran out of time is said in those words, because the
    # library's own message for it is empty and "ReadTimeout:" alone would
    # send a reader to the code to find out what was waited for.
    #
    # A refused answer gets a fixed reason (R-28). Parser messages quote the
    # answer, and validation messages, field paths and chained errors can
    # also contain rejected input. Keeping any of those would carry a paper
    # extract or a refused rationale into the state and run log. Less detail
    # here is preferable to retaining the text that the check refused.
    if isinstance(problem, httpx.TimeoutException):
        # The kind in brackets says which operation ran out: the library's
        # client raises one error for connecting, one for a read, one for a
        # write and one for waiting on its pool, and the limit applies to
        # each; "one read" here would have been false for the other three.
        return (f"no answer in the {_SECONDS_TO_WAIT_FOR_AN_ANSWER} seconds "
                "allowed for one operation on the connection "
                f"({type(problem).__name__})")
    if isinstance(problem, OutputParserException):
        return "OutputParserException: the answer was not in the format asked for"
    if isinstance(problem, ValidationError):
        return "ValidationError: the answer was not in the format asked for"
    return f"{type(problem).__name__}: {problem}"


def _say(line):
    """Hands a line to the screen, when the screen has asked to be told."""
    if say_while_waiting is not None:
        say_while_waiting(line)


def _in_words(seconds):
    """"2 minutes 30 seconds", as a person reads a wait."""
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if minutes == 1:
        parts.append("1 minute")
    elif minutes > 1:
        parts.append(f"{minutes} minutes")
    if seconds:
        parts.append(f"{seconds} seconds")
    return " ".join(parts)


def _wait_and_say_so(seconds, asked):
    """Sleeps the wait in slices, saying before each how long is left and
    how many times the model has been asked, so that the person reading the
    screen sees the wait moving and its end."""
    at_most = 1 + len(_WAITS_WHEN_THE_SERVICE_IS_OVERLOADED)
    left = seconds
    while left > 0:
        _say(f"The model is busy: asking again in {_in_words(left)} "
             f"({asked} of {at_most} asks made).")
        a_slice = min(_SECONDS_PER_SLICE_OF_A_WAIT, left)
        time.sleep(a_slice)
        left -= a_slice


def _answer_in(saved):
    """Returns the answer text from a saved reply, or None if it holds none."""
    # A saved reply is a file, and a file can be damaged or left behind by an
    # older version of this application. Reading the answer straight out of it
    # would stop a run with a message about a missing key or the wrong kind of
    # value, when what happened is that a recording cannot be replayed. That is
    # the same class of problem as an answer that no longer fits its schema, and
    # it is reported the same way.
    if not isinstance(saved, dict):
        return None
    answer = saved.get("answer")
    # An answer of nothing but spaces is not an answer. Returned as one, it
    # would reach an agent as an empty reply and be acted on as though the
    # model had said something.
    if not isinstance(answer, str) or not answer.strip():
        return None
    return answer


def _connection():
    """Return the model connection used by the application."""
    # The connection is created the first time this function is called. Later
    # calls return the same connection instead of creating a new one.
    global _chat
    if _chat is None:
        _chat = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            # At most one attempt for each invocation. In this library the
            # number is attempts including the first, so one means the
            # library never asks again on its own. Left at its default, it
            # asked six times for one invocation that met a 503, waiting
            # between attempts by its own clock: five requests this file had
            # neither counted nor spaced, which is the count and the wait the
            # free tier's limits rest on. Measured offline in
            # tests/unit/test_llm.py (R-13).
            #
            # The library retries nothing. The further attempts this
            # project makes, after a 503 or a request that ran out of
            # time, are made in ask_for, where each is counted and spaced
            # like a first one; any other failure is
            # reported and stops that step, and the repair loop stays for
            # malformed answers (OB-07).
            max_retries=1,
            # Every request gives up after this long without an answer, so
            # that a call the service never answers ends in a failure the
            # screen and the log can show, rather than in a connection held
            # open until somebody kills the process (R-25). The library
            # passes the number to its own HTTP client, which applies it to
            # each operation on the connection and not to the request as a
            # whole (the comment on the constant says what that leaves
            # unbounded).
            timeout=_SECONDS_TO_WAIT_FOR_AN_ANSWER,
        )
    return _chat


def _one_attempt(question, schema):
    """One request to the model for an answer in the schema, spaced by the
    limiter. Returns the response and nothing, or nothing and the problem.
    """
    # Kept apart from ask_for so that asking again after a 503 is the same
    # lines run twice rather than a second copy of them. Every kind of failure
    # is caught on purpose, not only the refusals that can be named. The
    # provider raises its own errors, and an unfamiliar one would otherwise end
    # the run and lose the work already done. The kind is kept in the problem,
    # so nothing is hidden. noqa tells the style checker that the wide catch
    # is meant.
    _LIMITER.wait()
    try:
        return _connection().with_structured_output(schema).invoke(question), None
    except Exception as problem:  # noqa: BLE001
        return None, problem


def ask(question):
    """Send one question to the model and return its response as text."""
    # Not used by the application, which asks through ask_for with the run's
    # budget; kept for a caller outside a run. It makes one attempt and asks
    # again for nothing, a 503 included: the wait and the second attempt belong
    # to a run, which has a count to spend them from.
    #
    # Three answers are possible and the caller has to tell them apart. Text is
    # the ordinary answer. A ModelFailure means the request did not succeed and
    # says why. None means DEMO_MODE was on and no response had been saved for
    # this question, which makes the missing response visible instead of letting
    # an incomplete run appear successful.
    if settings.demo_mode:
        saved = saved_replies.load_model_reply(settings.gemini_model, question)
        if saved is None:
            return None
        answer = _answer_in(saved)
        if answer is None:
            return ModelFailure(settings.gemini_model,
                                "a saved reply was found but holds no answer to replay")
        return answer

    _LIMITER.wait()
    try:
        reply = _connection().invoke(question)
    # Every kind of failure is caught on purpose, not only the refusals that
    # can be named. The provider raises its own errors, and an unfamiliar one
    # would otherwise end the run and lose the work already done. The kind is
    # kept in the reason, so nothing is hidden. noqa tells the style checker
    # that the wide catch is meant.
    except Exception as problem:  # noqa: BLE001
        return ModelFailure(settings.gemini_model, _describe(problem),
                            isinstance(problem, _ANSWERS_WORTH_ASKING_AGAIN))
    # Gemini may return a response in several parts. The .text property
    # combines them into one string, while .content contains the structured
    # parts.
    answer = reply.text

    if settings.record_replies:
        # The run is read here and handed in, because saved_replies cannot
        # read it itself; run_log.run_in_progress says why.
        saved_replies.save_model_reply(settings.gemini_model, question, answer,
                                       run_log.run_in_progress())

    return answer


def ask_for(question, schema, budget=None):
    """Send a question to the model and return a response matching a schema."""
    # Returns the answer and the budget, in that order. A run passes the budget
    # it holds and keeps the one that comes back, because every model request a
    # run makes goes through here, and a limit counted anywhere else would be a
    # limit on part of the run.
    #
    # Not every call to this module is counted. A caller that passes no budget is
    # not, and neither is ask(), which takes none. Nothing in the application uses
    # either: both call sites are in the planner and both pass the budget the run
    # is carrying. So a run is counted in full, and the module is not a place
    # where counting is enforced on everybody.
    #
    # The request is counted before it is sent, and it is counted whatever
    # happens next. A refused request, a badly formatted answer and a repair all
    # spend a request. So does a saved answer replayed in demonstration mode: a
    # demonstration whose live equivalent would run out of allowance is not a
    # demonstration of anything, and only counting the replays shows that.
    #
    # A caller with no budget passes none and is not counted. That is for a
    # caller outside a run, and a run always has one.
    #
    # The schema is a Pydantic class describing the fields the response must
    # have. A response missing a field, or holding the wrong kind of value, is
    # rejected here instead of reaching an agent that would treat it as usable.
    #
    # As with ask, three answers are possible: a checked response, a ModelFailure
    # saying the request did not succeed, or None in DEMO_MODE when no response
    # has been saved. A saved response that no longer matches its schema is
    # rejected in the same way a live one would be.
    saved_under = f"{question}\n[schema:{schema.__name__}]"

    # Counted before anything is sent, and before a saved answer is read.
    # Counted afterwards it would report a request already spent, which is a
    # record of the allowance going rather than a limit on it.
    if budget is not None:
        if not budget.has_a_model_request_left():
            return OutOfBudget(
                f"this run has used the {budget.model_request_limit} model "
                f"requests it was allowed"), budget
        budget = budget.with_one_model_request_used()

    if settings.demo_mode:
        saved = saved_replies.load_model_reply(settings.gemini_model, saved_under)
        if saved is None:
            return None, budget
        # A saved answer can stop matching its schema after the schema changes,
        # and the text in it can be damaged. A live request that cannot produce
        # a usable answer comes back as a ModelFailure, so a replayed one has
        # to do the same. Otherwise the two ways of running the application
        # fail differently, and the one that would end in a stack trace is the
        # one a demonstration uses.
        answer = _answer_in(saved)
        if answer is None:
            return ModelFailure(
                settings.gemini_model,
                "a saved reply was found but holds no answer to replay"), budget
        try:
            return schema.model_validate(json.loads(answer)), budget
        except (json.JSONDecodeError, ValidationError) as problem:
            # Do not repair an invalid saved reply. Demonstration mode would
            # read the same saved answer again, so another attempt could not
            # fix it.
            return ModelFailure(settings.gemini_model,
                                _describe(problem)), budget

    response, problem = _one_attempt(question, schema)

    # Asked again only when the service said it was overloaded, and only
    # while the run has a request to ask with: each further attempt is
    # counted before it is sent, exactly as the first was, so the run's count
    # stays a count of what left this computer (R-13).
    attempts = 1
    for wait in _WAITS_WHEN_THE_SERVICE_IS_OVERLOADED:
        if not _the_service_was_overloaded(problem):
            break
        if budget is not None and not budget.has_a_model_request_left():
            break
        if budget is not None:
            budget = budget.with_one_model_request_used()
        _wait_and_say_so(wait, attempts)
        _say(f"Asking the model again (ask {attempts + 1} of "
             f"{1 + len(_WAITS_WHEN_THE_SERVICE_IS_OVERLOADED)}).")
        response, problem = _one_attempt(question, schema)
        attempts += 1
    # Said only after a wait, so the screen's line does not stay on the last
    # wait once the run has moved on; a run that was never refused has had
    # nothing said to it.
    if attempts > 1 and problem is None:
        _say("The model answered; the run goes on.")

    if problem is not None:
        reason = _describe(problem)
        # Said in the reason, because a report reading "503" alone would
        # send somebody to add the retry that is already there.
        if attempts > 1:
            reason = f"{reason}; asked {attempts} times"
        return ModelFailure(
            settings.gemini_model, reason,
            isinstance(problem, _ANSWERS_WORTH_ASKING_AGAIN)), budget

    # A request that did not raise is not the same as an answer in the format
    # asked for. If the library ever hands back something else, saying so here
    # is better than letting it travel: without this the agent would be told
    # that demonstration mode had no saved answer, which is a different problem
    # with a different fix.
    if not isinstance(response, schema):
        return ModelFailure(
            settings.gemini_model,
            f"the model returned a {type(response).__name__} where a "
            f"{schema.__name__} was asked for"), budget

    if settings.record_replies:
        saved_replies.save_model_reply(settings.gemini_model, saved_under,
                                       response.model_dump_json(),
                                       run_log.run_in_progress())

    return response, budget
