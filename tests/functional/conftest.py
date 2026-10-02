"""What the functional tests share: the two edges replaced, and a store of
their own.
"""
# A functional test runs the real screen, graph, agents and sources, and
# replaces the two things the application does not own, the language model
# and the search service, with the synthetic replies in synthetic_replies.py
# (FT-01 to FT-09). PubMed is replaced at the network, so the real source code
# reads the reply; the model is replaced where the agents ask, because a saved
# model reply is filed under the exact question text and a hand-written
# fixture cannot be keyed that way without copying every prompt. Three
# smaller things are replaced beside them, each for a reason its fixture
# gives: the mode the tests run in, the waits between retries, and the file
# the store writes to.

import httpx
import pytest
import respx

from research_agent.agents import asking
from research_agent.config import settings
from research_agent.llm import ModelFailure
from research_agent.processing import storage
from research_agent.sources import http, pubmed
from research_agent.sources.pubmed import EUTILS
from tests.functional import synthetic_replies


class ModelThatReplays:
    """Stands in for the model, answering each request from the list given."""
    # Each answer is built into the record the caller asked for when it is
    # handed over, so a synthetic reply that no longer fits the format stops
    # the test with a ValidationError rather than being replayed past the
    # check. That is a check on the fixtures, not a replay of what the real
    # model call does with a badly formatted answer: llm.ask_for turns that
    # into a ModelFailure marked worth asking again, and the sentinel below is
    # how a test asks for one. One request is counted for each answer
    # supplied, which is what llm.ask_for counts for each attempt; its refusal
    # of a request when the allowance is spent is not replayed here, so these
    # tests do not exercise a spent allowance.

    def __init__(self, answers):
        self.answers = list(answers)
        self.questions_asked = []

    def __call__(self, question, schema, budget=None):
        self.questions_asked.append(question)
        if budget is not None:
            budget = budget.with_one_model_request_used()
        answer = self.answers.pop(0)
        if answer == synthetic_replies.AN_ANSWER_THAT_DOES_NOT_FIT:
            return ModelFailure(settings.gemini_model,
                                "the answer did not fit the format asked for",
                                worth_asking_again=True), budget
        return schema.model_validate(answer), budget


@pytest.fixture
def replayed_model(monkeypatch):
    def replay(*answers):
        model = ModelThatReplays(answers)
        monkeypatch.setattr(asking.llm, "ask_for", model)
        return model
    return replay


def _esearch(request):
    term = request.url.params.get("term", "")
    found = synthetic_replies.SEARCHES.get(term, [])
    if isinstance(found, str):
        # A search text mapped to a status is one the service refuses.
        return httpx.Response(int(found.split()[1]))
    return httpx.Response(200, json=synthetic_replies.esearch_reply(term))


def _efetch(request):
    sent = httpx.QueryParams(request.content.decode("utf-8"))
    asked_for = sent.get("id", "").split(",")
    return httpx.Response(200, text=synthetic_replies.efetch_reply(asked_for))


@pytest.fixture
def synthetic_pubmed(monkeypatch):
    """Answers the real PubMed source's requests from the synthetic papers."""
    # Demonstration mode is switched off here, whatever the settings file on
    # this computer says: with it on, the shared request handling replays a
    # saved reply instead of sending anything, and the parent fixture has
    # pointed the saved replies at an empty folder, so every search would come
    # back as a missing recording and the routes below would never be asked.
    # The test chooses its own mode rather than inheriting one.
    #
    # The rate limiter and the wait between retries are replaced as well, so a
    # test that plans a refused search does not wait the seconds a live retry
    # would; what is being tested is what the run does with the refusal, not
    # how long it waits for it.
    monkeypatch.setattr(settings, "demo_mode", False)
    monkeypatch.setattr(pubmed, "_limiter", lambda: http.RateLimiter(1000.0))
    monkeypatch.setattr(http, "FIRST_WAIT_SECONDS", 0.0)
    with respx.mock(assert_all_called=False) as router:
        router.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(side_effect=_esearch)
        router.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(side_effect=_efetch)
        yield router


@pytest.fixture
def temporary_store(tmp_path, monkeypatch):
    """A store of the test's own, reached through the screen's own call."""
    # The screen calls storage.save_approved without naming a store, and the
    # default is the real file under data/. The real function is kept and only
    # the name of the file is moved, so what the tests show is what storage
    # does, not what a stand-in does; and the results page, which reads the
    # store by the same name, reads what the screen wrote. Moving the write
    # alone, as this fixture once did by wrapping the function, is what let
    # the launcher's store part from its read-back on 17 September.
    store = tmp_path / "records-of-this-test.sqlite3"
    monkeypatch.setattr(storage, "DEFAULT_STORE", store)
    return store
