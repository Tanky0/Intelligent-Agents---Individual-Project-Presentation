"""Makes the replies the public offline demonstration replays (R-12).

    python scripts/make_demonstration_fixtures.py <folder>
"""
# The command line stays in the docstring because the docstring is printed
# as the usage text when the command is called without its one argument.
#
# The demonstration runs the application with no network and no model, replaying
# saved replies. Until now the only saved replies were fifteen recordings of real
# services, which are kept private and cannot leave this repository, so an export
# had no demonstration at all. This writes a set the export may carry: every
# reply the demonstration run needs, made from the invented papers and answers
# the functional tests already use, so that not one word in them came from a
# paper or a service.
#
# A reply is replayed under a name built from the request, and a model answer
# under the exact text of the prompt with the schema's name appended. Nothing
# but a run produces those prompts, so the fixtures are made by running the real
# graph once: the real Planning and Curation Agents, the real PubMed source
# reading invented XML at the network, and a stand-in model answering from the
# same invented records. Recording is turned on and demonstration mode off, so
# each service reply is written by the same code that writes a live one, and
# each model answer is written here in the same shape, under the same name.
#
# The folder given is where the files go, with a manifest beside them in which
# each carries a fixture origin: who wrote it, when, what it stands in for, that
# it was not copied from a paper, and the fingerprint it is about. That is the
# record the replay check and the export boundary both read (R-16). The approved
# folder is not written to: moving the files there is the person's decision,
# made by copying, so that nothing here can put a file where a demonstration
# reads from without somebody looking first.
#
# The question the demonstration asks is the invented one, drug X in condition
# Y. Invented papers under the real research question would read as real.

import json
import pathlib
import sys

import httpx
import respx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent import graph
from research_agent.agents import asking, curator, planner
from research_agent.config import settings
from research_agent.processing import run_log
from research_agent.schemas import ResearchRequest
from research_agent.sources import http, pubmed, saved_replies
from research_agent.sources.pubmed import EUTILS
from tests.functional import synthetic_replies

# The question and criteria the demonstration asks, and the answers the model
# is made to give, in the order the run asks: one plan of one search, one
# decision to finish, one screening of the three papers.
#
# A run with a refused search was tried first, to show the coverage warning,
# and cannot be replayed: a refused request leaves no saved reply, so in
# demonstration mode it fails for a different reason ("no reply has been
# saved"), the planner's next prompt quotes that reason, and the prompt no
# longer matches the answer recorded for it. A demonstration replays only a
# run whose every request answered. The coverage warning is shown by the
# functional tests and by the screen with invented papers instead.
QUESTION = "Does drug X treat condition Y?"
INCLUSION = "studies of drug X in condition Y reporting an outcome"
EXCLUSION = "reviews and opinion pieces"
PLAN_ONE_SEARCH = {"subgoals": [
    {"purpose": "find trials of drug X in condition Y", "source": "pubmed",
     "query": "drug X condition Y trial"},
]}
ANSWERS = [PLAN_ONE_SEARCH, synthetic_replies.MOVE_FINISH,
           synthetic_replies.SCREENING_THREE_PAPERS]

WRITTEN_BY = "the author of this project, by running this script"
WRITTEN_ON = "2026-09-16"


class ModelThatRecords:
    """Answers each request from the list, and writes the answer as a saved
    reply under the name replay will look it up by.
    """
    # The functional tests' stand-in answers and does not write. This one writes,
    # because the whole point of the run is the files it leaves. The name is
    # built the way llm.ask_for builds it, the prompt text with the schema's
    # name, so that replay in demonstration mode finds the answer when the same
    # agent asks the same question.

    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, question, schema, budget=None):
        if budget is not None:
            budget = budget.with_one_model_request_used()
        answer = schema.model_validate(self.answers.pop(0))
        saved_under = f"{question}\n[schema:{schema.__name__}]"
        saved_replies.save_model_reply(settings.gemini_model, saved_under,
                                       answer.model_dump_json(),
                                       run_log.run_in_progress())
        return answer, budget


def _esearch(request):
    term = request.url.params.get("term", "")
    found = synthetic_replies.SEARCHES.get(term, [])
    if isinstance(found, str):
        return httpx.Response(int(found.split()[1]))
    return httpx.Response(200, json=synthetic_replies.esearch_reply(term))


def _efetch(request):
    sent = httpx.QueryParams(request.content.decode("utf-8"))
    asked_for = sent.get("id", "").split(",")
    return httpx.Response(200, text=synthetic_replies.efetch_reply(asked_for))


def run_once_and_record(folder):
    """Runs the graph with the invented replies and leaves every reply in the
    folder. Returns the state the run ended in.
    """
    # Six things on four modules are set for the run and put back afterwards:
    # where captures go, the mode, recording, the two waits, and the model. Put
    # back because the test of this script goes on to replay the files it made
    # through the real code, and a stand-in model left in place would answer
    # that replay too and make the test prove nothing.
    was = (saved_replies.CAPTURED_DIR, settings.demo_mode, settings.record_replies,
           pubmed._limiter, http.FIRST_WAIT_SECONDS, asking.llm.ask_for)
    saved_replies.CAPTURED_DIR = folder
    settings.demo_mode = False
    settings.record_replies = True
    # No wait between requests: the seconds a live call waits for its slot are
    # not part of what is being made here.
    pubmed._limiter = lambda: http.RateLimiter(1000.0)
    http.FIRST_WAIT_SECONDS = 0.0
    asking.llm.ask_for = ModelThatRecords(ANSWERS)

    request = ResearchRequest(question=QUESTION, inclusion_criteria=INCLUSION,
                              exclusion_criteria=EXCLUSION)
    try:
        with respx.mock(assert_all_called=False) as router:
            router.get(url__startswith=f"{EUTILS}/esearch.fcgi").mock(side_effect=_esearch)
            router.post(url__startswith=f"{EUTILS}/efetch.fcgi").mock(side_effect=_efetch)
            return graph.run(request, planner, curator)
    finally:
        (saved_replies.CAPTURED_DIR, settings.demo_mode, settings.record_replies,
         pubmed._limiter, http.FIRST_WAIT_SECONDS, asking.llm.ask_for) = was


def manifest_for(folder, stands_in_for):
    """A manifest naming every file in the folder with a fixture origin."""
    files = {}
    for path in saved_replies.files_in(folder):
        here = saved_replies.fingerprint(path)
        files[path.name] = {
            "sha256": here,
            "approval_history": [{
                "approved_at": saved_replies.now(),
                "approved_sha256": here,
                "repository_head_at_review": "0" * 40,
                "working_tree_at_review": ("made by scripts/make_demonstration_"
                                           "fixtures.py; the head is filled in "
                                           "when the files are approved"),
                "fixture_origin": {
                    "written_by": WRITTEN_BY,
                    "written_on": WRITTEN_ON,
                    "what_it_stands_in_for": stands_in_for(path),
                    "copied_from_a_paper": False,
                    "applies_to_sha256": here,
                },
            }],
        }
    return {**saved_replies._manifest_header(), "files": files}


def what_it_stands_in_for(path):
    """One sentence per file saying which edge it replaces."""
    record = saved_replies._read(path)
    if record is not None and "question" in record:
        return "an answer from the model to one of the run's prompts"
    return "a reply from PubMed to one of the run's requests"


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    folder = pathlib.Path(sys.argv[1]).resolve()
    if folder.exists() and any(folder.iterdir()):
        print(f"{folder} is not empty. The fixtures are written into an empty "
              f"folder so that nothing already there is mistaken for one.")
        return 1
    folder.mkdir(parents=True, exist_ok=True)

    # The run log for the fixture run is kept beside the fixtures rather than
    # in the real log, so a run that made files is not read later as a run
    # that answered a question.
    run_log.DEFAULT_LOG = folder.parent / (folder.name + "_run_log.jsonl")

    state = run_once_and_record(folder)

    manifest = manifest_for(folder, what_it_stands_in_for)
    manifest_path = folder.parent / (folder.name + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    replies = saved_replies.files_in(folder)
    print(f"Run {state.run_id}: {len(state.screened)} papers screened, "
          f"{len(replies)} replies written to {folder}")
    print(f"Manifest: {manifest_path}")
    print("Nothing was written to the approved folder. To make these the "
          "demonstration's replies, copy them there with the manifest, "
          "after reading them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
