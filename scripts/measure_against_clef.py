"""Measures the two agents against one review from the CLEF 2019 collection:
retrieval recall, screening precision, extract accuracy and repeat
consistency.
"""
# The tutor's feedback on the design proposal asked for tests against a standard
# dataset. The proposal that answers it is in docs/testing.md under "The
# standard dataset": one Cochrane review from the CLEF 2019 technology-assisted
# review collection, CD011571, fifty of its PubMed papers, and four measurements
# against the decisions the review's authors made. The numbers in this file (a
# step limit of four, a thousand results a search, the baselines and the caps)
# are the ones settled there, and each is named once here so the script and the
# plan cannot drift apart without one of them being wrong on its face.
#
#     python scripts/measure_against_clef.py draw TOPIC QRELS
#         the fifty, from the collection's topic and qrels files
#     python scripts/measure_against_clef.py fetch
#         the fifty records, from PubMed; no model request
#     python scripts/measure_against_clef.py retrieve --cap N --instructions FILE
#         retrieval recall
#     python scripts/measure_against_clef.py screen --cap N --instructions FILE
#         screening precision, with the extract accuracy the check counts
#     python scripts/measure_against_clef.py screen-again --cap N --instructions FILE
#         --reference SCREENING_FILE
#         repeat consistency
#     python scripts/measure_against_clef.py audit-sample --reference SCREENING_FILE
#         the ten judgements read by hand for extract accuracy
#
# It keeps these rules on purpose, each from the proposal.
#
# The labels are kept apart from the requests. The identifiers are read to
# prepare the records; which of them the review's authors included is read
# only when a run's answer is scored, so the figure measures the agents and
# not their memory of the answers, the way the hand-screened sample is kept
# out of every prompt. Every step but the draw refuses a labels file
# that is not fifty distinct PubMed identifiers, because every later check
# compares against that file and would agree with a short or repeated one.
#
# Every step that spends model requests is given its cap on the command line
# and checks it before the first request: not below the step's baseline, and
# not above the ceiling the proposal's allocation rule gives the step. Nothing
# in the application counts model requests across runs, so the day's
# twenty and the day-two retry are kept by hand, by whoever runs this; what
# the script itself holds is the bounds of one step. The same steps refuse to
# start in demonstration mode, where the application replays saved answers
# and counts each as a request (llm.py): a measurement made from replays
# would measure the replay, and a repeat of one would agree with itself.
#
# A record PubMed did not return, or returned with no title, stops the
# preparation, and a preparation that failed in any way leaves no earlier
# records behind to be screened by mistake, because the earlier ones are
# removed before the fetch begins. A paper with no title cannot be screened,
# and drawing a substitute would change the fifty the labels describe. A
# record that came back with a title and no abstract stays, judged on what
# the prompt carries for it (SC-01).
#
# The repeat run is refused before any request unless the first run's
# file is a complete live screening, every judgement in it read and checked,
# and its records, request and model are the ones the repeat would use. A
# comparison of two runs with different inputs measures the inputs.
#
# Nothing a paper says is written into the repository. The fetched records and
# every attempt's output go under data/clef_tar/private/, which the repository
# ignores; each attempt is kept under its run identifier, so a retry does not
# overwrite the evidence of the attempt before it.

import argparse
import csv
import hashlib
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent import graph
from research_agent.agents import curator, planner
from research_agent.config import settings
from research_agent.processing import run_log
from research_agent.processing.validation import with_the_extracts_checked
from research_agent.schemas import (
    PaperRecord,
    ResearchRequest,
    ResearchState,
    ResultsBuffer,
    RunBudget,
    ScreenedPaper,
    SearchOrigin,
)
from research_agent.sources import pubmed

# The review, named by its Cochrane identifier and its title. The title is the
# research question the agents are given, because it is what the review's
# authors set out to answer; the criteria come from the review's protocol and
# are passed in from a file kept outside the repository.
TOPIC = "CD011571"
TITLE = "Antistreptococcal interventions for guttate and chronic plaque psoriasis"

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "clef_tar"
LABELS = DATA_DIR / "cd011571_fifty.csv"
# Ignored by the repository: the records hold abstracts, and the outputs hold
# the model's summaries of them.
PRIVATE_DIR = DATA_DIR / "private"
RECORDS = PRIVATE_DIR / "records.json"
FINGERPRINT = PRIVATE_DIR / "records.sha256"

# Fifty papers a run is the screening budget the proposal was asked to fit
# fit. Every paper the labels include is kept, so the recall figure's
# denominator is known before the run; the rest are drawn with a fixed
# starting number so the draw can be repeated exactly. The number is taken
# from the topic's identifier so it is not mistaken for the hand-screened
# sample's.
SAMPLE_SIZE = 50
SEED = 11571

# The retrieval run's limits, from the proposal. Four searches at a thousand
# results are forty-four NCBI attempts before any retry, under the hundred a
# run is allowed (OB-04). The hand-screened sample's measurement asks for
# ten thousand results a search. That would be one search request and a
# hundred fetches, 101 attempts, so the threshold could refuse a fetch
# inside the first search.
STEP_LIMIT = 4
RESULTS_PER_SEARCH = 1000

# What each step costs when every answer fits, from the code as it is: one
# request to plan and one after each search short of the step limit, which
# is the most the planned sequence can take, since a finish move ends the
# searching earlier; one request for each batch of ten papers screened. A cap
# below this is refused as a matter of policy: a retrieval that finished
# after one search would fit in two, but a cap that cannot cover the planned
# sequence leaves the run's end to chance rather than to the plan.
BASELINES = {"retrieve": 1 + (STEP_LIMIT - 1), "screen": SAMPLE_SIZE // 10,
             "screen-again": SAMPLE_SIZE // 10}

# The most each step may be given, from the allocation rule in docs/testing.md:
# the retrieval run ten, the screening run eighteen (the smaller of eighteen
# and what the day has left is the operator's to work out), the repeat ten.
# Checked here so a slip on the command line cannot spend a day's allowance
# on one step. The day-two retry's ten is the operator's rule as well.
CAPS = {"retrieve": 10, "screen": 18, "screen-again": 10}

# The extract accuracy sample read by hand: ten judgements, five judged
# include and five judged
# exclude, in an order fixed before the run so the sample cannot be chosen
# after the results are seen.
AUDIT_SIZE = 10
AUDIT_EACH = 5


class CuratorThatJudgesNothing:
    """Stands in for the Curation Agent in the retrieval run."""
    # Retrieval recall measures what the searches brought back, before
    # deduplication and
    # before the fifty places are shared out, and screening the papers would
    # spend requests on a figure that does not use them: five for fifty papers
    # when every answer fits, and more with repairs. The graph still runs the
    # hand-over, so the run ends the way a real one does.

    def screen(self, state):
        return [], state.budget


# The draw.

def pids_in(topic_text):
    """The PubMed identifiers a CLEF topic file lists after its query."""
    # The file is: a topic line, a title line, the Boolean query over several
    # lines, then "Pids:" and one identifier a line.
    pids = []
    after_the_marker = False
    for line in topic_text.splitlines():
        if line.strip() == "Pids:":
            after_the_marker = True
            continue
        if after_the_marker and line.strip().isdigit():
            pids.append(line.strip())
    return pids


def relevant_in(qrels_text, topic):
    """The identifiers a CLEF qrels file marks relevant for one topic."""
    # A qrels line is: topic, an unused column, identifier, label. The label is
    # 0 for not relevant and anything else for relevant.
    relevant = set()
    for line in qrels_text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == topic and parts[3] != "0":
            relevant.add(parts[2])
    return relevant


def draw_the_fifty(pids, relevant):
    """The fifty rows: every relevant paper, and the rest drawn from the others."""
    # Sorted by identifier at the end so the order the papers are screened in
    # carries nothing about their labels.
    included = sorted(pid for pid in pids if pid in relevant)
    excluded = sorted(pid for pid in pids if pid not in relevant)
    how_many_more = SAMPLE_SIZE - len(included)
    if how_many_more < 0:
        raise ValueError(f"{len(included)} relevant papers do not fit in {SAMPLE_SIZE}")
    drawn = random.Random(SEED).sample(excluded, how_many_more)
    rows = []
    for pid in sorted(included + drawn, key=int):
        if pid in relevant:
            rows.append({"pmid": pid, "include": "include"})
        else:
            rows.append({"pmid": pid, "include": "exclude"})
    return rows


def write_labels(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pmid", "include"])
        writer.writeheader()
        writer.writerows(rows)


def read_identifiers(path):
    """The fifty identifiers, in the order the papers are screened in."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row["pmid"] for row in csv.DictReader(handle)]


def read_included(path):
    """Which of the fifty the review's authors included. Read to score, not to ask."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["pmid"] for row in csv.DictReader(handle) if row["include"] == "include"}


def why_the_labels_are_not_the_fifty(identifiers):
    """Says what is wrong with the labels file, or nothing if it holds the fifty."""
    # The checks on the prepared records and on a repeat's reference compare
    # against this file, so they would agree with a file one row short or with
    # a row repeated, and every figure would then be against a denominator
    # nobody chose. Checked once, before any other step reads it.
    if len(identifiers) != SAMPLE_SIZE:
        return f"the labels hold {len(identifiers)} identifiers, not {SAMPLE_SIZE}"
    if len(set(identifiers)) != len(identifiers):
        return "the labels repeat an identifier"
    for pmid in identifiers:
        if not pmid.isdigit():
            return f"the labels hold something that is not a PubMed identifier: {pmid!r}"
    return ""


# The preparation: the records, fetched once and checked before every run.

def as_paper(record):
    """One PubMed record in the shared format, with no search in front of it."""
    # The origin's query is left empty on purpose: these papers reach the
    # Curation Agent from a labelled list rather than from a search, which is
    # the case the format keeps that field empty for (SC-01).
    abstract = (record.abstract or "").strip()
    if abstract:
        text_source = "abstract"
    else:
        text_source = "metadata"
    return PaperRecord(
        title=record.title,
        origins=[SearchOrigin(source="pubmed", query="")],
        abstract=abstract,
        authors=list(record.authors),
        year=record.year,
        journal=record.journal,
        publication_types=list(record.publication_types),
        pmid=record.pmid,
        doi=record.doi,
        text_source=text_source,
    )


def fetch_the_fifty(identifiers):
    """The records in the labels' order, and what is missing or short."""
    # Returns the papers, one reason for each identifier that cannot be
    # screened, the identifiers whose record came back with no abstract, and
    # the fetch errors. Two things make a record unusable: PubMed did not
    # return it, or returned it with no title, which the shared format refuses
    # (a PubMed record with no ArticleTitle element reads as an empty title).
    # The title is checked here rather than left to the format's own refusal,
    # so the preparation reports which identifier and why instead of stopping
    # on an exception. The caller decides what to do; this only reports, so
    # the decision is in one place.
    records, errors = pubmed.fetch(list(identifiers))
    by_identifier = {record.pmid: record for record in records}
    papers = []
    unusable = []
    without_abstract = []
    for pmid in identifiers:
        record = by_identifier.get(pmid)
        if record is None:
            unusable.append(f"{pmid} was not returned")
            continue
        if not record.title.strip():
            unusable.append(f"{pmid} came back with no title")
            continue
        paper = as_paper(record)
        if not paper.abstract:
            without_abstract.append(pmid)
        papers.append(paper)
    return papers, unusable, without_abstract, errors


def fingerprint_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_the_records(papers):
    """Writes the records out and their fingerprint beside them."""
    # The fingerprint is what the repeat run checks against the first
    # run's, to show it read the same records rather than fetching them again
    # and hoping PubMed had not changed.
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    dumped = []
    for paper in papers:
        dumped.append(paper.model_dump())
    RECORDS.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
    FINGERPRINT.write_text(fingerprint_of(RECORDS), encoding="utf-8")


def forget_the_records():
    """Removes an earlier preparation, so a failed one cannot leave it usable."""
    for path in (RECORDS, FINGERPRINT):
        if path.exists():
            path.unlink()


def why_the_records_are_not_ready(identifiers):
    """Says what is wrong with the saved records, or nothing if they are the fifty."""
    # Checked before every run that screens, because the records on disk are
    # what the model is shown: a preparation that failed, a sample redrawn
    # beside an old cache, or a file edited by hand would otherwise be screened
    # as though it were the fifty the labels describe.
    if not RECORDS.exists() or not FINGERPRINT.exists():
        return "the records are not prepared; run fetch"
    if fingerprint_of(RECORDS) != FINGERPRINT.read_text(encoding="utf-8").strip():
        return "the saved records do not match their fingerprint; run fetch"
    try:
        papers = load_the_records()
    except (ValueError, KeyError, TypeError) as problem:
        return f"the saved records could not be read ({problem}); run fetch"
    saved = [paper.pmid for paper in papers]
    if saved != list(identifiers):
        return (f"the saved records hold {len(saved)} identifiers and the labels "
                f"hold {len(identifiers)}, or the order differs; run fetch")
    return ""


def load_the_records():
    decoded = json.loads(RECORDS.read_text(encoding="utf-8"))
    papers = []
    for item in decoded:
        papers.append(PaperRecord.model_validate(item))
    return papers


# The request, and the cap.

def request_from(instructions_path):
    """The request every run is given: the review's title and the fixed criteria."""
    # The criteria are condensed from the review's protocol before the first run
    # and kept outside the repository, because the protocol is Cochrane's text.
    # The file holds two fields, and the request refuses either left blank.
    instructions = json.loads(instructions_path.read_text(encoding="utf-8"))
    return ResearchRequest(
        question=TITLE,
        inclusion_criteria=instructions["inclusion_criteria"],
        exclusion_criteria=instructions["exclusion_criteria"],
        results_per_search=RESULTS_PER_SEARCH,
    )


def why_the_cap_is_refused(cap, step):
    """Says why a step may not start with this cap, or nothing if it may."""
    if cap < BASELINES[step]:
        return (f"the cap is {cap} and {step} needs {BASELINES[step]} requests "
                "to cover its planned sequence when every answer fits; a cap "
                "that cannot is not started")
    if cap > CAPS[step]:
        return (f"the cap is {cap} and the allocation rule gives {step} at most "
                f"{CAPS[step]}")
    return ""


# Retrieval recall.

def retrieve(request, cap):
    """The real Planning Agent and graph, with nothing screened."""
    # Built here rather than through graph.run because that takes no budget:
    # the cap has to be on the state before the first request, and the
    # application's own limit of eighteen is not the share this step was
    # given. The steps are graph.run's, in its order, with no ranking because
    # nothing is screened.
    identifier = run_log.new_identifier()
    starting = ResearchState(
        request=request, run_id=identifier, step_limit=STEP_LIMIT,
        budget=RunBudget(model_request_limit=cap))
    run_log.note_the_request(request, identifier)
    with run_log.while_running(identifier):
        finished = graph.build_graph(planner, CuratorThatJudgesNothing()).invoke(starting)
    return ResearchState.model_validate(finished)


def papers_the_searches_returned(finished):
    """Everything the searches returned, wherever the run left it."""
    # At the hand-over the graph moves what the searches returned into
    # papers_before_deduplication and narrows papers to the fifty for
    # curation. A run that stopped before the hand-over, on a failed move or
    # a spent budget, never moved them, so they are still in papers. Reading
    # one field would count a stopped run as having returned nothing.
    if finished.results.ready:
        return finished.results.papers_before_deduplication
    return finished.results.papers


def retrieval_figures(finished, included):
    """The share of the labelled papers the searches brought back, with every
    search's fate and whether the measurement is whole.
    """
    # Completed means the run handed over and every search finished within its
    # limit. Partial means something is missing and said: a search that failed,
    # was partial, or returned exactly its limit, or a run that stopped early
    # after papers came back. Failed means the run stopped before any paper
    # came back, so there is nothing to measure.
    #
    # The labels are PubMed identifiers, so a returned paper is matched by its
    # PubMed identifier. The planner may search Semantic Scholar too, and a
    # paper from there can come back with no PubMed identifier; such a paper
    # cannot be matched to the labels, so it is counted and named by the
    # identifiers it does carry rather than dropped, and the figure's limit is
    # visible in the report.
    returned = set()
    without_pmid = []
    for paper in papers_the_searches_returned(finished):
        if paper.pmid:
            returned.add(paper.pmid)
        else:
            without_pmid.append({"doi": paper.doi, "paper_id": paper.paper_id,
                                 "source": paper.origins[0].source})
    found = sorted(included & returned)
    searches = []
    something_short = False
    for outcome in finished.results.outcomes:
        note = ""
        if outcome.result != "finished":
            note = f"{outcome.result}: {outcome.reason}"
            something_short = True
        elif outcome.papers_found >= finished.request.results_per_search:
            note = "returned exactly its limit, so it may have been truncated"
            something_short = True
        searches.append({"query": outcome.subgoal.query, "result": outcome.result,
                         "papers_found": outcome.papers_found, "note": note})

    stopped_early = bool(finished.stopped_because) or finished.needs_human_attention
    # Failed is judged on every paper that came back, not only the ones with
    # a PubMed identifier: a stopped run that returned one Semantic Scholar
    # paper has something to report, so it is partial, and calling it failed
    # would contradict the list of papers without an identifier beside it.
    nothing_came_back = not returned and not without_pmid
    if stopped_early and nothing_came_back:
        status = "failed"
    elif stopped_early or something_short:
        status = "partial"
    else:
        status = "completed"
    return {
        "status": status,
        "relevant": len(included),
        "relevant_found": len(found),
        "retrieval": len(found) / len(included),
        "papers_returned": len(returned),
        "papers_returned_without_pmid": len(without_pmid),
        "returned_identifiers": sorted(returned),
        "returned_without_pmid": without_pmid,
        "relevant_found_identifiers": found,
        "searches": searches,
        "handed_over": finished.results.ready,
        "stopped_because": finished.stopped_because,
        "needs_human_attention": finished.needs_human_attention,
        "attention_reason": finished.attention_reason,
        "model_requests_used": finished.budget.model_requests_used,
    }


# Screening precision, and repeat consistency.

def screen_the_fifty(request, papers, cap):
    """The real Curation Agent, given the fifty as the hand-over would."""
    # Returns what the agent returned, the budget it ended with and the run's
    # identifier. The answer is the list of judgements, or the failure that
    # stopped a batch.
    identifier = run_log.new_identifier()
    state = ResearchState(
        request=request, run_id=identifier,
        results=ResultsBuffer(papers=list(papers), ready=True),
        budget=RunBudget(model_request_limit=cap))
    run_log.note_the_request(request, identifier)
    with run_log.while_running(identifier):
        answer, budget = curator.screen(state)
    if isinstance(answer, list):
        answer = with_the_extracts_checked(answer)
    return answer, budget, identifier


def why_the_screening_is_incomplete(answer, identifiers):
    """Says what is wrong with a screening's answer, or nothing if it is whole."""
    # A failure from a batch is one kind of incomplete; a list that is not one
    # judgement for each of the fifty, in their order, is another, and the
    # application does not promise the second cannot happen.
    if not isinstance(answer, list):
        return f"the screening stopped at a failed batch: {answer}"
    judged = [item.paper.pmid for item in answer]
    if judged != list(identifiers):
        return (f"the screening returned {len(judged)} judgements for "
                f"{len(identifiers)} papers, or in a different order")
    return ""


def screening_figures(judged, included):
    """Precision and recall against the labels, and the extract check's count."""
    # Precision has no value when nothing was judged include, and the figure says
    # so rather than reporting a number. A paper judged uncertain counts as not
    # included and is counted on its own.
    judged_include = [item.paper.pmid for item in judged
                      if item.screening.decision == "include"]
    uncertain = [item.paper.pmid for item in judged
                 if item.screening.decision == "uncertain"]
    right = [pmid for pmid in judged_include if pmid in included]
    if judged_include:
        precision = len(right) / len(judged_include)
    else:
        precision = None
    found = sum(1 for item in judged if item.extract_found is True)
    return {
        "papers_judged": len(judged),
        "judged_include": len(judged_include),
        "judged_include_and_labelled_include": len(right),
        "precision": precision,
        "relevant": len(included),
        "recall": len(right) / len(included),
        "judged_uncertain": len(uncertain),
        "extracts_found_in_the_paper": found,
    }


def differences_between(first, second):
    """The judgements whose decision or score changed between two runs."""
    # The two lists are the same papers, which the checks on the reference
    # establish before the second run is made; a paper with no partner is
    # reported rather than skipped, so the count of papers compared is the
    # count of pairs and a short comparison cannot pass as a whole one.
    earlier = {}
    for item in first:
        earlier[item.paper.pmid] = item.screening
    decision_differs = []
    score_differs = []
    unmatched = []
    pairs = 0
    for item in second:
        before = earlier.get(item.paper.pmid)
        if before is None:
            unmatched.append(item.paper.pmid)
            continue
        pairs += 1
        if before.decision != item.screening.decision:
            decision_differs.append(item.paper.pmid)
        if before.relevance != item.screening.relevance:
            score_differs.append(item.paper.pmid)
    return {"papers_compared": pairs,
            "unmatched": unmatched,
            "decision_differs": decision_differs,
            "score_differs": score_differs}


def load_the_screening(path, identifiers):
    """A saved screening, read once and checked whole: the file as saved and
    its judgements in the shared format.
    """
    # Raises ValueError, saying why, for a file that is not a complete live
    # screening of exactly the fifty in order, that does not name its run, or
    # whose judgements are not what the application writes, a relevance score
    # outside its range, say, or a field missing. The repeat and the hand-read
    # sample both read from here, and the repeat reads before it spends, so
    # a file the comparison could not use is never discovered after the
    # requests have gone.
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        raise ValueError(f"the file could not be read ({problem})")
    if not isinstance(saved, dict) or saved.get("complete") is not True:
        raise ValueError("the file is not a complete screening")
    if saved.get("mode") != "live":
        raise ValueError("the file is not a live screening, or does not say which mode made it")
    if not saved.get("run_id"):
        raise ValueError("the file does not say which run made it")
    try:
        judged = []
        for item in saved["judgements"]:
            judged.append(ScreenedPaper.model_validate(item))
    except (KeyError, TypeError, ValueError) as problem:
        raise ValueError(f"the file's judgements are not in the shared format ({problem})")
    if [item.paper.pmid for item in judged] != list(identifiers):
        raise ValueError(f"the file holds {len(judged)} judgements and the labels hold "
                         f"{len(identifiers)} papers, or the order differs")
    return saved, judged


def load_the_reference(path, identifiers, request):
    """The first run of a comparison, checked before the repeat spends anything."""
    # Returns its judgements and its run identifier. Beyond being a complete
    # live screening of the fifty, the first run has to have read the records
    # prepared now and used the request and model the repeat will use, or the
    # comparison measures the inputs. Nothing is guessed from a file name.
    saved, judged = load_the_screening(path, identifiers)
    if saved.get("records_fingerprint") != FINGERPRINT.read_text(encoding="utf-8").strip():
        raise ValueError("the reference was made from different records; the repeat must "
                         "read the same ones")
    if saved.get("request") != request.model_dump():
        raise ValueError("the reference was made with a different question or criteria")
    if saved.get("model") != settings.gemini_model:
        raise ValueError(f"the reference was made with {saved.get('model')} and this "
                         f"repeat would use {settings.gemini_model}")
    return judged, saved["run_id"]


# Extract accuracy.

def audit_sample(judged):
    """The ten read by hand: five judged include and five judged exclude, in a
    fixed order."""
    # When a decision has fewer than five, the shortfall is filled from the other
    # decision and then from those judged uncertain, in the same order; when
    # fewer than ten judgements exist, all are read. The order is fixed by the
    # same starting number as the draw, before any result is seen.
    shuffled = list(judged)
    random.Random(SEED).shuffle(shuffled)
    by_decision = {"include": [], "exclude": [], "uncertain": []}
    for item in shuffled:
        by_decision[item.screening.decision].append(item.paper.pmid)

    chosen = by_decision["include"][:AUDIT_EACH] + by_decision["exclude"][:AUDIT_EACH]
    for decision in ("include", "exclude", "uncertain"):
        for pmid in by_decision[decision]:
            if len(chosen) >= AUDIT_SIZE:
                break
            if pmid not in chosen:
                chosen.append(pmid)
    return chosen


# What each attempt leaves behind. One file for each run identifier, so a
# retry sits beside the attempt it replaces rather than over it.

def write_attempt(name, identifier, request, cap, budget, content):
    """Writes one attempt's record and returns its path."""
    # Every attempt, whether it finished or failed, says what it was given and
    # what it spent: the run, the model, whether its answers were live or
    # replayed, the request, the cap, the requests used and the records'
    # fingerprint. A failed attempt with none of that would be a spent
    # allowance nobody could account for.
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    if FINGERPRINT.exists():
        fingerprint = FINGERPRINT.read_text(encoding="utf-8").strip()
    else:
        fingerprint = ""
    # The spending steps refuse demonstration mode before their first
    # request, so what reaches here was live. It is read from the settings
    # rather than written as a constant, and a later reader checks the field
    # instead of trusting that the refusal was in place when the file was made.
    if settings.demo_mode:
        mode = "demonstration"
    else:
        mode = "live"
    record = {
        "run_id": identifier,
        "model": settings.gemini_model,
        "mode": mode,
        "request": request.model_dump(),
        "cap": cap,
        "model_requests_used": budget.model_requests_used,
        "records_fingerprint": fingerprint,
    }
    record.update(content)
    path = PRIVATE_DIR / f"{name}_{identifier}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


# The steps.

def run_draw(files):
    if len(files) != 2:
        print("draw needs the topic file and the qrels file")
        return 1
    topic_text = pathlib.Path(files[0]).read_text(encoding="utf-8")
    qrels_text = pathlib.Path(files[1]).read_text(encoding="utf-8")
    rows = draw_the_fifty(pids_in(topic_text), relevant_in(qrels_text, TOPIC))
    write_labels(rows, LABELS)
    included = sum(1 for row in rows if row["include"] == "include")
    print(f"Wrote {len(rows)} identifiers to {LABELS}, {included} of them included.")
    return 0


def run_fetch(identifiers):
    # The earlier preparation is removed before the fetch, not once a failure
    # has been found, so that a preparation which fails in any way, a record
    # not returned, a record with no title, or the fetch itself raising,
    # leaves no earlier records behind to be screened as though it had
    # succeeded. Removing them first costs nothing when the fetch succeeds,
    # because the new records replace them.
    forget_the_records()
    papers, unusable, without_abstract, errors = fetch_the_fifty(identifiers)
    for error in errors:
        print(f"  fetch problem: {error.reason}")
    if unusable:
        print(f"{len(unusable)} of {len(identifiers)} records cannot be screened: "
              f"{'; '.join(unusable)}. Nothing is saved, any earlier records are "
              "removed, and nothing may be screened.")
        return 1
    save_the_records(papers)
    print(f"Saved {len(papers)} records to {RECORDS}; fingerprint in {FINGERPRINT}.")
    print(f"  {len(without_abstract)} have no abstract and are judged on the title: "
          f"{', '.join(without_abstract) or 'none'}")
    return 0


def run_retrieve(cap, instructions):
    refused = why_the_cap_is_refused(cap, "retrieve")
    if refused:
        print(refused)
        return 1
    request = request_from(instructions)
    # Made before the run so a spent run cannot fail to leave its record.
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    finished = retrieve(request, cap)

    figures = retrieval_figures(finished, read_included(LABELS))
    path = write_attempt("retrieval", finished.run_id, request, cap, finished.budget,
                         {"figures": figures})
    print(json.dumps(figures, indent=2))
    print(f"Written to {path}")
    if figures["status"] == "completed":
        return 0
    return 1


def run_screen(identifiers, cap, instructions, reference=None):
    """Screening precision, or repeat consistency when a reference is given.
    The checks come first: the cap, the records, and for a repeat the
    reference, all before a request."""
    if reference is None:
        step = "screen"
    else:
        step = "screen-again"
    refused = why_the_cap_is_refused(cap, step)
    if refused:
        print(refused)
        return 1
    request = request_from(instructions)
    refused = why_the_records_are_not_ready(identifiers)
    if refused:
        print(refused)
        return 1
    first = []
    first_run = ""
    if reference is not None:
        try:
            first, first_run = load_the_reference(reference, identifiers, request)
        except ValueError as refused:
            print(refused)
            return 1
    papers = load_the_records()

    answer, budget, identifier = screen_the_fifty(request, papers, cap)

    incomplete = why_the_screening_is_incomplete(answer, identifiers)
    if incomplete:
        path = write_attempt("failed", identifier, request, cap, budget,
                             {"complete": False, "failure": incomplete})
        print(f"{incomplete}. {budget.model_requests_used} requests were spent; no "
              f"figure is taken from it. Written to {path}")
        return 1

    figures = screening_figures(answer, read_included(LABELS))
    content = {"complete": True, "figures": figures}
    name = "screening"
    if reference is not None:
        figures["differences"] = differences_between(first, answer)
        content["reference"] = {"file": str(reference), "run_id": first_run}
        name = "screening_again"
    judgements = []
    for item in answer:
        judgements.append(item.model_dump())
    content["judgements"] = judgements
    path = write_attempt(name, identifier, request, cap, budget, content)
    print(json.dumps(figures, indent=2))
    print(f"Written to {path}")
    return 0


def run_audit_sample(identifiers, reference):
    try:
        saved, judged = load_the_screening(reference, identifiers)
    except ValueError as refused:
        print(f"the sample is read from a complete screening of the fifty, and this is "
              f"not one: {refused}")
        return 1
    chosen = audit_sample(judged)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRIVATE_DIR / f"audit_sample_{saved['run_id']}.json"
    path.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(f"The {len(chosen)} judgements to read, in order: {', '.join(chosen)}")
    print(f"Written to {path}")
    return 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["draw", "fetch", "retrieve", "screen",
                                         "screen-again", "audit-sample"])
    parser.add_argument("files", nargs="*", help="draw: the topic file and the qrels file")
    parser.add_argument("--cap", type=int, help="the model requests this step may make")
    parser.add_argument("--instructions", type=pathlib.Path,
                        help="the file holding the fixed inclusion and exclusion criteria")
    parser.add_argument("--reference", type=pathlib.Path,
                        help="a complete screening file, for screen-again and audit-sample")
    args = parser.parse_args(argv)

    if args.step == "draw":
        return run_draw(args.files)

    # Every other step starts from the labels, so they are read and checked
    # here, once, and handed to the step.
    identifiers = read_identifiers(LABELS)
    refused = why_the_labels_are_not_the_fifty(identifiers)
    if refused:
        print(refused)
        return 1
    if args.step == "fetch":
        return run_fetch(identifiers)
    if args.step == "audit-sample":
        if args.reference is None:
            print("audit-sample needs --reference")
            return 1
        return run_audit_sample(identifiers, args.reference)

    # The three remaining steps spend model requests.
    if args.cap is None or args.instructions is None:
        print(f"{args.step} needs --cap and --instructions")
        return 1
    if settings.demo_mode:
        print("demonstration mode is on, so the application would replay saved answers "
              "and count each as a request; a measurement of that would measure the "
              "replay, and it is refused")
        return 1
    if args.step == "retrieve":
        return run_retrieve(args.cap, args.instructions)
    if args.step == "screen":
        return run_screen(identifiers, args.cap, args.instructions)
    if args.reference is None:
        print("screen-again needs --reference, the first screening's file")
        return 1
    return run_screen(identifiers, args.cap, args.instructions, reference=args.reference)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
