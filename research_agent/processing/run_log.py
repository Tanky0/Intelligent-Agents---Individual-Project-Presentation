"""Writes down what a run did, while it is doing it."""
# The coverage report in `audit_log.py` reads a finished run and says what it
# missed. This is the other half of the same commitment: a line written as each
# thing happens, so a run can be followed afterwards by somebody who was not
# watching. A report built at the end describes the state a run finished in, and
# that state is missing the order things happened in and the time each took.
#
# What is written is what the proposal names: the query, the filters, the time,
# the source, the failures, and the decisions the reviewer made.
#
# The query is the one the Planning Agent wrote, and the filters are recorded
# beside it rather than inside it. Both services are told the years as their own
# parameters rather than as words in the query, so no single string is "the
# search as sent". A reader repeating it needs the query and the bounds
# together, which is why they are separate fields.
#
# Every line carries the identifier of the execution it belongs to, made once
# when the run starts and carried in the state. The file is shared by every run
# on the machine, so without it a decision recorded an hour later could not be
# tied to the run it was made in. Working the identifier out from the question
# and criteria was the first attempt and was wrong: two runs of the same
# question shared it, so the decisions of one could not be told from the other's.
#
# A line written where no run is to hand says "not recorded" rather than naming
# a run it does not belong to.
#
# One line for each event, as JSON. A file appended to a line at a time survives
# a run that stops half way, which a single record written at the end does not.
# A run interrupted mid-write leaves a line with no end, so the next write puts
# a newline in first: otherwise the following event would be joined onto the
# fragment and one interruption would cost two events.
#
# The reviewer's note is written as they wrote it, and it is free text. That is
# deliberate, because the note is the reason a decision can be argued with
# later, and it means a reviewer who quotes a paper in their note puts that
# quotation in the log. What is never written by this file is the paper's own
# text as the record holds it: no abstract, no full text, no extract, no
# summary, and no title. The identifier is what a reader follows, as the
# service and data boundaries in `docs/development.md` record.
#
# The file is not in the repository. It holds what runs did on one machine, and
# git ignores it for the same reason it ignores the buffer.
#
# Reading it back is done here as well, beside the writing. One file holds every
# run the computer has made, so anything that shows a run has to separate them
# first, and a reader kept in its own file would go on naming fields that had
# been renamed here. What reads it today is the screen (`ui/app.py`).

import contextlib
import contextvars
import datetime
import json
import threading
import uuid
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from research_agent.config import REPO_ROOT, settings

# The names of the request values that describe a search, and therefore the
# list of everything else, the keys and the contact address included. It is
# imported rather than written out again so that the log and the saved replies
# cannot disagree about what may be written down.
from research_agent.sources.saved_replies import _REQUEST_VALUES_TO_KEEP

DEFAULT_LOG = REPO_ROOT / "evidence" / "run_log.jsonl"

# Failures collected while writing the log itself. A log that stopped a run
# because it could not be written would be worse than the gap it records, so
# they are put here in the way saved_replies and the buffer do, and the
# coverage report reads them.
writing_failures = []

# Which run the lines written from deep inside a request belong to. The shared
# request handling records what it sent, and it is several calls below
# anything that knows which run is happening; threading an identifier through
# every source and every helper to reach it would change four sources and
# their tests to carry a value none of them uses.
#
# Held per execution context rather than as one value for the whole process.
# A single value was the first attempt and gave one run's requests to another:
# two runs at once, which two browser sessions are, share the process, and the
# second to start took the first's requests with it and left them recorded as
# nobody's when it finished.
_run_in_progress = contextvars.ContextVar("the run in progress", default="")

# Held while the log is being added to. Two runs at once share the file, and
# looking at its last byte and then appending are two steps: interleaved, they
# lose lines with nothing saying so.
_one_writer = threading.Lock()


@contextlib.contextmanager
def while_running(identifier):
    """Files what the requests inside it record under this run."""
    # Two things, and they are not the same thing. A run started in one thread
    # is invisible to another, because each carries its own context, and that is
    # what stops two runs at once taking each other's requests. Putting the old
    # value back at the end is what makes a run inside another give the first
    # one back; clearing would have left the outer run's later requests recorded
    # as nobody's.
    #
    # Called a handle rather than a token, the library's own word, because the
    # release check reads "token =" followed by a long value as a credential,
    # and this file travels in the release.
    handle = _run_in_progress.set(identifier or "")
    try:
        yield
    finally:
        _run_in_progress.reset(handle)


def run_in_progress():
    """The identifier of the run this code is running inside, or "" outside one."""
    # Written for one caller. A saved model reply should say which run produced
    # it, and the module that writes those replies is imported above by this
    # one, so it cannot import this module back: Python would find each waiting
    # for the other. llm.py can import this module, so it asks here and hands
    # the answer in, the way note_an_attempt takes a run and falls back to the
    # same variable when none is given.
    return _run_in_progress.get()


def new_identifier():
    """Names one execution, and is made once at the start of it."""
    # Worked out from the question and criteria was the first attempt and does
    # not identify an execution: two runs of the same question share it, so a
    # decision recorded later could belong to either. This is random, so it
    # names the occasion. It is carried in the state and handed to the screen,
    # which is why nothing has to recompute it.
    return uuid.uuid4().hex[:12]


def _now():
    """The time an event happened, to the second, in UTC."""
    # UTC rather than the computer's own zone, because a log read next to
    # another machine's should not need the reader to work out which hour is
    # which. To the second because a search takes seconds, and a finer figure
    # would suggest an accuracy the ordering does not have.
    return datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()


def _write(entry, run, log):
    """Adds one line, and records rather than raises if it cannot."""
    # A run that stopped because its log was unwritable would lose the work it
    # had done to a fault in the record of that work.
    #
    # The file is looked up here rather than being a default on each function,
    # because a default is fixed when the file is read and a test redirecting
    # the log would be redirecting something nothing used.
    #
    # One writer at a time. Looking at the end of the file and then appending to
    # it are two steps, and two runs at once interleaved them: a measurement of
    # two hundred events kept a hundred and ninety, and nothing said which ten
    # were gone. Two runs at once is what two browser sessions are, so this is
    # not a case that needs contriving. The lock covers only this process; two
    # interpreters writing to one log would be back where this started, and
    # nothing in this project runs that way.
    log = DEFAULT_LOG if log is None else log
    try:
        with _one_writer:
            log.parent.mkdir(parents=True, exist_ok=True)
            # A run stopped part way through a write leaves a line with no end.
            # Appending to it would join this event onto that fragment, so one
            # interruption would cost the event after it as well.
            unfinished = log.exists() and log.read_bytes()[-1:] not in (b"\n", b"")
            with log.open("a", encoding="utf-8") as open_file:
                if unfinished:
                    open_file.write("\n")
                open_file.write(
                    json.dumps({"run": run or "not recorded", **entry},
                               ensure_ascii=True) + "\n")
    except OSError as problem:
        writing_failures.append(f"a line could not be written to {log}: {problem}")


def note_the_request(request, run, log=None):
    """Records the question and criteria a run was given, and when it began."""
    # The criteria are written down because the screening decisions later in the
    # log cannot be judged without them: a paper excluded under one set of
    # criteria is not excluded under another, and a log that recorded only the
    # decisions would leave a reader unable to tell whether they were right.
    #
    # The model is written down as well, since 17 September. The proposal names
    # one model and the settings can name another, and the day that became a
    # real choice was the day the named model refused two live attempts with
    # "high demand": a run made on a different model has to say so where its
    # decisions are read, or a judgement could be credited to a model that did
    # not make it. The saved replies carry the name already; the log is where
    # a reader looks first.
    _write({
        "at": _now(),
        "event": "run started",
        "model": settings.gemini_model,
        "question": request.question,
        "include": request.inclusion_criteria,
        "exclude": request.exclusion_criteria,
        "from year asked for": request.from_year,
        "to year asked for": request.to_year,
        "results asked for per search": request.results_per_search,
    }, run, log)


def note_a_search(outcome, request, run, log=None):
    """Records one search: what was asked for, where, when, and what came back."""
    # The filters come from the request rather than from the outcome, because
    # the outcome says what a search returned and the request is what the search
    # was bounded by. They are separate fields rather than part of the query
    # because that is how they are sent: both services take the years as their
    # own parameters.
    #
    # What a service actually applied is not known here. A source may return a
    # smaller number of results than was asked for, and the planner checks the
    # years of what came back for that reason. So this records what was
    # asked for, and says so in the field names.
    _write({
        "at": _now(),
        "event": "search",
        "source": outcome.subgoal.source,
        "query": outcome.subgoal.query,
        "purpose": outcome.subgoal.purpose,
        "from year asked for": request.from_year,
        "to year asked for": request.to_year,
        "results asked for": request.results_per_search,
        "result": outcome.result,
        "papers found": outcome.papers_found,
        "what happened": outcome.reason,
    }, run, log)


def _kept(values):
    """The values that describe a search, as pairs, in the order they were sent."""
    # Pairs rather than a mapping, because a request may carry the same name
    # twice and a mapping keeps only the last of them. A reader repeating a
    # search that asked for two terms and saw one recorded would repeat a
    # different search.
    #
    # The allowed names are `saved_replies`'s, so the log and the replies cannot
    # disagree about what may be written down.
    pairs = values.items() if hasattr(values, "items") else (values or [])
    return [[name, value] for name, value in pairs
            if name.lower() in _REQUEST_VALUES_TO_KEEP]


def _address_without_credentials(url):
    """The address a reader can follow, and the search values inside it."""
    # A key or a password can be written into an address as well as passed
    # beside it, as user information before the host or as a value in the query.
    # Writing the address down whole was the first attempt and let both through,
    # so what is kept is the scheme, the host and the path, and the query is
    # read separately and put through the same allowed names as everything else.
    #
    # The port is kept, because an address without it is a different address.
    #
    # Blank values are kept as blanks rather than dropped. "retmax=" is a
    # request that was made, and a reader told nothing would think it never was.
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    plain = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return plain, _kept(parse_qsl(parts.query, keep_blank_values=True))


def note_an_attempt(source, method, url, params=None, data=None, attempt=1,
                    run=None, log=None):
    """Records what a request carried, as it is about to go out."""
    # Until this, the log held the bounds the request asked for and not what
    # went out. Both services take the years in their own wording and each has
    # its own maximum number of results, so a reader repeating a search from the
    # log was repeating what was asked for, which is not always what was sent.
    # The proposal promises searches a reader can repeat, and that gap was
    # written down as a reduction rather than closed. This closes it.
    #
    # An attempt, not a delivery. It is written before the request is handed to
    # the connection, so a request that never left the machine is recorded as
    # having been tried. That is the honest reading: what this proves is that
    # the run was about to ask, not that a service was asked.
    #
    # The method, the values beside the request, the values in its body and the
    # values in its address are kept apart. Merging them was the first attempt
    # and lost a value that appeared in two of them, and a reader who has to
    # guess whether a search was a GET or a POST cannot repeat it.
    #
    # Only the values that describe a search are written, under the same allowed
    # names the saved replies use, which keeps out the NCBI key, the Semantic
    # Scholar key and the contact address that travel with these requests. One
    # list rather than two, because the copy nobody was reading would be the one
    # that let a key through.
    address, in_the_address = _address_without_credentials(url)
    _write({
        "at": _now(),
        "event": "request attempt",
        "source": source,
        "method": method,
        "url": address,
        "values in the address": in_the_address,
        "values sent beside it": _kept(params),
        "values sent in the body": _kept(data),
        "attempt": attempt,
    }, run or _run_in_progress.get(), log)


def note_a_failure(what, why, run=None, log=None):
    """Records something that went wrong without stopping the run."""
    # Written here as well as collected on a list, because the list says what
    # went wrong and the log says when, which is what tells a reader whether two
    # failures were one service having a bad minute or a fault that ran through
    # the whole run.
    #
    # The identifier is optional because some failures happen where no run is to
    # hand, and a failure recorded without one is better than a failure not
    # recorded at all. Such a line says "not recorded" rather than naming a run
    # it does not belong to.
    _write({"at": _now(), "event": "failure", "what": what, "why": why},
           run, log)


def note_a_store(kept, run=None, log=None):
    """Records that the reviewer stored the approved papers, and how many."""
    # The decisions were already in the log, one line each, but nothing said
    # when the reviewer closed the review by storing, and the results page
    # says when a review finished. A count and a time, no paper's
    # words: which papers were kept is in the decision lines above it.
    _write({"at": _now(), "event": "approved papers stored",
            "papers stored": kept}, run, log)


def _named(paper):
    """How a paper is written in the log: by identifier, never by title."""
    # The title is the paper's own words, and this file holds none of them. One
    # place decides it so that two events cannot name the same paper two ways
    # and leave a reader unable to follow it from one line to the next.
    return paper.doi or paper.pmid or paper.paper_id or "no identifier"


def note_a_revision(judged, said_before, run=None, log=None):
    """Records that a paper was judged again after the reviewer sent it back."""
    # The decision either side of this is already in the log, so without it a
    # reader sees a paper rejected and then approved with nothing in between
    # saying that the judgement itself had changed.
    #
    # What the agent said before is given rather than looked up, because the
    # judgement that held it has been replaced by the time this is written.
    _write({
        "at": _now(),
        "event": "judged again",
        "paper": _named(judged.paper),
        "the agent had said": said_before,
        "the agent now says": judged.screening.decision,
        "relevance the agent gives now": judged.screening.relevance,
    }, run, log)


def note_a_decision(judged, choice, note, run=None, log=None):
    """Records what the reviewer decided about one paper, and when."""
    # The paper is named by its identifier rather than its title, because the
    # identifier is what a reader follows and the title is the paper's own
    # words. The note is the reviewer's own writing and is kept as they wrote
    # it: it is the reason the decision can be argued with later, and it is free
    # text, so a reviewer quoting a paper puts that quotation here.
    _write({
        "at": _now(),
        "event": "reviewer decision",
        "paper": _named(judged.paper),
        "decision": choice,
        "note": note,
        "the agent had said": judged.screening.decision,
        "relevance the agent gave": judged.screening.relevance,
    }, run, log)


def read(log=None):
    """Every line of the log, and a note where one could not be read."""
    # A damaged line is reported rather than passed over. A log is appended to
    # while a run works, so its last line can be half written, and a reader told
    # nothing would take the record as complete. Reading stops at nothing: the
    # events after a damaged line are the ones a reader most needs when
    # something went wrong in the middle.
    log = DEFAULT_LOG if log is None else log
    if not log.exists():
        return []
    try:
        written = log.read_text(encoding="utf-8")
    except OSError as problem:
        # A log that cannot be read at all must not stop whatever is showing
        # it. The screen is drawn for a reviewer who has papers to judge, and
        # a fault in the record of a run is not a reason to give them none of
        # it. What went wrong is returned so it can be shown rather than
        # looking like an empty log.
        return [{"event": "the log could not be read", "why": str(problem)}]
    entries = []
    for number, line in enumerate(written.splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"event": "unreadable line", "line number": number})
    return entries


@dataclass(frozen=True)
class RecordedRun:
    """One execution as the log holds it, and the lines it wrote."""
    # A record rather than a list of lines, because a reader choosing between
    # runs needs to know which is which before reading any of them, and the
    # question and the time are what they choose by.

    run: str
    started: str
    question: str
    entries: list


def runs(log=None):
    """Every execution in the log, the most recently started first."""
    # The file is shared by every run the computer has made, so a reader looking
    # for what one run did has to have them separated first. They are separated
    # by the identifier each line carries rather than by where the lines sit,
    # because two runs can be worked on in one afternoon and their lines then
    # interleave.
    #
    # Lines written where no run was to hand, and lines that could not be read at
    # all, are gathered together under "not recorded" rather than dropped or
    # attached to whichever run they lie next to. A reader who is shown a
    # complete-looking run that is missing a failure is worse off than one who is
    # shown that some lines belong to nothing.
    #
    # The most recently started first, because a person opening this has almost
    # always come to look at the run they have just made. A run with no time on
    # any of its lines is put last: it cannot be placed, and putting it first
    # would push the run somebody came for off the top.
    gathered = {}
    for entry in read(log):
        gathered.setdefault(entry.get("run", "not recorded"), []).append(entry)

    found = []
    for name, entries in gathered.items():
        # The first line that carries a time, rather than the first line: a
        # damaged line has no time, and a run whose first line was damaged
        # would otherwise be treated as a run that cannot be placed at all.
        started = next((entry["at"] for entry in entries if "at" in entry), "")
        question = next((entry["question"] for entry in entries
                         if entry.get("event") == "run started"
                         and "question" in entry), "")
        found.append(RecordedRun(name, started, question, entries))
    return sorted(found, key=lambda one: one.started, reverse=True)


def started_on_this_clock(stamp):
    """"17-Sep-2026 18:52" in this computer's time, from the log's stamp."""
    # The log writes UTC so that two computers' logs can be read side by side;
    # a person choosing a run wants the clock on the wall. A stamp the log does
    # not have, or one that cannot be read, is said to be not recorded rather
    # than shown as a blank.
    if not stamp:
        return "at a time not recorded"
    try:
        moment = datetime.datetime.fromisoformat(stamp)
    except ValueError:
        return stamp
    return f"{moment.astimezone():%d-%b-%Y %H:%M}"


def heading(recorded):
    """One line naming a run, for a reader choosing which to read."""
    # "Started 17-Sep-2026 18:52, Does drug X, 9148fbaa89bd": the time first,
    # in this computer's clock, the first three words of the question, the
    # identifier last. The order and wording decided on the evening of 17
    # September, in place of the identifier first and the log's UTC stamp
    # whole, which read as a record's line and not a person's. The identifier
    # is still given: two runs of the same question are told apart only by
    # it, and a list of identifiers on their own gives a reader nothing to
    # recognise a run by, which is why it comes last.
    when = started_on_this_clock(recorded.started)
    if recorded.question:
        question = " ".join(recorded.question.split()[:3])
    else:
        question = "question not recorded"
    return f"Started {when}, {question}, {recorded.run}"


def as_lines(entries):
    """The lines of one run as a person reads them, in the order written."""
    # Every field of an event is shown, rather than the fields this function
    # knows the names of. A reader naming its fields would quietly drop a field
    # added to an event later, and a record of what happened that leaves part of
    # it out is the failure this log exists to prevent. What it costs is that
    # the wording follows the field names, which is why those are written to be
    # read (`note_a_search` and the others above).
    #
    # The identifier is left out of each line because every line of one run
    # carries the same one, and `heading` says it once.
    #
    # A line nobody could read, and a log nobody could open, are each said in
    # their own words. Neither has a time or any fields, so shown the ordinary
    # way they would read "time not recorded", which describes a line that was
    # written badly rather than a record that cannot be got at.
    lines = []
    for entry in entries:
        if entry.get("event") == "unreadable line":
            lines.append(f"line {entry.get('line number')} of the log could not "
                         "be read")
            continue
        if entry.get("event") == "the log could not be read":
            lines.append(f"the log itself could not be read: {entry.get('why')}")
            continue
        rest = "; ".join(f"{name}: {value}" for name, value in entry.items()
                         if name not in ("run", "at", "event"))
        said = (f"{entry.get('at', 'time not recorded')}  "
                f"{entry.get('event', 'event not named')}")
        lines.append(f"{said}  {rest}" if rest else said)
    return lines
