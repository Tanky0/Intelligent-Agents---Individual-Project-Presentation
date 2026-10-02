"""What the run log records while a run is working, and how it reads (FR-15)."""
# Every test writes to a file pytest gives it, so none of them touches the log a
# real run would use.
#
# The proposal names six things: the query as sent, the filters, the time, the
# source, the failures and the reviewer's decisions. There is a test for each,
# because a log missing one of them is a log that looks complete.
#
# The tests at the end read a log back rather than writing one. They build the
# file line by line instead of calling the writing above it, because what is
# being checked there is how lines that arrived in a particular order, or that
# arrived damaged, are separated and shown.

import datetime
import json
import threading

from research_agent.processing import run_log
from research_agent.processing.run_log import (
    as_lines,
    heading,
    new_identifier,
    note_a_decision,
    note_a_failure,
    note_a_search,
    note_the_request,
    read,
    runs,
)
from research_agent.schemas import (
    Outcome,
    PaperRecord,
    ResearchRequest,
    ScreenedPaper,
    Screening,
    SearchOrigin,
    Subgoal,
)


def a_request(**changes):
    values = {
        "question": "Does drug X treat condition Y?",
        "inclusion_criteria": "trials reporting an outcome",
        "exclusion_criteria": "reviews and opinion pieces",
    }
    values.update(changes)
    return ResearchRequest(**values)


def an_outcome(**changes):
    values = {
        "subgoal": Subgoal(purpose="find trials of drug X", source="pubmed",
                           query="drug X AND trial"),
        "result": "finished",
        "papers_found": 12,
    }
    values.update(changes)
    return Outcome(**values)


def judged(**changes):
    values = {
        "paper": PaperRecord(
            title="A trial of drug X", pmid="12345678",
            abstract="Forty patients were given drug X.", text_source="abstract",
            origins=[SearchOrigin(source="pubmed", query="drug X")]),
        "screening": Screening(paper_number=1, decision="include", relevance=8,
                               reason="it reports an outcome",
                               summary="a trial of drug X",
                               extract="Forty patients were given drug X"),
    }
    values.update(changes)
    return ScreenedPaper(**values)


def test_the_query_the_agent_wrote_is_recorded(tmp_path):
    """The query and the bounds are recorded apart, because they are sent apart."""
    # Both services take the years as their own parameters rather than as words
    # in the query, so no single string is the search as sent, and a reader
    # repeating it needs the two together.
    log = tmp_path / "run.jsonl"

    note_a_search(an_outcome(), a_request(), "run one", log)

    assert read(log)[0]["query"] == "drug X AND trial"


def test_the_filters_asked_for_are_recorded_with_the_search(tmp_path):
    """The outcome says what came back; the request says what was asked for."""
    # What a service actually applied is not known here, which is why the field
    # names say "asked for" and why the planner checks the years of what came
    # back separately (ID-02).
    log = tmp_path / "run.jsonl"

    note_a_search(an_outcome(), a_request(from_year=2020, to_year=2024,
                                          results_per_search=25),
                  "run one", log)

    written = read(log)[0]
    assert written["from year asked for"] == 2020
    assert written["to year asked for"] == 2024
    assert written["results asked for"] == 25
    assert "2020" not in written["query"], "the years are sent as their own parameters"


def test_every_line_carries_the_time(tmp_path):
    """Nothing else records the time of a run's own events."""
    # Saved replies record when each reply was captured, which is a different
    # thing: it says when a service answered, not when a search was decided on
    # or a paper judged (UT-20).
    log = tmp_path / "run.jsonl"

    note_the_request(a_request(), "run one", log)
    note_a_search(an_outcome(), a_request(), "run one", log)
    note_a_failure("pubmed", "the service was busy", "run one", log)
    note_a_decision(judged(), "approved", "", "run one", log)

    times = [line["at"] for line in read(log)]
    assert len(times) == 4
    assert all(time.endswith("+00:00") for time in times), "written in UTC"


def test_the_source_is_recorded(tmp_path):
    log = tmp_path / "run.jsonl"

    note_a_search(an_outcome(), a_request(), "run one", log)

    assert read(log)[0]["source"] == "pubmed"


def test_the_model_a_run_used_is_recorded_with_its_request(tmp_path, monkeypatch):
    """The name the settings hold when the run starts, not the one the
    proposal names: a run made on another model has to say so beside the
    decisions that model made."""
    log = tmp_path / "run.jsonl"
    monkeypatch.setattr(run_log.settings, "gemini_model", "a-model-named-for-a-test")

    note_the_request(a_request(), "run one", log)

    assert read(log)[0]["model"] == "a-model-named-for-a-test"


def test_a_search_that_failed_says_so_and_says_why(tmp_path):
    """The outcomes keep a failed search too, so this is not the only record."""
    # What the log adds is when it happened and where it sat among the other
    # events, which is what says whether two failures were one bad minute.
    log = tmp_path / "run.jsonl"

    note_a_search(an_outcome(result="failed", papers_found=0,
                             reason="pubmed: service busy"), a_request(),
                  "run one", log)

    written = read(log)[0]
    assert written["result"] == "failed"
    assert written["what happened"] == "pubmed: service busy"


def test_a_failure_that_did_not_stop_the_run_is_recorded_with_its_time(tmp_path):
    """The list says what went wrong; the log says when."""
    # That is what tells a reader whether two failures were one service having a
    # bad minute or a fault running through the whole run.
    log = tmp_path / "run.jsonl"

    note_a_failure("the buffer", "no space left on the disk", None, log)

    written = read(log)[0]
    assert written["what"] == "the buffer"
    assert written["why"] == "no space left on the disk"
    assert written["at"]


def test_a_reviewer_decision_is_recorded_with_what_the_agent_had_said(tmp_path):
    """A reviewer disagreeing with the agent is the thing worth reading later,
    and it cannot be seen from the decision alone."""
    log = tmp_path / "run.jsonl"

    note_a_decision(judged(), "rejected", "it is a review, not a trial",
                    "run one", log)

    written = read(log)[0]
    assert written["decision"] == "rejected"
    assert written["note"] == "it is a review, not a trial"
    assert written["the agent had said"] == "include"
    assert written["relevance the agent gave"] == 8
    assert written["paper"] == "12345678"


def test_the_criteria_are_recorded_at_the_start(tmp_path):
    """A paper excluded under one set of criteria is not excluded under
    another, and a log of decisions alone cannot be judged."""
    log = tmp_path / "run.jsonl"

    note_the_request(a_request(), "run one", log)

    written = read(log)[0]
    assert written["question"] == "Does drug X treat condition Y?"
    assert written["include"] == "trials reporting an outcome"
    assert written["exclude"] == "reviews and opinion pieces"


def test_the_paper_s_own_text_does_not_reach_the_log(tmp_path):
    """The boundaries in `docs/development.md` send abstracts to the buffer."""
    # The abstract, the extract, the summary and the title are all the paper's
    # own words or built from them, and the identifier is what a reader follows.
    # What this does not cover is the reviewer's note, which is free text and
    # theirs to write.
    log = tmp_path / "run.jsonl"

    note_a_decision(judged(), "approved", "", "run one", log)

    written = log.read_text(encoding="utf-8")
    assert "Forty patients were given drug X" not in written, "abstract or extract"
    assert "a trial of drug X" not in written, "the summary"
    assert "A trial of drug X" not in written, "the title"
    assert "12345678" in written


def test_the_reviewer_s_note_is_written_as_they_wrote_it(tmp_path):
    """It is the reason a decision can be argued with later."""
    # It is also free text, so a reviewer who quotes a paper puts that quotation
    # in the log. That is a consequence of keeping their words, and it is
    # written down rather than left for somebody to find.
    log = tmp_path / "run.jsonl"

    note_a_decision(judged(), "rejected",
                    'it says "forty patients", which is not a trial', None, log)

    assert read(log)[0]["note"] == 'it says "forty patients", which is not a trial'


def test_a_run_that_stopped_half_way_still_has_the_lines_before_it(tmp_path):
    """One line for each event, so an interrupted run keeps what it did."""
    # A single record written at the end would lose exactly the run whose log is
    # most worth having.
    log = tmp_path / "run.jsonl"
    note_the_request(a_request(), "run one", log)
    note_a_search(an_outcome(), a_request(), "run one", log)

    with log.open("a", encoding="utf-8") as open_file:
        open_file.write('{"at": "half a line')

    assert [line["event"] for line in read(log)] == [
        "run started", "search", "unreadable line"]


def test_the_events_after_a_damaged_line_are_kept(tmp_path):
    """A line with no end would swallow the next event written after it."""
    # An interruption cost two events instead of one, and the second was lost
    # with nothing saying so.
    log = tmp_path / "run.jsonl"
    note_the_request(a_request(), "run one", log)
    with log.open("a", encoding="utf-8") as open_file:
        open_file.write('{"at": "half a line')

    note_a_failure("the buffer", "no space left", None, log)

    assert [line["event"] for line in read(log)] == [
        "run started", "unreadable line", "failure"]


def test_a_damaged_line_is_reported_rather_than_passed_over(tmp_path):
    """A reader told nothing would take the record as complete."""
    log = tmp_path / "run.jsonl"
    note_the_request(a_request(), "run one", log)
    with log.open("a", encoding="utf-8") as open_file:
        open_file.write("this was never JSON\n")

    damaged = [line for line in read(log) if line["event"] == "unreadable line"]
    assert damaged == [{"event": "unreadable line", "line number": 2}]


def test_every_event_carries_the_identifier_of_the_run_it_belongs_to(tmp_path):
    """The file is shared by every run on the machine."""
    # Without it a decision recorded an hour later could not be tied to the run
    # it was made in.
    log = tmp_path / "run.jsonl"
    request = a_request()

    note_the_request(request, "run one", log)
    note_a_search(an_outcome(), request, "run one", log)
    note_a_decision(judged(), "approved", "", "run one", log)

    assert {line["run"] for line in read(log)} == {"run one"}


def test_two_runs_of_the_same_question_are_two_runs_in_the_log(tmp_path):
    """Working the identifier out from the question was the first attempt."""
    # It named the question rather than the occasion, so the decisions of one
    # run could not be told from the other's.
    log = tmp_path / "run.jsonl"

    note_the_request(a_request(), new_identifier(), log)
    note_the_request(a_request(), new_identifier(), log)

    first, second = (line["run"] for line in read(log))
    assert first != second


def test_an_attempt_carries_the_run_it_was_made_inside(tmp_path):
    """The shared request handling is several calls below anything that knows
    which run is happening, so it is told once rather than handed an
    identifier through four sources that have no use for one (FR-15)."""
    log = tmp_path / "run.jsonl"

    with run_log.while_running("run one"):
        run_log.note_an_attempt("pubmed", "GET", "https://example.test/",
                                {"term": "x"}, log=log)

    assert read(log)[0]["run"] == "run one"


def test_a_request_made_after_the_run_is_not_filed_under_it(tmp_path):
    """The next thing to make a request, a measurement or somebody trying a
    source on its own, should not have it recorded against a run that had
    already finished."""
    log = tmp_path / "run.jsonl"

    with run_log.while_running("run one"):
        pass
    run_log.note_an_attempt("pubmed", "GET", "https://example.test/",
                            {"term": "x"}, log=log)

    assert read(log)[0]["run"] == "not recorded"


def test_two_runs_at_once_do_not_take_each_other_s_requests(tmp_path):
    """One value for the whole process gave one run's requests to another."""
    # Two runs at once is what two browser sessions are, and with a single value
    # the second to start took the first's requests with it, then left them
    # recorded as nobody's when it finished. Each thread has its own context, so
    # this shows the two are kept apart.
    started = threading.Barrier(2)

    def run_as(name):
        # A file each, because what is being checked is which run a line is
        # filed under. Two threads appending to one file could interleave, and
        # a test that failed for that reason would look like this one failing.
        with run_log.while_running(name):
            started.wait(timeout=5)
            run_log.note_an_attempt(name, "GET", "https://example.test/",
                                    {"term": "x"}, log=tmp_path / f"{name}.jsonl")

    one = threading.Thread(target=run_as, args=("run one",))
    other = threading.Thread(target=run_as, args=("run two",))
    one.start()
    other.start()
    one.join(timeout=5)
    other.join(timeout=5)

    assert read(tmp_path / "run one.jsonl")[0]["run"] == "run one"
    assert read(tmp_path / "run two.jsonl")[0]["run"] == "run two"


def test_a_run_inside_another_gives_the_first_one_back(tmp_path):
    """Clearing rather than putting back would lose the run around it, and
    leave the rest of that run's requests recorded as nobody's."""
    log = tmp_path / "run.jsonl"

    with run_log.while_running("the outer run"):
        with run_log.while_running("the inner run"):
            pass
        run_log.note_an_attempt("pubmed", "GET", "https://example.test/",
                                {"term": "x"}, log=log)

    assert read(log)[0]["run"] == "the outer run"


def test_the_run_in_progress_can_be_read_by_name():
    """The module that saves a model reply is imported by the log, so it
    cannot import the log back to read the run itself. llm.py reads it here
    and hands it in, and outside a run there is nothing to hand (FR-15)."""
    assert run_log.run_in_progress() == ""

    with run_log.while_running("run one"):
        assert run_log.run_in_progress() == "run one"

    assert run_log.run_in_progress() == ""


def test_no_event_is_lost_when_two_runs_write_at_once(tmp_path):
    """Looking at the end of the file and appending to it are two steps."""
    # Interleaved, they lose lines and say nothing: two hundred events kept a
    # hundred and ninety, and the ten that were gone left no trace. Two runs at
    # once is what two browser sessions are.
    log = tmp_path / "run.jsonl"
    each = 50

    def write_as(name):
        for number in range(each):
            run_log.note_a_failure(name, str(number), name, log)

    writers = [threading.Thread(target=write_as, args=(f"run {which}",))
               for which in range(4)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=30)

    written = read(log)
    assert len(written) == 4 * each, "an event was lost or written twice"
    assert not [line for line in written if line["event"] == "unreadable line"]
    for which in range(4):
        mine = [line["why"] for line in written if line["what"] == f"run {which}"]
        assert sorted(mine, key=int) == [str(number) for number in range(each)]


def test_only_the_values_that_describe_a_search_are_written(tmp_path):
    """The keys and the contact address travel with these requests, and one
    allowed list rather than two means the log and the saved replies cannot
    disagree about what may be written down (ID-05, ID-13)."""
    log = tmp_path / "run.jsonl"

    run_log.note_an_attempt(
        "pubmed", "GET", "https://example.test/",
        {"term": "drug X", "api_key": "a secret", "email": "a@example.test",
         "tool": "research-agent", "retmax": "50"}, run="run one", log=log)

    assert read(log)[0]["values sent beside it"] == [["term", "drug X"],
                                                     ["retmax", "50"]]


def test_an_address_is_written_without_what_it_carries(tmp_path):
    """An address can hold a key in its query and a password before its host,
    and writing it down whole let both through."""
    log = tmp_path / "run.jsonl"

    run_log.note_an_attempt(
        "pubmed", "GET",
        "https://someone:secret@example.test:8443/entrez?term=drug+X&api_key=k",
        run="run one", log=log)

    written = read(log)[0]
    assert written["url"] == "https://example.test:8443/entrez"
    assert written["values in the address"] == [["term", "drug X"]]
    assert "secret" not in json.dumps(written)
    assert "someone" not in json.dumps(written)


def test_a_name_sent_twice_is_kept_twice(tmp_path):
    """A mapping keeps only the last of them."""
    # A search that asked for two terms and recorded one is a different search,
    # and a reader repeating it would get different papers.
    log = tmp_path / "run.jsonl"

    run_log.note_an_attempt(
        "pubmed", "GET", "https://example.test/entrez?term=first&term=second",
        [("term", "beside first"), ("term", "beside second")],
        run="run one", log=log)

    written = read(log)[0]
    assert written["values in the address"] == [["term", "first"],
                                                ["term", "second"]]
    assert written["values sent beside it"] == [["term", "beside first"],
                                                ["term", "beside second"]]


def test_a_value_left_blank_is_written_as_blank(tmp_path):
    """"retmax=" is a request that was made, and a reader told nothing about
    it would think it never was."""
    log = tmp_path / "run.jsonl"

    run_log.note_an_attempt("pubmed", "GET",
                            "https://example.test/entrez?term=drug+X&retmax=",
                            run="run one", log=log)

    assert read(log)[0]["values in the address"] == [["term", "drug X"],
                                                     ["retmax", ""]]


def test_a_line_written_with_no_run_to_hand_says_so(tmp_path):
    """Better than naming a run it does not belong to."""
    # Some failures happen where no run is available, and a failure recorded
    # without one is still worth having.
    log = tmp_path / "run.jsonl"

    note_a_failure("something", "anything", None, log)

    assert read(log)[0]["run"] == "not recorded"


def test_the_log_is_looked_up_when_it_is_used_not_when_it_is_imported(
        tmp_path, monkeypatch):
    """A default is fixed when the file is read."""
    # Redirecting the log in a test was therefore redirecting something nothing
    # used, and the suite wrote its invented searches into a real run's file.
    somewhere_else = tmp_path / "redirected.jsonl"
    monkeypatch.setattr(run_log, "DEFAULT_LOG", somewhere_else)

    note_a_failure("something", "anything")

    assert somewhere_else.exists()
    assert read(somewhere_else)[0]["what"] == "something"


def test_reading_a_log_that_was_never_written_is_not_a_failure(tmp_path):
    assert read(tmp_path / "nothing here.jsonl") == []


def test_a_log_that_cannot_be_written_does_not_stop_the_run(tmp_path):
    """A run stopped by a fault in the record of its work would lose the work."""
    # The failure is collected instead, the way an unrecorded reply and an
    # unwritten buffer are, and the coverage report reads it (FR-05).
    # Read through the module, because the list itself is replaced for every
    # test so that one test's failure is not read as another's.
    before = len(run_log.writing_failures)
    a_folder_not_a_file = tmp_path / "in the way"
    a_folder_not_a_file.mkdir()

    note_a_failure("something", "anything", None, a_folder_not_a_file)

    assert len(run_log.writing_failures) == before + 1
    assert "could not be written" in run_log.writing_failures[-1]


def test_the_lines_are_json_one_to_a_line(tmp_path):
    """Read by people and by whatever reports on a run, so the format has to
    be one nothing has to guess at."""
    log = tmp_path / "run.jsonl"
    note_the_request(a_request(), "run one", log)
    note_a_search(an_outcome(), a_request(), "run one", log)

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["event"] for line in lines)


def a_log(tmp_path, *entries):
    """A log written line by line, so a test decides what it holds."""
    # The writing above is not used for the reading tests. What they check is
    # how lines that arrived in a particular order, or that arrived damaged, are
    # separated and shown, and building the file is the only way to put a line
    # in a state the writing would never produce.
    log = tmp_path / "run.jsonl"
    log.write_text("".join(json.dumps(entry) + "\n" for entry in entries),
                   encoding="utf-8")
    return log


def test_the_lines_of_two_runs_are_separated_by_the_identifier_they_carry(
        tmp_path):
    """One file holds every run the computer has made."""
    # Two runs worked on in one afternoon interleave in it, so where a line sits
    # says nothing about which run it belongs to.
    log = a_log(
        tmp_path,
        {"run": "one", "at": "2026-09-08T10:00:00+00:00", "event": "search"},
        {"run": "two", "at": "2026-09-08T11:00:00+00:00", "event": "search"},
        {"run": "one", "at": "2026-09-08T10:00:30+00:00", "event": "failure"},
    )

    found = {one.run: one for one in runs(log)}
    assert [entry["event"] for entry in found["one"].entries] == [
        "search", "failure"]
    assert [entry["event"] for entry in found["two"].entries] == ["search"]


def test_the_most_recently_started_run_is_first(tmp_path):
    """A person opening this has almost always come for the run they just
    made."""
    log = a_log(
        tmp_path,
        {"run": "older", "at": "2026-09-07T09:00:00+00:00", "event": "search"},
        {"run": "newer", "at": "2026-09-08T09:00:00+00:00", "event": "search"},
    )

    assert [one.run for one in runs(log)] == ["newer", "older"]


def test_a_run_no_line_of_which_carries_a_time_is_put_last(tmp_path):
    """It cannot be placed among the others."""
    # First would push the run somebody came to read off the top of the list.
    log = a_log(
        tmp_path,
        {"run": "no time", "event": "search"},
        {"run": "timed", "at": "2026-09-08T09:00:00+00:00", "event": "search"},
    )

    assert [one.run for one in runs(log)] == ["timed", "no time"]


def test_a_run_is_placed_by_its_first_line_that_carries_a_time(tmp_path):
    """A damaged first line has no time on it."""
    # Reading the first line only would treat the whole run as one that cannot
    # be placed, when every line after it says when it happened.
    log = a_log(tmp_path, {"run": "damaged first", "event": "search"},
                {"run": "damaged first", "at": "2026-09-08T09:00:00+00:00",
                 "event": "search"})

    assert runs(log)[0].started == "2026-09-08T09:00:00+00:00"


def test_lines_belonging_to_no_run_are_gathered_rather_than_dropped(tmp_path):
    """A failure recorded where no run was to hand still happened."""
    # Shown as part of whichever run it lies next to it would be blamed on that
    # run; left out, the reader is shown a complete-looking record with a
    # failure missing from it.
    log = tmp_path / "run.jsonl"
    log.write_text(
        json.dumps({"run": "one", "at": "2026-09-08T09:00:00+00:00",
                    "event": "search"}) + "\n"
        + json.dumps({"run": "not recorded", "at": "2026-09-08T09:00:01+00:00",
                      "event": "failure", "what": "the buffer"}) + "\n"
        + "this was never JSON\n",
        encoding="utf-8")

    found = {one.run: one for one in runs(log)}
    assert [entry["event"] for entry in found["not recorded"].entries] == [
        "failure", "unreadable line"]


def test_a_run_is_named_by_its_time_its_question_and_its_identifier(tmp_path):
    """The wording given on 17 September for the run selector: the time on this
    computer's clock first, then the first three words of the question, then
    the identifier, which two runs of the same question are told apart by.
    """
    # The expected time is made with the same conversion, so the test does not
    # depend on the zone of the computer running it.
    log = a_log(tmp_path,
                {"run": "abc123", "at": "2026-09-08T09:00:00+00:00",
                 "event": "run started", "question": "Does drug X treat Y?"})

    started = datetime.datetime.fromisoformat("2026-09-08T09:00:00+00:00")
    assert heading(runs(log)[0]) == (
        f"Started {started.astimezone():%d-%b-%Y %H:%M}, Does drug X, abc123")


def test_a_run_whose_time_cannot_be_read_says_so_in_its_name(tmp_path):
    """A damaged stamp is shown as it is, and a missing one is said to be not
    recorded, rather than either being a blank in the name."""
    damaged = a_log(tmp_path, {"run": "abc123", "at": "yesterday",
                               "event": "run started", "question": "test"})
    assert heading(runs(damaged)[0]) == "Started yesterday, test, abc123"

    missing = a_log(tmp_path, {"run": "abc123", "event": "run started",
                               "question": "test"})
    assert heading(runs(missing)[0]) == (
        "Started at a time not recorded, test, abc123")


def test_a_run_whose_question_was_never_written_says_so(tmp_path):
    """A run whose first line was lost has lines but no question, and a blank
    where the question goes reads as a run that was given none."""
    log = a_log(tmp_path, {"run": "abc123", "event": "search"})

    assert "question not recorded" in heading(runs(log)[0])


def test_a_field_the_reader_does_not_know_the_name_of_is_still_shown(tmp_path):
    """A reader naming its fields drops a field added to an event later."""
    # A record of what happened that leaves part of it out is the failure this
    # log exists to prevent.
    log = a_log(tmp_path,
                {"run": "one", "at": "2026-09-08T09:00:00+00:00",
                 "event": "search", "something added later": "worth reading"})

    assert "something added later: worth reading" in as_lines(runs(log)[0].entries)[0]


def test_a_line_nobody_could_read_is_shown_as_one(tmp_path):
    """It has no time and no fields."""
    # Shown the ordinary way it would say the time was not recorded, which
    # describes a line written badly rather than one that cannot be read.
    log = tmp_path / "run.jsonl"
    log.write_text("this was never JSON\n", encoding="utf-8")

    assert as_lines(runs(log)[0].entries) == [
        "line 1 of the log could not be read"]


def test_the_identifier_is_not_repeated_on_every_line_shown(tmp_path):
    """Every line of one run carries it, and the heading says it once."""
    log = a_log(tmp_path,
                {"run": "abc123", "at": "2026-09-08T09:00:00+00:00",
                 "event": "search", "source": "pubmed"})

    assert "abc123" not in as_lines(runs(log)[0].entries)[0]


def test_reading_a_log_that_was_never_written_finds_no_runs(tmp_path):
    assert runs(tmp_path / "nothing here.jsonl") == []


def test_a_log_that_cannot_be_read_at_all_is_reported_rather_than_raised(
        tmp_path):
    """Whatever is showing the log has papers to put in front of a reviewer."""
    # A fault in the record of a run is not a reason to give them none of it,
    # and an empty list would say the run did nothing.
    a_folder_not_a_file = tmp_path / "run.jsonl"
    a_folder_not_a_file.mkdir()

    shown = as_lines(runs(a_folder_not_a_file)[0].entries)

    assert shown[0].startswith("the log itself could not be read")
