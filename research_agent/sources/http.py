"""Provides the request handling shared by every source of papers."""
# Each source uses this module to identify the application, keep within the
# request limit set by its service, and deal with temporary failures the same
# way.
#
# Requests are attempted no more than three times. Responses that may indicate a
# temporary problem, including status 429 and server errors, are retried because
# a later attempt may succeed. A status of 400 means the service rejected the
# request itself, which would happen again, so those are returned without retry.
# Each wait is twice the one before it, so a service that is busy is given
# noticeably longer to recover between attempts. The Planning Agent bounds its repairs of an
# unusable model answer in the same spirit, though not with the same number: it
# allows one attempt and three repairs, which is what Figure 2 sets for that
# case. A fixed limit is
# necessary because the agent has finite time: after repeated failures, it must
# stop and either proceed with the available information or report the failure,
# rather than continue deliberating indefinitely, as Bratman, Israel and
# Pollack (1988) argue (DC-06).
#
# After the final unsuccessful attempt, the module returns a description of the
# failure instead of stopping the run with an exception. A failure returned this
# way can be counted, which is what the coverage report will need in order to say
# what a search missed. A reported search has to say how many records each source
# returned, so a source that could not be reached has to be visible rather than
# absent, which is what PRISMA-S asks of a search that is written up
# (Rethlefsen et al., 2021). It answers the
# risk the design identified: a search that returns part of the evidence but
# still reads as complete.

import time
from dataclasses import dataclass

import httpx

from research_agent import certificates
from research_agent.config import settings
from research_agent.processing import run_log
from research_agent.sources import saved_replies

# Must run before any connection is opened. See certificates.py (R-01).
certificates.install()

# NCBI asks that a large job be run at weekends or between 9 pm and 5 am
# Eastern Time (Sayers, 2022), and gives no number for "large". A hundred
# attempts in one run is this project's own threshold, chosen so that a run
# needing many identifier fetches counts as large (OB-04). It is kept
# here, where every request passes, and counted before each attempt goes out,
# so that retries and every search in the run are in the count. A run that
# reaches it has the rest of its NCBI attempts refused and reported rather
# than made: nothing here checks the window, so a run that carried on past the
# threshold outside it would have breached the guidance with nothing saying
# so, and a run inside the window is refused too, which is the cost of not
# checking.
NCBI_REQUESTS_PER_RUN = 100

# The sources that draw on NCBI's one allowance, named as they name themselves
# in a request. One since 19 September, when the PubMed Central source was
# removed as never used by a run; kept as a tuple so a second NCBI source
# would be counted under the same allowance (OB-01).
NCBI_SOURCES = ("pubmed",)

# How many attempts each run has made at NCBI, by run identifier. Attempts,
# not deliveries: an attempt is counted before the connection is asked, and a
# connection can fail before anything leaves the computer, so the count can
# run ahead of what NCBI received. That is the cautious side to be on. Per
# run rather than for the whole process because two browser sessions are two
# runs in one process, and one run's attempts must not spend another's
# threshold (R-19). A caller outside a run, such as the sample-drawing
# script, is counted under the empty identifier. Tests empty it between
# tests, in tests/conftest.py.
ncbi_attempts_by_run = {}

# Status codes worth trying again. Any other code will give the same answer.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# How long to wait after the first failure. Each further wait doubles it.
FIRST_WAIT_SECONDS = 1.0


@dataclass
class FetchError:
    """A failed request, described so that it can be counted and reported."""

    source: str
    url: str
    status: int | None
    reason: str
    attempts: int


class RateLimiter:
    """Keeps a minimum gap between requests to one service."""

    def __init__(self, requests_per_second):
        self.requests_per_second = requests_per_second
        self.min_gap = 1.0 / requests_per_second
        self.last_call = 0.0

    def set_rate(self, requests_per_second):
        """Changes the gap, used when a service reports its own limit."""
        if requests_per_second <= 0:
            return
        self.requests_per_second = requests_per_second
        self.min_gap = 1.0 / requests_per_second

    def wait(self):
        """Waits before the next request when the last one was too recent."""
        gap = time.monotonic() - self.last_call
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self.last_call = time.monotonic()


# The limiter each service uses, made the first time it is asked for. Kept
# here rather than in each source because two sources can share one allowance.
_limiters: dict[str, RateLimiter] = {}


def limiter_for(service, requests_per_second):
    """Returns the one limiter belonging to a service."""
    # A service counts every request from this application together, whichever
    # part of the application sent it, so there has to be one limiter for each
    # service rather than one for each source file. PubMed's search and fetch
    # both ask for "ncbi" and share a limiter for that reason: they are one
    # service with one allowance.
    #
    # A limiter made inside each call would be worse than none. It starts with
    # nothing to wait for, so two calls one after another would go out together,
    # which is exactly what the limit forbids.
    if service not in _limiters:
        _limiters[service] = RateLimiter(requests_per_second)
    return _limiters[service]


def _follow_reported_limit(limiter, headers):
    """Uses the request limit a service reports, when it reports one."""
    # Crossref returns its current limit with every response, as a number of
    # requests and the period they apply to. Following that figure means the
    # application keeps to a limit that has changed, rather than one written into
    # the code that would silently become wrong. Services that send no such
    # headers, such as NCBI, are left on the rate they were given.
    reported_requests = headers.get("x-rate-limit-limit")
    reported_period = headers.get("x-rate-limit-interval")
    if not reported_requests or not reported_period:
        return

    try:
        requests_allowed = float(reported_requests)
        # The period arrives as text such as "1s".
        seconds = float(reported_period.rstrip("s"))
    except ValueError:
        # An unexpected format is ignored rather than allowed to stop a run.
        return

    if seconds > 0:
        limiter.set_rate(requests_allowed / seconds)


def _count_or_refuse_an_ncbi_attempt(source, url, attempts_so_far):
    """Counts one more attempt against the run's NCBI threshold, or refuses it."""
    # Returns None when the attempt may go out, and a FetchError when the run
    # has already counted the hundred attempts this project treats as a large
    # job. The refusal is a failure like any other, so the search that met it
    # is reported as failed or partial and the coverage report says why. A
    # source that is not NCBI's is never counted: the threshold is
    # about one service's guidance, not about the run as a whole.
    if source not in NCBI_SOURCES:
        return None
    run = run_log.run_in_progress()
    counted = ncbi_attempts_by_run.get(run, 0)
    if counted >= NCBI_REQUESTS_PER_RUN:
        return FetchError(
            source, url, None,
            f"this run has already counted {NCBI_REQUESTS_PER_RUN} attempts "
            "at NCBI, the number this project treats as a large job, and NCBI "
            "asks that a large job be run at weekends or overnight in Eastern "
            "Time; no further attempt is made (OB-04)",
            attempts_so_far)
    ncbi_attempts_by_run[run] = counted + 1
    return None


def request(connection, method, url, source, params=None, data=None, limiter=None,
            max_attempts=3):
    """Returns a reply from the service, or a description of what went wrong."""
    # When DEMO_MODE is on, a saved reply is used and nothing is sent over the
    # network. When RECORD_REPLIES is on, each reply is saved so it can be used
    # that way later. See saved_replies.py.
    if settings.demo_mode:
        return _replay(method, url, source, params, data)

    status = None
    reason = "unknown"

    for attempt in range(1, max_attempts + 1):
        # Counted before the wait as well as before the request, so a refused
        # attempt neither waits for a slot nor takes one (OB-04).
        refused = _count_or_refuse_an_ncbi_attempt(source, url, attempt - 1)
        if refused is not None:
            return refused

        if limiter:
            limiter.wait()

        # Written before the request is handed over rather than after a reply
        # comes back, so a request that timed out is still recorded as having
        # been tried. It records an attempt and not a delivery: a connection
        # can fail before anything leaves the machine. This is the one place
        # every request passes through, which is why it is the place that can
        # say what each one carried.
        run_log.note_an_attempt(source, method, url, params, data, attempt)

        try:
            response = connection.request(method, url, params=params, data=data)
            status = response.status_code

            if status == 200:
                if limiter:
                    _follow_reported_limit(limiter, response.headers)
                if settings.record_replies:
                    # A reply that cannot be filed away is not a failed
                    # request. The result is returned either way.
                    saved_replies.save(method, url, params, data, response)
                return response

            if status not in RETRYABLE_STATUS:
                # The service rejected the request itself, so asking again
                # would give the same answer.
                return FetchError(source, url, status, f"status {status}", attempt)

            reason = f"status {status}"

        except (httpx.TimeoutException, httpx.TransportError) as error:
            reason = str(error)

        # Wait twice as long after each failure: one second, then two. Every
        # service used here asks callers to back off rather than keep trying at
        # the same rate, and Semantic Scholar requires it of applications
        # requesting a key. Doubling means a busy service is given noticeably
        # longer to recover, rather than being asked again almost immediately.
        if attempt < max_attempts:
            time.sleep(FIRST_WAIT_SECONDS * (2 ** (attempt - 1)))

    return FetchError(source, url, status, reason, max_attempts)


def get_json(connection, url, source, params=None, limiter=None, max_attempts=3):
    """Returns the reply read as JSON, or a description of the failure."""
    result = request(connection, "GET", url, source, params=params, limiter=limiter,
                     max_attempts=max_attempts)
    if isinstance(result, FetchError):
        return result

    try:
        return result.json()
    except ValueError as problem:
        # A reply with a status of 200 that is not readable JSON is a failed
        # request as far as the caller is concerned, and reporting it that way
        # keeps every failure countable. Letting it raise here would end a run
        # over one bad reply and lose the sources that answered.
        return FetchError(source, url, 200,
                          f"the reply was not readable JSON: {problem}", 1)


def post_text(connection, url, source, data, limiter=None, max_attempts=3):
    """Returns the reply as text, sending the values in the request body."""
    # The body is used rather than the web address because NCBI returned
    # incomplete replies once the list of identifiers made the address long
    # (R-02).
    result = request(connection, "POST", url, source, data=data, limiter=limiter,
                     max_attempts=max_attempts)
    if isinstance(result, FetchError):
        return result
    return result.text


def _replay(method, url, source, params, data):
    """Rebuilds a saved reply so the rest of the code cannot tell the difference."""
    # A request with no saved reply returns an error rather than silently doing
    # nothing, because a demonstration that quietly skipped a service would be
    # worse than one that stopped and said so.
    saved = saved_replies.load(method, url, params, data)
    if saved is None:
        return FetchError(source, url, None,
                          "DEMO_MODE is on but no reply has been saved for this "
                          "request. Run again with DEMO_MODE=false and "
                          "RECORD_REPLIES=true to save one.", 0)

    return httpx.Response(saved["status"],
                          content=saved["body"].encode("utf-8"),
                          headers=saved["headers"])
