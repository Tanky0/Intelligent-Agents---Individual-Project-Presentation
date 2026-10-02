"""Saves service replies to disk and plays them back later."""
# The application depends on services it does not control: PubMed and PubMed
# Central, Semantic Scholar, Crossref, and the language model. During a
# demonstration a daily free allowance may already be spent, a service may be
# down, or a slow connection may time out. None of those are faults in this code,
# but any of them would stop it running.
#
# Every reply is saved as it arrives, so whatever was run last can be run again
# with no network at all. Saving happens without being asked for, because a
# demonstration set that depends on remembering to switch something on beforehand
# is one that will eventually be missing.
#
# Replies are kept in two places, and the difference between them is the point.
#
#   data/captured_replies    everything the application receives, as it arrives.
#                            Not in the repository. Nothing is replayed from here.
#   data/recorded_replies    the replies approved for the demonstration. In the
#                            repository, listed in a manifest, and the only place
#                            DEMO_MODE reads from.
#
# Capture stays automatic; publication becomes deliberate. A reply reaches the
# demonstration set only by running scripts/approve_replies.py, which checks it
# first and writes its fingerprint into the manifest. Anything in that folder
# without a matching fingerprint is reported by the tests, whatever it contains.
#
# That split exists because invented replies have twice reached the folder used
# for demonstrations: nine when renaming tests/conftest.py removed the protection
# it provides, and one from a script that made a pretend request with recording
# still switched on. Both were found by luck rather than by a check. A manifest
# finds them by arithmetic instead (R-07).
#
# What is written down is chosen by a list of what to keep rather than a list of
# what to remove. A list of things to remove can only ever catch the ones
# somebody thought of: an earlier version removed anything called api_key, and
# let through access_token, authorization, password, the contact email and the
# session identifiers that NCBI sets on every reply. Keeping only what replay
# actually needs makes that class of mistake impossible rather than unlikely.
#
# The paper text is the one field that cannot be handled that way. It is a whole
# document in a single value, and a list of what to keep has nothing to choose
# between. It was therefore left alone, and that was wrong: an article fetched
# from PubMed Central carries its authors' contact details inside it, and
# approving those replies put 103 email-address occurrences, representing 89
# distinct addresses, into a public repository. Nobody consented to that, and
# none of it is needed to replay a search. The text is now rewritten before
# publication, taking out the parts that identify a person and leaving what the
# paper says (R-10).

import datetime
import hashlib
import json
import re

from research_agent.config import REPO_ROOT, settings
from research_agent.sources import publication_rights

# Everything received, as it arrives. Kept out of the repository.
CAPTURED_DIR = REPO_ROOT / "data" / "captured_replies"

# The approved replay set: the replies approved for the demonstration, and the
# only ones replayed.
#
# Older comments here, and the remediation log's entries on the contact
# details and the publication rights, call this the published folder. That
# was accurate while the folder's
# contents were published. Since 3 September they are not, and everything in
# this folder is recorded as kept for replaying here with no right to publish it
# established, so "published" now describes what happened rather than what
# these files are. Where a comment below still says published and means the act
# of promoting a reply into this folder, that is what it means (R-12).
APPROVED_DIR = REPO_ROOT / "data" / "recorded_replies"

# The fingerprint of every approved reply, written when it is approved.
MANIFEST_PATH = REPO_ROOT / "data" / "recorded_replies_manifest.json"

# The only request values that are written down, and the only ones used to work
# out a file name. They describe what was asked. Everything else says who was
# asking: a key, a contact address, a tool name. None of that belongs in a file
# in a public repository, and leaving it out of the file name as well means a
# saved reply still matches after a key or an address has changed.
# Every value here changes what comes back, so every one has to be part of the
# file name. A value that changes the reply but is left out would give two
# different requests the same name, and a demonstration could then replay a
# reply captured for a different question (R-05).
# "year" and "datetype" were missing until the date limits were built.
# Semantic Scholar takes the
# whole range as one "year" value, so a search with no date, one for 2020-2024
# and one for 1990-1999 all produced the same file name and could overwrite one
# another. PubMed keeps mindate and maxdate, so its ranges did differ, but
# "datetype" decides which date those bounds are read against, and pdat and edat
# gave identical names for otherwise identical requests. Checking the years of
# what comes back cannot recover a paper absent from the wrong reply, so the
# date has to be part of the identity and not only of the search.
_REQUEST_VALUES_TO_KEEP = (
    "db", "dbfrom", "term", "id", "retmax", "retmode", "rows", "query",
    "sort", "mindate", "maxdate", "datetype", "year",
    "field", "fields", "limit", "offset",
)

# The only reply headers that are written down. Replay needs the content type,
# and the rate limit figures so that a replayed run behaves like a real one.
# Everything else is dropped, including the cookies and session identifiers
# NCBI returns, which identify the person who made the request rather than
# saying anything about the paper that came back.
_REPLY_HEADERS_TO_KEEP = (
    "content-type", "x-rate-limit-limit", "x-rate-limit-interval",
)

# Words that must never appear in the part of a record describing the request
# and the reply. Each one identifies a person or a session rather than saying
# anything about a paper. Declared here, in the file that decides what may be
# written, so that approving a reply and checking the approved set both use one
# list rather than two that can drift apart.
NEVER_PUBLISH = (
    "ncbi-sid", "ncbi_sid", "set-cookie", "cookie",
    "authorization", "api_key", "access_token", "@",
)

# The parts of a record that describe the request and the reply. The paper text
# itself is not searched for the words above, because an abstract can contain
# an author's address and a title can contain almost anything. Refusing a reply
# on that basis would reject good papers for what they quote, which is why the
# text is redacted instead of judged.
#
# The run is on the list because a field left off it reaches the published
# folder unscreened. An identifier made on this computer is twelve hexadecimal
# characters and cannot carry a key or an address, so checking it costs
# nothing, and a field that skips the check is the arrangement this list
# exists to refuse.
DESCRIBING_FIELDS = ("captured_at", "method", "url", "asked", "status",
                     "headers", "model", "question", "run")

# An email address anywhere in the paper text, and the JATS elements that exist
# to hold one. PubMed Central returns the corresponding author's address, often
# every author's, in <email> elements inside <contrib>, and again as a mailto
# link. None of it is read by any adapter in this project: the adapters take
# titles, abstracts, authors and identifiers. Taking it out therefore costs
# replay nothing and costs the people named nothing either.
_AN_EMAIL_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_A_CONTACT_ELEMENT = re.compile(r"<(email|phone|fax)\b[^>]*>.*?</\1\s*>",
                                re.DOTALL | re.IGNORECASE)

# What replaces one. A marker rather than an empty string, so that a reply read
# later shows the text was changed on purpose and does not look as though the
# service returned nothing.
REDACTED = "[contact detail removed]"


def _contact_details_removed(text):
    """Returns the paper text with other people's contact details taken out."""
    # Two passes, because one is not enough. The first empties the elements that
    # exist to carry a contact detail, including their attributes, since a mailto
    # link keeps the address in the attribute rather than between the tags. The
    # second takes any address still loose in the text, which is where a
    # corresponding-author note or an acknowledgement puts one.
    #
    # A model answer is a record rather than a string, a summary and an extract
    # among other fields, and an extract quoting a corresponding author's line
    # carries the address inside that record. Until 16 September only a string
    # was rewritten and anything else was returned as it was, so such an
    # address reached the published copy whole. Records and lists are now
    # walked and every string inside them rewritten (R-21).
    #
    # Anything else, a number or nothing, is returned unchanged, so a record
    # holding an unexpected shape is refused elsewhere rather than damaged here.
    if isinstance(text, dict):
        rewritten = {}
        for name, value in text.items():
            rewritten[name] = _contact_details_removed(value)
        return rewritten
    if isinstance(text, list):
        rewritten = []
        for value in text:
            rewritten.append(_contact_details_removed(value))
        return rewritten
    if not isinstance(text, str):
        return text
    without_elements = _A_CONTACT_ELEMENT.sub(
        lambda found: f"<{found.group(1)}>{REDACTED}</{found.group(1)}>", text)
    return _AN_EMAIL_ADDRESS.sub(REDACTED, without_elements)


def configured_keys_found_in(record):
    """The configured keys whose value appears anywhere in this record."""
    # The word list refuses the names a key travels under, and the list of
    # request values to keep drops everything else at capture, so a key reaches
    # a captured file only by a mistake or by a hand: a source that put its
    # query into the address, or a file written to get past the list under a
    # name it does not know, "apikey" for "api_key". A check on names cannot
    # see either. The value can be seen, and comparing the three keys this
    # computer holds against the whole record costs three searches (R-21).
    #
    # The whole record rather than the describing fields, because a key is
    # never part of a paper and a service that echoes one in an error message
    # puts it in the body.
    text = json.dumps(record)
    found = []
    for name in ("ncbi_api_key", "semantic_scholar_api_key", "gemini_api_key"):
        value = getattr(settings, name, "")
        if value and value in text:
            found.append(name)
    return found


def holds_contact_details(text):
    """True when this text still carries an email address."""
    # Used by the tests against the published folder. Written as a question about
    # the text rather than about a file, so the same check covers a reply about to
    # be approved and one approved months ago.
    return bool(isinstance(text, str) and _AN_EMAIL_ADDRESS.search(text))


# Replies that could not be written down, in the order it happened. A failed
# save must not fail a request that succeeded, but a run that captured nothing
# while appearing to work would send somebody to a demonstration with no
# recording to approve, and the daily model allowance is twenty requests. The
# failure is collected here so the run report can say so.
recording_failures = []


def _values_worth_keeping(values):
    """Returns only the request values that describe what was asked."""
    kept = {}
    for name, value in (values or {}).items():
        if name.lower() in _REQUEST_VALUES_TO_KEEP:
            kept[name] = value
    return kept


def _headers_worth_keeping(headers):
    """Returns only the reply headers that replaying a reply needs."""
    kept = {}
    for name, value in headers.items():
        if name.lower() in _REPLY_HEADERS_TO_KEEP:
            kept[name] = value
    return kept


def _file_name(description):
    """Turns a description of a request into a short, repeatable file name."""
    text = json.dumps(description, sort_keys=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest[:20]}.json"


def _name_for(method, url, params, data):
    """Builds the file name for one request to a service."""
    # The name comes from the request itself, so asking the same thing again
    # finds the reply that was saved for it.
    return _file_name([method.upper(), url,
                       sorted(_values_worth_keeping(params).items()),
                       sorted(_values_worth_keeping(data).items())])


def _name_for_question(model, question):
    """Builds the file name for one question put to the language model."""
    # The same question asked of the same model finds the same saved answer.
    # Changing a single word produces a different name, so the earlier answer is
    # correctly treated as not matching.
    return _file_name(["model", model, question])


def _write(path, record):
    """Writes one record, and says whether it succeeded."""
    # The text is written to a second file first and then moved into place. A run
    # interrupted midway, or two runs writing at once, would otherwise leave a
    # half-written file that later looks like a saved reply and cannot be read.
    #
    # A failure here is reported rather than raised. The request itself already
    # succeeded, and losing a real result because the evidence of it could not be
    # filed away would be the wrong way round.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        partial.write_text(json.dumps(record, indent=2), encoding="utf-8")
        partial.replace(path)
        return True
    except OSError:
        return False


def _read(path):
    """Reads one record, or returns None if there is not a usable one."""
    # A file that cannot be read is treated the same as one that is not there.
    # Replay then reports that nothing was saved for this request, which is what
    # the caller can act on, rather than stopping the run with a complaint about
    # the shape of a file.
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None


def now():
    """The time this reply was saved, so a stale one can be recognised."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def save(method, url, params, data, response):
    """Writes one reply to the captured folder. False if it could not be written."""
    record = {
        "captured_at": now(),
        "method": method.upper(),
        "url": url,
        "asked": _values_worth_keeping(params) or _values_worth_keeping(data),
        "status": response.status_code,
        "headers": _headers_worth_keeping(response.headers),
        "body": response.text,
    }
    name = _name_for(method, url, params, data)
    written = _write(CAPTURED_DIR / name, record)
    if not written:
        # Collected here rather than returned for the caller to check, because
        # the caller ignoring it is exactly how this stayed invisible.
        recording_failures.append(f"{method.upper()} {url} could not be recorded")
    return written


def is_approved(path):
    """True when this file is one somebody approved, exactly as it stands."""
    # Checked before a reply is replayed rather than only afterwards by the
    # tests. A file with the right name dropped into the approved folder would
    # otherwise be played back during a demonstration, and only reported as
    # unapproved the next time the tests ran.
    #
    # Three things have to agree, not one. The file on disk, the fingerprint at
    # the top of its entry, and the fingerprint the latest approval recorded.
    # Checking only the first two let a redacted file be replayed: redaction
    # changes the file and updates the entry, and what it does not do is get the
    # file approved again, so the record of what somebody actually accepted was
    # the one part that still disagreed and the one part nothing read.
    #
    # A manifest that cannot be read means nothing is approved. That is the safe
    # direction: refusing to replay ends a demonstration, while replaying against
    # a manifest nobody can read is exactly what the manifest is there to stop.
    return _the_approved_reply_in(path) is not None


def _the_approved_reply_in(path):
    """The reply this file holds, if it is approved, or None if it is not."""
    # The bytes are read once, and everything after that is decided about those
    # bytes: their fingerprint, what they parse to, and what reading them finds.
    # The record returned is the one that was checked.
    #
    # Written this way because the two steps used to be separate. Replay asked
    # whether the file was approved, which read it, and then read it again to
    # return it. A file changed between those two reads would have been approved
    # as one thing and replayed as another, which is the gap the fingerprint
    # exists to close and was standing open beside it.
    # The bytes, not the text. Reading as text translates line endings on
    # Windows, so hashing what came back from that would give a different
    # fingerprint from the one recorded and nothing would ever replay.
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    here = hashlib.sha256(raw).hexdigest()

    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return None

    try:
        details = read_manifest()["files"].get(path.name)
    except ManifestProblem:
        return None

    if not details:
        return None
    if details.get("needs_approving_again"):
        return None

    latest = latest_approval(details)
    if latest is None:
        return None
    if here != details.get("sha256") or here != latest.get("approved_sha256"):
        return None

    # Approved is not the same as publishable. A reply whose fingerprints all
    # agree can still be one nobody ever decided could be published, which is
    # what every approval recorded before the rights check existed looks like
    # (R-11).
    if why_it_cannot_be_replayed(details, path, record, here):
        return None

    return record


def load(method, url, params, data):
    """Returns an approved reply for this request, or None if there is not one."""
    # Only the approved folder is read, and only files the manifest agrees with.
    # A reply that has been captured but not approved is deliberately not found,
    # so a demonstration cannot quietly replay something nobody has looked at.
    path = APPROVED_DIR / _name_for(method, url, params, data)
    return _the_approved_reply_in(path)


def save_model_reply(model, question, answer, run=None):
    """Writes one answer from the language model. Returns False on failure."""
    # The run that produced the answer goes inside the record, not into the
    # file name. The name is a hash of the model and the question, and writing
    # overwrites, so the same question asked in a later run replaces the
    # earlier answer. The capture time says when, and a time does not name an
    # execution: two runs on one day cannot be told apart by it, and nothing
    # on disk said which run an answer came from. Re-keying the name by run
    # was the alternative, and it was worse: the name is where the manifest
    # keeps each file's entry and what replay works out from the request, so
    # every entry and every lookup would have changed days before the
    # technical freeze, while the release boundary was still unproven
    # (R-12). A field disturbs none of that.
    #
    # The run is handed in rather than read from the run log, because the run
    # log imports this module and importing it back would make a cycle. When
    # there is no run the field is left out rather than written blank. An
    # answer captured outside any run and one captured before the field
    # existed then read the same, which is right: no run was recorded for
    # either, and the record cannot say which of the two it was.
    record = {
        "captured_at": now(),
        "model": model,
        "question": question,
        "answer": answer,
    }
    if run:
        record["run"] = run
    written = _write(CAPTURED_DIR / _name_for_question(model, question), record)
    if not written:
        recording_failures.append(f"an answer from {model} could not be recorded")
    return written


def load_model_reply(model, question):
    """Returns an approved answer for this question, or None if there is not one."""
    # As with load, an answer the manifest does not agree with is not replayed.
    path = APPROVED_DIR / _name_for_question(model, question)
    return _the_approved_reply_in(path)


def run_of_each_model_reply(folder=None):
    """Which run each saved model reply says it came from."""
    # One entry for each model reply in the folder, the approved folder unless
    # another is named: the file name, and the run written in the record, or
    # None when it carries no run. Service replies are left out, because
    # nothing writes a run into them.
    #
    # The caller gets the whole list rather than a yes or no, because a replay
    # set can be wrong in ways that need different words. Two runs mixed
    # together is the fault the field was added to catch. A reply with no run
    # is not that fault: no run was recorded for it, whether it was captured
    # before the field existed or outside a run, and the record cannot say
    # which. A check that failed on those would refuse every older recording.
    if folder is None:
        folder = APPROVED_DIR
    runs = {}
    for path in files_in(folder):
        record = _read(path)
        if isinstance(record, dict) and "question" in record:
            runs[path.name] = record.get("run")
    return runs


def count():
    """Returns how many replies have been captured but not yet approved."""
    return len(files_in(CAPTURED_DIR))


def files_in(folder):
    """Returns the saved replies in a folder, in a settled order."""
    if not folder.exists():
        return []
    return sorted(folder.glob("*.json"))


def fingerprint(path):
    """Returns the fingerprint of a file, used to notice it changing."""
    # Two files with the same fingerprint hold the same bytes. A file edited by
    # hand after it was approved gets a different one, which is how an alteration
    # is noticed rather than assumed not to happen.
    return hashlib.sha256(path.read_bytes()).hexdigest()


# What the manifest is, in the manifest, so that a reader does not have to come
# here to find out. The second version came after two history rewrites left the
# first one recording commits that no longer exist (R-11).
class ManifestProblem(Exception):
    """Raised when the manifest is there but cannot be used."""
    # A missing manifest is an ordinary state: nothing has been approved yet. A
    # damaged one, or one written by a version this code does not know, is not,
    # and treating the two alike would let an approval overwrite a record it
    # could not read. Whoever asked has to decide what to do: replay treats it as
    # nothing being approved, and the approval command stops without writing.


MANIFEST_VERSION = 2

WHAT_APPROVAL_MEANS = (
    "Approval means somebody accepted this exact file for replay during a "
    "demonstration. It says the file is the one that was accepted and has not "
    "changed since. It does not say a service produced what is in it, and it "
    "does not by itself say the contents may be published.")

WHAT_A_HEAD_AT_REVIEW_MEANS = (
    "The commit the repository stood at when the file was reviewed. It is not "
    "the commit that carries this record, because a manifest cannot hold the "
    "identifier of the commit that adds it. The commit that carries the record "
    "is the one that changed this file in the history.")

# Every rewrite of the history, and what it did to the identifiers recorded
# before it. Kept here because an entry naming a commit nobody can find reads
# as a mistake unless something says why it cannot be found.
HISTORY_REWRITES = [
    {
        "decision": "R-10",
        "completed_at": "2026-09-02T09:55:55+02:00",
        "what_it_did": ("Removed email addresses and the structured contact "
                        "elements around them from replies, in the current "
                        "files and in the history. Author names and "
                        "affiliations were deliberately left in place."),
        "effect_on_this_manifest": (
            "Identifiers recorded before this date name commits from the "
            "earlier history and cannot be reached from the current one."),
    },
    {
        "decision": "R-11",
        "completed_at": "2026-09-02T12:00:00+02:00",
        "what_it_did": ("Removed thirty reply paths from the history, being "
                        "the twenty withdrawn from the published set and ten "
                        "renamed or deleted earlier."),
        "effect_on_this_manifest": (
            "Identifiers changed again. Any recorded before this date names a "
            "commit that no longer exists."),
    },
]


def _manifest_header():
    """The part of the manifest that is the same whoever wrote it."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "what_approval_means": WHAT_APPROVAL_MEANS,
        "what_repository_head_at_review_means": WHAT_A_HEAD_AT_REVIEW_MEANS,
        "history_rewrites": HISTORY_REWRITES,
    }


def _as_version_two(entry):
    """Turns an entry written by the first version into the second's shape."""
    # The first version held one approval. The second holds all of them, because
    # a file approved, then changed by a redaction, then accepted again has three
    # facts about it and the first version could only keep the last. The approval
    # already recorded is kept as it was written and labelled with the history it
    # belongs to, rather than being restated as though it had happened now.
    if "approval_history" in entry:
        return entry

    # A file nobody has approved yet has nothing to carry forward. Building an
    # approval out of an empty entry would write one saying it happened at an
    # unknown commit on no date, which is not a weaker record than none: it is
    # a false one, and it would sit at the front of the history of every reply
    # approved from now on.
    if not entry:
        return {"sha256": None, "approval_history": []}

    where = entry.get("approval_commit") or {}
    return {
        "sha256": entry.get("sha256"),
        "approval_history": [{
            "approved_at": entry.get("approved_at"),
            "approved_sha256": entry.get("sha256"),
            "repository_head_at_review": where.get("commit", "unknown"),
            "working_tree_at_review": where.get("working_tree", "unknown"),
            "history": "before R-10",
            "note": ("Recorded before the history was rewritten, so the commit "
                     "named here cannot be reached from the current history."),
        }],
    }


def latest_approval(entry):
    """The most recent time somebody accepted this file, or None."""
    history = _as_version_two(entry).get("approval_history") or []
    return history[-1] if history else None


def read_manifest(manifest_path=None):
    """Returns what the manifest says, or an empty record if there is none."""
    # An older manifest is read into the current shape rather than refused, so
    # that a checkout from before the second version still reports what it
    # recorded. Reading it does not rewrite it: that happens when something is
    # approved and the manifest is written again.
    #
    # The path can be given rather than taken from this file, because the tests
    # redirect these folders and the check on the real one has to be able to say
    # which folder it means.
    path = manifest_path or MANIFEST_PATH

    if not path.exists():
        return {**_manifest_header(), "files": {}}

    # json.loads raises RecursionError when a file is nested more deeply than
    # Python can follow, and reading the file raises UnicodeDecodeError when its
    # bytes are not text. A manifest that fails either way is damaged, so it is
    # reported rather than treated as absent.
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            RecursionError) as problem:
        # A manifest that is there and cannot be read is not the same as no
        # manifest at all. Reading the two alike would let an approval
        # overwrite a damaged one and take every earlier approval with it.
        raise ManifestProblem(f"the manifest at {path} could not be read "
                              f"({problem}). It is there and damaged rather "
                              f"than absent, so nothing has been changed") from problem

    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ManifestProblem(f"the manifest at {path} is not in a shape this "
                              f"application recognises")

    version = manifest.get("manifest_version")

    # A manifest saying which version it is is taken as it stands. Filling in a
    # header it does not have would mean a manifest with its version, its
    # meanings or its record of the rewrites deleted would read here as though
    # it still had them, and every check on those would pass on a default this
    # file supplied rather than on what is written down.
    if version == MANIFEST_VERSION:
        return manifest

    # Only a manifest saying it is the first version, or saying nothing, is
    # converted. A later version was written by something that knew more than
    # this code does, and quietly reading it as the oldest shape would discard
    # whatever it knew. Stopping says so instead.
    if version is not None and version != 1:
        raise ManifestProblem(
            f"the manifest at {path} says it is version {version!r}, and this "
            f"application knows version 1 and version {MANIFEST_VERSION}. It "
            f"has not been read or changed")

    files = {}
    for name, entry in manifest["files"].items():
        files[name] = _as_version_two(entry)
    return {**_manifest_header(), **manifest, "files": files}


def what_should_not_be_published(record):
    """Returns the words in NEVER_PUBLISH that this record carries."""
    # Only the parts describing the request and the reply are looked at, for the
    # reason given beside DESCRIBING_FIELDS.
    described = {}
    for name in DESCRIBING_FIELDS:
        if name in record:
            described[name] = record[name]

    text = json.dumps(described).lower()
    found = []
    for word in NEVER_PUBLISH:
        if word in text:
            found.append(word)
    return found


def differences_from_manifest(folder=None, manifest_path=None, files_only=False):
    """Returns everything about the approved folder the manifest does not agree
    with, as a list of sentences.
    """
    # Four things are looked for, and each one is a way the demonstration set
    # could stop being what it says it is: a file nobody approved, a file that
    # was approved and is now gone, a file whose contents have changed since it
    # was approved, and one nobody decided could be published. An empty list
    # means the folder and the manifest agree.
    #
    # files_only leaves out the last of those. It is for the command that records
    # a fresh review of what is already published: that command exists to supply
    # the missing decisions, so refusing to run until they exist would leave no
    # way to ever supply them. It still refuses to run if the folder and the
    # manifest disagree about which files are there.
    #
    # What this establishes, exactly: a replayed file is listed under a recorded
    # fingerprint and has not changed since that entry was written. It does not
    # establish that a service produced the reply, that anybody read it closely,
    # or which code captured it. Somebody able to edit the reply can edit the
    # manifest as well. It is protection against a file arriving, changing or
    # disappearing unnoticed, not a proof of where the contents came from.
    folder = folder or APPROVED_DIR
    try:
        approved = read_manifest(manifest_path)["files"]
    except ManifestProblem as problem:
        # Reported as a difference rather than raised. This is the function
        # somebody runs to find out whether the folder is in order, and being
        # unable to read the manifest is the most serious answer it can give,
        # not a reason to stop with a traceback instead of an answer.
        return [str(problem)]

    problems = []

    on_disk = {}
    for path in files_in(folder):
        on_disk[path.name] = path

    for name in sorted(on_disk):
        if name not in approved:
            problems.append(f"{name} is in the folder but was never approved")
            continue

        entry = approved[name]
        here = fingerprint(on_disk[name])
        if here != entry.get("sha256"):
            problems.append(f"{name} has changed since it was approved")
            continue

        # The fingerprint at the top of an entry is a convenience: it says what
        # the file should be now. What makes it approved is the last approval
        # event, and that carries its own fingerprint. Checking only the first
        # would let a changed file pass as soon as somebody updated the one
        # line that is easy to find, leaving the record of what was actually
        # accepted saying something different a few lines below.
        latest = latest_approval(entry)
        if entry.get("needs_approving_again"):
            # Reported on its own, because it already says why the file and its
            # last approval no longer agree. Adding the general complaint under
            # it would report one fact twice and hide which came first.
            problems.append(f"{name} was {entry['needs_approving_again']}")
        elif latest is None:
            problems.append(f"{name} has no approval recorded against it")
        elif latest.get("approved_sha256") != here:
            problems.append(f"{name} is not the file its latest approval accepted")
        elif not files_only:
            # The replay question, not the publication one. This function
            # reports whether the set in this repository is what its manifest
            # says it is. Whether any of it may go into a public repository is
            # a different question, asked by why_it_cannot_be_exported at the
            # boundary, and answering it here would report every reply kept for
            # private replay as a fault.
            for reason in why_it_cannot_be_replayed(entry, on_disk[name]):
                problems.append(f"{name}: {reason}")

    for name in sorted(approved):
        if name not in on_disk:
            problems.append(f"{name} was approved but is no longer there")

    return problems


# What a reply kept for replaying here but never published is recorded as. It
# is not a publication basis and must never be read as one: it says the
# opposite, that nobody has established a right to publish this and nobody
# intends to. Keeping the two apart in the record is the whole point, because
# an entry that merely said nothing would look the same as one written before
# anybody thought to ask (R-11).
#
# Replaying is not publishing. These files sit where the work is done and are
# replayed to run the system without a network. What must not happen is one of
# them reaching a public repository. research_agent/release.py refuses any
# export carrying this classification, so the marking is now read at the
# boundary rather than being only a record of a decision. The release candidate
# stays open until an export has actually been built and checked from a clean
# commit (R-12).
KEPT_PRIVATE = "private sandbox only"


def why_it_cannot_be_replayed(entry, path, record=None, here=None):
    """Returns the reasons this file must not be replayed, if there are any."""
    # Replaying is not publishing, and the two questions are asked separately.
    # A reply kept for replaying here needs no right to publish, because nothing
    # is being published: the file is read where the work is done, to run
    # the system without a network. What it does need is for the record to say
    # that is what it is, and for that to have been said about these exact bytes.
    #
    # Written as its own function after the first version answered both questions
    # at once. A reply marked as not publishable was passed to a check called
    # "does the publication decision hold", which returned nothing to complain
    # about, so a refusal to publish was being read internally as permission to.
    # The record was right and the code reading it was not.
    #
    # There are three kinds of record it can be given, not two. A reply kept for
    # private replay, a reply somebody decided could be published, and a fixture
    # this project wrote for the offline demonstration. The third was added at
    # the export boundary and not here (R-16).
    latest = latest_approval(entry)
    if latest is None:
        return ["nobody has approved it"]

    scope = latest.get("replay_scope")
    if isinstance(scope, dict):
        if scope.get("scope") != KEPT_PRIVATE:
            return [(f"it is recorded with a replay scope nothing "
                     f"understands: {scope.get('scope')!r}")]
        if scope.get("applies_to_sha256") != (here or fingerprint(path)):
            return [("it was set aside for private replay as a different "
                     "version of the file")]
        return []

    # The export boundary asks a fixture where it came from and this side did
    # not, so a fixture written for the offline demonstration passed the check
    # on publishing it and was refused by the check on replaying it, with a
    # reason about publication that had nothing to do with what was wrong.
    # Nothing had noticed because no fixture existed yet: the rule was written
    # for files that were not there (R-16).
    #
    # What replay needs to know is what export needs to know, so the same
    # function answers both: that somebody recorded where this file came from,
    # and that they recorded it about these bytes.
    #
    # The fingerprint and the parsed record are handed on rather than left to
    # be worked out again. A caller that has already read the file is the
    # reason those two are parameters at all, and reading it a second time here
    # would put back the gap between what was checked and what is returned.
    if "fixture_origin" in latest:
        return why_the_fixture_origin_does_not_hold(latest["fixture_origin"],
                                                    path, entry, record, here)

    return why_the_publication_decision_does_not_hold(entry, path, record, here)


def why_it_cannot_be_exported(entry, path, record=None):
    """Returns the reasons this file must not go into a public repository."""
    # The other half of the pair. A reply kept for private replay fails here by
    # definition: being allowed to replay it is exactly not being allowed to
    # publish it, and this is the function that has to say so.
    #
    # It is asked at the boundary, by research_agent/release.py, about every file
    # in an export's replay set. It was written before anything called it, which
    # is a rule and no check, and a test in tests/unit/test_release.py makes this
    # refuse everything to prove the caller is really asking.
    latest = latest_approval(entry)
    if latest is None:
        return ["nobody has approved it"]

    scope = latest.get("replay_scope")
    if isinstance(scope, dict):
        return [(f"it is kept for private replay only: "
                 f"{scope.get('reason', 'no reason recorded')}")]

    problems = why_the_approval_says_too_little(latest)

    # A reply recorded as written for this project may be considered for export
    # under the project's own release policy. The origin record is evidence of
    # that decision; it is not proof that nobody else's words are in the file,
    # and it is not a legal clearance. Nothing can read a file and tell whether
    # somebody invented it, which is exactly why a person records it and why
    # what is recorded is a decision rather than a finding.
    if "fixture_origin" in latest:
        return problems + why_the_fixture_origin_does_not_hold(
            latest["fixture_origin"], path, entry, record)

    return problems + why_the_publication_decision_does_not_hold(entry, path,
                                                                 record)


def why_the_approval_says_too_little(latest):
    """Returns the reasons this approval is not a record somebody could check."""
    # Asked at the export boundary only, and about the approval itself rather
    # than about the file. Three fingerprints agreeing say the file has not
    # changed since it was accepted. They do not say when it was accepted, what
    # the person was reading when they accepted it, or whether anything else was
    # uncommitted at the time, and those are what let somebody go back and look.
    #
    # An entry carried over from the first version of the manifest records the
    # head as "unknown", which is true: the first version did not keep it. That
    # is an honest record of a private replay and not one to publish on, and this
    # is the check that says so at the boundary rather than in a document.
    problems = []

    when = str(latest.get("approved_at", "") or "").strip()
    if not when:
        problems.append("its approval does not say when it was given")
    else:
        try:
            datetime.datetime.fromisoformat(when)
        except ValueError:
            problems.append(f"its approval gives {when!r} as when it was "
                            f"given, which is not a time")

    if not looks_like_a_commit(latest.get("repository_head_at_review")):
        problems.append("its approval does not name where the repository stood "
                        "as the full identifier of a commit")

    if not str(latest.get("working_tree_at_review", "") or "").strip():
        problems.append("its approval does not record what was uncommitted "
                        "when it was given")

    return problems


def why_the_fixture_origin_does_not_hold(origin, path, entry=None, record=None,
                                         here=None):
    """Returns the reasons this is not a record of where a fixture came from."""
    # A fixture stands in for a service reply so a demonstration can run with no
    # network and publish only text this project wrote. That only holds if
    # somebody says so: the file itself looks exactly like a recording, which is
    # the point of it.
    #
    # So the record has to say who wrote it, when, what it stands in for, that it
    # was not copied from a paper, and which version of the file it is about. An
    # origin missing any of those describes something, but not this file.
    #
    # The file itself is read as well. A fixture that will not parse, or that is
    # not a shape replay can use, is not a fixture: it is a file in the folder
    # replay reads from, and the demonstration it is meant to make possible would
    # fail on it. Recording where something came from says nothing about whether
    # it is what it claims to be.
    #
    # The record and the fingerprint can be handed in by a caller that has
    # already read the file, the way the publication check takes them. Replay
    # reads the bytes once and decides everything about those bytes, so a second
    # read here would reopen the gap between what was checked and what is
    # returned. A caller that has not read it leaves them out.
    if not isinstance(origin, dict):
        return ["nothing records where it came from"]

    problems = []

    if here is None:
        here = fingerprint(path)
    if record is None:
        record = _read(path)
    if record is None:
        problems.append("it cannot be read, so it is not a reply anything "
                        "could replay")
    elif rebuilt_for_publication(record) is None:
        problems.append("it does not hold what replaying it would need")

    # All three, not one. The fingerprint at the top of the entry, the one the
    # approval recorded, and the one the origin is about are three statements
    # about the same file, and a file is only the file they all describe.
    # Checking the origin alone let the other two disagree with it.
    if isinstance(entry, dict):
        latest = latest_approval(entry) or {}
        if entry.get("sha256") != here:
            problems.append("the manifest records a different version of it")
        if latest.get("approved_sha256") != here:
            problems.append("its approval accepted a different version of it")
    for field, saying in (("written_by", "who wrote it"),
                          ("written_on", "when"),
                          ("what_it_stands_in_for", "what it stands in for")):
        if not str(origin.get(field, "")).strip():
            problems.append(f"its origin does not say {saying}")

    # A date that is a date. Checked only for being there, "when the fixtures
    # were made" satisfies the rule and tells nobody anything they could look
    # up, which makes the field look like a record while being a sentence.
    #
    # The whole value is read, not its first ten characters. Reading a prefix
    # accepted "2026-09-03 whatever anybody typed after it" as a date, which is
    # the same fault one step along: a field that looks checked and is not.
    # A plain calendar date is what this records, so a timestamp is refused
    # here too rather than half-read.
    written_on = str(origin.get("written_on", "")).strip()
    if written_on:
        try:
            datetime.date.fromisoformat(written_on)
        except ValueError:
            problems.append(f"its origin gives {written_on!r} as the date it "
                            f"was written, which is not a date")

    if origin.get("copied_from_a_paper") is not False:
        problems.append("its origin does not record that it was not copied "
                        "from a paper")

    if origin.get("applies_to_sha256") != here:
        problems.append("its recorded origin is about a different version of "
                        "the file")

    return problems


def why_the_publication_decision_does_not_hold(entry, path, record=None,
                                               here=None):
    """Returns the reasons the latest approval does not settle publication."""
    # An empty list means somebody decided this file could be published, the
    # decision was about these exact bytes, and reading the file now still agrees
    # with what they decided.
    #
    # Checked here rather than only when approving, because approving happened
    # once and replay happens every run. An approval recorded before this rule
    # existed says nothing about publication at all, and reading that silence as
    # permission is how the fifteen replies stayed replayable while the
    # publication question stood unasked (R-11).
    latest = latest_approval(entry)
    if latest is None:
        return ["nobody has approved it"]

    decision = latest.get("publication_basis")
    if not isinstance(decision, dict):
        return ["its latest approval does not say whether it may be published"]

    if decision.get("applies_to_sha256") != (here or fingerprint(path)):
        return [("the decision to publish it was taken about a different "
                 "version of the file")]

    # The record and its fingerprint can both be handed in by a caller that
    # has already read the file, so that replay reads it once rather than once
    # for each question asked of it. A caller that has not read it leaves them
    # out and they are worked out here.
    if record is None:
        record = _read(path)
    if record is None:
        return ["it cannot be read"]

    found = publication_rights.inspect(record)

    # Read again rather than trusted. The recorded finding and the file are
    # checked against each other, so an entry claiming a reply holds no article
    # text is worth no more than the reply agreeing with it.
    if decision.get("kind") == "no recognised article-text field":
        if found.finding != publication_rights.NO_ARTICLE_TEXT:
            return [(f"it was published as holding no article text, and "
                     f"reading it now finds {found.description}")]
        return []

    return publication_rights.problems_with_the_basis(
        decision, found, path.name, decision.get("content_was_changed", False))


def rebuilt_for_publication(record):
    """Returns a record holding only what replaying it needs, built field by
    field, or None if it is not a shape this application wrote.
    """
    # The record is not copied. It is built again from the same list of what to
    # keep that writing one uses, so a field nobody thought of cannot travel into
    # the repository by being present in the file. Copying the bytes and checking
    # them against a list of things to remove is the arrangement this module
    # exists to avoid, and approving a reply must not reintroduce it (R-07).
    #
    # The one field that list cannot govern is the paper text, so it is redacted
    # on the way through. This is the only route into the published folder, which
    # is why the removal belongs here and not in the command that calls it (R-10).
    if not isinstance(record, dict):
        return None

    if all(name in record for name in ("model", "question", "answer")):
        rebuilt = {
            "captured_at": record.get("captured_at"),
            "model": record["model"],
            "question": record["question"],
            "answer": _contact_details_removed(record["answer"]),
        }
        # The run is carried through on purpose, and that widens the list
        # this rebuild was built around: a field written at capture now
        # reaches the published folder. This is the one route in, so a
        # run dropped here would be captured and then lost at approval, and
        # the check for a replay set mixing runs would find no run on
        # anything. Only a record that has one gets the field, so an older
        # capture approved again is written exactly as it was before.
        if "run" in record:
            rebuilt["run"] = record["run"]
        return rebuilt

    if all(name in record for name in ("method", "url", "status", "headers", "body")):
        return {
            "captured_at": record.get("captured_at"),
            "method": record["method"],
            "url": record["url"],
            "asked": _values_worth_keeping(record.get("asked")),
            "status": record["status"],
            "headers": _headers_worth_keeping(record.get("headers") or {}),
            "body": _contact_details_removed(record["body"]),
        }

    return None


def looks_like_a_commit(text):
    """True when this is the full form of a commit identifier."""
    # Forty hexadecimal characters. A short form is refused because it stops
    # being unique as a repository grows, and a word like HEAD is refused because
    # it names wherever the repository happens to be standing rather than what
    # somebody reviewed.
    #
    # Checked here rather than only in the command that calls it. The command was
    # the only thing looking, so anything calling approve_reply directly could
    # record a single letter as the commit that was reviewed.
    return (isinstance(text, str) and len(text) == 40
            and all(c in "0123456789abcdef" for c in text))


def publication_changed_the_content(record, rebuilt):
    """True when publishing altered the words of the reply itself."""
    # Only the fields holding what came back are compared. Dropping a header or a
    # parameter the replay does not need is not a change to the work; taking a
    # contact address out of an abstract is. The difference matters because a
    # licence forbidding derivatives does not permit a changed version to be
    # distributed, so whether this happened decides whether such a licence can be
    # a basis at all (R-11).
    for field in ("body", "answer"):
        if field in record and record.get(field) != rebuilt.get(field):
            return True
    return False


def why_it_cannot_be_approved(path, publication_basis=None, record=None,
                              entry=None):
    """Returns the reasons this file must not be published, if there are any."""
    # Kept here rather than in the approval command so that the tests can use the
    # same checks the command uses. A test approving a reply by copying it would
    # be testing a path the application does not have.
    #
    # Three separate questions are asked, and a reply has to pass all of them:
    # whether it holds what replaying it needs, whether it carries anything about
    # the caller, and whether whatever article text it carries may be published
    # at all. The third is the one added last, and it is asked of the rebuilt
    # record rather than the file, because the rebuilt record is what would
    # actually be published (R-11).
    #
    # The record can be passed in rather than read here. Approving reads the file
    # once and checks and writes that same record: reading it again would leave
    # room for the file to change in between, so the reply inspected would not
    # have to be the reply published.
    if record is None:
        record = _read(path)
    if record is None:
        return ["it cannot be read"]

    rebuilt = rebuilt_for_publication(record)
    if rebuilt is None:
        return ["it does not hold what replaying it would need"]

    reasons = []
    if not rebuilt.get("captured_at"):
        reasons.append("it does not say when it was captured")

    for word in what_should_not_be_published(rebuilt):
        reasons.append(f"it contains '{word}', which identifies a caller or a session")

    for name in configured_keys_found_in(rebuilt):
        reasons.append(f"it contains the value of {name}, the key this computer "
                       "sends, under a name the word list does not know")

    # Whether publishing changed this reply is taken from what is already
    # recorded as well as from the file, for the reason given beside
    # _whether_publishing_changed_it. Asked only of the file, a reply approved
    # again from the published copy looks unchanged, and a licence that forbids
    # sharing a changed version would be accepted for the very reply it exists
    # to refuse.
    reasons.extend(publication_rights.problems_with_the_basis(
        publication_basis,
        publication_rights.inspect(rebuilt),
        path.name,
        _whether_publishing_changed_it(record, rebuilt, entry or {})))

    return reasons


# Where a reply waits between being approved and being published. It is a
# folder inside the published one, and files_in lists only a folder's own
# files rather than descending, so nothing staged is read as published or
# reported as unapproved while it waits. Until the manifest recording them is
# written they are not approved replies, only candidates.
STAGING_DIR_NAME = ".waiting_to_be_published"

# The key under which an in-memory manifest carries the replies staged against
# it. It never reaches the file: write_manifest takes it out and acts on it,
# which is what makes the two writes one act rather than two.
STAGED = "waiting_to_be_published"

# Publishing steps that did not finish, and could not be undone. Kept the same
# way recording_failures is: a caller that ignores it is how a problem stays
# invisible, so the command prints whatever is in here.
promotion_failures = []


def approve_reply(path, manifest, approval_commit=None, publication_basis=None):
    """Accepts one captured reply for publication, and records why it may be."""
    # Returns the reasons it was refused, or an empty list if it was accepted.
    # The record staged is the rebuilt one, not the file as it stands.
    #
    # Nothing enters the published folder here. The reply is written where
    # nothing reads from and its entry is added to the manifest held in memory;
    # writing the manifest is what publishes both. Written the obvious way round,
    # the reply was copied into the published folder first and the manifest
    # written after, so a manifest that failed to write left a file published
    # that no record mentioned, while the command that wrote it reported that
    # nothing counted as approved.
    #
    # What this gives is a promotion that fails closed: a step that does not
    # finish leaves nothing published that has no record. It is not one
    # indivisible change. Two things are written, and between them there is a
    # moment when the manifest names a reply that has not moved yet. That moment
    # is recoverable and it is visible, because the checks report a record naming
    # a file that is not there; it is not invisible, which is what the old order
    # was. Calling it atomic would claim more than it does.
    #
    # What is recorded with it is where the repository stood when it was
    # approved, which is not the same as the code that captured it. Capture does
    # not record a version, and saying so matters: a manifest full of one commit
    # could otherwise be read as saying every reply was captured under it.
    # Read once. Everything below asks about this record and publishes this
    # record, so there is no moment where the file could change between being
    # inspected and being written.
    record = _read(path)
    if record is None:
        return ["it cannot be read"]

    entry = _as_version_two(manifest["files"].get(path.name, {}))

    reasons = why_it_cannot_be_approved(path, publication_basis, record, entry)
    if reasons:
        return reasons

    if not approval_commit or not looks_like_a_commit(approval_commit.get("commit")):
        return [("the commit being reviewed is not a full commit identifier, "
                 "and an approval that cannot say what was reviewed is not a "
                 "record")]

    staged = APPROVED_DIR / STAGING_DIR_NAME / path.name
    if not _write(staged, rebuilt_for_publication(record)):
        return ["it could not be written ready for publication"]

    written = fingerprint(staged)

    # Added to what is already there rather than replacing it. A file approved
    # in August, changed by a redaction in September and accepted again is
    # three facts, and writing only the last would say the file had always been
    # what it is now.
    entry["sha256"] = written
    entry.pop("needs_approving_again", None)
    entry.setdefault("approval_history", []).append({
        "approved_at": now(),
        "approved_sha256": written,
        "repository_head_at_review": approval_commit["commit"],
        "working_tree_at_review": approval_commit.get("working_tree", "unknown"),
        "history": "after R-11",
        # What was found in the reply and what allows it to be published. A
        # reply holding no article text says so, which is a finding rather than
        # a silence: an entry with nothing written here would not say whether
        # the question had been asked (R-11).
        "publication_basis": _the_decision_taken(record, publication_basis,
                                                 written, entry),
    })
    manifest["files"][path.name] = entry
    manifest.setdefault(STAGED, {})[path.name] = staged
    return []


def _the_decision_taken(record, publication_basis, written, entry):
    """What was decided about publishing this reply, written down in full."""
    # Enough has to be recorded for the decision to be checked again later
    # without the captured file, which by then may be gone. That means the
    # fingerprint it was taken about, and whether publishing changed the reply,
    # because a licence forbidding changed versions turns on exactly that and
    # nothing in the published file alone can say.
    #
    # A reply holding nothing needing a basis records that finding rather than
    # leaving the field out. An entry silent on the question does not say the
    # question was asked, and fifteen entries silent on it are indistinguishable
    # from fifteen approved before anybody thought to ask.
    rebuilt = rebuilt_for_publication(record)
    found = publication_rights.inspect(rebuilt)

    # A reply holding nothing that needs a basis records that, whatever was
    # handed in. Writing a licence against a list of titles would put a claim
    # about one paper beside a reply that has nothing to do with it, and one
    # basis offered for a batch would land on every metadata reply in it.
    if found.finding == publication_rights.NO_ARTICLE_TEXT:
        return {"kind": "no recognised article-text field",
                "applies_to_sha256": written,
                "found": found.description}

    return {**publication_basis,
            "applies_to_sha256": written,
            "content_was_changed": _whether_publishing_changed_it(
                record, rebuilt, entry)}


def _whether_publishing_changed_it(record, rebuilt, entry):
    """Whether publishing this reply altered the words in it."""
    # Once true, always true. Approving again reads the file that was published
    # last time, and that file has already had its contact details taken out, so
    # comparing it with its own rebuilt form finds nothing and would answer no.
    # The answer would flip from yes to no on a second approval, and a licence
    # forbidding a changed version would then stop refusing exactly the reply it
    # should refuse.
    #
    # So an earlier decision that recorded a change is carried forward rather
    # than worked out again. It is a fact about what was published, and reading
    # the published file is not a way to find it out.
    if publication_changed_the_content(record, rebuilt):
        return True

    for approval in reversed(_as_version_two(entry).get("approval_history") or []):
        decision = approval.get("publication_basis")
        if isinstance(decision, dict) and "content_was_changed" in decision:
            return bool(decision["content_was_changed"])

    return False


def keep_private(manifest, folder=None, decided_at=None):
    """Records that the replies already published here are for replaying only."""
    # Returns the names recorded. Nothing is written to the folder: the files do
    # not change, only what the manifest says about them.
    #
    # These are real recordings from real services, kept so a demonstration can
    # run without a network. Establishing a right to publish each one would mean
    # reading every field of every response and forming a judgement about it,
    # including fields nothing here inspects. That work buys nothing, because the
    # public repository is to carry project-owned synthetic replies instead. So
    # the record says what they are rather than pretending a basis exists.
    #
    # What this does **not** do is stop anybody copying the folder. The release
    # check that will refuse to export these is not built; `why_it_cannot_be_exported`
    # is the rule it will use, and the release candidate stays open until
    # something calls it at the boundary. Saying otherwise here would describe
    # a guard as though writing it down had built it (R-12).
    folder = folder or APPROVED_DIR
    where = decided_at or {"commit": "unknown", "working_tree": "unknown"}
    recorded = []

    for path in files_in(folder):
        entry = _as_version_two(manifest["files"].get(path.name, {}))
        here = fingerprint(path)
        entry["sha256"] = here
        entry.pop("needs_approving_again", None)
        entry.setdefault("approval_history", []).append({
            "approved_at": now(),
            "approved_sha256": here,
            "repository_head_at_review": where.get("commit", "unknown"),
            "working_tree_at_review": where.get("working_tree", "unknown"),
            "history": "after R-11",
            # Recorded under its own name, not as a publication basis. It is
            # the opposite of one: a restriction on use, saying this may be
            # replayed here and nothing more. Writing it where a basis goes
            # made a refusal look like permission to anything reading the
            # record by field name.
            "replay_scope": {
                "scope": KEPT_PRIVATE,
                "applies_to_sha256": here,
                "publication_basis": None,
                "reason": ("No right to publish this has been established, and "
                           "none is claimed."),
            },
        })
        manifest["files"][path.name] = entry
        recorded.append(path.name)

    return recorded


def redact_approved(manifest, folder=None):
    """Takes contact details out of replies already published, and re-records
    their fingerprints. Returns the names of the files that changed.
    """
    # Redacting on the way in protects everything approved from now on. It does
    # nothing for the replies approved before the removal existed, and those are
    # the ones already public. This re-runs the same rebuilding step over the
    # published folder so that both sets are held to one standard.
    #
    # The fingerprint has to be written again with them. A redacted file no longer
    # matches the one the manifest recorded, and leaving that alone would make
    # every check report the folder as tampered with, which would bury a real
    # alteration among thirty-five false ones (R-10).
    folder = folder or APPROVED_DIR
    changed = []

    for path in files_in(folder):
        record = _read(path)
        rebuilt = rebuilt_for_publication(record)
        if rebuilt is None or rebuilt == record:
            continue
        if not _write(path, rebuilt):
            continue
        changed.append(path.name)
        entry = _as_version_two(manifest["files"].get(path.name, {}))
        entry["sha256"] = fingerprint(path)
        entry["redacted_at"] = now()
        # The approvals already recorded stay as they are, and none is added.
        # They were of the file as it was, and this is not that file any more.
        # Leaving the record saying otherwise would have the manifest assert
        # that somebody accepted something nobody has seen, which is the whole
        # of what approval is for. Until it is approved again the entry says
        # its latest approval is of an earlier version, and the checks say so.
        entry["needs_approving_again"] = (
            "changed by a redaction after it was last approved")
        manifest["files"][path.name] = entry

    return changed


def write_manifest(manifest, manifest_path=None):
    """Writes the manifest, and publishes whatever was approved against it."""
    # Returns True when the record and the files it describes are both in place,
    # and False when neither is: a failure part way through puts back what was
    # there before rather than leaving the two disagreeing.
    #
    # This is the moment a reply becomes published. Approving one only writes it
    # where nothing reads from, so a reply that never reaches here never reaches
    # the folder a demonstration reads.
    #
    # The order is deliberate. The record is written first, so that a failure
    # between the two leaves a record naming a file that is not there, which the
    # checks report out loud. The other way round leaves a published file that no
    # record names, which is the case the publication rights fault was about and
    # the one nothing would notice (R-11).
    #
    # If moving the files fails, what was there before is put back. That is a
    # best effort rather than a guarantee: putting it back can fail too, and a
    # run that is killed between the two writes leaves the halfway state on disk.
    # Both are reported rather than passed over, in promotion_failures, so that
    # the state is something somebody is told about rather than something they
    # find later.
    #
    # The part explaining what the manifest is comes from this file rather than
    # from whatever was read, so a manifest written by an older version comes
    # back with the current explanation rather than none.
    #
    # Written through a second file and moved into place, for the same reason a
    # reply is: a run interrupted midway would otherwise leave a manifest that
    # cannot be read, and every approved reply would then look unapproved.
    manifest_path = manifest_path or MANIFEST_PATH
    staged = manifest.pop(STAGED, {})

    was = manifest_path.read_bytes() if manifest_path.exists() else None
    ordered = {**_manifest_header(),
               "files": dict(sorted(manifest["files"].items()))}
    if not _write(manifest_path, ordered):
        return False

    published = _publish_the_staged_files(staged)
    if published is None:
        if not _put_back(manifest_path, was):
            promotion_failures.append(
                f"the manifest at {manifest_path} could not be put back after "
                f"publishing failed, so it may name replies that are not there")
        return False
    return True


def _publish_the_staged_files(staged):
    """Moves approved replies into the published folder, or moves none of them."""
    # Returns the names published, or None if any could not be and everything was
    # put back as it was. Each file replaced is kept until they have all moved,
    # because a reply being approved again over an earlier version has to be
    # recoverable if a later one in the same batch fails.
    replaced = {}
    moved = []
    try:
        for name, waiting in sorted(staged.items()):
            destination = APPROVED_DIR / name
            if destination.exists():
                replaced[name] = destination.read_bytes()
            waiting.replace(destination)
            moved.append(name)
        return moved
    except OSError as problem:
        promotion_failures.append(f"publishing stopped part way: {problem}")
        for name in moved:
            destination = APPROVED_DIR / name
            try:
                if name in replaced:
                    destination.write_bytes(replaced[name])
                else:
                    destination.unlink(missing_ok=True)
            except OSError as second:
                # Undoing failed as well. Nothing further can be done from
                # here, and the one thing that must not happen is silence: a
                # file left published that the record does not describe is the
                # state this whole arrangement exists to prevent.
                promotion_failures.append(
                    f"{name} could not be put back after publishing failed "
                    f"({second}), so it may be published without a record")
        return None


def _put_back(path, contents):
    """Restores a file to what it held before, or removes it if there was none."""
    # Returns whether it worked. Used only when publishing failed after the
    # manifest had been written: a manifest describing files that are not there
    # is worse than the approval simply not having happened, which is what this
    # turns it back into.
    #
    # A failure here is returned rather than raised, so that the caller reports
    # one refusal to publish instead of a crash on top of it. It is not passed
    # over: an earlier version returned nothing at all, which meant the one case
    # where the record and the folder are left disagreeing was also the one case
    # nobody was told about.
    try:
        if contents is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes(contents)
        return True
    except OSError:
        return False
