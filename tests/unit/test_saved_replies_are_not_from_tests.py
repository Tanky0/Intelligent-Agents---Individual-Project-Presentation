"""Looks for signs that an invented reply reached the demonstration folder."""
# The application saves each reply it receives so a run can be repeated with no
# network. Tests use invented replies, and one of those reaching this folder
# would be played back during a demonstration as though it came from a service. A
# made-up paper shown as a real result would be worse than the demonstration
# failing outright.
#
# tests/conftest.py is supposed to prevent it. This checks the outcome rather
# than the mechanism, because the mechanism has failed silently once already:
# renaming conftest.py removed the protection while every test still passed, and
# nine invented replies were written here before anyone noticed.
#
# Unlike the other tests, this one reads the real folder rather than a temporary
# one, so it is deliberately outside that protection.
#
# What this can and cannot show. The marker checks find the addresses and
# identifiers the tests in this project use. They cannot prove a reply is
# genuine: an invented one without those marks would pass, and one did. A file
# written by a script that made a pretend request went in unnoticed because its
# invented identifiers were not on the list.
#
# The manifest checks are the answer to that. They do not look at what is in a
# file at all. They ask whether every file in the folder was approved, whether
# everything approved is still there, and whether any of it has changed since.
# An invented reply fails those whatever it contains, because nobody approved it
# (R-07).
#
# Neither kind of check can prove a service produced what is in these files. What
# the manifest adds is that nothing arrives, leaves or changes without somebody
# having said so.

import datetime
import json
import pathlib

import pytest

from research_agent.sources import publication_rights, saved_replies

# Built here rather than read from the module, because tests/conftest.py points
# the module's folders at a temporary one for every test. Reading them would
# make this check look at an empty folder and pass without reading anything,
# which is the fault it exists to catch. parents[0] is tests/unit, parents[1]
# is tests, parents[2] is the project folder.
DATA = pathlib.Path(__file__).resolve().parents[2] / "data"
SAVED_REPLIES = DATA / "recorded_replies"
MANIFEST = DATA / "recorded_replies_manifest.json"

# Addresses and identifiers that only ever appear in invented replies. The
# services these stand in for do not use them.
INVENTED_MARKERS = (
    "example.test",       # the address used when a test needs any address
    "10.1000/",           # the DOI range reserved for examples
    "<PMID>12345678<",    # the paper identifier used in the test replies
    "<PMID>99999999<",
    '"111"',              # the identifiers a pretend search returns
)

# Things that should never appear because they identify a person or a session
# rather than describing a paper. Taken from the module that decides what may
# be written, so that one list is used here, when approving a reply, and when
# saving one, rather than three that can drift apart.
NEVER_SAVE = saved_replies.NEVER_PUBLISH + ("ncbi-phid",)

# The paper text itself is not searched for these. An abstract can legitimately
# contain an author's email address, and a title can contain almost anything.
# Only the parts describing the request and the reply are checked, which is
# where anything identifying the caller would end up.
#
# That reasoning is right about who these markers describe and wrong about what
# it implied, which was that the paper text needed no attention. It carries the
# contact details of people who are not the caller and never agreed to appear
# here. Those are dealt with separately, by the last test in this file, and by
# removal rather than refusal.
DESCRIBING_FIELDS = saved_replies.DESCRIBING_FIELDS

# The identifiers the tests use. A genuine reply never contains them, because
# no such paper exists.
INVENTED_IDENTIFIERS = ("9999990", "9999991", "9999992")


def saved_files():
    if not SAVED_REPLIES.exists():
        return []
    return sorted(SAVED_REPLIES.glob("*.json"))


def test_the_folder_being_checked_is_the_real_one():
    """Guards against this test quietly checking nothing."""
    # A wrong count of parent folders would point at a folder that does not
    # exist. Searching an empty folder finds no problems, so every check below
    # would pass while reading nothing at all.
    assert SAVED_REPLIES.exists(), f"{SAVED_REPLIES} does not exist"
    assert saved_files(), "no saved replies found, so nothing would be checked"


def test_no_saved_reply_came_from_a_test():
    """Looks for the addresses and identifiers only invented replies contain."""
    found = []

    for path in saved_files():
        text = path.read_text(encoding="utf-8")
        for marker in INVENTED_MARKERS:
            if marker in text:
                found.append(f"{path.name} contains {marker}")

    assert not found, (
        "Invented replies have reached the folder used for demonstrations. "
        "Delete them, and check that tests/conftest.py is still in place: "
        + "; ".join(found)
    )


def test_no_saved_reply_uses_an_identifier_invented_for_testing():
    """The tests refer to papers that do not exist. A genuine reply cannot
    mention them.
    """
    # An earlier version of this test judged by length instead, on the assumption
    # that a real reply is always long. That was wrong: a recorded link reply
    # asking which PubMed Central record matched one paper (made while that
    # source existed) has a complete answer of under two hundred characters,
    # and real replies were rejected as invented. How long a reply is says
    # nothing about where it came from.
    found = []

    for path in saved_files():
        text = path.read_text(encoding="utf-8")
        for identifier in INVENTED_IDENTIFIERS:
            if identifier in text:
                found.append(f"{path.name} mentions {identifier}")

    assert not found, (
        "Replies mentioning a paper invented for testing: " + "; ".join(found)
    )


@pytest.mark.parametrize("forbidden", NEVER_SAVE)
def test_nothing_identifying_a_person_or_session_was_saved(forbidden):
    """A key, a contact address or a session identifier in a public repository
    is the failure this whole arrangement exists to prevent."""
    found = []

    for path in saved_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        described = {name: record[name] for name in DESCRIBING_FIELDS
                     if name in record}
        if forbidden in json.dumps(described).lower():
            found.append(path.name)

    assert not found, (
        f"'{forbidden}' appears in: {', '.join(found)}. The list of fields kept "
        "in saved_replies.py has been widened or bypassed."
    )


def test_every_replayed_model_reply_comes_from_one_run():
    """A demonstration that replays answers from two runs would show a run that
    never happened.
    """
    # The file name cannot show it, because a later run overwrites the earlier
    # answer to the same question, and the capture time cannot either, because
    # a time does not name an execution. So each answer carries the run that
    # produced it and this reads them all (ID-20).
    #
    # Three outcomes, kept apart because they need different words. No model
    # reply in the folder is nothing to check, and is reported as a skip rather
    # than passed silently. A set in which no reply has a run had no run
    # recorded, whether the replies were captured before the field existed or
    # outside a run, and the record cannot say which; that is said in those
    # words, again as a skip, because a skip is the one result pytest reports
    # that is neither a pass nor a failure. A reply with no run beside one that
    # has a run, or two different runs, is the fault, and fails.
    runs = saved_replies.run_of_each_model_reply(SAVED_REPLIES)
    if not runs:
        pytest.skip("nothing to check: the approved folder holds no model reply")

    without_a_run = []
    found = set()
    for name, run in runs.items():
        if run is None:
            without_a_run.append(name)
        else:
            found.add(run)

    if not found:
        pytest.skip(f"{len(without_a_run)} approved model replies have no run: "
                    "none was recorded, whether they were captured before the "
                    "field existed or outside a run")

    assert not without_a_run, (
        "these replies have no run, beside replies that do: "
        + ", ".join(without_a_run))
    assert len(found) == 1, "the replay set mixes runs: " + ", ".join(sorted(found))


def test_every_saved_reply_says_when_it_was_captured():
    """A saved reply can be months old and still match. The date makes the age
    of the evidence visible rather than leaving it to be assumed."""
    undated = []

    for path in saved_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("captured_at"):
            undated.append(path.name)

    assert not undated, f"no capture date in: {', '.join(undated)}"


def test_every_approved_reply_matches_the_manifest():
    """Nothing arrives, leaves or changes without somebody having said so."""
    # This is the check that does not depend on recognising what is in a file.
    # A reply nobody approved fails it whatever it contains, which is what the
    # marker checks above could not do (R-07).
    problems = saved_replies.differences_from_manifest(SAVED_REPLIES, MANIFEST)

    assert not problems, (
        "the approved replies and the manifest disagree. Run "
        "'python scripts/approve_replies.py' to see the same list, and approve "
        "anything that should be there: " + "; ".join(problems)
    )


def test_the_manifest_exists_and_is_not_empty():
    """Guards against the check above quietly passing on nothing."""
    # An empty manifest and an empty folder agree with each other perfectly.
    approved = saved_replies.read_manifest(MANIFEST)["files"]

    assert approved, "the manifest lists no approved replies, so nothing is checked"
    assert len(approved) == len(saved_files())


def test_every_approved_reply_records_when_it_was_approved_and_its_fingerprint():
    """An approval with no date or no fingerprint is not much of a record."""
    # Who approved it is deliberately not recorded. It would be one more piece
    # of personal detail in a public repository, and it is not needed: the
    # repository has one author.
    incomplete = []

    for name, details in saved_replies.read_manifest(MANIFEST)["files"].items():
        latest = saved_replies.latest_approval(details)
        if not details.get("sha256") or not latest or not latest.get("approved_at"):
            incomplete.append(name)

    assert not incomplete, f"approved without a date or a fingerprint: {incomplete}"


def test_no_published_reply_carries_a_contact_detail():
    """No email address reaches the repository inside a paper."""
    # This reads the published folder rather than testing the removal on invented
    # text, because the removal working is not the claim that matters. The claim
    # is that nothing in this folder carries one, and only the folder can settle
    # that. Thirty-five replies were approved before the removal existed and seven
    # of them held a hundred and three occurrences between them, standing for
    # eighty-nine distinct addresses once case is ignored.
    #
    # Author names and affiliations are left in place. They are the bibliographic
    # record of a published paper and are public by intent. So the claim here is
    # about email addresses and the contact fields around them, not about contact
    # details in general: an address is the part that is collectable and unwanted,
    # and it is what a search never needed.
    carrying = []

    for path in saved_files():
        record = json.loads(path.read_text(encoding="utf-8"))
        for field in ("body", "answer"):
            if saved_replies.holds_contact_details(record.get(field)):
                carrying.append(f"{path.name} ({field})")

    assert not carrying, (
        "published replies still carry contact details. Run "
        "'python scripts/approve_replies.py redact' to take them out and record "
        "the fingerprints again: " + "; ".join(carrying)
    )


# The check itself lives in research_agent/sources/publication_rights.py, and
# is used here rather than copied. It began as a copy in this file, which meant
# the rule that decided what may be published lived only in a test: it could
# say afterwards that something had got in, and could not stop it. The approval
# command now asks the same question before anything is published, and a test
# holding its own version of the rule would be checking something other than
# what the application does (R-11).
_article_text_in = publication_rights.article_text_in

def test_no_published_reply_carries_the_text_of_a_paper():
    """Nothing published here contains an abstract or the body of an article."""
    # Being able to fetch something through a public interface says that access
    # was allowed, not that republishing it is. Crossref says most of its
    # metadata is free to reuse while abstracts may still belong to a publisher
    # or an author, and NLM says it claims no copyright in PubMed abstracts but
    # that others may. Neither grants permission to put them in a repository.
    #
    # Seventeen replies were withdrawn because they carried paper text without an
    # established publication basis. One further reply was withdrawn because it
    # had no usable replay content or established publication basis. Two saved
    # answers from the language model went with them for a different reason: a
    # prompt is prose, so nothing can read one and say whose words are in it, and
    # they wait for a recorded basis like everything else. Twenty in all (R-11).
    #
    # The rule left behind is deliberately one anybody can check rather than one
    # they have to trust: the published set holds identifiers, publication status
    # and search results, and no article text at all. A reply carrying article
    # text now has to earn its place through a recorded basis for publishing it,
    # which approval refuses to proceed without.
    #
    # This check stays even though approval enforces it. Approval guards the way
    # in, and this says what is actually in the folder however it arrived, which
    # is the difference that mattered when files were put there by hand.
    #
    # A fixture this project wrote for the offline demonstration is the third
    # kind of record, beside a reply kept private and a reply with a publication
    # basis. It holds an abstract and prompts like a recording does, because it
    # stands in for one, and no service or paper produced a word of it. What
    # says so is its recorded origin, the same record the export boundary reads,
    # and this check accepts it only when that origin holds for these bytes.
    # Until the five fixtures were copied into the folder on 17 September the
    # rule here knew two kinds of record and refused the third (R-16).
    unaccounted = _holding_article_text_nothing_accounts_for(
        saved_files(), manifest()["files"])

    assert not unaccounted, (
        "these replies hold, or may hold, the text of a paper, and nothing "
        "records that they are kept private or written by this project:\n  "
        + "\n  ".join(unaccounted))


def _holding_article_text_nothing_accounts_for(paths, listed):
    """The rule of the test above, over any files and any manifest entries, so
    the same rule can be asked of a made-up fixture as of the real folder."""
    unaccounted = []

    for path in paths:
        found = _article_text_in(json.loads(path.read_text(encoding="utf-8")))
        if not found:
            continue

        # A reply this check cannot read, or that holds article text, may stay
        # here only where the record says it is kept for replaying and will not
        # be published. Four of these are real search answers carrying the
        # query as the service understood it, which is free text nothing can
        # read for whose words are in it (R-12).
        entry = listed.get(path.name, {})
        latest = saved_replies.latest_approval(entry) or {}
        scope = latest.get("replay_scope") or {}
        if scope.get("scope") == saved_replies.KEPT_PRIVATE:
            continue

        # Or where the record says this project wrote it. The origin is asked
        # to hold, not merely to be there, because an origin about other bytes
        # or one that does not say the words are not a paper's would excuse a
        # file this check exists to find.
        origin = latest.get("fixture_origin")
        if origin is not None and not saved_replies.why_the_fixture_origin_does_not_hold(
                origin, path, entry):
            continue

        unaccounted.append(f"{path.name} is {found}")

    return unaccounted


def test_a_fixture_is_excused_only_by_an_origin_that_holds(tmp_path):
    """The origin has to say this project wrote it, about these bytes."""
    # One record that says so, one that says the text was copied from a paper,
    # and one about a different version of the file. Only the first excuses an
    # abstract. The other two are the ways a real recording could be dressed as
    # a fixture, by a field left at its default or by an origin pasted from
    # another entry.
    reply = _a_reply_carrying_an_abstract_and_an_address(tmp_path)
    here = saved_replies.fingerprint(reply)

    def an_entry(origin):
        return {reply.name: {"sha256": here, "approval_history": [{
            "approved_at": "2026-09-17T07:00:00+00:00",
            "approved_sha256": here,
            "repository_head_at_review": "a" * 40,
            "fixture_origin": origin}]}}

    written_here = {"written_by": "this project", "written_on": "2026-09-17",
                    "what_it_stands_in_for": "a fetch", "copied_from_a_paper": False,
                    "applies_to_sha256": here}
    copied = dict(written_here, copied_from_a_paper=True)
    other_bytes = dict(written_here, applies_to_sha256="0" * 64)

    assert _holding_article_text_nothing_accounts_for([reply], an_entry(written_here)) == []
    for origin in (copied, other_bytes):
        assert _holding_article_text_nothing_accounts_for([reply], an_entry(origin)) \
            == [f"{reply.name} is a Crossref record holding an abstract"]


# The detector the check above rests on, tried against each thing it has to
# tell apart. The real folder cannot do this: it holds only replies that pass,
# so reading it proves the answer for those files and nothing about the
# branches that would catch a different one.

@pytest.mark.parametrize(
    "record, expected",
    [
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"abstract": "<p>Findings.</p>"}}'},
         "a Crossref record holding an abstract"),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"title": ["A paper"]}}'},
         None),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"abstract": "   "}}'},
         None),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": "this is not json at all"},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '["a list where a record was expected"]'},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"abstract": 42}}'},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"abstract": ["Findings", "in pieces"]}}'},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://api.crossref.org/works/10.1/a",
          "body": '{"message": {"items": ["not a record"]}}'},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<Article><AbstractText>Findings.</AbstractText></Article>"},
         "an NCBI reply holding an abstract"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<article><body><p>Full text.</p></body></article>"},
         "an NCBI reply holding full text"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<PubmedArticle><ArticleTitle>A paper</ArticleTitle></PubmedArticle>"},
         ("a fetched article (<pubmedarticle>), whose shape is too large to "
          "write down, so finding no abstract in it is not "
          "evidence that there is none")),
        ({"url": "https://api.semanticscholar.org/graph/v1/paper/search",
          "body": '{"data": [{"abstract": "Findings."}]}'},
         "a Semantic Scholar record holding an abstract"),
        ({"url": "https://api.semanticscholar.org/graph/v1/paper/search",
          "body": '{"data": [{"title": "A paper", "abstract": null}]}'},
         None),
        ({"url": "https://api.semanticscholar.org/graph/v1/paper/search",
          "body": '{"nothing": "where data was expected"}'},
         "a Semantic Scholar reply that could not be inspected for article text"),
        ({"url": "https://api.crossref.org/works/10.1/a", "body": ""},
         "a Crossref reply that could not be inspected for article text"),
        ({"url": "https://api.semanticscholar.org/graph/v1/paper/search",
          "body": '{"data": [{"abstract": ["Findings", "in pieces"]}]}'},
         "a Semantic Scholar reply that could not be inspected for article text"),
        # An address this check was never written for is not an address known
        # to return no article text.
        ({"url": "https://api.example.test/papers", "body": '{"abstract": "Findings."}'},
         ("a reply from an address this check does not recognise: "
          "https://api.example.test/papers")),
        # XML read as XML, so a tag inside a comment is not the real tag, and
        # an abstract in a JATS article is found without needing a body.
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<article><!-- <AbstractText>hidden</AbstractText> --></article>"},
         ("a fetched article (<article>), whose shape is too large to "
          "write down, so finding no abstract in it is not "
          "evidence that there is none")),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<article><front><abstract><p>Findings.</p></abstract></front></article>"},
         "an NCBI reply holding an abstract"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<article><AbstractText></AbstractText></article>"},
         ("a fetched article (<article>), whose shape is too large to "
          "write down, so finding no abstract in it is not "
          "evidence that there is none")),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": "<article><unclosed>"},
         "an NCBI reply that could not be read as XML or JSON"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
          "body": '{"esearchresult": {"idlist": ["1", "2"]}}'},
         None),
        # A saved answer from the model is refused whatever it says. This one
        # plainly carries nobody else's words, and is refused anyway, because
        # reading it and deciding that is not a recorded basis.
        ({"model": "gemini", "question": "a question", "answer": "an answer",
          "captured_at": "2026-09-02T00:00:00+00:00"},
         ("a saved model reply, whose prompt cannot be read for whose words "
           "are in it")),
        # The same reply carrying the run that produced it is still that
        # shape: the run is the one field a model reply may or may not have
        # (ID-20). Any other extra field is still refused.
        ({"model": "gemini", "question": "a question", "answer": "an answer",
          "captured_at": "2026-09-10T00:00:00+00:00", "run": "0123456789ab"},
         ("a saved model reply, whose prompt cannot be read for whose words "
           "are in it")),
        ({"model": "gemini", "question": "a question", "answer": "an answer",
          "captured_at": "2026-09-10T00:00:00+00:00", "extra": "field"},
         "a saved reply in a shape this check does not recognise"),
        ({"something": "unrecognised"},
         "a saved reply in a shape this check does not recognise"),
        # A search address carrying an article body. The address is not
        # evidence of what came back.
        ({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
          "body": "<article><body><p>Full text.</p></body></article>"},
         "an NCBI reply holding full text"),
        # A search answer in JSON that carries an abstract is not a search
        # answer, whatever it was asked for.
        ({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
          "body": '{"result": {"1": {"abstract": "Findings."}}}'},
         "an NCBI reply holding an abstract"),
        # An article from PMC arrives with its elements in a namespace, so the
        # tag is not the bare name.
        ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
          "body": '<article xmlns="https://jats.nlm.nih.gov"><front>'
                  "<abstract><p>Findings.</p></abstract></front></article>"},
         "an NCBI reply holding an abstract"),
        # And one that plainly does carry somebody else's words, refused for
        # the same reason rather than for its length. A prompt shorter than
        # this can hold a whole abstract, which is why the length was dropped.
        ({"model": "gemini", "question": "Screen this: " + "a full abstract. " * 40,
          "answer": "included", "captured_at": "2026-09-02T00:00:00+00:00"},
         ("a saved model reply, whose prompt cannot be read for whose words "
           "are in it")),
        # A JSON field named as article text but holding a list is something
        # this check cannot read, not something it can pass.
        ({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
          "body": '{"abstract": ["Published paper", "text"]}'},
         "an NCBI reply that could not be inspected for article text"),
        ({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
          "body": '{"result": {"1": {"body": "Full text."}}}'},
         "an NCBI reply holding full text"),
    ])
def test_the_detector_tells_article_text_from_everything_else(record, expected):
    """Each answer the detector has to give, including the one for a reply it
    cannot read.
    """
    # That last one is the reason for this test. The first version of the
    # detector returned nothing when a body would not parse, which the check
    # above reads as "no article text found". A file nobody could inspect would
    # have passed the publication rule in silence, which is the shape of fault
    # this project keeps finding rather than a new one.
    assert _article_text_in(record) == expected


# What the manifest has to say about itself, and about each approval in it.
# These read the real manifest, like the checks above, because a manifest built
# for a test would only prove the test built it correctly.

def manifest():
    return saved_replies.read_manifest(MANIFEST)


def test_the_manifest_says_what_it_is_and_what_approval_means():
    """A record nobody can interpret is not a record."""
    # Somebody reading this repository has no reason to know what was meant by
    # approving a file, so the manifest says it rather than leaving it in a
    # remediation entry they would have to find.
    written = manifest()

    assert written["manifest_version"] == 2
    assert "has not changed since" in written["what_approval_means"]
    assert "does not by itself say the contents may be published" in \
        written["what_approval_means"]
    assert "cannot hold the identifier of the commit that adds it" in \
        written["what_repository_head_at_review_means"]


def test_the_manifest_records_the_rewrites_that_broke_its_own_references():
    """An entry naming a commit nobody can find reads as a mistake."""
    # Two rewrites left this manifest holding identifiers from histories that no
    # longer exist. Saying so is what turns an unreachable commit from a fault
    # into a fact with a date on it.
    rewrites = manifest()["history_rewrites"]

    assert [r["decision"] for r in rewrites] == ["R-10", "R-11"]
    for rewrite in rewrites:
        assert rewrite["completed_at"]
        assert rewrite["what_it_did"]
        assert rewrite["effect_on_this_manifest"]


def test_every_published_reply_appears_exactly_once_in_the_manifest():
    """Two entries for one file, or none, is a set nobody can check."""
    listed = manifest()["files"]
    on_disk = [path.name for path in saved_files()]

    assert sorted(listed) == sorted(on_disk)
    assert len(on_disk) == len(set(on_disk))


def test_the_latest_approval_is_of_the_file_as_it_is_now():
    """An approval of an earlier version of the file is not an approval of this
    one. Seven files were changed by a redaction after being approved, which is
    exactly how the two can drift apart (R-10)."""
    stale = []

    for name, details in manifest()["files"].items():
        latest = saved_replies.latest_approval(details)
        if latest.get("approved_sha256") != details.get("sha256"):
            stale.append(name)

    assert not stale, f"approved as they were, not as they are: {stale}"


def test_every_approval_says_where_the_repository_stood():
    """Without it the record says a file was accepted and not against what."""
    missing = []

    for name, details in manifest()["files"].items():
        for approval in details["approval_history"]:
            if not approval.get("repository_head_at_review"):
                missing.append(name)

    assert not missing, f"approved with no repository head recorded: {missing}"


def _is_a_commit_identifier(text):
    """True when this is the full form of a commit identifier."""
    # Forty hexadecimal characters. A short form is refused because it stops
    # being unique as a repository grows, and a word like HEAD is refused because
    # it names wherever the repository happens to be standing rather than what
    # was reviewed.
    return (isinstance(text, str) and len(text) == 40
            and all(c in "0123456789abcdef" for c in text))


# Skipped in the exported package, which the release command builds without
# `.git` on purpose: git can reach no commit where there is no repository, and
# on 1 October this failed there as though every approval named a lost commit.
# The missing `.git` entry is the only excuse; a git error in a checkout that
# has one still fails, because that may be a damaged repository.
@pytest.mark.skipif(
    not (SAVED_REPLIES.parents[1] / ".git").exists(),
    reason="an exported package has no repository for git to answer about")
def test_an_unreachable_head_is_only_accepted_for_a_declared_rewrite():
    """A commit that cannot be found is fine only where the manifest already
    says a rewrite removed it.
    """
    # An approval made before a rewrite names a commit from a history that no
    # longer exists, and that is the truth about it. One made after has no such
    # excuse: its commit has to be one this repository can reach.
    #
    # What the label is allowed to say comes from the manifest's own list of
    # rewrites, not from how the words begin. An earlier version accepted any
    # text starting with "before ", so "before lunch" excused an unreachable
    # commit as readily as "before R-10", and inventing an excuse was as easy as
    # typing one.
    import subprocess

    rewrites = manifest()["history_rewrites"]
    excuses = {"before " + rewrite["decision"] for rewrite in rewrites}

    unexplained = []
    for name, details in manifest()["files"].items():
        for approval in details["approval_history"]:
            head = approval["repository_head_at_review"]
            if not _is_a_commit_identifier(head):
                unexplained.append(f"{name}: {head!r} is not a commit identifier")
                continue

            reachable = subprocess.run(
                ["git", "merge-base", "--is-ancestor", head, "HEAD"],
                cwd=SAVED_REPLIES.parents[1], capture_output=True,
                check=False).returncode == 0
            if reachable:
                continue

            if approval.get("history") not in excuses:
                unexplained.append(
                    f"{name}: {head} cannot be reached, and "
                    f"{approval.get('history')!r} is not one of the rewrites "
                    f"the manifest declares ({sorted(excuses)})")

    assert not unexplained, unexplained


def _moment(text):
    """Reads a recorded time, or None if it is not one."""
    # Compared as times rather than as text. Both are written here in UTC, so
    # comparing the strings happens to work, but nothing makes them stay that
    # way: a time written with a different offset sorts by the digits it starts
    # with, and 2026-09-02T01:00+02:00 reads as earlier than the same instant
    # written as 2026-09-01T23:00Z. That is a comparison giving the wrong answer
    # quietly, which is the kind this file exists to remove.
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return moment


def test_no_approval_is_dated_before_the_reply_was_captured():
    """A file cannot have been accepted before it existed. A date that says so
    means the record was written from something other than what happened.
    """
    # A missing or unreadable time is a failure rather than a file to skip. An
    # earlier version passed over anything without a capture time, so removing
    # that one line was enough to put an approval beyond checking.
    impossible = []

    for path in saved_files():
        captured = _moment(json.loads(path.read_text(encoding="utf-8"))
                           .get("captured_at"))
        if captured is None:
            impossible.append(f"{path.name} does not say when it was captured, "
                              "so nothing can be compared with it")
            continue

        for approval in manifest()["files"][path.name]["approval_history"]:
            approved = _moment(approval.get("approved_at"))
            if approved is None:
                impossible.append(f"{path.name} has an approval with no "
                                  "readable time on it")
            elif approved < captured:
                impossible.append(f"{path.name} approved before it was captured")

    assert not impossible, impossible


def test_every_published_reply_has_an_approval_with_a_date_on_it():
    """A record of acceptance with nothing saying when is not a record."""
    # This says the real manifest has one. It does not say that approving again
    # keeps it, because nothing here approves anything: the test below does that
    # against a manifest of its own.
    for name, details in manifest()["files"].items():
        history = details["approval_history"]
        assert history, f"{name} has no approval at all"
        assert history[0].get("approved_at"), \
            f"{name} does not say when it was first approved"


def test_approving_a_second_time_leaves_the_first_approval_untouched(tmp_path):
    """Approving again adds to the record instead of replacing it."""
    # This is the invariant version two exists for, so it is proved by doing it
    # rather than by reading the manifest that already exists. Reading that one
    # can only show a history with something in it, which a writer that replaced
    # the entry every time would also produce.
    #
    # The reply is changed between the two approvals, because that is the case
    # that matters: a file approved, redacted, and accepted again. If the second
    # approval overwrote the first, the record would say the file had always been
    # what it is now, and the earlier version nobody can see any more would be
    # the one that went into the repository.
    #
    # It runs against folders of its own, so the real published replies are not
    # touched and nothing here is written where a demonstration reads from.
    captured = tmp_path / "captured"
    saved_replies.CAPTURED_DIR = captured
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    manifest_path = tmp_path / "manifest.json"

    reply = captured / "a_reply.json"
    reply.parent.mkdir(parents=True)
    reply.write_text(json.dumps({
        "captured_at": "2026-09-01T10:00:00+00:00",
        "method": "GET",
        "url": "https://api.crossref.org/works?query=anything",
        "asked": {"query": "anything"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"message": {"items": [{"title": ["a paper"]}]}}),
    }), encoding="utf-8")

    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    record = saved_replies.read_manifest(manifest_path)
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record, manifest_path)

    first = json.dumps(saved_replies.read_manifest(manifest_path)
                       ["files"]["a_reply.json"]["approval_history"][0],
                       sort_keys=True)

    # The reply is changed, and then accepted again at a different commit.
    held = json.loads(reply.read_text(encoding="utf-8"))
    held["body"] = json.dumps({"message": {"items": [{"title": ["another"]}]}})
    reply.write_text(json.dumps(held), encoding="utf-8")

    record = saved_replies.read_manifest(manifest_path)
    later = {"commit": "b" * 40, "working_tree": "changes not committed"}
    assert saved_replies.approve_reply(reply, record, later) == []
    assert saved_replies.write_manifest(record, manifest_path)

    history = (saved_replies.read_manifest(manifest_path)
               ["files"]["a_reply.json"]["approval_history"])

    assert len(history) == 2, "the second approval replaced the first"
    assert json.dumps(history[0], sort_keys=True) == first, \
        "the first approval was rewritten by the second"
    assert history[1]["repository_head_at_review"] == "b" * 40
    assert history[0]["approved_sha256"] != history[1]["approved_sha256"], \
        "the two approvals should be of different versions of the file"


def _a_captured_reply(folder):
    """Writes one reply for a test to approve, and returns where it is."""
    reply = folder / "a_reply.json"
    reply.parent.mkdir(parents=True, exist_ok=True)
    reply.write_text(json.dumps({
        "captured_at": "2026-09-01T10:00:00+00:00",
        "method": "GET",
        "url": "https://api.crossref.org/works?query=anything",
        "asked": {"query": "anything"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"message": {"items": [{"title": ["a paper"]}]}}),
    }), encoding="utf-8")
    return reply


def test_a_manifest_that_cannot_be_written_publishes_nothing(tmp_path,
                                                             monkeypatch):
    """Publishing fails closed, tested by breaking the half that can fail."""
    # This is the case the old order got wrong. The reply was copied into the
    # published folder and the manifest written afterwards, so a manifest that
    # could not be written left the reply sitting in a public folder with nothing
    # recording that anybody had accepted it, while the command reported that
    # nothing had been approved. Nothing may be published that no record
    # names, and that is the direction this protects.
    #
    # The failure is caused rather than waited for: writing is made to fail, so
    # the test proves what happens instead of hoping to see it one day.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    manifest_path = tmp_path / "manifest.json"
    reply = _a_captured_reply(tmp_path / "captured")

    record = saved_replies.read_manifest(manifest_path)
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []

    monkeypatch.setattr(saved_replies, "_write", lambda path, contents: False)
    assert saved_replies.write_manifest(record, manifest_path) is False

    assert saved_replies.files_in(saved_replies.APPROVED_DIR) == [], \
        "a reply was published although its record was never written"
    assert not manifest_path.exists()


def test_a_reply_that_cannot_be_published_leaves_the_manifest_alone(tmp_path,
                                                                    monkeypatch):
    """The same rule from the other end."""
    # Here the record writes and moving the file into place does not. Left as it
    # was, the manifest would name a reply that is not there, and every later
    # check would report the folder as tampered with because of something that
    # never happened. What was there before is put back instead.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    manifest_path = tmp_path / "manifest.json"
    reply = _a_captured_reply(tmp_path / "captured")

    record = saved_replies.read_manifest(manifest_path)
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []

    # Only the move out of the staging folder is broken. Refusing every move
    # would break writing the manifest as well, which is written through a
    # second file and moved into place, and the test would then pass without
    # ever reaching the step it is about. It did, until this was narrowed.
    moving = pathlib.Path.replace

    def refuse_to_move(self, destination):
        if saved_replies.STAGING_DIR_NAME in str(self):
            raise OSError("the file could not be moved into place")
        return moving(self, destination)

    monkeypatch.setattr(pathlib.Path, "replace", refuse_to_move)
    assert saved_replies.write_manifest(record, manifest_path) is False

    assert saved_replies.files_in(saved_replies.APPROVED_DIR) == []
    assert not manifest_path.exists(), \
        "the manifest recorded an approval of a file that was never published"


def test_a_redacted_reply_is_not_replayed_until_it_is_approved_again(tmp_path):
    """Replay checks what somebody accepted, not only what the entry says now."""
    # Redaction changes a published file and writes its new fingerprint into the
    # entry. What it cannot do is get the file approved again. So the top-level
    # fingerprint agrees with the file while the approval recording what somebody
    # actually looked at does not, and a check reading only the first would
    # replay a version nobody has seen.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    assert saved_replies.is_approved(published)

    # What redaction does: the file changes, the entry is brought up to date,
    # and no new approval is recorded.
    held = json.loads(published.read_text(encoding="utf-8"))
    held["body"] = json.dumps({"message": {"items": [{"title": ["changed"]}]}})
    published.write_text(json.dumps(held), encoding="utf-8")
    record = saved_replies.read_manifest()
    record["files"][reply.name]["sha256"] = saved_replies.fingerprint(published)
    record["files"][reply.name]["needs_approving_again"] = "changed by a redaction"
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published), \
        "a file changed after approval was replayed as though it had been seen"

    # And again with the note taken off, because the note is not what makes it
    # unapproved. Somebody tidying the manifest, or an older redaction that
    # never wrote one, must not be enough to make the file replayable: what
    # settles it is that the approval recorded a different file. Written as one
    # assertion, removing the fingerprint check left this test passing.
    record = saved_replies.read_manifest()
    del record["files"][reply.name]["needs_approving_again"]
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published), \
        "the file disagreed with its approval and was replayed anyway"

    # The note on its own, on a file that does match its approval. Redaction
    # always changes the file, so the note never arrives alone in practice and
    # the check on it would be proved by nothing. It is the part that will
    # still hold if something later marks a file for review without changing
    # it, so it is put in that position here deliberately.
    record = saved_replies.read_manifest()
    entry = record["files"][reply.name]
    agreed = saved_replies.fingerprint(published)
    entry["sha256"] = agreed
    entry["approval_history"][-1]["approved_sha256"] = agreed
    # The decision to publish is about particular bytes too, so it moves with
    # them. Approving again through the command does this properly; it is done
    # by hand here because the point of the next assertion is that everything
    # except the note agrees.
    entry["approval_history"][-1]["publication_basis"]["applies_to_sha256"] = agreed
    assert saved_replies.write_manifest(record)
    assert saved_replies.is_approved(published), "everything agrees again"

    record = saved_replies.read_manifest()
    record["files"][reply.name]["needs_approving_again"] = "somebody asked"
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published), \
        "a file marked as needing review was replayed anyway"


def test_a_manifest_that_cannot_be_read_stops_rather_than_reading_as_empty(tmp_path):
    """A damaged manifest is not an empty one."""
    # Read alike, an approval would write a new manifest over a damaged one and
    # take every earlier approval with it, which is the record this whole file
    # exists to protect. Replay treats the same failure as nothing being
    # approved, because refusing to replay ends a demonstration while replaying
    # against a manifest nobody can read is what the manifest is there to stop.
    damaged = tmp_path / "manifest.json"
    damaged.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(saved_replies.ManifestProblem):
        saved_replies.read_manifest(damaged)

    saved_replies.MANIFEST_PATH = damaged
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    published = saved_replies.APPROVED_DIR / "anything.json"
    published.parent.mkdir(parents=True)
    published.write_text("{}", encoding="utf-8")

    assert saved_replies.is_approved(published) is False


def test_a_manifest_from_a_later_version_is_not_read_as_an_older_one(tmp_path):
    """A version this code does not know was written by something that knew
    more than it does. Reading it as the oldest shape would quietly discard
    whatever that was, so it stops instead."""
    later = tmp_path / "manifest.json"
    later.write_text(json.dumps({"manifest_version": 3, "files": {}}),
                     encoding="utf-8")

    with pytest.raises(saved_replies.ManifestProblem):
        saved_replies.read_manifest(later)


def test_a_missing_manifest_is_an_ordinary_state(tmp_path):
    """Nothing has been approved yet, which is where every checkout starts."""
    empty = saved_replies.read_manifest(tmp_path / "not_there.json")

    assert empty["files"] == {}
    assert empty["manifest_version"] == saved_replies.MANIFEST_VERSION


@pytest.mark.parametrize("commit", ["", "unknown", "HEAD", "abc1234", "x" * 40,
                                    "A" * 40, None])
def test_an_approval_that_cannot_name_what_was_reviewed_is_refused(tmp_path,
                                                                   commit):
    """Checked where approving happens, not only in the command."""
    # The command asks git and validates the answer. Anything else calling
    # approval directly did not, so a single letter could be written down as the
    # commit somebody reviewed, and the manifest says every approval names one.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    reply = _a_captured_reply(tmp_path / "captured")

    record = saved_replies.read_manifest(tmp_path / "manifest.json")
    reasons = saved_replies.approve_reply(reply, record,
                                          {"commit": commit,
                                           "working_tree": "anything"})

    assert reasons, f"{commit!r} was accepted as the commit that was reviewed"
    assert saved_replies.files_in(saved_replies.APPROVED_DIR) == []


def test_an_entry_with_no_approval_on_it_is_not_replayed(tmp_path):
    """A fingerprint is not an approval."""
    # An entry can name a file and match it exactly while recording that nobody
    # ever accepted it. That is what a hand-edited manifest looks like, and it is
    # the shape somebody would write to make a file replayable without going
    # through approval, so the fingerprint agreeing is not enough on its own.
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    published = saved_replies.APPROVED_DIR / "a_reply.json"
    published.parent.mkdir(parents=True)
    published.write_text('{"captured_at": "2026-09-01T10:00:00+00:00"}',
                         encoding="utf-8")

    record = saved_replies.read_manifest()
    record["files"]["a_reply.json"] = {
        "sha256": saved_replies.fingerprint(published),
        "approval_history": [],
    }
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published), \
        "a file nobody approved was replayed because its fingerprint matched"


def test_a_reply_nobody_decided_could_be_published_is_not_replayed(tmp_path):
    """Approved is not the same as publishable."""
    # Every approval recorded before R-11 says a person accepted the file. None
    # of them says anything about whether its contents may be published, because
    # nothing asked. Reading that silence as permission is precisely how fifteen
    # replies stayed replayable while the question was open, so a latest approval
    # with no decision in it is refused.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    assert saved_replies.is_approved(published)

    # An approval in the shape they were written in before the question was
    # asked: fingerprints, a commit, and nothing about publishing.
    record = saved_replies.read_manifest()
    del record["files"][reply.name]["approval_history"][-1]["publication_basis"]
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published), \
        "a reply nobody decided could be published was replayed"
    assert saved_replies.differences_from_manifest(), \
        "the manifest check should say so too"
    assert saved_replies.differences_from_manifest(files_only=True) == [], \
        "the folder itself is in order, which is what reapproval needs"


def test_a_decision_taken_about_other_bytes_does_not_cover_this_file(tmp_path):
    """The decision is about a version of the file, not about its name."""
    # A record saying somebody decided this could be published, taken about
    # different contents, is worth no more than one saying nothing: what was
    # looked at is not what would be replayed.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    record = saved_replies.read_manifest()
    decision = (record["files"][reply.name]["approval_history"][-1]
                ["publication_basis"])
    decision["applies_to_sha256"] = "0" * 64
    assert saved_replies.write_manifest(record)

    assert not saved_replies.is_approved(published)


def _a_reply_carrying_an_abstract_and_an_address(folder):
    """A reply publishing will change, because it holds a contact address."""
    reply = folder / "with_an_abstract.json"
    reply.parent.mkdir(parents=True, exist_ok=True)
    reply.write_text(json.dumps({
        "captured_at": "2026-09-01T10:00:00+00:00",
        "method": "GET",
        "url": "https://api.crossref.org/works/10.0000/example",
        "asked": {},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"message": {"items": [{
            "title": ["a paper"],
            "abstract": "Background: findings. Write to a.author@example.org.",
        }]}}),
    }), encoding="utf-8")
    return reply


A_LICENCE_THAT_FORBIDS_CHANGES = {
    "kind": "open licence",
    "applies_to": "with_an_abstract.json",
    "licence": "CC BY-ND 4.0",
    "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/",
    "source": "https://doi.org/10.0000/example",
    "attribution": "A. Author and others, 2026",
    "changes": "a contact address was removed",
    "redaction": "a contact address was removed",
}


def test_a_recorded_change_is_not_forgotten_when_a_reply_is_approved_again(
        tmp_path):
    """Whether publishing changed a reply is a fact about what was published."""
    # Approving again reads the published file, which already had its contact
    # details taken out, so comparing that with its own rebuilt form finds
    # nothing. Worked out afresh, the answer would flip from yes to no, and a
    # licence that forbids sharing a changed version would stop refusing the one
    # reply it exists to refuse.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_reply_carrying_an_abstract_and_an_address(tmp_path / "captured")
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}

    # A licence forbidding changed versions is refused, because publishing
    # takes the address out and that is a change.
    record = saved_replies.read_manifest()
    assert saved_replies.approve_reply(reply, record, where,
                                       A_LICENCE_THAT_FORBIDS_CHANGES)

    # Published under one that permits it, the change is recorded.
    allows = {**A_LICENCE_THAT_FORBIDS_CHANGES, "licence": "CC BY 4.0",
              "licence_url": "https://creativecommons.org/licenses/by/4.0/"}
    record = saved_replies.read_manifest()
    assert saved_replies.approve_reply(reply, record, where, allows) == []
    assert saved_replies.write_manifest(record)

    decision = saved_replies.latest_approval(
        saved_replies.read_manifest()["files"]["with_an_abstract.json"]
    )["publication_basis"]
    assert decision["content_was_changed"] is True

    # Approved again from the published file, which has already been redacted.
    published = saved_replies.APPROVED_DIR / "with_an_abstract.json"
    record = saved_replies.read_manifest()
    assert saved_replies.approve_reply(published, record, where, allows) == []
    assert saved_replies.write_manifest(record)

    history = (saved_replies.read_manifest()["files"]["with_an_abstract.json"]
               ["approval_history"])
    assert history[-1]["publication_basis"]["content_was_changed"] is True, \
        "the second approval forgot that publishing had changed the reply"

    # And the consequence that matters: the licence forbidding changed versions
    # is still refused when the reply is approved again from the published
    # file. If the fact were worked out afresh it would now say nothing was
    # changed, and the one reply this licence exists to refuse would pass.
    record = saved_replies.read_manifest()
    refused = saved_replies.approve_reply(published, record, where,
                                          A_LICENCE_THAT_FORBIDS_CHANGES)

    assert any("changed version" in reason for reason in refused), \
        f"a NoDerivatives basis was accepted on reapproval: {refused}"


def test_a_basis_is_not_recorded_against_a_reply_that_needed_none(tmp_path):
    """A licence read from one paper must not be filed against another reply."""
    # One basis can no longer be spread across several replies holding article
    # text. It could still be written against the metadata replies approved in
    # the same command, which would put a claim about a particular paper beside
    # a list of identifiers that has nothing to do with it.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    offered = {"kind": "open licence", "applies_to": reply.name,
               "licence": "CC BY 4.0",
               "licence_url": "https://creativecommons.org/licenses/by/4.0/",
               "source": "https://doi.org/10.0000/somebody-elses-paper",
               "attribution": "somebody else", "changes": "none",
               "redaction": "nothing"}

    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where, offered) == []
    assert saved_replies.write_manifest(record)

    decision = saved_replies.latest_approval(
        saved_replies.read_manifest()["files"][reply.name])["publication_basis"]

    assert decision["kind"] == "no recognised article-text field", \
        "a licence was recorded against a reply that holds no article text"
    assert "source" not in decision


def test_a_replayed_reply_is_read_from_disk_only_once(tmp_path, monkeypatch):
    """What was checked is what is returned."""
    # Replay used to ask whether the file was approved, which read and
    # fingerprinted it, and then read it again to return what it held. A file
    # changed between those reads would have been approved as one thing and
    # replayed as another, which is precisely what the fingerprint is for and
    # was standing open beside it.
    #
    # Counting the reads is a plain way to say so. It cannot show that no two
    # reads could ever disagree; it shows there is only one, which is the
    # property that makes the question moot.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    reads = []
    real_bytes, real_text = pathlib.Path.read_bytes, pathlib.Path.read_text

    def count_bytes(self):
        if self == published:
            reads.append("bytes")
        return real_bytes(self)

    def count_text(self, *args, **kwargs):
        if self == published:
            reads.append("text")
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", count_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", count_text)

    assert saved_replies._the_approved_reply_in(published) is not None

    assert reads == ["bytes"], \
        f"the replayed file was read {len(reads)} times: {reads}"


def test_a_replayed_fixture_is_read_from_disk_only_once(tmp_path, monkeypatch):
    """The same property, for the other kind of record an entry can carry."""
    # A fixture takes a different route through the replay check from a captured
    # reply, and the first version of that route read the file again instead of
    # using what had just been read. Counting the reads for a captured reply
    # could not have shown it, because a captured reply never goes that way. Two
    # routes to the same guarantee need two counts (R-16).
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    published = saved_replies.APPROVED_DIR / "a_fixture.json"
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(json.dumps({
        "captured_at": "2026-09-09T00:00:00+00:00",
        "model": "a model",
        "question": "a question",
        "answer": "an answer",
    }), encoding="utf-8")
    written = saved_replies.fingerprint(published)

    manifest = saved_replies.read_manifest()
    manifest["files"][published.name] = {
        "sha256": written,
        "approval_history": [{
            "approved_at": "2026-09-09T00:00:00+00:00",
            "approved_sha256": written,
            "repository_head_at_review": "a" * 40,
            "working_tree_at_review": "nothing uncommitted",
            "fixture_origin": {
                "written_by": "the author of this test",
                "written_on": "2026-09-09",
                "what_it_stands_in_for": "an answer from the model",
                "copied_from_a_paper": False,
                "applies_to_sha256": written,
            },
        }],
    }
    assert saved_replies.write_manifest(manifest)

    reads = []
    real_bytes, real_text = pathlib.Path.read_bytes, pathlib.Path.read_text

    def count_bytes(self):
        if self == published:
            reads.append("bytes")
        return real_bytes(self)

    def count_text(self, *args, **kwargs):
        if self == published:
            reads.append("text")
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_bytes", count_bytes)
    monkeypatch.setattr(pathlib.Path, "read_text", count_text)

    assert saved_replies._the_approved_reply_in(published) is not None

    assert reads == ["bytes"], \
        f"the replayed fixture was read {len(reads)} times: {reads}"


def test_replaying_here_and_publishing_elsewhere_are_different_questions(tmp_path):
    """A reply kept for private replay may be replayed and may not be exported."""
    # The first version of this recorded the restriction where a publication
    # basis goes, and asked one question for both. A file marked "not for
    # publication" was then handed to a check called "does the publication
    # decision hold", which found nothing to complain about, so a refusal read
    # as permission. The record was right and the code reading it was not.
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    record = saved_replies.read_manifest()

    # Approved with a decision that it holds no recognised article-text field:
    # replayable here, and exportable.
    entry = record["files"][reply.name]
    assert saved_replies.why_it_cannot_be_replayed(entry, published) == []
    assert saved_replies.why_it_cannot_be_exported(entry, published) == []

    # Set aside for private replay: still replayable here, no longer
    # exportable. That is the whole distinction.
    saved_replies.keep_private(record, decided_at=where)
    assert saved_replies.write_manifest(record)

    entry = saved_replies.read_manifest()["files"][reply.name]
    assert saved_replies.why_it_cannot_be_replayed(entry, published) == []
    assert saved_replies.why_it_cannot_be_exported(entry, published), \
        "a reply kept for private replay was cleared for export"
    assert saved_replies.is_approved(published), "it still replays here"


def test_a_reply_set_aside_as_other_bytes_is_not_replayed(tmp_path):
    """Being set aside is about a version of the file, like everything else."""
    saved_replies.CAPTURED_DIR = tmp_path / "captured"
    saved_replies.APPROVED_DIR = tmp_path / "approved"
    saved_replies.MANIFEST_PATH = tmp_path / "manifest.json"

    reply = _a_captured_reply(tmp_path / "captured")
    record = saved_replies.read_manifest()
    where = {"commit": "a" * 40, "working_tree": "nothing uncommitted"}
    assert saved_replies.approve_reply(reply, record, where) == []
    # Published first: nothing is in the folder until the manifest is written,
    # and setting aside reads the folder.
    assert saved_replies.write_manifest(record)

    record = saved_replies.read_manifest()
    saved_replies.keep_private(record, decided_at=where)
    assert saved_replies.write_manifest(record)

    published = saved_replies.APPROVED_DIR / reply.name
    record = saved_replies.read_manifest()
    entry = record["files"][reply.name]
    entry["approval_history"][-1]["replay_scope"]["applies_to_sha256"] = "0" * 64

    assert saved_replies.why_it_cannot_be_replayed(entry, published)
