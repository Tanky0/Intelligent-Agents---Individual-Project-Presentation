"""Tries to get things past the release check that must not get past it."""
# The check refuses everything unless something allows it, so most of these are
# attempts to make a forbidden thing look like an allowed one: a private
# recording under a harmless name, an abstract in a column called something else,
# a fixture nothing describes, a key in a file nobody would search.
#
# Each case is one that would have worked at some point in this project, or is
# the same shape as one that did. R-11 published article text inside saved
# replies. R-12 published eighty-seven abstracts inside a screening file that the
# reply check never looked at. Both were files nobody had thought about, which is
# why the rule is a list of what may travel rather than a list of what may not.
#
# Every export here is built in a temporary folder. Nothing reads or writes the
# real repository.

import json
import pathlib
import sys

import pytest

from research_agent import release
from research_agent.sources import saved_replies

A_FIXTURE = {
    "captured_at": "2026-09-01T10:00:00+00:00",
    "method": "GET",
    "url": "https://api.crossref.org/works?query=anything",
    "asked": {"query": "anything"},
    "status": 200,
    "headers": {"content-type": "application/json"},
    "body": json.dumps({"message": {"items": [{"title": ["An invented paper"]}]}}),
}


def write(root, relative, text):
    """Writes one file into the export."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SAMPLE = (pathlib.Path(__file__).resolve().parents[2]
          / "data" / "hand_screened_sample")

# What a version two manifest says about itself. Written out here rather than
# taken from the module, so that a change to the real header is a change these
# tests notice rather than follow.
#
# The list of rewrites is empty, which is what a public repository built from
# this export would hold. R-10 and R-11 were rewrites of the working
# repository's history, and a repository whose first commit is the export has
# no earlier history to describe. The fixtures claimed an R-11 rewrite for a
# while, so every test here agreed with a check that asked a new repository
# to declare a rewrite it never had.
A_VERSION_TWO_HEADER = {
    "manifest_version": 2,
    "what_approval_means": "somebody accepted this file as it stood",
    "what_repository_head_at_review_means": "where the repository stood then",
    "history_rewrites": [],
}


def a_release(root):
    """Builds the smallest export that the check accepts."""
    # Written out rather than copied from the repository, so a change to the real
    # files cannot quietly change what these tests mean.
    #
    # It carries no screening file. An earlier version put an invented one here
    # with a single row, which the check refuses because the benchmark is fifty
    # rows with a known fingerprint. Every test asking for no problems then
    # failed. Worse, every test asking whether some problem was found passed on
    # that one instead of on the thing it was testing. A baseline that is not
    # clean makes a test pass for the wrong reason.
    write(root, "README.md", "# The agent\n\nHow to install and run it.\n")
    write(root, "LICENSE", "MIT License\n")
    write(root, "requirements.txt", "pydantic==2.13.4\n")
    write(root, "pyproject.toml", "[project]\nname = 'research-agent'\n")
    write(root, "research_agent/__init__.py", "")
    write(root, "tests/unit/test_something.py", "def test_it():\n    assert True\n")
    return root


def with_the_screening_files(root):
    """Adds the real benchmark files, copied from the repository."""
    # Copied rather than invented, because what the check compares them against
    # is the fingerprint taken when they were reduced. Anything made up here
    # would be refused, correctly, and would prove nothing about the rule.
    for name in ("screening_labels.csv", "screening_labels_check.csv"):
        write(root, f"data/hand_screened_sample/{name}",
              (SAMPLE / name).read_text(encoding="utf-8-sig"))
    return root


def with_a_fixture(root, name="a_fixture.json", origin=None, record=None,
                   approval=None):
    """Adds one synthetic fixture and the manifest entry describing it."""
    path = write(root, f"data/recorded_replies/{name}",
                 json.dumps(record if record is not None else A_FIXTURE,
                            indent=2))
    entry_origin = {
        "written_by": "the person who built this release",
        "written_on": "2026-09-03",
        "what_it_stands_in_for": "a Crossref search returning one record",
        "copied_from_a_paper": False,
        "applies_to_sha256": saved_replies.fingerprint(path),
    }
    if origin is not None:
        entry_origin = {**entry_origin, **origin} if origin else origin

    approved = {
        "approved_at": "2026-09-03T10:00:00+00:00",
        "approved_sha256": saved_replies.fingerprint(path),
        "repository_head_at_review": "a" * 40,
        "working_tree_at_review": "nothing uncommitted",
        "fixture_origin": entry_origin,
    }
    if approval:
        approved = {**approved, **approval}

    write(root, "data/recorded_replies_manifest.json", json.dumps({
        **A_VERSION_TWO_HEADER,
        "files": {name: {
            "sha256": saved_replies.fingerprint(path),
            "approval_history": [approved],
        }},
    }, indent=2))
    return path


def _repoint_the_manifest(root, path):
    """Points the entry's three fingerprints at the file as it stands now."""
    # Used where a test changes a fixture on purpose and wants the check to fail
    # for what is in the file rather than for the fingerprints disagreeing.
    manifest_path = root / "data" / "recorded_replies_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    now = saved_replies.fingerprint(path)
    entry = manifest["files"][path.name]
    entry["sha256"] = now
    entry["approval_history"][-1]["approved_sha256"] = now
    entry["approval_history"][-1]["fixture_origin"]["applies_to_sha256"] = now
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_everything_this_release_requires_is_a_thing_it_may_carry():
    """The two lists have to agree, and for a while they did not."""
    # One says what may be published and the other says what this project decided
    # a release of it must hold. A file required by the second and refused by the
    # first is a release nobody can build: it fails the safety check for carrying
    # it and the completeness check for leaving it out. That happened the moment
    # the evidence folder stopped being allowed as a whole, and the final test
    # output became required and forbidden at once.
    refused = [relative for relative in release.REQUIRED_FOR_THIS_RELEASE
               if relative not in release.ALLOWED_EXACT
               and not release._matches_a_pattern(relative)
               and not release._in_an_allowed_tree(relative)]

    assert refused == [], (
        "this project requires these of a release and the check refuses them: "
        + ", ".join(refused))


def test_every_document_this_project_has_is_classified():
    """A document left out of both lists is left out with nobody deciding."""
    # This reads the real docs/ folder, where the rest of this file deliberately
    # does not read the repository. The other tests ask whether the rules are
    # right. This one asks whether anything has been left outside the rules, and
    # an invented folder could not answer that.
    #
    # It found two. One says in its own first paragraph that it is kept in
    # the repository because evidence nobody else can read cannot show that
    # the work was planned, and the export would have left it out. The
    # other holds the architecture decisions the traceability table draws on.
    # Neither was refused by anything; both were simply never mentioned.
    #
    # Every file under docs/ is looked at, not the markdown at the top of it. A
    # note in a folder below, or a document saved as anything else, is exactly
    # the kind of file this project has twice published without meaning to, and
    # a test that only reads *.md would have said the folder was accounted for.
    root = pathlib.Path(__file__).resolve().parents[2]
    unclassified = _documents_nobody_classified(root)

    assert unclassified == [], (
        "these are neither allowed into a release nor recorded as deliberately "
        "kept back: " + ", ".join(unclassified))


def _documents_nobody_classified(root):
    """The files under docs/ that neither list accounts for."""
    # The kept-back list is read from the root being checked, and is empty
    # where the file is absent. That is not a loophole: in the repository the
    # documents the list names are then unclassified and reported, so a list
    # deleted by accident fails this check. In an export the list is absent by
    # design, and every document under docs/ there is one the allow list
    # names, so the same rule passes without it. An earlier version asserted
    # the list was present, which made the test fail in the export it exists
    # to protect (found in review on 21 September).
    kept_back = release.kept_back(root)
    unclassified = []
    for path in (root / "docs").rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in release.ALLOWED_EXACT:
            continue
        if relative in kept_back:
            continue
        unclassified.append(relative)
    return sorted(unclassified)


def test_an_export_carrying_no_list_passes_when_its_documents_are_allowed(tmp_path):
    """An export leaves the kept-back list behind on purpose."""
    # The shape the classifying test failed in before 22 September: no list,
    # and only exported documents under docs/.
    write(tmp_path, "docs/testing.md", "# Testing\n")
    write(tmp_path, "docs/development.md", "# Development\n")

    assert _documents_nobody_classified(tmp_path) == []


def test_a_document_nothing_allows_is_reported_with_the_list_and_without_it(tmp_path):
    """A stray document is refused whether or not the list is there."""
    write(tmp_path, "docs/testing.md", "# Testing\n")
    write(tmp_path, "docs/notes.md", "a note nobody read\n")

    assert _documents_nobody_classified(tmp_path) == ["docs/notes.md"]

    write(tmp_path, release.NOT_EXPORTED_LIST,
          json.dumps({release.NOT_EXPORTED_LIST: "the list itself"}))

    assert _documents_nobody_classified(tmp_path) == ["docs/notes.md"]


def test_nothing_is_both_allowed_and_kept_back():
    """The two lists are answers to one question, so they cannot overlap."""
    # A path in both says a file may travel and must not, and whichever list is
    # read first decides. A rule that depends on the order it is read in is not
    # a rule.
    root = pathlib.Path(__file__).resolve().parents[2]
    both = sorted(set(release.ALLOWED_EXACT) & set(release.kept_back(root)))

    assert both == [], (
        "these are allowed into a release and recorded as kept back at the "
        "same time: " + ", ".join(both))


def test_a_minimal_release_is_accepted(tmp_path):
    """A check that refused everything would stop the release being made."""
    # This is the one that has to pass, and it is first because the rest of the
    # file means nothing without it. A rule that refused every release would stop
    # any release being made.
    assert release.problems_in(a_release(tmp_path)) == []


def test_a_release_with_a_fixture_and_its_origin_is_accepted(tmp_path):
    """The ordinary case for the public demonstration set."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)

    assert release.problems_in(tmp_path) == []


def test_an_empty_export_is_refused(tmp_path):
    """Nothing to publish is not the same as nothing wrong."""
    # A copying step that silently did nothing would otherwise produce a folder
    # that passes every check in this file.
    problems = release.problems_in(tmp_path)

    assert any("empty" in problem for problem in problems)


def test_an_export_missing_its_readme_is_refused(tmp_path):
    """The brief asks for a README, and a release without one is broken."""
    a_release(tmp_path)
    (tmp_path / "README.md").unlink()

    assert any("README.md" in problem for problem in release.problems_in(tmp_path))


def test_a_private_recording_renamed_to_look_harmless_is_refused(tmp_path):
    """The R-11 case, under a different name."""
    # A recording kept for replaying in the private sandbox is refused by what
    # the manifest says about it, not by what it is called. Renaming it changes
    # the file name and nothing else.
    a_release(tmp_path)
    path = write(tmp_path, "data/recorded_replies/harmless_example.json",
                 json.dumps(A_FIXTURE, indent=2))
    write(tmp_path, "data/recorded_replies_manifest.json", json.dumps({
        "manifest_version": 2,
        "files": {"harmless_example.json": {
            "sha256": saved_replies.fingerprint(path),
            "approval_history": [{
                "approved_at": "2026-09-03T10:00:00+00:00",
                "approved_sha256": saved_replies.fingerprint(path),
                "repository_head_at_review": "a" * 40,
                "working_tree_at_review": "nothing uncommitted",
                "replay_scope": {
                    "scope": saved_replies.KEPT_PRIVATE,
                    "applies_to_sha256": saved_replies.fingerprint(path),
                    "publication_basis": None,
                    "reason": "No right to publish this has been established.",
                },
            }],
        }},
    }, indent=2))

    problems = release.problems_in(tmp_path)

    assert any("private replay" in problem for problem in problems), problems


def test_the_export_check_actually_asks_the_publication_rule(tmp_path,
                                                             monkeypatch):
    """Proves the rule is called, not merely written."""
    # why_it_cannot_be_exported existed for a while with nothing calling it,
    # which is the state R-12 described as a rule without a check. Made to refuse
    # everything, the export must fail; if it still passed, the rule would be
    # decoration.
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    assert release.problems_in(tmp_path) == []

    monkeypatch.setattr(saved_replies, "why_it_cannot_be_exported",
                        lambda entry, path, record=None: ["refused for a test"])

    assert any("refused for a test" in problem
               for problem in release.problems_in(tmp_path))


def test_the_real_screening_files_are_accepted(tmp_path):
    """The baseline the two tests below are measured against."""
    a_release(tmp_path)
    with_the_screening_files(tmp_path)

    assert release.problems_in(tmp_path) == []


def test_an_abstract_in_another_column_of_a_screening_file_is_refused(tmp_path):
    """The R-12 case, moved."""
    # The columns are checked, not the file name, so putting the abstract under
    # another heading in the same file does not help. A screening file carried
    # here holds three columns and no others.
    a_release(tmp_path)
    with_the_screening_files(tmp_path)
    write(tmp_path, "data/hand_screened_sample/screening_labels.csv",
          "pmid,include,rationale,notes\n"
          "32768971,exclude,\"review\",\"Background: this study examined...\"\n")

    problems = release.problems_in(tmp_path)

    assert any("columns" in problem for problem in problems), problems


def test_an_abstract_in_the_reason_column_is_refused(tmp_path):
    """The same text, in a column that is allowed to hold text."""
    # Three columns with the abstract in the reason satisfies every rule about
    # shape, which is why the file is compared with the fingerprint taken when it
    # was reduced. Counting columns would let this through, and the reason column
    # is the obvious place to put it.
    a_release(tmp_path)
    with_the_screening_files(tmp_path)

    lines = (SAMPLE / "screening_labels.csv").read_text(
        encoding="utf-8-sig").splitlines()
    lines[1] = ('32768971,exclude,"Background: this study examined the effect '
                'of ivermectin on viral clearance in hospitalised patients"')
    write(tmp_path, "data/hand_screened_sample/screening_labels.csv",
          "\n".join(lines) + "\n")

    problems = release.problems_in(tmp_path)

    assert any("does not say what it said" in problem for problem in problems), \
        problems


def test_a_table_holding_a_column_of_article_text_is_refused(tmp_path):
    """Asked of the rule directly, because no such file is allowed today."""
    # The only tables a release carries are the two screening files, so a table
    # with an abstract column cannot currently reach an export at all: it is
    # refused earlier, for being a file nothing allows. The rule is kept for
    # whenever another table is added to the list, and tested here so that
    # adding one does not also have to remember this.
    path = write(tmp_path, "results.csv",
                 "pmid,abstract\n32768971,\"Background: this study...\"\n")

    problems = release._problems_with_a_table(path, "evidence/results.csv")

    assert any("text of a paper" in problem for problem in problems), problems


def test_a_prose_file_nobody_allowed_is_refused(tmp_path):
    """The answer to an abstract pasted into a note."""
    # Nothing here can read prose and say whose words are in it. So a document
    # travels only if its exact path was written down by somebody who read it,
    # and a new one is refused until that happens.
    a_release(tmp_path)
    write(tmp_path, "docs/notes_to_self.md",
          "Background: this study examined the effect of ivermectin...\n")

    problems = release.problems_in(tmp_path)

    assert any("nothing allows it" in problem for problem in problems), problems


def test_a_fixture_nothing_describes_is_refused(tmp_path):
    """A file in the replies folder with no manifest entry at all."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    write(tmp_path, "data/recorded_replies/unlisted.json",
          json.dumps(A_FIXTURE, indent=2))

    problems = release.problems_in(tmp_path)

    assert any("no manifest entry" in problem for problem in problems), problems


def test_a_fixture_changed_after_its_origin_was_recorded_is_refused(tmp_path):
    """The record is about a version of the file, as everywhere else here."""
    a_release(tmp_path)
    path = with_a_fixture(tmp_path)
    path.write_text(json.dumps({**A_FIXTURE, "status": 404}, indent=2),
                    encoding="utf-8")

    problems = release.problems_in(tmp_path)

    assert any("different version" in problem for problem in problems), problems


@pytest.mark.parametrize("origin, missing", [
    ({"written_by": ""}, "who wrote it"),
    ({"written_on": ""}, "when"),
    ({"what_it_stands_in_for": ""}, "what it stands in for"),
    ({"copied_from_a_paper": True}, "not copied from a paper"),
    ({"copied_from_a_paper": None}, "not copied from a paper"),
])
def test_a_fixture_whose_origin_says_too_little_is_refused(tmp_path, origin,
                                                           missing):
    """Nothing can read a file and tell that somebody invented it."""
    # That is why the origin is recorded rather than detected, and why a record
    # that leaves out who, when, or that it was not copied is not a record.
    a_release(tmp_path)
    with_a_fixture(tmp_path, origin=origin)

    problems = release.problems_in(tmp_path)

    assert any(missing in problem for problem in problems), problems


def test_a_fixture_that_is_not_readable_json_is_refused(tmp_path):
    """A fixture is read, not only fingerprinted."""
    # Recording where something came from says nothing about whether it is what
    # it claims to be. This one is described perfectly and is not a reply, so the
    # demonstration it exists for would fail on it.
    a_release(tmp_path)
    path = with_a_fixture(tmp_path)
    path.write_text("{ not json at all", encoding="utf-8")
    _repoint_the_manifest(tmp_path, path)

    problems = release.problems_in(tmp_path)

    assert any("cannot be read" in problem for problem in problems), problems


def test_a_fixture_that_is_not_a_shape_replay_can_use_is_refused(tmp_path):
    """Readable, and not a saved reply."""
    a_release(tmp_path)
    path = with_a_fixture(tmp_path)
    path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    _repoint_the_manifest(tmp_path, path)

    problems = release.problems_in(tmp_path)

    assert any("replaying it would need" in problem for problem in problems), \
        problems


@pytest.mark.parametrize("which, expected", [
    ("sha256", "the manifest records a different version"),
    ("approved_sha256", "its approval accepted a different version"),
    ("applies_to_sha256", "recorded origin is about a different version"),
])
def test_each_of_the_three_fingerprints_has_to_agree(tmp_path, which, expected):
    """Three statements about the same file, and the file is only the file all
    three describe. Checking the origin alone let the other two drift."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)

    manifest = json.loads(
        (tmp_path / "data" / "recorded_replies_manifest.json")
        .read_text(encoding="utf-8"))
    entry = manifest["files"]["a_fixture.json"]
    if which == "sha256":
        entry["sha256"] = "0" * 64
    elif which == "approved_sha256":
        entry["approval_history"][-1]["approved_sha256"] = "0" * 64
    else:
        entry["approval_history"][-1]["fixture_origin"]["applies_to_sha256"] = "0" * 64
    write(tmp_path, "data/recorded_replies_manifest.json",
          json.dumps(manifest, indent=2))

    problems = release.problems_in(tmp_path)

    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("written_on", [
    "when the fixtures were made",
    # The whole value has to be a date. Only the first ten characters were
    # read, so anything typed after a real date was accepted along with it,
    # and the field looked checked without being checked.
    "2026-09-03 or thereabouts",
    "2026-09-03T10:00:00+00:00",
    "03/09/2026",
])
def test_a_date_that_is_not_a_date_is_refused(tmp_path, written_on):
    """A field checked only for being filled in is a sentence, not a record."""
    a_release(tmp_path)
    with_a_fixture(tmp_path, origin={"written_on": written_on})

    problems = release.problems_in(tmp_path)

    assert any("not a date" in problem for problem in problems), problems


def test_a_plain_date_is_accepted(tmp_path):
    """The baseline the four above are measured against."""
    a_release(tmp_path)
    with_a_fixture(tmp_path, origin={"written_on": "2026-09-03"})

    assert release.problems_in(tmp_path) == []


@pytest.mark.parametrize("approval, expected", [
    ({"approved_at": ""}, "does not say when it was given"),
    ({"approved_at": "some time in September"}, "which is not a time"),
    ({"repository_head_at_review": "a" * 7}, "full identifier of a commit"),
    ({"repository_head_at_review": "unknown"}, "full identifier of a commit"),
    ({"working_tree_at_review": ""}, "what was uncommitted"),
])
def test_an_approval_recording_too_little_is_not_one_to_publish_on(
        tmp_path, approval, expected):
    """Three fingerprints agreeing is not the whole of an approval."""
    # They say the file has not changed since somebody accepted it. They do not
    # say when it was accepted, what the person was reading at the time, or
    # whether anything else was uncommitted, and those are what let a reader go
    # back and look. An entry carried over from the first version of the manifest
    # records the head as "unknown", truthfully, and that is an honest record of
    # a private replay rather than a basis for publishing.
    a_release(tmp_path)
    with_a_fixture(tmp_path, approval=approval)

    problems = release.problems_in(tmp_path)

    assert any(expected in problem for problem in problems), problems


def test_a_manifest_entry_that_is_not_a_record_is_reported_not_raised(tmp_path):
    """A wrong type used to stop the check with a traceback."""
    # A release check that crashes tells somebody less than one that names the
    # file, and the crash is easy to read as the tool being broken rather than
    # the release being wrong.
    a_release(tmp_path)
    path = with_a_fixture(tmp_path)
    manifest = json.loads(
        (tmp_path / "data" / "recorded_replies_manifest.json")
        .read_text(encoding="utf-8"))
    manifest["files"]["a_fixture.json"] = 42
    write(tmp_path, "data/recorded_replies_manifest.json",
          json.dumps(manifest, indent=2))

    problems = release.problems_in(tmp_path)

    assert any("not a record" in problem for problem in problems), problems
    assert path.exists()


def _with_a_manifest_where(tmp_path, **changes):
    """Rewrites the manifest header, keeping the entry it describes."""
    manifest_path = tmp_path / "data" / "recorded_replies_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field, value in changes.items():
        if value is _GONE:
            manifest.pop(field, None)
        else:
            manifest[field] = value
    write(tmp_path, "data/recorded_replies_manifest.json",
          json.dumps(manifest, indent=2))


_GONE = object()


@pytest.mark.parametrize("field", ["what_approval_means",
                                   "what_repository_head_at_review_means"])
def test_a_version_two_manifest_missing_what_it_promises_is_refused(tmp_path,
                                                                    field):
    """Version two says it explains itself. One that does not, does not."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    _with_a_manifest_where(tmp_path, **{field: "   "})

    problems = release.problems_in(tmp_path)

    assert any(field in problem for problem in problems), problems


def test_a_manifest_with_no_record_of_rewrites_at_all_is_refused(tmp_path):
    """The field has to be there. Absent, nobody knows whether the commits the
    approvals name can still be reached, and the manifest does not say."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    _with_a_manifest_where(tmp_path, history_rewrites=_GONE)

    problems = release.problems_in(tmp_path)

    assert any("history_rewrites" in problem for problem in problems), problems


def test_a_record_of_rewrites_that_is_not_a_list_is_refused(tmp_path):
    """A sentence where a list belongs is a field filled in, not a record."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    _with_a_manifest_where(tmp_path, history_rewrites="none that I know of")

    problems = release.problems_in(tmp_path)

    assert any("history_rewrites" in problem for problem in problems), problems


def test_a_new_repository_may_say_it_has_rewritten_nothing(tmp_path):
    """An empty list is the true answer for the repository being planned."""
    # R-10 and R-11 rewrote this sandbox's history. The public repository is
    # built from one commit and has none, so it has nothing to declare. Written
    # as "must not be empty", the rule made the only honest manifest a new
    # repository could carry the one thing it would refuse, and the fixtures here
    # invented an R-11 entry to get past it.
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    _with_a_manifest_where(tmp_path, history_rewrites=[])

    assert release.problems_in(tmp_path) == []


def test_a_manifest_that_cannot_be_read_is_refused(tmp_path):
    """A damaged manifest is not an empty one, here as everywhere else."""
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json", "{ not json")

    problems = release.problems_in(tmp_path)

    assert any("manifest" in problem for problem in problems), problems


def test_a_manifest_describing_a_file_that_is_not_there_is_refused(tmp_path):
    """The other half of the disagreement."""
    a_release(tmp_path)
    path = with_a_fixture(tmp_path)
    path.unlink()

    problems = release.problems_in(tmp_path)

    assert any("not there" in problem for problem in problems), problems


# The credential-shaped strings below are built rather than written out. A file
# holding something indistinguishable from a live key is worth avoiding in a
# repository whether or not the key is real: a reader cannot tell, a secret
# scanner cannot tell, and this check cannot tell either, so it refused the
# tests that prove it works. Assembling them here keeps the file clean while
# the check still sees exactly the string it would see in earnest.
#
# This file is published with the rest of the tests, so anything left whole
# here is a value the check would find in its own test suite.
A_TOKEN = "gh" + "p_" + "a" * 30
A_PERSONAL_TOKEN = "github" + "_pat_" + "b" * 25
A_KEY = "abc123" + "def456" + "ghi789"
AN_ADDRESS = "somebody@" + "somewhere.co.uk"
A_COOKIE_HEADER = "Set-" + "Cooki" + "e: session" + "id=" + A_KEY
A_SESSION = "ncbi" + "_sid=" + A_KEY
AN_AUTHORIZATION = "authorization: " + "Bear" + "er " + A_KEY
A_PERSONAL_PATH = "rootdir: C:\\Users\\someone\\Documents\\research-agent"


@pytest.mark.parametrize("relative, text, what", [
    ("README.md", "Run it with NCBI_API" + "_KEY=" + A_KEY,
     "a key from the environment file"),
    ("docs/testing.md", "token: " + A_TOKEN, "a GitHub token"),
    ("docs/testing.md", "cloned with " + A_PERSONAL_TOKEN, "a GitHub token"),
    ("README.md", "Written by " + AN_ADDRESS, "an email address"),
    ("docs/testing.md", A_COOKIE_HEADER, "a cookie header"),
    ("docs/testing.md", "Cooki" + "e: " + A_SESSION, "a cookie header"),
    ("docs/testing.md", "the reply carried " + A_SESSION,
     "an NCBI session identifier"),
    ("docs/testing.md", AN_AUTHORIZATION, "an authorization header"),
    ("data/recorded_replies_manifest.json",
     '{"files": {}, "headers": {"' + AN_AUTHORIZATION.replace(": ", '": "')
     + '"}}', "an authorization header"),
    ("README.md", "api" + "_key = " + "A" * 20, "a credential"),
])
def test_a_secret_under_an_unexpected_name_is_refused(tmp_path, relative, text,
                                                      what):
    """Not the environment file, which nothing copies. A key pasted into a
    document is the one that travels, because nobody thinks of it as a key.
    """
    # The cookie and the session identifier are here because they are what this
    # project actually leaked (R-06), and the JSON one because every reply this
    # project saves is JSON: a header written out as a header was the only shape
    # the first version of these patterns could find.
    #
    # What is asserted is the finding, not that something was found. The manifest
    # case makes a broken manifest as well, so a test asking only for problems
    # would pass on those and say nothing about the pattern it is here for.
    a_release(tmp_path)
    write(tmp_path, relative, text)

    problems = release.problems_in(tmp_path)

    assert any(f"holds {what}" in problem for problem in problems), \
        f"{what} was not noticed in {relative}: {problems}"


@pytest.mark.parametrize("text, value", [
    ("Run it with NCBI_API" + "_KEY=" + A_KEY, A_KEY),
    ("token: " + A_TOKEN, A_TOKEN),
    ("Written by " + AN_ADDRESS, AN_ADDRESS),
    (A_COOKIE_HEADER, A_KEY),
    ("the reply carried " + A_SESSION, A_KEY),
    (A_PERSONAL_PATH, "someone"),
])
def test_what_was_found_is_never_repeated_back(text, value):
    """A check meant to prevent a leak must not cause one."""
    # An earlier version printed the matched text to make the report easier to
    # act on, which put the value on the screen, into whatever the terminal saved
    # and into any evidence file the run was captured in. The report names the
    # file and the line, and the person opens the file.
    found = release._secrets_in(text, "docs/testing.md")

    assert found, "nothing was found, so this proves nothing about the report"
    for problem in found:
        assert value not in problem, problem
        assert text not in problem, problem


@pytest.mark.parametrize("text", [
    "write to someone@example.com about it",
    "the address the tests use is a.author@example.org",
    "a service at https://www.ncbi.nlm.nih.gov/home/about/policies/",
    "a path written as an example: C:" + chr(92) + "Users" + chr(92) + "<name>",
])
def test_what_is_written_as_an_example_is_not_refused(tmp_path, text):
    """A check that refused the project's own documents would be turned off."""
    # Each of these was refused when the rules were first written: addresses at
    # the domains reserved for examples, a policy page cited in the remediation
    # log, and a path written to show the shape of a path rather than to name
    # one. A rule that refuses the project's own documents is one somebody turns
    # off.
    a_release(tmp_path)
    write(tmp_path, "README.md", "# The agent\n\n" + text + "\n")

    assert release.problems_in(tmp_path) == []


@pytest.mark.parametrize("path", [
    "C:\\Users\\someone\\Documents\\research-agent",
    # The form Python prints, and the form this project's tooling produces.
    # It is caught by the rule for a home folder rather than the one for a
    # drive letter, which is why the cases are listed here: which alternative
    # does the work is an accident of how the pattern was written, and these
    # four are what must be refused.
    #
    # Inside JSON a path is written the other way, with each separator
    # doubled, and none of these rules sees that. It is found by reading the
    # JSON back, which is tested further down.
    #
    # Built from pieces for the reason given further down this file. The
    # backslash form can be written out whole because the source doubles its
    # separators. These cannot, so written whole they would be real personal
    # paths in a file this project publishes.
    "C:/Users/" + "someone" + "/Documents/research-agent",
    "/home/" + "someone" + "/research-agent",
    "/Users/" + "someone" + "/research-agent",
])
def test_a_path_naming_somebody_s_machine_is_refused(tmp_path, path):
    """Test output run from a home folder says who ran it and where."""
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md", f"rootdir: {path}\n")

    problems = release.problems_in(tmp_path)

    assert any("own machine" in problem for problem in problems), problems


# Built from pieces, as the keys above are: this file travels in the release
# and the rule reads it, so written whole the names would refuse this test.
THE_NAMES = ["Cla" + "ude", "co" + "dex", "The " + "owner"]


@pytest.mark.parametrize("name", THE_NAMES)
def test_a_name_outside_the_readme_s_statement_is_refused(tmp_path, name):
    """The release names nobody outside the README's statement on the use of
    generative AI, decided on 19 September. Any case is refused, so a sentence
    starting with the name is caught as well as one in the middle."""
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md", f"# Test plan\n\nRead by {name} on the 16th.\n")

    problems = release.problems_in(tmp_path)

    assert [problem for problem in problems if "names" in problem] == [
        (f"docs/testing.md, line 3: names \"{name.lower()}\", which a release "
         f"names only in the README's statement on the use of generative AI")]


def test_the_readme_s_statement_may_name_the_assistants(tmp_path):
    """The one place the brief asks the tools to be acknowledged."""
    # The statement runs from its heading to the next heading, and the rest of
    # the README is read like any other file, so a name above or below the
    # statement is still refused.
    first, second = THE_NAMES[0], THE_NAMES[1]
    a_release(tmp_path)
    write(tmp_path, "README.md",
          "# The agent\n\nHow to install and run it.\n\n## Use of generative AI\n\n"
          f"{first} was used to proofread; {second} reviewed.\n\n## Licence\n\nMIT.\n")
    assert release.problems_in(tmp_path) == []

    write(tmp_path, "README.md",
          f"# The agent\n\n## Use of generative AI\n\n{first} was used.\n\n"
          f"## Licence\n\nReviewed by {second}.\n")
    problems = release.problems_in(tmp_path)

    assert [problem for problem in problems if "names" in problem] == [
        (f"README.md, line 9: names \"{second}\", which a release names only in "
         f"the README's statement on the use of generative AI")]


def test_a_name_broken_across_two_lines_is_still_found(tmp_path):
    """Wrapped prose puts the two words of the third name on two lines."""
    # Read line by line on 19 September, the check passed four docstrings
    # written that way. The phrase is matched with any white space between its
    # words, and the line reported is the one the first word is on.
    first, second = THE_NAMES[2].split()
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md",
          f"# Test plan\n\nThe screen was read by {first}\n{second} on the 16th.\n")

    problems = release.problems_in(tmp_path)

    assert [problem for problem in problems if "names" in problem] == [
        (f"docs/testing.md, line 3: names \"{THE_NAMES[2].lower()}\", which a "
         f"release names only in the README's statement on the use of generative AI")]


def test_the_files_a_release_carries_name_nobody_and_keep_dates_only():
    """The rule of 19 September read against this repository's own files, not
    a sandbox.
    """
    # The sandbox tests show the two readers work; this shows the tree passes
    # them, which is what the release depends on and what the cutting pieces of
    # 19 September were for. Every text file a release would carry is read: the
    # exact paths this check can read, and the code and tests the patterns
    # allow. A file named before it exists, the final test output, is skipped
    # until it does.
    root = pathlib.Path(__file__).resolve().parents[2]
    carried = []
    for relative in sorted(release.ALLOWED_EXACT):
        readable = pathlib.Path(relative).suffix in release.READABLE
        if readable and (root / relative).is_file():
            carried.append(relative)
    for pattern in release.ALLOWED_PATTERNS:
        for path in sorted(root.glob(pattern)):
            carried.append(path.relative_to(root).as_posix())

    problems = []
    for relative in carried:
        text = (root / relative).read_text(encoding="utf-8-sig")
        problems += release._names_in(text, relative)
        problems += release._clock_times_in(text, relative)

    assert len(carried) > 50, "the release carries more files than this"
    assert problems == []


def test_the_deduplication_code_s_body_owner_is_not_a_name(tmp_path):
    """The third name is matched as a phrase: the code's "body owner", the
    record whose text is kept when duplicates are combined, is not it."""
    a_release(tmp_path)
    write(tmp_path, "research_agent/dedup.py",
          "# The body owner's abstract is kept.\nbody_owner = None\n")

    assert release.problems_in(tmp_path) == []


def test_a_clock_time_in_a_document_is_refused_and_one_in_evidence_is_not(tmp_path):
    """The documents carry dates only, decided on 19 September; the evidence
    files are the machine's record and keep their times. A ratio written with
    a full stop, 12.31 to 1, is not a time."""
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md", "# Test plan\n\nBody text 12.31 to 1.\n")
    assert release.problems_in(tmp_path) == []

    write(tmp_path, "docs/testing.md", "# Test plan\n\nRun started at 09:41 local.\n")
    problems = release.problems_in(tmp_path)
    assert [problem for problem in problems if "clock" in problem] == [
        ("docs/testing.md, line 3: holds a clock time, and the documents of a "
         "release carry dates only")]

    write(tmp_path, "docs/testing.md", "# Test plan\n")
    write(tmp_path, "evidence/execution/live-run-2026-09-18/README.md",
          "# The run\n\n| 8 | 18 Sep, 19:01 to 20:04 |\n")
    assert release.problems_in(tmp_path) == []


def test_a_run_log_is_read_line_by_line(tmp_path):
    """The run log is one JSON record per line, so the whole file is not JSON;
    each line gets the reading a JSON file gets: the secret patterns on the
    decoded fields and the check for a field holding a paper's text."""
    a_release(tmp_path)
    log = "evidence/execution/live-run-2026-09-18/run_98af89fa5cdc.jsonl"
    write(tmp_path, log,
          '{"event": "run started", "run": "98af89fa5cdc"}\n'
          '{"event": "search", "query": "drug X"}\n')
    assert release.problems_in(tmp_path) == []

    write(tmp_path, log,
          '{"event": "run started"}\n'
          '{"event": "paper", "abstract": "Forty patients were given drug X."}\n'
          'not json\n')
    problems = release.problems_in(tmp_path)

    assert f"{log}, line 2: has a field holding the text of a paper (abstract)" \
        in problems
    assert f"{log}, line 3: is not readable JSON" in problems


@pytest.mark.parametrize("text, what", [
    ('"set-' + 'cooki' + 'e": ["session' + 'id=' + A_KEY + '"]',
     "a cookie header"),
    ('"authorization": ["' + 'Bear' + 'er ' + A_KEY + '"]',
     "an authorization header"),
])
def test_a_header_saved_as_a_list_of_values_is_still_found(tmp_path, text, what):
    """A service may return a header's values as a list."""
    # This project does not save one that way, and the patterns are what would
    # have to notice if it ever did. Without the bracket they matched a header
    # written out as a header and a header saved as a plain field, and stopped
    # one character short of the third shape.
    #
    # Asserting the finding rather than that something was found. Asked only
    # whether there were problems, this passed on any refusal at all and would
    # have gone on passing with the pattern removed.
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md", text)

    problems = release.problems_in(tmp_path)

    assert any(f"holds {what}" in problem for problem in problems), problems


def test_a_kind_of_file_this_check_cannot_read_is_refused(tmp_path):
    """An archive is not safe because nothing could look inside it."""
    # Skipping a file nothing can read would let it pass unchecked, which is the
    # same mistake the publication check made with replies it could not
    # inspect.
    #
    # It is put in the replay set because that is the one folder allowed as a
    # whole, so it is the only place a file of an unexpected kind can reach. Put
    # anywhere else it is refused one step earlier, for being a path nothing
    # allows, which is also correct and is tested above.
    a_release(tmp_path)
    path = tmp_path / "data" / "recorded_replies" / "run.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04 not really a zip")

    problems = release.problems_in(tmp_path)

    assert any("not one" in problem or "cannot read" in problem
               for problem in problems), problems


def test_an_archive_anywhere_else_is_refused_for_not_being_allowed(tmp_path):
    """The step before: nothing allows it, so its kind never comes up."""
    a_release(tmp_path)
    path = tmp_path / "evidence" / "run.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK\x03\x04 not really a zip")

    problems = release.problems_in(tmp_path)

    assert any("nothing allows it" in problem for problem in problems), problems


def test_a_file_that_is_allowed_but_unreadable_is_refused(tmp_path):
    """Allowed by name, and not text. The name is not the contents."""
    a_release(tmp_path)
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "testing.md").write_bytes(b"\xff\xfe\x00 binary")

    problems = release.problems_in(tmp_path)

    assert any("could not be read" in problem for problem in problems), problems


@pytest.mark.skipif(sys.platform == "win32",
                    reason="making a link needs a privilege Windows does not "
                           "give by default, so the case is checked directly")
def test_a_link_pointing_outside_the_export_is_refused(tmp_path):
    """A link would have this check read one file and publish another."""
    export = tmp_path / "export"
    a_release(export)
    outside = tmp_path / "secrets.txt"
    outside.write_text("NCBI_API" + "_KEY=" + A_KEY, encoding="utf-8")
    (export / "README.md").unlink()
    (export / "README.md").symlink_to(outside)

    problems = release.problems_in(export)

    assert any("link" in problem for problem in problems), problems


def test_a_repositorys_own_git_folder_is_not_read(tmp_path):
    """A deliberate limit, pinned so that it stays deliberate."""
    # The same checks are meant to be run against a fresh clone of the published
    # repository, and a clone carries the objects git wrote. Read as files, every
    # one of them would be reported as something nothing allows, several hundred
    # of them, and the report would be useless.
    #
    # What this does not say is anything about the published history. That is a
    # different question, and this project needed two rewrites to answer it.
    a_release(tmp_path)
    write(tmp_path, ".git/config", "[core]\n\trepositoryformatversion = 0\n")
    write(tmp_path, ".git/objects/ab/cdef", "not text and not allowed anywhere")

    assert release.problems_in(tmp_path) == []


def test_a_path_that_escapes_the_export_is_refused_by_the_same_rule(tmp_path):
    """The rule the test above needs a privilege to exercise, asked directly."""
    # Windows does not let an ordinary account make a symbolic link, so the case
    # that matters is checked here rather than left untested on the machine this
    # project is built on.
    export = tmp_path / "export"
    a_release(export)
    outside = tmp_path / "outside.txt"
    outside.write_text("anything", encoding="utf-8")

    assert release._escapes(export, outside)
    assert not release._escapes(export, export / "README.md")



@pytest.mark.parametrize("relative", [
    "private/research_agent/agents/leak.py",
    "private/research_agent/leak.py",
    "private/tests/unit/test_leak.py",
    "private/tests/test_leak.py",
    "private/scripts/leak.py",
    "docs/research_agent/agents/leak.py",
    "anything/at/all/scripts/leak.py",
])
def test_a_folder_shaped_like_the_application_is_not_the_application(tmp_path,
                                                                     relative):
    """The patterns are matched from the root, not from the end of the path."""
    # PurePosixPath.match compares a relative pattern against the last parts of a
    # path, so research_agent/*/*.py said yes to
    # private/research_agent/agents/leak.py, and so did a copy under docs/. A
    # folder somebody had set aside could carry anything out of the repository by
    # being shaped like the application, and because the same rule chooses what
    # to copy and judges it afterwards, both steps agreed it belonged.
    a_release(tmp_path)
    write(tmp_path, relative, "SECRET = 'never reviewed'\n")

    problems = release.problems_in(tmp_path)

    assert any("nothing allows it" in problem for problem in problems), problems


@pytest.mark.parametrize("relative", [
    "research_agent/graph.py",
    "research_agent/sources/crossref.py",
    "tests/conftest.py",
    "tests/unit/test_something.py",
    "scripts/approve_replies.py",
])
def test_the_application_itself_is_still_allowed(tmp_path, relative):
    """The other half. A rule that refused the code would stop any release."""
    a_release(tmp_path)
    write(tmp_path, relative, "VALUE = 1\n")

    assert release.problems_in(tmp_path) == []


@pytest.mark.parametrize("shaped_like", [
    "NOTICE.md",
    "docs/testing.md",
    "docs/development.md",
    "evidence/testing/final_test_output.txt",
])
def test_a_folder_given_a_required_file_s_name_is_not_that_file(tmp_path,
                                                                shaped_like):
    """Asked whether the path existed, the completeness check took a folder."""
    # Four empty folders, named NOTICE.md and research_agent/fake.py and two more
    # named like tests, made an empty export report as a finished submission. The
    # safety check did not notice either, because it walks files and steps over
    # ordinary folders. Neither check saw a folder given a file's name.
    a_release(tmp_path)
    (tmp_path / shaped_like).mkdir(parents=True, exist_ok=True)

    missing = release.what_a_release_is_missing(tmp_path)

    assert any(shaped_like in line for line in missing), missing


def test_an_export_of_empty_folders_is_not_a_finished_submission(tmp_path):
    """All four folders at once, which is how this was found."""
    for relative in release.REQUIRED_FOR_THIS_RELEASE:
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    for relative in ("research_agent/fake.py", "research_agent/other.py",
                     "tests/unit/test_fake.py", "tests/integration/test_fake.py",
                     "data/recorded_replies/fake.json"):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)

    missing = release.what_a_release_is_missing(tmp_path)

    assert len(missing) >= len(release.REQUIRED_FOR_THIS_RELEASE), missing
    assert any("the application itself" in line for line in missing), missing
    assert any("tests/unit" in line for line in missing), missing
    assert any("tests/integration" in line for line in missing), missing
    assert any("recorded_replies" in line for line in missing), missing


def _everything_a_release_needs_except(root, left_out):
    """Writes every required file but one, each empty, under that folder."""
    # Empty files are enough here: the completeness check asks whether the path
    # is a file, and what the file says is a person's reading. Writing the real
    # pictures would make the test about their bytes instead.
    for relative in release.REQUIRED_FOR_THIS_RELEASE:
        if relative != left_out:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    return root


@pytest.mark.parametrize("left_out", [".env.example", "requirements-dev.txt"])
def test_a_release_without_what_the_readme_names_is_not_finished(tmp_path,
                                                                 left_out):
    """The README tells a reader to use both, and nothing asked for either."""
    # It says to install the test dependencies from requirements-dev.txt and to
    # copy .env.example to .env before running. Both were files a release was
    # allowed to carry and never asked for, so an export that lost one passed
    # this check while the installation and testing instructions in its own
    # README named a file it did not hold.
    _everything_a_release_needs_except(tmp_path, left_out)

    missing = release.what_a_release_is_missing(tmp_path)

    assert any(line.startswith(left_out) for line in missing), missing


def test_the_record_of_execution_is_required_of_a_release(tmp_path):
    """The brief asks for evidence of execution and nothing asked for it."""
    # The README and the testing page name the live run's pictures, the three
    # early live checks and the notice on the search page. The two lists named
    # them too, but only as files a release may carry, so an export that lost
    # them passed the completeness check while both documents went on citing
    # them.
    #
    # Everything else the project requires is written here as an empty file, so
    # what comes back is the record of execution and nothing else. The record is
    # taken from the allowed list, the same place the requirement is, because a
    # second list written by hand is what this change exists to avoid.
    for relative in release.REQUIRED_FOR_THIS_RELEASE:
        if not relative.startswith("evidence/execution/"):
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    missing = release.what_a_release_is_missing(tmp_path)

    expected = set()
    for relative in release.ALLOWED_EXACT:
        if relative.startswith("evidence/execution/"):
            expected.add(relative)
    reported = set()
    for line in missing:
        if line.startswith("evidence/execution/"):
            reported.add(line.split(":")[0])

    assert reported == expected, missing


def test_a_path_written_the_way_json_writes_one_is_found(tmp_path):
    """JSON doubles every separator, so the rule for a path never sees one."""
    # What the file holds is two backslashes where the rule looks for one. It is
    # found by reading the JSON back and scanning what it decodes to, which is
    # the same pass that finds a field name spelled out in escapes.
    a_release(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json",
          json.dumps({"files": {}, "ran_in": "C:" + chr(92) + "Users"
                      + chr(92) + "someone" + chr(92) + "research-agent"}))

    problems = release.problems_in(tmp_path)

    assert any("own machine" in problem for problem in problems), problems


def test_a_field_name_written_in_escapes_is_found(tmp_path):
    """A rule matching bytes reads escapes as escapes. A reader reads the word."""
    # The name here is spelled out one character at a time so that the file does
    # not hold it, which is exactly what a leak written by a program that escapes
    # everything would look like.
    a_release(tmp_path)
    escaped = "".join(f"{chr(92)}u{ord(letter):04x}"
                      for letter in "authorization")
    write(tmp_path, "data/recorded_replies_manifest.json",
          '{"files": {}, "' + escaped + '": "' + "Bear" + "er " + A_KEY + '"}')

    problems = release.problems_in(tmp_path)

    assert any("holds an authorization header" in problem
               for problem in problems), problems


def test_a_credential_name_does_not_reach_across_a_line_ending(tmp_path):
    r"""Written with \s the value could sit on the line after the name."""
    # A document ending a sentence with "password:" was then read as giving
    # whatever word began the next line, which is a refusal somebody works
    # around rather than acts on.
    a_release(tmp_path)
    write(tmp_path, "docs/testing.md",
          "The service asks for a password:\n" + "A" * 20 + " is the answer\n")

    assert release.problems_in(tmp_path) == []


def _escaped(word):
    """A word written the way JSON is allowed to write it, one escape a letter."""
    # The file then does not hold the word at all, which is what a leak written
    # by something that escapes everything looks like, and what a rule matching
    # on the bytes cannot see.
    return "".join(f"{chr(92)}u{ord(letter):04x}" for letter in word)


@pytest.mark.parametrize("body, what", [
    # The name spelled out in escapes, its value moved into a list. Each half
    # on its own got past a different check: the escapes past the scan of the
    # bytes, the list past the pairing of a name with its value.
    ('{"' + _escaped("authorization") + '": ["' + "Bear" + "er " + A_KEY + '"]}',
     "an authorization header"),
    ('{"' + _escaped("set-cookie") + '": ["session' + 'id=' + A_KEY + '"]}',
     "a cookie header"),
    # A session identifier recorded as a number. Only strings were read, so
    # this was the one shape nothing looked at.
    ('{"' + _escaped("ncbi_sid") + '": 1234567}',
     "an NCBI session identifier"),
    # Nested, escaped and in a list at once. The name is paired with the
    # values directly under it, as a header maps to what it carries. A value
    # under some further name is read under that name as well.
    ('{"headers": {"' + _escaped("authorization") + '": ["'
     + "Bear" + "er " + A_KEY + '"]}}', "an authorization header"),
])
def test_a_secret_hidden_in_the_shape_of_the_json_is_still_found(tmp_path, body,
                                                                 what):
    """Read as bytes and read again as what it decodes to."""
    a_release(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json", body)

    problems = release.problems_in(tmp_path)

    assert any(f"holds {what}" in problem for problem in problems), problems


@pytest.mark.parametrize("body", [
    '{"' + A_TOKEN + '": "anything"}',
    '{"' + A_TOKEN + '": ["anything"]}',
    '{"' + A_TOKEN + '": {"inside": "anything"}}',
    '{"records": [{"' + A_TOKEN + '": 1}]}',
])
def test_a_secret_used_as_a_field_name_is_found_and_never_quoted(tmp_path, body):
    """A key is a place a credential can sit, and one nobody looks at."""
    # Saying which field it was put the field's name in the report, so where the
    # name was the credential the report printed it. Where something was found is
    # given by counting fields instead.
    a_release(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json", body)

    problems = release.problems_in(tmp_path)

    assert any("holds a GitHub token" in problem for problem in problems), \
        problems
    for problem in problems:
        assert A_TOKEN not in problem, problem


def test_where_something_was_found_is_said_by_counting(tmp_path):
    """The report has to be usable as well as safe."""
    # Positions are what it gives, so a person can count to the field rather than
    # read its name back out of a report that should not hold it.
    found = release._secrets_in_the_decoded_json(
        {"status": 200, "headers": {"authorization": ["Bear" + "er " + A_KEY]}},
        "x.json")

    assert any("item 1 of field 1 of field 2 of the top level" in problem
               for problem in found), found


@pytest.mark.parametrize("value", ['{"inside": "anything"}', '{}', '[]'])
def test_an_escaped_secret_in_a_name_over_a_container_is_still_found(tmp_path,
                                                                     value):
    """The case that needs the names to be read on their own."""
    # A plain credential in a field name is caught by the scan of the bytes,
    # because the file holds it. Written in escapes it is not, and where its
    # value is an object or a list there is no name-and-value pair to read
    # either. Reading every name is the only thing that sees this one, and until
    # this test existed nothing could tell whether that reading was happening.
    a_release(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json",
          '{"' + _escaped(A_TOKEN) + '": ' + value + '}')

    problems = release.problems_in(tmp_path)

    assert any("holds a GitHub token" in problem for problem in problems), \
        problems
    for problem in problems:
        assert A_TOKEN not in problem, problem


@pytest.mark.parametrize("shape", [
    '{"values": ["' + "Bear" + "er " + A_KEY + '"]}',
    '{"a": {"b": ["' + "Bear" + "er " + A_KEY + '"]}}',
    '{"value": "' + "Bear" + "er " + A_KEY + '"}',
    '[["' + "Bear" + "er " + A_KEY + '"]]',
])
@pytest.mark.parametrize("name", ["authorization", None])
def test_a_value_is_read_under_every_name_it_sits_below(tmp_path, shape, name):
    """Which container somebody puts it in is not something to depend on."""
    # Only the nearest name was carried down, so a value one container further
    # down was read under whatever that container happened to be called:
    # authorization holding an object holding a list got past this, while the
    # same value directly under the same name did not.
    #
    # The name is written in escapes in half of these, so the scan of the file as
    # it is written cannot see it either and the pairing is the only thing left.
    a_release(tmp_path)
    written = name if name is None else _escaped(name)
    write(tmp_path, "data/recorded_replies_manifest.json",
          '{"' + (written or "authorization") + '": ' + shape + '}')

    problems = release.problems_in(tmp_path)

    assert any("holds an authorization header" in problem
               for problem in problems), problems


@pytest.mark.parametrize("opened, closed, deep", [
    # Deep enough that reading the decoded file runs out of room, and shallow
    # enough that json itself still parses it.
    ("[", "]", 2000),
    ('{"a": ', "}", 2000),
    # Deep enough that json runs out of room parsing it, which is a different
    # place for the same failure and needs its own answer.
    ("[", "]", 20000),
], ids=["lists", "objects", "too deep to parse"])
def test_a_file_nested_deeper_than_python_will_follow_is_refused(tmp_path,
                                                                 opened, closed,
                                                                 deep):
    """A JSON file is read a level at a time, twice, and so is parsed that way."""
    # A file two thousand containers deep raised an error nothing caught, and the
    # whole check stopped with a traceback rather than saying which file was
    # wrong. A file nothing can read all the way through is refused, which is the
    # same answer this check gives to any other file it cannot inspect.
    a_release(tmp_path)
    write(tmp_path, "data/recorded_replies_manifest.json",
          opened * deep + "1" + closed * deep)

    problems = release.problems_in(tmp_path)

    assert any("nested too deeply" in problem for problem in problems), problems


def test_a_manifest_that_is_not_text_is_reported_not_raised(tmp_path):
    """The same answer a manifest that will not parse gets."""
    # Reading a file raises UnicodeDecodeError when its bytes are not text, and
    # that was not among the things read_manifest answered for, so a damaged
    # manifest stopped the check instead of being reported by it.
    a_release(tmp_path)
    with_a_fixture(tmp_path)
    (tmp_path / "data" / "recorded_replies_manifest.json").write_bytes(
        b'{"files": "\xff\xfe not utf-8"}')

    problems = release.problems_in(tmp_path)

    assert any("manifest" in problem for problem in problems), problems
