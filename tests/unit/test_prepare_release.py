"""Checks the command that builds a release, not only the rule that judges one."""
# The rule is tested next door. What is tested here is the part only the command
# does, and it is the part that went wrong first: which tree the files come from.
#
# An earlier version read the working directory. Anything modified, and anything
# not committed at all, travelled with the release. This project has twice
# published something nobody had looked at, and both times the file was one that
# happened to be on disk. Reading a commit rather than a folder is the
# difference the command exists to make.
#
# Each test builds a small repository of its own. Nothing reads or writes the
# real one, and the tests run git rather than pretending to: what is being
# checked is that the export comes out of a commit, and a stand-in for git could
# not show that.

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "prepare_release.py")


def git(repo, *arguments):
    """Runs git in a repository built for a test."""
    return subprocess.run(["git", "-C", str(repo), *arguments],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture
def exporter():
    """The command as it stands on disk, loaded from its path."""
    spec = importlib.util.spec_from_file_location("prepare_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repository(tmp_path, exporter, monkeypatch):
    """A small repository holding one commit that would pass the check."""
    repo = tmp_path / "sandbox"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "somebody@example.com")
    git(repo, "config", "user.name", "Somebody")

    write(repo, "README.md", "# The agent\n\nHow to install and run it.\n")
    write(repo, "LICENSE", "MIT License\n")
    write(repo, "requirements.txt", "pydantic==2.13.4\n")
    write(repo, "pyproject.toml", "[project]\nname = 'research-agent'\n")
    write(repo, "research_agent/__init__.py", "")
    write(repo, "tests/unit/test_something.py", "def test_it():\n    assert True\n")

    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "The first commit")

    monkeypatch.setattr(exporter, "REPO", repo)
    return repo


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


SAMPLE = (pathlib.Path(__file__).resolve().parents[2]
          / "data" / "hand_screened_sample")

EVIDENCE = (pathlib.Path(__file__).resolve().parents[2]
            / "evidence" / "execution")


def with_the_execution_evidence(root):
    """Adds the real record of execution, which a release now requires."""
    # Copied rather than invented, for the same reason as the screening files.
    # The check holds a fingerprint of every picture it allows, so an invented
    # screenshot is refused, and the test would then fail for a reason that has
    # nothing to do with what it asks about.
    for path in sorted(EVIDENCE.rglob("*")):
        if path.is_file():
            target = root / "evidence" / "execution" / path.relative_to(EVIDENCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return root


def _a_fixture_and_its_manifest():
    """One synthetic reply and the manifest describing where it came from."""
    reply = json.dumps({
        "captured_at": "2026-09-01T10:00:00+00:00",
        "method": "GET",
        "url": "https://api.crossref.org/works?query=anything",
        "asked": {"query": "anything"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"message": {"items": [{"title": ["An invented paper"]}]}}),
    }, indent=2)
    fingerprint = hashlib.sha256(reply.encode("utf-8")).hexdigest()
    manifest = json.dumps({
        "manifest_version": 2,
        "what_approval_means": "somebody accepted this file as it stood",
        "what_repository_head_at_review_means": "where the repository stood",
        # Empty, because a repository whose first commit is this export has no
        # earlier history to describe. R-10 and R-11 rewrote the working
        # repository's history, not this one's.
        "history_rewrites": [],
        "files": {"a_fixture.json": {
            "sha256": fingerprint,
            "approval_history": [{
                "approved_at": "2026-09-03T10:00:00+00:00",
                "approved_sha256": fingerprint,
                "repository_head_at_review": "a" * 40,
                "working_tree_at_review": "nothing uncommitted",
                "fixture_origin": {
                    "written_by": "the person who built this release",
                    "written_on": "2026-09-03",
                    "what_it_stands_in_for": "a Crossref search",
                    "copied_from_a_paper": False,
                    "applies_to_sha256": fingerprint,
                },
            }],
        }},
    }, indent=2)
    return reply, manifest


_REPLY, _MANIFEST = _a_fixture_and_its_manifest()

# What has to be in a commit for an export of it to be a finished submission.
# Written out rather than copied from the real repository, so that a change
# there cannot quietly change what these tests mean. The two screening files
# are the exception: the check compares them with the fingerprint taken when
# they were reduced, so anything invented would be refused, correctly. The
# record of execution is the second exception, copied by the function above
# for the same reason.
COMPLETE_ENOUGH = {
    "NOTICE.md": "# Notice\n\nWhat the licence covers.\n",
    ".gitattributes": "* text=auto eol=lf\n",
    # Required since 23 September, because the README's own instructions name
    # both: the test dependencies come from the first and the second is what
    # the reader copies to .env before running. The key name here is left
    # without a value, which is what the real example file does and what the
    # rule on keys expects to see.
    "requirements-dev.txt": "pytest==9.1.1\n",
    ".env.example": "# Copy this to .env and fill it in.\nGEMINI_API_KEY=\n",
    # The plan, the architecture decisions and the data boundaries were
    # required until 19 September, when the release was cut to the ten
    # documents a marker needs; they stay in the repository unexported. The
    # test plan and the functional test report were required until 22
    # September, when the testing page took in every row of them. On 23
    # September the matrix, the account of the design, the development log and
    # the remediation log became this one page, and are kept back in their turn.
    "docs/development.md": "# Development\n",
    # Required since 14 September, when the tutor's feedback asked for them
    # (TF-01, TF-02). Added here by hand for the reason the comment above
    # gives.
    "docs/class_diagram.md": "# Class diagram\n",
    "docs/sequence_diagram.md": "# Sequence diagram\n",
    # Required since 19 September: the one page on testing asked for as the
    # marker's first stop.
    "docs/testing.md": "# Testing: what was planned, what was run, what it found\n",
    "data/hand_screened_sample/README.md": "# The hand-screened sample\n",
    "data/hand_screened_sample/screening_labels.csv":
        (SAMPLE / "screening_labels.csv").read_text(encoding="utf-8-sig"),
    "data/hand_screened_sample/screening_labels_check.csv":
        (SAMPLE / "screening_labels_check.csv").read_text(encoding="utf-8-sig"),
    "evidence/testing/final_test_output.txt": "411 passed\n",
    "research_agent/graph.py": "def build():\n    return None\n",
    "research_agent/schemas.py": "class Request:\n    pass\n",
    "tests/integration/test_together.py": "def test_it():\n    assert True\n",
    "data/recorded_replies/a_fixture.json": _REPLY,
    "data/recorded_replies_manifest.json": _MANIFEST,
}


def test_a_release_is_built_from_a_commit_and_passes(exporter, repository,
                                                     tmp_path):
    """The ordinary case, and the one everything below is measured against."""
    destination = tmp_path / "export"

    assert exporter.main.__name__ == "main"
    assert exporter.why_the_commit_cannot_be_exported("HEAD") is None

    written = exporter.copy_out_of(exporter.commit_identifier("HEAD"), destination)

    assert "README.md" in written
    assert (destination / "README.md").exists()


def test_a_file_that_was_never_committed_cannot_travel(exporter, repository,
                                                       tmp_path):
    """The failure the move to a commit was for."""
    # A file left by a probe, a draft, a note to self: none of it is in the
    # commit, so none of it is in the release. Reading the folder, all of it
    # would have been.
    write(repository, "research_agent/left_behind.py", "SECRET = 'not reviewed'\n")
    destination = tmp_path / "export"

    written = exporter.copy_out_of(exporter.commit_identifier("HEAD"), destination)

    assert "research_agent/left_behind.py" not in written
    assert not (destination / "research_agent" / "left_behind.py").exists()


def test_a_change_that_was_not_committed_does_not_travel(exporter, repository,
                                                         tmp_path):
    """What is exported is the reviewed version, not the edited one."""
    write(repository, "README.md", "# Edited after the commit\n")
    destination = tmp_path / "export"

    exporter.copy_out_of(exporter.commit_identifier("HEAD"), destination)

    assert "Edited after the commit" not in \
        (destination / "README.md").read_text(encoding="utf-8")


def test_a_dirty_working_tree_stops_the_export(exporter, repository):
    """The commit is what travels, but a dirty tree means the commit is not
    what anybody is looking at, and they would assume it was."""
    write(repository, "README.md", "# Changed and not committed\n")

    wrong = exporter.why_the_commit_cannot_be_exported("HEAD")

    assert wrong and "not committed" in wrong


def test_a_commit_git_does_not_know_stops_the_export(exporter, repository):
    """A tag that does not exist is a release built from nothing."""
    wrong = exporter.why_the_commit_cannot_be_exported("no-such-tag")

    assert wrong and "does not know" in wrong


def test_a_tag_is_accepted_and_resolved_to_its_commit(exporter, repository):
    """Releases are built from a tag, and the record names the commit."""
    git(repository, "tag", "unit11-submission-v1.0")

    assert exporter.why_the_commit_cannot_be_exported("unit11-submission-v1.0") is None
    identifier = exporter.commit_identifier("unit11-submission-v1.0")
    assert len(identifier) == 40
    assert identifier == exporter.commit_identifier("HEAD")


def test_a_destination_inside_the_repository_is_refused(exporter, repository):
    """A release built inside the thing being released joins the next one."""
    wrong = exporter.why_the_destination_will_not_do(repository / "evidence" / "out")

    assert wrong and "inside the repository" in wrong


def test_a_destination_that_is_the_repository_is_refused(exporter, repository):
    """The same mistake, written more directly."""
    assert exporter.why_the_destination_will_not_do(repository)


def test_a_destination_that_is_not_empty_is_refused(exporter, repository,
                                                    tmp_path):
    """Whatever was in the folder would become part of the release."""
    destination = tmp_path / "export"
    destination.mkdir()
    write(destination, "something_already_here.txt", "from an earlier run\n")

    wrong = exporter.why_the_destination_will_not_do(destination)

    assert wrong and "not empty" in wrong


def test_an_empty_folder_that_does_not_exist_yet_is_accepted(exporter,
                                                             repository,
                                                             tmp_path):
    """A rule that refused every destination would stop a release being made."""
    assert exporter.why_the_destination_will_not_do(tmp_path / "not_yet") is None


def test_the_inventory_records_what_was_written_and_from_where(exporter,
                                                               repository,
                                                               tmp_path):
    """What the first public commit and a fresh clone are checked against."""
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)
    identifier = exporter.commit_identifier("HEAD")
    written = exporter.copy_out_of(identifier, destination)
    inventory = exporter.inventory_of(identifier, written, True)

    assert exporter.write_exclusively(
        record, json.dumps(inventory, indent=2)) is None

    assert not (destination / "release_inventory.json").exists(), \
        "the record of the export belongs beside it, not inside it"
    assert record.exists()

    recorded = json.loads(record.read_text(encoding="utf-8"))
    assert recorded["exported_from"] == identifier
    assert recorded["demonstration_required"] is True
    assert set(recorded["files"]) == set(written)


def _an_export(exporter, tmp_path):
    """An export and its record, ready to be checked against each other."""
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)
    identifier = exporter.commit_identifier("HEAD")
    written = exporter.copy_out_of(identifier, destination)
    exporter.write_exclusively(
        record, json.dumps(exporter.inventory_of(identifier, written, True),
                           indent=2))
    return record, destination


def test_an_untouched_export_matches_its_record(exporter, repository, tmp_path):
    """The baseline the rest of these are measured against."""
    record, destination = _an_export(exporter, tmp_path)

    assert exporter.differences_from_inventory(record, destination) == []


def test_a_changed_file_is_noticed_against_the_record(exporter, repository,
                                                      tmp_path):
    """What the record is for: a file edited between the export and the first
    commit, or between the commit and a clone."""
    record, destination = _an_export(exporter, tmp_path)
    write(destination, "README.md", "# Changed after the export\n")

    differences = exporter.differences_from_inventory(record, destination)

    assert any("not the file that was exported" in line for line in differences)


def test_a_deleted_file_is_noticed_against_the_record(exporter, repository,
                                                      tmp_path):
    """A release missing something it was supposed to carry."""
    record, destination = _an_export(exporter, tmp_path)
    (destination / "README.md").unlink()

    differences = exporter.differences_from_inventory(record, destination)

    assert any("was exported and is not here" in line for line in differences)


def test_a_file_that_was_never_exported_is_noticed(exporter, repository,
                                                   tmp_path):
    """The direction that matters most: something added after the check."""
    # Everything in the export was read and allowed. A file appearing afterwards
    # was not, and the record is the only thing that would say so.
    record, destination = _an_export(exporter, tmp_path)
    write(destination, "notes.md", "an abstract, pasted in after the check\n")

    differences = exporter.differences_from_inventory(record, destination)

    assert any("is here and was not exported" in line for line in differences)


def test_a_record_named_for_a_folder_with_another_name_is_still_read(
        exporter, repository, tmp_path):
    """A clone is called whatever the person cloning it chose."""
    # The record used to be found by adding a suffix to the folder's name, so the
    # check could only be run where the export was built, which is the one place
    # nobody needed it. It is named on the command line instead.
    record, destination = _an_export(exporter, tmp_path)
    clone = tmp_path / "somebody_elses_clone"
    destination.replace(clone)

    assert exporter.differences_from_inventory(record, clone) == []


_A_COMMIT = '"exported_from": "' + "a" * 40 + '"'
_ASKED = '"demonstration_required": true'


@pytest.mark.parametrize("written, expected", [
    ("{ not json at all", "could not be read"),
    ('["a", "list"]', "not a record"),
    ('{' + _ASKED + ', "files": {"README.md": "' + "a" * 64 + '"}}',
     "which commit"),
    ('{"exported_from": "HEAD", ' + _ASKED + ', "files": {}}', "which commit"),
    ('{' + _A_COMMIT + ', "files": {}}', "whether the export was required"),
    ('{' + _A_COMMIT + ', "demonstration_required": "yes", "files": {}}',
     "whether the export was required"),
    ('{' + _A_COMMIT + ', ' + _ASKED + ', "files": []}', "does not list"),
    ('{' + _A_COMMIT + ', ' + _ASKED + ', "files": {}}', "does not list"),
    ('{' + _A_COMMIT + ', ' + _ASKED + ', "files": {"README.md": "ok"}}',
     "does not give a fingerprint"),
])
def test_a_record_that_is_not_a_record_is_reported_not_believed(
        exporter, repository, tmp_path, written, expected):
    """A record read without being checked is a record that agrees with
    anything. One whose files are a list matches nothing and disagrees with
    nothing, so a folder checked against it comes back as exactly right."""
    record, destination = _an_export(exporter, tmp_path)
    record.write_text(written, encoding="utf-8")

    differences = exporter.differences_from_inventory(record, destination)

    assert any(expected in line for line in differences), differences


def test_a_folder_that_is_not_there_is_reported(exporter, repository, tmp_path):
    """Nothing to compare is not the same as nothing different."""
    record, _ = _an_export(exporter, tmp_path)

    differences = exporter.differences_from_inventory(record,
                                                      tmp_path / "not_here")

    assert any("not a folder" in line for line in differences), differences


def test_a_missing_record_is_reported_rather_than_passed(exporter, repository,
                                                         tmp_path):
    """No record is not the same as nothing wrong."""
    record, destination = _an_export(exporter, tmp_path)
    record.unlink()

    assert exporter.differences_from_inventory(record, destination)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="making a link needs a privilege Windows does not "
                           "give by default")
def test_a_link_added_after_the_export_is_noticed(exporter, repository,
                                                  tmp_path):
    """The one addition the comparison could not see."""
    # Links were skipped rather than reported, so a link put into a clone after
    # the export was the one kind of thing added that this said nothing about,
    # and it is the kind that reads a file from somewhere nobody exported.
    record, destination = _an_export(exporter, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("anything", encoding="utf-8")
    (destination / "shortcut.md").symlink_to(outside)

    differences = exporter.differences_from_inventory(record, destination)

    assert any("is a link" in line for line in differences), differences




@pytest.fixture
def a_complete_commit(repository):
    """The little repository with everything the assignment asks for in it."""
    # Used wherever a test needs the command to get all the way through, because
    # a candidate that fails the completeness check never reaches the steps after
    # it, and a test written against an incomplete one would be asserting about a
    # path the command does not take.
    for relative, text in COMPLETE_ENOUGH.items():
        write(repository, relative, text)
    with_the_execution_evidence(repository)
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "Everything a release needs")
    return repository


def _building_beside(tmp_path, exporter):
    """Where a candidate is written before it has a name of its own."""
    return tmp_path / ("export" + exporter.BUILDING_SUFFIX)


def _run(exporter, monkeypatch, *arguments):
    """Runs the command with these arguments and returns its status."""
    monkeypatch.setattr(exporter.sys, "argv",
                        ["prepare_release.py", *arguments])
    return exporter.main()


def test_a_copy_that_stops_part_way_leaves_no_release_behind(exporter,
                                                             repository,
                                                             tmp_path,
                                                             monkeypatch):
    """A half-written export must not look like a finished one."""
    # Copying straight into the destination and failing on the fourth file left a
    # folder holding three, which the next run would refuse for not being empty
    # and which somebody could mistake for a release. It is built next door and
    # moved into place only once every file is written and both checks have
    # passed.
    destination = tmp_path / "export"

    written = []
    real = exporter.copy_out_of

    def fail_part_way(commit, into):
        for relative in exporter.what_to_copy(commit)[:2]:
            target = into / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"partly written")
            written.append(relative)
        raise OSError("the disk went away")

    monkeypatch.setattr(exporter, "copy_out_of", fail_part_way)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists(), \
        "a folder that stopped part way was left where a release goes"
    assert written, "the test did not actually copy anything before failing"
    assert real is not exporter.copy_out_of


def test_a_candidate_left_by_an_earlier_run_stops_the_next_one(exporter,
                                                               repository,
                                                               tmp_path,
                                                               monkeypatch):
    """What was refused stays where it was built, and is not written over."""
    # Whatever is in it is an export nobody has looked at yet, and a run that
    # quietly replaced it would take away the only copy of the thing that went
    # wrong.
    destination = tmp_path / "export"
    _building_beside(tmp_path, exporter).mkdir()

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()


def test_the_record_is_not_written_over_one_already_there(exporter,
                                                          repository,
                                                          tmp_path):
    """The folder has to be empty, and the record sits beside it."""
    # Checked only for the folder, a record left by an earlier run would be
    # replaced without anybody being told, and a link left there would be
    # followed to wherever it pointed.
    destination = tmp_path / "export"
    exporter.inventory_beside(destination).write_text("{}", encoding="utf-8")

    wrong = exporter.why_the_destination_will_not_do(destination)

    assert wrong and "already there" in wrong


def test_a_destination_that_is_a_file_is_refused(exporter, repository,
                                                 tmp_path):
    """Asked whether it was empty first, an existing file made iterdir raise,
    so a command given a file name stopped with a traceback instead of a
    sentence saying what was wrong with it."""
    destination = tmp_path / "export"
    destination.write_text("not a folder\n", encoding="utf-8")

    wrong = exporter.why_the_destination_will_not_do(destination)

    assert wrong and "not a folder" in wrong


def test_the_command_refuses_a_destination_that_is_a_file(exporter, repository,
                                                          tmp_path,
                                                          monkeypatch):
    """The same case, end to end, with nothing built."""
    destination = tmp_path / "export"
    destination.write_text("not a folder\n", encoding="utf-8")

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not _building_beside(tmp_path, exporter).exists()


def test_a_record_that_cannot_be_written_says_what_stopped_it(exporter,
                                                              tmp_path):
    """A full disk and a folder nobody may write to are not "it was already
    there", which is what a true-or-false answer made them. Somebody reading
    that goes to look at a file that is not the problem."""
    stopped = exporter.write_exclusively(
        tmp_path / "no_such_folder" / "record.json", "{}")

    assert stopped and "already there" not in stopped

    assert exporter.write_exclusively(tmp_path / "record.json", "{}") is None
    assert "already there" in exporter.write_exclusively(
        tmp_path / "record.json", "{}")


def test_the_command_refuses_and_writes_nothing_when_the_tree_is_dirty(
        exporter, repository, tmp_path, monkeypatch):
    """End to end: a non-zero status, and no folder built."""
    write(repository, "README.md", "# Changed and not committed\n")
    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert not _building_beside(tmp_path, exporter).exists()


def test_the_command_judges_what_it_built_before_giving_it_the_release_name(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The order that matters: copy, judge the copies, then name them."""
    # The check is given the folder that was written, so what it reads is what
    # would be published. Given the repository instead, it would be approving one
    # tree while another was published. Given the destination before the checks
    # had run, a candidate that failed one of them would already be sitting under
    # the name a finished release has, with a record beside it matching it
    # exactly.
    destination = tmp_path / "export"
    looked_at = {}
    real = exporter.release.problems_in

    def remember(root):
        looked_at["root"] = root
        looked_at["named yet"] = destination.exists()
        return real(root)

    monkeypatch.setattr(exporter.release, "problems_in", remember)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0
    assert looked_at["root"] == _building_beside(tmp_path, exporter)
    assert looked_at["named yet"] is False
    assert (destination / "README.md").exists()
    assert not _building_beside(tmp_path, exporter).exists()


@pytest.mark.parametrize("which", ["problems_in", "what_a_release_is_missing"])
def test_a_candidate_that_fails_a_check_never_gets_the_release_name(
        exporter, a_complete_commit, tmp_path, monkeypatch, which):
    """Either check failing leaves the candidate named as one."""
    # The record is not written either. A record beside no folder refuses the
    # next run and describes an export nobody has.
    destination = tmp_path / "export"
    monkeypatch.setattr(exporter.release, which,
                        lambda root, *rest: ["something is wrong with it"])

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert not exporter.inventory_beside(destination).exists(), \
        "a record was written for an export that was refused"
    assert _building_beside(tmp_path, exporter).is_dir(), \
        "the candidate was thrown away instead of being left to look at"


def test_a_record_that_cannot_be_written_stops_the_release(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """Without a record there is nothing to check a clone against."""
    # So the export does not get its name either. A folder published with no
    # record of what left here is the state the record exists to prevent.
    destination = tmp_path / "export"
    monkeypatch.setattr(exporter, "write_exclusively",
                        lambda path, text: "the folder is read-only")

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert _building_beside(tmp_path, exporter).is_dir()


def test_a_move_that_fails_leaves_the_candidate_and_takes_back_the_record(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The export passed and could not be put in place."""
    # Reported rather than raised, and the record written moments earlier is
    # taken away again, because a record beside no folder refuses the next run
    # and lists an export nobody can look at.
    destination = tmp_path / "export"

    def refuse(self, target):
        raise OSError("the folder is in use")

    monkeypatch.setattr(exporter.pathlib.Path, "replace", refuse)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert _building_beside(tmp_path, exporter).is_dir()
    assert not exporter.inventory_beside(destination).exists()


def test_an_empty_folder_already_at_the_destination_is_used(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """Making the folder first is an ordinary thing to do."""
    # It is allowed, and renaming onto a folder that exists fails on Windows, so
    # the empty one is removed just before the move. rmdir refuses a folder
    # holding anything, so it cannot take away something somebody wanted.
    destination = tmp_path / "export"
    destination.mkdir()

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0
    assert (destination / "README.md").exists()


def test_a_safe_but_unfinished_export_is_refused_for_being_unfinished(
        exporter, repository, tmp_path, monkeypatch):
    """Safe and finished are separate questions, answered separately."""
    # This little repository holds nothing that must be kept out of a public
    # repository, and it is nowhere near a submission: no documents, no
    # benchmark, no test output, no integration tests. The command says so in
    # those terms rather than reporting it as unsafe, because telling somebody
    # their release is dangerous when it is merely incomplete sends them looking
    # in the wrong place.
    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1

    building = _building_beside(tmp_path, exporter)
    assert not destination.exists(), \
        "an unfinished export was given the name a finished one has"
    assert exporter.release.problems_in(building) == [], \
        "nothing in it must be kept out of a repository"
    assert exporter.release.what_a_release_is_missing(building), \
        "and it is not a finished submission"


def test_the_command_succeeds_when_the_export_is_safe_and_complete(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The one that has to pass, so that a failure anywhere else means
    something. Everything the assignment asks for is in the commit."""
    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0
    assert exporter.inventory_beside(destination).exists()


def _a_private_recording_and_its_entry():
    """One recording of a real service, kept for private replay only, as the
    fifteen in the sandbox are recorded: a replay scope on its approval."""
    reply = json.dumps({
        "captured_at": "2026-08-28T10:00:00+00:00",
        "method": "GET",
        "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        "asked": {"db": "pubmed", "term": "a real search"},
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"esearchresult": {"count": "0", "idlist": []}}),
    }, indent=2)
    fingerprint = hashlib.sha256(reply.encode("utf-8")).hexdigest()
    entry = {
        "sha256": fingerprint,
        "approval_history": [{
            "approved_at": "2026-09-03T10:00:00+00:00",
            "approved_sha256": fingerprint,
            "repository_head_at_review": "b" * 40,
            "working_tree_at_review": "nothing uncommitted",
            "replay_scope": {
                "scope": "private sandbox only",
                "reason": "a recording of a real service, not for publishing",
                "applies_to_sha256": fingerprint,
            },
        }],
    }
    return reply, entry


def test_a_recording_kept_private_is_left_out_and_the_fixture_travels(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The replay folder holds both kinds of file, and only one may leave."""
    # The fifteen recordings of real services sit beside the demonstration's
    # fixtures in the one folder allowed whole, and the check on the export
    # refuses any recording kept private. Copying the folder as it stood refused
    # every export that carried a demonstration. The copy now leaves those files
    # out and writes the manifest without their entries (R-12).
    reply, entry = _a_private_recording_and_its_entry()
    write(a_complete_commit, "data/recorded_replies/a_recording.json", reply)
    manifest = json.loads(_MANIFEST)
    manifest["files"]["a_recording.json"] = entry
    write(a_complete_commit, "data/recorded_replies_manifest.json",
          json.dumps(manifest, indent=2))
    git(a_complete_commit, "add", "-A")
    git(a_complete_commit, "commit", "-q", "-m", "A recording beside the fixture")
    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0

    assert (destination / "data/recorded_replies/a_fixture.json").exists()
    assert not (destination / "data/recorded_replies/a_recording.json").exists()
    exported = json.loads((destination / "data/recorded_replies_manifest.json")
                          .read_text(encoding="utf-8"))
    assert set(exported["files"]) == {"a_fixture.json"}
    record = json.loads(exporter.inventory_beside(destination)
                        .read_text(encoding="utf-8"))
    assert "data/recorded_replies/a_recording.json" not in record["files"]
    written_manifest = (destination / "data/recorded_replies_manifest.json").read_bytes()
    assert record["files"]["data/recorded_replies_manifest.json"] == \
        hashlib.sha256(written_manifest).hexdigest(), \
        "the record fingerprints the manifest as written, not the commit's copy"


def test_the_draft_flag_is_read_and_does_not_become_a_folder(exporter,
                                                             repository,
                                                             tmp_path,
                                                             monkeypatch):
    """The flag was inspected and never removed from the list."""
    # Counted as a third argument, it failed the length check, so the command
    # that used it printed its help and did nothing. Every flag has to be taken
    # out before what is left is counted.
    for relative, text in COMPLETE_ENOUGH.items():
        if not relative.startswith("data/recorded_replies"):
            write(repository, relative, text)
    with_the_execution_evidence(repository)
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "Everything but the fixtures")

    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "--no-demonstration", "HEAD",
                str(destination)) == 0, \
        "the flag was not read, or was counted as a folder"


def test_without_the_draft_flag_the_missing_demonstration_is_reported(
        exporter, repository, tmp_path, monkeypatch):
    """The same export, asked to be a finished one."""
    for relative, text in COMPLETE_ENOUGH.items():
        if not relative.startswith("data/recorded_replies"):
            write(repository, relative, text)
    with_the_execution_evidence(repository)
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "Everything but the fixtures")

    destination = tmp_path / "export"

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert any("replay" in line for line in
               exporter.release.what_a_release_is_missing(
                   _building_beside(tmp_path, exporter)))


def _a_finished_export(exporter, tmp_path, monkeypatch):
    """An export the command built and approved, with its record beside it."""
    destination = tmp_path / "export"
    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0
    return exporter.inventory_beside(destination), destination


def test_the_verify_flag_says_yes_to_the_export_it_was_written_for(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The baseline. A check that never says yes is one nobody runs."""
    record, destination = _a_finished_export(exporter, tmp_path, monkeypatch)

    assert _run(exporter, monkeypatch, "--verify-inventory", str(record),
                str(destination)) == 0


def test_the_verify_flag_says_no_when_a_file_changed_afterwards(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """What the record is for, asked through the command rather than the
    function underneath it. The helpers were tested and the flag was not, so
    nothing showed that running it did anything at all."""
    record, destination = _a_finished_export(exporter, tmp_path, monkeypatch)
    write(destination, "README.md", "# Changed after the export\n")

    assert _run(exporter, monkeypatch, "--verify-inventory", str(record),
                str(destination)) == 1


def test_the_verify_flag_needs_both_a_record_and_a_folder(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """Given one path it prints its help rather than guessing which was meant."""
    _, destination = _a_finished_export(exporter, tmp_path, monkeypatch)

    assert _run(exporter, monkeypatch, "--verify-inventory",
                str(destination)) == 1


def test_a_folder_that_matches_its_record_can_still_be_refused(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """A record says which bytes were exported. It does not say anybody was
    allowed to export them.
    """
    # Somebody who copied a refused export faithfully, or who added a file and
    # wrote it into the record, would have a folder matching its record exactly.
    # So the two checks are asked again here, and not only the fingerprints.
    record, destination = _a_finished_export(exporter, tmp_path, monkeypatch)
    added = write(destination, "docs/notes_to_self.md",
                  "an abstract, pasted in after the check\n")
    inventory = json.loads(record.read_text(encoding="utf-8"))
    inventory["files"]["docs/notes_to_self.md"] = hashlib.sha256(
        added.read_bytes()).hexdigest()
    record.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    assert exporter.differences_from_inventory(record, destination) == [], \
        "the record and the folder were meant to agree in this test"
    assert _run(exporter, monkeypatch, "--verify-inventory", str(record),
                str(destination)) == 1


def test_a_file_changed_while_the_checks_ran_is_not_released(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """A file can change between being recorded and being checked."""
    # The record used to be written after the checks, from a second reading of
    # the folder, so a file changed between the two readings was fingerprinted
    # and released without ever having been checked. The record is written first
    # now and the folder is compared with it again after the checks.
    #
    # Nothing hostile is needed to cause this. The repository this was built in
    # sits inside a folder that synchronises itself.
    destination = tmp_path / "export"
    real = exporter._judge

    def judge_then_change_something(folder, expect_a_demonstration, note=""):
        outcome = real(folder, expect_a_demonstration, note)
        (folder / "README.md").write_text("# Changed while the checks ran\n",
                                          encoding="utf-8")
        return outcome

    monkeypatch.setattr(exporter, "_judge", judge_then_change_something)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert not exporter.inventory_beside(destination).exists()


def test_a_file_added_while_the_checks_ran_is_not_released(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The other half. A file that appears after the checks would be moved
    into the release while being absent from the record of what was exported,
    so the release would carry a file no record mentions."""
    destination = tmp_path / "export"
    real = exporter._judge

    def judge_then_add_something(folder, expect_a_demonstration, note=""):
        outcome = real(folder, expect_a_demonstration, note)
        write(folder, "research_agent/appeared.py", "SECRET = 'not reviewed'\n")
        return outcome

    monkeypatch.setattr(exporter, "_judge", judge_then_add_something)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert not exporter.inventory_beside(destination).exists()


def test_a_file_the_copy_did_not_report_is_caught_by_the_record(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The rules would pass it. The record is what does not."""
    # A file written into the export but left out of what the copy reported is an
    # allowed kind of file in an allowed place, so every rule here lets it
    # through. What notices is the arithmetic between the folder and the record
    # of what was supposed to be written into it.
    destination = tmp_path / "export"
    real = exporter.copy_out_of

    def copy_and_leave_one_out(commit, into):
        written = real(commit, into)
        write(into, "research_agent/unreported.py", "# not in the record\n")
        return written

    monkeypatch.setattr(exporter, "copy_out_of", copy_and_leave_one_out)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()


def test_the_record_says_which_standard_the_export_was_held_to(
        exporter, repository, tmp_path, monkeypatch):
    """A rehearsal and a submission are different exports, and the record has
    to say which one it describes."""
    for relative, text in COMPLETE_ENOUGH.items():
        if not relative.startswith("data/recorded_replies"):
            write(repository, relative, text)
    with_the_execution_evidence(repository)
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "Everything but the fixtures")

    destination = tmp_path / "export"
    assert _run(exporter, monkeypatch, "--no-demonstration", "HEAD",
                str(destination)) == 0

    record = exporter.inventory_beside(destination)
    recorded = json.loads(record.read_text(encoding="utf-8"))

    assert recorded["demonstration_required"] is False


def test_a_flag_typed_later_cannot_lower_the_standard_an_export_was_built_to(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The standard is read from the record, not from the command line."""
    # Taken from the flag, the same folder passed or failed depending on what
    # somebody typed afterwards, and nothing said which answer it had been built
    # for. A submission release verified with --no-demonstration would have been
    # held to the standard of a rehearsal.
    destination = tmp_path / "export"
    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0

    # The manifest goes with the replies. Left behind, it describes fifteen
    # files that are not there and the safety check refuses the export for
    # that, which is correct and is not what this test is about: the first
    # version of it passed on those refusals and said nothing at all about the
    # flag it was named after.
    record = exporter.inventory_beside(destination)
    shutil.rmtree(destination / "data" / "recorded_replies")
    (destination / "data" / "recorded_replies_manifest.json").unlink()
    inventory = json.loads(record.read_text(encoding="utf-8"))
    for relative in list(inventory["files"]):
        if relative.startswith("data/recorded_replies"):
            del inventory["files"][relative]
    record.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    assert exporter.differences_from_inventory(record, destination) == [], \
        "the record and the folder were meant to agree in this test"
    assert exporter.release.problems_in(destination) == [], \
        "nothing in it must be kept out, so only completeness is left to fail"
    assert _run(exporter, monkeypatch, "--no-demonstration",
                "--verify-inventory", str(record), str(destination)) == 1, \
        "the flag lowered the standard the export was built to"


def test_the_record_describes_the_commit_and_not_the_folder(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """A file changed after the copy must not become the record's truth."""
    # The record used to be built by reading the folder back. A file changed
    # between git handing it over and that reading was fingerprinted as it then
    # stood, so the record said the commit contained bytes the commit had never
    # held, and every comparison afterwards agreed with it. The fingerprints come
    # from the bytes git handed over instead.
    destination = tmp_path / "export"
    real = exporter.copy_out_of

    def copy_then_change_one(commit, into):
        written = real(commit, into)
        (into / "README.md").write_text("# Changed after the copy\n",
                                        encoding="utf-8")
        return written

    monkeypatch.setattr(exporter, "copy_out_of", copy_then_change_one)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()
    assert not exporter.inventory_beside(destination).exists()


def test_the_record_holds_the_fingerprints_git_gave(exporter, a_complete_commit,
                                                    tmp_path, monkeypatch):
    """What copy_out_of returns is what the record is made of."""
    destination = tmp_path / "export"
    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0

    recorded = json.loads(
        exporter.inventory_beside(destination).read_text(encoding="utf-8"))
    identifier = recorded["exported_from"]

    for relative, fingerprint in recorded["files"].items():
        from_git = subprocess.run(
            ["git", "-C", str(a_complete_commit), "show",
             f"{identifier}:{relative}"], capture_output=True, check=True).stdout
        assert hashlib.sha256(from_git).hexdigest() == fingerprint, relative


def test_a_file_changed_during_the_move_is_not_reported_as_published(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The last comparison, which nothing exercised until now."""
    # Moving the folder is the final thing that touches these files, and without
    # a test that changes one while it happens, removing that comparison would
    # not have altered a single result.
    destination = tmp_path / "export"
    real = pathlib.Path.replace

    def move_then_change_something(self, target):
        outcome = real(self, target)
        (pathlib.Path(target) / "README.md").write_text(
            "# Changed during the move\n", encoding="utf-8")
        return outcome

    monkeypatch.setattr(exporter.pathlib.Path, "replace",
                        move_then_change_something)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert destination.is_dir(), \
        "the move did happen, so the folder is where the report says"


def test_a_rehearsal_is_not_reported_as_complete_for_submission(
        exporter, repository, tmp_path, monkeypatch, capsys):
    """A rehearsal left out something a submission needs, on purpose."""
    # Saying everything the assignment asks for is in it would be reporting the
    # presence of the one thing it was built without.
    for relative, text in COMPLETE_ENOUGH.items():
        if not relative.startswith("data/recorded_replies"):
            write(repository, relative, text)
    with_the_execution_evidence(repository)
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "Everything but the fixtures")

    destination = tmp_path / "export"
    assert _run(exporter, monkeypatch, "--no-demonstration", "HEAD",
                str(destination)) == 0

    record = exporter.inventory_beside(destination)
    assert _run(exporter, monkeypatch, "--no-demonstration",
                "--verify-inventory", str(record), str(destination)) == 0

    said = capsys.readouterr().out
    assert "records a rehearsal" in said, said
    assert "Everything the assignment asks for is in it." not in said, said


def _meddle_with_the_record_during_the_checks(exporter, monkeypatch, record,
                                              meddle):
    """Does something to the record between the checks and the move."""
    real = exporter._judge

    def judge_then_meddle(folder, expect_a_demonstration, note=""):
        outcome = real(folder, expect_a_demonstration, note)
        meddle(record)
        return outcome

    monkeypatch.setattr(exporter, "_judge", judge_then_meddle)


def _rewrite(record):
    record.write_text(record.read_text(encoding="utf-8").replace(
        '"demonstration_required": true',
        '"demonstration_required": false'), encoding="utf-8")


def _remove(record):
    record.unlink()


def _make_it_a_folder(record):
    record.unlink()
    record.mkdir()


def _make_it_unreadable(record):
    record.write_bytes(b"\xff\xfe\x00 not text at all")


@pytest.mark.parametrize("meddle, said", [
    (_rewrite, "is not the record this run wrote"),
    (_remove, "is not there"),
    (_make_it_a_folder, "is not a file"),
    (_make_it_unreadable, "is not the record this run wrote"),
], ids=["rewritten", "removed", "replaced by a folder", "made unreadable"])
def test_a_record_disturbed_during_the_checks_stops_the_release(
        exporter, a_complete_commit, tmp_path, monkeypatch, meddle, said,
        capsys):
    """The record is asked about before the folder is given its name."""
    # It was asked about after the move for a while, so a record altered while
    # the checks ran was found only once the candidate already stood under the
    # name a finished release has. Read directly rather than through a check that
    # answers, a record removed or replaced by a folder stopped the command with
    # a traceback, which is easy to read as the tool being broken rather than the
    # release being wrong.
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)
    _meddle_with_the_record_during_the_checks(exporter, monkeypatch, record,
                                              meddle)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists(), \
        "the candidate was given the name of a release before this was found"
    assert _building_beside(tmp_path, exporter).is_dir(), \
        "what was built should still be there to look at"

    # The reason, not only the refusal. Asked for the exit code alone, these
    # all passed with each separate reason taken out, because reading a record
    # that is gone raises something the catch-all turns into a refusal anyway.
    # What is worth having is a sentence that sends somebody to the right
    # thing, and asserting it is what keeps those apart.
    assert said in capsys.readouterr().out


def test_the_record_is_written_with_the_endings_it_is_compared_against(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """What a record holds is decided here, not by the machine it was made on."""
    # Left to translate, this wrote one thing and held another: the file carried
    # Windows endings while the text it is compared with carried Unix ones, and
    # the two only agreed because reading the file translated them back. A record
    # is read on other machines, so its bytes are settled here.
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 0
    assert b"\r\n" not in record.read_bytes()
    assert json.loads(record.read_bytes().decode("utf-8"))["files"]


@pytest.mark.skipif(sys.platform == "win32",
                    reason="making a link needs a privilege Windows does not "
                           "give by default, so the case is checked directly")
def test_a_record_replaced_by_a_link_is_refused_whatever_it_holds(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """A link holding the right text is still not the record this run wrote."""
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)

    def swap_in_a_link(the_record):
        elsewhere = tmp_path / "elsewhere.json"
        elsewhere.write_bytes(the_record.read_bytes())
        the_record.unlink()
        the_record.symlink_to(elsewhere)

    _meddle_with_the_record_during_the_checks(exporter, monkeypatch, record,
                                              swap_in_a_link)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not destination.exists()


def test_a_link_is_refused_by_the_rule_even_where_one_cannot_be_made(tmp_path,
                                                                     exporter):
    """The rule the test above needs a privilege to exercise, asked directly."""
    record = tmp_path / "record.json"
    record.write_text("anything", encoding="utf-8")

    assert exporter.why_the_record_is_not_the_one_written(
        record, "anything") is None
    assert exporter.why_the_record_is_not_the_one_written(
        record, "something else")
    assert exporter.why_the_record_is_not_the_one_written(
        tmp_path / "not_there.json", "anything")
    assert exporter.why_the_record_is_not_the_one_written(tmp_path, "anything")


def test_a_record_something_else_changed_is_not_deleted_by_the_cleanup(
        exporter, a_complete_commit, tmp_path, monkeypatch, capsys):
    """The record this run wrote is taken away again. Nothing else is."""
    # A run that writes a record and is then refused removes it, because a record
    # beside no folder refuses the next run and lists an export nobody has. It
    # used to remove whatever was at that path, so a record something else had
    # changed while the checks ran was deleted when those checks refused the
    # release, and the evidence of the change went with it.
    destination = tmp_path / "export"
    record = exporter.inventory_beside(destination)

    def refuse_and_change_the_record(root):
        record.write_text("changed by something else", encoding="utf-8")
        return ["something is wrong with it"]

    monkeypatch.setattr(exporter.release, "problems_in",
                        refuse_and_change_the_record)

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert record.read_text(encoding="utf-8") == "changed by something else"
    assert "left where it is" in capsys.readouterr().out


def test_the_record_this_run_wrote_is_taken_away_when_it_is_refused(
        exporter, a_complete_commit, tmp_path, monkeypatch):
    """The other half, so the rule above cannot be met by never deleting."""
    destination = tmp_path / "export"
    monkeypatch.setattr(exporter.release, "problems_in",
                        lambda root: ["something is wrong with it"])

    assert _run(exporter, monkeypatch, "HEAD", str(destination)) == 1
    assert not exporter.inventory_beside(destination).exists()


@pytest.mark.parametrize("written, said", [
    ("[" * 20000 + "1" + "]" * 20000, "could not be read"),
    (None, "could not be read"),
], ids=["nested too deeply", "not text"])
def test_a_record_nothing_can_read_is_refused_not_raised(exporter,
                                                         a_complete_commit,
                                                         tmp_path, monkeypatch,
                                                         written, said, capsys):
    """A record is read before anything is compared against it."""
    # Read without answering for what can go wrong, a record nested deeper than
    # Python will follow, or written in bytes that are not text, stopped the
    # command with a traceback. Neither is a record, and both should be reported
    # as one that cannot be read.
    record, destination = _a_finished_export(exporter, tmp_path, monkeypatch)
    if written is None:
        record.write_bytes(b'{"a": "\xff\xfe not utf-8"}')
    else:
        record.write_text(written, encoding="utf-8")

    assert _run(exporter, monkeypatch, "--verify-inventory", str(record),
                str(destination)) == 1
    assert said in capsys.readouterr().out


def test_the_command_writes_no_bytecode_into_the_export_it_reads(tmp_path):
    """The exported copy must not alter the export it is there to vouch for."""
    # Importing the project package saved a .pyc file beside each module, inside
    # the export, and the command's own comparison then reported those files as
    # additions no export had produced. A reader checking an export would have
    # been told by this command that the export must not be published, when
    # the only thing wrong with it was what the command had just written
    # into it.
    #
    # Run as a separate process on purpose. Every other test in this file loads
    # the command out of this repository and checks a sandbox export that holds
    # no copy of it, so the imports never come from the folder being checked and
    # none of those tests could see this happen.
    export = tmp_path / "export"
    (export / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPT, export / "scripts" / "prepare_release.py")
    shutil.copytree(SCRIPT.parents[1] / "research_agent",
                    export / "research_agent",
                    ignore=shutil.ignore_patterns("__pycache__"))

    # Either of these two settings stops Python writing bytecode, or sends it
    # somewhere else, whatever the command itself does. Inherited from whoever
    # ran the tests, they would make this pass with the line taken out of the
    # command, which is the one thing it must not do.
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)

    # Given no arguments the command prints its usage and stops with 1, so the
    # status it returns is not a failure of this test.
    finished = subprocess.run(
        [sys.executable, "export/scripts/prepare_release.py"],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        check=False)

    left_behind = []
    for path in sorted(export.rglob("*")):
        if path.name == "__pycache__" or path.suffix == ".pyc":
            left_behind.append(str(path.relative_to(export)))

    assert left_behind == [], left_behind
    # The usage text is the proof that the imports were reached. Without it the
    # check above would also pass for a command that stopped before importing
    # anything, which is the one way it could be satisfied by accident.
    assert "<commit-or-tag>" in finished.stdout
