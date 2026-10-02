"""Checks that git hands the same bytes to everybody."""
# Two things here fingerprint exact bytes. The manifest records what each saved
# reply was approved as, and replay refuses a reply whose bytes have changed. The
# record of an export lists every file it carries, and a fresh clone is checked
# against it.
#
# Neither survives git deciding line endings from the setting on whoever cloned
# the repository. This one was checked out with core.autocrlf=true, so its files
# are held with Windows line endings while git stored them with Unix ones.
# Verifying a clone against the record of an export was tried before
# `.gitattributes` existed, and fifty-nine of sixty-eight files disagreed.
#
# These tests read the real `.gitattributes` and the real manifest, where most
# tests here deliberately invent what they need. The rules being tested are the
# ones in this repository, and a copy of them would only show that a copy works.

import hashlib
import json
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ATTRIBUTES = REPO / ".gitattributes"
MANIFEST = REPO / "data" / "recorded_replies_manifest.json"

# Cloning three times is slower than the tests around it, and this is the only
# way to see what a clone actually gets. The alternative is reading the rules
# and believing them.
pytestmark = pytest.mark.skipif(not shutil.which("git"),
                                reason="git is needed to see what a clone gets")

# The export carries these tests and not the repository: the release command
# copies files out of a commit and leaves `.git` behind on purpose. The two
# checks below that ask git about this repository have nothing to ask there,
# and on 1 October a reader running the documented command in the exported
# package saw them fail as though the rules were wrong. They are skipped on
# one condition only, the missing `.git` entry; once that entry exists, any
# git failure still fails the test, because it may be a damaged repository.
# The clone tests above make their own repository, so they still run.
_NOTHING_FOR_GIT_TO_HOLD_HERE = pytest.mark.skipif(
    not (REPO / ".git").exists(),
    reason="an exported package has no repository for git to answer about")


def git(where, *arguments, **settings):
    """Runs git somewhere, with settings applied to that one command."""
    passed = []
    for name, value in settings.items():
        passed += ["-c", f"{name.replace('_', '.')}={value}"]
    return subprocess.run(["git", *passed, "-C", str(where), *arguments],
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("relative, expected", [
    ("README.md", {"text": "auto", "eol": "lf"}),
    ("research_agent/release.py", {"text": "auto", "eol": "lf"}),
    ("data/recorded_replies/anything.json",
     {"text": "unset", "whitespace": "cr-at-eol"}),
    ("evidence/testing/a_screenshot.png", {"binary": "set"}),
])
@_NOTHING_FOR_GIT_TO_HOLD_HERE
def test_the_rules_say_what_they_are_meant_to_say(relative, expected):
    """Asked of git rather than read off the file."""
    # A rule can be written and mean something else. git check-attr says what git
    # will actually do with a path, which is the thing that matters.
    for attribute, wanted in expected.items():
        line = git(REPO, "check-attr", attribute, "--", relative).strip()

        assert line.endswith(f" {wanted}"), line


def _a_repository_using_the_real_rules(tmp_path):
    """A small repository with this project's attributes and three files."""
    repo = tmp_path / "sandbox"
    (repo / "data" / "recorded_replies").mkdir(parents=True)
    shutil.copyfile(ATTRIBUTES, repo / ".gitattributes")

    # Written with Windows endings, as everything in this checkout is. What the
    # rules do with them is the whole question.
    (repo / "README.md").write_bytes(b"# The agent\r\n\r\nHow to run it.\r\n")
    reply = repo / "data" / "recorded_replies" / "a_reply.json"
    reply.write_bytes(b'{\r\n  "captured_at": "2026-09-04T00:00:00+00:00"\r\n}\r\n')
    picture = repo / "a_picture.png"
    picture.write_bytes(bytes.fromhex("89504e470d0a1a0a") + b"\r\n not text \x00")

    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "somebody@example.com")
    git(repo, "config", "user.name", "Somebody")
    git(repo, "config", "core.autocrlf", "true")
    git(repo, "add", "-A")
    git(repo, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "One commit")
    return repo, reply.read_bytes(), picture.read_bytes()


@pytest.mark.parametrize("setting", ["true", "false", "input"])
def test_a_clone_gets_the_same_bytes_whatever_the_setting(tmp_path, setting):
    """The three settings somebody cloning this could have."""
    # A saved reply has to come out byte for byte, because its fingerprint was
    # taken from those bytes and replay refuses anything else. Ordinary text has
    # to come out the same way everywhere, because the record of an export
    # fingerprints it too.
    repo, reply_bytes, picture_bytes = _a_repository_using_the_real_rules(tmp_path)
    clone = tmp_path / f"clone_{setting}"
    subprocess.run(["git", "-c", f"core.autocrlf={setting}", "clone", "-q",
                    str(repo), str(clone)], check=True, capture_output=True)

    assert (clone / "data" / "recorded_replies" / "a_reply.json").read_bytes() \
        == reply_bytes, "a saved reply came out of the clone changed"
    assert (clone / "a_picture.png").read_bytes() == picture_bytes
    assert b"\r\n" not in (clone / "README.md").read_bytes(), \
        "ordinary text came out with the endings of whoever cloned it"


def _every_fingerprint_on(entry):
    """Each fingerprint the manifest holds for one reply."""
    # Three of them, and they are three statements about the same file. The one
    # at the top of the entry says what the file should be now. The approval
    # carries its own, which is what somebody actually accepted. The record of
    # what may be done with it afterwards carries a third, saying which version
    # that decision was about.
    #
    # All three, because checking the first alone would let a change to the easy
    # one pass while the record of what was accepted said something else. The
    # same reason the export check reads all three.
    latest = entry["approval_history"][-1]
    found = {"the entry": entry["sha256"],
             "its approval": latest["approved_sha256"]}
    for where in ("replay_scope", "fixture_origin"):
        if where in latest:
            found[where.replace("_", " ")] = latest[where]["applies_to_sha256"]
    return found


@_NOTHING_FOR_GIT_TO_HOLD_HERE
def test_every_saved_reply_git_holds_is_the_one_its_approval_accepted():
    """The rules above, asked of this repository rather than a made-up one."""
    # What git is holding for each reply is compared with every fingerprint the
    # manifest records for it. The index is read rather than the working files,
    # because the staged blob is what this commit will store and what a clone
    # will later be given.
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    disagreeing = []
    for name, entry in sorted(manifest["files"].items()):
        held = subprocess.run(
            ["git", "-C", str(REPO), "show", f":data/recorded_replies/{name}"],
            capture_output=True, check=True).stdout
        now = hashlib.sha256(held).hexdigest()
        for what, fingerprint in _every_fingerprint_on(entry).items():
            if fingerprint != now:
                disagreeing.append(f"{name} ({what})")

    assert disagreeing == [], (
        "git is holding a different version of these than the manifest "
        "records: " + ", ".join(disagreeing))
