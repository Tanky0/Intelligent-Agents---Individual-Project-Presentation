"""Moves captured replies into the approved replay set (R-07)."""
# Replies are captured automatically as they arrive, into a folder that is not in
# the repository. Nothing is replayed from there. This is the one way a reply
# reaches the approved replay set, and it checks each one before it goes.
#
# Reaching that set is not permission to publish. What may be replayed and what
# may be published are recorded separately, and everything currently in the set
# is recorded as kept for replaying here only, never for publishing. The public
# demonstration will replay project-owned synthetic fixtures instead (R-12).
#
#     python scripts/approve_replies.py            show what is waiting, and
#                                                  whether the approved set still
#                                                  agrees with the manifest
#     python scripts/approve_replies.py all        approve everything waiting
#     python scripts/approve_replies.py <name>     approve one file by name
#     python scripts/approve_replies.py existing   fingerprint replies that were
#                                                  already approved before there
#                                                  was a manifest. Refused once a
#                                                  manifest exists, so it cannot
#                                                  be used to bless a file that
#                                                  was never captured
#     python scripts/approve_replies.py reapprove  record that you have reviewed
#                                                  the replies already in the
#                                                  replay set, against the commit
#                                                  the repository stands at now
#     python scripts/approve_replies.py keep-private
#                                                  record that the replies
#                                                  already in the replay set are
#                                                  kept for replaying here
#                                                  only, with no right to
#                                                  publish them established
#                                                  or claimed
#     python scripts/approve_replies.py redact     take contact details out of
#                                                  replies approved before that
#                                                  removal existed, and record
#                                                  their new fingerprints
#
# The checks and the writing are in saved_replies.py, so that the tests use the
# same ones. This file decides what the person running it asked for and says what
# happened.
#
# A reply is not copied. It is built again from the list of what to keep, so a
# field nobody thought of cannot reach the repository by being in the file.
#
# Every approved reply has its fingerprint written into
# data/recorded_replies_manifest.json with the day it was approved and where the
# repository stood at the time. Replaying checks that fingerprint, and the
# tests fail if the folder holds a file nobody approved, is missing one that was
# approved, or holds one whose contents have changed. That is the check that
# would have caught both times an invented reply reached this folder.
#
# Approving is deliberate on purpose. Capture has to be automatic, or a
# demonstration set will one day be missing because nobody remembered to switch
# it on. Publishing has to be asked for, or the folder fills with whatever
# happened to be running.

import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent.sources import saved_replies


def _looks_like_a_commit(text):
    """True when this is the full form of a commit identifier."""
    # Checked rather than trusted. git normally answers rev-parse with forty hex
    # characters, but it answers with whatever it has: in a repository with no
    # commits yet it fails, and in some states it returns a name instead. A
    # manifest entry saying the review happened at "HEAD" would be worse than one
    # saying nothing, because it looks like a record.
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def approval_commit():
    """Returns where the repository stood when this approval was made."""
    # The whole commit rather than the short form, because a short one can stop
    # being unique as a repository grows. Whether anything was uncommitted is
    # recorded too: a commit on its own would suggest the code was exactly that,
    # when most of the work can be sitting unsaved beside it.
    #
    # git is run against the project folder rather than wherever the command
    # happened to be started from. Returns None when it cannot be established,
    # and then nothing is approved.
    #
    # An earlier version recorded the word "unknown" and carried on. The manifest
    # now says every approval names the commit that was reviewed, and a record
    # saying "unknown" does not say what somebody looked at: it says only that
    # somebody looked at something. Refusing is the honest answer, and the fix is
    # to run the command where git works rather than to keep a record that cannot
    # be checked.
    root = str(pathlib.Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        waiting = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                                 capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None

    if not _looks_like_a_commit(commit.stdout.strip()):
        return None

    if waiting.stdout.strip():
        state = "changes not committed"
    else:
        state = "nothing uncommitted"
    return {"commit": commit.stdout.strip(), "working_tree": state}


def publication_basis_from(arguments):
    """Reads the basis for publishing article text from a file named on the
    command line, or returns None when none was given.
    """
    # Returns False when one was named and could not be read, so that the caller
    # stops rather than approving without it.
    #
    # It is a file rather than something typed after the command, because it has
    # to say several things at once: which licence, whose work, and what was
    # taken out. A licence named in a shell would be recorded once and never seen
    # again; a file can be committed beside the manifest and read by anybody
    # checking the claim (R-11).
    #
    # A reply carrying no article text needs no basis, and most do not carry any.
    # Leaving this out is the ordinary case, not an oversight.
    if "--basis" not in arguments:
        return None

    where = arguments.index("--basis") + 1
    if where >= len(arguments):
        print("--basis has to be followed by the path to a file describing it.")
        return False

    try:
        return json.loads(pathlib.Path(arguments[where]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as problem:
        print(f"The basis file could not be read: {problem}")
        return False


def _describe(path):
    """One line saying what a captured reply is, so a person can recognise it."""
    record = saved_replies._read(path)
    if not isinstance(record, dict):
        return "unreadable"

    captured = record.get("captured_at", "no date")
    if "model" in record:
        return f"an answer from {record['model']}, captured {captured}"

    asked = record.get("asked") or {}
    wanted = asked.get("term") or asked.get("query") or asked.get("id") or ""
    return f"{record.get('url', 'no address')} {wanted}".strip() + f", captured {captured}"


def show_what_is_waiting():
    """Prints what has been captured, what is approved, and any disagreement."""
    waiting = saved_replies.files_in(saved_replies.CAPTURED_DIR)
    approved = saved_replies.files_in(saved_replies.APPROVED_DIR)

    print(f"captured and waiting: {len(waiting)}")
    for path in waiting:
        print(f"  {path.name}  {_describe(path)}")

    print(f"approved for the demonstration: {len(approved)}")

    problems = saved_replies.differences_from_manifest()
    if problems:
        print("the approved folder and the manifest do not agree:")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("the approved folder matches the manifest")


def choose(wanted, manifest):
    """Returns the files to approve, or None if the request cannot be met."""
    waiting = saved_replies.files_in(saved_replies.CAPTURED_DIR)

    if wanted == "existing":
        # The one-off for replies that were already approved before there was a
        # manifest. Once one exists this is refused, so it cannot be used to
        # approve a file that never went through capture.
        if manifest["files"]:
            print("There is already a manifest, so 'existing' is not available.")
            print("Approve a captured reply by name, or with 'all'.")
            return None
        return saved_replies.files_in(saved_replies.APPROVED_DIR)

    if wanted == "all":
        # Everything waiting is shown before anything is published. The cache
        # can hold replies from an abandoned run or a script somebody tried
        # once, and a valid record is not the same as one that belongs in a
        # demonstration.
        if not waiting:
            return waiting
        print("About to approve:")
        for path in waiting:
            print(f"  {path.name}  {_describe(path)}")
        if input("Type yes to approve all of these: ").strip().lower() != "yes":
            print("Nothing approved.")
            return None
        return waiting

    chosen = [path for path in waiting if path.name == wanted]
    if not chosen:
        print(f"{wanted} is not waiting to be approved.")
        return None
    return chosen


def redact_what_is_published():
    """Re-runs the removal of contact details over the replies already in
    the replay set, and writes their new fingerprints.
    """
    # Separate from approving because it changes files nobody is approving. It is
    # safe to run twice: a reply with nothing left to take out is not rewritten,
    # so a second run reports nothing and leaves the manifest alone.
    manifest = saved_replies.read_manifest()
    changed = saved_replies.redact_approved(manifest)

    if not changed:
        print("No reply in the replay set is carrying a contact detail.")
        return 0

    for name in changed:
        print(f"  redacted {name}")

    if not saved_replies.write_manifest(manifest):
        print("The manifest could not be written, so the fingerprints are wrong.")
        return 1

    print(f"{len(changed)} redacted, and their fingerprints recorded again.")
    print("Each one now needs approving again. It is not the file that was "
          "accepted, so its last approval no longer describes it.")
    return 0


def _basis_for(basis, path):
    """The basis to use for one reply, or None when none was given for it."""
    # A basis names the reply it was written for, and the name is never filled in
    # here. An earlier version did fill it in when it was missing, which quietly
    # undid the whole point: one licence description, written after reading one
    # paper, was rebound to every file the command was given and approved them
    # all. Convenience defeating the rule it was attached to.
    #
    # A basis file may instead hold one basis for each reply, keyed by file name,
    # which is how a batch of different papers is approved in one command.
    if not isinstance(basis, dict):
        return basis
    if _is_a_basis_for_each_file(basis):
        return basis.get(path.name)
    return basis


def _is_a_basis_for_each_file(basis):
    """True when this file holds one basis for each reply rather than one basis."""
    # Told apart by shape: a single basis says what kind it is, and a collection
    # of them is names to objects. Anything else is left as a single basis, and
    # the checks then say what is missing from it.
    if "kind" in basis:
        return False
    return bool(basis) and all(isinstance(value, dict) for value in basis.values())


def keep_what_is_published_private(manifest):
    """Records that the replies in the replay set are for replaying here only."""
    # Asks first, because it writes down a decision about what this project will
    # and will not publish, and that is a person's decision rather than a
    # consequence of running a command.
    #
    # It is the opposite of approving. Approving says a reply may be published
    # and why; this says nobody has established that and nobody will, and that
    # the release check must keep these out of any public repository once it
    # exists. It does not exist yet, and this command says so rather than
    # implying the marking is itself a barrier.
    disagreements = saved_replies.differences_from_manifest(files_only=True)
    if disagreements:
        print("The replay set and the manifest do not agree, so nothing "
              "was recorded:")
        for line in disagreements:
            print(f"  {line}")
        return 1

    published = saved_replies.files_in(saved_replies.APPROVED_DIR)
    if not published:
        print("There is nothing in the replay set to record.")
        return 0

    where_we_are = approval_commit()
    if where_we_are is None:
        print("Where the repository stands could not be established, so "
              "nothing was recorded.")
        return 1

    print("About to record these as kept for replaying here only, "
          "with no right to publish them established or "
          "claimed:")
    for path in published:
        print(f"  {path.name}  {_describe(path)}")
    print("Nothing here prevents the folder being copied. The release check "
          "that must exclude them before any public repository is created is "
          "not built yet (R-12).")
    if input("Type yes to record that: ").strip().lower() != "yes":
        print("Nothing recorded.")
        return 0

    recorded = saved_replies.keep_private(manifest, decided_at=where_we_are)
    if not saved_replies.write_manifest(manifest):
        print("The manifest could not be written, so nothing was recorded.")
        for failure in saved_replies.promotion_failures:
            print(f"  {failure}")
        return 1

    print(f"{len(recorded)} recorded as private replay material.")
    return 0


def reapprove_what_is_published(manifest):
    """Records a new approval for replies already in the replay set."""
    # Needed because approving reads the folder replies are captured into, and
    # the fifteen replies already published were captured long ago: that folder
    # is empty, and on a fresh checkout it does not exist. So the ordinary
    # command reports that nothing is waiting and cannot record that anybody has
    # looked at what is published now.
    #
    # That matters after a history rewrite. Every approval those files carry was
    # recorded against a commit that no longer exists, and until somebody reviews
    # them again there is no approval anybody can check. This is how that review
    # gets written down.
    #
    # It is not a way to bless a file that never went through capture. Only files
    # already in the replay set and already in the manifest are offered:
    # anything else has to go through approval the ordinary way.
    # The folder and the manifest have to agree about which files are there
    # before anything is recorded. A published file nobody approved, or an
    # entry whose file has gone, means the set is not what it says it is, and
    # recording a review of it would put a person's name to that state. The
    # missing publication decisions are left out of this check, because
    # supplying them is what this command is for.
    disagreements = saved_replies.differences_from_manifest(files_only=True)
    if disagreements:
        print("The replay set and the manifest do not agree, so nothing "
              "was reapproved:")
        for line in disagreements:
            print(f"  {line}")
        print("Deal with these first. A review recorded now would be a review "
              "of a set that is not what the manifest describes.")
        return 1

    published = saved_replies.files_in(saved_replies.APPROVED_DIR)
    if not published:
        print("There is nothing in the replay set to reapprove.")
        return 0

    known = published

    where_we_are = approval_commit()
    if where_we_are is None:
        print("Where the repository stands could not be established, so "
              "nothing was reapproved.")
        return 1

    basis = publication_basis_from(sys.argv[2:])
    if basis is False:
        return 1

    print(f"About to record that you have reviewed these at {where_we_are['commit']}")
    print(f"  ({where_we_are['working_tree']})")
    for path in known:
        print(f"  {path.name}  {_describe(path)}")
    if input("Type yes if you have reviewed all of these: ").strip().lower() != "yes":
        print("Nothing reapproved.")
        return 0

    done = 0
    for path in known:
        reasons = saved_replies.approve_reply(path, manifest, where_we_are,
                                              _basis_for(basis, path))
        if reasons:
            print(f"  REFUSED {path.name}")
            for reason in reasons:
                print(f"          {reason}")
        else:
            done += 1
            print(f"  reapproved {path.name}")

    if not saved_replies.write_manifest(manifest):
        print("Nothing was published, and nothing counts as reapproved.")
        for failure in saved_replies.promotion_failures:
            print(f"  {failure}")
        return 1

    print(f"{done} of {len(known)} reapproved.")
    return 0 if done == len(known) else 1


def main():
    if len(sys.argv) == 1:
        show_what_is_waiting()
        return 0

    if sys.argv[1] == "redact":
        return redact_what_is_published()

    try:
        manifest = saved_replies.read_manifest()
    except saved_replies.ManifestProblem as problem:
        print(problem)
        return 1

    if sys.argv[1] == "reapprove":
        return reapprove_what_is_published(manifest)

    if sys.argv[1] == "keep-private":
        return keep_what_is_published_private(manifest)

    chosen = choose(sys.argv[1], manifest)
    if chosen is None:
        return 1

    if not chosen:
        print("Nothing is waiting to be approved.")
        return 0

    where_we_are = approval_commit()
    if where_we_are is None:
        print("Where the repository stands could not be established, so nothing "
              "was approved.")
        print("Every approval has to name the commit that was reviewed. Run "
              "this where git can read the project.")
        return 1

    basis = publication_basis_from(sys.argv[2:])
    if basis is False:
        return 1

    approved_now = 0

    for path in chosen:
        reasons = saved_replies.approve_reply(path, manifest, where_we_are,
                                              _basis_for(basis, path))
        if reasons:
            print(f"  REFUSED {path.name}")
            for reason in reasons:
                print(f"          {reason}")
        else:
            approved_now += 1
            print(f"  approved {path.name}")

    if not saved_replies.write_manifest(manifest):
        print("Nothing was published, and nothing counts as approved.")
        for failure in saved_replies.promotion_failures:
            print(f"  {failure}")
        return 1

    print(f"{approved_now} of {len(chosen)} approved.")

    # A file that was refused stays where it is, so the reason can be read and
    # the same command run again once it is dealt with.
    if approved_now < len(chosen):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
