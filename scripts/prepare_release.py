"""Copies a reviewed commit out of this repository and checks it (R-12).

    python scripts/prepare_release.py <commit-or-tag> <folder>
    python scripts/prepare_release.py --check-only <folder>
    python scripts/prepare_release.py --verify-inventory <record> <folder>

Add --no-demonstration to any of these while the offline fixtures do not exist
yet. It asks only that the export be safe, not that it be finished, and nothing
that will be submitted should use it.
"""
# The docstring is longer than the one-to-three-line rule allows because it is
# printed as the usage text when the command is called wrongly, and the usage
# has to name the three forms and the one option.
#
# The commit is what gets copied, not the working directory. An earlier version
# read the folder this file is in, so anything modified or not yet committed
# travelled with the release: a half-finished document, a file left by a probe, a
# draft nobody had reviewed. This project has been caught by stray files twice,
# and reading a commit is the difference between exporting what was reviewed and
# exporting whatever happens to be on disk.
#
# The working tree has to be clean as well. It is not what is copied, but a dirty
# tree means the commit being exported is not the state anybody is looking at,
# and somebody would reasonably assume it was.
#
# The folder is where the public repository will be built. It must be empty, and
# it must be outside this repository: building a release inside the thing being
# released puts the release into the next one.
#
# The order matters. The files are written out of the commit into a folder
# named .building. A record of those bytes is written next, before anything is
# judged. The folder is then compared with that record, judged, compared again,
# renamed, and compared once more under its own name.
#
# That order prevents two things. What is judged and what is moved are the same
# folder, so a check cannot pass on one folder while another is published. And a
# file that changes between being recorded and being checked is found, which
# matters because this repository sits inside a folder that synchronises
# itself.
#
# Nothing here creates a repository or publishes anything. It leaves a folder, a
# record of what is in it, and a report saying whether it may become one.

import hashlib
import json
import pathlib
import re
import subprocess
import sys

# An exported copy of this command imports the package out of the export it
# belongs to, and the cached bytecode lands there. In the release rehearsal of
# 24 September 2026 that put six files into the export before anything was
# compared, so both checks refused a folder nothing was wrong with. A command
# that vouches for an export must not change it
# (R-34; found during the R-12 rehearsal).
sys.dont_write_bytecode = True

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent import release
from research_agent.sources import saved_replies

REPO = pathlib.Path(__file__).resolve().parents[1]

# The replay folder and its manifest, as the export names them. The same two
# paths the check in release.py reads.
REPLIES = "data/recorded_replies"
MANIFEST = "data/recorded_replies_manifest.json"

# Written next to the export folder rather than inside it, listing every file
# and its fingerprint. The first commit of the public repository and a fresh
# clone of it are checked against this, so that what was reviewed here is what
# ends up published.
#
# Outside the folder because it is a record of the export, not part of the
# release. Written inside, it would be a file the check does not allow, and
# allowing it would mean the export carries a list of its own contents that
# nobody uses.
INVENTORY_SUFFIX = "_inventory.json"

# Written where a folder is built and left there if it is refused, so that a
# candidate nobody approved never occupies the name a finished release has.
BUILDING_SUFFIX = ".building"

A_COMMIT = re.compile(r"^[0-9a-f]{40}$")
A_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def _git(*arguments):
    """Runs git in this repository and returns its output, or None."""
    try:
        done = subprocess.run(["git", "-C", str(REPO), *arguments],
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout


def why_the_commit_cannot_be_exported(commit):
    """Says why this commit is not one to build a release from, or None."""
    resolved = _git("rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved is None:
        return f"git does not know '{commit}'"

    identifier = resolved.strip()
    if not A_COMMIT.match(identifier):
        return f"'{commit}' did not resolve to a commit"

    waiting = _git("status", "--porcelain")
    if waiting is None:
        return "git could not say whether the working tree is clean"
    if waiting.strip():
        return ("the working tree has changes that are not committed. The "
                "export is taken from the commit, so this would build a "
                "release nobody is looking at")
    return None


def commit_identifier(commit):
    """The full identifier of the commit being exported."""
    return _git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()


def files_in(commit):
    """Every path the commit holds, in the order git lists them."""
    listed = _git("ls-tree", "-r", "--name-only", commit)
    return [] if listed is None else [line for line in listed.splitlines() if line]


def what_to_copy(commit):
    """The paths in that commit which a release may carry."""
    # Taken from the same lists the check reads, so the two cannot disagree about
    # what belongs. A file copied by one rule and judged by another is a file
    # that passes because two people wrote two lists.
    chosen = []
    for relative in files_in(commit):
        if (relative in release.ALLOWED_EXACT
                or release._matches_a_pattern(relative)
                or release._in_an_allowed_tree(relative)):
            chosen.append(relative)
    return sorted(chosen)


def why_the_destination_will_not_do(destination):
    """Says why a release must not be built here, or None."""
    if destination == REPO or REPO in destination.parents:
        return (f"{destination} is inside the repository being exported. A "
                f"release built there becomes part of the next one")
    if destination in REPO.parents:
        return f"{destination} contains the repository being exported"

    # A link and a file are both refused before the folder is asked whether it
    # is empty. Asked first, an existing file made iterdir raise, so a command
    # given a file name stopped with a traceback rather than a sentence.
    #
    # The command resolves the path before it gets here, which follows a link
    # already; this is for anything calling it directly.
    if destination.is_symlink():
        return (f"{destination.name} is a link. A release is built into a "
                f"folder, not through one into wherever it points")
    if destination.exists() and not destination.is_dir():
        return (f"{destination.name} is already there and is not a folder")
    if destination.is_dir() and any(destination.iterdir()):
        return (f"{destination.name} is not empty. A release is built into an "
                f"empty folder so that nothing already there travels with it")

    # The record of the export is written next to the folder, so it is part of
    # what has to be free. Checked only for the folder, an existing record
    # would be overwritten, and a link there would be followed.
    record = inventory_beside(destination)
    if record.is_symlink():
        return f"{record.name} is a link, and the record must be a plain file"
    if record.exists():
        return (f"{record.name} is already there. The record of an export is "
                f"written fresh, not over whatever a previous run left")
    return None


def _out_of_the_commit(commit, relative):
    """The bytes of one file as the commit holds them."""
    return subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{relative}"],
        capture_output=True, check=True).stdout


def replies_kept_private_in(commit):
    """The names of the saved replies the commit's manifest keeps private."""
    # The replay folder holds two kinds of file: recordings of real services,
    # kept for replaying here and never for publishing,
    # and fixtures this project wrote for the public demonstration. The manifest
    # tells them apart, by the replay scope on a recording's latest approval.
    #
    # A commit with no manifest, or one that cannot be read, keeps nothing
    # private here; the check on the export then finds the folder and the
    # manifest disagreeing and says so, which is the right place for that to be
    # said.
    try:
        contents = _out_of_the_commit(commit, MANIFEST)
    except subprocess.CalledProcessError:
        return set()
    try:
        manifest = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return set()
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return set()

    private = set()
    for name, entry in manifest["files"].items():
        if not isinstance(entry, dict):
            continue
        latest = saved_replies.latest_approval(entry)
        if latest is not None and isinstance(latest.get("replay_scope"), dict):
            private.add(name)
    return private


def _manifest_without(contents, names):
    """The manifest's bytes with those entries taken out, written the way the
    application writes a manifest so the export's copy reads the same."""
    manifest = json.loads(contents.decode("utf-8"))
    kept = {}
    for name, entry in manifest["files"].items():
        if name not in names:
            kept[name] = entry
    manifest["files"] = dict(sorted(kept.items()))
    return json.dumps(manifest, indent=2).encode("utf-8")


def copy_out_of(commit, destination):
    """Writes the allowed files of that commit into the folder."""
    # Returns each path with the fingerprint of the bytes git handed over, not
    # of the file afterwards. A list of names was returned to begin with, and the
    # record was made by reading the folder a moment later. A file that changed
    # in between was recorded as what the commit contained, and the record named
    # a commit those bytes never came from.
    #
    # Each file is read out of the commit rather than copied off disk, so a file
    # changed since the commit cannot travel. A link in the working tree cannot
    # be followed either, because git holds contents rather than paths.
    #
    # The replies the manifest keeps private are not copied, and the manifest is
    # written without their entries. The replay folder is the one folder allowed
    # whole, and it holds the fifteen recordings of real services beside the
    # fixtures written for the demonstration; the check on the export refuses
    # any recording kept private, so copying the folder as it stands refused
    # every export that carried a demonstration. Moving the recordings to a
    # folder of their own was the other way, and it was not taken days before
    # the freeze: it would have changed the replay path, the ignore rules and
    # the documents that say where the recordings are, for the same result.
    # The manifest written here is a derived file, and the record of the export
    # fingerprints it as written, so a clone is checked against what left and
    # not against the commit's copy (R-12).
    private = replies_kept_private_in(commit)
    written = {}
    for relative in what_to_copy(commit):
        if relative.startswith(REPLIES + "/") and relative[len(REPLIES) + 1:] in private:
            continue
        contents = _out_of_the_commit(commit, relative)
        if relative == MANIFEST and private:
            contents = _manifest_without(contents, private)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        written[relative] = hashlib.sha256(contents).hexdigest()
    return written


def inventory_beside(destination):
    """Where the record of an export is written."""
    return destination.parent / (destination.name + INVENTORY_SUFFIX)


def write_exclusively(path, text):
    """Writes a file that must not already exist. None, or what stopped it."""
    # Used for the record of an export. Written with the ordinary call, an
    # existing record would be replaced and a link there would be followed to
    # wherever it pointed.
    #
    # The bytes are written as they are given, with no line endings changed.
    # Left to translate, the record on disk carried Windows endings while the
    # text it is compared against carried Unix ones. The two agreed only because
    # reading the file translated them back. A record is read on other machines,
    # so what it holds is decided here.
    #
    # It returned only true or false for a while. A full disk and a folder nobody
    # may write to were both reported as "it was already there", which sends
    # somebody to look at a file that is not the problem.
    try:
        with open(path, "x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError:
        return f"{path.name} is already there"
    except OSError as problem:
        return f"{path.name} could not be written ({problem})"
    return None


def why_the_record_is_not_the_one_written(record, as_written):
    """Says why the record on disk is not what this run wrote, or None."""
    # Every comparison of the export is against the object this run built, in
    # memory, so that a record edited part way through cannot become what the
    # folder is measured by. That leaves the file itself unread, and the file is
    # what a clone will be checked against later, so it is compared here.
    #
    # It answers rather than raises. Reading it directly, a record deleted,
    # replaced by a folder, or written in bytes that are not text stopped the
    # command with a traceback, which is harder to act on than a sentence and
    # easy to read as the tool being broken rather than the release being wrong.
    # A link is refused whatever it points at, because a record is a plain file
    # and one that is not is not answering for itself.
    if record.is_symlink():
        return f"{record.name} is a link, and the record must be a plain file"
    if not record.exists():
        return f"{record.name} is not there"
    if not record.is_file():
        return f"{record.name} is not a file"
    try:
        held = record.read_bytes()
    except OSError as problem:
        return f"{record.name} could not be read ({problem})"
    if held != as_written.encode("utf-8"):
        return f"{record.name} is not the record this run wrote"
    return None


def inventory_of(commit, written, expect_a_demonstration):
    """What was exported, as a record a clone can be checked against."""
    # Built from the fingerprints copy_out_of took of the bytes git handed over.
    # Nothing here reads the folder, so there is no moment between the copy and
    # the record in which a file could change and have the change written down
    # as though it were what the commit held.
    #
    # It records which standard the export was held to as well as which bytes
    # left here. Without that, --no-demonstration was a question asked again by
    # whoever ran the check next: the same folder passed or failed depending on a
    # flag typed later, and nothing said which answer the export was built for.
    return {
        "exported_from": commit,
        "demonstration_required": expect_a_demonstration,
        "files": dict(sorted(written.items())),
    }


def why_the_inventory_will_not_do(inventory):
    """Says why this is not a record of an export, or None."""
    # Read for its shape before anything is compared against it. A record whose
    # files are a list, or whose fingerprints are the word "ok", would otherwise
    # match nothing and disagree with nothing, and a folder checked against it
    # would be reported as being exactly what was exported.
    if not isinstance(inventory, dict):
        return "it is not a record"

    commit = inventory.get("exported_from")
    if not isinstance(commit, str) or not A_COMMIT.match(commit):
        return ("it does not say which commit was exported, as the full "
                "identifier of one")

    # Required rather than assumed when absent. Nothing has been released yet,
    # so there is no older record to be kind to, and a default here would be
    # this file deciding which standard an export was held to.
    if not isinstance(inventory.get("demonstration_required"), bool):
        return ("it does not say whether the export was required to carry a "
                "demonstration")

    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        return "it does not list what was exported"

    for relative in sorted(files, key=str):
        if not isinstance(relative, str) or not relative.strip():
            return "it lists something that is not a path"
        if not isinstance(files[relative], str) \
                or not A_FINGERPRINT.match(files[relative]):
            return f"it does not give a fingerprint for {relative}"
    return None


def read_the_record(record):
    """Reads and checks a record of an export. (inventory, None) or (None, why)."""
    # Read once and passed around afterwards. It used to be opened again for
    # every question asked of it, so the fingerprints could be compared against
    # one version of the file and the standard read out of another, and a check
    # meant to settle what an export is could be answered from two different
    # records without noticing.
    if record.is_symlink():
        return None, f"{record.name} is a link, and a record is read as a plain file"
    if not record.exists():
        return None, f"there is no record of an export at {record}"

    # RecursionError among them. json.loads raises it when a file is nested
    # more deeply than Python can follow, and a record nobody can read to the
    # end is one nothing can be checked against.
    try:
        inventory = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError,
            RecursionError) as problem:
        return None, f"the record of the export could not be read ({problem})"

    wrong = why_the_inventory_will_not_do(inventory)
    if wrong:
        return None, f"{record.name} is not a record of an export: {wrong}"
    return inventory, None


def differences_from_inventory(record, folder, inventory=None):
    """Returns how a folder differs from the export that record describes."""
    # Run against the first commit of the public repository and against a fresh
    # clone of it, so that what was reviewed here is what somebody else gets.
    #
    # The record is named rather than worked out from the folder's name. A clone
    # is called whatever the person cloning it chose, and the record of the
    # export stays where it was written, so deriving one from the other meant the
    # check could only be run in the one place it was least needed.
    #
    # A caller that has already read the record hands it in, so that one reading
    # answers every question. Callers that have not read it leave it out.
    #
    # This was written before anything called it, which is a state this project
    # has been in twice. A record nothing reads does not check anything.
    if inventory is None:
        inventory, wrong = read_the_record(record)
        if wrong:
            return [wrong]

    if folder.is_symlink() or not folder.is_dir():
        return [f"{folder} is not a folder"]

    recorded = inventory["files"]

    problems = []
    here = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        if relative == release.NOT_READ \
                or relative.startswith(release.NOT_READ + "/"):
            continue

        # Links are reported rather than skipped. Skipped, a link added to a
        # clone after the export was the one addition this said nothing
        # about.
        if path.is_symlink():
            problems.append(f"{relative}: is a link, and an export holds files")
            continue
        if path.is_dir():
            continue
        try:
            here[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as problem:
            problems.append(f"{relative}: could not be read ({problem})")

    for relative in sorted(set(recorded) | set(here)):
        if relative not in here:
            problems.append(f"{relative}: was exported and is not here")
        elif relative not in recorded:
            problems.append(f"{relative}: is here and was not exported")
        elif here[relative] != recorded[relative]:
            problems.append(f"{relative}: is not the file that was exported")

    return problems


def _report(lines, heading, closing):
    """Prints a list of findings under a heading."""
    print(f"\n{len(lines)} {heading}:\n")
    for line in lines:
        print(f"  {line}")
    print(f"\n{closing}")


def _judge(folder, expect_a_demonstration, note=""):
    """Asks the two questions of a folder. None when it passes both."""
    # Safe and finished are different states and are reported separately.
    # Telling somebody their release is unsafe when it is only incomplete sends
    # them looking in the wrong place.
    #
    # The note is added to whichever report is printed, so that a run which built
    # something can say where what it built has been left.
    problems = release.problems_in(folder)
    if problems:
        _report(problems, "reasons this must not be published",
                "Nothing was published. Deal with these and run it again."
                + note)
        return 1

    missing = release.what_a_release_is_missing(folder, expect_a_demonstration)
    if missing:
        _report(missing,
                f"files this project requires of a release and "
                f"{folder.name} does not carry",
                "That is a release that is safe and not finished." + note)
        return 1
    return None


def verify(record, folder, expect_a_demonstration):
    """Checks a folder against the record of the export it should be."""
    # All three questions, not only whether the bytes match. A record says the
    # files are the ones that left here; it does not say they were ever allowed
    # to. Somebody who copied a refused export faithfully would have a folder
    # that matched its record perfectly.
    inventory, wrong = read_the_record(record)
    if wrong:
        print(wrong)
        return 1

    differences = differences_from_inventory(record, folder, inventory)
    if differences:
        _report(differences, "differences from what was exported",
                "This is not the export that was checked.")
        return 1

    # The record decides, not the flag. Passing --no-demonstration here would
    # otherwise hold a submission release to the standard of a rehearsal.
    asked_for = inventory["demonstration_required"]
    if asked_for and not expect_a_demonstration:
        print(f"{record.name} records an export that had to carry a "
              f"demonstration, so --no-demonstration does not apply to it and "
              f"has been ignored.")

    if _judge(folder, asked_for) is not None:
        return 1

    print(f"{folder.name} holds exactly what {record.name} says was exported, "
          f"and every file in it is one a release may carry.")
    if asked_for:
        print("Every file this project requires for its repository release "
              "is present.")
    else:
        # A rehearsal was built to a lower standard on purpose, and saying it
        # is complete would be reporting that the thing it deliberately left
        # out is there.
        print(f"{record.name} records a rehearsal, built without the offline "
              f"demonstration. It meets that reduced standard and has not been "
              f"shown to be complete for submission.")
    print(f"The {release.NOT_READ} folder, if there is one, was not read. What "
          f"a published history holds is a separate question from what its "
          f"files hold, and this project has needed two history rewrites to "
          f"answer it.")
    return 0


def main():
    # Both flags are taken out before the rest is counted. Written to remove
    # only one of them, --no-demonstration was left in the list and counted as
    # a folder, so the command that used it always printed its help instead of
    # running.
    arguments = sys.argv[1:]
    check_only = "--check-only" in arguments
    verify_only = "--verify-inventory" in arguments
    expect_a_demonstration = "--no-demonstration" not in arguments
    arguments = [argument for argument in arguments
                 if argument not in ("--check-only", "--no-demonstration",
                                     "--verify-inventory")]

    if verify_only:
        if len(arguments) != 2:
            print(__doc__)
            return 1
        return verify(pathlib.Path(arguments[0]).resolve(),
                      pathlib.Path(arguments[1]).resolve(),
                      expect_a_demonstration)

    if check_only:
        if len(arguments) != 1:
            print(__doc__)
            return 1
        return _judge_only(pathlib.Path(arguments[0]).resolve(),
                           expect_a_demonstration)

    if len(arguments) != 2:
        print(__doc__)
        return 1
    return _build(arguments[0], pathlib.Path(arguments[1]).resolve(),
                  expect_a_demonstration)


def _judge_only(destination, expect_a_demonstration):
    """Reads a folder somebody else built and says whether it may be published."""
    wrong = _judge(destination, expect_a_demonstration)
    if wrong is not None:
        return wrong
    _say_what_this_did_and_did_not_establish(destination)
    return 0


def _build(commit, destination, expect_a_demonstration):
    """Copies a commit out, checks the copies, and only then names them."""
    wrong = why_the_commit_cannot_be_exported(commit)
    if wrong:
        print(f"Nothing was exported: {wrong}")
        return 1

    wrong = why_the_destination_will_not_do(destination)
    if wrong:
        print(f"Nothing was exported: {wrong}")
        return 1

    identifier = commit_identifier(commit)

    # Built next door and moved into place only once every file is written and
    # every check has passed. Copying straight into the destination and failing
    # part way left a folder holding some of a release, which the next run
    # would then refuse for not being empty, and which somebody could mistake
    # for a finished one.
    building = destination.parent / (destination.name + BUILDING_SUFFIX)
    if building.exists() or building.is_symlink():
        print(f"{building.name} is left over from a run that did not finish, "
              f"or from one that was refused. Look at it, then remove it.")
        return 1

    try:
        building.mkdir(parents=True)
        written = copy_out_of(identifier, building)
    except (OSError, subprocess.CalledProcessError) as problem:
        print(f"Copying stopped part way ({problem}). What is in "
              f"{building.name} is not a release and must not be used.")
        return 1

    print(f"Wrote {len(written)} files from {identifier[:12]} into "
          f"{building.name}")

    # The record is written first, from the fingerprints of the bytes git
    # handed over, and every step after it is compared back against those.
    #
    # It was written after the checks to begin with, from a second reading of
    # the folder. A file changed between the two readings was recorded and
    # released without having been checked, and a file that appeared between
    # them was moved into the release with nothing describing it. Reading the
    # folder at all left the same gap one step earlier: whatever was on disk
    # when the record was written became what the record said the commit held.
    #
    # Taken from the copy, the record describes the commit and nothing else,
    # and each step afterwards can ask whether the folder still matches it.
    record = inventory_beside(destination)
    inventory = inventory_of(identifier, written, expect_a_demonstration)
    as_written = json.dumps(inventory, indent=2)
    stopped = write_exclusively(record, as_written)
    if stopped:
        print(f"\nThe record of the export could not be written ({stopped}), "
              f"so there would be nothing to check a clone against. What was "
              f"built is in {building.name} and has not been moved.")
        return 1

    staying = (f" What was built is in {building.name} and has not been "
               f"renamed.")

    # Judged where it was built, and moved only if it passes both questions.
    # Renamed first, a candidate that failed either one sat under the name of a
    # finished release, with a record beside it that matched it exactly. A
    # record says which bytes were exported. It does not say anybody was
    # allowed to export them.
    if _judge(building, expect_a_demonstration, staying) is not None:
        _undo(record, as_written)
        return 1

    # Compared with the record after the checks rather than before them. This
    # catches a file changed at any point since git handed it over: while the
    # copy was running, between the copy and the checks, or while the checks
    # were reading it. It catches a file that appeared as well, which would be
    # an allowed kind of file that every rule above lets through.
    #
    # There was a second comparison here, before the checks. It was removed
    # because no test could tell whether it was there. Every case it caught,
    # this one caught as well.
    if _differs(record, building, staying, inventory):
        _undo(record, as_written)
        return 1

    # The record itself, before the folder is given the name of a release. It
    # was asked about after the move for a while, so a record altered while the
    # checks ran was found only once the candidate already carried the name of
    # a finished release. The folder has not been renamed here and can still be
    # looked at where it was built.
    #
    # Nothing is removed on this path. A record that is not the one this run
    # wrote has been touched by something else, and deleting it would remove
    # the evidence of that.
    wrong = why_the_record_is_not_the_one_written(record, as_written)
    if wrong:
        print(f"\n{wrong}. Nothing has been moved, and neither the export nor "
              f"the record has been taken away.{staying}")
        return 1

    try:
        # An empty folder already at the destination is allowed, and renaming
        # onto one fails on Windows. It is removed rather than written into,
        # because rmdir refuses a folder holding anything and so cannot take
        # away something somebody wanted.
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        building.replace(destination)
    except OSError as problem:
        _undo(record, as_written)
        print(f"\nThe export passed every check and could not be put in place "
              f"({problem}).{staying}")
        return 1

    # And both again under its own name. Moving the folder is the last thing
    # that touches any of this, and what is reported as ready to publish should
    # be what was actually read.
    if _differs(record, destination,
                f" It is in {destination.name} and must not be published "
                f"until somebody has looked at why.", inventory):
        return 1

    wrong = why_the_record_is_not_the_one_written(record, as_written)
    if wrong:
        print(f"\n{wrong}. What is in {destination.name} was checked, and the "
              f"record beside it is not the one it was checked against, so "
              f"nothing should be published from either.")
        return 1

    _say_what_this_did_and_did_not_establish(destination)
    return 0


def _differs(record, folder, note, inventory=None):
    """True when the folder is not what the record describes, having said so."""
    # A run that built the record hands the same object in each time. Reading the
    # file again for every comparison meant the folder was checked against
    # whatever the record said at that moment, and a record edited between two
    # comparisons would have been believed by the second.
    differences = differences_from_inventory(record, folder, inventory)
    if not differences:
        return False
    _report(differences,
            f"differences between {folder.name} and the record of what was "
            f"written into it",
            f"Something changed after the export was written, so what is here "
            f"is not what was checked.{note}")
    return True


def _undo(record, as_written):
    """Removes the record written for an export that did not get its name."""
    # A record beside no folder is worse than none. The next run is refused for
    # it being there, and whoever reads it has a list of an export that does not
    # exist. So the one this run wrote is taken away again.
    #
    # Only the one this run wrote. It used to remove whatever was at that path,
    # so a record something else had changed while the checks ran was deleted
    # when those checks then refused the release, and the evidence of the change
    # went with it. The bytes are compared first, and anything else is left where
    # it is with a reason printed.
    wrong = why_the_record_is_not_the_one_written(record, as_written)
    if wrong:
        print(f"{wrong}, so it has been left where it is. It describes an "
              f"export that was not put in place, and the next run will refuse "
              f"to write over it.")
        return
    try:
        record.unlink()
    except OSError as problem:
        print(f"{record.name} was written and could not be removed "
              f"({problem}). It describes an export that was not put in place.")


def _say_what_this_did_and_did_not_establish(destination):
    """The report at the end, in the terms the checks can actually support."""
    print(f"\nEvery file in {destination.name} is one a release may carry. "
          f"None of the credential or personal-information patterns recognised "
          f"by this project was found in the text it checked or in the names "
          f"and values of decoded JSON fields.")
    print("That is not the same as there being no secret in it: a list of "
          "patterns finds the shapes somebody thought of. Nor is it the same "
          "as being ready. A person still has to read the documents, because "
          "nothing here can.")


if __name__ == "__main__":
    sys.exit(main())
