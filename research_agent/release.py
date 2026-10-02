"""Decides whether a prepared export may become a public repository."""
# Two things reached a public repository without anybody deciding they could.
# The first was article text inside saved service replies. The second was the
# abstracts of eighty-seven papers inside a screening file that the reply check
# never looked at. Both were found by somebody reading, afterwards. What was
# missing was a rule at the boundary, which is what this file is (R-12).
#
# The boundary is a folder, not this repository. An export is prepared by copying
# allowed paths into a new directory, and this reads that directory: the same
# bytes that would be committed. Checking one folder and publishing another
# would let a check pass on something other than what is published.
#
# Everything is refused unless something says it may go. That is the opposite of
# listing what to remove, and it is deliberate: a list of what to remove only
# ever holds the problems somebody has already had. Both incidents were files
# nobody had thought about, and a list of forbidden things would have missed
# both, exactly as it did.
#
# What this cannot do is read prose and tell whose words are in it. So no file
# holding prose travels unless its exact path is written down here, and adding
# one is a deliberate act by a person who has read it. That is the same shape as
# approving a reply: a check cannot establish provenance, and a person can.

import csv
import hashlib
import json
import pathlib
import re

from research_agent.sources import saved_replies

# Files an export cannot be a release without. An export missing these is not a
# lean release, it is a broken one, and the brief asks for a README and a way
# to install and run the thing.
REQUIRED = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
)

# The paths that may travel, written one at a time where they hold prose.
#
# Code and tests are matched by pattern, because a new module is read in review
# like any other code and its shape is checked by everything else here. Prose
# is not: a document can hold an abstract, a screenshot can hold a session
# cookie, and neither is something a rule can see. So each prose file is named,
# and a new one fails this check until somebody adds it having read it.
ALLOWED_PATTERNS = (
    "research_agent/*.py",
    "research_agent/*/*.py",
    "tests/*.py",
    "tests/*/*.py",
    "scripts/*.py",
)

ALLOWED_EXACT = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    # One theme setting and the measurement behind it, read in full. Streamlit
    # reads it from the folder the screen is started in, so a release without
    # it draws the dark outline the session of 16 September failed (UX-01).
    # Allowed and not required: the copy takes every allowed file of the
    # commit, and the required list is for what a release is no submission
    # without.
    ".streamlit/config.toml",
    # The documents a marker needs, decided on 19 September and consolidated on
    # 23 September. The rest of the repository's documents stay out of the
    # release, listed with their reasons in the file NOT_EXPORTED_LIST names
    # below, and the facts the README needs from them are in the README.
    #
    # One development page since 23 September. The traceability matrix, the
    # development log, the remediation log and the account of the design became
    # one document because the brief asks twice for work that is concise and
    # self-contained, and four documents repeating each other is neither. The
    # four stay in the repository, kept back, so nothing written is lost.
    "docs/development.md",
    # The one document on testing, the marker's first stop, decided on
    # 19 September: figures, identifiers, file names and test names,
    # pointing into the development page; it quotes no paper and no reply,
    # and was read in full before it was named here. Since 22 September it
    # holds what the test plan and the functional test report held, and
    # those two are kept back, so a marker reads one page and not two more
    # that repeat it.
    "docs/testing.md",
    # The two diagrams hold class, field, column, function and test names and
    # nothing that came from a paper or a service. Each was read in full
    # before it was named here, like every other prose file.
    "docs/class_diagram.md",
    "docs/sequence_diagram.md",
    "data/hand_screened_sample/README.md",
    "data/hand_screened_sample/screening_labels.csv",
    "data/hand_screened_sample/screening_labels_check.csv",
    # The CLEF review's fifty identifiers with the collection's labels, one
    # identifier and one of two labels a row, under the collection's MIT
    # licence, and the note beside them saying where they came from. Both
    # were read in full; the records fetched for the identifiers stay in the
    # ignored folder beside them (R-12).
    "data/clef_tar/README.md",
    "data/clef_tar/cd011571_fifty.csv",
    "data/recorded_replies_manifest.json",
    # Evidence, one file at a time. The folder was allowed as a whole for a
    # while, which meant any note or screenshot dropped into it travelled
    # unread. So each artefact is named here by a person who has read it.
    #
    # One is named before it exists: the final test output. The completeness
    # check below requires it, and the two lists have to agree. A file
    # required by one and refused by the other is a release that can never
    # be built. It is written from the last full test run before an export,
    # and a person reads it before that export, like every other prose file.
    "evidence/testing/final_test_output.txt",
    "evidence/ux-01-form-focus.png",
    "evidence/ux-01-form-focus-overlap.png",
    # The three early live checks of 26 and 28 August. They are the evidence
    # behind the certificate fault, the truncated replies and the shared
    # request pool, and the testing page and the rows for the two searches
    # and the status check name them. Read in full on 22 September before
    # being named here: DOIs, titles cut to seventy characters, counts and
    # timings; no abstract, key or address.
    # They were tracked and pointed at but on no list, so a release would
    # have carried documents naming files it did not hold (found by the
    # release check on the tree, 22 September).
    "evidence/execution/crossref_live_check_2026-08-26.txt",
    "evidence/execution/pubmed_batch_size_check_2026-08-28.txt",
    "evidence/execution/semantic_scholar_live_check_2026-08-28.txt",
    # The live run of 18 September: its sixteen screenshots, the run log's
    # lines for it and the index that says which picture belongs to which
    # attempt. Decided on 19 September as the evidence of execution the
    # brief asks for. The pictures are fingerprinted below like the two
    # above; the log is read line by line as JSON, with the same checks a
    # JSON file gets.
    "evidence/execution/live-run-2026-09-18/README.md",
    "evidence/execution/live-run-2026-09-18/run_98af89fa5cdc.jsonl",
    "evidence/execution/live-run-2026-09-18/live-01-form-evening.png",
    "evidence/execution/live-run-2026-09-18/live-01-form.png",
    "evidence/execution/live-run-2026-09-18/live-02-headline-evening-37flash.png",
    "evidence/execution/live-run-2026-09-18/live-02-headline-evening.png",
    "evidence/execution/live-run-2026-09-18/live-02-headline-lite.png",
    "evidence/execution/live-run-2026-09-18/live-02-headline.png",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-attempt3-37flash.png",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-evening-37flash.png",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-evening.png",
    "evidence/execution/live-run-2026-09-18/live-03-coverage.png",
    "evidence/execution/live-run-2026-09-18/live-03-headline-attempt3-37flash.png",
    "evidence/execution/live-run-2026-09-18/live-04-paper.png",
    "evidence/execution/live-run-2026-09-18/live-05-revised.png",
    "evidence/execution/live-run-2026-09-18/live-06-store.png",
    "evidence/execution/live-run-2026-09-18/live-07-log.png",
    "evidence/execution/live-run-2026-09-18/live-08-store-again.png",
    # The NCBI notice on the Search page, 20 September, fingerprinted below
    # like the sixteen above (OB-03).
    "evidence/execution/notice-on-the-search-page-2026-09-20.png",
}

# The documents this project has and does not export are listed, each with
# the reason, in a file beside them that is itself not exported. The list
# lives outside this file so that the release, which carries this file, names
# none of the documents it leaves behind; what a reader needs from them is in
# the README and the exported documents.
#
# The allowed list above says what may travel. Nothing said anything about
# the rest, so a document the lists did not account for was simply left out.
# One such document said in its own first paragraph that it is kept as evidence
# the work was planned, and it went missing from a release all the same. A
# test asks that every document in docs/ appears in one list or the other,
# so leaving one out has to be written down.
NOT_EXPORTED_LIST = "docs/not_exported.json"


def kept_back(root):
    """The documents under root kept out of the release, each with its reason."""
    # Empty where the list is absent: an export has no list, and nothing an
    # export is checked for depends on it. The test that classifies every
    # document reads it from the repository, where it is.
    path = root / NOT_EXPORTED_LIST
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# The one folder allowed as a whole, because every file in it is reconciled
# against the manifest below: named there, fingerprinted there, and refused if
# the manifest does not describe it. Nothing else is allowed by folder.
#
# evidence/ was here for a while, which contradicted the rule three paragraphs
# up. A new evidence/notes.md holding an abstract, or any screenshot at all,
# would have been copied and accepted without anybody reading it. Evidence is
# named one file at a time in ALLOWED_EXACT like every other prose file.
ALLOWED_TREES = (
    "data/recorded_replies",
)

# Images an export may carry, each named and each with somebody recorded as
# having looked at it. Nothing here can read the text in a picture, so a
# screenshot can hold a session identifier, a paper, or somebody's desktop, and
# this check would see a valid PNG. The record is the review.
#
# The value is the fingerprint of the bytes somebody read, not their name. A
# name would say a picture was reviewed and still let a different picture travel
# under the same path, which is the whole failure this is here to stop. Who read
# each one is written beside it, because a comparison of bytes cannot hold it.
#
# Both of these were taken on 9 September 2026 for the accessibility check
# on the form (UX-01) and read in
# full before being named here. They show the form with no run loaded, so
# there is no paper and no identifier in either.
REVIEWED_IMAGES = {
    "evidence/ux-01-form-focus.png":
        "5926776a3eeca4bc1c5ea004a49e179f52d30dda623134dc00f6d3bd7f77e575",
    "evidence/ux-01-form-focus-overlap.png":
        "dc794274c19c0f6b934d0c183d5ec1934d7a28af2459e850cf652c9433ef3e45",
    # The sixteen screenshots of the live run of 18 September, taken at the
    # screen during the eight attempts and decided into the release on 19
    # September as the evidence of execution. They show the form, the
    # headline and coverage boxes, one paper's card with its summary and
    # extract (live-04), the cards after the send-back, the results page
    # and the record page; no key, no address, and the card's text is one
    # paper's summary and extract as the screen showed them, which the
    # decision weighed. Fingerprinted on 19 September from the files as
    # copied into the repository.
    "evidence/execution/live-run-2026-09-18/live-01-form-evening.png":
        "cd352de470c70ac17cda831de1456d8c2f108e771b9f5e5ad97f6d95e389db7b",
    "evidence/execution/live-run-2026-09-18/live-01-form.png":
        "018f70abb9023bb91df6149ba55ae99f50a8f57b21de15856ffcdda64d6fe799",
    "evidence/execution/live-run-2026-09-18/live-02-headline-evening-37flash.png":
        "6b8272023b0d70311eebddff938eae0e3f01bad9c2dc2c9a11287ba17fb2728f",
    "evidence/execution/live-run-2026-09-18/live-02-headline-evening.png":
        "2e5081dbf3ee5ba1a2018ecf6911f6bacb45d9041380c1766f2f9b1a4f7c983a",
    "evidence/execution/live-run-2026-09-18/live-02-headline-lite.png":
        "546c679e7b3cd2bc5303ab60e53c05c77092fb1820d6750a792632ec1ef69f6e",
    "evidence/execution/live-run-2026-09-18/live-02-headline.png":
        "cefece813d155607a78cbb7db717c600a490a42602a87747ae7393f3eaa301d1",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-attempt3-37flash.png":
        "2467d0252a82ce35b608baeaad2c955c0ebf8937fbffdc42e56a286bf888c33f",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-evening-37flash.png":
        "dce11cc3cb108095a9218b21b349b3aaad469fcb8419bb0c43fd2e7b8c4cf4b9",
    "evidence/execution/live-run-2026-09-18/live-03-coverage-evening.png":
        "7d9a62ee3785df4bc7bafb0f6c53eaceda8c74d135410d3caa785e0ed72b5915",
    "evidence/execution/live-run-2026-09-18/live-03-coverage.png":
        "d268d787716e794baf3c1c30db0cf16b6e5983c767a6c53202beec4278958d7d",
    "evidence/execution/live-run-2026-09-18/live-03-headline-attempt3-37flash.png":
        "2ad75ebb55e7843e7e6a787e574ba6adfa99871e3026679b99a03d3af6248c33",
    "evidence/execution/live-run-2026-09-18/live-04-paper.png":
        "621909cb80819d306cf1b6d8f09fb95061ab238c41b88ab99fdbf56554754547",
    "evidence/execution/live-run-2026-09-18/live-05-revised.png":
        "b4c8ec16d9068a90f74cdefec3952804cc9d6f07ffa4bceb0bfccfcca888a184",
    "evidence/execution/live-run-2026-09-18/live-06-store.png":
        "bdb53bcf9ef3f8d27bd560c7aa9357409ae19393fd1c21bb626f55dc7b8bf1b3",
    "evidence/execution/live-run-2026-09-18/live-07-log.png":
        "1e5d75ebfceac27da171b5039dd86aa71b9b968a511794cea4120e74bbec9922",
    "evidence/execution/live-run-2026-09-18/live-08-store-again.png":
        "a7564c17244b72544796896e1a8da60e76ceca45c11583fd57deb9d57b244fa5",
    # The NCBI disclaimer and copyright notice on the Search page, taken on
    # 20 September at 2880 by 1470: the heading, the two paragraphs in
    # NCBI's wording and the link to the disclaimer page, under the form's
    # two numbers and above the record's line. The picture the sixteen above
    # lacked, since the form and the results pages were cut off above the
    # notice (OB-03). Read before it was named here; it holds no paper, no
    # key and no address.
    "evidence/execution/notice-on-the-search-page-2026-09-20.png":
        "8b4feaa52c04c2a77c29d89d63bdf0d37aa280f2a12b63cd488a682a89538a57",
}

# The columns a published screening file may hold. Checked rather than the file
# name, because the abstracts that started R-12 would have travelled just as
# happily in a file called anything else. A column nobody listed is refused
# whether it is called abstract, body, notes or summary.
SCREENING_COLUMNS = ["pmid", "include", "rationale"]

# The two screening files a release carries, and what each one says. The
# fingerprint is of the parsed rows, taken when the files were reduced on
# 3 September, and it is the same one tests/unit/test_screening_labels.py
# holds. Written in both places on purpose: this stops a changed benchmark
# leaving the repository, and that stops it changing inside it.
#
# Counting columns is not enough. Three columns with an abstract pasted into
# the reason is a shape nothing else here would refuse, and the fingerprint is
# what notices.
SCREENING_FILES = {
    "screening_labels.csv": {
        "rows": 50, "included": 3,
        "fingerprint":
            "64639fc20fc4c2c9a35e79b0738339329e2f4fbf41c30c148faff00afc53e624"},
    "screening_labels_check.csv": {
        "rows": 50, "included": 5,
        "fingerprint":
            "ffaa59b25c13911ab3899e774a67c60f7d72e512a917f97d7b341e4a94fd7fb5"},
}

# Field names that hold the text of a paper, in any file that has fields. The
# same names the publication check looks for, because a column called abstract
# is an abstract wherever it is.
ARTICLE_TEXT_FIELDS = {"abstract", "abstracttext", "body", "full_text",
                       "fulltext", "article", "text", "content"}

# What a credential looks like when somebody has not meant to include one.
# Matched against the text of every file that travels, under any name, because
# a key pasted into a note is still a key.
#
# This is a list of shapes somebody has thought of. A value it does not match
# is a value it does not match, and nothing more. It does not establish that an
# export holds no secret, and the report says so in those words.
SECRETS = (
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "a GitHub token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "a GitHub token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "an API key"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "a Google API key"),
    # Horizontal space around the separator, as in the environment-file rule
    # below and for the same reason. Written with \s the value could sit on the
    # line after the name, so a document ending a sentence with "password:" was
    # read as giving whatever word began the next line.
    (re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token)"
                r"[ \t]*[=:][ \t]*[\"']?[A-Za-z0-9_\-]{16,}"), "a credential"),
    # Both directions of a cookie: the one a service sets and the one a request
    # sends back. Only set-cookie was listed, so the header carrying a session
    # identifier to the service was the one shape not looked for.
    #
    # The quote before the separator is what makes these find a header saved as
    # JSON, which is the shape everything in the replay set has. Without it the
    # patterns only matched a header written out as a header, and a leak in
    # this project would be a field in a file. The optional bracket is for a
    # service that returns a header's values as a list, which is a shape this
    # project does not save but does not refuse either.
    (re.compile(r"(?i)\b(set-)?cookie[\"']?\s*[=:]\s*\[?\s*"
                r"[\"']?[A-Za-z0-9_\-]+="), "a cookie header"),
    # The session identifier this project actually leaked (R-06), by name and
    # with a value. The name on its own is not matched: the remediation log
    # discusses it at length, and a rule that refused the document describing
    # the leak is a rule somebody turns off.
    (re.compile(r"(?i)\bncbi[_-]?sid[\"']?\s*[=:]\s*[\"']?[A-Za-z0-9]"),
     "an NCBI session identifier"),
    # A whole authorization header, scheme and value. Written as the field name
    # alone it matched the list of header names this project scrubs, which is
    # the code that prevents the leak rather than a leak.
    (re.compile(r"(?i)\bauthorization[\"']?\s*[=:]\s*\[?\s*[\"']?"
                r"(bearer|basic|token)\s+\S"), "an authorization header"),
    # Horizontal space only. Written with \s this crossed the line ending, so a
    # key name with an empty value read as though the following line were its
    # value, and the example environment file, whose values are all empty, was
    # reported as holding a key.
    (re.compile(r"(?i)\b(ncbi_api_key|gemini_api_key|semantic_scholar_api_key)"
                r"[ \t]*=[ \t]*\S+"), "a key from the environment file"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
     "an email address"),
)

# What this list cannot do, written beside the list rather than in a document
# read somewhere else.
#
# Each file is read as the bytes on disk, and a JSON file is read a second
# time as what it decodes to, by _secrets_in_the_decoded_json below, because
# a field name spelled out as escape sequences in the file reads as itself to
# a person and matches nothing as bytes. That second reading was built after
# R-12 named the escaped case as an open limit; what stays open is a name
# assembled at runtime, or one hidden in a form nothing here decodes, and a
# credential nobody has invented a shape for is not looked for at all.
#
# Which is why the report says no value matched a pattern this project
# recognises, and never that there is no secret.

# The domains kept aside for writing examples, which is what a test address and
# a line of documentation use. example.com, .net and .org and the names ending
# .test, .example, .invalid and .localhost are reserved for exactly this, so an
# address at one of them belongs to nobody and reaches nobody. Without this the
# check refused the project's own tests for using someone@example.com.
FOR_EXAMPLES = re.compile(r"(?i)@([a-z0-9.\-]+\.)?"
                          r"(example\.(com|net|org)"
                          r"|test|example|invalid|localhost)$")

# A path written to show the shape of a path rather than to name one. Anything
# holding <name> or a similar placeholder is describing, not carrying.
A_PLACEHOLDER = re.compile(r"[<>{}]")

# Addresses in an export that name somebody's own machine. A path like
# C:\Users\<name>\ says who ran the tests and where, which is not something a
# release has to carry.
#
# The part after the folder has to look like a name. Written as "anything up to
# the next separator" it also matched the source of this pattern where it
# appears in a file, so the check refused itself.
#
# A Windows path written with forward slashes, which is the form Python prints,
# is caught by the third alternative rather than the first: C:/Users/<name>
# holds /Users/<name>. The first was widened to accept either separator for a
# while, and a mutation test showed nothing could tell whether that change was
# there, because this already refused every case it was meant to add. It was
# taken out again. All four forms are checked in tests/unit/test_release.py.
#
# Inside JSON a path is usually written with each separator doubled. That form
# is not matched here: this looks for one separator and the file holds two. It
# is found by _secrets_in_the_decoded_json, which reads the file back as what
# it decodes to.
PERSONAL_PATHS = re.compile(r"(?i)([A-Z]:\\Users\\[A-Za-z0-9._\- ]+"
                            r"|/home/[A-Za-z0-9._\-]+"
                            r"|/Users/[A-Za-z0-9._\-]+)")

# The suffixes this check knows how to read. Anything else is refused rather
# than skipped. An archive or a compiled file is not safe because nothing here
# could look inside it, and skipping a file it cannot read would let that file
# pass unchecked.
READABLE = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".toml", ".cfg",
            ".ini", ".example", ".gitignore", ""}

# Names that belong in one place in a release: the README's statement on the
# use of generative AI, which the brief asks for. That statement is where the
# tools used are described, so the account of them is complete and in one
# place; everywhere else the record says what was done, found or decided, and
# when. The wording was reviewed and revised for clarity on 19 September. The
# names are matched whatever their case, and the third is matched as a
# phrase, so the deduplication code's "body owner" is not caught.
#
# Written in pieces, because this file travels in the release and the check
# reads it: written whole, the rule would refuse its own source, which is
# the reason the tests build their keys the same way.
NAMED_IN_ONE_PLACE = ("cla" + "ude", "co" + "dex", "the " + "owner")
THE_ONE_PLACE = "## Use of generative AI"

# A clock time in a document. The exported documents carry dates only, from
# the same decision: a time of day in a plan or a log is scheduling, not a
# fact a reader of the submission needs. Evidence files keep their times,
# because they are the machine's record; the screenshots' index in the
# evidence folder is one of them.
A_CLOCK_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
DOCUMENTS = (".md",)
EVIDENCE = "evidence/"

# Images an export may carry as evidence. They are not read for text, and the
# report says so, because a screenshot can hold a session identifier or an
# abstract and nothing here can see it. They are listed by exact path like
# prose for that reason.
IMAGES = {".png", ".jpg", ".jpeg"}

# The one name never read. A repository keeps its commits under .git, and the
# same checks are meant to be run against a fresh clone of the published
# repository as against the export it was built from. Read there, every object
# git wrote would be reported as a file nothing allows, several hundred of
# them, and the report would be useless.
#
# This is a limit, not a clearance, and the command says so where it reports.
# What is in a published history is a different question from what is in its
# files, and this project needed two history rewrites to learn that.
#
# An export built by scripts/prepare_release.py has no .git of its own: the
# files are read out of a commit and written into an empty folder.
NOT_READ = ".git"


# What this project has decided a release of it must contain. Kept apart from
# the checks above, and reported separately, because the two answer different
# questions and get confused when they share one list.
#
# problems_in asks whether anything here must not be published. This asks
# whether what was decided on is here at all. An export can hold nothing the
# checks refuse and still be useless: four packaging files and an empty package
# would satisfy every rule above while being no submission at all.
#
# It was called FOR_THE_BRIEF, which claimed more than it could. The brief asks
# for a README, for evidence of development and for the remediation to be
# documented; it does not name a project plan or a file of architecture
# decisions. Those are here because this project chose them as its evidence,
# and one of them says in its own first paragraph that it belongs in the
# repository. Naming the list after the brief made a choice of this project's
# read as a requirement of the assignment.
REQUIRED_FOR_THIS_RELEASE = {
    "README.md": "the README the brief asks for",
    "LICENSE": "the licence",
    "NOTICE.md": "the notice saying what the licence does and does not cover",
    "requirements.txt": "the dependencies needed to install it",
    # The README's own instructions name both of these: it says to install the
    # test dependencies from the first and to copy the second to .env before
    # running. They were allowed but not asked for, so an export that lost
    # them passed this check while its installation and testing instructions
    # told the reader to use files it did not carry.
    "requirements-dev.txt": "the dependencies the README's test instructions "
                            "install",
    ".env.example": "the settings file the README tells the reader to copy "
                    "before running",
    ".gitattributes": "what git hands over, so a clone holds the same bytes "
                      "the record of the export describes",
    "pyproject.toml": "how the project is built and its tests are run",
    "docs/development.md": "the design requirements and how they were met, how "
                           "the Unit 6 design became this implementation, "
                           "evidence of development over the module, and the "
                           "remediation the brief asks to be documented",
    "docs/testing.md": "the one page on testing a reader starts from: what was "
                       "planned, what was run and what it found, with the "
                       "functional testing the brief requires",
    "docs/class_diagram.md": "the class diagram the tutor's feedback asked for",
    "docs/sequence_diagram.md": "the sequence diagram the tutor's feedback asked for",
    "data/hand_screened_sample/README.md": "how the benchmark was drawn",
    "data/hand_screened_sample/screening_labels.csv": "the benchmark",
    "data/hand_screened_sample/screening_labels_check.csv": "the second sample",
    "evidence/testing/final_test_output.txt": "the test output the brief asks for",
}

# The brief asks for evidence of execution, and the folder the allowed list
# names one file at a time is where this project put it. Nothing asked for any
# of it, so an export that lost the live run's pictures would have passed this
# check while the README and the testing page went on citing them.
#
# Taken from the allowed list rather than written out a second time. The two
# lists have to agree, and twenty-two paths kept by hand would part company the
# first time one was renamed.
for _named_in_the_allowed_list in sorted(ALLOWED_EXACT):
    if _named_in_the_allowed_list.startswith("evidence/execution/"):
        REQUIRED_FOR_THIS_RELEASE[_named_in_the_allowed_list] = (
            "one of the execution records this release carries as evidence "
            "that the agent ran")


def what_a_release_is_missing(export_root, expect_a_demonstration=True):
    """Returns what this project requires of a release and this export lacks."""
    # Separate from problems_in on purpose. That one refuses things; this one
    # asks whether the thing is finished. A release that fails this is not
    # dangerous, it is incomplete, and telling somebody the difference is the
    # point of two lists.
    #
    # What it can answer is whether a file is there. Whether the file says what
    # the assignment needs it to say is a person's reading, so nothing here is
    # worded as the assignment being satisfied.
    #
    # expect_a_demonstration is off for an export made to try the machinery
    # before the fixtures exist. It is on for anything that will be submitted,
    # because a demonstration nobody can run is not a demonstration.
    if not export_root.is_dir():
        return [f"{export_root} is not there"]

    # Every one of these asks for a file. Asked whether the path existed, it
    # took a folder as an answer. A folder called NOTICE.md, another called
    # research_agent/fake.py and two more named like tests made an empty export
    # report as a finished submission. The safety check did not notice either,
    # because it walks files and steps over ordinary folders. Neither check saw
    # a folder given a file's name.
    missing = []
    for relative, what in REQUIRED_FOR_THIS_RELEASE.items():
        if not (export_root / relative).is_file():
            missing.append(f"{relative}: {what}")

    application = [path for path in (export_root / "research_agent").rglob("*.py")
                   if path.is_file() and path.name != "__init__.py"]
    if len(application) < 2:
        missing.append("research_agent/: the application itself")

    for folder, what in (("tests/unit", "the unit tests"),
                         ("tests/integration", "the integration tests")):
        if not [path for path in (export_root / folder).glob("test_*.py")
                if path.is_file()]:
            missing.append(f"{folder}/: {what}")

    if expect_a_demonstration:
        replies = export_root / "data" / "recorded_replies"
        if not replies.is_dir() or not [path for path in replies.glob("*.json")
                                        if path.is_file()]:
            missing.append("data/recorded_replies/: at least one reply the "
                           "demonstration can replay without a network")

    return missing


def _matches_a_pattern(relative):
    """True when this path is allowed by one of the code patterns."""
    # Matched from the root of the export, not from the end of the path.
    #
    # match() compares a relative pattern against the last parts of a path, so
    # research_agent/*/*.py accepted private/research_agent/agents/leak.py, and
    # so did docs/research_agent/agents/leak.py. A folder somebody had put aside
    # could carry anything out of the repository by being shaped like the
    # application, and the same rule chose the files to copy and judged them
    # afterwards, so both steps agreed it was allowed. full_match compares the
    # whole path.
    for pattern in ALLOWED_PATTERNS:
        if pathlib.PurePosixPath(relative).full_match(pattern):
            return True
    return False


def _in_an_allowed_tree(relative):
    """True when this path is inside a folder checked by its contents."""
    for tree in ALLOWED_TREES:
        if relative == tree or relative.startswith(tree + "/"):
            return True
    return False


def _every_file_in(root):
    """Every file under this folder, with its path relative to it."""
    # A link to a folder is kept in the list rather than stepped over with the
    # ordinary folders, so that _escapes can refuse it. Stepped over, a link
    # would be neither refused nor read, and an export could carry one that
    # nobody had looked at.
    #
    # A repository's own .git folder is not among them. See NOT_READ.
    found = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == NOT_READ or relative.startswith(NOT_READ + "/"):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        found.append((path, relative))
    return found


def _escapes(root, path):
    """True when this path is a link, or resolves outside the export."""
    # A link is refused rather than followed. Following one would mean checking a
    # file inside the export and publishing whatever the link pointed at, which
    # is the same mistake as checking one tree and publishing another.
    if path.is_symlink():
        return True
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        return True
    return False


def _fingerprint_of(path):
    """The fingerprint of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_of(path):
    """The file's text, or None if it is not text this check can read."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _line_of(text, position):
    """Which line of the text this position falls on, counting from one."""
    return text.count("\n", 0, position) + 1


def _secrets_in(text, relative, where=None):
    """The reasons this text must not be published, if there are any."""
    # What was found is never repeated back. An earlier version printed the
    # matched text to make the report easier to act on, which put the key on the
    # screen, into whatever the terminal saved, and into any evidence file the
    # run was captured in. A check meant to prevent a leak must not cause one.
    # The report gives the file, the kind of thing found and the line. The person
    # opens the file.
    #
    # A caller reading decoded JSON gives the field instead of a line, because
    # the decoded value is not in the file at that position.
    problems = []
    for pattern, what in SECRETS:
        for found in pattern.finditer(text):
            if what == "an email address" and FOR_EXAMPLES.search(found.group(0)):
                continue
            if where is None:
                place = f"line {_line_of(text, found.start())}"
            else:
                place = where
            problems.append(f"{relative}, {place}: holds {what}")
            break

    # Web addresses are taken out first. A published policy page on a service's
    # own site is not a folder on anybody's computer, and reading it as one
    # made this refuse a document for citing its sources.
    without_links = re.sub(r"https?://\S+", "", text)
    for found in PERSONAL_PATHS.finditer(without_links):
        if A_PLACEHOLDER.search(found.group(0)):
            continue
        if where is None:
            place = f"line {_line_of(without_links, found.start())}"
        else:
            place = where
        problems.append(f"{relative}, {place}: holds a path naming somebody's "
                        f"own machine")
        break

    return problems


def _outside_the_one_place(text, relative):
    """The text a release is read for names in: all of it, except the README's
    statement on the use of generative AI.
    """
    # The statement runs from its heading to the next heading of the same level
    # or the end of the file. Only the README has one; any other file is read
    # whole. The statement is blanked rather than cut out, keeping its line
    # breaks, so a line number reported for the rest of the file is the line
    # a person will find it on.
    if relative != "README.md":
        return text
    start = text.find(THE_ONE_PLACE)
    if start == -1:
        return text
    end = text.find("\n## ", start + len(THE_ONE_PLACE))
    if end == -1:
        end = len(text)
    blanked = re.sub(r"[^\n]", " ", text[start:end])
    return text[:start] + blanked + text[end:]


def _names_in(text, relative):
    """The names this release keeps to the README's statement, found elsewhere."""
    # One report per name and file, with the first line it is on, so the person
    # opens the file rather than being sent every line of it.
    #
    # The words of a name are matched with any white space between them, not
    # the one space they are written with, because wrapped prose breaks a
    # phrase across two lines: read line by line on 19 September, the check
    # passed four docstrings that carried "the" at the end of one line and
    # "owner" at the start of the next.
    problems = []
    read = _outside_the_one_place(text, relative)
    for name in NAMED_IN_ONE_PLACE:
        pattern = r"\s+".join(re.escape(word) for word in name.split())
        found = re.search(pattern, read, re.IGNORECASE)
        if found is not None:
            problems.append(f"{relative}, line {_line_of(read, found.start())}: "
                            f"names \"{name}\", which a release names only in "
                            f"the README's statement on the use of generative AI")
    return problems


def _clock_times_in(text, relative):
    """A time of day in an exported document, which carries dates only."""
    # Evidence files are not documents and keep their times. One report per
    # file, with the first line.
    if not relative.endswith(DOCUMENTS) or relative.startswith(EVIDENCE):
        return []
    found = A_CLOCK_TIME.search(text)
    if found is None:
        return []
    line = _line_of(text, found.start())
    return [(f"{relative}, line {line}: holds a clock time, and the documents "
             f"of a release carry dates only")]


def _problems_with_a_log(text, relative):
    """The run log's lines, each read as the JSON it is."""
    # A log is one JSON record per line, so the file as a whole is not JSON and
    # the reading a .json file gets would refuse it. Each line gets that reading
    # instead: the secret patterns against the decoded fields, and the check
    # that no field holds the text of a paper.
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{relative}, line {number}: is not readable JSON")
            continue
        problems.extend(_secrets_in_the_decoded_json(parsed, relative,
                                                     where=f"line {number}"))
        holding = _article_text_fields_in(parsed)
        if holding:
            problems.append(f"{relative}, line {number}: has a field holding "
                            f"the text of a paper ({', '.join(sorted(holding))})")
    return problems


def _secrets_in_the_decoded_json(value, relative, where="the top level",
                                 under=()):
    """The same rules, asked of what a JSON file decodes to."""
    # The file is scanned as it is written as well, and this is the second pass
    # over it. JSON may write any character as an escape, so a field name spelled
    # out in escapes reads as itself to a person and does not appear in the file
    # at all. A Windows path is the same problem: JSON doubles every separator,
    # so the file holds two where the rule for a path looks for one.
    #
    # A name and its value are put back together as "name: value" before being
    # read, because that is the shape these rules are written for. Read apart, a
    # field called authorization and a value beginning Bearer match nothing.
    #
    # Every name a value sits under is paired with it, not only the nearest one.
    # Carrying the nearest name alone, a value one container further down was
    # read under whatever that container was called. A list of values under
    # authorization was found; the same list one object deeper was not.
    #
    # Names are read as well as values, because a credential can sit in a field
    # name.
    #
    # Where something was found is given by counting fields, not by naming one.
    # Naming the field put the field's name in the report, so where the name was
    # the credential the report printed it.
    problems = []
    if isinstance(value, dict):
        for position, (name, inside) in enumerate(value.items(), start=1):
            here = f"field {position} of {where}"
            problems.extend(_secrets_in(str(name), relative,
                                        f"in the name of {here}"))
            if not isinstance(inside, (dict, list)):
                problems.extend(_paired_with_every_name(
                    _as_text(inside), (*under, str(name)), relative, here))
            problems.extend(_secrets_in_the_decoded_json(
                inside, relative, here, (*under, str(name))))
    elif isinstance(value, list):
        for position, item in enumerate(value, start=1):
            here = f"item {position} of {where}"
            if not isinstance(item, (dict, list)):
                text = _as_text(item)
                problems.extend(_secrets_in(text, relative, f"in {here}"))
                problems.extend(_paired_with_every_name(text, under, relative,
                                                        here))
            problems.extend(_secrets_in_the_decoded_json(item, relative, here,
                                                         under))
    return problems


def _paired_with_every_name(text, names, relative, here):
    """Reads this value once under each of the names it sits below."""
    problems = []
    for name in names:
        problems.extend(_secrets_in(f"{name}: {text}", relative, f"in {here}"))
    return problems


def _as_text(value):
    """A JSON value that is not a container, written out as text."""
    # Numbers and true and false are read too. A session identifier recorded as a
    # number is still a session identifier, and skipping everything that was not
    # a string left that one shape unlooked at.
    if value is None:
        return ""
    return str(value)


def _article_text_fields_in(value):
    """Every field name in this JSON that means the text of a paper."""
    found = set()
    if isinstance(value, dict):
        for name, inside in value.items():
            if str(name).lower() in ARTICLE_TEXT_FIELDS:
                found.add(str(name))
            found |= _article_text_fields_in(inside)
    elif isinstance(value, list):
        for item in value:
            found |= _article_text_fields_in(item)
    return found


def canonical_rows(rows):
    """The text a screening file's rows are fingerprinted as."""
    # Its parsed rows rather than its bytes, so saving the file again with
    # different line endings is not a change to what it says, while changing a
    # paper, a decision or a reason is.
    return json.dumps([[row["pmid"], row["include"], row["rationale"]]
                       for row in rows], ensure_ascii=False, sort_keys=True)


def _problems_with_a_screening_file(path, relative):
    """The reasons this screening file must not be published."""
    # Its columns are checked, not its name. The eighty-seven abstracts in R-12
    # were in files whose names said abstracts, and renaming them would have
    # changed nothing about what they held.
    #
    # The columns alone are not enough either. Three columns with an abstract
    # pasted into the reason satisfies every rule about shape, so the file is
    # compared against the fingerprint taken when it was reduced. That is the
    # same fingerprint the benchmark test holds, and it is what makes this a
    # check on the contents rather than on their arrangement.
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as problem:
        return [f"{relative}: could not be read as a table ({problem})"]

    if not rows:
        return [f"{relative}: holds no rows"]

    columns = list(rows[0])
    if columns != SCREENING_COLUMNS:
        return [(f"{relative}: holds the columns {columns}, and a screening "
                 f"file published here holds {SCREENING_COLUMNS}")]

    problems = []
    name = pathlib.PurePosixPath(relative).name
    expected = SCREENING_FILES.get(name)
    if expected is None:
        return [(f"{relative}: is not one of the screening files a release "
                 f"carries")]

    if len(rows) != expected["rows"]:
        problems.append(f"{relative}: holds {len(rows)} rows and the sample is "
                        f"{expected['rows']}")

    included = sum(1 for row in rows if row["include"] == "include")
    if included != expected["included"]:
        problems.append(f"{relative}: holds {included} included and the sample "
                        f"decided {expected['included']}")

    identifiers = [row["pmid"] for row in rows]
    if not all(pmid.isdigit() for pmid in identifiers):
        problems.append(f"{relative}: holds something that is not an identifier")
    if len(set(identifiers)) != len(identifiers):
        problems.append(f"{relative}: repeats an identifier")
    if not all(row["include"] in ("include", "exclude") for row in rows):
        problems.append(f"{relative}: holds a decision that is neither include "
                        f"nor exclude")
    if not all(row["rationale"].strip() for row in rows):
        problems.append(f"{relative}: holds a decision with no reason")

    fingerprint = hashlib.sha256(
        canonical_rows(rows).encode("utf-8")).hexdigest()
    if fingerprint != expected["fingerprint"]:
        problems.append(f"{relative}: does not say what it said when it was "
                        f"reduced. A reason holding an abstract would look "
                        f"like this, and so would an edited decision")

    return problems


def _problems_with_a_table(path, relative):
    """The reasons any other table must not be published."""
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as problem:
        return [f"{relative}: could not be read as a table ({problem})"]

    if not rows:
        return []

    holding = [name for name in rows[0]
               if str(name).lower() in ARTICLE_TEXT_FIELDS]
    if holding:
        return [(f"{relative}: has a column holding the text of a paper "
                 f"({', '.join(holding)})")]
    return []


def _problems_with_the_replies(root):
    """The reasons the saved replies in this export must not be published."""
    # Every file in the replies folder is asked about, and the manifest is what
    # answers. A recording kept for replaying in the private sandbox fails here
    # by definition, which is the rule written when replay and publication were
    # separated; a fixture written for this project passes if the manifest says
    # where it came from and the file is the one it describes.
    folder = root / "data" / "recorded_replies"
    manifest_path = root / "data" / "recorded_replies_manifest.json"

    # Both absent is a release with no replay set, which is allowed: the system
    # runs against the services, and an offline demonstration is something the
    # release may or may not carry.
    if not folder.exists() and not manifest_path.exists():
        return []

    # One without the other is not. An earlier version returned here as soon as
    # the folder was missing, so a manifest listing fifteen replies that were
    # not there passed the check with nothing said, and a run that reported it
    # as clean was quoted as evidence that the check worked.
    if not manifest_path.exists():
        return [("data/recorded_replies_manifest.json: replies are here and "
                 "nothing describes them")]

    try:
        manifest = saved_replies.read_manifest(manifest_path)
    except saved_replies.ManifestProblem as problem:
        return [f"data/recorded_replies_manifest.json: {problem}"]

    problems = []
    listed = manifest["files"]

    # A manifest saying it is version two has to carry what version two says it
    # carries. Left out, a reader is missing the part that explains what an
    # approval means and which rewrites the recorded commits belong to, and
    # nothing here noticed because every check went straight to the files.
    for field in ("what_approval_means", "what_repository_head_at_review_means"):
        if not str(manifest.get(field, "")).strip():
            problems.append(f"data/recorded_replies_manifest.json: does not "
                            f"carry {field}, which version two says it does")

    # The record of the rewrites has to be there and has to be a list. It does
    # not have to hold anything.
    #
    # Written as "must not be empty", this asked a public repository to claim a
    # rewrite it had never had. The two rewrites this project made were of
    # this sandbox's history, for the contact details and the publication
    # rights (R-11). A repository whose first commit is the export has no
    # earlier history and nothing to say about it. An empty list is the true
    # answer
    # there, and an invented entry would be a worse record than none.
    if not isinstance(manifest.get("history_rewrites"), list):
        problems.append("data/recorded_replies_manifest.json: does not carry "
                        "history_rewrites as a list, which version two says "
                        "it does")

    # An entry that is a number or a list would reach latest_approval and raise
    # rather than produce a reason. A release check that stops with a traceback
    # tells somebody less than one that says which file is wrong.
    for name, entry in sorted(listed.items()):
        if not isinstance(entry, dict):
            problems.append(f"data/recorded_replies_manifest.json: the entry "
                            f"for {name} is not a record")
    listed = {name: entry for name, entry in listed.items()
              if isinstance(entry, dict)}

    if not folder.exists():
        for name in sorted(listed):
            problems.append(f"data/recorded_replies/{name}: the manifest "
                            f"describes it and it is not there")
        return problems

    # Everything in the folder, not only the JSON at the top of it. Looking at
    # one shape in one place would leave every other shape unchecked, and this
    # is the one folder allowed as a whole.
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(root).as_posix()

        if path.is_dir():
            problems.append(f"{relative}: the replay set is a flat folder, and "
                            f"nothing checks what is inside a folder in it")
            continue

        if path.suffix.lower() != ".json":
            problems.append(f"{relative}: the replay set holds saved replies, "
                            f"and this is not one")
            continue

        if path.parent != folder:
            problems.append(f"{relative}: is not at the top of the replay set, "
                            f"so no manifest entry can name it")
            continue

        entry = listed.get(path.name)
        if entry is None:
            problems.append(f"{relative}: no manifest entry describes it")
            continue

        for reason in saved_replies.why_it_cannot_be_exported(entry, path):
            problems.append(f"{relative}: {reason}")

    for name in sorted(listed):
        if "/" in name or "\\" in name or name != pathlib.Path(name).name:
            problems.append("data/recorded_replies_manifest.json: names an "
                            "entry that is not a plain file name")
        elif not (folder / name).exists():
            problems.append(f"data/recorded_replies/{name}: the manifest "
                            f"describes it and it is not there")

    return problems


def problems_in(export_root):
    """Returns every reason this export must not become a public repository."""
    # An empty list means each file in it is one somebody allowed, is the kind of
    # thing this check can read, and, for the files that could carry somebody
    # else's words, is described by a record saying where it came from. None of
    # the credential or personal-information patterns written above was found in
    # the text it checked or in the names and values of decoded JSON fields.
    #
    # It does not mean the export holds no secret. It means nothing matching a
    # shape somebody thought of was found, which is a smaller statement and the
    # only one a list of patterns can make. Nor does it mean the export is
    # correct, or that the documents are true, or that a person would be happy to
    # publish it. It means nothing in it is a thing this project has already
    # published by accident twice.
    problems = []

    if not export_root.exists():
        return [f"{export_root} is not there"]

    files = _every_file_in(export_root)
    if not files:
        return [f"{export_root} is empty, and an empty export is not a release"]

    present = {relative for _, relative in files}
    for name in REQUIRED:
        if name not in present:
            problems.append(f"{name}: a release cannot be without it")

    for path, relative in files:
        if _escapes(export_root, path):
            problems.append(f"{relative}: is a link, or points outside the "
                            f"export")
            continue

        allowed = (relative in ALLOWED_EXACT
                   or _matches_a_pattern(relative)
                   or _in_an_allowed_tree(relative))
        if not allowed:
            problems.append(f"{relative}: nothing allows it into a release")
            continue

        suffix = path.suffix.lower()
        if suffix in IMAGES:
            # Read by a person, not by this, so what is required is the record
            # that a person read it. Skipping images because nothing here can
            # inspect them is the same mistake as passing a reply because
            # nothing could read it.
            reviewed = REVIEWED_IMAGES.get(relative)
            if reviewed is None:
                problems.append(f"{relative}: is a picture, and nothing here "
                                f"can read what is in one. No record says "
                                f"anybody has looked at it")
            elif reviewed != _fingerprint_of(path):
                problems.append(f"{relative}: is not the picture somebody "
                                f"looked at")
            continue

        if suffix not in READABLE:
            problems.append(f"{relative}: is a kind of file this check cannot "
                            f"read, so nothing can say what is in it")
            continue

        text = _text_of(path)
        if text is None:
            problems.append(f"{relative}: could not be read")
            continue

        problems.extend(_secrets_in(text, relative))
        problems.extend(_names_in(text, relative))
        problems.extend(_clock_times_in(text, relative))

        if suffix == ".jsonl":
            problems.extend(_problems_with_a_log(text, relative))

        if suffix == ".csv":
            if relative.startswith("data/hand_screened_sample/"):
                problems.extend(_problems_with_a_screening_file(path, relative))
            else:
                problems.extend(_problems_with_a_table(path, relative))

        if suffix == ".json":
            # A file nested deeper than Python will follow is refused, not
            # skipped and not allowed to stop the run. Both readings below go
            # down a level at a time, so a file two thousand lists deep raised
            # an error the command did not catch, and the whole check stopped
            # with a traceback instead of saying which file was wrong.
            #
            # A file nothing can read all the way through is exactly the case
            # the rule above is for: what cannot be inspected is refused.
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as problem:
                problems.append(f"{relative}: is not readable JSON ({problem})")
            except RecursionError:
                problems.append(f"{relative}: is nested too deeply to read")
            else:
                try:
                    # Read a second time, as what it decodes to rather than as
                    # the text above. Every reply this project saves is JSON,
                    # so a leak here would be a field in a file, and a rule
                    # matching the text does not see one written in escapes.
                    problems.extend(_secrets_in_the_decoded_json(parsed,
                                                                 relative))

                    # The replay set is asked about below instead, where the
                    # manifest says what each file is allowed to hold.
                    if relative.startswith("data/recorded_replies"):
                        holding = set()
                    else:
                        holding = _article_text_fields_in(parsed)
                except RecursionError:
                    problems.append(f"{relative}: is nested too deeply to read")
                else:
                    if holding:
                        problems.append(f"{relative}: has a field holding the "
                                        f"text of a paper "
                                        f"({', '.join(sorted(holding))})")

    problems.extend(_problems_with_the_replies(export_root))
    return sorted(set(problems))
