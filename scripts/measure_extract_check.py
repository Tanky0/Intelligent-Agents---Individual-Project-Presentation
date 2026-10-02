"""Counts how often the extract check rejects a window cut from an abstract."""
# `processing/validation.py` answers one question: do the extract's words appear
# in the paper. It is deliberately strict about where a match may sit, because a
# changed number inside a real sentence is the failure that matters. Strictness
# costs the other way, and the SC-02 row has said since it was written that
# nobody had measured the cost.
#
# This is a step towards that measurement rather than the measurement itself.
# It cuts windows of whole words out of real abstracts and counts how many the
# rule rejects. What it reports is a rejection rate for windows of these shapes,
# and that is not the same as a false-refusal rate.
#
# The difference matters and is the reason this file says so twice. A window is
# a substring of the abstract, but a substring is not always an honest
# quotation: cut "improved in 28" out of "improved in 28 - 30 patients" and the
# number has lost half of itself. The rule rejects that on purpose, and
# `tests/unit/test_validation.py` requires it. A rejection counted here may
# therefore be the rule working. Telling the two apart needs quotations a person
# or the model actually produced, judged one by one, which is what remains open.
#
# The abstracts come out of replies this project captured from PubMed. Nothing
# here reaches a service and nothing here asks the model, so it costs none of
# the twenty model requests a day (OB-07) and can be repeated exactly from the
# same replies.
#
# There is no result yet. Run against `data/recorded_replies/` it reports that
# there is nothing to measure: of the fifteen files there, seven are Crossref
# records with no abstract field, four are PubMed link replies and four are
# searches returning identifiers. None of them carries abstract text. That is
# what those files hold today; it is not a claim about every reply this project
# ever received. The replies the live run captures on 17 to 18 September are
# what this is waiting for, and the folder to read can be given on the command
# line.
#
# No paper text is printed. A rejection is described by what the window starts
# and ends with, and the count of each kind is what a reader needs to judge the
# rule. Printing the windows would put paper text into `evidence/`, which is the
# boundary R-11 was spent enforcing.
#
# Run it with the project's interpreter, from the repository root:
#
#     python scripts/measure_extract_check.py [folder of saved replies]

import json
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_agent.processing.validation import _CARRIES_A_NUMBER_ON, is_in_the_paper
from research_agent.sources import pubmed

REPLIES = Path(__file__).resolve().parent.parent / "data" / "recorded_replies"

# How long a window to cut, in words. These are sampling choices, not claims
# about how long a quotation is. Short windows are included because a window
# has two ends whatever its length, so shorter ones put more ends into the
# count, and the ends are what the rule is strict about. Longer ones are
# closer to what the Curation Agent is asked for, which is a sentence.
LENGTHS = (3, 6, 12)


def abstracts_from(path):
    """The abstracts in one saved reply, and whether it could be read."""
    # Returns the papers and a reason the file could not be used, or None when
    # there was nothing wrong with it. A reply that cannot be parsed is reported
    # rather than passed over: a measurement that quietly reads fewer papers
    # than it was given is the failure this project has already had twice, and
    # it looks exactly like a measurement that worked.
    #
    # A reply that is not a PubMed article set is not a problem. Searches, link
    # replies and Crossref records are ordinary things to find in that folder
    # and hold no abstracts, so they are outside what this reads rather than
    # faults in it.
    #
    # The source's own reader is used rather than a second one written here. A
    # measurement that parsed the XML its own way would be measuring the rule
    # against text no run would ever hold, and the difference would be
    # invisible.
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        return [], f"{path.name}: the file could not be read ({problem})"

    body = record.get("body", "")
    if "<PubmedArticle" not in body:
        return [], None
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as problem:
        return [], f"{path.name}: the PubMed reply could not be parsed ({problem})"

    found = [pubmed._read_article(article) for article in root.iter("PubmedArticle")]
    return [paper for paper in found if paper.abstract], None


def windows_from(abstract):
    """Runs of whole words, cut out of the abstract character for character."""
    # Cut at word boundaries rather than at a fixed number of characters, so
    # that a window is made of whole words the way a quotation is. The text
    # between the first and last word is taken from the abstract itself, so the
    # spacing and the punctuation inside it are the paper's own.
    #
    # A window can still cut through something larger than a word, a numerical
    # expression above all, which is why what comes out of this is called a
    # window and not a quotation.
    words = []
    at = 0
    for word in abstract.split():
        start = abstract.index(word, at)
        words.append((start, start + len(word)))
        at = start + len(word)

    for length in LENGTHS:
        for first in range(len(words) - length + 1):
            start = words[first][0]
            end = words[first + length - 1][1]
            yield abstract[start:end]


def kind_of(character):
    """Which of five groups this character falls into."""
    # The groups are the ones the rule keeps apart, so a count by group says
    # what the ends of the rejected windows looked like. It does not say which
    # end was the one refused, nor which condition refused it: the rule reads
    # both ends and the text around them, and this reads the two characters.
    #
    # It is still worth counting. A rejection rate on its own says nothing about
    # where to look, and a count showing that the rejected windows nearly all
    # end in a digit points at the numerical rules rather than at the letters.
    #
    # The dashes are taken from `validation.py` rather than written out again.
    # They are one rule, and a second copy of it here would be a second rule
    # that has to agree.
    if character.isdigit():
        return "a digit"
    if character in ".,":
        return "a decimal separator"
    if character in _CARRIES_A_NUMBER_ON:
        return "a sign or dash"
    if character.isalpha():
        return "a letter"
    return "other punctuation"


def measure(papers):
    """Tries every window against the paper it came out of."""
    # Kept apart from the printing so that a test can run it on abstracts it
    # wrote itself. A measurement whose only path to a result is a folder of
    # real replies is one nobody can check.
    tried = 0
    rejected = Counter()
    rejected_papers = set()
    for paper in papers:
        for window in windows_from(paper.abstract):
            tried += 1
            if not is_in_the_paper(window, paper):
                rejected[(kind_of(window[0]), kind_of(window[-1]))] += 1
                rejected_papers.add(paper.pmid)
    return tried, rejected, rejected_papers


def main(folder=REPLIES):
    papers = []
    problems = []
    for path in sorted(Path(folder).glob("*.json")):
        found, problem = abstracts_from(path)
        papers.extend(found)
        if problem:
            problems.append(problem)

    for problem in problems:
        print(f"Not read: {problem}")

    if not papers:
        print(f"No abstracts were found in {folder}, so nothing was measured. "
              "That is a failure of the measurement, not a result.")
        return 1

    tried, rejected, rejected_papers = measure(papers)
    if not tried:
        print(f"{len(papers)} abstracts were read, and none of them is long "
              f"enough for a window of {min(LENGTHS)} words. Nothing was "
              "measured. That is a failure of the measurement, not a result.")
        return 1

    print(f"Abstracts read: {len(papers)}")
    print(f"Windows tried: {tried}")
    print(f"Windows rejected: {sum(rejected.values())}")
    print(f"Rejection rate: {sum(rejected.values()) / tried:.6f}")
    print(f"Abstracts with at least one rejection: {len(rejected_papers)}")
    print()
    print("A rejection here is not by itself a fault. A window can cut a "
          "numerical expression in half, and the rule refuses that on "
          "purpose. Reading these as false refusals needs the windows judged "
          "one by one.")
    print()
    if rejected:
        print("What the rejected windows start and end with:")
        for (starts, ends), how_many in rejected.most_common():
            print(f"  starts with {starts}, ends with {ends}: {how_many}")
    else:
        print("No window of these shapes was rejected.")

    # A file that could not be read means the figure above covers less than
    # what was asked for, and a rate reported as though it were complete is
    # the thing this project has been caught by before.
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else REPLIES))
