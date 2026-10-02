"""Draws a random sample of PubMed papers to be screened by hand (SC-01)."""
# The sample is random and drawn from a fixed starting number, which records how
# the fifty were chosen. It does not on its own let the draw be repeated: PubMed
# returns a different set as papers are added, so drawing again would give the
# same fifty only if the same papers came back in the same order. The fifty in
# the file are what the recall figure is measured against. Choosing them by hand
# would push that figure towards papers that are easy to find.
#
# Two samples are drawn. The second is a check on the first, and leaves out every
# paper already in it so the two do not overlap. The reason for a second sample
# is recorded in data/hand_screened_sample/README.md.
#
#     python scripts/get_abstracts_to_screen.py          draws the first sample
#     python scripts/get_abstracts_to_screen.py check    draws the check sample
#
# The papers themselves are fetched by the PubMed source rather than by this
# file. That code is already tested, and having one place that reads PubMed
# replies means a sample cannot be built by different rules from the ones the
# agent will use.

import csv
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_agent.config import settings
from research_agent.sources.http import FetchError
from research_agent.sources.pubmed import fetch, search

QUERY = "ivermectin AND (covid-19 OR sars-cov-2)"
SAMPLE_SIZE = 50
DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data/hand_screened_sample"

# The draw itself, with the abstracts to be read. It is written where the
# repository ignores it, because an abstract is somebody else's text and
# publishing one needs a basis nobody has established (R-12). Screening is done
# from this file; what is prepared for release is the labels beside it.
#
# This script does not write the labels file. Reducing a draw to its labels is
# a deliberate act by the person who screened it, done once and recorded, and
# the labels are then maintained by hand. A command that produced them
# automatically would be able to overwrite a benchmark whose decisions cannot
# be recovered, which is the same reason a redraw is refused when the draw
# already exists.
DRAWN_DIR = DATA_DIR / "drawn_for_screening"

# Fixed starting numbers, so either sample can be drawn again exactly.
SAMPLES = {
    "first": {"seed": 42, "file": DRAWN_DIR / "abstracts_to_screen.csv",
              "labels": DATA_DIR / "screening_labels.csv"},
    "check": {"seed": 43, "file": DRAWN_DIR / "abstracts_to_screen_check.csv",
              "labels": DATA_DIR / "screening_labels_check.csv"},
}

COLUMNS = ["pmid", "year", "journal", "publication_types", "title", "abstract",
           "include", "rationale"]


def papers_already_drawn(earlier_file):
    """Returns the papers in an earlier sample, so the next one avoids them."""
    # An empty answer for a file that is not there is right only when there is no
    # earlier sample. It is given the committed labels for that reason: a file
    # the repository always carries, rather than one that may have been drawn on
    # somebody else's machine.
    if not earlier_file.exists():
        return set()

    drawn = set()
    with earlier_file.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            drawn.add(row["pmid"].strip())
    return drawn


def why_the_first_sample_cannot_be_read(labels):
    """Says why the first sample's labels are unusable, or None if they are fine."""
    # The check sample is drawn to avoid every paper in the first, so the list it
    # avoids has to be the whole list. A file that is missing, short, or holding
    # something other than identifiers would leave papers in the pool that are
    # already screened, and the two samples would overlap. Reported here rather
    # than discovered later in a recall figure that counts a paper twice.
    if not labels.exists():
        return f"{labels.name} is not there"

    try:
        with labels.open(encoding="utf-8-sig", newline="") as f:
            identifiers = [row["pmid"].strip() for row in csv.DictReader(f)]
    except (OSError, KeyError) as problem:
        return f"{labels.name} could not be read ({problem})"

    if len(identifiers) != SAMPLE_SIZE:
        return (f"{labels.name} holds {len(identifiers)} identifiers, and the "
                f"first sample is {SAMPLE_SIZE}")
    if len(set(identifiers)) != SAMPLE_SIZE:
        return f"{labels.name} repeats an identifier"
    if not all(pmid.isdigit() for pmid in identifiers):
        return f"{labels.name} holds something that is not an identifier"
    return None


def as_row(record):
    """Turns one paper into a line of the spreadsheet, ready to be screened."""
    return {
        "pmid": record.pmid,
        "year": record.year,
        "journal": record.journal,
        "publication_types": "; ".join(record.publication_types),
        "title": record.title,
        "abstract": record.abstract,
        "include": "",        # filled in by hand: include or exclude
        "rationale": "",      # filled in by hand: one short line
    }


def write_spreadsheet(path, rows):
    """Writes the sample out where it can be opened and screened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so that Excel shows accented characters correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Draws one sample. Returns 0 when it worked, and 1 when it did not."""
    # The number matters because anything running this needs to be able to tell
    # the difference without reading the words. Saying nothing went wrong when
    # something did is how a failed step gets built on.
    if len(sys.argv) > 1:
        which = sys.argv[1]
    else:
        which = "first"

    if which not in SAMPLES:
        print(f"Unknown sample '{which}'. Use 'first' or 'check'.")
        return 1

    destination = SAMPLES[which]["file"]
    if destination.exists():
        # Screening decisions are written into this file by hand and cannot be
        # recovered, so a second run must never quietly replace them.
        print(f"{destination.name} already exists. Delete it first to redraw.")
        return 1

    if not settings.ncbi_api_key:
        # NCBI allows three requests a second without a key and ten with one,
        # so the sample can still be drawn. Saying it would be refused was
        # wrong, and would have sent somebody looking for a fault that is not
        # there.
        print("NCBI_API_KEY is missing from .env. The sample can still be "
              "drawn, but more slowly: NCBI allows three requests a second "
              "without a key and ten with one.")

    # Checked before the search, not after it. The check sample cannot be drawn
    # without the list of papers to leave out, so asking PubMed first would
    # spend a request on a draw that is going to be refused anyway. Read from
    # the labels rather than from the draw: the draw sits in a folder the
    # repository ignores, so on a fresh checkout it is not there, and asked for
    # it this returned an empty set and the check sample was drawn with nothing
    # left out, which is the one thing it exists to avoid.
    if which == "check":
        wrong = why_the_first_sample_cannot_be_read(SAMPLES["first"]["labels"])
        if wrong:
            print(f"The check sample cannot be drawn: {wrong}")
            return 1

    found = search(QUERY, limit=2000)
    if isinstance(found, FetchError):
        print(f"The search failed: {found.reason}")
        return 1
    print(f"The query matched {len(found)} papers.")

    available = found
    if which == "check":
        already_drawn = papers_already_drawn(SAMPLES["first"]["labels"])
        available = [pmid for pmid in found if pmid not in already_drawn]
        print(f"Leaving out {len(already_drawn)} already drawn, "
              f"{len(available)} remain.")

    # Fewer than fifty is a smaller sample, not a sample. Taking what there is
    # would give a benchmark of a different size from the one the recall figure
    # is defined against, and the only sign of it would be a row count nobody
    # looked at.
    if len(available) < SAMPLE_SIZE:
        print(f"Only {len(available)} papers are available and the sample is "
              f"{SAMPLE_SIZE}. Nothing was drawn.")
        return 1

    random.seed(SAMPLES[which]["seed"])
    chosen = sorted(random.sample(available, SAMPLE_SIZE))

    records, errors = fetch(chosen)
    rows = []
    for record in records:
        rows.append(as_row(record))

    write_spreadsheet(destination, rows)

    without_abstract = 0
    for row in rows:
        if not row["abstract"]:
            without_abstract += 1

    print(f"Wrote {len(rows)} papers to {destination}")
    print(f"  {without_abstract} have no abstract. Judge these on the title and "
          f"publication type, not automatically as excluded.")
    if errors:
        print(f"  {len(errors)} groups of papers could not be fetched.")
        # The file is still written, because the papers that did come back are
        # worth having. The number says the sample is short of what was asked
        # for, so nothing is built on it as though it were complete.
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
