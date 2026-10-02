"""Checks the diagnostic that counts what the extract check rejects."""
# The diagnostic cuts windows of whole words out of real abstracts and counts
# how many the rule rejects. That count only means anything if the windows are
# the paper's own words, character for character: a generator that dropped a
# space or joined two sentences would produce text no reader wrote, and the
# figure would measure the generator instead of the rule.
#
# What a rejection means is a separate question, and these tests do not settle
# it. A window can cut a numerical expression in half, and `test_validation.py`
# requires the rule to refuse that, so some rejections are the rule working.
# Telling those apart from real false refusals needs quotations somebody
# actually wrote, which is still to come (SC-02).
#
# Nothing here reads the saved replies, calls a service or asks the model.

import importlib.util
import json
import pathlib
from dataclasses import dataclass

import pytest

SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "measure_extract_check.py")


@pytest.fixture
def measuring():
    """The script as it stands on disk, loaded from its path."""
    # Loaded by its path so that what is tested is the file somebody runs. An
    # import by name would depend on where the interpreter was started from and
    # on what else is on its path, and could find something other than this
    # file without saying so.
    spec = importlib.util.spec_from_file_location("measure_extract_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class APaper:
    """A paper carrying the three fields the diagnostic reads."""
    # The check looks at the title and the abstract, and the counting names a
    # paper by its identifier. A whole PaperRecord would bring with it the
    # source of the text, the origins of the search and the rules that hold
    # between them, none of which this asks about, and a reader would have to
    # work out which of those the test depended on.

    abstract: str
    title: str = "A trial of drug X"
    pmid: str = "12345678"


AN_ABSTRACT = (
    "Forty patients were given drug X. Scores improved in 28 per cent of "
    "them, against 12 per cent in the control group. No serious harm was "
    "reported during the eight weeks of follow-up."
)


def a_reply(path, body):
    """A saved reply written in the shape the capture writes."""
    # The tests that use it are about what happens to a folder of files: which
    # of them are read, which are reported, and what is done with the abstracts
    # that come out. Handing the functions a parsed record instead would test
    # the counting and leave the reading, which is the half of it that decides
    # whether anything is measured at all.
    path.write_text(json.dumps({"url": "https://eutils.ncbi.nlm.nih.gov/",
                                "asked": {}, "status": 200, "headers": {},
                                "body": body}), encoding="utf-8")


def an_article(abstract="Forty patients were given drug X and were followed."):
    """A PubMed reply holding one article, with the abstract a test chooses."""
    # Written out here rather than copied from a captured reply, because a test
    # that carried real article text would put paper text into the repository,
    # which is the boundary R-11 was spent enforcing.
    return (
        "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
        "<PMID>12345678</PMID>"
        "<Article><ArticleTitle>A trial of drug X</ArticleTitle>"
        f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>"
        "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )


def test_every_window_is_the_paper_s_own_words(measuring):
    """A window the generator invented would make its rejection meaningless,
    because the rule would be right to refuse it."""
    cut = list(measuring.windows_from(AN_ABSTRACT))

    assert cut, "the generator produced nothing to measure"
    for window in cut:
        assert window in AN_ABSTRACT


def test_windows_start_and_end_at_a_word(measuring):
    """Cutting mid-word would measure a shape nobody quotes."""
    for window in measuring.windows_from(AN_ABSTRACT):
        at = AN_ABSTRACT.index(window)
        before = AN_ABSTRACT[at - 1] if at else " "
        after = AN_ABSTRACT[at + len(window):at + len(window) + 1] or " "
        assert before.isspace(), "a window began inside a word"
        assert after.isspace(), "a window ended inside a word"


def test_the_lengths_sampled_are_three_six_and_twelve(measuring):
    """Written out rather than compared with the constant."""
    # Read from the constant, this test passes whatever the constant says, so
    # it would protect nothing. The three lengths are a sampling choice, and a
    # change to them changes what any figure means.
    assert measuring.LENGTHS == (3, 6, 12)
    assert {len(window.split()) for window in measuring.windows_from(AN_ABSTRACT)} \
        == {3, 6, 12}


def test_a_window_that_cuts_a_number_in_half_is_rejected(measuring):
    """This is the rule working, not failing, and it is why the count is a
    rejection count rather than a false-refusal count.
    """
    # The window ends at a word boundary and still leaves half of a spaced
    # range behind. `test_validation.py` requires that refusal.
    spaced_range = APaper(abstract="Scores improved in 28 - 30 patients "
                                   "overall in the treated group today")

    tried, rejected, papers = measuring.measure([spaced_range])

    assert tried > 0
    assert sum(rejected.values()) >= 1, "no rejection was counted"
    assert rejected[("a letter", "a digit")] >= 1
    assert papers == {"12345678"}


def test_ordinary_prose_is_not_rejected(measuring):
    """A diagnostic that reported rejections on plain sentences would be
    counting its own faults."""
    plain = APaper(abstract="Forty patients were given the treatment and all "
                            "of them were followed for eight weeks")

    tried, rejected, papers = measuring.measure([plain])

    assert tried > 0
    assert rejected == {}
    assert papers == set()


@pytest.mark.parametrize("character, named", [
    ("7", "a digit"),
    (".", "a decimal separator"),
    ("-", "a sign or dash"),
    ("x", "a letter"),
    (")", "other punctuation"),
])
def test_the_groups_are_the_ones_the_rule_keeps_apart(measuring, character, named):
    """A total says how often, and this says what the ends looked like."""
    # It does not say which end was refused or which condition refused it, and
    # the count is worth having anyway: rejected windows that nearly all end in
    # a digit point at the numerical rules rather than at the letters.
    assert measuring.kind_of(character) == named


def test_an_abstract_too_short_for_a_window_is_a_non_result(measuring, tmp_path,
                                                            capsys):
    """No windows means nothing was measured, and a rate over nothing is not
    a number at all.
    """
    # Dividing by the count was what it did before, so the whole measurement
    # ended in an error message about arithmetic rather than about abstracts.
    a_reply(tmp_path / "short.json", an_article("Brief abstract."))

    answer = measuring.main(tmp_path)

    assert answer == 1
    assert "Nothing was measured" in capsys.readouterr().out


def test_a_reply_that_cannot_be_read_is_reported(measuring, tmp_path, capsys):
    """A measurement that quietly reads fewer papers than it was given looks
    exactly like one that worked. This project has been caught by that twice.
    """
    a_reply(tmp_path / "good.json", an_article())
    a_reply(tmp_path / "damaged.json",
            "<PubmedArticleSet><PubmedArticle>truncated")

    answer = measuring.main(tmp_path)

    said = capsys.readouterr().out
    assert "damaged.json" in said
    assert "could not be parsed" in said
    assert answer == 1, "a figure covering less than was asked for is not a pass"


def test_a_reply_holding_no_abstract_is_not_a_problem(measuring, tmp_path,
                                                      capsys):
    """Searches, link replies and Crossref records are ordinary things to
    find in that folder. Reporting them as faults would bury a real one."""
    a_reply(tmp_path / "good.json", an_article())
    a_reply(tmp_path / "crossref.json", '{"message": {"title": ["A paper"]}}')

    answer = measuring.main(tmp_path)

    said = capsys.readouterr().out
    assert "Not read:" not in said
    assert "Abstracts read: 1" in said
    assert answer == 0
