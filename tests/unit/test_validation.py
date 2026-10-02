"""Looking for the extract in the paper it was taken from (SC-02, SC-03)."""
# A fabricated extract is harder to notice than a fabricated summary, because it
# looks like evidence. Most of what is below is about the two kinds of near-miss
# that matter: a quotation that differs only in how it was typeset, which has to
# pass, and one whose words or numbers differ, which has to fail.
#
# The number cases are here because they were found rather than imagined. A
# plain substring search reported "5 patients improved" as a quotation of "15
# patients improved", and "5 mg" as one of "0.5 mg".

import pytest

from research_agent.processing.validation import is_in_the_paper, with_the_extracts_checked
from research_agent.schemas import PaperRecord, ScreenedPaper, Screening, SearchOrigin

ABSTRACT = (
    "Forty patients were given drug X for twelve weeks.\n"
    "Symptoms improved in 28 of them, and three withdrew."
)


def a_paper(title="A trial of drug X", abstract=ABSTRACT, **changes):
    values = {
        "title": title,
        "abstract": abstract,
        "text_source": "abstract" if abstract else "metadata",
        "origins": [SearchOrigin(source="pubmed", query="drug X")],
    }
    values.update(changes)
    return PaperRecord(**values)


def judged(extract, paper=None, decision="include"):
    return ScreenedPaper(
        paper=paper or a_paper(),
        screening=Screening(
            paper_number=1,
            decision=decision,
            relevance=7,
            reason="it reports an outcome for drug X",
            summary="a trial of drug X in condition Y",
            extract=extract,
        ),
    )


def test_a_sentence_copied_from_the_abstract_is_found():
    assert is_in_the_paper("Symptoms improved in 28 of them", a_paper()) is True


def test_a_sentence_that_is_not_in_the_paper_is_not_found():
    assert is_in_the_paper("Symptoms improved in every patient", a_paper()) is False


def test_a_no_break_space_is_spacing_like_any_other():
    """A no-break space should not make a copied quotation look invented."""
    # Publishers set one between a number and what follows it. It changes the
    # layout between the same words and nothing else, so the spacing rule
    # should treat it as an ordinary space. It does, because Python counts it
    # as whitespace when the text is split into words; this test is what holds
    # the rule to that, since nothing else in this file writes one.
    spaced = a_paper(abstract="Symptoms improved in" + chr(0xA0) + "28 of them.")

    assert is_in_the_paper("improved in 28 of them", spaced) is True


def test_the_title_counts_as_the_paper_as_well():
    """A paper with no abstract still has to support its extract somehow."""
    # The title is shown to the model along with the abstract, so it is part of
    # what an extract can honestly have come from.
    metadata_only = a_paper(abstract="")

    assert is_in_the_paper("A trial of drug X", metadata_only) is True


def test_full_text_is_not_searched():
    """Full text is not sent for screening, so an extract cannot come from it."""
    # The boundaries in `docs/development.md` keep full text out of the model's
    # requests. A
    # sentence found only there is one the model could not have read, and
    # accepting it would let an invented quotation pass whenever the run happened
    # to hold a long paper.
    with_a_body = a_paper(
        full_text="Drug X was well tolerated throughout the study.",
        text_source="full text", full_text_licence="CC BY")

    assert is_in_the_paper("Drug X was well tolerated throughout", with_a_body) is False


@pytest.mark.parametrize("extract", [
    "Symptoms   improved  in 28 of them",
    "Symptoms improved\nin 28 of them",
    "Symptoms improved in 28 of them ",
])
def test_spacing_does_not_decide_whether_a_quotation_is_real(extract):
    """An abstract arrives with the line breaks its publisher used."""
    # The same words laid out differently are the same words, and a reviewer sent
    # to look at a real quotation because of a line break has been sent for
    # nothing.
    assert is_in_the_paper(extract, a_paper()) is True


def test_case_is_not_ignored():
    """Folding case made "5 mm" match "5 mM", which is a different unit."""
    # A model copying a sentence word for word reproduces its capitals, so the
    # cost of keeping case is small and the unit is worth keeping.
    assert is_in_the_paper("SYMPTOMS IMPROVED IN 28 OF THEM", a_paper()) is False


def test_a_changed_unit_is_not_hidden_by_case():
    dosed = a_paper(abstract="The 5 mM solution was used throughout.")

    assert is_in_the_paper("5 mm", dosed) is False
    assert is_in_the_paper("5 mM", dosed) is True


@pytest.mark.parametrize("typographic, plain", [
    ("‘quoted’ result", "'quoted' result"),
    ("“quoted” result", "\"quoted\" result"),
    ("and so on… in all", "and so on... in all"),
])
def test_the_same_words_in_different_typography_are_the_same_words(typographic, plain):
    """Folding these is this project's decision, not a measurement."""
    # The same sentence appears with either form of quotation mark in different
    # records, so a quotation copied faithfully can differ in nothing else.
    assert is_in_the_paper(typographic, a_paper(abstract=plain)) is True
    assert is_in_the_paper(plain, a_paper(abstract=typographic)) is True


@pytest.mark.parametrize("near_miss", [
    "Symptoms improved in 27 of them",
    "Symptoms improved in most of them",
    "Forty patients were given drug Y for twelve weeks",
    "Symptoms improved in 28 of the patients",
])
def test_a_changed_word_or_number_is_not_found(near_miss):
    """Deliberately strict, and the near misses are the point."""
    # A changed number is the failure that matters most: it reads like the paper,
    # it survives a glance, and it is a claim the paper does not make.
    assert is_in_the_paper(near_miss, a_paper()) is False


def test_a_shorter_quotation_that_is_really_there_is_found():
    """Strictness is about the words, not about how many of them there are."""
    assert is_in_the_paper("three withdrew", a_paper()) is True


@pytest.mark.parametrize("extract, abstract", [
    ("5 patients improved", "15 patients improved during the trial."),
    ("5 mg reduced symptoms", "A dose of 0.5 mg reduced symptoms."),
    ("5 mg reduced symptoms", "A dose of .5 mg reduced symptoms."),
    ("improved in 28", "Symptoms improved in 28.5 per cent of patients."),
    ("5 mg was given", "A dose of -5 mg was given."),
    ("5 mg was given", "A change of - 5 mg was given."),
    ("5 mg was given", "A change of −5 mg was given."),
    ("improved in 28", "Symptoms improved in 28-30 per cent."),
    ("improved in 28", "Symptoms improved in 28–30 per cent."),
    ("improved in 28", "Symptoms improved in 28 - 30 per cent."),
    ("50 patients", "150 patients were recruited."),
])
def test_a_number_that_is_part_of_a_longer_one_is_not_a_match(extract, abstract):
    """A substring search reports a different number as a quotation."""
    # "5 patients improved" sits inside "15 patients improved", and "5 mg" inside
    # "0.5 mg". Both read as the paper's own words and neither is what it says,
    # which is the failure this file exists to catch, so the match has to line up
    # at both ends.
    assert is_in_the_paper(extract, a_paper(abstract=abstract)) is False


@pytest.mark.parametrize("extract, abstract", [
    ("Symptoms improved in 28.", "Symptoms improved in 28.5 per cent of patients."),
    ("-5 mg", "A dose of 10-5 mg was used."),
    ("- 5 mg", "A dose of 10 - 5 mg was used."),
    (".5 mg", "A dose of 10.5 mg was used."),
    (".5 mg", "A dose of 0.5 mg was used."),
    (".5 mg", "A change of -.5 mg was seen."),
    (".5 mg", "A change of - .5 mg was seen."),
    (",5 mg", "A dose of 10,5 mg was used."),
])
def test_the_punctuation_a_quotation_starts_or_ends_with_is_checked_too(
        extract, abstract):
    """Asking only about letters and digits left the punctuation as a way in."""
    # A quotation ending in a full stop was accepted inside a decimal, because
    # the rule was asked about the full stop rather than about the digit in
    # front of it, and one beginning with a minus sign was accepted inside
    # "10-5".
    assert is_in_the_paper(extract, a_paper(abstract=abstract)) is False


@pytest.mark.parametrize("extract, abstract", [
    ("Symptoms improved in 28.", "Symptoms improved in 28. Three withdrew."),
    ("-5 mg", "A change of -5 mg was seen."),
    (".5 mg", "A dose of .5 mg was used."),
    ("-.5 mg", "A change of -.5 mg was seen."),
])
def test_that_punctuation_check_does_not_refuse_an_honest_quotation(
        extract, abstract):
    """A full stop ending a sentence and a minus sign starting a number are
    ordinary, and refusing them would send the reviewer to look at everything."""
    assert is_in_the_paper(extract, a_paper(abstract=abstract)) is True


def test_a_quotation_ending_in_a_number_is_still_found_at_the_end_of_a_sentence():
    """A full stop that ends a sentence is not a decimal point."""
    # Treating every following full stop as a continuation would refuse honest
    # quotations, which is the opposite mistake.
    counted = a_paper(abstract="Symptoms improved in 28.")

    assert is_in_the_paper("Symptoms improved in 28", counted) is True


@pytest.mark.parametrize("extract", [
    "improved in 28 of them Forty",
    "benefit was observed",
])
def test_a_quotation_cannot_be_assembled_from_two_parts_of_the_record(extract):
    """The title and the abstract are searched one at a time."""
    # Run together, a title of "No benefit" and an abstract starting "was
    # observed" would accept "benefit was observed", which neither of them says.
    split = a_paper(
        title="No benefit", abstract="was observed in either group of patients.")

    assert is_in_the_paper(extract, split) is False


def test_a_word_may_not_run_into_the_match_either():
    """The same reasoning as for numbers, applied to words."""
    dosed = a_paper(abstract="The pretreatment group improved.")

    assert is_in_the_paper("treatment group improved", dosed) is False


@pytest.mark.parametrize("abstract", [
    "The prétreatment group improved.",
    "The treatmenté group improved.",
    "The alphatreatment group improved.",
])
def test_a_letter_in_any_alphabet_runs_into_the_match(abstract):
    """Asking only about the English letters left the others as a way in."""
    # An abstract writing an accent or a Greek letter against a word is not
    # quoting that word on its own, and a check that only knew A to Z reported
    # "treatment" as found inside "pretreatment" spelt with an accent.
    assert is_in_the_paper("treatment", a_paper(abstract=abstract)) is False


# The word the next two tests use is written with the letter and its accent
# as two characters rather than one. Both spellings look the same on the
# page, so they are built from their code points: a reader can then see
# which form is meant, which is the whole of what these two check.
WRITTEN_APART = "The pre" + chr(0x0301) + "treatment group improved."
WRITTEN_WHOLE = "pr" + chr(0x00E9) + "treatment group"


def test_an_accent_written_as_its_own_character_still_runs_into_the_match():
    """The same word as the test above, spelt the other way round."""
    # An accent is not a letter, so a rule that asked only about letters let
    # "treatment" be found inside it. The test above refuses the same word when
    # the publisher wrote the accented letter whole, which is how this one got
    # past for as long as it did.
    apart = a_paper(abstract=WRITTEN_APART)

    assert is_in_the_paper("treatment", apart) is False


def test_a_word_carrying_an_accent_is_still_found_when_it_is_quoted_whole():
    """Refusing the accent as a way in must not refuse the word that has one."""
    # The two spellings are compared as one, so a quotation does not turn on
    # which form the publisher used: the extract here is written with the letter
    # whole and the abstract with the accent apart.
    apart = a_paper(abstract=WRITTEN_APART)

    assert is_in_the_paper(WRITTEN_WHOLE, apart) is True


@pytest.mark.parametrize("extract, abstract", [
    ("-5 mg", "The dose fell to 1e-5 mg overall."),
    ("-5 mg", "The dose fell to 1E-5 mg overall."),
    ("-5 mg", "The dose fell to 1.e-5 mg overall."),
    ("-5 mg", "The dose fell to 1.E-5 mg overall."),
])
def test_an_exponent_carries_a_number_on_as_a_digit_does(extract, abstract):
    """A letter between the digit and the sign hid the rest of the number."""
    # "1e-5" is one value written in scientific notation, and "-5" quoted out of
    # it is a different one, which is what this file exists to refuse. The last
    # two spell the same value with a decimal point before the marker, which
    # Python reads as 1e-05: the first version of this rule asked for a digit
    # immediately before the marker, and the point stood in that place.
    assert is_in_the_paper(extract, a_paper(abstract=abstract)) is False


# The two tests below turn on a combining mark, which shows as nothing on its
# own, so the words are built from code points. U+0483 is a mark written with
# Cyrillic; it is used here because it lies outside the five blocks of marks an
# earlier version of the rule listed, and no composing settles it, so it can
# only pass by the boundary asking Unicode what a mark is.
MARKED_IN_THE_ABSTRACT = "The treatment" + chr(0x0483) + " group improved."
QUOTED_WITH_ITS_MARK = "treatment" + chr(0x0483)


def test_a_mark_outside_the_listed_blocks_still_runs_into_the_match():
    """The blocks were a guess about which alphabets a paper would use."""
    # This project screens papers in any of them. The mark here is a real one in
    # Cyrillic and was in none of the five blocks named before, so "treatment"
    # was accepted as a quotation of the longer word carrying it.
    marked = a_paper(abstract=MARKED_IN_THE_ABSTRACT)

    assert is_in_the_paper("treatment", marked) is False


def test_an_extract_that_ends_in_a_mark_still_has_an_ending_boundary():
    """A mark is not a letter, so the ending check was not applied at all."""
    # The extract here ends in the mark, and the abstract carries on with an "s"
    # after it. Without this, the "s" simply vanished from what was called a
    # quotation.
    with_an_s = a_paper(abstract="The treatment" + chr(0x0483) + "s group.")

    assert is_in_the_paper(QUOTED_WITH_ITS_MARK, with_an_s) is False


def test_a_word_quoted_with_its_mark_is_still_found():
    """Refusing the mark as a way in must not refuse the word that carries it."""
    # Without this, the two tests above could be passed by refusing every word
    # with a mark in it, which would send the reviewer to check every paper
    # written in an alphabet that uses them.
    marked = a_paper(abstract=MARKED_IN_THE_ABSTRACT)

    assert is_in_the_paper(QUOTED_WITH_ITS_MARK, marked) is True


@pytest.mark.parametrize("extract, abstract", [
    ("a change of -5 mg", "We saw a change of -5 mg overall."),
    ("1.e-5 mg", "The dose fell to 1.e-5 mg overall."),
    ("28-30 per cent", "Symptoms improved in 28-30 per cent."),
    ("Symptoms improved in 28,", "Symptoms improved in 28, and three withdrew."),
])
def test_a_whole_numerical_expression_quoted_as_it_stands_is_found(extract, abstract):
    """Refusing half a range must not turn into refusing the whole of one."""
    # The rule is about a quotation stopping inside a number, not about numbers
    # being unquotable, and a check that refused these would send the reviewer
    # to look at every paper with a range in it.
    assert is_in_the_paper(extract, a_paper(abstract=abstract)) is True


def test_a_dash_is_not_the_same_character_as_a_minus_sign():
    """Folding them would make a range and a subtraction the same text."""
    # They look alike and do not mean alike, and accepting a changed sign is
    # the kind of thing this file exists to refuse.
    ranged = a_paper(abstract="Symptoms improved in 28–30 per cent.")

    assert is_in_the_paper("improved in 28-30 per cent", ranged) is False
    assert is_in_the_paper("improved in 28–30 per cent", ranged) is True


def test_an_empty_extract_is_not_found():
    """The record refuses a blank extract, so this is for a caller that is not
    a run. Answered false rather than true, because nothing was found."""
    assert is_in_the_paper("", a_paper()) is False


def test_every_judgement_comes_back_with_the_question_answered():
    given = [
        judged("Symptoms improved in 28 of them"),
        judged("Symptoms improved in every patient"),
    ]

    checked = with_the_extracts_checked(given)

    assert [item.extract_found for item in checked] == [True, False]


def test_an_excluded_paper_is_checked_too():
    """An exclusion resting on a sentence that is not in the paper is as wrong
    as an inclusion resting on one, and the reviewer may want the paper."""
    given = [judged("Symptoms improved in every patient", decision="exclude")]

    assert with_the_extracts_checked(given)[0].extract_found is False


def test_the_judgements_that_were_given_are_left_as_they_were():
    """The judgements are evidence of what the model said."""
    # Rewritten in place, there would be nothing to compare a later run against,
    # and the record would no longer show what was checked and what was claimed.
    given = [judged("Symptoms improved in 28 of them")]

    with_the_extracts_checked(given)

    assert given[0].extract_found is None, "not looked for, on the caller's copy"
