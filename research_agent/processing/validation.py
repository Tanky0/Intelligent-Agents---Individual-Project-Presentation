"""Checks that a judgement's extract is really in the paper (SC-02)."""
# The Curation Agent is asked for a summary built from the paper's own wording
# and the sentence that supports it, copied word for word. Asking is not the
# same as getting: a model can produce a sentence that reads like the paper and
# is not in it, and a fabricated extract is harder to notice than a fabricated
# summary, because it looks like evidence. So the extract is looked for in the
# paper rather than taken on trust.
#
# What this answers, and what it does not. It answers whether those words appear
# in the paper. It does not say the extract supports the summary, that the
# summary is a fair reading of the paper, or that the model had that sentence in
# front of it: a shortened paper is cut before it reaches the request, so a
# sentence in the part not shown would still be found here. Whether the evidence
# holds is the reviewer's to judge (DC-08).
#
# The answer is written into the judgement rather than acted on. Nothing is
# thrown away and no paper is refused: a paper whose extract could not be found
# is one the reviewer should look at. Storage is where that answer becomes a
# refusal, because storing an unverifiable claim is not the reviewer's to
# authorise (SC-02).
#
# The title and the abstract are searched, one at a time rather than joined. Run
# together, a quotation could be assembled from the end of one and the start of
# the other: a title of "No benefit" and an abstract beginning "was observed"
# would between them accept "benefit was observed", which neither says. Full
# text is not searched, because it is not sent for screening, as the service
# and data boundaries in `docs/development.md` record, so an extract found only
# there is one the model could not have read.
#
# Three kinds of difference are ignored and no others.
#
# Spacing, because an abstract arrives with the line breaks its publisher used
# and the same words laid out differently are the same words.
#
# A short list of typographic characters: the curly quotation marks and the
# ellipsis. This is a decision made for this project rather than something
# measured about the services. The same sentence is written with either form of
# quotation mark in different records, and a quotation copied faithfully can
# differ in nothing else. The dashes are not in that list. A minus sign, a range
# dash and a hyphen look alike and do not mean alike, and treating them as one
# would make "minus 5" and "5 to 10" the same text.
#
# And where the match may begin and end. A word may not run into it, in any
# alphabet, and a number may not be part of a longer numerical expression. So
# "5 patients improved" is not found in "15 patients improved", "5 mg" is not
# found in "0.5 mg" or ".5 mg" or "- 5 mg", "improved in 28" is not found in
# "improved in 28-30", and "treatment" is not found in "prétreatment". Without
# that, a substring search reports a different number as a quotation, which is
# the failure this file exists to catch.
#
# Case is not ignored. Folding it made "5 mm" match "5 mM", which is a different
# unit, and a model copying a sentence word for word reproduces its capitals.
# Anything else counts as not found: a paraphrase, a changed word, a dropped
# clause and a different number. A reviewer sent to look at a real quotation
# that differed by a comma has lost a minute; a claim the paper does not make,
# passed as verified, is what the count of claims absent from the abstract
# exists to catch (SC-03).

import re
import unicodedata

# The same words in a different typeface. Folding these is this project's
# choice, made because the same sentence appears with either form in different
# records, not a finding about what any service returns.
#
# The dashes are deliberately not in here. A minus sign, a range dash and a
# hyphen look alike and do not mean alike: folding them would make "5 to 10"
# and "minus 5" the same text, and accepting a changed sign or a changed range
# is the kind of thing this file exists to refuse.
_SAME_CHARACTER = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "…": "...",
}

# Every character that can carry a number on from where it stopped: the
# hyphen and the various dashes, the minus sign, and the two signs. A number
# followed by one of these and another digit is a range or a subtraction, and
# quoting half of it is quoting something the paper does not say.
#
# The hyphen and the plus are written as themselves; the rest are given by code
# point. On the page a figure dash, an en dash, an em dash and a minus sign are
# nearly the same mark, so a reader checking this list against the rules above
# could not tell which was which, and neither could somebody adding one. It
# also keeps the em dash out of the file, which is what this project asks of
# every file it writes.
_CARRIES_A_NUMBER_ON = "-+" + "".join(chr(point) for point in (
    0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212))


def _is_a_mark(character):
    # A letter and the accent written after it can be two characters, and the
    # accent is not a letter, so a rule that asked only about letters found
    # "treatment" inside "pre", an acute accent, and "treatment". Composing the
    # text settles that wherever the pair has a letter of its own; this settles
    # the rest, which have no composed form to become.
    return unicodedata.category(character)[0] == "M"


def _marks_on_a_letter():
    # Asked of Unicode rather than listed by hand. An earlier version of this
    # named five blocks of marks and called them the rest, and U+0483, which is
    # written with Cyrillic, sits outside all five: "treatment" was quoted out
    # of the longer word spelt with it. A list of blocks is really a guess about
    # which alphabets a paper will be written in, and this project screens
    # papers in any of them.
    runs = []
    for point in range(0x110000):
        if not _is_a_mark(chr(point)):
            continue
        if runs and runs[-1][1] == point - 1:
            runs[-1][1] = point
        else:
            runs.append([point, point])
    # Given as ranges of code points rather than written out, because a
    # combining mark on its own is invisible on the page and a reader could not
    # check the list against the rule it serves.
    return "".join(f"{chr(first)}-{chr(last)}" for first, last in runs)


_MARKS_ON_A_LETTER = _marks_on_a_letter()


def _comparable(text):
    # Composed first, so a letter written as a letter and a separate accent is
    # the same text as the letter written whole. Two records of one paper can
    # use either form, and a quotation should not turn on which the publisher
    # chose. What this does not do is make the accent disappear: an accented
    # letter is still a letter, and still refuses a match that runs into it.
    text = unicodedata.normalize("NFC", text)
    for odd, plain in _SAME_CHARACTER.items():
        text = text.replace(odd, plain)
    return " ".join(text.split())


def _found_whole(wanted, within):
    """True when the words of one appear in the other, lining up at both ends."""
    # A plain substring search accepts a shorter number inside a longer one, so
    # "5 patients" is found in "15 patients" and "5 mg" in "0.5 mg". A quotation
    # that changes a number is the failure that matters most here, and a check
    # that reports it as a match is worse than no check.
    #
    # Two things have to line up. A word may not run into the match, which is
    # asked of every letter in any alphabet rather than of the English ones
    # alone: an abstract that writes a Greek letter or an accent next to a word
    # is not quoting that word on its own. An accent written as its own
    # character is asked about as well, at both ends and whether it falls inside
    # the quotation or just outside it, because it is not a letter and a rule
    # that looked only for letters let "treatment" be quoted out of the longer
    # word when the publisher spelt the accent apart.
    #
    # And a number may not be part of a longer numerical expression. That covers
    # a digit or a decimal separator on either side, a leading decimal point as
    # in ".5", a sign whether or not it is spaced away from the number, a range
    # like "28-30", and an exponent written either "1e-5" or "1.e-5": "-5"
    # quoted out of one of those is a different number, and the letter between
    # the digit and the sign hid that. A full stop that ends a sentence is not a
    # continuation, so a quotation ending in a number is still found at the end
    # of one.
    #
    # The punctuation a quotation begins or ends with is looked at too, not only
    # its letters and digits. A quotation ending in a full stop was accepted
    # inside a decimal, because the rules were asked about the full stop rather
    # than about the digit in front of it, and one beginning with a minus sign
    # was accepted inside "10-5". So a trailing decimal separator may not be
    # followed by a digit, and a leading sign or decimal separator may not follow
    # one: ".5 mg" is a quotation of ".5 mg" and not of "10.5 mg".
    signs = re.escape(_CARRIES_A_NUMBER_ON)
    marks = _MARKS_ON_A_LETTER
    before, after = "", ""
    # A mark counts as part of a word at either end, as well as a letter. It is
    # not a letter itself, so an extract that ended in one was given no ending
    # boundary at all, and "treatment" with an accent after it was accepted as
    # a quotation of the same word with an "s" after that.
    if wanted[0].isalnum() or wanted[0] == "_" or _is_a_mark(wanted[0]):
        before = rf"(?<!\w)(?<![{marks}])"
    if wanted[0].isdigit():
        before += rf"(?<![.,])(?<![{signs}])(?<![{signs}] )"
    if wanted[0] in _CARRIES_A_NUMBER_ON:
        # The digit before an exponent marker carries the number on as surely
        # as a digit against the sign does: "-5" quoted out of "1e-5" is a
        # different number, and the letter between them hid that. A decimal
        # point may stand between the digit and the marker, as in "1.e-5",
        # which Python reads as the same value, so that form is asked about
        # too.
        before += r"(?<!\d)(?<!\d )(?<![0-9][eE])(?<![0-9][.,][eE])"
    if wanted[0] in ".,":
        # A decimal separator can be carried on from either side. A digit
        # before it makes the number longer, as "10.5" does for ".5", and a
        # sign before it changes what the number is, as "-.5" does.
        before += rf"(?<!\d)(?<!\d )(?<![{signs}])(?<![{signs}] )"
    if wanted[-1].isalnum() or wanted[-1] == "_" or _is_a_mark(wanted[-1]):
        after = rf"(?!\w)(?![{marks}])"
    if wanted[-1].isdigit():
        after += rf"(?![.,]\d)(?! ?[{signs}] ?\d)"
    if wanted[-1] in ".,":
        after += r"(?!\d)"
    return re.search(before + re.escape(wanted) + after, within) is not None


def is_in_the_paper(extract, paper):
    """True when the extract appears in the paper's title or in its abstract."""
    # An extract of nothing is not found. The record already refuses a blank one,
    # so this is for a caller that built a judgement some other way rather than
    # for anything a run can produce.
    wanted = _comparable(extract)
    if not wanted:
        return False
    return any(_found_whole(wanted, _comparable(part))
               for part in (paper.title, paper.abstract) if part)


def with_the_extracts_checked(screened):
    """Returns the screened papers with extract_found answered on each."""
    # A new list of new records. The judgements are evidence of what the model
    # said, and a check that rewrote them in place would leave nothing to compare
    # a later run against.
    #
    # Every paper is checked, including the ones the agent excluded. An exclusion
    # resting on a sentence that is not in the paper is as wrong as an inclusion
    # resting on one, and the reviewer may well disagree with the exclusion and
    # want the paper.
    return [
        item.model_copy(update={
            "extract_found": is_in_the_paper(item.screening.extract, item.paper)})
        for item in screened
    ]
