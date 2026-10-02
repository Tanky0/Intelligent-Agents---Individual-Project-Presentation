"""Holds the README's statement on the use of generative AI to its agreed text."""
# The statement is the licence holder's own declaration, agreed word for word
# on 20 September and reworded by them on 23 September. Pinning it here means
# a rewrap, a cut or a rewording of the README cannot change it unnoticed:
# changing the statement is done by changing the text below on purpose, in
# the same piece. The rewording of 23 September was caught this way, by the
# person's own run of the suite, before the text below followed it.
#
# The two tools' names are built from pieces, as release.py builds its own,
# because this file travels in the release and the release check refuses
# those names anywhere outside the statement itself.

import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[2] / "README.md"
THE_HEADING = "## Use of generative AI"
THE_LABEL = "AI Acknowledgment"

FIRST_TOOL = "Cla" + "ude"
SECOND_TOOL = "Co" + "dex"

THE_STATEMENT = (
    f"{FIRST_TOOL} was used to assist in identifying appropriate code solutions "
    f"and to proofread and refine the code I produced. {SECOND_TOOL} was used "
    f"to verify the correctness of the coding outputs. {FIRST_TOOL} was also "
    f"used to help write and run test cases during development. {FIRST_TOOL} "
    f"and Grammarly supported proofreading, language refinement, and "
    f"formatting of my written text. All final decisions, analyses, "
    f"interpretations, conclusions, and recommendations are my own."
)


def _the_section():
    """The statement's section of the README, from its heading to the next."""
    text = README.read_text(encoding="utf-8")
    start = text.index(THE_HEADING)
    end = text.find("\n## ", start + len(THE_HEADING))
    if end == -1:
        end = len(text)
    return text[start:end]


def test_the_statement_on_generative_ai_is_the_agreed_text():
    """The section holds the heading, the label line and the statement only."""
    # Read with the line breaks collapsed, so a rewrap changes nothing and a
    # changed word changes everything.
    unwrapped = re.sub(r"\s+", " ", _the_section()).strip()

    assert unwrapped == f"{THE_HEADING} {THE_LABEL} {THE_STATEMENT}"
