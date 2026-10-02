"""Checks that no design decision is left without a written reason."""
# While a decision is still waiting for its reason, the line carrying it is
# marked TODO(reason). This test lists any that remain. Run it before the code
# is finished:
#
#     pytest -m rationale
#
# It is left out of the ordinary test run, because a reminder that work is still
# to be done would otherwise appear as a failure beside real ones.

import pathlib

import pytest

# The folder holding the application code. parents[0] is tests/unit,
# parents[1] is tests, parents[2] is the project folder itself.
CODE = pathlib.Path(__file__).resolve().parents[2] / "research_agent"


def test_the_folder_being_checked_is_the_right_one():
    """Runs in the ordinary test run, unlike the check below."""
    # A wrong count above would point this test at a folder that does not exist.
    # Searching an empty folder finds nothing, so the check below would pass
    # while reading no code at all. This test failed to exist once already and
    # the check was silently passing as a result.
    assert CODE.exists(), f"{CODE} does not exist, so nothing would be checked"

    files = list(CODE.rglob("*.py"))
    assert len(files) > 1, f"only {len(files)} files found in {CODE}"


@pytest.mark.rationale
def test_every_design_decision_has_a_written_reason():
    outstanding = []

    # rglob looks in this folder and every folder inside it.
    for path in CODE.rglob("*.py"):
        lines = path.read_text(encoding="utf8").splitlines()
        # enumerate gives the position of each line as well as the line, so a
        # marker can be reported with the line it is on.
        for number, line in enumerate(lines, start=1):
            if "TODO(reason)" in line:
                outstanding.append(f"{path.relative_to(CODE)}:{number}")

    message = ("design decisions still waiting for a written reason: "
               + ", ".join(outstanding))
    assert not outstanding, message
