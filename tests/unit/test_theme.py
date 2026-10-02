"""The settings in `.streamlit/config.toml`, and what each was set for."""
# Streamlit reads the file from the folder the screen is started in. Nothing
# here draws a page. What can be checked without a browser is that each colour
# set clears the ratio it was set to clear, that the toolbar setting is the one
# whose own description hides the Deploy control and keeps the appearance
# menu, that the violet the page's own style rule draws is the accent the
# file sets, and that every key is one Streamlit knows, so that a misspelt key
# does not leave things as they were without a word.

import pathlib
import tomllib

import streamlit.config

from research_agent.ui import app

SETTINGS = pathlib.Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"

# The pages of Streamlit's two appearances: white, and gray100 of its dark
# palette. The settings under test leave both alone, so colours are
# measured against them.
LIGHT_PAGE = "#ffffff"
DARK_PAGE = "#0e1117"


def channel(value):
    """One channel of sRGB, 0 to 255, as WCAG's relative luminance takes it."""
    value = value / 255
    if value <= 0.03928:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_colour):
    red = int(hex_colour[1:3], 16)
    green = int(hex_colour[3:5], 16)
    blue = int(hex_colour[5:7], 16)
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def contrast(one, other):
    """WCAG's contrast ratio, the lighter colour over the darker."""
    lighter = max(luminance(one), luminance(other))
    darker = min(luminance(one), luminance(other))
    return (lighter + 0.05) / (darker + 0.05)


def the_settings():
    return tomllib.loads(SETTINGS.read_text(encoding="utf-8"))


def test_the_dark_outline_clears_three_to_one_against_the_page():
    """The session of 16 September measured the answer buttons'
    outline at 1.97 to 1 in the dark appearance, and WCAG 2.2 asks 3 to 1 of
    the boundary of a control (UX-01, criterion 1.4.11)."""
    outline = the_settings()["theme"]["dark"]["borderColor"]

    assert contrast(outline, DARK_PAGE) >= 3.0


def test_the_light_outline_clears_three_to_one_against_the_page_and_a_face():
    """The same session measured the light appearance's outline at 1.45 to 1
    (#D6D6D9 on white), and it was decided on 19 September that it goes up too.
    """
    # A plain button's face on the light page is white, so the outline is
    # measured against white for both (UX-01, criterion 1.4.11).
    outline = the_settings()["theme"]["borderColor"]

    assert contrast(outline, LIGHT_PAGE) >= 3.0


def test_the_accent_clears_three_to_one_against_the_page_it_is_drawn_on():
    """The accent draws the outline of the note box while it is written in and the
    switch on the results page when it is on, and 1.4.11 asks 3 to 1 of a
    control's boundary.
    """
    # The violet #440099 chosen on 17 September is 12.31 to 1 on the light page
    # and 1.54 on the dark one, so the dark appearance has its own accent, the
    # lighter violet of the pair chosen; each is measured against the page it
    # is drawn on.
    settings = the_settings()

    assert contrast(settings["theme"]["primaryColor"], LIGHT_PAGE) >= 3.0
    assert contrast(settings["theme"]["dark"]["primaryColor"], DARK_PAGE) >= 3.0


def test_the_current_step_is_written_in_the_accent_of_its_appearance():
    """The current step is written in the accent of its appearance, and 4.5 to
    1 is asked of it as text (1.4.3).
    """
    # The progress line writes the step the person is on in Streamlit's
    # violet text (decided on 18 September, UX-05), and the file sets that
    # violet to the accent of each appearance, so the step is #440099 on the
    # light page and the lighter violet on the dark one, where #440099 is all
    # but invisible.
    settings = the_settings()
    light, dark = settings["theme"], settings["theme"]["dark"]

    assert light["violetTextColor"] == light["primaryColor"]
    assert dark["violetTextColor"] == dark["primaryColor"]
    assert contrast(light["violetTextColor"], LIGHT_PAGE) >= 4.5
    assert contrast(dark["violetTextColor"], DARK_PAGE) >= 4.5


def test_the_panels_text_and_lines_clear_their_ratios():
    """The panel is the accent's violet with white text on both appearances (asked
    for on 17 September, UX-05), and 1.4.3 asks 4.5 to 1 of text.
    """
    # The two lines drawn across it in picture 20 are white too, held to the 3
    # to 1 of a boundary. The boxes Streamlit draws in the panel take their own
    # text colour on their own violet, and 4.5 is asked of them as well; the
    # blue one is the panel's instruction box since the second session of 19
    # September, and is the same box as the other three.
    panel = the_settings()["theme"]["sidebar"]

    assert panel["backgroundColor"] == the_settings()["theme"]["primaryColor"]
    assert contrast(panel["textColor"], panel["backgroundColor"]) >= 4.5
    assert contrast(panel["borderColor"], panel["backgroundColor"]) >= 3.0
    for colour in ("green", "red", "yellow", "blue"):
        text = panel[f"{colour}TextColor"]
        box = panel[f"{colour}BackgroundColor"]
        assert contrast(text, box) >= 4.5, colour
        assert contrast(box, panel["backgroundColor"]) >= 1.5, "shows as a box"
    assert (panel["blueTextColor"], panel["blueBackgroundColor"]) == \
        (panel["greenTextColor"], panel["greenBackgroundColor"])


def test_the_pages_style_rule_draws_the_buttons_in_the_owners_violets():
    """The two action buttons are lettered by a style rule in the page, in the
    violet the theme file sets, and bordered in the lighter violet of the pair.
    """
    # The page cannot read the file at run time without a branch for a missing
    # setting, so the lettering colour is written in both places and kept the
    # same here: a change to one without the other would leave the buttons in a
    # violet the panel is not. The figures given for the border (3.73 against
    # the panel, 3.3 against the face) are held to the 3 to 1 of a control's
    # boundary.
    accent = the_settings()["theme"]["primaryColor"]
    page = pathlib.Path(app.__file__).read_text(encoding="utf-8")

    rule_start = page.index(".st-key-store button, .st-key-send-back button")
    rule = page[rule_start:page.index("}", rule_start)]
    assert f"color: {accent};" in rule
    border = rule[rule.index("border: 3px solid ") + len("border: 3px solid "):][:7]
    assert contrast(accent, LIGHT_PAGE) >= 4.5, "the lettering on the white face"
    assert contrast(border, accent) >= 3.0, "the border against the panel"
    assert contrast(border, LIGHT_PAGE) >= 3.0, "the border against the face"


def test_every_setting_is_one_streamlit_knows():
    """Streamlit ignores a key it does not know without saying so. Asked for
    an unknown key, it raises, which is the check."""
    settings = the_settings()
    for name in settings["theme"]:
        if name not in ("dark", "sidebar"):
            streamlit.config.get_option(f"theme.{name}")
    for name in settings["theme"]["dark"]:
        streamlit.config.get_option(f"theme.dark.{name}")
    for name in settings["theme"]["sidebar"]:
        streamlit.config.get_option(f"theme.sidebar.{name}")
    for name in settings["client"]:
        streamlit.config.get_option(f"client.{name}")


def test_the_toolbar_hides_the_deploy_control_and_keeps_the_appearance_menu():
    """Asked on 17 September: whether Streamlit's Deploy control, which publishes
    an application to its cloud and is nothing of this project's, could go
    without losing the Settings menu and its choice of appearance (UX-05).
    """
    # Streamlit's own description of the setting says which value does that,
    # and the description is read here rather than remembered, so a Streamlit
    # that changes the meaning fails the test.
    assert the_settings()["client"]["toolbarMode"] == "viewer"

    streamlit.config.get_config_options(force_reparse=True)
    described = streamlit.config._config_options["client.toolbarMode"].description
    viewer = described[described.index('"viewer"'):described.index('"minimal"')]
    assert "Hide the developer options" in viewer
    assert "deploy button" in viewer
    assert "theme toggle" in described and "viewer options" in described
