"""Tests for loading the settings (UT-01)."""
# The project folder is found by counting how many folders to go up from
# config.py. Moving that file changes the count, and getting it wrong does not
# raise anything: the .env file is simply not found, every setting keeps its
# empty default, and the first request to a service is refused for a reason that
# points at the request rather than at the missing file. That happened once
# during development, so it is checked here.

from research_agent.config import REPO_ROOT, settings


def test_the_project_folder_was_found():
    assert (REPO_ROOT / "pyproject.toml").exists(), (
        f"The project folder was worked out as {REPO_ROOT}, which holds no "
        "pyproject.toml. The number of folders counted in config.py is wrong."
    )


def test_the_settings_file_is_looked_for_in_the_project_folder():
    """.env is deliberately kept out of the repository, so its example is
    checked instead. Both sit in the same place."""
    assert (REPO_ROOT / ".env.example").exists()


def test_the_ncbi_rate_depends_on_whether_a_key_is_present():
    """NCBI allows three requests a second without a key and ten with one."""
    # Working the figure out from the key means it cannot be left at ten after
    # the key has been removed, which would send requests faster than allowed.
    assert settings.ncbi_rate_limit in (3.0, 10.0)

    if settings.ncbi_api_key:
        assert settings.ncbi_rate_limit == 10.0
    else:
        assert settings.ncbi_rate_limit == 3.0


def test_the_identifying_text_carries_a_contact_address():
    """Crossref uses this to decide which queue handles the request."""
    identifying_text = settings.user_agent("crossref")

    assert "uoeo-research-agent" in identifying_text
    assert "mailto:" in identifying_text
    assert "crossref" in identifying_text
