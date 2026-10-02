"""Loads shared settings from environment variables."""
# API rate limits and caller details are defined here so every source uses the
# same values. Centralising these settings prevents individual source adapters
# from becoming inconsistent or outdated, and reduces the risk of one adapter
# breaching a service's usage terms while the others comply.
#
# These settings support compliance with each service's usage requirements. NCBI
# sets request limits and requires applications to provide a tool name and
# contact email. Crossref does not require identification, but routes clients
# that identify themselves to a better-served request pool.

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The folder holding pyproject.toml, one level above this package. A wrong
# count here loads no .env at all and fails silently, so a test pins it.
REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    ncbi_api_key: str = ""
    ncbi_email: str = ""
    ncbi_tool: str = "uoeo-research-agent"

    semantic_scholar_api_key: str = ""
    crossref_email: str = ""

    # When true, saved replies are used instead of calling the services, so a
    # run can be repeated with no network. See sources/saved_replies.py.
    demo_mode: bool = False

    # Every reply is saved as it arrives, so whatever was run last can be run
    # again with no network. On by default, because a demonstration set that
    # depends on remembering to switch something on beforehand is one that will
    # eventually be missing. Tests are kept out of it by tests/conftest.py.
    record_replies: bool = True

    @property
    def ncbi_rate_limit(self) -> float:
        """Return the requests per second NCBI permits for this client."""
        # NCBI allows three per second without a key and ten with one. The value
        # is derived from the key rather than fixed, so the limiter cannot be
        # left at ten after the key is removed.
        if self.ncbi_api_key:
            return 10.0
        return 3.0

    def user_agent(self, service: str) -> str:
        """Returns the text identifying this application to a service."""
        # This is what Crossref uses to decide which queue handles the request.
        # NCBI asks for the same details, but expects them alongside the request
        # rather than in this text, so the PubMed source sends them separately.
        contact = self.crossref_email or self.ncbi_email or "unknown"
        return f"uoeo-research-agent/0.1 ({service}; mailto:{contact})"


settings = Settings()
