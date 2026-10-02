"""Keeps DOI comparison independent of connections and certificate setup."""
# Crossref and deduplication must agree about which identifier a record carries.
# Sharing this function avoids two rules drifting apart, while
# keeping it here means comparing identifiers does not load the HTTP handling.

_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)


def normalise_doi(raw):
    """Keeps differences in source formatting from counting one paper twice."""
    # Sources can return a DOI as an address or with different capitalisation.
    # Both still identify the same paper, so comparison uses the plain DOI.
    doi = raw.strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi
