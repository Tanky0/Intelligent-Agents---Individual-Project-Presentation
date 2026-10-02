"""Checks the rule that decides whether a saved reply may be published."""
# The rule has two halves and both can fail quietly. Reading a reply for the text
# of a paper can miss some, and judging a basis can accept one that says nothing.
# A mistake in either publishes somebody else's work, which is the failure R-11
# was about, so each case is written down rather than reasoned about once.
#
# Nothing here calls a service or a model. Every reply is invented, and none of
# them is written anywhere: these tests ask the rule questions, they do not
# approve anything.

import json

import pytest

from research_agent.sources import publication_rights as rights

A_LICENCE = {
    "kind": "open licence",
    "applies_to": "a_reply.json",
    "licence": "CC BY 4.0",
    "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    "source": "https://doi.org/10.0000/example",
    "attribution": "A. Author and others, in a journal, 2026",
    "changes": "contact details were removed",
    "redaction": "contact details were removed",
}


@pytest.mark.parametrize("record, why", [
    ({"url": "https://example.test/search", "body": "{}"},
     "an address the check does not know"),
    ({"url": "https://api.crossref.org/works", "body": "not json at all"},
     "a Crossref body that will not parse"),
    ({"url": "https://eutils.ncbi.nlm.nih.gov/efetch", "body": "<broken"},
     "NCBI XML that will not parse"),
    ({"url": "https://api.semanticscholar.org/graph/v1", "body": "{}"},
     "a Semantic Scholar body with no data in it"),
    ({"something": "unrecognised"},
     "a record in no shape this application writes"),
    ({"url": "https://api.crossref.org/works", "body": 12},
     "a body that is not text"),
])
def test_a_reply_that_cannot_be_read_is_refused_however_good_the_basis(record, why):
    """A basis cannot answer for text nobody could find."""
    # This is the failure the three answers exist to prevent. An earlier version
    # used one string for both "article text found" and "could not be read", and
    # judged a basis against whether that string was empty. A valid licence then
    # published every case in this list: the reply was not known to be safe, it
    # was merely unreadable, and unreadable was being treated as safe.
    #
    # Refusing here is not a judgement that these replies hold article text. It
    # is a refusal to guess either way about a file nobody can inspect.
    found = rights.inspect(record)
    assert found.finding == rights.CANNOT_INSPECT, why

    problems = rights.problems_with_the_basis(A_LICENCE, found, "a_reply.json")
    assert problems, f"{why} was published on a licence that cannot describe it"


def test_a_reply_holding_nothing_needs_no_basis():
    """Most replies are titles and identifiers, which are facts about papers
    rather than the papers. Requiring a basis for those would make the rule
    something people work around rather than use."""
    record = {"url": "https://api.crossref.org/works",
              "body": json.dumps({"message": {"items": [{"title": ["a paper"]}]}})}

    found = rights.inspect(record)

    assert found.finding == rights.NO_ARTICLE_TEXT
    assert rights.problems_with_the_basis(None, found, "a_reply.json") == []


def test_an_abstract_without_a_basis_is_refused():
    """The case that started R-11: ten Crossref records holding abstracts were
    published because nothing asked."""
    record = {"url": "https://api.crossref.org/works",
              "body": json.dumps({"message": {"items": [
                  {"title": ["a paper"], "abstract": "Background: this study..."}]}})}

    found = rights.inspect(record)

    assert found.finding == rights.ARTICLE_TEXT_FOUND
    assert rights.problems_with_the_basis(None, found, "a_reply.json")


def test_a_basis_written_for_another_reply_does_not_cover_this_one():
    """One basis for a whole batch would let a licence read from one paper
    stand for a different paper approved in the same command."""
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    problems = rights.problems_with_the_basis(A_LICENCE, found, "another.json")

    assert any("written for" in problem for problem in problems)


def test_a_licence_forbidding_changes_is_refused_when_the_reply_was_changed():
    """The unresolved risk R-11 recorded, refused rather than repeated."""
    # Publishing can remove contact details, and whether it did is measured for
    # each reply rather than assumed. A NoDerivatives licence permits sharing the
    # work and not a changed version of it, so where the removal did change a
    # reply, the fix for R-10 is what makes that reply unpublishable here.
    #
    # Reading a redaction as making a changed version is this project's cautious
    # choice. A licence written about adaptations does not obviously speak to
    # removing an email address, and the refusal says as much rather than
    # presenting it as settled.
    basis = {**A_LICENCE, "licence": "CC BY-ND 4.0",
             "licence_url": "https://creativecommons.org/licenses/by-nd/4.0/"}
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    unchanged = rights.problems_with_the_basis(basis, found, "a_reply.json",
                                               content_was_changed=False)
    changed = rights.problems_with_the_basis(basis, found, "a_reply.json",
                                             content_was_changed=True)

    assert unchanged == [], "an unchanged reply under this licence is fine"
    assert any("changed version" in problem for problem in changed)


def test_a_public_domain_dedication_does_not_require_attribution():
    """CC0 asks for nothing back. Requiring attribution for it would be
    inventing an obligation, which is as wrong as missing one."""
    basis = {**A_LICENCE, "licence": "CC0 1.0",
             "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/"}
    basis.pop("attribution")
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    problems = rights.problems_with_the_basis(basis, found, "a_reply.json",
                                              content_was_changed=True)

    assert problems == []


def test_share_alike_requires_the_changed_version_to_be_offered_alike():
    """A ShareAlike licence asks that a changed version carry compatible terms."""
    # The licence would accept some other licences its author considers
    # compatible. No list of those is kept here, so the basis is asked for the
    # same licence, which is narrower than the licence requires. That is this
    # project's rule and the refusal says so rather than attributing it to the
    # licence.
    basis = {**A_LICENCE, "licence": "CC BY-SA 4.0",
             "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/"}
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    silent = rights.problems_with_the_basis(basis, found, "a_reply.json",
                                            content_was_changed=True)
    said = rights.problems_with_the_basis(
        {**basis, "shared_alike_under": "CC BY-SA 4.0"}, found, "a_reply.json",
        content_was_changed=True)

    assert any("compatible terms" in problem for problem in silent)
    assert said == []


def test_a_licence_nobody_has_read_is_not_a_basis():
    """Refused because nobody read it, not because it forbids anything."""
    basis = {**A_LICENCE, "licence": "some publisher's own terms"}
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    assert rights.problems_with_the_basis(basis, found, "a_reply.json")


def test_written_permission_has_to_be_something_somebody_could_produce():
    """A name and a date are not evidence. What it covers, where it is kept,
    and that whoever gave it could give it, are the parts that make it
    checkable by somebody who was not there."""
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")
    thin = {"kind": "written permission", "applies_to": "a_reply.json",
            "granted_by": "somebody", "granted_on": "last week",
            "redaction": "contact details were removed"}

    problems = rights.problems_with_the_basis(thin, found, "a_reply.json")

    assert any("covers" in problem for problem in problems)
    assert any("kept" in problem for problem in problems)
    assert any("authorise" in problem for problem in problems)


def test_full_text_is_not_published_on_written_permission_here():
    """This project's own rule rather than a rule of copyright, and the
    refusal says so. Permission from a rights holder can allow republication;
    what it cannot do is let a reader check the terms for themselves."""
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND,
                              "an NCBI reply holding full text")
    permission = {"kind": "written permission", "applies_to": "a_reply.json",
                  "granted_by": "the publisher", "granted_on": "2026-09-01",
                  "covers": "this article", "evidence": "docs/permissions/",
                  "grantor_could_authorise": True,
                  "redaction": "contact details were removed"}

    problems = rights.problems_with_the_basis(permission, found, "a_reply.json")

    assert any("its own rule" in problem for problem in problems)


@pytest.mark.parametrize("body, why", [
    ("<mystery><paper>Complete article prose</paper></mystery>",
     "valid XML whose outermost element nothing recognises"),
    ('{"mystery": "Complete article prose"}',
     "valid JSON naming nothing the check knows"),
    ('{"header": {"type": "esearch"}, "unexpected": {"text": "prose"}}',
     "JSON with an NCBI header but no answer the check knows"),
])
def test_an_unknown_but_valid_ncbi_shape_is_not_read_as_safe(body, why):
    """Finding nothing only means something in an answer whose shape is known."""
    # NCBI serves several answers, and a fetch returns whatever the article
    # itself contains. So a well-formed reply that mentions none of the names
    # looked for has not been shown to hold no article text: it has been shown to
    # use different names. Both bodies here carry a whole article in an element
    # nothing looks at, and both were read as safe until the recognised shapes
    # were listed.
    found = rights.inspect({"url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
                            "body": body})

    assert found.finding == rights.CANNOT_INSPECT, why


@pytest.mark.parametrize("body", [
    '{"header": {"type": "esearch"}, "esearchresult": {"idlist": ["1", "2"]}}',
    '{"header": {"type": "elink"}, "linksets": [{"ids": ["1"]}]}',
    "<eSearchResult><IdList><Id>1</Id></IdList></eSearchResult>",
    "<eLinkResult><LinkSet><IdList><Id>1</Id></IdList></LinkSet></eLinkResult>",
])
def test_the_ncbi_answers_the_project_actually_uses_still_read_as_safe(body):
    """The shapes the demonstration set is made of, kept working."""
    # A rule tight enough to refuse everything would be no use. These are the
    # answers the searches and link lookups return: counts, identifiers and the
    # query as it was understood. Every name that can appear in one is written
    # down, so finding no abstract in one really does mean there is none.
    found = rights.inspect({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
                            "body": body})

    assert found.finding == rights.NO_ARTICLE_TEXT


@pytest.mark.parametrize("body", [
    "<eSearchResult><paper>Complete article prose</paper></eSearchResult>",
    '{"esearchresult": {}, "mystery": "Complete article prose"}',
    '{"header": {"type": "elink"}, "linksets": [{"note": "prose"}]}',
])
def test_a_known_answer_carrying_unknown_fields_is_refused(body):
    """The outermost name is not the shape."""
    # An answer that merely begins <eSearchResult> can hold anything inside it,
    # and both of the first two here carry a whole article under a name nothing
    # looks at. Checking the outermost element let them through, which made the
    # list a label allowlist rather than a shape allowlist. Every name in the
    # answer is checked now.
    found = rights.inspect({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
                            "body": body})

    assert found.finding == rights.CANNOT_INSPECT


def test_a_fetched_article_is_never_read_as_holding_no_article_text():
    """A paper's own shape is too large to write down."""
    # A search answer can be listed element by element, so finding no abstract in
    # one means something. An article cannot: it is whatever the publisher put
    # in it, and the elements this check looks for are a handful of the names it
    # might use. So a fetch that shows no abstract is reported as something this
    # check could not read, rather than as a paper without one.
    #
    # The cost is that a genuinely metadata-only fetch needs a recorded basis it
    # may not strictly need. That is the direction to be wrong in.
    found = rights.inspect({
        "url": "https://eutils.ncbi.nlm.nih.gov/efetch.fcgi",
        "body": "<PubmedArticleSet><PubmedArticle><PMID>1</PMID>"
                "</PubmedArticle></PubmedArticleSet>"})

    assert found.finding == rights.CANNOT_INSPECT
    assert "too large to write down" in found.description


def test_the_licence_link_has_to_be_the_licence():
    """A link nobody follows is not evidence of anything."""
    # Attribution under these licences means saying where the terms are, so the
    # basis carries a link. Checking only that something was typed there let
    # https://example.invalid/not-the-licence stand for the licence as readily as
    # the real address, which is a record that looks complete and shows nothing.
    wrong = {**A_LICENCE, "licence_url": "https://example.invalid/not-the-licence"}
    other = {**A_LICENCE,
             "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/"}
    found = rights.Inspection(rights.ARTICLE_TEXT_FOUND, "an abstract")

    assert rights.problems_with_the_basis(wrong, found, "a_reply.json")
    assert rights.problems_with_the_basis(other, found, "a_reply.json"), \
        "the link has to be to the licence named, not to some other one"
    assert rights.problems_with_the_basis(A_LICENCE, found, "a_reply.json") == []


@pytest.mark.parametrize("body, why", [
    (('<eSearchResult><IdList><Id paper="Complete article prose">1</Id>'
      '</IdList></eSearchResult>'),
     "an attribute holding the article, on an element from the list"),
    (("<eSearchResult><IdList><Id>Complete article prose</Id></IdList>"
      "</eSearchResult>"),
     "the article as the text of a field that holds identifiers"),
    ('{"esearchresult": {"idlist": ["Complete article prose"]}}',
     "the same in JSON"),
    ('{"esearchresult": {"count": "not a number"}}',
     "a count that is not a number"),
    ('{"esearchresult": {"idlist": "1"}}',
     "a list of identifiers that is not a list"),
])
def test_a_known_field_holding_something_it_should_not_is_refused(body, why):
    """Names are not a shape."""
    # Checking that every name in an answer is one of the expected names still
    # left the values free, so the article was simply moved inside a field that
    # was on the list. An identifier is digits, a count is a number, and a list
    # of identifiers is a list; anything else in those places is a reply this
    # check cannot read.
    found = rights.inspect({"url": "https://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
                            "body": body})

    assert found.finding == rights.CANNOT_INSPECT, why


@pytest.mark.parametrize("url, why", [
    ("https://evil.example/?next=eutils.ncbi.nlm.nih.gov",
     "the service name in a query parameter anybody can write"),
    ("https://eutils.ncbi.nlm.nih.gov.evil.example/esearch.fcgi",
     "the service name as the start of somebody else's domain"),
    ("http://eutils.ncbi.nlm.nih.gov/esearch.fcgi",
     "the right host over a connection this project does not make"),
    ("https://api.crossref.org.evil.example/works",
     "the same trick on Crossref"),
])
def test_the_address_is_read_as_a_host_and_not_as_text(url, why):
    """Which reader runs is decided by the host, not by a substring."""
    # Looking for the service name anywhere in the address meant anybody who
    # could write a URL could choose how their reply was inspected. A body is
    # then read by the wrong reader, and a body no reader was written for is read
    # as one that was.
    found = rights.inspect({"url": url,
                            "body": '{"esearchresult": {"idlist": ["1"]}}'})

    assert found.finding == rights.CANNOT_INSPECT, why


@pytest.mark.parametrize("url", [
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed",
    "https://api.crossref.org/works/10.0000/example",
    "https://api.semanticscholar.org/graph/v1/paper/search",
])
def test_the_real_addresses_still_reach_their_readers(url):
    """A rule that refused the project's own addresses would be no use."""
    assert rights._which_service(url) is not None
