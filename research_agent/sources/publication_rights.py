"""Says whether a saved reply may be published, and on what basis."""
# Publishing a reply into the repository is republishing whatever it holds. A
# search result listing titles, authors and identifiers is bibliographic fact,
# and Crossref says plainly that this metadata is not subject to copyright. An
# abstract is not that. Crossref's own terms say abstracts are subject to
# publisher or author copyright, and an article body from PMC carries whatever
# licence its publisher gave it. So the two cannot travel the same route into a
# public folder, and the difference has to be found by reading the reply rather
# than by trusting where it came from (R-11).
#
# This lives beside the code that publishes rather than in a test, because a
# test can only say afterwards that something got in. The check that matters is
# the one standing in the doorway.
#
# Reading a reply gives one of three answers, and keeping them apart is the
# whole point of the arrangement:
#
#   NO_ARTICLE_TEXT     nothing was found that needs a basis.
#   ARTICLE_TEXT_FOUND  an abstract or an article body is in there.
#   CANNOT_INSPECT      it could not be read for either.
#
# The third is not the first. An unreadable body is not a body known to be safe,
# and a service this was not written for is not a service known to return no
# article text. A basis can answer the second, because somebody has looked at the
# reply and said what allows it to be published. Nothing answers the third: a
# basis describing text nobody could find is a statement about an unknown.
#
# An earlier version of this file collapsed the second and third into one string
# and let a basis clear either. A valid-looking licence then published a
# malformed Crossref reply, a broken piece of NCBI XML and a reply from an
# address nothing recognised. That is the failure this separation exists to stop.

import collections
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET

# What an element holding the text of a paper is called, in the XML NCBI
# returns. PubMed uses AbstractText, and articles from PMC use the JATS names.
TEXT_ELEMENTS = {"abstracttext": "an abstract", "abstract": "an abstract",
                 "body": "full text"}

NO_ARTICLE_TEXT = "no recognised article-text field"
ARTICLE_TEXT_FOUND = "article text found"
CANNOT_INSPECT = "cannot inspect"

# The NCBI answers whose shape this check knows. Finding no article text is
# only meaningful in one of these: they are lists of identifiers and links,
# where the absence of an abstract element means there is no abstract. In an
# answer of some other shape it means only that the names looked for were not
# used, which is not the same thing at all.
#
# The outermost element of the XML answers. PubmedArticleSet is included
# because it is the answer to a fetch, and one holding no abstract element
# genuinely holds no abstract.
# The answers whose whole shape is known, element by element. A search or a
# link lookup returns counts, identifiers and the query as it was understood,
# and nothing else, so every name that may appear in one can be written down.
# That is what makes finding no article text in one mean something.
#
# Checking only the outermost element was not enough. An answer that merely
# begins <eSearchResult> can hold anything at all inside it, and
# <eSearchResult><paper>Complete article prose</paper></eSearchResult> passed
# as safe. What is checked now is every element in the answer.
NCBI_SEARCH_ELEMENTS = {
    "esearchresult", "count", "retmax", "retstart", "querykey", "webenv",
    "idlist", "id", "translationset", "translation", "from", "to",
    "translationstack", "termset", "term", "field", "explode", "op",
    "querytranslation", "errorlist", "warninglist", "phrasenotfound",
    "fieldnotfound", "phraseignored", "quotedphrasenotfound",
    "outputmessage", "error",
}

# The XML names and the JSON names differ for the same answer: the XML uses
# LinkSetDb and IdList, the JSON uses linksetdbs and ids. Both are listed,
# because a reply is checked against this whichever form it arrived in.
NCBI_LINK_ELEMENTS = {
    "elinkresult", "linkset", "linksets", "linksetdb", "linksetdbs",
    "dbfrom", "dbto", "idlist", "ids", "id", "link", "links", "linkname",
    "linksetdbhistory", "querykey", "webenv", "info", "error", "idurllist",
    "idurlset", "objurl", "url", "linkinfo", "score",
}

# The same answers in JSON, by the name at the top of them.
NCBI_JSON_ANSWERS = {"esearchresult": NCBI_SEARCH_ELEMENTS,
                     "linksets": NCBI_LINK_ELEMENTS}

# The XML answers, by their outermost element.
NCBI_XML_ANSWERS = {"esearchresult": NCBI_SEARCH_ELEMENTS,
                    "elinkresult": NCBI_LINK_ELEMENTS}

# The containers a fetch returns, in JATS and in PubMed's own format. These
# hold a paper, and their shape is far too large to write down, so an absence
# of the elements looked for is not evidence that they hold no article text.
# They are named here only so the refusal can say what they are: a fetch is
# refused as something this check cannot read rather than passed as safe.
NCBI_ARTICLE_CONTAINERS = {"pubmedarticleset", "pubmedarticle", "article",
                           "pmc-articleset", "esummaryresult"}

# What reading a reply produced: which of the three answers, and a sentence
# saying what was found, for somebody reading a refusal.
Inspection = collections.namedtuple("Inspection", "finding description")

# Used inside the JSON search below to mean the same as CANNOT_INSPECT, kept
# separate only because that function is answering about one field rather than
# about the whole reply.
_UNREADABLE = "unreadable"

# The licences a body may be published under here, and what each one asks for.
#
#   attribution_required  whether credit has to be given. CC0 is a dedication
#                         to the public domain and does not require it, so
#                         demanding it would be inventing an obligation.
#   changes_allowed       whether a changed version may be distributed. The
#                         NoDerivatives licences say a changed version may not
#                         be shared, and publishing here can change a reply by
#                         removing contact details from it. Whether it did is
#                         measured for each reply rather than assumed: a reply
#                         that was not changed is not a changed version, and
#                         refusing it would be inventing a restriction.
#                         Treating a redaction as making one is this project's
#                         conservative reading, since a licence written about
#                         adaptations does not obviously speak to removing an
#                         email address (R-11).
#   share_alike           whether a changed version has to be offered under the
#                         same terms. ShareAlike also allows some other
#                         licences the licensor considers compatible, and no
#                         list of those is kept here, so this asks for the same
#                         licence. That is narrower than the licence requires
#                         and is a rule of this project, not of the licence.
#   url                   the licence itself. The basis has to link to this
#                         one, so that a link nobody follows cannot stand in
#                         for a licence nobody read.
#
# A licence not listed is not refused because it forbids anything. It is
# refused because nobody has read it, and a licence nobody has read is not a
# basis for anything.
LICENCES = {
    "CC0 1.0": {
        "attribution_required": False, "changes_allowed": True,
        "share_alike": False,
        "url": "https://creativecommons.org/publicdomain/zero/1.0/"},
    "CC BY 4.0": {
        "attribution_required": True, "changes_allowed": True,
        "share_alike": False,
        "url": "https://creativecommons.org/licenses/by/4.0/"},
    "CC BY-SA 4.0": {
        "attribution_required": True, "changes_allowed": True,
        "share_alike": True,
        "url": "https://creativecommons.org/licenses/by-sa/4.0/"},
    "CC BY-NC 4.0": {
        "attribution_required": True, "changes_allowed": True,
        "share_alike": False,
        "url": "https://creativecommons.org/licenses/by-nc/4.0/"},
    "CC BY-NC-SA 4.0": {
        "attribution_required": True, "changes_allowed": True,
        "share_alike": True,
        "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"},
    "CC BY-ND 4.0": {
        "attribution_required": True, "changes_allowed": False,
        "share_alike": False,
        "url": "https://creativecommons.org/licenses/by-nd/4.0/"},
}

# How a piece of article text may come to be publishable. An open licence is
# the ordinary route. Written permission is the other, and it has to name who
# gave it, when, what it covers and where it is kept, so that it can be
# produced if it is ever asked for.
BASIS_KINDS = ("open licence", "written permission")


def _has_text(value):
    """True when a field holds something a reader would call text."""
    return isinstance(value, str) and value.strip() != ""


def _local_name(tag):
    """The element's own name, with any namespace taken off the front."""
    # An article from PMC arrives with its elements in a namespace, so the tag
    # reads {http://...}abstract rather than abstract. Comparing whole tags
    # misses those, which is a silent pass on exactly the files this rule is
    # about.
    return tag.rpartition("}")[2].lower()


def _abstract_text_in(items):
    """Says whether any of these records holds an abstract, or that it cannot say."""
    # Returns True, False, or None for cannot say. An abstract that is not text
    # is the third case rather than the second: a list holding the abstract in
    # pieces is not nothing, it is something this check cannot read, and reading
    # it as nothing is how an abstract would pass.
    for item in items:
        if not isinstance(item, dict):
            return None
        abstract = item.get("abstract")
        if abstract is None:
            continue
        if not isinstance(abstract, str):
            return None
        if abstract.strip():
            return True
    return False


def _article_text_in_ncbi_xml(article):
    """Says which element holding paper text was found, reading the XML."""
    for element in article.iter():
        holding = TEXT_ELEMENTS.get(_local_name(element.tag))
        if holding and "".join(element.itertext()).strip():
            return f"an NCBI reply holding {holding}"
    return None


def _text_named_anywhere_in(answer):
    """What this JSON holds under a name meaning article text."""
    # Three answers, the same three the whole check gives: the kind of text
    # found, None for none, or _UNREADABLE when something is named as article
    # text but is not text. A list holding an abstract in pieces is the third
    # case. Reading it as the second is how an abstract passes.
    #
    # NCBI answers a search in JSON rather than XML, and a search reply carries
    # identifiers. One carrying an abstract is not a search reply, whatever its
    # address says, so it is looked for rather than assumed absent.
    if isinstance(answer, dict):
        for name, value in answer.items():
            holding = TEXT_ELEMENTS.get(_local_name(str(name)))
            if holding:
                if _has_text(value):
                    return holding
                if value is not None and not isinstance(value, str):
                    return _UNREADABLE
            found = _text_named_anywhere_in(value)
            if found:
                return found
    elif isinstance(answer, list):
        for item in answer:
            found = _text_named_anywhere_in(item)
            if found:
                return found
    return None


# What a value in one of those answers is allowed to look like. Names alone
# were not enough: <Id>Complete article prose</Id> uses a name from the list
# and holds a whole article, and so does {"idlist": ["Complete article prose"]}.
# A field with a settled form is checked against it.
_AN_IDENTIFIER = re.compile(r"^(PMC)?[0-9]{1,12}$")
_A_NUMBER = re.compile(r"^-?[0-9]{1,12}$")
_A_TOKEN = re.compile(r"^[A-Za-z0-9_.:|-]{1,120}$")

# The fields whose form is settled, and the form. Anything else in the lists
# above is free text by design.
NCBI_VALUE_SHAPES = {
    "id": _AN_IDENTIFIER, "ids": _AN_IDENTIFIER, "idlist": _AN_IDENTIFIER,
    "link": _AN_IDENTIFIER, "links": _AN_IDENTIFIER,
    "count": _A_NUMBER, "retmax": _A_NUMBER, "retstart": _A_NUMBER,
    "querykey": _A_NUMBER, "score": _A_NUMBER, "explode": _A_TOKEN,
    "webenv": _A_TOKEN, "dbfrom": _A_TOKEN, "dbto": _A_TOKEN,
    "linkname": _A_TOKEN, "field": _A_TOKEN, "op": _A_TOKEN,
    # The header names the request that was answered.
    "type": _A_TOKEN, "version": _A_TOKEN,
}

# The fields that hold text with no settled form: the query as the service
# understood it, and messages about it. A reply carrying any of them cannot be
# read for article text, because nothing here can tell a search expression from
# a paragraph of a paper.
#
# They were bounded by length for a while, on the reasoning that a search
# expression is a line or two. That is the same measurement this project
# already threw out for saved model replies, and it failed the same way: eight
# hundred characters of abstract-like prose sat in querytranslation and passed.
# Length is not provenance. It was not provenance the first time either, and
# putting it back under a different field name did not make it so.
#
# Nothing that replays needs these. The PubMed adapter reads
# esearchresult.idlist and nothing else, so a reply carrying them is refused
# rather than measured, and a reply built without them can still be replayed.
NCBI_FREE_TEXT_FIELDS = {
    "querytranslation", "translationset", "translationstack", "translation",
    "term", "from", "to", "errorlist", "warninglist", "error", "info",
    "outputmessage", "phrasenotfound", "fieldnotfound", "phraseignored",
    "quotedphrasenotfound", "linkinfo",
}


# The fields that hold a collection rather than a value, and what kind. A
# field named as a list of identifiers has to be a list of them: given a bare
# string instead, the check below would look at one value and find it well
# formed, which is a shape nobody expected passing as one that was.
NCBI_CONTAINER_TYPES = {
    "idlist": list, "ids": list, "links": list, "linksets": list,
    "linksetdbs": list, "linkset": list, "linksetdb": list,
    "translationset": list, "translationstack": list, "termset": list,
    "linksetdbhistory": list, "idurllist": list, "idurlset": list,
    "objurl": list, "errorlist": dict,
    "warninglist": dict, "esearchresult": dict, "header": dict,
}


def _the_container_is_wrong_for(name, value):
    """Says why this field is not the kind of collection it should be, or None."""
    expected = NCBI_CONTAINER_TYPES.get(name)
    if expected is None or isinstance(value, expected):
        return None
    return (f"{name} holds {type(value).__name__} where "
            f"{expected.__name__} was expected")


def _the_value_is_wrong_for(name, value):
    """Says why this value does not belong under this name, or None."""
    # Anything that is not text is refused outright. A list is walked, because
    # idlist holds its identifiers in one.
    if isinstance(value, list):
        for item in value:
            wrong = _the_value_is_wrong_for(name, item)
            if wrong:
                return wrong
        return None

    if isinstance(value, dict):
        return None            # walked separately, name by name

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        value = str(value)

    if not isinstance(value, str):
        return f"{name} holds something that is not text"

    if name in NCBI_FREE_TEXT_FIELDS and value.strip():
        return (f"{name} holds text with no settled form, which cannot be read "
                f"for whether a paper is in it")

    shape = NCBI_VALUE_SHAPES.get(name)
    if shape is not None:
        return None if shape.match(value.strip()) else (
            f"{name} holds {value.strip()[:40]!r}, which is not the form that "
            f"field takes")

    # Every remaining name in the lists above holds an identifier, a count or a
    # token, and each of those has a shape checked just above. A name reaching
    # here has been allowed into the lists without anybody saying what it may
    # contain, so it is refused rather than passed.
    return f"{name} has no settled form, so nothing can be concluded from it"


def _wrong_values_in_json(answer):
    """The first value in this answer that does not belong, or None."""
    if isinstance(answer, dict):
        for name, value in answer.items():
            wrong = (_the_container_is_wrong_for(str(name).lower(), value)
                     or _the_value_is_wrong_for(str(name).lower(), value))
            if wrong:
                return wrong
            wrong = _wrong_values_in_json(value)
            if wrong:
                return wrong
    elif isinstance(answer, list):
        for item in answer:
            wrong = _wrong_values_in_json(item)
            if wrong:
                return wrong
    return None


def _wrong_values_in_xml(root):
    """The first element text or attribute that does not belong, or None."""
    # Attributes are refused outright. These answers carry their content in
    # element text, so an attribute is something this check was not written for,
    # and <Id paper="Complete article prose">1</Id> is what that looks like.
    for element in root.iter():
        name = _local_name(element.tag)
        if element.attrib:
            return (f"<{name}> carries attributes, which these answers do not "
                    f"use: {', '.join(sorted(element.attrib))}")
        text = (element.text or "").strip()
        if text:
            wrong = _the_value_is_wrong_for(name, text)
            if wrong:
                return wrong
    return None


def _every_name_in(answer):
    """Every field name appearing anywhere in this JSON, lowercased."""
    # The whole answer is walked rather than its top level, because a field
    # holding an article can sit at any depth and the point is to find names
    # nobody expected.
    names = set()
    if isinstance(answer, dict):
        for name, value in answer.items():
            names.add(str(name).lower())
            names |= _every_name_in(value)
    elif isinstance(answer, list):
        for item in answer:
            names |= _every_name_in(item)
    return names


# The hosts each reader was written for, exactly. Matched against the host part
# of the address and nothing else.
#
# Looking for the name anywhere in the address was how this started, and
# https://evil.example/?next=eutils.ncbi was read as a reply from NCBI: the
# name appeared in it, in a query parameter anybody can write. An address is
# not a claim about who answered, but if it is going to choose which reader
# runs, the part of it that names a host has to be the part that is read.
SERVICE_HOSTS = {
    "api.crossref.org": "crossref",
    "doi.org": "crossref",
    "dx.doi.org": "crossref",
    "api.semanticscholar.org": "semantic scholar",
    "eutils.ncbi.nlm.nih.gov": "ncbi",
    "www.ncbi.nlm.nih.gov": "ncbi",
    "pmc.ncbi.nlm.nih.gov": "ncbi",
}


def _which_service(url):
    """Which reader this address calls for, or None for one nothing knows."""
    # Only https is recognised. A recorded reply that arrived over plain http is
    # not one this project makes, and reading it as though it were would be
    # accepting the shape of an answer nobody here asked for.
    try:
        address = urllib.parse.urlparse(url or "")
    except ValueError:
        return None
    if address.scheme != "https":
        return None
    return SERVICE_HOSTS.get((address.hostname or "").lower())


def _found(description):
    """Article text was found, and this says what it was."""
    return Inspection(ARTICLE_TEXT_FOUND, description)


def _unreadable(description):
    """It could not be read, which is never something a basis can answer."""
    return Inspection(CANNOT_INSPECT, description)


# What the first answer actually establishes, said in those words. It is not
# "there is no third-party text in this reply": the Crossref and Semantic
# Scholar readers look for a field named abstract, and a Crossref record can
# carry unstructured references or an ethics statement under other names. What
# was checked is what is said.
_NOTHING_FOUND = Inspection(
    NO_ARTICLE_TEXT,
    "no recognised abstract or full-text field was found in it")


def _inspect_a_service_reply(record):
    """Reads a reply from a service for the text of a paper."""
    # The body decides, not the address. An earlier version guessed from the
    # address, decided Crossref returned only publication status, and missed ten
    # replies holding abstracts.
    body = record.get("body", "")
    if not isinstance(body, str):
        return _unreadable("its body is not text, so nothing could be read")

    service = _which_service(record.get("url", ""))

    if service == "crossref":
        try:
            message = json.loads(body)["message"]
        except (json.JSONDecodeError, TypeError, KeyError):
            return _unreadable("a Crossref reply that could not be inspected "
                               "for article text")
        items = message.get("items", [message]) if isinstance(message, dict) else None
        if items is None:
            return _unreadable("a Crossref reply that could not be inspected "
                               "for article text")
        found = _abstract_text_in(items)
        if found is None:
            return _unreadable("a Crossref reply that could not be inspected "
                               "for article text")
        return _found("a Crossref record holding an abstract") if found else _NOTHING_FOUND

    if service == "semantic scholar":
        try:
            papers = json.loads(body)["data"]
        except (json.JSONDecodeError, TypeError, KeyError):
            return _unreadable("a Semantic Scholar reply that could not be "
                               "inspected for article text")
        if not isinstance(papers, list):
            return _unreadable("a Semantic Scholar reply that could not be "
                               "inspected for article text")
        found = _abstract_text_in(papers)
        if found is None:
            return _unreadable("a Semantic Scholar reply that could not be "
                               "inspected for article text")
        if found:
            return _found("a Semantic Scholar record holding an abstract")
        return _NOTHING_FOUND

    if service == "ncbi":
        # Every reply from this service is inspected, whatever the address
        # asked for. Trusting the address to say what came back is how the
        # first version of this let a search reply through: a recording can
        # hold something other than what its address suggests, and that is
        # exactly the case a guard is for.
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            pass
        else:
            holding = _article_text_in_ncbi_xml(root)
            if holding:
                return _found(holding)

            outermost = _local_name(root.tag)
            expected = NCBI_XML_ANSWERS.get(outermost)
            if expected is None:
                if outermost in NCBI_ARTICLE_CONTAINERS:
                    return _unreadable(
                        f"a fetched article (<{outermost}>), whose shape is "
                        "too large to write down, so finding no abstract in it "
                        "is not evidence that there is none")
                return _unreadable(
                    f"an NCBI reply whose outermost element is <{outermost}>, "
                    "which is not one of the answers this check knows")

            unexpected = sorted({_local_name(element.tag)
                                 for element in root.iter()} - expected)
            if unexpected:
                return _unreadable(
                    f"an NCBI {outermost} holding elements this check does not "
                    f"know: {', '.join('<' + name + '>' for name in unexpected)}")

            wrong = _wrong_values_in_xml(root)
            if wrong:
                return _unreadable(f"an NCBI {outermost} where {wrong}")
            return _NOTHING_FOUND

        try:
            answer = json.loads(body)
        except json.JSONDecodeError:
            return _unreadable("an NCBI reply that could not be read as XML "
                               "or JSON")
        found = _text_named_anywhere_in(answer)
        if found == _UNREADABLE:
            return _unreadable("an NCBI reply that could not be inspected for "
                               "article text")
        if found:
            return _found(f"an NCBI reply holding {found}")

        # Finding no field named as article text is only reassuring in an
        # answer whose shape is known. NCBI serves several, and a fetch returns
        # whatever the article itself contains, so a well-formed reply naming
        # nothing familiar is a reply this check could not read rather than one
        # known to be safe: {"mystery": "..."} holds a whole article and
        # mentions none of the names looked for.
        if not isinstance(answer, dict):
            return _unreadable("an NCBI reply that is not a record, so its "
                               "shape is not one this check knows")

        known = [name for name in NCBI_JSON_ANSWERS if name in answer]
        if not known:
            return _unreadable("an NCBI reply in a shape this check does not "
                               "know, so nothing can be concluded from what it "
                               "does not contain")

        # Every name in it, not only the one at the top. An answer that merely
        # begins with esearchresult can carry anything beside it, and one that
        # did carried a whole article under a name of its own.
        expected = set(NCBI_JSON_ANSWERS[known[0]]) | {"header", "type", "version"}
        unexpected = sorted(_every_name_in(answer) - expected)
        if unexpected:
            return _unreadable(
                f"an NCBI {known[0]} holding fields this check does not know: "
                f"{', '.join(unexpected)}")

        wrong = _wrong_values_in_json(answer)
        if wrong:
            return _unreadable(f"an NCBI {known[0]} where {wrong}")
        return _NOTHING_FOUND

    return _unreadable("a reply from an address this check does not recognise: "
                       + (record.get("url") or "none given"))


def inspect(record):
    """Reads a saved reply and says which of the three answers it gives."""
    # A saved answer from the language model is a different kind of record. It
    # holds a question and an answer rather than a reply from a service, and the
    # question can carry paper text: the Curation Agent's prompts will hold
    # abstracts, and the Planning Agent's already hold titles. Nothing can read a
    # prompt and tell whose words are in it, so it is reported as article text
    # found rather than as unreadable: somebody who knows what went into the
    # prompt can say what allows it to be published, which is exactly what a
    # basis is for.
    #
    # An earlier version let a model reply through when it was under two thousand
    # characters, on the reasoning that an abstract is longer than that. Many
    # abstracts are shorter, and a prompt of seven hundred characters carrying
    # one passed. Length is not provenance, and measuring it was a guess wearing
    # the clothes of a rule.
    if not isinstance(record, dict):
        return _unreadable("a saved reply that could not be read")

    if "url" not in record:
        # The run is set aside before the shape is compared, because it is
        # the one field a model reply may or may not have: a reply captured
        # before the field existed, or outside a run, has none, and both are
        # this application's own shape. Listing two exact shapes was the
        # alternative, and it would need a third the next time a field is added.
        fields = sorted(record)
        if "run" in fields:
            fields.remove("run")
        if fields != ["answer", "captured_at", "model", "question"]:
            return _unreadable("a saved reply in a shape this check does not "
                               "recognise")
        return _found("a saved model reply, whose prompt cannot be read for "
                      "whose words are in it")

    return _inspect_a_service_reply(record)


def article_text_in(record):
    """The description when something needs a basis, or None when nothing does."""
    # Kept for reading a reply on its own. The gate uses inspect() instead,
    # because this cannot tell a refusal that a basis may answer from one that
    # it may not.
    found = inspect(record)
    return None if found.finding == NO_ARTICLE_TEXT else found.description


def problems_with_the_basis(basis, found, name=None, content_was_changed=False):
    """Returns the reasons this basis cannot support publishing this reply."""
    # An empty list means the basis is one that can be recorded and checked
    # later. It does not mean the licence is the right one for that article: only
    # the person approving can know that. What is refused here is a basis too
    # vague to be checked at all, because a basis nobody can check is the same as
    # none, and writing one into the manifest would make the record look stronger
    # than it is.
    #
    # content_was_changed says whether publishing altered the reply. It matters
    # because this project removes contact details on the way in, and a licence
    # forbidding derivatives does not permit a changed version to be distributed.
    if found.finding == NO_ARTICLE_TEXT:
        return []

    if found.finding == CANNOT_INSPECT:
        # No basis answers this, so none is asked for. A basis says what allows
        # a known thing to be published; here nobody knows what is in the file.
        return [(f"it could not be read for article text "
                 f"({found.description}), and a basis cannot describe what "
                 f"nobody could find")]

    if not isinstance(basis, dict):
        return [(f"it holds {found.description}, and no basis for "
                 f"publishing it was given")]

    problems = []

    # A basis is written for one reply. Applying one to everything approved in
    # the same command would let a licence read from one paper stand for
    # another, which is the mistake it exists to prevent.
    if name is not None and basis.get("applies_to") != name:
        problems.append(f"the basis was written for "
                        f"{basis.get('applies_to') or 'nothing named'}, not for "
                        f"{name}")

    kind = basis.get("kind")
    if kind not in BASIS_KINDS:
        problems.append("the basis does not say which of "
                        + " or ".join(BASIS_KINDS) + " it rests on")

    # This project's own rule, not a statement of the law. Permission from a
    # rights holder can allow an article body to be republished. It is refused
    # here because a licence anybody can read is checkable by anybody, and a
    # permission held in somebody's correspondence is not, which for a body of
    # somebody else's text is a difference worth being cautious about.
    if "full text" in found.description and kind == "written permission":
        problems.append("this project publishes full text only under an open "
                        "licence, so a reader can check the terms without "
                        "asking anybody. That is its own rule, not a rule of "
                        "copyright")

    if kind == "open licence":
        problems.extend(_problems_with_a_licence(basis, content_was_changed))

    if kind == "written permission":
        for field, saying in (("granted_by", "who gave it"),
                              ("granted_on", "when it was given"),
                              ("covers", "what it covers"),
                              ("evidence", "where the permission is kept")):
            if not _has_text(basis.get(field)):
                problems.append(f"written permission has to say {saying}")
        if basis.get("grantor_could_authorise") is not True:
            problems.append("written permission has to record that whoever "
                            "gave it was able to authorise republication")

    if not _has_text(basis.get("redaction")):
        problems.append("the basis has to say what was taken out of the reply "
                        "before it was published, or that nothing was")

    return problems


def _problems_with_a_licence(basis, content_was_changed):
    """The reasons an open licence does not support publishing this reply."""
    named = basis.get("licence")
    terms = LICENCES.get(named)
    if terms is None:
        return ["the licence named is not one of " + ", ".join(sorted(LICENCES))]

    problems = []
    given = (basis.get("licence_url") or "").strip().rstrip("/")
    if given != terms["url"].rstrip("/"):
        # Compared with the licence's own address rather than only checked for
        # being non-empty. A link nobody follows is not evidence of anything,
        # and https://example.invalid/not-the-licence satisfied the earlier
        # version of this as readily as the real one.
        problems.append(f"the basis has to link to {named} itself, at "
                        f"{terms['url']}")
    if not _has_text(basis.get("source")):
        problems.append("the basis has to say which work it covers, by DOI or "
                        "address")

    if terms["attribution_required"] and not _has_text(basis.get("attribution")):
        problems.append(f"{named} requires attribution, and none was given")

    if content_was_changed:
        if not terms["changes_allowed"]:
            problems.append(f"{named} does not permit a changed version to be "
                            "shared, and publishing changed this reply by "
                            "removing contact details. Treating that removal "
                            "as making a changed version is this project's "
                            "cautious reading rather than a settled one")
        if not _has_text(basis.get("changes")):
            problems.append("publishing changed this reply, and the basis has "
                            "to say what was changed")
        if terms["share_alike"] and basis.get("shared_alike_under") != named:
            problems.append(f"{named} requires a changed version to be offered "
                            f"under compatible terms. This project asks for "
                            f"{named} itself, which is narrower than the "
                            f"licence requires, because no list of the "
                            f"licences it would accept is kept here")

    return problems
