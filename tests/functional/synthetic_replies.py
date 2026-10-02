"""The project-owned synthetic replies the functional tests replay (FT-01 to
FT-09).
"""
# Every reply here was invented for these tests on 14 September 2026, and
# that sentence is the record of where they came from:
# nothing below is a recording of PubMed or of the model, and no paper below is
# a real one. The titles, abstracts, authors and journals are invented, and the
# identifiers are nine digits beginning with 9, a shape chosen so they cannot
# be mistaken for the identifiers in this project's recordings and samples;
# whether PubMed has ever issued them is not something this project has
# checked or relies on. What keeps them out of the class of thing R-12 was
# about is that the text is invented and says so here, not the numbers. A
# functional test built on these shows that the system behaves the same way
# twice, and says nothing about what a service or the model returns; the live
# run supplies that separately.
#
# They are kept as Python rather than as data files for two reasons. A module
# under tests/ is read in review like any other code, and the release check
# already allows it by pattern; a data file holding what looks like an abstract
# would be one more file that has to be named and read one at a time. And the
# record a fetch returns is built here from PAPERS, so an invented paper's
# title, abstract and authors are written once; which identifiers a search
# finds is written separately in SEARCHES, on purpose, so a search can name a
# record the fetch does not hold, and the judgements below are written
# separately too, with their extracts copied from those abstracts by hand.
#
# What the model would answer is written as records in the formats the agents
# ask for. The stand-in in conftest.py builds each one when it is handed over,
# so a reply that stopped fitting the format stops the test there; that is a
# check on these fixtures, not a replay of what the real model call does with
# a badly formatted answer, which the sentinel at the end of this file stands
# for.

# The three invented papers. The abstract is what the extract check reads, so
# every extract below is a sentence taken from one of these word for word.
PAPERS = {
    "900000001": {
        "title": "A randomised trial of drug X in condition Y",
        "year": "2019",
        "journal": "Journal of Invented Trials",
        "author": ("One", "A"),
        "publication_type": "Randomized Controlled Trial",
        "abstract": ("Forty patients with condition Y were given drug X for eight "
                     "weeks. Symptoms improved in most of them."),
    },
    "900000002": {
        "title": "Drug X and condition Y: a narrative review",
        "year": "2020",
        "journal": "Journal of Invented Reviews",
        "author": ("Two", "B"),
        "publication_type": "Review",
        "abstract": ("This review summarises what is known about drug X in "
                     "condition Y. No new patients were studied."),
    },
    "900000003": {
        "title": "Adverse events of drug X in a small cohort",
        "year": "2021",
        "journal": "Journal of Invented Trials",
        "author": ("Three", "C"),
        "publication_type": "Journal Article",
        "abstract": ("Twelve patients taking drug X were followed for adverse "
                     "events. Two stopped treatment because of nausea."),
    },
}

# What each search text finds, in the order PubMed would list them. A search
# text mapped to a status is answered with that status instead of a list, so
# a test can plan a search that the service refuses. The identifier ending in
# 9 belongs to no paper above, which is how a search that finds more than the
# fetch returns is made (FT-03).
SEARCHES = {
    "drug X condition Y trial": ["900000001", "900000002", "900000003"],
    "drug X condition Y": [],
    "drug X randomised": ["900000001", "900000002", "900000003"],
    "drug X trial and cohort": ["900000001", "900000003", "900000009"],
    "drug X harms": "status 503",
}


def esearch_reply(term):
    found = SEARCHES.get(term, [])
    return {"esearchresult": {"count": str(len(found)), "idlist": list(found)}}


def _article_xml(pmid):
    paper = PAPERS[pmid]
    last_name, initials = paper["author"]
    return f"""  <PubmedArticle>
    <MedlineCitation>
      <PMID>{pmid}</PMID>
      <Article>
        <Journal><Title>{paper["journal"]}</Title>
          <JournalIssue><PubDate><Year>{paper["year"]}</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>{paper["title"]}</ArticleTitle>
        <Abstract><AbstractText>{paper["abstract"]}</AbstractText></Abstract>
        <AuthorList>
          <Author><LastName>{last_name}</LastName><Initials>{initials}</Initials></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType>{paper["publication_type"]}</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
"""


def efetch_reply(pmids):
    """The XML PubMed would send for the papers asked for, in the shape the
    source reads; an identifier no paper has is left out, as PubMed leaves out
    a record it does not hold."""
    articles = "".join(_article_xml(pmid) for pmid in pmids if pmid in PAPERS)
    return f'<?xml version="1.0"?>\n<PubmedArticleSet>\n{articles}</PubmedArticleSet>'


# What the model would answer, one record for each request the agents make.
# The names say which step each answers, and a test lists them in the order
# its run asks.

PLAN_TRIAL_AND_HARMS = {"subgoals": [
    {"purpose": "find trials of drug X in condition Y", "source": "pubmed",
     "query": "drug X condition Y trial"},
    {"purpose": "find reports of harm from drug X", "source": "pubmed",
     "query": "drug X harms"},
]}

PLAN_ONE_BROAD_SEARCH = {"subgoals": [
    {"purpose": "find studies of drug X in condition Y", "source": "pubmed",
     "query": "drug X condition Y"},
]}

PLAN_TRIAL_AND_COHORT = {"subgoals": [
    {"purpose": "find trials and cohorts of drug X", "source": "pubmed",
     "query": "drug X trial and cohort"},
]}

MOVE_CONTINUE = {"move": "continue", "reason": "the next planned search is still worth running"}
MOVE_FINISH = {"move": "finish", "reason": "enough has been found to screen"}
MOVE_REFINE = {"move": "refine",
               "reason": "the search returned nothing, so the wording is narrowed",
               "new_query": "drug X randomised"}

# Judgements for the three papers in the order the searches list them: the
# trial included, the review excluded, the cohort included with a lower score,
# so the ranked list puts the trial first and the review last (FR-11, FR-16).
SCREENING_THREE_PAPERS = {"screenings": [
    {"paper_number": 1, "decision": "include", "relevance": 9,
     "reason": "a randomised trial of drug X reporting an outcome in condition Y",
     "summary": "forty patients were given drug X and symptoms improved in most",
     "extract": "Symptoms improved in most of them"},
    {"paper_number": 2, "decision": "exclude", "relevance": 2,
     "reason": "a narrative review with no new patients",
     "summary": "a review of what is known, with no new patients studied",
     "extract": "No new patients were studied"},
    {"paper_number": 3, "decision": "include", "relevance": 6,
     "reason": "a cohort reporting adverse events of drug X",
     "summary": "twelve patients were followed and two stopped because of nausea",
     "extract": "Two stopped treatment because of nausea"},
]}

# The same judgements for a run that fetched the trial and the cohort only.
SCREENING_TRIAL_AND_COHORT = {"screenings": [
    SCREENING_THREE_PAPERS["screenings"][0],
    {**SCREENING_THREE_PAPERS["screenings"][2], "paper_number": 2},
]}

# A second opinion on the cohort paper after a reviewer sent it back: the
# model now excludes it, so the revised record needs a new decision (FR-19).
SCREENING_AGAIN_THE_COHORT = {"screenings": [
    {"paper_number": 1, "decision": "exclude", "relevance": 3,
     "reason": "on a second reading the cohort reports harms, not the outcome asked about",
     "summary": "twelve patients were followed for adverse events",
     "extract": "Two stopped treatment because of nausea"},
]}

# Stands for an answer that came back but did not fit the format asked for.
# The stand-in model turns it into the failure the real one returns, marked
# worth asking again, so the repairs Figure 2 allows are spent on it (DC-06).
AN_ANSWER_THAT_DOES_NOT_FIT = "an answer that does not fit the format"
