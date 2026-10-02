# Notice: what the licence covers

The MIT licence in `LICENSE` covers material created for this project,
including:

* the code in `research_agent/`, `scripts/` and `tests/`;
* the project documentation in `docs/` and the README;
* the fictional papers and responses used in the demonstration and
  functional tests; and
* the hand-screened labels in `data/hand_screened_sample/`.

The hand-screened files record this project's own inclusion decisions and
rationales about published papers. They contain no text reproduced from those
papers.

The MIT licence does not cover third-party material used by or retrieved
through the project:

* CLEF 2019 test data. The fifty identifiers and labels in
  `data/clef_tar/cd011571_fifty.csv` were selected from the CLEF 2019
  Technology-Assisted Review collection, published at
  `github.com/CLEF-TAR/tar` under its own MIT licence (copyright 2017 Leif
  Azzopardi). The file remains subject to the source collection's licence.
* Records returned by external services. Data from PubMed, Semantic Scholar
  and Crossref remains subject to those services' terms, as described in the
  README. Saved responses used for replay are kept private and are excluded
  from the public export.
* Published paper text. Titles, abstracts and extracts remain the work of
  their authors and publishers. The project may retain a short extract
  alongside a paper identifier to make a claim traceable to its source. Full
  abstracts used during a run remain in the private run buffer and are not
  published with the repository.
* Language-model responses. Gemini responses are used under Google's
  applicable terms; this project does not license them.
* National Library of Medicine notices. The NLM disclaimer and copyright
  notice displayed in the interface are reproduced as required by NCBI and
  remain subject to their original terms.

The project's own code and documentation may be reused under the MIT licence.
Third-party data, paper content and service output must be obtained and used
under the terms of their original sources.
