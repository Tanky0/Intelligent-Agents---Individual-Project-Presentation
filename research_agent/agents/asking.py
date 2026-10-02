"""Asking the model for a record, and repairing an answer that is not one."""
# Both agents need the same thing: send a question, get back a record in the
# format asked for, and try again a bounded number of times when what comes back
# is not one. It lived in the Planning Agent while that was the only agent
# making requests. The Curation Agent needs the same rule, and a second copy
# would be a second repair limit that could drift from this one.
#
# Figure 2 allows up to three repairs, and they happen inside the agent, so a
# failure that reaches the graph has already used the attempts it was allowed
# (DC-06).

from research_agent import llm

# The model gets one first attempt followed by up to three repairs, as allowed
# by Figure 2 (DC-06). Keeping the number here makes the repair limit easy to
# find and change. It also matters for the free-tier request allowance.
REPAIRS_ALLOWED = 3


def ask_for_a_record(question, schema, budget):
    """Asks the model for a record in the required format and repairs it if needed."""
    # Returns the answer and the budget, in that order. The caller may receive
    # the requested record, a ModelFailure, an OutOfBudget, or None when
    # demonstration mode is on but no saved answer is available.
    #
    # Every attempt is counted, the first and each repair alike. Counting only
    # the first would make three repairs cost one request in the record and four
    # in the allowance, which is the wrong way round for a limit meant to keep
    # something back.
    #
    # A run that spends its last request part way through repairing gets the
    # OutOfBudget back from that attempt, and the repairs stop there. It is not a
    # failure to repair; there was nothing left to repair with.
    #
    # Only answers that fail the required format are repaired. Other failures are
    # returned immediately instead of using more model requests on a problem that
    # another attempt is unlikely to fix (OB-07).
    answer, budget = llm.ask_for(question, schema, budget)

    for repair in range(1, REPAIRS_ALLOWED + 1):
        if isinstance(answer, schema) or answer is None:
            return answer, budget
        if not answer.worth_asking_again:
            return answer, budget
        # Tell the model what was wrong with its previous answer so the
        # repair attempt can correct that problem instead of simply repeating
        # the same request.
        answer, budget = llm.ask_for(
            f"{question}\n\nThe previous answer could not be used because it "
            f"did not fit the format asked for. This is attempt {repair + 1}. "
            "Answer only in the format described above, with no other text.",
            schema, budget)

    return answer, budget
