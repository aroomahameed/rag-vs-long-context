def recall_at_k(retrieved, gold, k=5):
    """1.0 if any gold file is in the top k retrieved sources, else 0.0."""
    if not gold:
        return None  # unanswerable questions have no gold file
    return float(any(s in gold for s in retrieved[:k]))


def mrr(retrieved, gold):
    """1 / rank of the first gold file (1.0 = it was first)."""
    if not gold:
        return None
    for rank, s in enumerate(retrieved, 1):
        if s in gold:
            return 1 / rank
    return 0.0