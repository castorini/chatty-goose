
def reciprocal_rank_fusion(list1, list2, k=60):
    """
    Implements a reciprocal rank fusion as define in
    ``Reciprocal Rank fusion outperforms Condorcet and individual Rank Learning Methods`` by Cormack, Clarke and Buettcher.
    Parameters:
        k: term to avoid vanishing importance of lower-ranked documents. Default value is 60 (default value used in their paper).
        output: a file pointer to write the results. Sys.stdout is the default.
    """
    doc_scores = {}
    for pos, docid in enumerate(list1, start=1):
        doc_scores[docid] = doc_scores.get(docid, 0.0) + 1.0 / (k + pos)

    for pos, docid in enumerate(list2, start=1):
        doc_scores[docid] = doc_scores.get(docid, 0.0) + 1.0 / (k + pos)
    doc_scores= {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}

    return [*doc_scores]

