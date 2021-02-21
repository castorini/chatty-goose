from typing import Dict, List, Tuple

from pyserini.search import JSimpleSearcherResult


def reciprocal_rank_fusion(*hit_lists: List[JSimpleSearcherResult], k=60):
    """
    Implements reciprocal rank fusion as defined in
    "Reciprocal Rank Fusion Outperforms Condorcet and individual Rank Learning Methods" by Cormack, Clarke and Buettcher.
    
    Parameters:
        hit_lists: lists of hits to merge using reciprocal rank fusion
        k: term to avoid vanishing importance of lower-ranked documents (default 60 from original paper)
    """
    if len(hit_lists) < 1:
        return []

    doc_scores: Dict[str, Tuple[float, JSimpleSearcherResult]] = {}
    for hits in hit_lists:
        for pos, hit in enumerate(hits, start=1):
            cur_rank = doc_scores.get(hit.docid, (0.0, hit))[0]
            doc_scores[hit.docid] = (cur_rank + 1.0 / (k + pos), hit)

    # Sort by highest score
    result = [
        score_hit[1]
        for _, score_hit in sorted(iter(doc_scores.items()), key=lambda kv: -kv[1][0])
    ]
    return result
