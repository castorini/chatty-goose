import sys 
# sys.path.append('./pyserini') 
import logging
from os import path
from typing import Dict, List, Tuple

from pygaggle.rerank.transformer import MonoBERT
from pyserini.search import JSimpleSearcherResult, SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher
from chatty_goose.settings import SearcherSettings


def reciprocal_rank_fusion(
    hit_lists: List[List[JSimpleSearcherResult]], k: int = 60
) -> List[JSimpleSearcherResult]:
    """
    Implements reciprocal rank fusion as defined in
    "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" by Cormack, Clarke and Buettcher.
    
    Parameters:
        hit_lists: lists of hits to merge using reciprocal rank fusion
        k: term to avoid vanishing importance of lower-ranked documents (default 60 from original paper)
    """
    if len(hit_lists) == 0:
        return []

    if len(hit_lists) == 1:
        return hit_lists[0]

    doc_scores: Dict[str, Tuple[float, JSimpleSearcherResult]] = {}
    for hits in hit_lists:
        for pos, hit in enumerate(hits, start=1):
            cur_rank = doc_scores.get(hit.docid, (0.0, hit))[0]
            doc_scores[hit.docid] = (cur_rank + 1.0 / (k + pos), hit)

    # Sort by highest score
    result = []
    for _, score_hit in sorted(iter(doc_scores.items()), key=lambda kv: -kv[1][0]):
        score_hit[1].score = score_hit[0]  #update score with rrf fusion score 
        result.append(score_hit[1])
    return result


def build_bert_reranker(
    name_or_path: str = "castorini/monobert-large-msmarco-finetune-only",
    device: str = None,
):
    """Returns a BERT reranker using the provided model name or path to load from"""
    model = MonoBERT.get_model(name_or_path, device=device)
    tokenizer = MonoBERT.get_tokenizer(name_or_path)
    return MonoBERT(model, tokenizer)


def build_searcher(settings: SearcherSettings) -> SimpleSearcher:
    if settings.index_path==None:
        logging.info(
            "Cannot find bm25 index, skip bm25 search!")
        searcher=None
    else: 
        if path.isdir(settings.index_path):
            searcher = SimpleSearcher(settings.index_path)
        else:
            searcher = SimpleSearcher.from_prebuilt_index(settings.index_path)
        searcher.set_bm25(float(settings.k1), float(settings.b))
        logging.info(
            "Initializing BM25, setting k1={} and b={}".format(settings.k1, settings.b)
        )
        if settings.rm3:
            searcher.set_rm3(
                settings.fb_terms, settings.fb_docs, settings.original_query_weight
            )
            logging.info(
                "Initializing RM3, setting fbTerms={}, fbDocs={} and originalQueryWeight={}".format(
                    settings.fb_terms, settings.fb_docs, settings.original_query_weight
                )
            )
    return searcher

def build_dense_searcher(settings: SearcherSettings) -> SimpleDenseSearcher:
    if (settings.index_path==None) or (settings.query_encoder==None):
        logging.info(
            "Cannot find dense index or query encoder, skip dense search!")
        searcher = None
    else:
        logging.info(
            "Load dense index: {}".format(settings.index_path))
        if path.isdir(settings.index_path):
            searcher = SimpleDenseSearcher(settings.index_path, settings.query_encoder)
        else:
            searcher = SimpleDenseSearcher.from_prebuilt_index(settings.index_path, settings.query_encoder)
        # logging.info(
        #     "Load, setting k1={} and b={}".format(settings.k1, settings.b)
        # )
    return searcher
