import sys 
sys.path.append('../pyserini') # Here we import pyserini from our local, since the current version pyserini did not support dense retrieval with embedding as input (See line 67)
import logging
import json
from typing import List, Optional, Union
from pyserini.hsearch import HybridSearcher
from pyserini.dsearch import SimpleDenseSearcher, DenseSearchResult
from chatty_goose.cqr import ConversationalQueryRewriter
from chatty_goose.util import reciprocal_rank_fusion
from pygaggle.rerank.base import Query, Reranker, hits_to_texts
from pyserini.search import JSimpleSearcherResult, SimpleSearcher


__all__ = ["RetrievalPipeline"]


class RetrievalPipeline:
    """
    End-to-end conversational passage retrieval pipeline

    Parameters:
        searcher (SimpleSearcher): Pyserini searcher for Lucene index
        reformulators (List[ConversationalQueryRewriter]): List of CQR methods to use for first-stage retrieval
        searcher_num_hits (int): number of hits returned by searcher - default 10
        early_fusion (bool): flag to perform fusion before second-stage retrieval - default True
        reranker (Reranker): optional reranker for second-stage retrieval
        reranker_query_index (int): retriever index to use for reranking query - defaults to last retriever
        reranker_query_reformulator (ConversationalQueryRewriter): CQR method for generating reranker query,
                                                                   overrides reranker_query_index if provided
    """

    def __init__(
        self,
        searcher: SimpleSearcher,
        dense_searcher: Optional[SimpleSearcher],
        reformulators: List[ConversationalQueryRewriter],
        searcher_num_hits: int = 10,
        early_fusion: bool = True,
        reranker: Reranker = None,
        reranker_query_index: int = -1,
        reranker_query_reformulator: ConversationalQueryRewriter = None,
        context_searcher: Optional[SimpleSearcher] = None,
    ):
        self.searcher = searcher
        self.dense_searcher = dense_searcher
        self.reformulators = reformulators
        self.searcher_num_hits = int(searcher_num_hits)
        self.early_fusion = early_fusion
        self.reranker = reranker
        self.reranker_query_index = reranker_query_index
        self.reranker_query_reformulator = reranker_query_reformulator
        self.context_searcher = context_searcher

    def retrieve(self, query, context: Optional[str] = None) -> List[JSimpleSearcherResult]:
        cqr_hits = []
        cqr_queries = []
        for cqr in self.reformulators:
            sparse_hits, dense_hits = None, None
            new_query = cqr.rewrite(query, context)
            # Sparse search
            if self.searcher!=None:
                sparse_hits = self.searcher.search(new_query, k=self.searcher_num_hits)
            # Dense search
            if self.dense_searcher!=None:
                if cqr.name=='Cqe': #CQE embedding is generated during query rewritting, so here we directly input CQE embeddings for dense retrieval
                    dense_hits = self.dense_searcher.search(cqr.query_embs, k=self.searcher_num_hits)
                else:
                    dense_hits = self.dense_searcher.search(new_query, k=self.searcher_num_hits)
                    
            hits = self._hybrid_results(dense_hits, sparse_hits, 0.1, self.searcher_num_hits)
            cqr_hits.append(hits)
            cqr_queries.append(new_query)
        

        # Merge results from multiple CQR methods if required
        if self.early_fusion or self.reranker is None:
            cqr_hits = reciprocal_rank_fusion(cqr_hits)

        # Return results if no reranker
        if self.reranker is None:
            return cqr_hits

        # Get query for reranker
        if self.reranker_query_reformulator is None:
            rerank_query = cqr_queries[self.reranker_query_index]
        else:
            rerank_query = self.reranker_query_reformulator.rewrite(query)

        # Rerank results
        if self.early_fusion:
            results = self.rerank(
                rerank_query, cqr_hits[:self.searcher_num_hits])
        else:
            # Rerank all CQR results and fuse together
            results = []
            for hits in cqr_hits:
                results = self.rerank(rerank_query, hits)
            results = reciprocal_rank_fusion(results)
        return results

    def rerank(self, query, hits):
        if self.reranker is None:
            logging.info("Reranker not available, skipping reranking")
            return hits

        reranked = self.reranker.rerank(Query(query), hits_to_texts(hits))
        reranked_scores = [r.score for r in reranked]

        # Reorder hits with reranker scores
        reranked = list(zip(hits, reranked_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        reranked_hits = [r[0] for r in reranked]
        return reranked_hits

    def reset_history(self):
        for cqr in self.reformulators:
            cqr.reset_history()

        if self.reranker_query_reformulator:
            self.reranker_query_reformulator.reset_history()

    def get_context(self, docid: Union[str, int]) -> Optional[str]:
        if not self.context_searcher:
            return None
        doc = self.context_searcher.doc(docid)
        if doc is not None:
            return json.loads(doc.raw())['contents']
        return None
    # Directly copy from pyserini (https://github.com/castorini/pyserini/blob/master/pyserini/hsearch/_hybrid.py)
    @staticmethod 
    def _hybrid_results(dense_results, sparse_results, alpha, k, normalization=False, weight_on_dense=False):
        if (dense_results==None) or (sparse_results==None):
            if dense_results==None:
                return sparse_results[:k]
            else:
                return dense_results[:k]
        dense_hits = {hit.docid: hit.score for hit in dense_results}
        sparse_hits = {hit.docid: hit.score for hit in sparse_results}
        hybrid_result = []
        min_dense_score = min(dense_hits.values()) if len(dense_hits) > 0 else 0
        max_dense_score = max(dense_hits.values()) if len(dense_hits) > 0 else 1
        min_sparse_score = min(sparse_hits.values()) if len(sparse_hits) > 0 else 0
        max_sparse_score = max(sparse_hits.values()) if len(sparse_hits) > 0 else 1
        for doc in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if doc not in dense_hits:
                sparse_score = sparse_hits[doc]
                dense_score = min_dense_score
            elif doc not in sparse_hits:
                sparse_score = min_sparse_score
                dense_score = dense_hits[doc]
            else:
                sparse_score = sparse_hits[doc]
                dense_score = dense_hits[doc]
            if normalization:
                sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
                               / (max_sparse_score - min_sparse_score)
                dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
                              / (max_dense_score - min_dense_score)
            score = alpha * sparse_score + dense_score if not weight_on_dense else sparse_score + alpha * dense_score
            hybrid_result.append(DenseSearchResult(doc, score))
        return sorted(hybrid_result, key=lambda x: x.score, reverse=True)[:k]
