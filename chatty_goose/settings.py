from pydantic import BaseSettings
import typing
from chatty_goose.types import PosFilter

__all__ = ["SearcherSettings", "HqeSettings", "NtrSettings", "CqeSettings"]


class SearcherSettings(BaseSettings):
    """Settings for Anserini searcher"""

    index_path: typing.Any  # Pre-built index name or path to Lucene index
    k1: float = 0.82  # BM25 k parameter
    b: float = 0.68  # BM25 b parameter
    rm3: bool = False  # use RM3
    fb_terms: int = 10  # RM3 number of expansion trees
    fb_docs: int = 10  # RM3 number of documents
    original_query_weight: float = 0.8  # RM3 weigh to assign initial query

class DenseSearcherSettings(BaseSettings):
    """Settings for Pyserini dsearcher"""

    index_path: typing.Any # Pre-built index name or path to faiss index
    query_encoder: str # path to huggingface model or hub

class CqrSettings(BaseSettings):
    verbose: bool = False


class HqeSettings(CqrSettings):
    """Settings for HQE with defaults tuned on CAsT"""

    M: int = 5  # number of aggregate historical queries
    eta: float = 10.0  # QPP threshold for first stage retrieval
    R_topic: float = 4.5  # topic keyword threshold
    R_sub: float = 3.5  # subtopic keyword threshold
    filter: PosFilter = PosFilter.POS  # 'no' or 'pos' or 'stp'

class CqeSettings(CqrSettings):
    """Settings for CQE model for NTR"""
    l2_threshold: float = 10.5
    model_name: str = "castorini/cqe"
    max_context_length: int = 100
    max_query_length: int = 36


class NtrSettings(CqrSettings):
    """Settings for T5 model for NTR"""

    model_name: str = "castorini/t5-base-canard"
    max_length: int = 64
    num_beams: int = 10
    early_stopping: bool = True
