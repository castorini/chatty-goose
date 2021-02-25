from pydantic import BaseSettings

from chatty_goose.types import PosFilter

__all__ = ["SearcherSettings", "HQESettings", "T5Settings"]


class SearcherSettings(BaseSettings):
    """Settings for Anserini searcher"""

    index_path: str  # Lucene index path
    k1: float = 0.82  # BM25 k parameter
    b: float = 0.68  # BM25 b parameter
    rm3: bool = False  # use RM3
    fb_terms: int = 10  # RM3 number of expansion trees
    fb_docs: int = 10  # RM3 number of documents
    original_query_weight: float = 0.8  # RM3 weigh to assign initial query


class CQRSettings(BaseSettings):
    verbose: bool = False


class HQESettings(CQRSettings):
    """Settings for HQE with defaults tuned on CAsT"""

    M: int = 5  # number of aggregate historical queries
    eta: float = 10.0  # QPP threshold for first stage retrieval
    R_topic: float = 4.5  # topic keyword threshold
    R_sub: float = 3.5  # subtopic keyword threshold
    filter: PosFilter = PosFilter.POS  # 'no' or 'pos' or 'stp'


class T5Settings(CQRSettings):
    """Settings for T5 model for NTR"""

    model_name: str = "castorini/t5-base-canard"
    max_length: int = 64
    num_beams: int = 10
    early_stopping: bool = True
