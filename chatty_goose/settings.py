from enum import Enum
from typing import Union
from pydantic import BaseSettings


__all__ = ["SearcherSettings", "HQESettings", "T5Settings"]


class PosFilter(str, Enum):
    NO = "no"
    POS = "pos"
    STP = "stp"


class SearcherSettings(BaseSettings):
    """Settings for Anserini searcher"""

    k1: float = 0.82  # BM25 k parameter
    b: float = 0.68  # BM25 b parameter
    rm3: bool = False  # use RM3
    fb_terms: int = 10  # RM3 number of expansion trees
    fb_docs: int = 10  # RM3 number of documents
    original_query_weight: float = 0.8  # RM3 weigh to assign initial query


class HQESettings(BaseSettings):
    """Settings for HQE"""

    M: int  # number of aggregate historical queries
    eta: float  # QPP threshold for first stage retrieval
    R_topic: float  # topic keyword threshold
    R_sub: float  # subtopic keyword threshold
    filter: PosFilter = PosFilter.POS  # 'no' or 'pos' or 'stp'


class T5Settings(BaseSettings):
    """Settings for T5 model"""

    model_name: str = "castorini/t5-base-canard"
    max_length: int = 64
    num_beams: int = 10
    early_stopping: bool = True
