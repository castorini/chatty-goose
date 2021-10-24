from enum import Enum


class PosFilter(str, Enum):
    NO = "no"
    POS = "pos"
    STP = "stp"

class CqrType(str, Enum):
    HQE = "hqe"
    CQE = "cqe"
    T5 = "t5"
    FUSION = "fusion"
    CQE_T5_FUSION = "cqe_t5_fusion"
