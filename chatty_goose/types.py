from enum import Enum


class PosFilter(str, Enum):
    NO = "no"
    POS = "pos"
    STP = "stp"

class CqrType(str, Enum):
    HQE = "hqe"
    T5 = "t5"
    FUSION = "fusion"
