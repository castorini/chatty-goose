# Multi-stage Conversational Passage Retrieval: An Approach to Fusing Term Importance Estimation and Neural Query Rewriting

---

[![PyPI](https://img.shields.io/pypi/v/chatty-goose?color=brightgreen)](https://pypi.org/project/chatty-goose/)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

## Installation

1. Make sure Java 11+ and Python 3.6+ are installed

2. Install the `chatty-goose` PyPI module

```
pip install chatty-goose
```

3. If you are using T5 or BERT, make sure to install [PyTorch 1.5+](https://pytorch.org/) using your specific platform instructions

## Quickstart Guide

The following example shows how to initialize a searcher and build a CQR agent from scratch using HQE and T5 as first-stage retrievers, and a BERT reranker. To see a working example agent, see [chatty_goose/agents/cqragent.py](chatty_goose/agents/cqragent.py).

First, load a searcher

```
from pyserini.search import SimpleSearcher

# Option 1: load a prebuilt index
searcher = SimpleSearcher.from_prebuilt_index("INDEX_NAME_HERE")
# Option 2: load a local Lucene index
searcher = SimpleSearcher("PATH_TO_INDEX")

searcher.set_bm25(0.82, 0.68)
```

Next, initialize one or more first-stage retrievers

```
from chatty_goose.retrievers import HQE, T5_NTR
from chatty_goose.settings import HQESettings, T5Settings

hqe = HQE(searcher, HQESettings())
t5 = T5_NTR(T5Settings())
```

Load a reranker

```
from chatty_goose.util import build_bert_reranker

reranker = build_bert_reranker()
```

Create a new CQR object

```
from chatty_goose.cqr import CQR

cqr = CQR(searcher, [hqe, t5], searcher_num_hits=10, reranker=reranker)
```

And we're done! Simply call `cqr.retrieve(query)` to retrieve passages, or call `cqr.reset_history()` to reset the conversational history of the retrievers.

## Running Experiments

1. Clone the repo and all submodules (`git submodule update --init --recursive`)

2. Clone and build [Anserini](https://github.com/castorini/anserini) for evaluation tools

3. Install dependencies

```
pip install -r requirements.txt
```

4. Follow the instructions under [docs/cqr_experiments.md](docs/cqr_experiments.md) to run experiments using HQE, T5, or fusion.
