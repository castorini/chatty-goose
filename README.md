# Chatty Goose

## Multi-stage Conversational Passage Retrieval: An Approach to Fusing Term Importance Estimation and Neural Query Rewriting

---

[![PyPI](https://img.shields.io/pypi/v/chatty-goose?color=brightgreen)](https://pypi.org/project/chatty-goose/)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

## Installation

1. Make sure Java 11+ and **Python 3.7+** are installed

2. Install the `chatty-goose` PyPI module

```
pip install chatty-goose
```

3. If you are using T5 or BERT, make sure to install [PyTorch 1.4.0 - 1.7.1](https://pytorch.org/) using your specific platform instructions. Note that PyTorch 1.8 is currently incompatible due to the `transformers` version we currently use. Also make sure to install the corresponding [torchtext](https://pypi.org/project/torchtext/) version.

4. Download the English model for spaCy

```
python -m spacy download en_core_web_sm
```

## Quickstart Guide

The following example shows how to initialize a searcher and build a `ConversationalQueryRewriter` agent from scratch using HQE and T5 as first-stage retrievers, and a BERT reranker. To see a working example agent, see [chatty_goose/agents/chat.py](chatty_goose/agents/chat.py).

First, load a searcher

```
from pyserini.search import SimpleSearcher

# Option 1: load a prebuilt index
searcher = SimpleSearcher.from_prebuilt_index("INDEX_NAME_HERE")
# Option 2: load a local Lucene index
searcher = SimpleSearcher("PATH_TO_INDEX")

searcher.set_bm25(0.82, 0.68)
```

Next, initialize one or more first-stage CQR retrievers

```
from chatty_goose.cqr import Hqe, Ntr
from chatty_goose.settings import HqeSettings, NtrSettings

hqe = Hqe(searcher, HqeSettings())
ntr = Ntr(NtrSettings())
```

Load a reranker

```
from chatty_goose.util import build_bert_reranker

reranker = build_bert_reranker()
```

Create a new `RetrievalPipeline`

```
from chatty_goose.pipeline import RetrievalPipeline

rp = RetrievalPipeline(searcher, [hqe, ntr], searcher_num_hits=50, reranker=reranker)
```

And we're done! Simply call `rp.retrieve(query)` to retrieve passages, or call `rp.reset_history()` to reset the conversational history of the retrievers.

## Running Experiments

1. Clone the repo and all submodules (`git submodule update --init --recursive`)

2. Clone and build [Anserini](https://github.com/castorini/anserini) for evaluation tools

3. Install dependencies

```
pip install -r requirements.txt
```

4. Follow the instructions under [docs/cqr_experiments.md](docs/cqr_experiments.md) to run experiments using HQE, T5, or fusion.

## Example Agent

To run an interactive conversational search agent with ParlAI, simply run [`chat.py`](chatty_goose/agents/chat.py). By default, we use the CAsT 2019 pre-built Pyserini index, but it is possible to specify other indexes using the `--from_prebuilt` flag. See the file for other possible arguments:

```
python -m chatty_goose.agents.chat
```

Alternatively, run the agent using ParlAI's command line interface:

```
python -m parlai interactive --model chatty_goose.agents.chat:ChattyGooseAgent
```

We also provide instructions to deploy the agent to Facebook Messenger using ParlAI under [`examples/messenger`](examples/messenger/README.md).
