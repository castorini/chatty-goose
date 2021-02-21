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

<!--
## Quickstart Guide

1. Load a new pretrained agent
```
TODO
```
-->

## Running Experiments

1. Clone the repo and all submodules (`git submodule update --init --recursive`)

2. Clone and build [Anserini](https://github.com/castorini/anserini) for evaluation tools

3. Install dependencies
```
pip install -r requirements.txt
```

4. Follow the appropriate instructions under `docs/` to run experiments using HQE, T5, or fusion.

<!--
## Data Preparation

We first deduplicate and generate CAR and Msmarco corpus with the format "Docid \t Document", and merge them.

```shell=bash
mkdir corpus
# Corpus dedupe
python data/trec-cast-tools/src/main/python/car_trecweb.py \
         $CARCorpus \
         ./corpus/
python data/trec-cast-tools/src/main/python/marco_trecweb.py \
         $MsMarcoCorpus \
         ./corpus/ \
         ./duplicate_file/msmarco_duplicate_list_v1.0.txt
# Corpus combine
cat $CARCorpus.xml $MsMarcoCorpus.xml > ./corpus/CAsT_collection.tsv
```

After that you can index CAsT_collection.tsv following the instruction in [Anserini: BM25 Baselines on MS MARCO Passage Retrieval](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md). To run conversational query reformulation using T5 or HQE, please see the readme in [cqr_t5](https://github.com/jacklin64/Conversational-IR/tree/master/cqr_t5) and [cqr_hqe](https://github.com/jacklin64/Conversational-IR/tree/master/cqr_hqe).
-->
