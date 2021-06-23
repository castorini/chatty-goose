# Experiments for CQR with CAsT 2020

## Data Preparation

1. Download [evaluation](https://github.com/daltonj/treccastweb/blob/master/2020/2020_manual_evaluation_topics_v1.0.json) input query JSON file. This file can be found under `data/treccastweb/2020` if you cloned the submodules for this repo.

2. Download the evaluation answer files for [evaluation](https://trec.nist.gov/data/cast/2020qrels.txt).

## Run CQR retrieval

The run is similar to the run for the [CQR experiment for CAsT2019](./cqr_experiments.md#run-cqr-retrieval). For canonical runs, you also need to specify an extra `--context_index` flag to define the index from which the canonical passage is retrieved from.

```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe \
      --context_index msmarco-passage \
      --hits 1000 \
      --index cast2019 \
      --qid_queries $input_query_json \
      --output ./output/hqe_bm25 \
```

The index `cast2019` can still be used to perform bm25 search since `cast2019` and `cast2020` share the same corpus. 

In the naive run, only the `raw_utterance` is used. In the canonical run, the passage corresponding to `manual_canonical_result_id` is also used in the context. 

## Evaluate CQR results

Evaluation is done in same way as described in [CQR experiment for CAsT2019](./cqr_experiments.md#evaluate-cqr-results).

## Evaluation results

Results for the CAsT 2020 evaluation dataset are provided below for both naive and canonical runs. As of writing, we use `spacy==2.2.4` with the English model `en_core_web_sm==2.2.5`, and `transformers==4.0.0`.

### Naive

|             | HQE BM25 |    T5 BM25      | Fusion BM25 |
| ----------- | :------: | :-------------: | :---------: |
| mAP         |  0.1155  |     0.1236      |   0.1385    |
| Recall@1000 |  0.5316  |     0.5551      |   0.6063    |
| NDCG@1      |  0.1635  |     0.1639      |   0.2027    |
| NDCG@3      |  0.1640  |     0.1620      |   0.1880    |

---------

### Canonical

|             | HQE BM25 | HQE BM25 + BERT |   T5 BM25      | T5 BM25 + BERT |   Fusion BM25  | Fusion BM25 + BERT |
| ----------- | :------: | :-------------: | :------------: | :------------: | :------------: | :----------------: |
| mAP         |  0.0890  |    0.2009       |    0.1305      |    0.2565      |     0.1356     |      0.2565        |
| Recall@1000 |  0.5083  |    0.5083       |    0.5461      |    0.5461      |     0.6235     |      0.5461        |
| NDCG@1      |  0.1482  |    0.3417       |    0.1855      |    0.4151      |     0.2388     |      0.4151        | 
| NDCG@3      |  0.1337  |    0.3129       |    0.1817      |    0.3970      |     0.2102     |      0.3970        | 


## Reproduction Log

+ Results reproduced by [@saileshnankani](https://github.com/saileshnankani) on 2021-06-22 (commit [`52323c1`](https://github.com/castorini/chatty-goose/commit/52323c1f4c15fae207c219feeb3086cc823e595b)) (Non-Bert - Tuna and BERT - Colab with Tesla P100-PCIE-16GB)
