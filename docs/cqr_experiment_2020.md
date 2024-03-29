# Experiments for CQR with CAsT 2020

## Data Preparation

1. Download [evaluation](https://github.com/daltonj/treccastweb/blob/master/2020/2020_manual_evaluation_topics_v1.0.json) input query JSON file. This file can be found under `data/treccastweb/2020` if you cloned the submodules for this repo.

2. Download the evaluation answer files for [evaluation](https://trec.nist.gov/data/cast/2020qrels.txt).

## Run and evaluate CQR retrieval

The run is similar to the run for the [CQR experiment for CAsT2019](./cqr_experiments.md#run-cqr-retrieval). For canonical runs, you also need to specify an extra `--context_index` flag to define the index from which the canonical passage is retrieved from. `--add_response` controls how many previous response you want to add to the context; 0 represents using historical query only.


The index `cast2019` can still be used to perform bm25 search since `cast2019` and `cast2020` share the same corpus. 

In the naive run, only the `raw_utterance` is used. In the canonical run, the passage corresponding to `manual_canonical_result_id` is also used in the context. 

Results for the CAsT 2020 evaluation dataset are provided below for both naive and canonical runs. As of writing, we use `spacy==2.2.4` with the English model `en_core_web_sm==2.2.5`, and `transformers==4.0.0`.

### Historical Query only

```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe or t5 or hqe_t5_fusion \
      --hits 1000 \
      --sparse_index cast2019 \
      --qid_queries $input_query_json \
      --output ./output/result \

python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1 -mrecall.1000 -mmap $qrel ./output/result.trec
```

|             | HQE BM25 |    T5 BM25      | Fusion BM25 |
| ----------- | :------: | :-------------: | :---------: |
| mAP         |  0.1155  |     0.1236      |   0.1386    |
| Recall@1000 |  0.5316  |     0.5551      |   0.6063    |
| NDCG@1      |  0.1635  |     0.1639      |   0.2015    |
| NDCG@3      |  0.1640  |     0.1620      |   0.1879    |

---------

### One Canonical Response

```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe or t5 or hqe_t5_fusion \
      --hits 1000 \
      --sparse_index cast2019 \
      --qid_queries $input_query_json \
      --output ./output/result \
      --add_response 1 \

python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1 -mrecall.1000 -mmap $qrel ./output/result.trec
```

|             | HQE BM25 |    T5 BM25      | Fusion BM25 |
| ----------- | :------: | :-------------: | :---------: |
| mAP         |  0.1061  |     0.1271      |   0.1478    |
| Recall@1000 |  0.5887  |     0.5625      |   0.6594    |
| NDCG@1      |  0.1306  |     0.1807      |   0.1959    |
| NDCG@3      |  0.1244  |     0.1652      |   0.1925    |

---------

### Two Canonical Response

```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe or t5 or hqe_t5_fusion \
      --hits 1000 \
      --sparse_index cast2019 \
      --qid_queries $input_query_json \
      --output ./output/result \
      --add_response 2 \

python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1 -mrecall.1000 -mmap $qrel ./output/result.trec
```

|             | HQE BM25 |   T5 BM25      |   Fusion BM25  |
| ----------- | :------: | :------------: | :------------: |
| mAP         |  0.0948  |    0.1319      |     0.1446     |
| Recall@1000 |  0.5755  |    0.5747      |     0.6618     |
| NDCG@1      |  0.1306  |    0.1955      |     0.2011     |
| NDCG@3      |  0.1185  |    0.1785      |     0.1967     |

---------

### BERR Reranking
```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe_t5_fusion \
      --hits 1000 \
      --sparse_index cast2019 \
      --qid_queries $input_query_json \
      --output ./output/result \
      --add_response 2 \
      --rerank

python -m pyserini.eval.trec_eval -c -mndcg_cut.3,1 -mrecall.1000 -mmap $qrel ./output/result.trec
```

|             |   Fusion BM25 Rerank  |
| ----------- | :------------: |
| mAP         |     0.2861     |
| Recall@1000 |     0.6618     |
| NDCG@1      |     0.4623     |
| NDCG@3      |     0.4202     |

This [link](https://colab.research.google.com/drive/1KBm-BJAy9Yhb5b7NMuuW4v_gj8KJ0VAv?usp=sharing) is the Colab demo for HQE-T5 early fusion with BERT reranking.
## Reproduction Log


