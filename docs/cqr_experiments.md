# Experiments for Conversational Query Reformulation

## Data Preparation

1. Download the pre-built CAsT 2019 index using Pyserini. This will download the entire index to `~/.cache/pyserini`.

```
from pyserini.search import SimpleSearcher
SimpleSearcher.from_prebuilt_index('cast2019')
```

2. Download either the [training](https://github.com/daltonj/treccastweb/blob/master/2019/data/training/train_topics_v1.0.json) and [evaluation](https://github.com/daltonj/treccastweb/blob/master/2019/data/evaluation/evaluation_topics_v1.0.json) input query JSON files. These files can be found under `data/treccastweb/2019/data` if you cloned the submodules for this repo.

3. Download the evaluation answer files for [training](https://github.com/daltonj/treccastweb/blob/master/2019/data/training/train_topics_mod.qrel) or [evaluation](https://trec.nist.gov/data/cast/2019qrels.txt). The training answer file is found under `data/treccastweb/2019/data`.

## Run CQR retrieval

The following command is for HQE, but you can also run `experiments.retrieve_with_t5` and `experiments.retrieve_with_fusion` for the corresponding experiments.

```shell=bash
python -m experiments.retrieve_with_hqe \
      --hits 1000 \
      --index $anserini_index_path \
      --qid_queries $input_query_json \
      --output ./output/hqe_bm25 \
```

Running the experiment will output the retrieval results at the specified location in TSV format. By default, this will perform retrieval using only BM25, but you can add the `--rerank` flag to further rerank these results using BERT.

## Evaluate CQR results

Remove query-document pairs with zero relevance from the answer file.

```shell=bash
awk -F " " '{if ($4>0) print($1 " " $2 " " $3 " " $4)}' $answer_file_path > ./output/answer_file
```

Finally, we convert the TSV file to TREC format and use the TREC tool to evaluate the resuls in terms of Recall@1000, mAP and NDCG@1,3.

```shell=bash
python $path_to_anserini/tools/scripts/msmarco/convert_msmarco_to_trec_run.py \
      --input ./output/hqe_bm25.tsv \
      --output ./output/hqe_bm25.trec

$path_to_anserini/tools/eval/trec_eval.9.0.4/trec_eval \
      -c -mrecall.1000 -mmap -mndcg_cut.1,3 \
      ./output/answer_file \
      ./output/hqe_bm25.trec
```

## Evaluation results

The results of the eval set are slightly different from the numbers reported in the paper due to slight implementation differences from Huggingface and SpaCy versions.

| HQE + BM25  |  Eval  |
| ----------- | :----: |
| mAP         | 0.2105 |
| Recall@1000 | 0.7306 |
| NDCG@1      | 0.2640 |
| NDCG@3      | 0.2606 |

| T5 + BM25   |  Eval  |
| ----------- | :----: |
| mAP         | 0.2236 |
| Recall@1000 | 0.7383 |
| NDCG@1      | 0.2813 |
| NDCG@3      | 0.2922 |

| Fusion + BM25 |  Eval  |
| --------------| :----: |
| mAP           | 0.3454 |
| Recall@1000   | 0.7383 |
| NDCG@1        | 0.5284 |
| NDCG@3        | 0.5216 |
