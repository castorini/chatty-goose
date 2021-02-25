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

The following command is for HQE, but you can also run other CQR methods using `t5` or `fusion` instead of `hqe` as the input to the `--experiment` flag.

```shell=bash
python -m experiments.run_retrieval \
      --experiment hqe \
      --hits 1000 \
      --index $anserini_index_path \
      --qid_queries $input_query_json \
      --output ./output/hqe_bm25 \
```

Running the experiment will output the retrieval results at the specified location in TSV format. By default, this will perform retrieval using only BM25, but you can add the `--rerank` flag to further rerank these results using BERT. For other command line arguments, see [run_retrieval.py](experiments/run_retrieval.py).

## Evaluate CQR results

Convert the TSV file from above to TREC format and use the TREC tool to evaluate the resuls in terms of Recall@1000, mAP and NDCG@1,3.

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

Results for the CAsT 2019 evaluation dataset are provided below. The results may be slightly different from the numbers reported in the paper due to implementation differences between Huggingface and SpaCy versions.

|             | HQE BM25 | HQE BM25 + BERT | T5 BM25 | T5 BM25 + BERT | Fusion BM25 | Fusion BM25 + BERT |
| ----------- | :------: | :-------------: | :-----: | :------------: | :---------: | :----------------: |
| mAP         |  0.2109  |     0.3058      | 0.2250  |     0.3555     |   0.2575    |                    |
| Recall@1000 |  0.7322  |     0.7322      | 0.7392  |     0.7392     |   0.8028    |                    |
| NDCG@1      |  0.2640  |     0.4745      | 0.2842  |     0.5751     |   0.3353    |                    |
| NDCG@3      |  0.2606  |     0.4798      | 0.2954  |     0.5464     |   0.3236    |                    |
