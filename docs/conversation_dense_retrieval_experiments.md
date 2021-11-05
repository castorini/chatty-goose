# Experiments for Conversational Dense Retrieval
This is the Chatty-goose reproduction of our paper: *[Contextualized Query Embeddings for Conversational Search](https://arxiv.org/abs/2104.08707)* Sheng-Chieh Lin, Jheng-Hong Yang and Jimmy Lin. Note that, due to quey latency concern, different from our orginal paper and [CQE github](https://github.com/castorini/CQE), in this repo, we use HSW dense index; thus, the result may be slightly different.
## Data Preparation

1. Download either the [training](https://github.com/daltonj/treccastweb/blob/master/2019/data/training/train_topics_v1.0.json) and [evaluation](https://github.com/daltonj/treccastweb/blob/master/2019/data/evaluation/evaluation_topics_v1.0.json) input query JSON files. These files can be found under `data/treccastweb/2019/data` if you cloned the submodules for this repo.

Pass your pathname to a variable
default:
```shell=bash
export input_query_json=data/treccastweb/2019/data
```

2. Download the evaluation answer files for [training](https://github.com/daltonj/treccastweb/blob/master/2019/data/training/train_topics_mod.qrel) or [evaluation](https://trec.nist.gov/data/cast/2019qrels.txt). The training answer file is found under `data/treccastweb/2019/data`.

## Run CQE retrieval

The following command is for CQE, but you can also run other CQR methods using `t5` or `cqe_t5_fusion` as the input to the `--experiment` flag (Currently, dense retrieval does not support HQE since it requires longer query sequence). Running the command for the first time will download the CAsT 2019 index (or whatever index is specified for the `--index` flag). It is also possible to supply a path to a local directory containing the index. Note that now the hsw dense index is stored at Orca:store/scratch/s269lin/cast/indexes. So now just input the dense_index as absolute path.
### CQE BM25 Retrieval

```shell=bash
python -m experiments.run_retrieval \
      --experiment cqe \
      --index cast2019 \
      --hits 1000 \
      --qid_queries $input_query_json \
      --output ./output/cqe_bm25 \
```
### CQE Dense Retrieval
```shell=bash
python -m experiments.run_retrieval \
      --experiment cqe \
      --dense_index=../cast/indexes \
      --hits 1000 \
      --qid_queries $input_query_json \
      --output ./output/cqe_dpr \
```
### CQE Sparse-Dense Hybrid Retrieval
```shell=bash
python -m experiments.run_retrieval \
      --experiment cqe \
      --dense_index=../cast/indexes \
      --index cast2019 \
      --hits 1000 \
      --qid_queries $input_query_json \
      --output ./output/cqe_hybrid \
      --cqe_l2_threshold 12 \
```
### CQE fuse T5 Sparse-Dense Hybrid Retrieval
```shell=bash
python -m experiments.run_retrieval \
      --experiment cqe_t5_fusion \
      --dense_index=../cast/indexes \
      --index cast2019 \
      --hits 1000 \
      --qid_queries $input_query_json \
      --output ./output/cqe_t5_hybrid \
      --cqe_l2_threshold 12 \
```

The experiment will output the retrieval results at the specified location in TSV format. For other command line arguments, see [run_retrieval.py](../experiments/run_retrieval.py).

## Evaluate CQR results

Convert the TSV file from above to TREC format and use the TREC tool to evaluate the resuls in terms of Recall@1000, mAP and NDCG@1,3.

```shell=bash
python $path_to_anserini/tools/scripts/msmarco/convert_msmarco_to_trec_run.py \
      --input ./output/cqe_t5_hybrid.tsv \
      --output ./output/cqe_t5_hybrid.trec

$path_to_anserini/tools/eval/trec_eval.9.0.4/trec_eval \
      -c -mrecall.1000 -mmap -mndcg_cut.1,3 \
      ./output/answer_file \
      ./output/cqe_t5_hybrid.trec
```

## Evaluation results

Results for the CAsT 2019 evaluation dataset are provided below. The results may be slightly different from the numbers reported in the paper due to implementation differences between Huggingface and SpaCy versions. As of writing, we use `spacy==2.2.4` with the English model `en_core_web_sm==2.2.5`, and `transformers==4.0.0`. Note that the Recall@1000 reported in [CQE paper]((https://arxiv.org/abs/2104.08707)) are using rel greater than 2 but in the repo, to be consistent with other previous experiments, we use rel greater than 1.

|             | CQE BM25 | CQE Dense Retrieval | CQE Hybrid | T5 BM25 | T5 Dense Retrieval | T5 Hybrid | CQE+T5 Fusion |
| ----------- | :------: | :-------------: | :-------------: | :-----: | :------------: | :---------: | :----------------: |
| mAP         |  0.2059  |     0.2616      |     0.2997      | 0.2250  |     0.2512     |   0.3043    |       0.3391       |
| Recall@1000 |  0.7705  |     0.7248      |     0.7984      | 0.7392  |     0.6734     |   0.7856    |       0.8376       |
| NDCG@1      |  0.3030  |     0.5082      |     0.4971      | 0.2842  |     0.4841     |   0.5077    |       0.5318       |
| NDCG@3      |  0.2740  |     0.4924      |     0.5032      | 0.2954  |     0.4688     |   0.5065    |       0.5226       |

## Reproduction Log

