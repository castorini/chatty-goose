---
title: 'Conversational Question Reformaulation'
author: Matt Yang
---

T5 for Conversational Query Reformulation (CQR)
===

## Table of Contents

[TOC]

Overview
---
This is the runbook of fintuning T5 for the conversational query reformulation task for TREC-CAsT 2019 dataset.

Training data preparation
---
We first use a conversational query reformulation dataset from [CANARD](https://sites.google.com/view/qanta/projects/canard).
> [Dataset Paper (EMNLP'19)](http://users.umiacs.umd.edu/~jbg/docs/2019_emnlp_sequentialqa.pdf)

First, we use the dataset and preprocessing script released by the authors. Note that you have to install [SpaCy](https://spacy.io/) by yourself.
The following guide is based on a snapshot (hashkey: da8639) of the [CANARD github repo](https://github.com/aagohary/canard):
```shell=bash
git clone https://github.com/aagohary/canard
# run preprocessing script
cd canard
# modify & run preprocessing script
sed 's/seq2seq\/release/release/g' preprocess.sh > preprocess.mod.sh
bash preprocess.mod.sh

# prepare a tsv file for finetuning T5
paste data/seq2seq/train-src.txt data/seq2seq/train-tgt.tsv > history_query_pair.tsv
```
And we have the trainig data for T5 `history_query_pair.train.tsv`

Replicating T5 fintuning for CQR
---
Here we show how to use [Text-To-Text Transfer Transformer (T5)](https://github.com/google-research/text-to-text-transfer-transformer) model from its original github repo to finetune T5 as a CQR module. The following command will train a T5-base model for 4k iterations to predict queries from passages. We assume you put the tsv training file in `gs://your_bucket/data/history_query_pairs.train.tsv` (download from above). Also, change `your_tpu_name`, `your_tpu_zone`, `your_project_id`, and `your_bucket` accordingly.

```bash
t5_mesh_transformer  \
  --tpu="your_tpu_name" \
  --gcp_project="your_project_id" \
  --tpu_zone="your_tpu_zone" \
  --model_dir="gs://your_bucket/models/" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/data/history_query_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch = 131072"
```

Predicting Queries from History
---
To begin, follow [T5 repo](https://github.com/google-research/text-to-text-transfer-transformer) to install packages. The following guide will show you how to predict the reformulated queries with the trained T5 model and the data from CANARD.
```bash
pip install t5[gcp]

gsutil cp canard/data/seq2seq/test-src.txt gs://your_bucket/data/test-src.canard.txt
```

```bash
### BEAM search decoding
for BEAM in {1,5,10,15,20}; do
	t5_mesh_transformer \
		--tpu="your_tpu_name" \
		--gcp_project="your_project_id" \
		--tpu_zone="your_tpu_zone" \
		--model_dir="gs://your_buckets/models/" \
		--gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
		--gin_file="infer.gin" \
		--gin_file="beam_search.gin" \
		--gin_param="Bitransformer.decode.beam_size = ${BEAM}" \
		--gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
		--gin_param="infer_checkpoint_step = 1004000" \
		--gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
		--gin_param="Bitransformer.decode.max_decode_length = 64" \
		--gin_param="input_filename = 'gs://your_buckets/data/test-src.canard.txt'" \
		--gin_param="output_filename = 'gs://your_buckets/data/test_pred_beam_${BEAM}.txt'" \
		--gin_param="tokens_per_batch = 131072" \
		--gin_param="Bitransformer.decode.temperature = 0.0" \
		--gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1"
done
```

Preparing Inference File for CAsT
---
We here prepare a preprocessing script for you to transform CAsT queries into a CANARD compatible format:
```bash
python cast_preprocess.py <treccastweb/data> <data-split> <output-dir>
```
After transforming the queries, just use the same procedure but replace the `'inputfilename = ...'` with CAsT queries for CQR




## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::

###### tags: `Templates` `Documentation`

