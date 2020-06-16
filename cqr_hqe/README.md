HQE for Conversational Query Reformulation
===

Requirement
---
You have to install [Anserini](https://github.com/castorini/anserini) and [SpaCy](https://spacy.io/) and follow the guideline on [TREC web](https://github.com/daltonj/treccastweb) downloading the corpus and data.

Data Preparation
---
We generate corpus with the format "Docid \t Document" and then index the corpus using [Anserini](https://github.com/castorini/anserini). With corpus.tsv and index file, we can now run HQE. The input query file ([train](https://github.com/daltonj/treccastweb/blob/master/2019/data/training/train_topics_v1.0.json) and [evaluation](https://github.com/daltonj/treccastweb/blob/master/2019/data/evaluation/evaluation_topics_v1.0.json)) for HQE is the json format.

Run HQE using Anserini
---
```shell=bash
python3 ./retrieve_with_hqe.py \
      --hits 1000 \
      --index $index \
      --qid_queries $input_query \
      --output ./hqe_bm25 \
      --corpus $corpus
```
