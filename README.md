# Query Reformulation using Query History for Passage Retrieval in Conversational Search
Requirement
---
You have to install [Anserini](https://github.com/castorini/anserini) and [SpaCy](https://spacy.io/) and follow the guideline on [TREC web](https://github.com/daltonj/treccastweb) downloading the corpus and data.

Data Preparation
---
We first deduplicate and generate CAR and Msmarco corpus with the format "Docid \t Document", and merge them.

```shell=bash
mkdir corpus
# Corpus dedupe
python trec-cast-tools/src/main/python/car_trecweb.py \
         $CARCorpus \
         ./corpus/
python trec-cast-tools/src/main/python/marco_trecweb.py \
         $MsMarcoCorpus \
         ./corpus/ \
         ./duplicate_file/msmarco_duplicate_list_v1.0.txt
# Corpus combine
cat $CARCorpus.xml $MsMarcoCorpus.xml > ./corpus/CAsT_collection.tsv
```
After that you can index CAsT_collection.tsv following the instruction in [Anserini: BM25 Baselines on MS MARCO Passage Retrieval](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md). To run conversational query reformulation using T5 or HQE, please see the readme in [cqr_t5](https://github.com/jacklin64/Conversational-IR/tree/master/cqr_t5) and [cqr_hqe](https://github.com/jacklin64/Conversational-IR/tree/master/cqr_hqe).