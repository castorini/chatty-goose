# -*- coding: utf-8 -*-
'''
Anserini: A Lucene toolkit for replicable information retrieval research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import argparse
import time
import json
import numpy as np
import collections
from util import read_corpus, word_score, exact_key_word, query_expansion
# Pyjnius setup
import sys
sys.path += ['../anserini/src/main/python'] # We assume you clone anserini outside of the current folder
from pyjnius_setup import configure_classpath
configure_classpath('../anserini') # We assume you clone anserini outside of the current folder
from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HQE for CAsT.')
    parser.add_argument('--qid_queries', required=True, default='', help='query id - query mapping file')
    parser.add_argument('--output', required=True, default='', help='output file')
    parser.add_argument('--index', required=True, default='', help='index path')
    parser.add_argument('--hits', default=10, help='number of hits to retrieve')
    parser.add_argument('--corpus', help='input corpus with the format docid \t document')
    # See our MS MARCO documentation to understand how these parameter values were tuned.
    parser.add_argument('--k1', default=0.82, help='BM25 k1 parameter')
    parser.add_argument('--b', default=0.68, help='BM25 b parameter')
    parser.add_argument('--rm3', action='store_true', default=False, help='use RM3')
    parser.add_argument('--fbTerms', default=10, type=int, help='RM3 parameter: number of expansion terms')
    parser.add_argument('--fbDocs', default=10, type=int, help='RM3 parameter: number of documents')
    parser.add_argument('--originalQueryWeight', default=0.8, type=float, help='RM3 parameter: weight to assign to the original query')
    # HQE related hyperparameters. The default is tuned on CAsT train data
    parser.add_argument('--M0', default=5, type=int, help='aggregate historcial queries for first stage (BM25) retrieval')
    parser.add_argument('--M1', default=1, type=int, help='aggregate historcial queries for second stage (BERT) retrieval')
    parser.add_argument('--eta0', default=10, type=float, help='QPP threshold for first stage (BM25) retrieval')
    parser.add_argument('--eta1', default=12, type=float, help='QPP threshold for second stage (BERT) retrieval')
    parser.add_argument('--R0_topic', default=4.5, type=float, help='Topic keyword threshold for first stage (BM25) retrieval')
    parser.add_argument('--R1_topic', default=4, type=float, help='Topic keyword threshold for second stage (BERT) retrieval')
    parser.add_argument('--R0_sub', default=3.5, type=float, help='Subtopic keyword threshold for first stage (BM25) retrieval')
    parser.add_argument('--R1_sub', default=3, type=float, help='Subtopic keyword threshold for second stage (BERT) retrieval')
    parser.add_argument('--QR_method', default='hqe', help='input_query') #origin, concat, hqe
    parser.add_argument('--filter', default='pos', help='filter word method') #pos, no
    args = parser.parse_args()
    searcher = JSearcher(JString(args.index))
    searcher.setBM25Similarity(float(args.k1), float(args.b))
    print('Initializing BM25, setting k1={} and b={}'.format(args.k1, args.b))
    if args.rm3:
        searcher.setRM3Reranker(args.fbTerms, args.fbDocs, args.originalQueryWeight)
        print('Initializing RM3, setting fbTerms={}, fbDocs={} and originalQueryWeight={}'.format(args.fbTerms, args.fbDocs, args.originalQueryWeight))



    id_to_doc = read_corpus(args.corpus)
    with open(args.output+'.tsv', 'w') as fout0:
        with open(args.output+'.doc.tsv', 'w') as fout1:

            total_query_count = 0
            with open(args.qid_queries) as json_file:
                data = json.load(json_file)


            for session in data:
                session_num = str(session['number'])
                # Since we use different hyperparameters for first stage and second stage retrieval, we maintain two keyword(sub keyword) lists
                # For first stage
                key_word_list = collections.defaultdict(list)
                subkey_word_list = collections.defaultdict(list)
                # For second stage
                key_word_list1 = collections.defaultdict(list)
                subkey_word_list1 = collections.defaultdict(list)
                hist_query = []



                start_time = time.time()
                for turn_id, conversations in enumerate(session['turn']):



                    query = conversations['raw_utterance']
                    hist_query.append(query)

                    query_for_bert  = query
                    query_for_anserini = query
                    total_query_count+=1

                    conversation_num = str(conversations['number'])
                    qid=session_num+"_"+conversation_num



                    if (args.QR_method =='hqe'):
                        hits = searcher.search(JString(query.encode('utf8')), 1)

                        proc_query = word_score(query, searcher, JString)



                        # extract subtopic keyword
                        subkey_word_list = exact_key_word(proc_query, turn_id, args.R0_sub, args.R0_topic, subkey_word_list, args.filter)
                        subkey_word_list1 = exact_key_word(proc_query, turn_id, args.R1_sub, args.R1_topic, subkey_word_list1, args.filter)



                        # extract topic keyword
                        key_word_list = exact_key_word(proc_query, turn_id, args.R0_topic, 100000, key_word_list, args.filter)
                        key_word_list1 = exact_key_word(proc_query, turn_id, args.R1_topic, 100000, key_word_list1, args.filter)





                        key_word = query_expansion(key_word_list, 0, turn_id+1)
                        key_word1 = query_expansion(key_word_list1, 0, turn_id+1)

                        subkey_word = ''
                        if not ( (hits[0].score > args.eta0) ):
                            end_turn = turn_id+1
                            start_turn = end_turn - args.M0
                            if start_turn < 0:
                                start_turn = 0
                            subkey_word = query_expansion(subkey_word_list, start_turn, end_turn)

                        subkey_word1 = ''
                        if not ( (hits[0].score > args.eta1) ):
                            end_turn = turn_id+1
                            start_turn = end_turn - args.M1
                            if start_turn < 0:
                                start_turn = 0
                            subkey_word1 = query_expansion(subkey_word_list1, start_turn, end_turn)


                    # Generate Query for Anserini
                    if (args.QR_method =='hqe'):
                        if turn_id!=0:
                            query_for_anserini = key_word + ' ' + subkey_word + ' ' + query_for_anserini
                    elif args.QR_method =='origin':
                        query_for_anserini = query
                    elif args.QR_method=='concat':
                        hist_word=''
                        if args.filter=='no':
                            for i in range( min(args.M0,(len(hist_query)-1)) ):
                                hist_word = hist_word+ ' ' + hist_query[-i-2]
                        elif args.filter=='pos':
                            for i in range( min(args.M0,(len(hist_query)-1)) ):
                                proc_query = word_score(hist_query[-i-2], searcher, JString)
                                for idx, word in enumerate(proc_query['word']):
                                    if(('NN' in proc_query['pos'][idx]) or ('JJ' in proc_query['pos'][idx]) ):
                                        hist_word = hist_word+ ' ' + word
                        query_for_anserini= hist_word + ' ' + query_for_anserini


                    # Generate Query for Bert
                    if (args.QR_method =='hqe'):
                        if turn_id!=0:
                            query_for_bert = key_word1 + ' ' + subkey_word1 + ' ' + query_for_bert
                    elif args.QR_method =='origin':
                        query_for_bert = query
                    elif args.QR_method=='concat':
                        hist_word=''
                        if args.filter=='no':
                            for i in range( min(args.M1,(len(hist_query)-1)) ):
                                hist_word = hist_query[-i-2] + ' ' + hist_word
                            query_list=hist_word.split(' ')
                            word_num=len(query_list)
                            if (word_num>64): # BERT with default query maximum length 64
                                hist_word=''
                                for word in query_list[(word_num-64):(word_num)]:
                                    hist_word = hist_word+ ' '+word
                        elif args.filter=='pos':
                            for i in range( min(args.M1,(len(hist_query)-1)) ):
                                proc_query = word_score(hist_query[-i-2], searcher, JString)
                                for idx, word in enumerate(proc_query['word']):
                                    if(('NN' in proc_query['pos'][idx]) or ('JJ' in proc_query['pos'][idx]) ):
                                        hist_word = hist_word+ ' ' + word
                        query_for_bert= hist_word + ' ' + query_for_bert


                    hits = searcher.search(JString(query_for_anserini.encode('utf8')), int(args.hits))
                    for rank in range(len(hits)):
                        docno = hits[rank].docid
                        doc = id_to_doc[docno]


                        fout0.write('{}\t{}\t{}\n'.format(qid, docno, rank + 1))
                        fout1.write('{}\t{}\t{}\t{}\n'.format(qid, docno, query_for_bert, doc))


                time_per_query = (time.time() - start_time) / (turn_id + 1)
                print('Retrieving session {} with {} queries ({:0.3f} s/query)'.format(session['number'], turn_id+1, time_per_query))



    print('total Query Counts %d'%(total_query_count))
    print('Done!')



