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
from spacy.lang.en.stop_words import STOP_WORDS
# Pyjnius setup
import sys
sys.path += ['../anserini/src/main/python']
from pyjnius_setup import configure_classpath
configure_classpath('../anserini')

from jnius import autoclass
JString = autoclass('java.lang.String')
JSearcher = autoclass('io.anserini.search.SimpleSearcher')
import collections
from util import word_score, exact_key_word, list_to_query_expansion

# nlp = spacy.load('en', parser = False)
# def space_extend( matchobj):
#     return ' ' + matchobj.group(0) + ' '
# def pre_proc(text):
#     text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t', space_extend, text)
#     text = text.strip(' \n')
#     text = re.sub('\s+', ' ', text)
#     return text

# def _str(s):
#         """ Convert PTB tokens to normal tokens """
#         if (s.lower() == '-lrb-'):
#             s = '('
#         elif (s.lower() == '-rrb-'):
#             s = ')'
#         elif (s.lower() == '-lsb-'):
#             s = '['
#         elif (s.lower() == '-rsb-'):
#             s = ']'
#         elif (s.lower() == '-lcb-'):
#             s = '{'
#         elif (s.lower() == '-rcb-'):
#             s = '}'
#         return s

# def process(parsed_text):
#     output = {'word': [],
#               'lemma': [],
#               'pos': [],
#               'pos_id': [],
#               'ent': [],
#               'ent_id': [],
#               'offsets': [],
#               'sentences': []}

#     for token in parsed_text:
#         #[(token.text,token.idx) for token in parsed_sentence]
#         output['word'].append(_str(token.text))
#         pos = token.tag_
#         output['pos'].append(pos)
#         # output['pos_id'].append(token2id(pos, POS, 0))

#         ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
#         output['ent'].append(ent)
#         # output['ent_id'].append(token2id(ent, ENT, 0))

#         output['lemma'].append(token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower())
#         output['offsets'].append((token.idx, token.idx + len(token.text)))

#     word_idx = 0
#     for sent in parsed_text.sents:
#         output['sentences'].append((word_idx, word_idx + len(sent)))
#         word_idx += len(sent)

#     assert word_idx == len(output['word'])
#     return output

# def word_score(query, method='BM25'):
#     nlp_query = nlp(pre_proc(query))
#     proc_query = process(nlp_query)
#     query_words = proc_query['word']
#     proc_query['score'] = []

#     if (method=='BM25'):
#         for query_word in query_words:
#             hits = searcher.search(JString(query_word.encode('utf8')), int(args.hits))
#             try:
#                 score = hits[0].score
#                 proc_query['score'].append(score)
#                 #print(query_word+'\t'+str(score))
#             except:
#                 proc_query['score'].append(-1)
#     elif (method=='idf'):
#         for query_word in query_words:
#             try:
#                 score = searcher.idf(JString(query_word.encode('utf8')))
#                 proc_query['score'].append(score)
#                 #print(query_word+'\t'+str(score))
#             except:
#                 proc_query['score'].append(-1)

#     return proc_query




# def exact_key_word(proc_query, turn_id, low, up, key_word_list, filter):
#     if filter=='no':
#         for i, word in enumerate(proc_query['word']):
#             if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
#                 key_word_list[turn_id].append(word)
#     elif filter=='pos':
#         for i, word in enumerate(proc_query['word']):
#             if(('NN' in proc_query['pos'][i]) or ('JJ' in proc_query['pos'][i]) ):
#                 if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
#                     key_word_list[turn_id].append(word)
#     elif filter=='stp':
#         for i, word in enumerate(proc_query['word']):
#             if(word not in STOP_WORDS):
#                 if ( (proc_query['score'][i]>=(low))&(proc_query['score'][i]<up)   ):
#                     key_word_list[turn_id].append(word)
#     return key_word_list

# def list_to_query_expansion(key_word_list, start_turn, end_turn):
#     query_expansion = ''
#     for turn in range(start_turn, end_turn):
#         for word in key_word_list[turn]:
#             query_expansion= query_expansion+' ' + word
#     return query_expansion

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




    id_to_doc = {}
    with open(args.corpus, 'r') as f:
        print('Read Corpus...')
        for line in f:
            try:
                docid ,doc=line.strip().split("\t")
                id_to_doc[docid] = doc
            except:
                print("skip %s"%(line.strip()))


        # print('Read Retrieved Document')
        # qid_docid_retrieved = collections.defaultdict(list)
        # with open(args.retrieved_doc, 'r') as f:
        #     for line in f.readlines():
        #         data = line.strip().split(" ")
        #         qid = data[0]
        #         docid = data[2]
        #         qid_docid_retrieved[qid].append(docid)
    # query_bert_score ={}
    # with open('../large_bert_eval_score_ref.txt', 'r') as f:
    #     for line in f.readlines():
    #         qid, _, bert_score=line.strip().split('\t')
    #         query_bert_score[qid] = np.exp(float(bert_score))

    # id_to_query = {}
    # with open(args.input_query) as txt_file:
    #             for line in txt_file.readlines():
    #                 qid, query = line.split('\t')
    #                 query ,_ = query.split('\n')
    #                 id_to_query[qid] = query

    # hist_id_to_query = {}
    # with open('../eval_set/cast/hist_query_dictionary/id_to_query.txt') as txt_file:
    #             for line in txt_file.readlines():
    #                 qid, query = line.split('\t')
    #                 query ,_ = query.split('\n')
    #                 hist_id_to_query[qid] = query


    with open(args.output+'.tsv', 'w') as fout0:
        with open(args.output+'.doc.tsv', 'w') as fout1:

            total_query_count = 0
            with open(args.qid_queries) as json_file:
                data = json.load(json_file)

            qid_list = []
            hist_query_list = []
            # drop_POS_list = ['WP', 'WRB', 'DT', 'PRP', 'VBZ'] #VBZ


            scores = []
            for session in data:
                session_num = str(session['number'])





                key_word_list = collections.defaultdict(list)
                subkey_word_list = collections.defaultdict(list)

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





                        key_word = list_to_query_expansion(key_word_list,0, turn_id+1)
                        key_word1 = list_to_query_expansion(key_word_list1,0, turn_id+1)

                        subkey_word = ''
                        if not ( (hits[0].score > args.eta0) ):
                            end_turn = turn_id+1
                            start_turn = end_turn - args.M0
                            if start_turn < 0:
                                start_turn = 0
                            subkey_word = list_to_query_expansion(subkey_word_list, start_turn, end_turn)

                        subkey_word1 = ''
                        if not ( (hits[0].score > args.eta1) ):
                            end_turn = turn_id+1
                            start_turn = end_turn - args.M1
                            if start_turn < 0:
                                start_turn = 0
                            subkey_word1 = list_to_query_expansion(subkey_word_list1, start_turn, end_turn)



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
                            if (word_num>64):
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


                    # print(query_for_anserini)
                    hits = searcher.search(JString(query_for_anserini.encode('utf8')), int(args.hits))
                    # retrieve_scores[qid] = hits[0].score

                    qid_list.append(qid)
                    for rank in range(len(hits)):
                        docno = hits[rank].docid
                        doc = id_to_doc[docno]


                        # if args.retrieved_doc==None:
                        #     docno = hits[rank].docid

                        #     if docno[:3]=='CAR':
                        #         try:
                        #             split_doc=hits[rank].content.split('\n')
                        #             split_len=len(split_doc)
                        #             if (split_len!=3):
                        #                 doc = ''
                        #                 for i in range(1, (split_len-1)):
                        #                     doc += split_doc[i] + ' '
                        #                 # print('CAR: %d'%(len(hits[rank].content.split('\n'))))
                        #             else:

                        #                 doc = split_doc[1]
                        #         except:

                        #             try:
                        #                 doc = except_id_to_doc[docno]
                        #             except:
                        #                 print('Skip '+docno+' Passage...Rank @%d'%(rank))
                        #                 # docno = 'FAKE_id'
                        #                 doc = 'FAKE DOCUMENT'

                        #     else:
                        #         try:
                        #             split_doc=hits[rank].content.split('\n')
                        #             split_len=len(split_doc)
                        #             if (split_len!=5):
                        #                 doc = ''
                        #                 for i in range(2, (split_len-2)):
                        #                     doc += split_doc[i] + ' '
                        #                 # print('Others: %d'%(len(hits[rank].content.split('\n'))))
                        #             else:
                        #                 doc = split_doc[2]
                        #         except:
                        #             try:
                        #                 doc = except_id_to_doc[docno]
                        #             except:
                        #                 print('Skip ' + docno + ' Passage...Rank @%d'%(rank))
                        #                 # docno = 'FAKE_id'
                        #                 doc = 'FAKE DOCUMENT'

                        # else:
                        #     docno = qid_docid_retrieved[qid][rank]
                        #     doc = doc_dic[docno]

                        fout0.write('{}\t{}\t{}\n'.format(qid, docno, rank + 1))
                        fout1.write('{}\t{}\t{}\t{}\n'.format(qid, docno, query_for_bert, doc))


                time_per_query = (time.time() - start_time) / (turn_id + 1)
                print('Retrieving session {} with {} queries ({:0.3f} s/query)'.format(session['number'], turn_id+1, time_per_query))



    print('total Query Counts %d'%(total_query_count))
    print('Done!')



