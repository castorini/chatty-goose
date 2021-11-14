# Now we import pyserini directly from folder since input embeddings to SimpleDenseSearcher.search are not upgraded
import sys 
# sys.path.append('../pyserini') 
import argparse
import json
import time

from chatty_goose.cqr import Hqe, Ntr, Cqe
from chatty_goose.pipeline import RetrievalPipeline
from chatty_goose.settings import SearcherSettings, DenseSearcherSettings, HqeSettings, NtrSettings, CqeSettings
from chatty_goose.types import CqrType, PosFilter
from chatty_goose.util import build_bert_reranker, build_searcher, build_dense_searcher

from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher

def parse_experiment_args():
    parser = argparse.ArgumentParser(description='CQR experiments for CAsT 2019.')
    parser.add_argument('--experiment', type=str, help='Type of experiment (cqe, hqe, t5, hqe_t5_fusion, cqe_t5_fusion)')
    parser.add_argument('--qid_queries', required=True, default='', help='query id - query mapping file')
    parser.add_argument('--output', required=True, default='', help='output file')
    parser.add_argument('--sparse_index', default=None, help='bm25 index path')
    parser.add_argument('--dense_index', default=None, help='dense index path')
    parser.add_argument('--context_index', default='cast2019', help='index for searching context text')
    parser.add_argument('--query_encoder', default='castorini/tct_colbert-v2-msmarco', help='query encoder model path')
    parser.add_argument('--hits', default=10, help='number of hits to retrieve')
    parser.add_argument('--rerank', action='store_true', help='rerank BM25 output using BERT')
    parser.add_argument('--reranker_device', default='cuda', help='reranker device to use')
    parser.add_argument('--late_fusion', action='store_true', help='perform late instead of early fusion')
    parser.add_argument('--verbose', action='store_true', help='verbose log output')
    parser.add_argument('--context_field', default='manual_canonical_result_id', help='doc id for additional context')
    parser.add_argument('--add_response', type=int, default=0, help='How many response to add in context')
    parser.add_argument('--run_name', type=str, default=None, help='run file name printed in trec file')


    # Parameters for BM25. See Anserini MS MARCO documentation to understand how these parameter values were tuned
    parser.add_argument('--k1', default=0.82, help='BM25 k1 parameter')
    parser.add_argument('--b', default=0.68, help='BM25 b parameter')
    parser.add_argument('--rm3', action='store_true', default=False, help='use RM3')
    parser.add_argument('--fb_terms', default=10, type=int, help='RM3 parameter: number of expansion terms')
    parser.add_argument('--fb_docs', default=10, type=int, help='RM3 parameter: number of documents')
    parser.add_argument('--original_query_weight', default=0.8, type=float, help='RM3 parameter: weight to assign to the original query')

    # Parameters for HQE. The default values are tuned on CAsT train data
    parser.add_argument('--M0', default=5, type=int, help='aggregate historcial queries for first stage (BM25) retrieval')
    parser.add_argument('--M1', default=1, type=int, help='aggregate historcial queries for second stage (BERT) retrieval')
    parser.add_argument('--eta0', default=10, type=float, help='QPP threshold for first stage (BM25) retrieval')
    parser.add_argument('--eta1', default=12, type=float, help='QPP threshold for second stage (BERT) retrieval')
    parser.add_argument('--R0_topic', default=4.5, type=float, help='Topic keyword threshold for first stage (BM25) retrieval')
    parser.add_argument('--R1_topic', default=4, type=float, help='Topic keyword threshold for second stage (BERT) retrieval')
    parser.add_argument('--R0_sub', default=3.5, type=float, help='Subtopic keyword threshold for first stage (BM25) retrieval')
    parser.add_argument('--R1_sub', default=3, type=float, help='Subtopic keyword threshold for second stage (BERT) retrieval')
    parser.add_argument('--filter', default='pos', help='filter word method (no, pos, stp')

    # Parameters for T5
    parser.add_argument('--t5_model_name', default='castorini/t5-base-canard', help='T5 model name')
    parser.add_argument('--max_length', default=64, help='T5 max sequence length')
    parser.add_argument('--num_beams', default=10, help='T5 number of beams')
    parser.add_argument('--no_early_stopping', action='store_false', help='T5 disable early stopping')
    parser.add_argument('--t5_device', default='cuda', help='T5 device to use')

    # Parameters for CQE
    parser.add_argument('--cqe_model_name', default='castorini/tct_colbert-v2-msmarco-cqe', help='CQE model name')
    parser.add_argument('--cqe_l2_threshold', default=10.5, help='Term weight threashold for select terms')
    parser.add_argument('--cqe_max_context_length', default=100, help='CQE max context length')
    parser.add_argument('--cqe_max_query_length', default=36, help='CQE max query length')
    parser.add_argument('--cqe_device', default='cpu', help='CQE device to use')

    # Return args
    args = parser.parse_args()
    return args
    

def run_experiment(rp: RetrievalPipeline):
    with open(args.output + ".trec", "w") as fout:
        total_query_count = 0
        with open(args.qid_queries) as json_file:
            data = json.load(json_file)

        qr_total_time = 0
        initial_time = time.time()
        for session in data:
            session_num = str(session["number"])
            start_time = time.time()
            manual_context_buffer = [None for i in range(len(session["turn"]))]

            for turn_id, conversations in enumerate(session["turn"]):
                query = conversations["raw_utterance"]
                total_query_count += 1

                conversation_num = str(conversations["number"])
                qid = session_num + "_" + conversation_num

                # qr_start_time = time.time()
                # qr_total_time += time.time() - qr_start_time

                if args.add_response!=0:
                    docid = conversations[args.context_field]
                    manual_context_buffer[turn_id] = rp.get_context(docid)
                # We don't use the current context for retrieval but save the context for next turn
                hits = rp.retrieve(query, manual_context_buffer[turn_id])

                for rank in range(len(hits)):
                    docno = hits[rank].docid
                    score = hits[rank].score
                    fout.write("{} Q0 {} {} {} {}\n".format(qid, docno, rank + 1, score, args.run_name))

            rp.reset_history()
            time_per_query = (time.time() - start_time) / (turn_id + 1)
            print(
                "Retrieving session {} with {} queries ({:0.3f} s/query)".format(
                    session["number"], turn_id + 1, time_per_query
                )
            )

        time_per_query = (time.time() - initial_time) / (total_query_count)
        qr_total_time = 0 
        for reformulator in rp.reformulators:
            qr_total_time+=reformulator.total_latency
        qr_time_per_query = qr_total_time / (total_query_count)
        print(
            "Retrieving {} queries ({:0.3f} s/query, QR {:0.3f} s/query)".format(
                total_query_count, time_per_query, qr_time_per_query
            )
        )

    print("total Query Counts %d" % (total_query_count))
    print("Done!")


if __name__ == "__main__":
    args = parse_experiment_args()
    assert (args.sparse_index!=None) or (args.dense_index!=None), "Must input at least one index for search"
    if args.sparse_index==None:
        assert (args.context_index!=None) or (args.add_response==0), "Must input argument context_index"
    else:
        args.context_index = args.sparse_index
    if args.run_name==None:
        args.run_name = 'chatty-goose_' + args.experiment
    experiment = CqrType(args.experiment)

    searcher_settings = SearcherSettings(
        index_path=args.sparse_index,
        k1=args.k1,
        b=args.b,
        rm3=args.rm3,
        fb_terms=args.fb_terms,
        fb_docs=args.fb_docs,
        original_query_weight=args.original_query_weight,
    )

    if experiment == CqrType.HQE or experiment == CqrType.HQE_T5_FUSION:
        #Currently, dense retrieval does not support HQE since it requires longer query sequence
        assert (args.dense_index==None), "HQE does not support dense retrieval. Do not input dense index while using HQE."
    dense_searcher_settings = DenseSearcherSettings(
        index_path=args.dense_index,
        query_encoder=args.query_encoder,
    )

    searcher = build_searcher(searcher_settings)
    dense_searcher = build_dense_searcher(dense_searcher_settings)

    # Initialize CQR and reranker
    reformulators = []
    reranker_query_reformulator = None
    reranker = build_bert_reranker(device=args.reranker_device) if args.rerank else None

    if experiment == CqrType.HQE or experiment == CqrType.HQE_T5_FUSION:
        hqe_bm25_settings = HqeSettings(
            M=args.M0,
            eta=args.eta0,
            R_topic=args.R0_topic,
            R_sub=args.R0_sub,
            filter=PosFilter(args.filter),
            verbose=args.verbose,
        )
        hqe_bm25 = Hqe(searcher, hqe_bm25_settings)
        reformulators.append(hqe_bm25)

    if experiment == CqrType.T5 or experiment == CqrType.HQE_T5_FUSION or experiment == CqrType.CQE_T5_FUSION:
        # Initialize T5 NTR
        t5_settings = NtrSettings(
            model_name=args.t5_model_name,
            max_length=args.max_length,
            num_beams=args.num_beams,
            early_stopping=not args.no_early_stopping,
            verbose=args.verbose,
        )
        t5 = Ntr(t5_settings, device=args.t5_device)
        reformulators.append(t5)

    if experiment == CqrType.HQE:
        hqe_bert_settings = HqeSettings(
            M=args.M1,
            eta=args.eta1,
            R_topic=args.R1_topic,
            R_sub=args.R1_sub,
            filter=PosFilter(args.filter),
        )
        reranker_query_reformulator = Hqe(searcher, hqe_bert_settings)

    if experiment == CqrType.CQE or experiment == CqrType.CQE_T5_FUSION:
        cqe_settings = CqeSettings(
            model_name=args.cqe_model_name,
            l2_threshold=args.cqe_l2_threshold,
            max_context_length=args.cqe_max_context_length,
            max_query_length=args.cqe_max_query_length,
            verbose=args.verbose,
        )
        cqe = Cqe(cqe_settings, device=args.cqe_device)
        reformulators.append(cqe)

    rp = RetrievalPipeline(
        searcher,
        dense_searcher,
        reformulators,
        searcher_num_hits=args.hits,
        early_fusion=not args.late_fusion,
        reranker=reranker,
        reranker_query_reformulator=reranker_query_reformulator,
        add_response = args.add_response,
        context_index_path = args.context_index
    )
    run_experiment(rp)
