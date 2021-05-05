import logging

from chatty_goose.cqr import Hqe, Ntr
from chatty_goose.pipeline import RetrievalPipeline
from chatty_goose.settings import HqeSettings, NtrSettings
from chatty_goose.types import CqrType, PosFilter
from parlai.core.agents import Agent, register_agent
from pyserini.search import SimpleSearcher


@register_agent("ChattyGooseAgent")
class ChattyGooseAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt = None):
        parser.add_argument('--name', type=str, default='CQR', help="The agent's name.")
        parser.add_argument('--cqr_type', type=str, default='fusion', help="hqe, t5, or fusion")
        parser.add_argument('--episode_done', type=str, default='[END]', help="end signal for interactive mode")
        parser.add_argument('--hits', type=int, default=50, help="number of hits to retrieve from searcher")

        # Pyserini
        parser.add_argument('--k1', default=0.82, help='BM25 k1 parameter')
        parser.add_argument('--b', default=0.68, help='BM25 b parameter')
        parser.add_argument('--from_prebuilt', type=str, default='cast2019', help="Pyserini prebuilt index")
       
        # T5
        parser.add_argument('--from_pretrained', type=str, default='castorini/t5-base-canard', help="Huggingface T5 checkpoint")
        
        # HQE
        parser.add_argument('--M', default=5, type=int, help='aggregate historcial queries for first stage (BM25) retrieval')
        parser.add_argument('--eta', default=10, type=float, help='QPP threshold for first stage (BM25) retrieval')
        parser.add_argument('--R_topic', default=4.5, type=float, help='topic keyword threshold for first stage (BM25) retrieval')
        parser.add_argument('--R_sub', default=3.5, type=float, help='subtopic keyword threshold for first stage (BM25) retrieval')
        parser.add_argument('--filter', default='pos', help='filter word method: no, pos, stp')
        parser.add_argument('--verbose', action='store_true')
        return parser

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.name = opt["name"]
        self.episode_done = opt["episode_done"]
        self.cqr_type = CqrType(opt["cqr_type"])

        # Initialize searcher
        searcher = SimpleSearcher.from_prebuilt_index(opt["from_prebuilt"])
        searcher.set_bm25(float(opt["k1"]), float(opt["b"]))

        # Initialize retrievers
        retrievers = []
        if self.cqr_type == CqrType.HQE or self.cqr_type == CqrType.FUSION:
            hqe_settings = HqeSettings(
                M=opt["M"],
                eta=opt["eta"],
                R_topic=opt["R_topic"],
                R_sub=opt["R_sub"],
                filter=PosFilter(opt["filter"]),
                verbose=opt["verbose"],
            )
            hqe = Hqe(searcher, hqe_settings)
            retrievers.append(hqe)
        if self.cqr_type == CqrType.T5 or self.cqr_type == CqrType.FUSION:
            t5_settings = NtrSettings(model_name=opt["from_pretrained"], verbose=opt["verbose"])
            t5 = Ntr(t5_settings)
            retrievers.append(t5)

        self.rp = RetrievalPipeline(searcher, retrievers, int(opt["hits"]))

    def observe(self, observation):
        # Gather the last word from the other user's input
        self.query = observation.get("text", "")
        if observation.get("episode_done") or self.query == self.episode_done:
            logging.info("Resetting agent history")
            self.rp.reset_history()

    def act(self):
        if self.query == self.episode_done:
            return {"id": self.id, "text": "Session finished"}

        # Retrieve hits
        hits = self.rp.retrieve(self.query)
        if len(hits) == 0:
            result = "Sorry, I couldn't find any results"
        else:
            result = hits[0].raw
        return { "id": self.id, "text": result }


if __name__ == "__main__":
    from parlai.scripts.interactive import Interactive

    Interactive.main(model="ChattyGooseAgent", cqr_type="fusion")
