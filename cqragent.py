from parlai.core.agents import register_agent, Agent
from parlai.core.torch_agent import History
from pyserini.search import SimpleSearcher
import json
from cqr_hqe.model import HQE
from cqr_t5.model import T5CQR
from cqr_hqe.util import reciprocal_rank_fusion

@register_agent("ConversationalSearcher")
class ConversationSearchAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt=None):
        parser.add_argument('--name', type=str, default='h2oloo', help="The agent's name.")
        parser.add_argument('--cqr_model', type=str, default='T5', help="HQE, T5 or fusion")
        # pyserini
        parser.add_argument('--k1', default=0.82, help='BM25 k1 parameter')
        parser.add_argument('--b', default=0.68, help='BM25 b parameter')
        parser.add_argument('--from_prebuilt', type=str, default='cast2019', help="Pyserini prebuilt index.")
        # T5
        parser.add_argument('--from_pretrained', type=str, default='castorini/t5-base-canard', help="HGF T5 CQR checkpoint.")
        parser.add_argument('--episode_done', type=str, default='[END]', help="End signal for interactive mode")
        # HQE
        parser.add_argument('--M', default=5, type=int, help='aggregate historcial queries for first stage (BM25) retrieval')
        parser.add_argument('--eta', default=10, type=float, help='QPP threshold for first stage (BM25) retrieval')
        parser.add_argument('--R_topic', default=4.5, type=float, help='Topic keyword threshold for first stage (BM25) retrieval')
        parser.add_argument('--R_sub', default=3.5, type=float, help='Subtopic keyword threshold for first stage (BM25) retrieval')
        parser.add_argument('--filter', default='pos', help='filter word method: pos, no, stp')
        parser.add_argument('--verbose', action='store_true')
        return parser

    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.name = opt['name']
        self.episode_done = opt['episode_done']
        self.cqr_model = opt['cqr_model']
        if self.cqr_model == 'T5':
            self.id = 'T5'
        elif self.cqr_model == 'HQE':
            self.id = 'HQE'
        elif self.cqr_model == 'fusion':
            self.id = 'HQET5Fusion'
        if self.cqr_model == 'T5' or self.cqr_model == 'fusion':
            self.T5  = T5CQR(opt['from_pretrained'])
        if self.cqr_model == 'HQE' or self.cqr_model == 'fusion':
            self.HQE = HQE(opt['M'], opt['eta'], opt['R_topic'], opt['R_sub'], opt['filter'], opt['verbose'])
        self.searcher = SimpleSearcher.from_prebuilt_index(opt['from_prebuilt'])
        self.searcher.set_bm25(float(opt['k1']), float(opt['b']))
    def observe(self, observation):
        # Gather the last word from the other user's input
        self.query = observation.get('text', '')
        if observation.get('episode_done') or self.query == self.episode_done:
            print('Reset History')
            if self.cqr_model == 'T5' or self.cqr_model == 'fusion':
                self.T5.reset_history()
            if self.cqr_model == 'HQE' or self.cqr_model == 'fusion':
                self.HQE.reset_history()

    def act(self):
        # Always return a string like this.
        if self.query == self.episode_done:

            return {
                'id': self.id,
                'text': 'Session finished'}

        if self.cqr_model == 'T5':
            rewrite_text = self.T5.rewrite(self.query)
            hits = self.searcher.search(rewrite_text, 1)
            docid = hits[0].docid
        if self.cqr_model == 'HQE':
            rewrite_text = self.HQE.rewrite(self.query, self.searcher)
            hits = self.searcher.search(rewrite_text, 1)
            docid = hits[0].docid
        if self.cqr_model == 'fusion':
            t5_rewrite_text = self.T5.rewrite(self.query)
            hits = self.searcher.search(t5_rewrite_text, 1000)
            t5_docids = [hits[i].docid for i in range(1000)]

            hqe_rewrite_text = self.HQE.rewrite(self.query, self.searcher)
            hits = self.searcher.search(hqe_rewrite_text, 1000)
            hqe_docids = [hits[i].docid for i in range(1000)]
            docid = reciprocal_rank_fusion(t5_docids, hqe_docids)[0]


        return {
            'id': self.id,
            'text': self.searcher.doc(docid).raw(),
        }

if __name__ == '__main__':
    from parlai.scripts.display_data import DisplayData
    from parlai.scripts.display_model import DisplayModel
    from parlai.core.teachers import register_teacher, DialogTeacher

    @register_teacher("my_teacher")
    class MyTeacher(DialogTeacher):
        def __init__(self, opt, shared=None):
            # opt is the command line arguments.

            # What is this shared thing?
            # We make many copies of a teacher, one-per-batchsize. Shared lets us store

            # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
            # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
            # the fold name (train/valid/test) + ".txt"
            opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
            self.id = 'teacher'
            super().__init__(opt, shared)

        def setup_data(self, datafile):
            # filename tells us where to load from.
            # We'll just use some hardcoded data, but show how you could read the filename here:
            print(f" ~~ Loading from {datafile} ~~ ")

            # setup_data should yield tuples of ((text, label), new_episode)
            # That is ((str, str), bool)

            # first episode
            # notice how we have call, response, and then True? The True indicates this is a first message
            # in a conversation
            yield ('Frank Zappa Disbandment What group disbanded?', 'What group disbanded?'), True
            # Next we have the second turn. This time, the last element is False, indicating we're still going
            yield ('Zappa and the Mothers of Invention When did they disband?', 'When did Zappa and the Mothers of Invention disband?'), False
            yield ("In late 1969, Zappa broke up the band. What kind of music did they play?", 'What kind of music did Zappa and the Mothers of Invention play?'), False

    #DisplayData.main(task="my_teacher")

    #DisplayModel.main(task='my_teacher', model='t5cqrbm25ranker')

    from parlai.scripts.interactive import Interactive
    Interactive.main(model='ConversationalSearcher', cqr_model='fusion')

