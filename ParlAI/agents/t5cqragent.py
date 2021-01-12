from parlai.core.agents import register_agent, Agent
from parlai.core.torch_agent import History
from pyserini.search import SimpleSearcher
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json


@register_agent("t5cqrbm25ranker")
class T5CQRBM25Agent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser):
        parser.add_argument('--name', type=str, default='T5CQR-BM25', help="The agent's name.")
        return parser
        
    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.id = 'T5CQRBM25Ranker'
        self.name = opt['name']
        self.searcher = SimpleSearcher.from_prebuilt_index('cast2019')

        self.model = T5ForConditionalGeneration.from_pretrained('castorini/t5-base-canard')
        self.tokenizer = T5Tokenizer.from_pretrained('castorini/t5-base-canard')
        self.history = []
    
    def rewrite(self, last_word, history):

        source_text = ' '.join(self.history)
        input_ids  = self.tokenizer(source_text, return_tensors="pt", add_special_tokens=True).input_ids
        output_ids = self.model.generate(input_ids)
        rewrite_text = self.tokenizer.decode(output_ids[0, 0:], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        
        return rewrite_text

    def observe(self, observation):
        # Gather the last word from the other user's input
        words = observation.get('text', '')

        if words:
            last_word = words
        else:
            last_word = ""
        
        self.history += [last_word]

        rewrite_text = self.rewrite(last_word, self.history)

        hits = self.searcher.search(rewrite_text)
        self.responses = [json.loads(hits[i].raw) for i in range(5)]
    
    def act(self):
        # Always return a string like this.
        return {
            'id': self.id,
            'text': f"{self.responses}",
        }
