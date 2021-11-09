import logging
import time
import torch
from typing import Optional
from pyserini.search import querybuilder
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from .cqr import ConversationalQueryRewriter
from chatty_goose.settings import CqeSettings
import numpy as np
from numpy import linalg as LA

name = "Cqe"
__all__ = [name]


class Cqe(ConversationalQueryRewriter):
    """Neural Transfer Reformulation using a trained T5 model"""

    def __init__(self, settings: CqeSettings = CqeSettings(), device: str = None):
        super().__init__(name, verbose=settings.verbose)
        self.name=name
        # Model settings
        if settings.model_name:
            self.l2_threshold = settings.l2_threshold
            self.max_query_length = settings.max_query_length
            self.max_context_length = settings.max_context_length
            self.max_length = self.max_query_length + self.max_context_length
            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(device)
            if self.verbose:
                logging.info(f"Initializing CQE using model {settings.model_name}...")
            logging.info(f"Initializing CQE using model {settings.model_name}...")
            self.model = BertModel.from_pretrained(settings.model_name)
            self.model.to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(settings.model_name)
            self.has_model = True
            self.history_query = []
            self.history = []
        if (not self.has_model):
            raise Exception('Neither query encoder model nor encoded queries provided. Please provide at least one')

    def rewrite(self, query: str, context: Optional[str] = None, response_num: Optional[int] = 0) -> str:
        start_time = time.time()
        self.turn_id += 1

        # If the passage from canonical result (context) is provided, it is added to history.
        # Since canonical passage can be large and there is limit on length of tokens,
        # only one passage for the new query is used at a time.
        self.history_query += [query]
        self.history += [query]
        if response_num!=0:
            src_ctx = '[CLS] ' +"|".join(self.history_query[:-response_num] + self.history[-2*response_num:-1])
        else:
            src_ctx = '[CLS] ' +"|".join(self.history_query[:-1])
        src_q = ' [Q] ' + self.history[-1] + '[MASK]' * self.max_query_length
        if self.turn_id == 0:
            input_ids = self.tokenizer.encode('[CLS] ' +src_q, max_length=self.max_query_length, add_special_tokens=False)
            token_type_ids =[0] + [1] * (len(input_ids)-1)
            self.context_length = 3
            effective_token_weights = [0]*4 + [1]*(self.max_query_length-4)
        else:
            src_ctx_token_ids = self.tokenizer.encode(src_ctx, max_length=self.max_context_length, add_special_tokens=False)
            src_query_token_ids = self.tokenizer.encode(src_q, max_length=self.max_query_length, add_special_tokens=False)
            input_ids = src_ctx_token_ids + src_query_token_ids
            token_type_ids = [0]*len(src_ctx_token_ids) + [1]*len(src_query_token_ids)
            self.context_length = len(src_ctx_token_ids) - 1 + 3
            effective_token_weights = [0] + [1]*(len(src_ctx_token_ids)-1) + [0]*3 + [1]*(self.max_query_length-3)
        input_tokens = []
        for input_id in input_ids[1:]:
            input_tokens.append(self.tokenizer.decode(input_id).replace(' ','') ) #transform whitepiece token into text by removing ' '

        input_ids = torch.LongTensor([input_ids]).to(self.device)
        token_type_ids = torch.LongTensor([token_type_ids]).to(self.device)
        


        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids)
        embeddings = outputs.last_hidden_state.detach().cpu().numpy()

        self.query_embs =  np.average(embeddings, axis=1, weights=effective_token_weights).astype(np.float32)

        # Generate reformulated query with term weight
        query_token_weights = np.squeeze(LA.norm(embeddings[:,1:,:], axis=-1))
        
        rewrite_text = self.build_query(input_tokens, query_token_weights, self.l2_threshold)
        if context:
            self.history += [context]

        self.total_latency += time.time() - start_time
        return rewrite_text

    def reset_history(self):
        super().reset_history()
        self.history_query = []
        self.history = []

    def build_query(self, query_tokens, query_token_weights, threshold):
        context_token_num = self.context_length
        mean_l2 = query_token_weights.mean()

        should = querybuilder.JBooleanClauseOccur['should'].value
        boolean_query_builder = querybuilder.get_boolean_query_builder()
        is_context = True
        term_weight = 0
        term =''

        for i, token in enumerate(query_tokens): 
            if token == '[MASK]':

                continue
            if '##' in token:
                term += token[2:]
                term_weight = max(term_weight, query_token_weights[i])
            else:
                if ( (term_weight > threshold) or (i>=context_token_num)): #10.5, 12
                    try:
                        term = querybuilder.get_term_query(term)
                        boost = querybuilder.get_boost_query(term, term_weight/mean_l2)
                        boolean_query_builder.add(boost, should)
                    except:
                        pass

                term = token
                term_weight = query_token_weights[i]

        return boolean_query_builder.build()

