import logging
import time
import torch
from typing import Optional

from chatty_goose.settings import NtrSettings
from spacy.lang.en import English
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .cqr import ConversationalQueryRewriter

__all__ = ["Ntr"]


class Ntr(ConversationalQueryRewriter):
    """Neural Transfer Reformulation using a trained T5 model"""

    def __init__(self, settings: NtrSettings = NtrSettings(), device: str = None):
        super().__init__("Ntr", verbose=settings.verbose)

        # Model settings
        self.max_length = settings.max_length
        self.num_beams = settings.num_beams
        self.early_stopping = settings.early_stopping

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        if self.verbose:
            logging.info(f"Initializing T5 using model {settings.model_name}...")
        self.model = (
            T5ForConditionalGeneration.from_pretrained(settings.model_name)
            .to(device)
            .eval()
        )
        self.tokenizer = T5Tokenizer.from_pretrained(settings.model_name)
        self.nlp = English()
        self.history_query = []
        self.history = []

    def rewrite(self, query: str, context: Optional[str] = None, response_num: Optional[int] = 0) -> str:
        start_time = time.time()

        # If the passage from canonical result (context) is provided, it is added to history.
        # Since canonical passage can be large and there is limit on length of tokens,
        # only one passage for the new query is used at a time.

        # if len(self.history) >= 2 and self.has_canonical_context:
        #     self.history.pop(-2)
        #     self.has_canonical_context = False
        self.history_query += [query]
        self.history += [query]
        

        # Build input sequence from query and history
        if response_num!=0:
            src_text = " ||| ".join(self.history_query[:-response_num] + self.history[-2*response_num:])
        else:
            src_text = " ||| ".join(self.history_query)
            

        src_text = " ".join([tok.text for tok in self.nlp(src_text)])
        input_ids = self.tokenizer(
            src_text, return_tensors="pt", add_special_tokens=True
        ).input_ids.to(self.device)

        # Generate new sequence
        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=self.early_stopping,
        )

        # Decode output
        rewrite_text = self.tokenizer.decode(
            output_ids[0, 0:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        if context:
            self.history += [context]
        self.total_latency += time.time() - start_time
        return rewrite_text

    def reset_history(self):
        super().reset_history()
        self.history = []
        self.history_query = []
