import logging
import time
import torch

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
        self.history = []

    def rewrite(self, query: str) -> str:
        start_time = time.time()
        self.turn_id += 1

        # Build input sequence from query and history
        self.history += [query]
        src_text = " ||| ".join(self.history)
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
        self.total_latency += time.time() - start_time
        return rewrite_text

    def reset_history(self):
        super().reset_history()
        self.history = []
