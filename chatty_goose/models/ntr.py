from spacy.lang.en import English
from transformers import T5ForConditionalGeneration, T5Tokenizer

__all__ = ["T5_NTR"]


class T5_NTR(object):
    def __init__(
        self,
        model_name: str = "castorini/t5-base-canard",
        max_length: int = 64,
        num_beams: int = 10,
        early_stopping: bool = True,
    ):
        print(f"Initializing {model_name}...")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.nlp = English()

        self.history = []
        self.max_length = max_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping

    def rewrite(self, query):
        # Build input sequence from query and history
        self.history += [query]
        src_text = " ||| ".join(self.history)
        src_text = " ".join([tok.text for tok in self.nlp(src_text)])
        input_ids = self.tokenizer(
            src_text, return_tensors="pt", add_special_tokens=True
        ).input_ids

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
        return rewrite_text

    def reset_history(self):
        self.history = []
