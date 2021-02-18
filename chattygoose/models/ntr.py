from transformers import T5Tokenizer, T5ForConditionalGeneration

__all__ = ["T5_NTR"]


class T5_NTR(object):
    def __init__(self, pretrained_ckpt):
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_ckpt)
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_ckpt)
        except ValueError as e:
            print(str(e))
            return None

        self.history = []
        print(f"Initializing {pretrained_ckpt}...")

    def rewrite(self, query):
        self.history += [query]
        source_text = " ".join(self.history)
        input_ids = self.tokenizer(
            source_text, return_tensors="pt", add_special_tokens=True
        ).input_ids
        output_ids = self.model.generate(input_ids)
        rewrite_text = self.tokenizer.decode(
            output_ids[0, 0:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )

        return rewrite_text

    def reset_history(self):
        self.history = []
