from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

import argparse

def main():
    # load t5 tf model to transformers model
    config = T5Config.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('base/model.ckpt-1004000.index', from_tf=True, config=config)

    # get t5 tokenizer ready too
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # save model & tokenizer together to the folder to upload to huggingface
    model.save_pretrained("./torch/t5-base-canard")
    tokenizer.save_pretrained("./torch/t5-base-canard")

if __name__ == '__main__':
    main()
