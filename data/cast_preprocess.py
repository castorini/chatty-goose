import json
import argparse

from pathlib import Path
from spacy.lang.en import English

PATH = "treccastweb/2019/data/evaluation/"
FILE = "evaluation_topics_v1.0.json"
OUTPUT = "test-src.cast.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("split")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    nlp = English()

    with open(
        Path(args.data_dir, args.split, args.split + "_topics_v1.0.json"), "r"
    ) as fin, open(Path(args.output_dir, args.split + "-src.cast.txt"), "w") as fout:
        js_list = json.load(fin)
        for topic in js_list:
            history = []
            topic_id = topic["number"]
            turns = topic["turn"]
            print(f"========TOPIC {topic_id}=======")
            for turn in turns:
                turn_id = turn["number"]
                raw_query = turn["raw_utterance"]
                src_text = " ||| ".join(history + [raw_query])
                src_text = " ".join([tok.text for tok in nlp(src_text)])
                history += [raw_query]
                print(src_text)
                fout.write(src_text + "\n")


if __name__ == "__main__":
    main()
