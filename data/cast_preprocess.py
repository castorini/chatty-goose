import json
import argparse

from pathlib import Path
from spacy.lang.en import English


def main():
    """Prepare inference file by transforming CAsT queries into CANARD format"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_queries", help="Input CAsT query JSON")
    parser.add_argument("output_dir", help="Output inference file path")
    args = parser.parse_args()

    nlp = English()

    with open(Path(args.input_queries), "r") as fin, open(
        Path(args.output_dir), "w"
    ) as fout:
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
