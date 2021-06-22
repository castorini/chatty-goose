import collections
import re
import time
from typing import Optional

import spacy
from chatty_goose.settings import HqeSettings
from pyserini.search import SimpleSearcher

from .cqr import ConversationalQueryRewriter

__all__ = ["Hqe"]

nlp = spacy.load("en_core_web_sm")
STOP_WORDS = nlp.Defaults.stop_words


class Hqe(ConversationalQueryRewriter):
    """Historical Query Expansion for conversational query reformulation"""

    def __init__(self, searcher: SimpleSearcher, settings: HqeSettings = HqeSettings()):
        super().__init__("Hqe", verbose=settings.verbose)

        # Model settings
        self.M = settings.M
        self.eta = settings.eta
        self.R_topic = settings.R_topic
        self.R_sub = settings.R_sub
        self.pos_filter = settings.filter
        self.searcher = searcher

        # History
        self.key_word_list = collections.defaultdict(list)
        self.subkey_word_list = collections.defaultdict(list)

    def rewrite(self, query: str, context: Optional[str] = None) -> str:
        start_time = time.time()
        self.turn_id += 1
        self.key_word_extraction(context+" "+query if context else query)
        if self.turn_id != 0:
            hits = self.searcher.search(query, 1)
            key_word = self.query_expansion(
                self.key_word_list, 0, self.turn_id)
            subkey_word = ""
            if len(hits) == 0 or hits[0].score <= self.eta:
                end_turn = self.turn_id + 1
                start_turn = end_turn - self.M
                if start_turn < 0:
                    start_turn = 0
                subkey_word = self.query_expansion(
                    self.subkey_word_list, start_turn, end_turn
                )
            query = key_word + " " + subkey_word + " " + query
        self.total_latency += time.time() - start_time
        return query

    def reset_history(self):
        super().reset_history()
        self.key_word_list = collections.defaultdict(list)
        self.subkey_word_list = collections.defaultdict(list)

    def key_word_extraction(self, query):
        proc_query = self.calc_word_score(query)
        # Extract topic keyword
        if self.pos_filter == "no":
            for i, word in enumerate(proc_query["word"]):
                if proc_query["score"][i] >= self.R_topic:
                    self.key_word_list[self.turn_id].append(word)
                if (proc_query["score"][i] >= self.R_sub) & (
                    proc_query["score"][i] < self.R_topic
                ):
                    self.subkey_word_list[self.turn_id].append(word)
        elif self.pos_filter == "pos":
            for i, word in enumerate(proc_query["word"]):
                if ("NN" in proc_query["pos"][i]) or ("JJ" in proc_query["pos"][i]):
                    if proc_query["score"][i] >= self.R_topic:
                        self.key_word_list[self.turn_id].append(word)
                    if (proc_query["score"][i] >= self.R_sub) & (
                        proc_query["score"][i] < self.R_topic
                    ):
                        self.subkey_word_list[self.turn_id].append(word)
        elif self.pos_filter == "stp":
            for i, word in enumerate(proc_query["word"]):
                if word not in STOP_WORDS:
                    if proc_query["score"][i] >= self.R_topic:
                        self.key_word_list[self.turn_id].append(word)
                    if (proc_query["score"][i] >= self.R_sub) & (
                        proc_query["score"][i] < self.R_topic
                    ):
                        self.subkey_word_list[self.turn_id].append(word)

    def calc_word_score(self, query):
        nlp_query = nlp(pre_process(query))
        proc_query = process(nlp_query)
        query_words = proc_query["word"]
        proc_query["score"] = []

        for query_word in query_words:
            hits = self.searcher.search(query_word, 1)
            try:
                score = hits[0].score
                proc_query["score"].append(score)
            except:
                proc_query["score"].append(-1)

        return proc_query

    @staticmethod
    def query_expansion(key_word_list, start_turn, end_turn):
        query_expansion = ""
        for turn in range(start_turn, end_turn + 1):
            for word in key_word_list[turn]:
                query_expansion = query_expansion + " " + word
        return query_expansion


def pre_process(text):
    text = re.sub(
        u"-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t",
        _space_extend,
        text,
    )
    text = text.strip(" \n")
    text = re.sub("\s+", " ", text)
    return text


def process(parsed_text):
    output = {
        "word": [],
        "lemma": [],
        "pos": [],
        "pos_id": [],
        "ent": [],
        "ent_id": [],
        "offsets": [],
        "sentences": [],
    }

    for token in parsed_text:
        output["word"].append(_str(token.text))
        pos = token.tag_
        output["pos"].append(pos)

    return output


def _space_extend(matchobj):
    return " " + matchobj.group(0) + " "


def _str(s):
    """ Convert PTB tokens to normal tokens """
    if s.lower() == "-lrb-":
        s = "("
    elif s.lower() == "-rrb-":
        s = ")"
    elif s.lower() == "-lsb-":
        s = "["
    elif s.lower() == "-rsb-":
        s = "]"
    elif s.lower() == "-lcb-":
        s = "{"
    elif s.lower() == "-rcb-":
        s = "}"
    return s
