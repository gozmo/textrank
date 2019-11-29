import streamlit as st
import os
import json
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from itertools import product
from collections import defaultdict
from random import random

STOPWORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]

class Graph:
    def __init__(self):
        self.edges = defaultdict(lambda : defaultdict(int))
        self.nodes = {}
        self.d = 0.85

    def add_edge(self, source, target):
        self.edges[source][target] += 1

        self.add_node(source)
        self.add_node(target)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 10 * random()

    def calculate_score(self, source_node):
        score = 0
        for target_node in self.edges[source_node].keys():
            score += self.nodes[target_node] / float(sum(self.edges[target_node].values()))
        return score

    def loop(self, iterations):
        for _ in range(iterations):
            for node in self.nodes.keys():
                score = self.calculate_score(node)
                self.nodes[node] = (1 - self.d) + self.d + score
        print(self.nodes)


class TextRank:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Create a blank Tokenizer with just the English vocab
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.window_size = 5
        self.graph = Graph()

    def build_graph_folder(self):
        for json_file in os.listdir("data/"):
            path = os.path.join("data", json_file)
            sentence = self.__read_file(path)
            self.loop(sentence)



    def __read_file(self, path):
        with open(path, "r") as f:
            json_content = json.loads(f.read())
        return json_content["summary"]

    def loop(self, document):
        doc = self.nlp(document)
        for sentence in doc.sents:
            tokenized_sentence = self.__tokenize_sentence(sentence)

            for start_offset in range(len(tokenized_sentence)):
                end_offset = start_offset + self.window_size
                if len(tokenized_sentence) < end_offset:
                    break
                window = tokenized_sentence[start_offset:end_offset]
                self.__update_graph(window)

    def __tokenize_sentence(self, sentence):
        tokenized_sentence = self.tokenizer(str(sentence))
        tokenized_sentence = filter(lambda token: str(token) not in STOPWORDS, tokenized_sentence)
        tokenized_sentence = map(lambda x: str(x), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.strip(), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.lower(), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.replace(",", ""), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.replace(".", ""), tokenized_sentence)
        return list(tokenized_sentence)


    def __update_graph(self, window):
        all_combinations = product(window, repeat=2)
        for source, target in all_combinations:
            self.graph.add_edge(source, target)

    def train(self, iterations):
        self.graph.loop(iterations)

    def top(self):
        a = [(score, node) for node, score in self.graph.nodes.items()]
        b = sorted(a, reverse=True)

        keywords = b[:20]
        return keywords

textrank_example = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strictinequations, and nonstrict inequations are considered. Upper bounds forcomponents of a minimal set of solutions and algorithms of construction ofminimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimalsupporting set of solutions can be used in solving all the considered  typessystems and systems of mixed types."

st.title("TextRank example")

document = st.text_input("text to be processed", textrank_example, "input_document")


textrank = TextRank()
textrank.loop(document)
textrank.train(30)

keywords = textrank.top()

st.table(keywords)
