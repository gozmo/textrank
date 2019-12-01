import os
import json
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from itertools import product
from collections import defaultdict
from random import random

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


class TextRank:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Create a blank Tokenizer with just the English vocab
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.window_size = 3
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
            tokenized_sentence = self.tokenizer(str(sentence))
            for start_offset in range(len(tokenized_sentence)):
                end_offset = start_offset + self.window_size
                if len(tokenized_sentence) < end_offset:
                    break
                window = tokenized_sentence[start_offset:end_offset]
                self.__update_graph(window)

    def __update_graph(self, window):
        all_combinations = product(window, repeat=2)
        for source, target in all_combinations:
            self.graph.add_edge(source, target)

    def train(self, iterations):
        self.graph.loop(iterations)
        self.graph.top()

    def top(self):
        a = [(score, node) for node, score in self.graph.nodes.items()]
        a = sorted(a)
        return a[-20:])



# sample_text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strictinequations, and nonstrict inequations are considered. Upper bounds forcomponents of a minimal set of solutions and algorithms of construction ofminimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimalsupporting set of solutions can be used in solving all the considered  typessystems and systems of mixed types."

# textrank = TextRank()
# #textrank.build_graph()
# textrank.loop(sample_text)
# textrank.train(30)

