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

    def add_edge(self, source, target):
        self.edges[source][target] += 1

        self.add_node(source)
        self.add_node(target)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 10 * random.random()

    def calculate_score(self, node):
            for self.edges[node]




class TextRank:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Create a blank Tokenizer with just the English vocab
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.window_size = 3
        self.graph = Graph()

    def run(self):
        for json_file in os.listdir("data/"):
            path = os.path.join("data", json_file)
            sentence = self.__read_file(path)
            self.__loop(sentence)
        print(self.graph)


    def __read_file(self, path):
        with open(path, "r") as f:
            json_content = json.loads(f.read())
        return json_content["summary"]

    def __loop(self, document):
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




textrank = TextRank()
textrank.run()
