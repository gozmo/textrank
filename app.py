import pudb
import streamlit as st
import os
import json
import spacy
import colour
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from itertools import product
from collections import defaultdict
from random import random

#STOPWORDS = ["a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your"]

class TextRank:
    def __init__(self, window_size, iterations, keywords, ngram_size):
        self.nlp = spacy.load("en_core_web_sm")
        # Create a blank Tokenizer with just the English vocab
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.window_size = window_size
        self.edges = defaultdict(lambda : defaultdict(int))
        self.nodes = defaultdict(float)
        self.d = 0.85
        self.iterations = iterations
        self.keywords = keywords
        self.ngram_size = ngram_size

    def process_document(self, document):
        doc = self.nlp(document)
        self.tokenized_sentences = []
        for sentence in doc.sents:
            tokenized_sentence = self.__tokenize_sentence(sentence)
            self.tokenized_sentences.append(tokenized_sentence)

            for start_offset in range(len(tokenized_sentence)):
                end_offset = start_offset + self.window_size
                if len(tokenized_sentence) < end_offset:
                    break
                window = tokenized_sentence[start_offset:end_offset]
                ngrams = self.calc_grams(window)
                self.__update_graph(ngrams)
        self.loop()
        keywords = self.top()
        return keywords
        # merge = self.merge_keywords(keywords)
        # return merge

    def calc_grams(self, window):
        grams = []
        for idx in range(len(window)):
            end_index = idx + self.ngram_size
            if len(window) < end_index:
                break
            gram = " ".join(window[idx:end_index])
            grams.append(gram)
        return grams


    def __tokenize_sentence(self, sentence):
        tokenized_sentence = self.tokenizer(str(sentence))
        #tokenized_sentence = filter(lambda token: str(token) not in STOPWORDS, tokenized_sentence)
        tokenized_sentence = map(lambda x: str(x), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.strip(), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.lower(), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.replace(",", ""), tokenized_sentence)
        tokenized_sentence = map(lambda x: x.replace(".", ""), tokenized_sentence)
        return list(tokenized_sentence)


    def __update_graph(self, window):
        all_combinations = product(window, repeat=2)
        for source, target in all_combinations:
            self.add_edge(source, target)

    def top(self):
        kw = sorted(self.nodes.items(), reverse=True, key=lambda x:x[1])
        kw = list(map(lambda x:x[0], kw))
        return kw[:self.keywords]

    def add_edge(self, source, target):
        self.edges[source][target] += 1

        self.add_node(source)
        self.add_node(target)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 3 * random()

    def calculate_score(self, source_node):
        score = 0
        for target_node in self.edges[source_node].keys():
            score += self.nodes[target_node] / float(sum(self.edges[target_node].values()))
        return score

    def loop(self):
        for _ in range(self.iterations):
            for node in self.nodes.keys():
                score = self.calculate_score(node)
                self.nodes[node] = (1 - self.d) + self.d + score
    
    def merge_keywords(self, keywords):
        merged_keywords = []
        for sentence in self.tokenized_sentences:
            a = []
            for token in sentence:
                is_keyword = token in keywords
                if is_keyword:
                    a.append(token)
                elif not is_keyword and 0 < len(a):
                    merged_keywords.append(a)
                    a = []
            if 0 < len(a):
                merged_keywords.append(a)
        
        return set([" ".join(a) for a in merged_keywords])
                
    def get_heatmap(self):
        heatmap = []
        max_value = max(self.nodes.values())
        for sentence in self.tokenized_sentences:
            sentence_map = [(token, self.nodes[token]/max_value) for token in sentence]
            heatmap.extend(sentence_map)
        return heatmap

textrank_example = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strictinequations, and nonstrict inequations are considered. Upper bounds forcomponents of a minimal set of solutions and algorithms of construction ofminimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimalsupporting set of solutions can be used in solving all the considered  typessystems and systems of mixed types."

st.title("TextRank example")

document = st.text_input("text to be processed", textrank_example, "input_document")

window_size = st.slider("window_size", 2, 10, step=1, value=2)
iterations = st.slider("iterations", 1, 50, step=1, value=20)
keywords = st.slider("keywords", 1, 40, step=1, value=5)
ngram_size = st.slider("ngram_size", 1, 5, step=1, value=1)

textrank = TextRank(window_size, iterations, keywords, ngram_size)
keywords = textrank.process_document(document)
heatmap = textrank.get_heatmap()

st.table(keywords)

markdown_text = ""
blue = colour.Color("#bfbfbf")
colors = list(blue.range_to(colour.Color("black"),11))
for token, score in heatmap:
    idx = int(score * 10)
    markdown_token = f"<span style=\"color:{colors[idx]}\">{token}</span>"
    markdown_text += " " + markdown_token
st.markdown(markdown_text, unsafe_allow_html=True)
