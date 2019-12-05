from collections import Counter
from collections import defaultdict
from itertools import combinations
from minio import Minio
from minio.error import ResponseError
from pathlib import Path
from random import random
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import graphviz
import feedparser
import urllib
import colour
import getpass
import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
import sys
import glob
import os
import pudb
import spacy
import streamlit as st

MINIO_URL = 'alicia:12921'
CACHE_FOLDER = 'data_cache/'
def fetch_secrets():
    secrets = {}
    home = str(Path.home())
    with open(f'{home}/.ssh/secrets.properties', 'r') as f:
        for line in f.readlines():
            key, value = line.split('=')
            secrets[key.strip()] = value.strip()
    return secrets

class Datawarehouse:
    def __init__(self):
        secrets = fetch_secrets()
        self.minio_client = Minio(MINIO_URL,
                             access_key=secrets['datawarehouse-pub'],
                             secret_key=secrets['datawarehouse-prv'],
                             secure=False)
    def get_available_corpus(self):
        files =  self.minio_client.list_objects('labeled-corpora')
        return [f.object_name for f in files]

    def download(self, corpus):
        version = self.minio_client.list_objects(bucket_name='labeled-corpora', prefix=corpus)
        corpus_with_version =  [v.object_name for v in version][-1]
        try:
            os.makedirs(CACHE_FOLDER + corpus_with_version)
        except:
            pass
        for oo in self.minio_client.list_objects(bucket_name='labeled-corpora', prefix=corpus_with_version):
            object_name = oo.object_name

            data_stream = self.minio_client.get_object(bucket_name='labeled-corpora', object_name=object_name)
                #
                #
            file_name = CACHE_FOLDER + object_name

                #
                #
            with open(file_name, 'wb') as file_data:
                for d in data_stream:
                    file_data.write(d)
        return CACHE_FOLDER + corpus_with_version

@st.cache
def arxiv():
    #rss_feed_ids = ["cs", "stat"]
    rss_feed_ids = ["cs"]
    base_url = "http://export.arxiv.org/rss/"

    papers = dict()
    for feed_id in rss_feed_ids:
        feed_url = urllib.parse.urljoin(base_url, feed_id)

        posts = feedparser.parse(feed_url)

        for item in posts.entries[0:100]:
            title = item.title
            summary = item.summary
            link = item.link
            temp = urllib.parse.urlparse(link)
            filename = temp.path.replace("/abs/","")
            papers[title] = summary
    return papers

class TextRank:
    def __init__(self, window_size, iterations, num_keywords, allowed_pos, use_lemma, weighted_graph, use_stopwords,
            token_or_sentence="token", merge_keywords=True, damping_factor=0.85):
        self.nlp = spacy.load("en_core_web_sm")

        self.window_size = window_size
        self.edges = defaultdict(lambda : defaultdict(int))
        self.nodes = defaultdict(float)
        self.d = damping_factor
        self.iterations = iterations
        self.num_keywords = num_keywords
        self.allowed_pos = allowed_pos
        self.use_lemma = use_lemma
        self.weighted_graph = weighted_graph
        self.use_stopwords = use_stopwords
        self.counter = Counter()
        self.filtered_counter = Counter()
        self.token_or_sentence = token_or_sentence
        self.merge_keywords = merge_keywords

    def process_document(self, document):
        doc = self.nlp(document)
        self.tokenized_sentences = []
        for sentence in doc.sents:
            if self.token_or_sentence == "token":
                tokenized_sentence = self.__tokenize_sentence(sentence)
                self.tokenized_sentences.append(tokenized_sentence)

            for window in self.get_window(tokenized_sentence):
                self.__update_graph(window)
            self.__update_filtered_word_count([token for _, token in tokenized_sentence if token != None])
            self.__update_word_count([token.text for token in sentence if not token.is_stop and not token.pos_ == "PUNCT"])
        self.loop()
        keywords = self.top()
        if self.merge_keywords:
            keywords= self.merge(keywords)
        keywords = sorted(keywords)
        return keywords


    def loop(self):
        key_to_follow =list(self.nodes.keys())[1]
        for _ in range(self.iterations):
            for node in self.nodes.keys():
                score = self.calculate_score(node)
                self.nodes[node] = (1 - self.d) + self.d * score

    def calculate_score(self, source_node):
        score = 0
        for target_node in self.edges[source_node].keys():
            target_node_all_edge_weights = float(sum(self.edges[target_node].values()))
            edge_weight_to_source_node = self.edges[target_node][source_node]
            target_node_score = self.nodes[target_node]

            score += (edge_weight_to_source_node / target_node_all_edge_weights) * target_node_score 

        return score

    def __tokenize_sentence(self, sentence):
        return [self.clean_token(token) for token in sentence]

    def clean_token(self, token):
        original_token = token.text
        if token.pos_ in ["SYM",  "PUNCT", "SPACE"] and not self.use_stopwords:
            return (original_token, None)
        if token.pos_ not in self.allowed_pos and len(self.allowed_pos) > 0:
            return (original_token, None)
        if token.is_stop and not self.use_stopwords:
            return (original_token, None)

        if self.use_lemma:
            cleaned_token = token.lemma_
        else:
            cleaned_token = token.text

        cleaned_token = cleaned_token.strip()
        cleaned_token = cleaned_token.lower()

        if cleaned_token == "":
            cleaned_token = None
        elif cleaned_token == "-":
            cleaned_token = None
            
        return (original_token, cleaned_token)

    def get_window(self, tokenized_sentence):
        for start_idx in range(len(tokenized_sentence)):
            if tokenized_sentence[start_idx][1] == None:
                continue
            end_idx_start = start_idx + self.window_size

            for end_idx in range(end_idx_start, len(tokenized_sentence)):
                window = [cleaned_token for original_token, cleaned_token in tokenized_sentence[start_idx:end_idx] if cleaned_token != None]
                if len(window) == self.window_size:
                    print(window)
                    yield window
                    break

    def __update_graph(self, window):
        all_combinations = combinations(window,2)
        for source, target in all_combinations:
            self.add_node(source)
            self.add_node(target)
            self.add_edge(source, target)
            self.add_edge(target, source)

    def __update_word_count(self, window):
        self.counter.update(window)

    def __update_filtered_word_count(self, window):
        self.filtered_counter.update(window)

    def add_edge(self, source, target):
        if self.weighted_graph:
            self.edges[source][target] += 1
        else:
            self.edges[source][target] = 1

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 10 * random()

    def top(self):
        kw_sorted = sorted(self.nodes.items(), reverse=True, key=lambda x:x[1])
        kw = list(map(lambda x:x[0], kw_sorted))
        return kw[:self.num_keywords]

    def merge(self, keywords):
        merged_keywords = []
        for sentence in self.tokenized_sentences:
            kw = []
            sw = []
            for original_token, cleaned_token in sentence:
                if 2 < len(sw):
                    sw = list()
                elif 0 < len(kw) and cleaned_token is None:
                    sw.append(original_token)

                if cleaned_token in keywords:
                    kw.extend(sw)
                    sw = list()
                    kw.append(original_token)
                elif 0 < len(kw):
                    keyword = " ".join(kw)
                    keyword = keyword.lower()
                    merged_keywords.append(keyword)
                    kw = list()
                    sw = list()

        unique = [keyword for keyword in merged_keywords if not any([keyword in k2 for k2 in merged_keywords if keyword != k2])]
        unique = set(unique)
        return unique
                
    def get_heatmap(self,heatmap_type):
        heatmap = []
        max_value = max(self.nodes.values())
        for sentence in self.tokenized_sentences:
            if heatmap_type == "node_score":
                score_function = lambda x:(0.0001 + self.nodes[x])/max_value
            elif heatmap_type == "graph":
                score_function = lambda x: 0.2 if self.nodes[x] == 0.0 else 1.0
            sentence_map = [(original, score_function(cleaned)) for original, cleaned in sentence]
            heatmap.extend(sentence_map)
        return heatmap

    def get_word_frequencies(self):
        return self.counter.most_common(self.num_keywords)

    def get_filtered_word_frequencies(self):
        keywords = [word for word, count in self.filtered_counter.most_common(self.num_keywords)]
        if self.merge_keywords:
            keywords = self.merge(keywords)
        keywords = sorted(keywords)
        return keywords

def plot_graph(edges):
    graph = graphviz.Digraph()
    for source, targets in edges.items():
        for target, weight in targets.items():
            graph.edge(source, target)

    st.graphviz_chart(graph)

def visualize_heatmap(heatmap, titel):
    markdown_text = ""
    blue = colour.Color("#dfdfdf")
    colors = list(blue.range_to(colour.Color("black"),11))
    for token, score in heatmap:
        idx = int(score * 10)
        markdown_token = f"<span style=\"color:{colors[idx]}\">{token}</span>"
        markdown_text += " " + markdown_token
    st.markdown("## " + titel)
    st.markdown(markdown_text, unsafe_allow_html=True)

examples = {"News Example 1": """They stand only six points above the drop zone with a rejuvenated Jose Mourinho returning to """
                      """Old Trafford next followed by the Manchester derby at the weekend."""
                      """Three Premier League managers have already been sacked with Mauricio Pochettino waiting in the wings for a big job."""
                      """ But smiling Solskjaer is trying to remain positive believing in his long term plan to get the club back on track."""
                      """ Solskjaer said: “It's that time of year and it’s never nice to see your colleagues lose their jobs."""
                      """ “It doesn't make me more concerned, I'm just focusing on my job and that's doing as well as I can, look forward to the next game and looking long term and planning things with the board."""
                      """ “These next two games are a great chance to improve things.”"""
                      """Solskjaer actually needs to win those next two and Everton at home to have more points than Mourinho did after 17 games last season when The Special One became the sacked one."""
                      """Solskjaer said: “When you change managers halfway through the season it’s because a club isn't where it wants to be.""",
    "News Example 2": 
                 """“And 2020 will then be the year we finally put behind us the arguments and uncertainty over Brexit."""
                 """“We will get Parliament working on the people’s priorities — delivering 50,000 more nurses and 20,000 more police,"""
                 """creating millions more GP appointments, and taking urgent action on the cost of living."""
                 """“But if the Conservatives don’t get a majority, then on Friday 13th we will have the nightmare of a hung Parliament"""
                 """with Jeremy Corbyn as Prime"""
                 """Minister propped up by Nicola Sturgeon’s SNP."""
                  """“Next year will be Groundhog Day in Parliament with MPs arguing every day about the referendum and businesses and"""
                  """families left in limbo, unable to plan their futures.”"""
                  """'TRANSFORMATIVE'"""
                  """Meanwhile Chancellor Sajid Javid signalled there would be further tax cuts in subsequent Tory Budgets, telling the"""
                  """Spectator magazine: “Whenever we can cut tax and give people back more of their hard-earned cash, we will do that.”"""
                  """He also vowed to take advantage of historically low interest rates to borrow for major infrastructure investment"""
                  """projects. In a candid interview, Mr Javid admitted the Tories should have cashed in on used those low interest rates"""
                    """to borrow and spend more in recent years."""
                    """He pledged to turn on the spending taps if the Tories win a majority — but he insisted the splurge would be"""
                    """abandoned if interest rates rise again."""
                    """The Chancellor said: “For a variety of factors, I think rates are going to remain low for a long time. """,
    "TextRank Example":"""Compatibility of systems of linear constraints over the set of natural numbers. Criteria of """
                    """compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are """
                    """considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal """
                    """generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for """
                    """constructing a minimal supporting set of solutions can be used in solving all the considered  types systems and """
                    """systems of mixed types.""",
    "Paper Example 1": """We introduce a new type of deep contextualized word representation that models both (1) complex """
                   """characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts """
                    """(i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional """
                    """language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily """
                    """added to existing models and significantly improve the state of the art across six challenging NLP problems, """
                    """including question answering, textual entailment and sentiment analysis. We also present an analysis showing that """
                    """exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types """
                    """of semi-supervision signals.""",
    "Paper Example 2": """Natural language processing tasks, such as question answering, machine translation, reading """
                      """  comprehension,  and  summarization,  are  typically approached  with  supervised  learning on  task specific """
                        """datasets. We demonstrate that language models begin to learn these tasks without any explicit supervision when """
                        """trained on a new dataset of millions of webpages called WebText. When conditioned on a document plus questions, the """
                        """answers generated by the language model reach 55 F1 on the CoQA dataset - matching or exceeding the performance of 3 """
                        """out of 4 baseline systems without  using  the  127,000+  training  examples.The capacity of the language model is """
                        """essential to the success of zero-shot task transfer and increasing it improves performance in a log-linear fashion """
                        """across tasks. Our largest model, GPT-2,is a 1.5B parameter Transformer that achieves state of the art results on 7 """
                        """out of 8 tested language modeling datasets in a zero-shot setting but still underfits WebText.   Samples from """
                        """the model reflect these improvements and contain coherent paragraphs of text. These findings suggest a promising path """
                        """towards building language processing systems which learn to perform tasks from their naturally occurring """
                        """demonstrations."""}
 
def load_data(path):
    all_files = glob.glob(path + "/*.txt")
    corpus = dict()
    for filename in all_files:
        with open(filename, "r") as f:
            file_content = f.read()
        corpus[filename] = file_content
    return corpus

@st.cache
def download_and_process_corpus(corpus_name):
    dw = Datawarehouse()
    path = dw.download(corpus_name)
    return load_data(path)

# Textrank parameters

text_input= st.sidebar.selectbox("Choose text", ["News Example 1", "News Example 2", "Paper Example 1","Paper Example 2", "TextRank Example", "Seal Data", "Arxiv", "User Input"]) 
if text_input == "User Input":
    document = st.text_input("Custom text", "", "input_document")
elif text_input == "Seal Data":
    dw = Datawarehouse()
    corpus_name = st.sidebar.selectbox("Choose a corpus", list(dw.get_available_corpus()), 0)
    corpus = download_and_process_corpus(corpus_name)
    contract_id = st.sidebar.selectbox("Choose a contract", list(corpus.keys()))
    start_idx_percent = st.sidebar.slider("Start index", 0, 100, step=1, value=0)
    end_idx_percent = st.sidebar.slider("End index", 0, 100, step=1, value=100)
    document = corpus[contract_id]
    start_idx = int(start_idx_percent/100 * len(document))
    end_idx = int(end_idx_percent/100 * len(document))
    document = document[start_idx:end_idx]
elif text_input == "Arxiv":
    corpus = arxiv()
    paper_title = st.sidebar.selectbox("Choose a contract", list(corpus.keys()))
    document = corpus[paper_title]
    document = document.replace("<p>", "")
    document = document.replace("</p>", "")
else:
    document = examples[text_input]

window_size = st.sidebar.slider("Window Size", 2, 10, step=1, value=2)
iterations = st.sidebar.slider("iterations", 1, 50, step=1, value=20)
damping_factor = st.sidebar.slider("Damping Factor", 1, 100, step=1, value=85) /100
num_keywords = st.sidebar.slider("Keywords to Extract", 1, 40, step=1, value=5)
allowed_pos = st.sidebar.multiselect("Limit to POS", ["NOUN", "ADJ", "VERB", "PROPN", "PRON", "ADV", "INTJ",  "CCONJ", "DET","X"]) 
use_lemma = st.sidebar.radio("Use lemma", ["No", "Yes"]) == "Yes"
use_stopwords = st.sidebar.radio("Include stopwords", ["No", "Yes"]) == "Yes"
weighted_graph = st.sidebar.radio("Weighted edges", ["Yes", "No"]) == "Yes"
merge_keywords = st.sidebar.radio("Merge keywords", ["Yes", "No"]) == "Yes"

textrank = TextRank(window_size, iterations, num_keywords, allowed_pos, use_lemma, weighted_graph, use_stopwords, "token", merge_keywords,damping_factor)
keywords = textrank.process_document(document)

st.markdown("## TextRank Keywords")
st.table(keywords)

st.markdown("## Word Frequency")
word_count = textrank.get_word_frequencies()
st.table(word_count)

st.markdown("## Filtered Word Frequency")
filtered_word_count = textrank.get_filtered_word_frequencies()
st.table(filtered_word_count)


heatmap_node_score = textrank.get_heatmap("node_score")
visualize_heatmap(heatmap_node_score, "Word scores heatmap")
heatmap_graph = textrank.get_heatmap("graph")
visualize_heatmap(heatmap_graph, "Active Words")

show_graph = st.sidebar.radio("Show graph plot", ["No", "Yes"]) == "Yes"
if show_graph:
    plot_graph(textrank.edges)
