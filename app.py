from itertools import combinations
from random import random
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import graphviz
import feedparser
import urllib
import colour
import spacy
import streamlit as st


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

class TextRank:
    def __init__(self, window_size, num_keywords, allowed_pos, use_lemma, weighted_graph, use_stopwords, merge_keywords):
        self.nlp = spacy.load("en_core_web_sm")
        self.window_size = window_size
        self.edges = defaultdict(lambda : defaultdict(int))
        self.nodes = defaultdict(float)
        self.num_keywords = num_keywords
        self.allowed_pos = allowed_pos
        self.use_lemma = use_lemma
        self.weighted_graph = weighted_graph
        self.use_stopwords = use_stopwords
        self.merge_keywords = merge_keywords
        self.iterations = 10

    def process_document(self, document):
        doc = self.nlp(document)
        self.tokenized_sentences = []
        for sentence in doc.sents:
            tokenized_sentence = self.__tokenize_sentence(sentence)
            self.tokenized_sentences.append(tokenized_sentence)

            for window in self.get_window(tokenized_sentence):
                self.__update_graph(window)
        self.loop()

    def loop(self):
        d = 0.85
        key_to_follow =list(self.nodes.keys())[1]
        for _ in range(self.iterations):
            for node in self.nodes.keys():
                score = self.calculate_score(node)
                self.nodes[node] = (1 - d) + d * score


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
                    yield window
                    break

    def __update_graph(self, window):
        all_combinations = combinations(window,2)
        for source, target in all_combinations:
            self.__add_node(source)
            self.__add_node(target)
            self.__add_edge(source, target)
            self.__add_edge(target, source)

    def __update_word_count(self, window):
        self.counter.update(window)

    def __update_filtered_word_count(self, window):
        self.filtered_counter.update(window)

    def __add_edge(self, source, target):
        if self.weighted_graph:
            self.edges[source][target] += 1
        else:
            self.edges[source][target] = 1

    def __add_node(self, node):
        if node not in self.nodes:
            self.nodes[node] = 10 * random()

    def get_keywords(self):
        keywords = self.__top_keywords()
        if self.merge_keywords:
            keywords= self.__merge(keywords)
        keywords = sorted(keywords)
        return keywords

    def __top_keywords(self):
        kw_sorted = sorted(self.nodes.items(), reverse=True, key=lambda x:x[1])
        kw = list(map(lambda x:x[0], kw_sorted))
        return kw[:self.num_keywords]

    def __merge(self, keywords):
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



example =  """Compatibility of systems of linear constraints over the set of natural numbers. Criteria of """ \
            """compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are """\
            """considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal """ \
            """generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for """ \
            """constructing a minimal supporting set of solutions can be used in solving all the considered  types systems and """ \
            """systems of mixed types."""

text_input= st.sidebar.selectbox("Choose text", ["TextRank Example", "Arxiv", "User Input"]) 
if text_input == "User Input":
    document = st.text_input("Custom text", "", "input_document")
elif text_input == "Arxiv":
    corpus = arxiv()
    paper_title = st.sidebar.selectbox("Choose a contract", list(corpus.keys()))
    document = corpus[paper_title]
    document = document.replace("<p>", "")
    document = document.replace("</p>", "")
else:
    document = example

window_size = st.sidebar.slider("Window Size", 2, 10, step=1, value=2)
num_keywords = st.sidebar.slider("Keywords to Extract", 1, 40, step=1, value=5)
allowed_pos = st.sidebar.multiselect("Limit to POS", ["NOUN", "ADJ", "VERB", "PROPN", "PRON", "ADV", "INTJ",  "CCONJ", "DET","X"]) 
use_lemma = st.sidebar.radio("Use lemma", ["No", "Yes"]) == "Yes"
use_stopwords = st.sidebar.radio("Include stopwords", ["No", "Yes"]) == "Yes"
weighted_graph = st.sidebar.radio("Weighted edges", ["Yes", "No"]) == "Yes"
merge_keywords = st.sidebar.radio("Merge keywords", ["Yes", "No"]) == "Yes"

textrank = TextRank(window_size, num_keywords, allowed_pos, use_lemma, weighted_graph, use_stopwords, merge_keywords)
textrank.process_document(document)
keywords = textrank.get_keywords()

st.markdown("## TextRank Keywords")
st.table(keywords)

heatmap_node_score = textrank.get_heatmap("node_score")
visualize_heatmap(heatmap_node_score, "Word scores heatmap")
heatmap_graph = textrank.get_heatmap("graph")
visualize_heatmap(heatmap_graph, "Active Words")

show_graph = st.sidebar.radio("Show graph plot", ["No", "Yes"]) == "Yes"
if show_graph:
    plot_graph(textrank.edges)
