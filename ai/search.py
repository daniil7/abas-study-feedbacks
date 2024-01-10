import re
import pathlib

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
# from pymorphy2 import MorphAnalyzer

from ai.sentence_tokenizer import nltk_sentence_tokenizer as get_sentences

ASPECTS_FILE = "data/aspects.txt"

def load_aspects():
    if pathlib.Path(ASPECTS_FILE).exists():
        with open(ASPECTS_FILE, encoding="UTF-8") as f:
            aspects = f.read().split('\n')
    else:
        aspects = []
    return aspects

def clean_text(text: str):
    # Убираем ссылки
    clean = re.sub(u'(http|ftp|https):\/\/[^ ]+', '', text)
    # Убираем все неалфавитные символы кроме дефиса и апострофа
    clean = re.sub(u'[^а-я^А-Я^\w^\-^\']', ' ', clean)
    # Убираем тире
    clean = re.sub(u' - ', ' ', clean)
    # Убираем дубликаты пробелов
    clean = re.sub(u'\s+', ' ', clean)
    # Убираем пробелы в начале и в конце строки
    clean = clean.strip().lower()
    return clean


# class MethodSubstring():
# 
#     def set_aspects(self, aspects_list=None):
#         if aspects_list is not None:
#             self.aspects_list = aspects_list
#         else:
#             self.aspects_list = load_aspects()
# 
#     def __init__(self):
#         self.set_aspects(None)
# 
#     def find_aspects(self, text: str):
#         aspects = {}
#         for aspect in self.aspects_list:
#             aspects[aspect] = []
#         # Разделение на предложения
#         sentences = get_sentences(text)
#         # Очистка
#         sentences_words = []
#         for sentence in sentences:
#             words = clean_text(sentence).split(" ")
#             sentences_words.append(words)
#         # Лемматизация
#         morph = MorphAnalyzer()
#         for sentence_idx in range(len(sentences_words)):
#             lemmatized = [morph.parse(word)[0].normal_form for word in sentences_words[sentence_idx]]
#             for aspect in self.aspects_list:
#                 if aspect in lemmatized:
#                     aspects[aspect].append(sentences[sentence_idx])
#         return aspects
# 
#     def process(self, text: str):
#         return self.find_aspects(text)

class MethodSimilarity():

    def transformers_tokenizer(self, sentence: str) -> torch.Tensor:
        tokens = self.transformers_auto_tokenizer(sentence, return_tensors='pt')
        vector = self.transformers_model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0).numpy()

    def set_aspects(self, aspects_list=None):
        if aspects_list is not None:
            self.aspects_list = aspects_list
        else:
            self.aspects_list = load_aspects()

    def __init__(self, tokenizer="distiluse"):
        self.set_aspects(None)

        self.aspects_list = load_aspects()

        if tokenizer == "distiluse":
            self.tokenizer = self.transformers_tokenizer
            self.transformers_auto_tokenizer = AutoTokenizer.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
        elif tokenizer == "sbert-pq":
            self.tokenizer = self.transformers_tokenizer
            self.transformers_auto_tokenizer = AutoTokenizer.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
        else:
            self.tokenizer = None
            raise Exception("Invalid tokenizer.")

    def calc_similarity(self, vector1, vector2):
        return np.dot(vector1, vector2) / \
               (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def find_aspects(self, text: str, min_similarity: int):
        aspects = {}
        for aspect in self.aspects_list:
            aspects[aspect] = []
        # Разделение на предложения
        sentences = get_sentences(text)
        # Схожесть
        for sentence in sentences:
            sentence_vector = self.tokenizer(sentence)
            similarities = [(aspect, self.calc_similarity(self.tokenizer(aspect), sentence_vector))
                              for aspect in self.aspects_list]
            # similarities.sort(key=lambda x: x[1], reverse=True)
            # if similarities and similarities[0][1] > min_similarity:
            #     aspects[similarities[0][0]].append(sentence)
            for similarity in similarities:
                if similarity[1] > min_similarity:
                    aspects[similarity[0]].append(sentence)
        return aspects

    def process(self, text: str, min_similarity: int = 0.2):
        return self.find_aspects(text, min_similarity)
