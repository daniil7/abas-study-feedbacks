import re

import torch
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModel

def load_aspects():
    with open("data/aspects.txt") as f:
        aspects = f.read().split('\n')
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

def get_sentences(text: str):
    # Токенизация
    return nltk.sent_tokenize(text, language="russian")


class MethodSubstring():

    def __init__(self):
        self.aspects_list = load_aspects()

    def find_aspects(self, text: str):
        aspects = {}
        for aspect in self.aspects_list:
            aspects[aspect] = []
        # Разделение на предложения
        sentences = get_sentences(text)
        # Очистка
        sentences_words = []
        for sentence in sentences:
            words = clean_text(sentence).split(" ")
            words = [token for token in words if token not in stopwords]
            sentences_words.append(words)
        # Лемматизация
        morph = MorphAnalyzer()
        for sentence_idx in range(len(sentences_words)):
            lemmatized = [morph.parse(word)[0].normal_form for word in sentences_words[sentence_idx]]
            for aspect in self.aspects_list:
                if aspect in lemmatized:
                    aspects[aspect].append(sentences[sentence_idx])
        return aspects

    def process(self, text: str):
        return self.find_aspects(text)

class MethodSimilarity():

    def transformers_tokenizer(self, sentence: str) -> torch.Tensor:
        tokens = self.transformers_tokenizer(sentence, return_tensors='pt')
        vector = self.transformers_model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0).numpy()

    def __init__(self, tokenizer="distiluse"):

        self.aspects_list = load_aspects()

        if tokenizer == "distiluse": # Хорошо
            self.tokenizer = self.transformers_tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/sentence-transformers/distiluse-base-multilingual-cased-v2", local_files_only=True)
        elif tokenizer == "sbert-pq": # Средне
            self.tokenizer = self.transformers_tokenizer
            self.transformers_tokenizer = AutoTokenizer.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
            self.transformers_model = AutoModel.from_pretrained("ai_models/inkoziev/sbert_pq", local_files_only=True)
        else:
            self.tokenizer = None
            raise Exception("Invalid tokenizer. Available: spacy, rubert")

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
            similarities.sort(key=lambda x: x[1], reverse=True)
            if similarities[0][1] > min_similarity:
                aspects[similarities[0][0]].append(sentence)
        return aspects

    def process(self, text: str, min_similarity: int = 0.2):
        return self.find_aspects(text, min_similarity)
