from os import listdir
from os.path import isfile, join
from pathlib import Path

import re
import pathlib

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pymorphy2 import MorphAnalyzer

# Подгружаем все размеченные аспекты из файлов и сохраняем в памяти

DATA_DIRECTORY = "stage_2_keywords_search/search_annotated/"

# Предварительно удаляем расщирение .txt
aspect_file_names = [
    f[:-4]
        for f
        in listdir(DATA_DIRECTORY)
        if isfile(join(DATA_DIRECTORY, f))
]

# Объект аспект - массив предложений
actual_aspects_sentences = {}
# Список всех уникальных предложений
all_aspects_senteces = []
for file_name in aspect_file_names:
    if file_name != "мусор":
        with open(DATA_DIRECTORY+file_name+".txt", "r", encoding="utf-8") as f:
            actual_aspects_sentences[file_name] = f.read().split('\n')
            all_aspects_senteces += actual_aspects_sentences[file_name]
all_aspects_senteces = list(set(all_aspects_senteces))

# Если правильно отнесли +1, неправильно отнесли/не отнесли когда требовалось -1
def calc_acc(aspects: list, sentence: str) -> int:
    result = 0
    for aspect, sentences in actual_aspects_sentences.items():
        if aspect in aspects and sentence in sentences:
            result += 1
        if (aspect not in aspects) ^ (sentence not in sentences): # xor
            result -= 1
    return result

def load_aspects():
    return aspect_file_names

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

# TODO сделать поддержку синонимов

class MethodSubstring():

    def set_aspects(self, aspects_list=None):
        if aspects_list is not None:
            self.aspects_list = aspects_list
        else:
            self.aspects_list = load_aspects()

    def __init__(self):
        self.set_aspects(None)

    def find_aspects(self, sentence: str):
        aspects = []
        # Очистка
        sentence_words = clean_text(sentence).split(" ")
        # Лемматизация
        morph = MorphAnalyzer()
        lemmatized = [morph.parse(word)[0].normal_form for word in sentence_words]
        for aspect in self.aspects_list:
            if aspect in lemmatized:
                aspects.append(aspect)
        return aspects

    def process(self, text: str):
        return self.find_aspects(text)

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

    def find_aspects(self, sentence: str, min_similarity: int):
        aspects = []
        # Схожесть
        sentence_vector = self.tokenizer(sentence)
        similarities = [(aspect, self.calc_similarity(self.tokenizer(aspect), sentence_vector))
                          for aspect in self.aspects_list]
        for similarity in similarities:
            if similarity[1] > min_similarity:
                aspects.append(similarity[0])
        return aspects

    def process(self, text: str, min_similarity: int = 0.4):
        return self.find_aspects(text, min_similarity)


search_substring = MethodSubstring()
substring_acc_list = []

search_similarity = MethodSubstring()
similarity_acc_list = []

print("Initialized")

for sentence in all_aspects_senteces:
    aspects = search_substring.process(sentence)
    substring_acc_list.append(calc_acc(aspects, sentence))

print(substring_acc_list)
