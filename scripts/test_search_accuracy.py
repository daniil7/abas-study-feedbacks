
from pathlib import Path

import pandas as pd

from ai.search import MethodSimilarity as SearchSimilarity, MethodSubstring as SearchSubstring

search_similarity = SearchSimilarity()
search_substring = SearchSubstring()

print("Initialized")

DATA_DIRECTORY = "stage_2_keywords_search/search_annotated/"

# TODO calc accuracy of search methods
