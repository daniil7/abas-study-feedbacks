from typing import Callable, Optional
from itertools import chain

from domain.sentiment_analysis import Sentiment
from domain.sentiaspect_evaluation import SentiAspect, AspectRating


def make_combined_sentiaspect_evaluator(
    absa_analyzer: Callable[[str], list[SentiAspect]],
    evaluation_strategy: Callable[[list[SentiAspect]], AspectRating]
) -> Callable[[list[str]], AspectRating]:
    def sentiaspect_evaluator(texts: list[str]) -> AspectRating:
        sentiaspects = list(chain.from_iterable(absa_analyzer(text) for text in texts))
        return evaluation_strategy(sentiaspects)
    return sentiaspect_evaluator
