from typing import Callable, Optional
from itertools import chain

from domain.sentiment_analysis import Sentiment
from domain.sentiaspect_evaluation import SentiAspect, AspectRating


def make_sentiaspect_evaluator(
    segmentizer: Callable[[str], list[str]],
    aspect_classifier: Callable[[str], Optional[str]],
    sentiment_analyzer: Callable[[str], Sentiment],
    evaluation_strategy: Callable[[list[SentiAspect]], AspectRating]
) -> Callable[[list[str]], AspectRating]:
    def sentiaspect_evaluator(texts: list[str]) -> AspectRating:
        segments = list(chain.from_iterable([segmentizer(text) for text in texts]))
        aspects = [ aspect_classifier(segment) for segment in segments ]
        sentiaspects = [
            (aspect, sentiment_analyzer(segment))
            for segment, aspect in zip(segments, aspects)
            if aspect is not None
        ]
        return evaluation_strategy(sentiaspects)
    return sentiaspect_evaluator
