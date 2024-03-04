from typing import Callable

from ai.sentiment_analysis import Sentiment
from ai.sentiaspect_evaluation import SentiAspect, AspectRating


def make_sentiaspect_evaluator(
    segmentatizer: Callable[[str], list[str]],
    aspect_classifier: Callable[[str], str],
    sentiment_analyzer: Callable[[str], Sentiment],
    evaluation_strategy: Callable[[list[SentiAspect]], AspectRating]
) -> Callable[[str], AspectRating]:
    def sentiaspect_evaluator(text: str) -> AspectRating:
        segments = segmentatizer(text)
        sentiaspects = [
            (aspect_classifier(segment), sentiment_analyzer(segment))
            for segment in segments
        ]
        return evaluation_strategy(sentiaspects)
    return sentiaspect_evaluator
