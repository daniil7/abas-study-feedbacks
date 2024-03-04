from typing import Callable, Optional

from domain.sentiment_analysis import Sentiment
from domain.sentiaspect_evaluation import SentiAspect, AspectRating


def make_sentiaspect_evaluator(
    segmentatizer: Callable[[str], list[str]],
    aspect_classifier: Callable[[str], Optional[str]],
    sentiment_analyzer: Callable[[str], Sentiment],
    evaluation_strategy: Callable[[list[SentiAspect]], AspectRating]
) -> Callable[[str], AspectRating]:
    def sentiaspect_evaluator(text: str) -> AspectRating:
        segments = segmentatizer(text)
        aspects = [ aspect_classifier(segment) for segment in segments ]
        sentiaspects = [
            (aspect, sentiment_analyzer(segment))
            for segment, aspect in zip(segments, aspects)
            if aspect is not None
        ]
        # sentiaspects = [
        #     (aspect_classifier(segment), sentiment_analyzer(segment))
        #     for segment in segments
        # ]
        return evaluation_strategy(sentiaspects)
    return sentiaspect_evaluator
