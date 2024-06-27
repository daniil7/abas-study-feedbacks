import math
import random
from operator import itemgetter
from itertools import groupby
from typing import Optional
from sentence_transformers.SentenceTransformer import Callable
from domain.sentiaspect_evaluation.sentiaspect_evaluator import make_sentiaspect_evaluator
from domain.sentiment_analysis.sentiment import Sentiment
from domain.sentiaspect_evaluation import SentiAspect, AspectRating

def dummy_segmentizer(text: str):
    batch_size = 10
    num_splits = math.floor(len(text) / batch_size)
    if num_splits > 0:
        return [ text[i * batch_size : i + 1 * batch_size] for i in range(num_splits-1) ]
    return [ text ]

def make_dummy_classifier(classes = list[str]) -> Callable[[str], Optional[str]]:
    def dummy_classifier(_: str) -> Optional[str]:
        if random.random() < 0.2:
            return None
        return random.choice(classes)
    return dummy_classifier

def dummy_sentiment_analyzer(_: str) -> Sentiment:
    return random.choice([Sentiment.NEUTRAL, Sentiment.NEGATIVE, Sentiment.POSITIVE])

def dummy_evaluation_strategy(sentiaspects: list[SentiAspect]) -> AspectRating:
    key = itemgetter(0)
    sentiaspects.sort(key=key)
    return {
        key: random.random()
        for key, _ in groupby(sentiaspects, key=key)
    }

def test_sentiaspect_evaluator():

    evaluator = make_sentiaspect_evaluator(
        segmentatizer=dummy_segmentizer,
        aspect_classifier=make_dummy_classifier(
            classes=["груша", "яблоко", "банан"],
        ),
        sentiment_analyzer=dummy_sentiment_analyzer,
        evaluation_strategy=dummy_evaluation_strategy
    )

    result = evaluator("Пробное предложение номер один. Пробное предложение номер два. Пробное предложение номер три.")

    assert isinstance(result, dict)
