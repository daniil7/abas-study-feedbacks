from statistics import mean
from itertools import groupby
from operator import itemgetter

from domain.sentiment_analysis import Sentiment
from domain.sentiaspect_evaluation import SentiAspect, AspectRating

def avg_evaluation_strategy(sentiaspects: list[SentiAspect]) -> AspectRating:

    def to_rating(sa: SentiAspect) -> float:
        _, sentiment = sa

        return {
            Sentiment.POSITIVE: 1., Sentiment.NEUTRAL: 0., Sentiment.NEGATIVE: -1.
        }[sentiment]

    key = itemgetter(0)
    sentiaspects.sort(key=key)

    return {
        key: mean(map(to_rating, group))
        for key, group in groupby(sentiaspects, key=key)
    }
