from domain.sentiaspect_evaluation.evaluation_strategies.avg_evaluation_strategy import avg_evaluation_strategy
from domain.sentiaspect_evaluation.sentiaspect import SentiAspect
from domain.sentiment_analysis.sentiment import Sentiment


def test_avg_evaluation_strategy():
    result = avg_evaluation_strategy([
        ('аспект1', Sentiment.POSITIVE),
        ('аспект1', Sentiment.POSITIVE),
        ('аспект1', Sentiment.NEGATIVE),
        ('аспект1', Sentiment.NEUTRAL),
        ('аспект2', Sentiment.NEUTRAL),
        ('аспект3', Sentiment.NEGATIVE),
    ])
    assert abs(0.25 - result['аспект1']) < 1e-8
    assert abs(0.0 - result['аспект2']) < 1e-8
    assert abs(-1.0 -result['аспект3']) < 1e-8
