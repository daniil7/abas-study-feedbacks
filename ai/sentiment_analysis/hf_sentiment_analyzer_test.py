from typing import Callable

from ai.sentiment_analysis.hf_sentiment_analyzer import make_hf_sentiment_analyzer
from ai.sentiment_analysis import Sentiment

def test_hf_sentiment_analyzer():

    analyzer = make_hf_sentiment_analyzer('seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka')

    assert callable(analyzer)

    assert isinstance(analyzer('Электив понравился.'), Sentiment)
    assert isinstance(analyzer('Учились блазонированию (на специальном геральдическом языке словесно описывали, как выглядит герб).'), Sentiment)
    assert isinstance(analyzer('Изучали историю гербов, геральдики, геральдические фигуры и их значение.'), Sentiment)
    assert isinstance(analyzer('Было очень интересно!'), Sentiment)
