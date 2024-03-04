from transformers import pipeline
from typing import Callable

from ai.sentiment_analysis.sentiment import Sentiment


def make_hf_sentiment_analyzer(model_name: str) -> Callable[[str], Sentiment]:
    model = pipeline("text-classification", model_name)
    def hf_sentiment_analyzer(text: str) -> Sentiment:
        result = model(text)[0]
        label = result['label']

        if label == 'NEGATIVE':
            return Sentiment.NEGATIVE
        elif label == 'POSITIVE':
            return Sentiment.POSITIVE
        elif label == 'NEUTRAL':
            return Sentiment.NEUTRAL
        else:
            raise ValueError(f"Unknown label: {label}")

    return hf_sentiment_analyzer
