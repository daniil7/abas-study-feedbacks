from transformers import pipeline
from typing import Callable

from ai.sentiment_analysis.sentiment import Sentiment


def make_hf_sentiment_analyzer(model_name: str) -> Callable[[str], Sentiment]:
    """
    Creates a HuggingFace AI model for sentiment analysing of given text.

    :param str model_name: An ai model that located in ai_modes/... in the form of <huggingface_username>/<model_name>
    :return: Sentiment classifier
    :rtype: Callable[[str], Sentiment]
    """
    model = pipeline("text-classification", model_name)
    def hf_sentiment_analyzer(text: str) -> Sentiment:
        result = model(text)[0]
        label = result['label']

        if label == 'NEGATIVE':
            return Sentiment.NEGATIVE
        if label == 'POSITIVE':
            return Sentiment.POSITIVE
        if label == 'NEUTRAL':
            return Sentiment.NEUTRAL
        raise ValueError(f"Unknown label: {label}")

    return hf_sentiment_analyzer
