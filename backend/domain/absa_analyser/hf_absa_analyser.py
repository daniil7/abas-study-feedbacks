import os
from transformers import pipeline
from typing import Callable

from domain.sentiaspect_evaluation.sentiaspect import SentiAspect
from domain.sentiment_analysis.sentiment import Sentiment


def make_hf_absa_analyzer(model_name: str, aspects: list[str]) -> Callable[[str], list[SentiAspect]]:
    """
    Creates a HuggingFace AI model for sentiment analysing of given text.

    :param str model_name: An domain model that located in ai_modes/... in the form of <huggingface_username>/<model_name>
    :return: Sentiment classifier
    :rtype: Callable[[str], Sentiment]
    """
    model = pipeline("text-classification", os.path.join("saved_models", model_name))
    def hf_absa_analyzer(text: str) -> list[SentiAspect]:

        sentiaspects = list()

        for aspect in aspects:
            result = model({'text': text, 'text_pair': aspect})
            label = result['label']
            print(text, aspect, label)
            if label == 'NOT-PRESENT':
                continue
            elif label == 'NEGATIVE':
                sentiaspects.append((aspect, Sentiment.NEGATIVE))
            elif label == 'POSITIVE':
                sentiaspects.append((aspect, Sentiment.POSITIVE))
            elif label == 'NEUTRAL':
                sentiaspects.append((aspect, Sentiment.NEUTRAL))
            else:
                raise ValueError(f"Unknown label: {label}")

        return sentiaspects

    return hf_absa_analyzer
