from functools import lru_cache
from typing import Annotated, Callable, Optional
from fastapi import Depends, Body
from torch import Tensor, nn
from sentence_transformers import SentenceTransformer

from domain.sentiaspect_evaluation import (
    make_sentiaspect_evaluator,
    avg_evaluation_strategy,
    AspectRating
)

from domain.text_segmentation import sentence_segmentizer
from domain.aspect_classification import make_embeds_sim_classifier
from domain.sentiment_analysis import make_hf_sentiment_analyzer, Sentiment

# Cache the models to avoid reinitialization on every request 
@lru_cache
def get_embeddings_model() -> Callable[[str], Tensor]:
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    def encode(s: str) -> Tensor:
        return model.encode(s, convert_to_tensor=True) # type: ignore
    return encode

@lru_cache
def get_sentiment_analyzer() -> Callable[[str], Sentiment]:
    return make_hf_sentiment_analyzer('seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka')

def get_aspect_classifier(
    embeddings_model: Annotated[
        Callable[[str], Tensor],
        Depends(get_embeddings_model),
    ],
    aspect_labels: Annotated[list[str], Body(description="Список слов, отражающих тот или иной аспект.")]
) -> Callable[[str], Optional[str]]:
    return make_embeds_sim_classifier(aspect_labels, embeddings_model, 0.3, nn.CosineSimilarity())


def get_sentiaspect_evaluator(
    aspect_classifier: Annotated[
        Callable[[str], str],
        Depends(get_aspect_classifier),
    ],
    sentiment_analyzer: Annotated[
        Callable[[str], Sentiment],
        Depends(get_sentiment_analyzer),
    ]
) -> Callable[[str], AspectRating]:
    return make_sentiaspect_evaluator(
        sentence_segmentizer, aspect_classifier, sentiment_analyzer, avg_evaluation_strategy
    )
