from fastapi import FastAPI, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache
from typing import Annotated, Callable, Optional
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn


from ai.sentiaspect_evaluation import (
    make_sentiaspect_evaluator, 
    avg_evaluation_strategy,
    AspectRating
)
from ai.text_segmentation import sentence_segmentizer
from ai.aspect_classification import make_embeds_sim_classifier
from ai.sentiment_analysis import make_hf_sentiment_analyzer, Sentiment



app = FastAPI()

origins = [
    settings.frontend_url
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache
def get_embeddings_model() -> Callable[[str], Tensor]:
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    def encode(s: str) -> Tensor:
        return model.encode(s, convert_to_tensor=True) # type: ignore
    
    return encode

@lru_cache
def get_aspect_classifier(
    embeddings_model: Annotated[
        Callable[[str], Tensor],
        Depends(get_embeddings_model),
        "The function that returns sentence embeddings."
    ],
    aspect_labels: Annotated[list[str], Body(description="Список слов, отражающих тот или иной аспект.")]
) -> Callable[[str], Optional[str]]:
    return make_embeds_sim_classifier(aspect_labels, embeddings_model, 0.3, nn.CosineSimilarity())


@lru_cache
def get_sentiment_analyzer() -> Callable[[str], Sentiment]:
    return make_hf_sentiment_analyzer('cointegrated/rubert-tiny2-sentiment')


@lru_cache
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



@app.post("/")
async def root(
    text: Annotated[str, Body(description="Текст отзыва")],
    sentiaspect_evaluator: Annotated[
        Callable[[str], AspectRating],
        Depends(get_sentiaspect_evaluator),
    ]
) -> AspectRating:
    """
    Аспектно-сентиментный анализ отзыва.
    """
    return sentiaspect_evaluator(text)
