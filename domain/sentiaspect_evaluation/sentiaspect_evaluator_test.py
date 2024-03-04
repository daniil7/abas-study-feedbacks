import torch
import torch.nn.functional as F
from torch import Tensor
from domain.sentiaspect_evaluation import evaluation_strategies

from domain.sentiaspect_evaluation.sentiaspect_evaluator import make_sentiaspect_evaluator
from domain.aspect_classification.embeds_sim_classifier import make_embeds_sim_classifier
from domain.sentiment_analysis.hf_sentiment_analyzer import make_hf_sentiment_analyzer
from domain.sentiaspect_evaluation.evaluation_strategies.avg_evaluation_strategy import avg_evaluation_strategy
from domain.text_segmentation.sentence_segmentizer import sentence_segmentizer
from domain.sentiaspect_evaluation.aspect_rating import AspectRating

def dummy_embeddings_model(_: str) -> Tensor:
    return torch.rand(10)

def cosine_similarity_metric(emb1: Tensor, emb2: Tensor) -> float:
    return F.cosine_similarity(emb1, emb2, dim=0, eps=1e-8)

def test_sentiaspect_evaluator():

    evaluator = make_sentiaspect_evaluator(
        segmentatizer=sentence_segmentizer,
        aspect_classifier=make_embeds_sim_classifier(
            classes=["груша", "яблоко", "банан"],
            embeddings_model=dummy_embeddings_model,
            sim_threshold=0.8,
            sim_metric=cosine_similarity_metric,
        ),
        sentiment_analyzer=make_hf_sentiment_analyzer("seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka"),
        evaluation_strategy=avg_evaluation_strategy
    )

    result = evaluator("Пробное предложение номер один. Пробное предложение номер два. Пробное предложение номер три.")

    assert isinstance(result, dict)
