import secrets

import torch
import torch.nn.functional as F
from torch import Tensor

from domain.aspect_classification.embeds_sim_classifier import make_embeds_sim_classifier

def dummy_embeddings_model(_: str) -> Tensor:
    return torch.rand(10)

def cosine_similarity_metric(emb1: Tensor, emb2: Tensor) -> float:
    return F.cosine_similarity(emb1, emb2, dim=0, eps=1e-8)

def test_embeds_sim_classifier():
    classes = ["class1", "class2", "class3"]
    sim_threshold = 0.8
    classifier = make_embeds_sim_classifier(
        classes=classes,
        embeddings_model=dummy_embeddings_model,
        sim_threshold=sim_threshold,
        sim_metric=cosine_similarity_metric
    )
    assert callable(classifier)
    for _ in range(20):
        result = classifier(secrets.token_hex(16))
        assert (isinstance(result, str) and result in classes) or result is None
