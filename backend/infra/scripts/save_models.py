import os

from transformers import pipeline, AutoModel
from sentence_transformers import SentenceTransformer

MODELS_PATH="saved_models"

MODELS_SENTIMENT = [
    "seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka", # emotional analysis
    "danil7/rubert-base-cased-absa-study-feedbacks", # absa analysis
]

MODELS_SENTENCE = [
    "sentence-transformers/distiluse-base-multilingual-cased-v2", # search tokens
    # "inkoziev/sbert_pq", # search tokens
]

# download transformers models
for model_name in MODELS_SENTIMENT:
    save_path = os.path.join(MODELS_PATH, model_name)

    pipeline(model=model_name).save_pretrained(save_path)

# download sentence transformers models
for model_name in MODELS_SENTENCE:
    save_path = os.path.join(MODELS_PATH, model_name)
    
    model = SentenceTransformer(model_name)
    model.save(save_path)
