from transformers import pipeline
from sentence_transformers import SentenceTransformer

MODELS_PATH="ai_models/"

MODELS = [
    "seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka", # emotional analysis
]

MODELS_SENTENCE = [
    "sentence-transformers/distiluse-base-multilingual-cased-v2", # search tokens
    "inkoziev/sbert_pq", # search tokens
]

# download transformers models
for model in MODELS:
    pipeline(model=model).save_pretrained(MODELS_PATH+model)

# download sentence transformers models
for model_name in MODELS_SENTENCE:
    model = SentenceTransformer(model_name)
    model.save(MODELS_PATH+model_name)
