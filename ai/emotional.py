from transformers import pipeline


class Emotional:
    def __init__(self):
        self.model = pipeline("text-classification", "ai_models/seninoseno/rubert-base-cased-sentiment-study-feedbacks-solyanka", max_length=512)

    def predict(self, input) -> tuple[str, str]:
        group = self.model(input, truncation=True)[0]
        group_label = group['label']
        return group_label
