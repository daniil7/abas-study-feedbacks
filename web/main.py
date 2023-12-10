from fastapi import FastAPI
from pydantic import BaseModel

from search import MethodSimilarity
from emotional import Emotional

app = FastAPI()

search = MethodSimilarity()
emotional = Emotional()

class Item(BaseModel):
    text: str

@app.post("/")
async def root(item: Item):
    text = item.text
    result = {}
    all_aspects = search.process(text)
    for aspect, sentences in all_aspects.items():
        if len(sentences) == 0:
            continue
        result[aspect] = {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": len(sentences)}
        for sentence in sentences:
            predict = emotional.predict(sentence)
            result[aspect][predict.lower()] += 1
        result[aspect]["score"] = result[aspect]["positive"] / result[aspect]["total"] - result[aspect]["negative"] / result[aspect]["total"]
    return {"message": result}
