from fastapi import FastAPI
from pydantic import BaseModel

from ai.search import MethodSimilarity
from ai.emotional import Emotional

app = FastAPI()

search = MethodSimilarity()
emotional = Emotional()

default_aspects = search.aspects_list

class Item(BaseModel):
    text: str
    aspects_list: list|None = None

@app.post("/")
async def root(item: Item):
    text = item.text
    if item.aspects_list:
        search.set_aspects(item.aspects_list)
    else:
        search.set_aspects(default_aspects)
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
