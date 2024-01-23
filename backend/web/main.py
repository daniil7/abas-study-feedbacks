from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.web.settings import settings
from pydantic import BaseModel
from pydantic import Field

from ai.search import MethodSimilarity
from ai.emotional import Emotional

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

search = MethodSimilarity()
emotional = Emotional()

default_aspects = search.aspects_list

class Request(BaseModel):
    text: str = Field(description="Текст одного или нескольких отзывов.")
    aspects_list: list|None = Field(default=None, description="Опциональный параметр, список слов, отражающих тот или иной аспект.")

@app.post("/")
async def root(item: Request):
    """
    Маршрут, выполняющий поиск аспектов и оценку их эмоциональной окраски

    Формат входных данных:

    ```
    {
      "text": "<текст>" # Текст одного или нескольких отзывов.
      "aspects_list": [...] # Опциональный параметр, список слов, отражающих тот или иной аспект.
    }
    ```

    Формат выходных данных:

    ```
    {
      "<аспект>": { # строка, относящаяся к аспекту.
        "score": <число>, # общая оценка эмоциональной окраски данного аспекта во всех отзывах [0, 1].
        "positive": <число>, # количество фрагментов, содержащих упоминание аспекта, с положительной эмоциональной окраской.
        "negative": <число>, # количество фрагментов, содержащих упоминание аспекта, с негативной эмоциональной окраской.
        "neutral": <число>, # количество фрагментов, содержащих упоминание аспекта, с нейтральной эмоциональной окраской.
        "total": <число> # общее количество фрагментов, содержащих упоминание аспекта.
      },
      ...
    }
    ```
    """
    text = item.text
    if item.aspects_list:
        search.set_aspects(item.aspects_list)
    else:
        search.set_aspects(default_aspects)
    result = {}
    all_aspects = search.process(text, 0.3)
    for aspect, sentences in all_aspects.items():
        if len(sentences) == 0:
            continue
        result[aspect] = {"score": 0, "positive": 0, "negative": 0, "neutral": 0, "total": len(sentences)}
        for sentence in sentences:
            predict = emotional.predict(sentence)
            result[aspect][predict.lower()] += 1
        result[aspect]["score"] = result[aspect]["positive"] / result[aspect]["total"] - result[aspect]["negative"] / result[aspect]["total"]
    return result
