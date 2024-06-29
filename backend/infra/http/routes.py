from fastapi import APIRouter, Depends, Body
from typing import Annotated, Callable

from domain.sentiaspect_evaluation import AspectRating
from infra.load_data.load_aspects import load_aspects

from infra.http.dependencies import get_sentiaspect_evaluator

router = APIRouter()

@router.get("/aspects")
async def get_aspects() -> list[str]:
    """
    Предустановленный список аспектов
    """
    return load_aspects()

@router.post("/")
async def root(
    texts: Annotated[list[str], Body(description="Текст отзыва")],
    sentiaspect_evaluator: Annotated[
        Callable[[list[str]], AspectRating],
        Depends(get_sentiaspect_evaluator),
    ]
) -> AspectRating:
    """
    Аспектно-сентиментный анализ отзыва.
    """
    return sentiaspect_evaluator(texts)
