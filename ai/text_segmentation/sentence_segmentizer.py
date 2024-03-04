from razdel import sentenize

def sentence_segmentizer(text: str) -> list[str]:
    return [
        _.text for _ in sentenize(text)
    ]
