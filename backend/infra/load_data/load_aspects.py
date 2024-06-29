import pathlib

ASPECTS_FILE = "data/aspects.txt"

def load_aspects():
    if pathlib.Path(ASPECTS_FILE).exists():
        with open(ASPECTS_FILE, encoding="UTF-8") as f:
            aspects = f.read().split('\n')
    else:
        aspects = []
    return list(filter(lambda s: len(s) > 0, aspects))
