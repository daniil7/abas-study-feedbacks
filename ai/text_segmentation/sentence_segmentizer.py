from razdel import sentenize

def sentence_segmentizer(text: str) -> list[str]:
    """
    Split plain text on sentences

    :param str text: String of plain text
    :return: List of sentences
    :rtype: list of str
    """
    return [
        _.text for _ in sentenize(text)
    ]
