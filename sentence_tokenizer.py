import nltk

def nltk_sentence_tokenizer(text: str):
    # Токенизация
    return nltk.sent_tokenize(text, language="russian")
