from keybert import KeyBERT

def Content_Analysis(text):
    doc = text
    kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = kw_model.extract_keywords(doc,stop_words="english",keyphrase_ngram_range=(1,1))
    return keywords[0][0]

if __name__ == '__main__':
    test = Content_Analysis('i am too fat. if i do not control eating, i will become unhealthy. i hate that.')
    print(test)