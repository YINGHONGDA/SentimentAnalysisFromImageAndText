from sentence_transformers import SentenceTransformer,util

def get_Relatedness(text,image):
    model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    embedding1 = model.encode(text)
    embedding2 = model.encode(image)
    cosine_scores = util.pytorch_cos_sim(embedding1,embedding2)

    return cosine_scores.item()

if __name__ == '__main__':
    test = get_Relatedness("unhealthy","flower")
    print(test)