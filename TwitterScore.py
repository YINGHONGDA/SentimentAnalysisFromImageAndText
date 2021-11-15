from nltk import word_tokenize
from nltk.tag import pos_tag
import nltk
import pandas as pd
from nltk.corpus import sentiwordnet as swn


def get_TwitterScore(text):
    sentence = text
    text = pos_tag([i for i in word_tokenize(str(text))])
    word_fq = nltk.FreqDist(text)
    wor_list = word_fq.most_common()

    key = []
    part = []
    frequency = []

    for i in range(len(wor_list)):
        key.append(wor_list[i][0][0])
        part.append(wor_list[i][0][1])
        frequency.append(wor_list[i][1])

    textdf = pd.DataFrame({'key': key,
                           'part': part,
                           'frequency': frequency
                           }, columns=['key', 'part', 'frequency'])

    n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']
    v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    a = ['JJ', 'JJR', 'JJS']
    r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']

    for i in range(len(textdf['key'])):
        z = textdf.iloc[i, 1]

        if z in n:
            textdf.iloc[i, 1] = 'n'
        elif z in v:
            textdf.iloc[i, 1] = 'v'
        elif z in a:
            textdf.iloc[i, 1] = 'a'
        elif z in r:
            textdf.iloc[i, 1] = 'r'
        else:
            textdf.iloc[i, 1] = ''

    x = []

    for i in range(len(textdf['key'])):
        m = list(swn.senti_synsets(textdf.iloc[i, 0], textdf.iloc[i, 1]))
        s = 0
        ra = 0
        if len(m) > 0:
            for j in range(len(m)):
                s += (m[j].pos_score() - m[j].neg_score()) / (j + 1) # give weight to word
                ra += 1 / (j + 1)
            x.append(s / ra)
        else:
            x.append(0)
    print(pd.concat([textdf, pd.DataFrame({'score': x})], axis=1))

    return sum(x)


if __name__ == '__main__':
    content = input("please input your tweet:")
    test = get_TwitterScore(content)
    print(test)
