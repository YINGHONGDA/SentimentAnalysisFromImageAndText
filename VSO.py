def get_SentimentScore(image_classification):
    with open('/home/yinghongda/DataSet/VSO.txt','r') as f:
        line = f.readlines()
        vso = {}
        for i in line:
            i = i.strip()
            k = i.split(' ')[0]
            v = i.split(' ')[1]
            vso[k] = v
    score = vso[image_classification].strip('[]')
    score = score.split(':')[1]
    return float(score)

if __name__ == '__main__':
    test = get_SentimentScore('adorable_child')
    print(test)