from konlpy.tag import Twitter

pos_tagger = Twitter()

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def read_raw_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        print('loading data')
        data = [line.split('\t') for line in f.read().splitlines()]

        print('pos tagging to token')
        data = [tokenize(row[1]) for row in data[1:]]
    return data

test=read_raw_data('ratings_train.txt')

import numpy as np
result=[]

for i in range(len(test)):
    a=[]
    for j in test[i]:
        a.append(j.split("/")[0])
    result.append(a)
    print(i)
