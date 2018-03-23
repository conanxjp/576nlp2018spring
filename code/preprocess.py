import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from nltk import (word_tokenize, pos_tag)
from nltk.corpus import (sentiwordnet as swn, words)
from nltk.metrics.distance import (edit_distance, jaccard_distance)
import re
from tqdm import tqdm
import config as cf

cf.configure('2014', 'rest', 'glove', 'test')
dataPath = cf.ROOT_PATH + cf.DATA_PATH + cf.DATA_FILE
embeddingPath = cf.ROOT_PATH + cf.DATA_PATH + cf.WORD2VEC_FILE

def parse2014(filepath):
    """
    parse 2014 raw data in xml format
    only tested for restaurant data
    """
    data = pd.DataFrame(columns = ['id', 'text', 'aspect', 'polarity'])
    # no good way to pick terms to corresponding aspect term yet
    aspectTerm_dict = {
                        'food': [],
                        'service': [],
                        'price': [],
                        'ambience': [],
                        'anecdotes/miscellaneous': []
                      }
    tree = et.parse(filepath)
    root = tree.getroot()
    sentences = root.findall('sentence');
    i = 0
    for sentence in tqdm(sentences):
        id = sentence.attrib.get('id')
        text = sentence.find('text').text
        # TODO categorize term words/phrases into aspect terms
        # aspectTerms = child.find('aspectTerms')
        # if aspectTerms != None:
        #     for term in aspectTerms.findall('aspectTerm'):
        #         terms.append(term.attrib.get('term'))
        for category in sentence.find('aspectCategories').findall('aspectCategory'):
            data.loc[i] = [id, text, category.attrib.get('category'), category.attrib.get('polarity')]
            i = i + 1
    return data

def tokenize(data):
    wordData = []
    for s in data:
        wordData.append([w for w in word_tokenize(s.lower())])
    return wordData

def cleanup(wordData):
    return wordData

# def correctSpell(data):
#
# def createVocabulary(data):

def split(data):
    words = []
    for str in data:
        words = np.append(words, str.split())
    return sorted(set(words))

def filterWordDict(words, dictpath):
    dictVoc = []
    newDict = []
    with open(dictpath) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            if word in words:
                dictVoc.append(word)
                newDict.append(line)
    f.close()
    unknownWords = [word for word in words if word not in dictVoc]
    with open('test.txt', 'w+') as f:
        for line in newDict:
            f.write(line)
    with open('unknown.txt', 'w+') as f:
        for word in unknownWords:
            f.write(word + '\n')

def loadWordVec(filepath):
    voc = []
    i = 0
    with open(filepath) as f:
        for line in f:
            values = line.split()
            voc.append(values[0])
            i = i + 1
            vector = np.array(values[1:], dtype = 'float32')
    print(len(voc))
    f.close()

data = parse2014(dataPath)
# words = cleanup(data['text'])
for word in sorted(set(tokenize(data['text']))):
    print(word)


# loadWordVec(embeddingPath)
# filterWordDict(words, embeddingPath)
