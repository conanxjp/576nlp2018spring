import xml.etree.ElementTree as et
import numpy as np
import pandas as pd
from nltk import (word_tokenize, pos_tag)
from nltk.corpus import sentiwordnet as swn
from nltk.metrics import edit_distance
import hunspell
import re
from tqdm import tqdm
import config as cf

cf.configure('2014', 'rest', 'glove', 'train')
dataPath = cf.ROOT_PATH + cf.DATA_PATH + cf.DATA_FILE
embeddingPath = cf.ROOT_PATH + cf.DATA_PATH + cf.WORD2VEC_FILE
# loading dictionaries for hunspell is a bit wierd, you have to put the dictionaries
# in a root-derivative folder path e.g. a folder ~/some-other-path is not allowed
hobj = hunspell.HunSpell(cf.HUNSPELL_PATH + cf.HUNSPELL_DICT[0],
                         cf.HUNSPELL_PATH + cf.HUNSPELL_DICT[1])

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
    writeCSV(data, cf.ROOT_PATH + cf.DATA_PATH + 'rest_train_2014_raw.csv')
    return data

def writeCSV(dataframe, filepath):
    dataframe.to_csv(filepath)

def tokenize(data):
    wordData = []
    for s in data:
        wordData.append([w for w in word_tokenize(s.lower())])
    return wordData

def cleanup(wordData):
    dictionary = embeddingDict(embeddingPath)
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, correctDashWord)
    wordData = cleanOp(wordData, re.compile(r'-'), dictionary, cleanDashWord)
    wordData = cleanOp(wordData, re.compile(r':'), dictionary, parseTime)
    wordData = cleanOp(wordData, re.compile('\+'), dictionary, parsePlus)
    wordData = cleanOp(wordData, re.compile(r'\d+'), dictionary, parseNumber)
    wordData = cleanOp(wordData, re.compile(r''), dictionary, correctSpell)
    return wordData

def cleanOp(wordData, regex, dictionary, op):
    for i, sentence in enumerate(wordData):
        if bool(regex.search(sentence)):
            newSentence = ''
            for word in word_tokenize(sentence.lower()):
                if bool(regex.search(word)) and word not in dictionary:
                    word = op(word)
                newSentence = newSentence + ' ' + word
            wordData[i] = newSentence
    return wordData

def parseTime(word):
    time_re = re.compile(r'^(([01]?\d|2[0-3]):([0-5]\d)|24:00)(pm|am)?$')
    if not bool(time_re.match(word)):
        return word
    else:
        dawn_re = re.compile(r'0?[234]:(\d{2})(am)?$')
        earlyMorning_re = re.compile(r'0?[56]:(\d{2})(am)?$')
        morning_re = re.compile(r'((0?[789])|(10)):(\d{2})(am)?$')
        noon_re = re.compile(r'((11):(\d{2})(am)?)|(((0?[01])|(12)):(\d{2})pm)$')
        afternoon_re = re.compile(r'((0?[2345]):(\d{2})pm)|((1[4567]):(\d{2}))$')
        evening_re = re.compile(r'((0?[678]):(\d{2})pm)|(((1[89])|20):(\d{2}))$')
        night_re = re.compile(r'(((0?9)|10):(\d{2})pm)|((2[12]):(\d{2}))$')
        midnight_re = re.compile(r'(((0?[01])|12):(\d{2})am)|(0?[01]:(\d{2}))|(11:(\d{2})pm)|(2[34]:(\d{2}))$')
        if bool(noon_re.match(word)):
            return 'noon'
        elif bool(evening_re.match(word)):
            return 'evening'
        elif bool(morning_re.match(word)):
            return 'morning'
        elif bool(earlyMorning_re.match(word)):
            return 'early morning'
        elif bool(night_re.match(word)):
            return 'night'
        elif bool(midnight_re.match(word)):
            return 'midnight'
        elif bool(dawb_re.match(word)):
            return 'dawn'
        else:
            return word

def parsePlus(word):
    return re.sub('\+', ' +', word)

def parseNumber(word):
    if bool(re.search(r'\d+', word)):
        return word
    else:
        search = re.search(r'\d+', word)
        pos = search.start()
        num = search.group()
        return word[:pos] + ' %s ' % num + parseNumber(word[pos+len(num):])

# def translateSymbol(word):

def checkSpell(word):
    global hobj
    return hobj.spell(word)

def correctSpell(word):
    global hobj
    suggestions = hobj.suggest(word)
    if len(suggestions) != 0:
        distance = [edit_distance(word, s) for s in suggestions]
        return suggestions[distance.index(min(distance))]
    else:
        return word

def createVocabulary(wordData):
    words = sorted(set([word for l in wordData for word in l.split(' ')]))
    global embeddingPath
    vocabulary = filterWordEmbedding(words, embeddingPath)
    return vocabulary

def splitDashWord(word):
    if '-' not in word:
        return [word]
    else:
        return word.split('-')

def cleanDashWord(word):
    return ''.join([s + ' ' for s in word.split('-')])

def correctDashWord(word):
    splittedWords = word.split('-')
    for i, word in enumerate(splittedWords):
        if not checkSpell(word):
            splittedWords[i] = correctSpell(word)
    return ''.join([s + '-' for s in splittedWords])[:-1]

def embeddingDict(embeddingPath):
    dictionary = []
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            dictionary.append(word)
    f.close()
    return dictionary

def filterWordEmbedding(words, embeddingPath):
    vocabulary = []
    filteredEmbeddingDict = []
    words = [word.lower() for word in words]
    with open(embeddingPath) as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            if word in words:
                vocabulary.append(word)
                filteredEmbeddingDict.append(line)
    f.close()
    unknownWords = [word for word in words if word not in vocabulary]
    with open(cf.ROOT_PATH + cf.DATA_PATH + 'glove_6B_filtered.txt', 'w+') as f:
        for line in filteredEmbeddingDict:
            f.write(line)
    with open('unknown.txt', 'w+') as f:
        for i, word in enumerate(unknownWords):
            f.write(word + '\t' + correctedWords[i] + '\n')

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
# wordData = tokenize(data['text'])
data['text'] = cleanup(data['text'])
trainVoc = createVocabulary(data['text'])
writeCSV(data, cf.ROOT_PATH + cf.DATA_PATH + 'rest_train_2014_processed.csv')
