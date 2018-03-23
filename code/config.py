"""
All constants configuration for the project is here
"""
import os


###################################################
#   file system configuration                     #
###################################################
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "/data/SemEval2014/"
DOMAIN = "Restaurants"
DATA_FILE = "Restaurants_Trial.xml"
WORD2VEC_FILE = "glove.6B.50d.txt"


def configure(year, domain, embedding, aim):
    """
    set file related parameters
    year: 2014 or 2016
    domain: rest or laptop
    embedding: glove or word2vec
    aim: trial, train or test
    """
    global DATA_FOLDER, DOMAIN, DATA_FILE, WORD2VEC_FILE
    DATA_FOLDER = "data/SemEval%s/" % year
    domainDict = {"rest": "Restaurants", "laptop": "Laptops"}
    DOMAIN = domainDict[domain]
    fileDict = {"trial": "Restaurants_Trial.xml", "train": "%s_Train.xml" % DOMAIN, "test": "%s_Test_Data_phaseB.xml" % DOMAIN}
    DATA_FILE = fileDict[aim]
    embeddingDict = {"glove": "glove.6B.50d.txt", "word2vec": "GoogleNews-vectors-negative300.bin"}
    WORD2VEC_FILE = embeddingDict[embedding]
