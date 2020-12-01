########################################################
########################################################
############                        ####################
############       PACKAGES         ####################
############                        ####################
########################################################
########################################################

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  # math fun
import matplotlib.pyplot as plt  # plotting

# NLP /Text standardization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# POS Tagging
from nltk.corpus import brown
import nltk.tokenize.treebank
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import string

a_file = open(("./texts/sample.txt").lower())  # DylanThomas/sample.txt
Sample = a_file.readlines()

a_file = open(("./texts/DylanThomas.txt").lower())  # DylanThomas/sample.txt
DylanThomas = a_file.readlines()


def standardize_text(text):
    list_1 = []
    for line in text:
        # result = input_str.translate(string.punctuation)
        words = nltk.word_tokenize(line)
        words = [word.lower() for word in words if word.isalpha()]
        lemmatizer = WordNetLemmatizer()
        std_text = lemmatizer.lemmatize(str(words))
        list_1.append(std_text)
    return list_1


def POS_tagging(text_string):
    for line in text_string:
        str1 = ""
        x = str1.join(line)
        x1 = str(x)
        text_str_tok = x1.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text_str_tok)
        print('Word tags for text:', nltk.pos_tag(tokens, tagset="universal"))


x = standardize_text(Sample)
POS_tagging(x)

#y = standardize_text(DylanThomas)
# POS_tagging(y)
