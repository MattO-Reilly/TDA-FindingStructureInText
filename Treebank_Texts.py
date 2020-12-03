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
import re

# NLP /Text standardization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, webtext
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


with open("./texts/DylanThomas.txt") as f:
    DylanThomas = f.read()


# Find List of Books from Project Gutenberg
nltk.corpus.gutenberg.fileids()
emma = webtext.raw('austen-emma.txt')

import re
alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    list_sentences = []
    text = " " + text + " "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" +
                  alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets +
                  "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    for sentence in text:
        sentence = text.split("<stop>")
        sentence = sentence[:-1]
        sentence = [s.strip() for s in sentence]
        list_sentences.append(sentence)
    return list_sentences


def standardize_text_line(text):
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


x = standardize_text_line(Sample)
POS_tagging(x)

# DylanThomas_Sentences = split_into_sentences(DylanThomas)

# for x in DylanThomas_Sentences:
# POS_tagging(x)


Emma_sentences = split_into_sentences(emma)

for sentence in Emma_sentences:
    POS_tagging(sentence)
