import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  # math fun
import matplotlib.pyplot as plt  # plotting
import re
from urllib import request

# NLP /Text standardization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, webtext
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# POS Tagging
from nltk.corpus import brown
import nltk.tokenize.treebank
from nltk.tokenize import TreebankWordTokenizer, sent_tokenize
import string

# Persistent Homology
from sklearn.feature_extraction.text import CountVectorizer
from ripser import *  # persistent homology package

# analyzing Persistence Diagrams
from persim import *
import sklearn
import persim
import gudhi
