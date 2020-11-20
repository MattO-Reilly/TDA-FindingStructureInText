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

# Persistent Homology
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ripser import ripser, Rips  # persistent homology package
from persim import plot_diagrams, bottleneck  # analyzing Persistence Diagrams
import sklearn

import xml.etree.ElementTree as ET  # Get around for a parsing error
import gudhi

# NLP /Text standardization
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string

from pathlib import Path
########################################################
########################################################
############                        ####################
############       Text File        ####################
############                        ####################
########################################################
########################################################


# NLTK for word processing

'''
def open_file(path):
    root = ET.fromstring(("./texts/" + str(path)))
    root = tree.getroot()
    a_file = open(root).lower()
    path = a_file.readlines()
    return path



def open_file(text_file):
    data_folder = Path("./texts/")
    file_to_open = str(data_folder) + str(text_file)
    root = ET.fromstring(file_to_open)
    f = open(root).lower()
    text_file = f_file.readlines()

'''

a_file = open(("./texts/sample.txt").lower())  # DylanThomas/sample.txt
Sample = a_file.readlines()
a_file = open(("./texts/DylanThomas.txt").lower())  # DylanThomas/sample.txt
DylanThomas = a_file.readlines()
a_file = open(("./texts/Hemingway.txt").lower())  # DylanThomas/sample.txt
Hemingway = a_file.readlines()

########################################################
########################################################
############                        ####################
############   WORD COUNT VECTORS   ####################
############                        ####################
########################################################
########################################################


def standardize_text(text):
    global std_text
    input_str = str(text)
    # result = input_str.translate(string.punctuation)
    words = nltk.word_tokenize(input_str)
    words = [word.lower() for word in words if word.isalpha()]
    stop_words = set(stopwords.words("english"))
    std_text = [i for i in words if not i in stop_words]
    print(std_text)
    return std_text


def wourd_count(std_text):
    CountVec = CountVectorizer(ngram_range=(0, 1))
    Count_data = CountVec.fit_transform(std_text)
    cv_dataframe = pd.DataFrame(
        Count_data.toarray(), columns=CountVec.get_feature_names())
    # Convert Array to single vector.
    global takens_vector
    takens_vector = np.concatenate(np.array(cv_dataframe))
    return takens_vector

########################################################
########################################################
############                        ####################
############   Takens Embedding     ####################
############                        ####################
########################################################
########################################################


def takensEmbedding(data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay * dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')
        global embeddedData
    embeddedData = np.array([data[0:len(data) - delay * dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(
            embeddedData, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
    return embeddedData


def plot_embedding(embeddedData, dimension):
    if dimension == 2:
        # Embedded into 2 Dimensions
        global embedding_2d
        embedding_2d = embeddedData
        # Plot into 2D
        fig = plt.figure()
        # plot the 2D embedding
        ax = fig.add_subplot(3, 1, dimension)
        ax.plot(embedding_2d[0, :], embedding_2d[1, :]);
        plt.show()
        return embedding_2d
    if dimension == 3:
        # Embedded into 3 Dimensions
        global embedding_3d
        embedding_3d = embeddedData
        # Plot into 3D
        fig = plt.figure()
        # plot the 3D embedding
        ax = fig.add_subplot(3, 1, 3, projection='3d')
        ax.plot(embedding_3d[0, :],
                embedding_3d[1, :], embedding_3d[2, :]);
        plt.show()
        return embedding_3d


########################################################
########################################################
############                        ####################
############  Persistent Homology   ####################
############                        ####################
########################################################
########################################################

def plot_persistence_diagram(embedding_dim):
    if embedding_dim == 2:
        diagrams = ripser(embedding_2d)['dgms']
        plot_diagrams(diagrams, show=True)
    if embedding_dim == 3:
        diagrams = ripser(embedding_3d)['dgms']
        plot_diagrams(diagrams, show=True)


# persim.bottleneck(dgm1, dgm2, matching=False)


# Dylan Thomas
print("DylanThomas")
standardize_text(DylanThomas)
wourd_count(std_text)
takensEmbedding(takens_vector, 1, 3)
plot_embedding(embeddedData, 3)
plot_persistence_diagram(3)


# Sample Text
print("Sample text")
standardize_text(Sample)
wourd_count(std_text)
takensEmbedding(takens_vector, 1, 3)
plot_embedding(embeddedData, 3)
plot_persistence_diagram(3)

# Hemingway Text
print("Hemingway")
standardize_text(Hemingway)
wourd_count(std_text)
takensEmbedding(takens_vector, 1, 3)
plot_embedding(embeddedData, 3)
plot_persistence_diagram(3)
# gudhi.plot_persistence_barcode(embedding_3d)
