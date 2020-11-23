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
from sklearn.feature_extraction.text import CountVectorizer
from ripser import *  # persistent homology package
# analyzing Persistence Diagrams
from persim import *
import sklearn
import persim
import gudhi

# NLP /Text standardization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

a_file = open(("./texts/f-scott.txt").lower())  # DylanThomas/sample.txt
fscott = a_file.readlines()


########################################################
########################################################
############                        ####################
############   WORD COUNT VECTORS   ####################
############                        ####################
########################################################
########################################################


def standardize_text(text):
    global std_text
    list_1 = []
    for i in text:
        # result = input_str.translate(string.punctuation)
        words = nltk.word_tokenize(i)
        words = [word.lower() for word in words if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        std_text = [i for i in words if not i in stop_words]
        lemmatizer = WordNetLemmatizer()
        std_text = lemmatizer.lemmatize(str(std_text))
        print(std_text)
        List_1.append(std_text)


def wourd_count(std_text):
    CountVec = CountVectorizer(ngram_range=(0, 1))
    Count_data = CountVec.fit_transform([std_text])
    cv_dataframe = pd.DataFrame(
        Count_data.toarray(), columns=CountVec.get_feature_names())
    print(cv_dataframe)
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


def plot_persistence(text, embeddedData, dimension):
    global d
    if dimension == 2:
        # Embedded into 2 Dimensions
        embedding_2d = embeddedData
        # Plot into 2D
        fig = plt.figure()
        # plot the 2D embedding
        ax = fig.add_subplot(3, 1, dimension)
        ax.plot(embedding_2d[0, :], embedding_2d[1, :]);
        plt.show()
        d = ripser(embedding_2d)['dgms']
        plot_diagrams(d, show=True)
    if dimension == 3:
        # Embedded into 3 Dimensions
        embedding_3d = embeddedData
        # Plot into 3D
        fig = plt.figure()
        # plot the 3D embedding
        ax = fig.add_subplot(3, 1, 3, projection='3d')
        ax.plot(embedding_3d[0, :],
                embedding_3d[1, :], embedding_3d[2, :]);
        plt.show()
        d = ripser(embedding_3d)['dgms']
        plot_diagrams(d, show=True)



# Dylan Thomas
print("DylanThomas")
for i in
standardize_text(DylanThomas)
wourd_count(std_text)
print(max(takens_vector))
takensEmbedding(takens_vector, 1, 3)
plot_persistence(DylanThomas, embeddedData, 3)
d1 = d


# Sample Text
print("Sample text")
standardize_text(Sample)
wourd_count(std_text)
print(takens_vector)
takensEmbedding(takens_vector, 1, 3)
plot_persistence(Sample, embeddedData, 3)
d2 = d

# Hemingway Text
print("Hemingway")
standardize_text(Hemingway)
wourd_count(std_text)
print(len(std_text))
x = takensEmbedding(takens_vector, 1, 3)
plot_persistence(Hemingway, embeddedData, 3)
d3 = d

# f-scott Text
print("f-scott")
standardize_text(fscott)
wourd_count(std_text)
print(len(std_text))
y = takensEmbedding(takens_vector, 1, 3)
plot_persistence(fscott, embeddedData, 3)
d4 = d


# distance_bottleneck, (matching, D) = persim.bottleneck(
# hemingway, fscott, matching=True)

#print(persim.bottleneck(d1, d2))
