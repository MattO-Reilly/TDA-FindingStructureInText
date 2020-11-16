########################################################
########################################################
############                        ####################
############    PACKAGES            ####################
############                        ####################
########################################################
########################################################

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  # math fun
import matplotlib.pyplot as plt  # plotting
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ripser import ripser  # persistent homology package
from persim import plot_diagrams, bottleneck  # analyzing Persistence Diagrams


a_file = open("DylanThomas.txt")  # sample.txt
lines = a_file.readlines()

########################################################
########################################################
############                        ####################
############   WORD COUNT VECTORS   ####################
############                        ####################
########################################################
########################################################


CountVec = CountVectorizer(ngram_range=(0, 1))
Count_data = CountVec.fit_transform(lines)

# create dataframe
cv_dataframe = pd.DataFrame(
    Count_data.toarray(), columns=CountVec.get_feature_names())
print(cv_dataframe)

data = np.array(cv_dataframe)  # Array of length 4 (In this case)
takens_vector = np.concatenate(data)  # Convert Array to single vector.

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
    embeddedData = np.array([data[0:len(data) - delay * dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(
            embeddedData, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
    return embeddedData


# Embedded into 2 Dimensions
embedded_data2 = takensEmbedding(takens_vector, 1, 2)
# Plot into 3D
fig = plt.figure()
# plot the 2D embedding
ax = fig.add_subplot(3, 1, 2)
ax.plot(embedded_data2[0, :], embedded_data2[1, :]);
plt.show()


# Embedded into 3 Dimensions
embedded_data3 = takensEmbedding(takens_vector, 1, 3)
# Plot into 3D
fig = plt.figure()
# plot the 3D embedding
ax = fig.add_subplot(3, 1, 2, projection='3d')
ax.plot(embedded_data3[0, :], embedded_data3[1, :], embedded_data3[2, :]);
plt.show()


########################################################
########################################################
############                        ####################
############  Persistent Homology   ####################
############                        ####################
########################################################
########################################################

diagrams = ripser(np.transpose(embedded_data3))['dgms']
plot_diagrams(diagrams, show=True)


#persim.bottleneck(dgm1, dgm2, matching=False)
