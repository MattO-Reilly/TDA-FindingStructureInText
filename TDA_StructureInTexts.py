import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math  # math fun
import matplotlib.pyplot as plt  # plotting
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

a_file = open("sample.txt")
lines = a_file.readlines()


CountVec = CountVectorizer(ngram_range=(0, 1))
# transform
Count_data = CountVec.fit_transform(lines)

# create dataframe
cv_dataframe = pd.DataFrame(
    Count_data.toarray(), columns=CountVec.get_feature_names())
print(cv_dataframe.to_string())


data = np.array(cv_dataframe)  # Array of length 4 (In this case)
takens_vector = np.concatenate(data)


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
embedded_data2 = takensEmbedding(data, 1, 2)

# Plot into 2D
fig = plt.figure()

# plot the 2D embedding
ax = fig.add_subplot(3, 1, 2, projection='3d')
ax.plot(embedded_data2[0, :], embedded_data2[1, :]);
plt.show()

########################################################
########################################################
########################################################
########################################################


''' NOT WORKING YET

# Embedded into 3 Dimensions
embedded_data3 = takensEmbedding(data, 1, 3)
# plot the 3D embedding
ax = fig.add_subplot(3, 1, 3, projection='3d')
ax.plot(embedded_data3[0, :], embedded_data3[1, :], embedded_data3[2, :]);
plt.show()

'''
