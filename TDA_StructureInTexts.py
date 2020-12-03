########################################################
########################################################
############                        ####################
############       PACKAGES         ####################
############                        ####################
########################################################
########################################################

from packages import *

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
    list_1 = []
    for line in text:
        # result = input_str.translate(string.punctuation)
        words = nltk.word_tokenize(line)
        words = [word.lower() for word in words if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        std_text = [i for i in words if not i in stop_words]
        lemmatizer = WordNetLemmatizer()
        std_text = lemmatizer.lemmatize(str(std_text))
        list_1.append(std_text)
    print(list_1)
    return list_1


def word_count(std_text):
    appended_data = pd.DataFrame()
    CountVec = CountVectorizer(ngram_range=(0, 1))
    Count_data = CountVec.fit_transform(std_text)
    cv_dataframe = pd.DataFrame(
        Count_data.toarray(), columns=CountVec.get_feature_names())
    takens_vector = np.concatenate(np.array(cv_dataframe, dtype=np.float32))
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
    if dimension == 2:
        # Embedded into 2 Dimensions
        embedding_2d = embeddedData
        # Plot into 2D
        fig = plt.figure()
        # plot the 2D embedding
        ax = fig.add_subplot(3, 1, dimension)
        ax.plot(embedding_2d[0, :], embedding_2d[1, :]);
        plt.show()
        d = ripser(embedding_2d)['dgms'][0]
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
        d = ripser(embedding_3d)['dgms'][0]
        plot_diagrams(d, show=True)
    return d


########################################################
########################################################
############                        ####################
############     Prepare texts      ####################
############                        ####################
########################################################
########################################################



# Sample Text
print("Sample text")
list_Sample = standardize_text(Sample)
vector_Sample = word_count(list_Sample)
takensEmbedding(vector_Sample, 1, 3)
dgms_Sample = plot_persistence(Sample, embeddedData, 3)
# Birth and death array (Used for bottleneck distance)
sample_d = np.array(dgms_Sample)

# Dylan Thomas
print("Dylan Thomas")
list_DylanThomas = standardize_text(DylanThomas)
vector_DylanThomas = word_count(list_DylanThomas)
takensEmbedding(vector_DylanThomas, 1, 3)
dgms_DylanThomas = plot_persistence(Sample, embeddedData, 3)
# Birth and death array (Used for bottleneck distance)
DylanThomas_d = np.array(dgms_DylanThomas)

# Hemingway Text
print("Hemingway")
list_Hemingway = standardize_text(Hemingway)
vector_Hemingway = word_count(list_Hemingway)
takensEmbedding(vector_Hemingway, 1, 3)
dgms_Hemingway = plot_persistence(Sample, embeddedData, 3)
# Birth and death Diagrams array(Used for bottleneck distance)
Hemingway_d = np.array(dgms_Hemingway)

# f-scott Text
print("f-scott")
list_fscott = standardize_text(fscott)
vector_fscott = word_count(list_fscott)
takensEmbedding(vector_fscott, 1, 3)
dgms_fscott = plot_persistence(fscott, embeddedData, 3)
# Birth and death Diagrams array (Used for bottleneck distance)
fscott_d = np.array(dgms_fscott)


########################################################
########################################################
############                        ####################
############    Bottleneck dist     ####################
############                        ####################
########################################################
########################################################

distance_bottleneck = persim.bottleneck(
    sample_d, DylanThomas_d, matching=False)
print(
    f"The Bottleneck Distance between Arianna Grande and Dylan Thomas is: {distance_bottleneck}")

distance_bottleneck = persim.bottleneck(
    Hemingway_d, DylanThomas_d, matching=False)
print(
    f"The Bottleneck Distance between Hemingway and Dylan Thomas is: {distance_bottleneck}")

distance_bottleneck = persim.bottleneck(
    Hemingway_d, fscott_d, matching=False)
print(
    f"The Bottleneck Distance between Hemingway and F. Scott Fitzgerald is: {distance_bottleneck}")

distance_bottleneck = persim.bottleneck(
    fscott_d, DylanThomas_d, matching=False)
print(
    f"The Bottleneck Distance between F. Scott Fitzgerald and Dylan Thomas is: {distance_bottleneck}")

distance_bottleneck = persim.bottleneck(
    fscott_d, sample_d, matching=False)
print(
    f"The Bottleneck Distance between F. Scott Fitzgerald and Arianna Grande is: {distance_bottleneck}")
