########################################################
########################################################
############                        ####################
############       PACKAGES         ####################
############                        ####################
########################################################
########################################################

from packages import *
from functions import *

########################################################
########################################################
############                        ####################
############       Text File        ####################
############                        ####################
########################################################
########################################################


data = pd.read_csv("./data/bbc-text.csv")
print(f"Shape : {data.shape}, \n\nColumns: {data.columns}, \n\nCategories: {data.category.unique()}")
data.head().append(data.tail())


class DataPreparation:
    def __init__(self, data, column='text'):
        self.df = data
        self.column = column

    def preprocess(self):
        self.tokenize()
        self.remove_stopwords()
        self.remove_non_words()
        self.lemmatize_words()

        return self.df

    def tokenize(self):
        self.df['clean_text'] = self.df[self.column].apply(nltk.word_tokenize)
        print("Tokenization is done.")

    def remove_stopwords(self):
        stopword_set = set(nltk.corpus.stopwords.words('english'))

        def rem_stopword(words): return [
            item for item in words if item not in stopword_set]

        self.df['clean_text'] = self.df['clean_text'].apply(rem_stopword)
        print("Remove stopwords done.")

    def remove_non_words(self):
        """
            Remove all non alpha characters from the text data
            :numbers: 0-9
            :punctuation: All english punctuations
            :special characters: All english special characters
        """
        regpatrn = '[a-z]+'
        def rem_special_chars(x): return [
            item for item in x if re.match(regpatrn, item)]
        self.df['clean_text'] = self.df['clean_text'].apply(rem_special_chars)
        print("Removed non english characters is done.")

    def lemmatize_words(self):
        lemma = nltk.stem.wordnet.WordNetLemmatizer()

        def on_word_lemma(x): return [lemma.lemmatize(w, pos='v') for w in x]

        self.df['clean_text'] = self.df['clean_text'].apply(on_word_lemma)
        print("Lemmatization on the words.")


data_prep = DataPreparation(data)
cleanse_df = data_prep.preprocess()
# cleanse_df['clean_text']

wordcount_df = []

print(cleanse_df.head())

for i in cleanse_df['clean_text']:
    wordcountvec = word_count(i)
    data = takensEmbedding(wordcountvec, 1, 2)
    diagrams = ripser(data)['dgms']
    Persistence_array = np.array(diagrams)
    wordcount_df.append(Persistence_array)

for a, b in itertools.combinations(cleanse_df['clean_text'], 2):
    distance_bottleneck = persim.bottleneck(a, b, matching=False)
'''

with open("./texts/DylanThomas.txt") as f:
    DylanThomas_lines = f.readlines()
DylanThomas_lines = [x.strip() for x in DylanThomas_lines]

#"./texts/sample.txt"
#"./texts/DylanThomas.txt"
#"./texts/Hemingway.txt"
#"./texts/f-scott.txt"

with open("./texts/DylanThomas.txt") as f:
    DylanThomas_sentence = f.read()


list_Sample = standardize_text_sentence(DylanThomas_sentence)
print(list_Sample)
x = word_count(list_Sample)
print(x)
embeddedData = takensEmbedding(x, 1, 2)
dgms_Sample = plot_persistence(DylanThomas_sentence, embeddedData, 2)

list_Sample = standardize_text_line(DylanThomas_lines)
print(list_Sample)
x = word_count(list_Sample)
print(x)
embeddedData = takensEmbedding(x, 1, 2)
dgms_Sample = plot_persistence(DylanThomas_lines, embeddedData, 2)



# Sample Text
print("Sample text")
list_Sample = standardize_text_line(DylanThomas)
vector_Sample = word_count(list_Sample)
x = takensEmbedding(vector_Sample, 1, 2)
dgms_Sample = plot_persistence(x, embeddedData, 2)
# Birth and death array (Used for bottleneck distance)
sample_d = np.array(dgms_Sample)


# Dylan Thomas
print("Dylan Thomas")
list_DylanThomas = standardize_text(DylanThomas)
vector_DylanThomas = word_count(list_DylanThomas)
takensEmbedding(vector_DylanThomas, 1, 2)
dgms_DylanThomas = plot_persistence(Sample, embeddedData, 2)
# Birth and death array (Used for bottleneck distance)
DylanThomas_d = np.array(dgms_DylanThomas)

# Hemingway Text
print("Hemingway")
list_Hemingway = standardize_text(Hemingway)
vector_Hemingway = word_count(list_Hemingway)
takensEmbedding(vector_Hemingway, 1, 2)
dgms_Hemingway = plot_persistence(Sample, embeddedData, 2)
# Birth and death Diagrams array(Used for bottleneck distance)
Hemingway_d = np.array(dgms_Hemingway)

# f-scott Text
print("f-scott")
list_fscott = standardize_text(fscott)
vector_fscott = word_count(list_fscott)
takensEmbedding(vector_fscott, 1, 2)
dgms_fscott = plot_persistence(fscott, embeddedData, 2)
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
    '''
