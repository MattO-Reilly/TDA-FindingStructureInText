from packages import *
from functions import *


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


vectorizer = CountVectorizer()
data_prep = DataPreparation(data)
cleanse_df = data_prep.preprocess()

wordcount_df = []
for i in cleanse_df['clean_text']:
    vectorizer.fit(i)
    vector = vectorizer.fit_transform(i)
    vector = vector.toarray()
    vector = np.concatenate(vector)
    wordcount_df.append(list(vector))

cleanse_df['wordcountvec'] = wordcount_df


pd.set_option('display.max_colwidth', None)
cleanse_df.to_excel(r'./data/text_vectors1.xlsx', index=False)
