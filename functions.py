from packages import *

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def standardize_text_sentence(text):
    sentences_list = []
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
    sentences = text.split("<stop>")
    #sentences = sentences[:-1]
    #sentences = text.split("<stop>")
    for i in sentences:
        words = nltk.word_tokenize(i)
        std_text = [word.lower() for word in words if word.isalpha()]
        lemmatizer = WordNetLemmatizer()
        i = lemmatizer.lemmatize(str(std_text))
        # i = ''.join(x for x in i if x not in string.punctuation)
        sentences_list.append(i)
    return sentences_list


def standardize_text_line(text):
    list_1 = []
    for line in text:
        # result = input_str.translate(string.punctuation)
        words = nltk.word_tokenize(line)
        std_text = [word.lower() for word in words if word.isalpha()]
        # stop_words = set(stopwords.words("english"))
        # std_text = [i for i in words if not i in stop_words]
        lemmatizer = WordNetLemmatizer()
        std_text = lemmatizer.lemmatize(str(std_text))
        list_1.append(std_text)
    return list_1


def word_count(std_text):
    np.set_printoptions(threshold=sys.maxsize)
    appended_data = pd.DataFrame()
    CountVec = CountVectorizer(ngram_range=(0, 1))
    Count_data = CountVec.fit_transform(std_text)
    cv_dataframe = pd.DataFrame(
        Count_data.toarray(), columns=CountVec.get_feature_names())
    takens_vector = np.concatenate(np.array(cv_dataframe, dtype=object))
    return takens_vector


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


def POS_tagging(text_string):
    list_1 = []
    for i in text_string:
        str1 = ""
        x = str1.join(i)
        x1 = str(x)
        text_str_tok = x1.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text_str_tok)
        tags = nltk.pos_tag(tokens, tagset="universal")
        list_1.append(tags)
    return list_1


def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test
