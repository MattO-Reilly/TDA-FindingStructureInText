########################################################
########################################################
############                        ####################
############       PACKAGES         ####################
############                        ####################
########################################################
########################################################

from packages import *

a_file = open(("./texts/sample.txt").lower())  # DylanThomas/sample.txt
Sample = a_file.readlines()


with open("./texts/DylanThomas.txt") as f:
    DylanThomas = f.read()

# Crime and Punishment by Fyodor Dostoevsky (Project Gutenberg)
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')
raw.find("PART I")
raw.rfind("End of Project Gutenberg's Crime")
raw = raw[5338:5577]

alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    sentences_list = []
    text = text.lower()
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
    sentences = [sentence[:-1] for sentence in text.split("<stop>")]
    for i in sentences:
        i = ''.join(x for x in i if x not in string.punctuation)
        sentences_list.append([i])
    return sentences_list


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


def word_count(std_text):
    appended_data = pd.DataFrame()
    CountVec = CountVectorizer(ngram_range=(0, 1))
    Count_data = CountVec.fit_transform(std_text)
    cv_dataframe = pd.DataFrame(
        Count_data.toarray(), columns=CountVec.get_feature_names())
    takens_vector = np.concatenate(np.array(cv_dataframe, dtype=np.float32))
    return takens_vector


DylanThomas_Sentences = split_into_sentences(DylanThomas)
print(DylanThomas_Sentences)
DylanThomas_POS = POS_tagging(DylanThomas_Sentences)
word_count(DylanThomas_Sentences)
print(DylanThomas_POS[0])
