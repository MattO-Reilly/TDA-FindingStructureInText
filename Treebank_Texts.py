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
    list_sentences = []
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
    for sentence in text:
        sentence = text.split("<stop>")
        sentence = sentence[:-1]
        sentence = [s.strip() for s in sentence]
        list_sentences.append(sentence)
    return list_sentences


def standardize_text_line(text):
    list_1 = []
    for line in text:
        # result = input_str.translate(string.punctuation)
        words = nltk.word_tokenize(line)
        words = [word.lower() for word in words if word.isalpha()]
        lemmatizer = WordNetLemmatizer()
        std_text = lemmatizer.lemmatize(str(words))
        list_1.append(std_text)
    return list_1


def POS_tagging(text_string):
    for line in text_string:
        str1 = ""
        x = str1.join(line)
        x1 = str(x)
        text_str_tok = x1.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text_str_tok)
        print('Word tags for text:', nltk.pos_tag(tokens, tagset="universal"))


x = standardize_text_line(Sample)
POS_tagging(x)

# DylanThomas_Sentences = split_into_sentences(DylanThomas)

# for x in DylanThomas_Sentences:
# POS_tagging(x)


Emma_sentences = split_into_sentences(raw)

for sentence in Emma_sentences:
    POS_tagging(sentence)
    print(sentence)
