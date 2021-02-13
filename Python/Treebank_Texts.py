
from packages import *
from functions import *

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


DylanThomas_Sentences = standardize_text_sentence(DylanThomas)
print(DylanThomas_Sentences)
DylanThomas_POS = POS_tagging(DylanThomas_Sentences)
print(DylanThomas_POS)
