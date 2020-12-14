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

'''

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
