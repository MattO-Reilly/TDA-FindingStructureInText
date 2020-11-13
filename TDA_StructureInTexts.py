import pandas as pd
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

cv_dataframe
