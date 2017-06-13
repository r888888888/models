import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pprint

def _tag_tokenizer(x):
  return x.split()

data = pd.read_csv("posts_chars.csv")
cv = CountVectorizer(min_df=0.003, tokenizer=_tag_tokenizer)
cv.fit(data["character"])
chars = set(cv.vocabulary_.keys())
total = data.shape[0]
matches = data[data["character"].isin(chars)].shape[0]
pprint.pprint(data[data["character"].isin(chars)].groupby("character").count())
print("percentage: {}".format(100 * matches / total))