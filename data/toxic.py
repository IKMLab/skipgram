import glovar
import os
import pandas as pd


train = pd.read_csv(os.path.join(glovar.DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(glovar.DATA_DIR, 'test.csv'))
token_set = set([])
for _, row in train.iterrows():
    token_set.update(nltk.word_tokenize(row['comment_text']))
for _, row in test.iterrows():
    token_set.update(nltk.word_tokenize(row['comment_text']))
print(len(token_set))

