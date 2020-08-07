import pandas as pd
import pickle

from spacy.lang.en import English

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('train.csv')

data['comment_text'] = data['comment_text'].str.lower() #lowercase
data['comment_text'] = data['comment_text'].str.replace('"', '') #remove double quotes
data['comment_text'] = data['comment_text'].str.replace('\'', '') #remove single quotes
data['comment_text'] = data['comment_text'].str.replace(' / ', ' ') #remove slashes
data['comment_text'] = data['comment_text'].str.replace('/', ' ') #remove slashes
data['comment_text'] = data['comment_text'].str.replace('.', ' ') #remove periods
data['comment_text'] = data['comment_text'].str.replace('\n', ' ') #remove new line characters

en = English()
#nlp = spacy.load('en')

def simple_tokenizer(doc, model=en):
    parsed = model(doc)
    return([t.lower_ for t in parsed if (t.is_alpha)&(not t.like_url)&(not t.is_stop)&(not t.is_punct)])

tfidf = TfidfVectorizer(tokenizer=simple_tokenizer, stop_words="english", max_features=7000)
tfidf.fit(data['comment_text'])
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

train_dtm = tfidf.transform(data['comment_text'])

cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

for c in cols:
  logreg = LogisticRegression()
  logreg.fit(train_dtm, data[c])
  pickle.dump(logreg, open(c + '_model.pkl', 'wb'))