import pickle
import pandas as pd
from tqdm import tqdm

with open('preproc_files/good_tags.pickle', 'rb') as g:
    good_tags = pickle.load(g)
g.close()

def extract_tag_features(df):
    tag_df = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tag_df.append({f't_{tag}': 1 if tag in row.tags else 0 for tag in good_tags})
    return pd.DataFrame(tag_df)


#--------------------------#
# Text preprocessing block #
#--------------------------#
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
exclude = set(string.punctuation)
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^(a-zA-Z)\s]','', text)
    text = re.sub(r'\s+', ' ', text.replace('\n', ' '))
    return text

def remove_stopwords(words):
    list_clean = [w for w in words.split(' ') if not w in stop_words]
    return ' '.join(list_clean)


def lemmatize_text(text):
    text = ''.join(ch for ch in text.lower() if ch.isalpha() or ch == ' ')
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    preprocessed = ' '.join(lemma for lemma in lemmatized if lemma not in stop_words)
    return preprocessed

def preprocess_text_lemmatized(base):
    base['text'] = base['Title'] + " " + base['Body']
    base['text'] = base['text'].apply(clean_text)
    base['text'] = base['text'].apply(lemmatize_text)
    return base

def preprocess_text_not_lemmatized(base):
    base['text'] = base['Title'] + " " + base['Body']
    base['text'] = base['text'].apply(clean_text)
    base['text'] = base['text'].apply(lemmatize_text)
    return base
