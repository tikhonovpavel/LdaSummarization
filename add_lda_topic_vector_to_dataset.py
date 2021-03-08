import glob
import pickle

import gensim
import torch
from tqdm import tqdm

import nltk

nltk.download('wordnet')

# with open('F:/workspace/LdaSummarization/dictionary_large_2020_12_05.pkl', 'rb') as f:
#     tm_dictionary = pickle.load(f)

with open('F:/workspace/LdaSummarization/lda_model_large_2020_12_08.pkl', 'rb') as f:
    lda_model, tm_dictionary = pickle.load(f)

stemmer = nltk.SnowballStemmer('english')


def lemmatize(text):
    return nltk.WordNetLemmatizer().lemmatize(text, pos='v')


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize(token))
    return result

MODE = 'train'

assert MODE in ('train', 'valid', 'test')

limit = 9999999
pt_files = sorted(glob.glob('F:/workspace/LdaSummarization/bert_data/cnndm' + '.' + MODE + '.[0-9]*.pt'))[:limit]

for pt in tqdm(pt_files):
    pt_result = []

    res = torch.load(pt)

    for r in res:
        bow_vector = tm_dictionary.doc2bow(preprocess(' '.join(r['src_txt'])))
        article_topic = sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1])

        r['topics'] = article_topic

    torch.save(res, pt.replace('bert_data', 'bert_data_with_topics'))
