import pandas as pd
import pickle

import torch
from pytorch_transformers import BertTokenizer

try:
    with open('f:/workspace/LdaSummarization/lda_model_large_2020_12_08.pkl', 'rb') as f:
        lda_model, tm_dictionary = pickle.load(f)
except FileNotFoundError:
    with open('/content/LdaSummarization/lda_model_large_2020_12_08.pkl', 'rb') as f:
        lda_model, tm_dictionary = pickle.load(f)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
           'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

# result = {}

emb = pickle.load(open('F:/workspace/LdaSummarization/src/embeddings_vocab.pkl', 'rb'))


topics_weighted_embeddings = {}
for t in range(lda_model.num_topics):
    df_temp = pd.DataFrame(lda_model.show_topic(t, topn=10))
    df_temp[2] = df_temp[1] / df_temp[1].sum()

    df_temp = df_temp.rename(columns={0: 'word', 1: 'prob0', 2: 'prob'})

    print(df_temp)

    print()
    # result[t] = df_temp[['word', 'prob']].T.to_dict().values()
    for _, row in df_temp.iterrows():

        word_index = tokenizer.vocab[row.word]

        if t not in topics_weighted_embeddings:
            topics_weighted_embeddings[t] = torch.zeros(emb[word_index].shape)

        topics_weighted_embeddings[t] += emb[word_index] * row.prob

