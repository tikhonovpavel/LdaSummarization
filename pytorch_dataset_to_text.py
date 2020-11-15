import glob
import pickle

import torch
from tqdm import tqdm

limit = 10
pt_files = pts = sorted(glob.glob('F:/workspace/PreSumm/bert_data/cnndm' + '.' + 'train' + '.[0-9]*.pt'))[:limit]

result = []
for pt in tqdm(pt_files):
    res = torch.load(pt)
    for r in res:
        result.append(r['src_txt'])

# with open('F:/workspace/PreSumm/cnndm_text.txt', 'w'):


pickle.dump(result, open('F:/workspace/PreSumm/cnndm_list.pkl', 'wb'))
