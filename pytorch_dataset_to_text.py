import glob
import pickle

import torch
from tqdm import tqdm

limit = 100
pt_files = pts = sorted(glob.glob('F:/workspace/LdaSummarization/bert_data/cnndm' + '.' + 'train' + '.[0-9]*.pt'))[:limit]

result = []
for pt in tqdm(pt_files):
    res = torch.load(pt)
    for r in res:
        result.append({
            'srt_txt': r['src_txt'],
            'tgt_txt': r['tgt_txt'],
        })

# with open('F:/workspace/PreSumm/cnndm_text.txt', 'w'):


pickle.dump(result, open('F:/workspace/LdaSummarization/cnndm_src_tgt_full.pkl', 'wb'))
