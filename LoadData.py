import transformers
from transformers import DebertaTokenizerFast, T5EncoderModel, T5Config
from typing import Callable, Optional, Union, Tuple, List
from torch.utils import data
import pickle
import torch
import numpy as np
import random
import pandas as pd
def pkl_load(one_path):
    with open(one_path, 'rb') as f:
        return pickle.load(f)
    
class DataLoad(data.Dataset):
    def __init__(self, data_path='./data/data_mfe_ss_510k.pkl',
                 tokenizer_path='./tokenizer/',
                 mode='train',
                 max_length=128,
                 mask_ratio=0.15,
                 ):
        self.mode = mode
        symbolic = ['(','.',')']
        lst_voc = ['PAD']
        for a1 in symbolic:
            for a2 in symbolic:
                 for a3 in symbolic:
                    lst_voc.extend([f'{a1}{a2}{a3}'])
        self.dic_voc = dict(zip(lst_voc, range(len(lst_voc))))

        self.data = pkl_load(data_path)
        self.data_x = []
        self.data_y = []
        self.data_ss = []
        
        if mode == 'train':
            for seq,ss, mfe in self.data[:-2000]:
                seq = seq.replace('T', 'U')
                self.data_x.append(seq)
                self.data_y.append(mfe)
                self.data_ss.append(ss)
        else:
            for seq,ss, mfe in self.data[-2000:]:
                seq = seq.replace('T', 'U')
                self.data_x.append(seq)
                self.data_y.append(mfe)
                self.data_ss.append(ss)
        self.tokenizer = DebertaTokenizerFast.from_pretrained(tokenizer_path, use_fast=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_ratio = mask_ratio

        self.mask_fn = self.normal_mask
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.tokenize_ss = self._tokenize_ss

    def __getitem__(self, index):

        seq_rna = self.data_x[index]
        mfe_rna = self.data_y[index]
        ss_rna = self.data_ss[index]
        ss_category = self.tokenize_ss(ss_rna).numpy()
        
        seq_rna = self.tokenizer(seq_rna,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt",
                                )
        
        y = seq_rna.input_ids[0].numpy().copy()
        mask_rna, label_mask = self.mask_fn(seq_rna)
        x = mask_rna.input_ids[0].numpy()
        attention_mask = mask_rna.attention_mask[0].numpy()
        return x, mfe_rna,ss_category, attention_mask, y,label_mask

    def __len__(self):
        return len(self.data_x)
    
    def _tokenize_ss(self,ss_rna):
        num_tokens = int(len(ss_rna)/3)
        ss_categories = [-1]
        for i in range(0, 3*126,3):
            if i+3>len(ss_rna):
                ss_categories.insert(0, 0)
            else:
                a1 = ss_rna[-1-i]
                a2 = ss_rna[-2-i]
                a3 = ss_rna[-3-i]
                category = self.dic_voc[f'{a3}{a2}{a1}']
                ss_categories.insert(0,category)
        ss_categories.insert(0, -2)
        return torch.tensor(ss_categories)


    def normal_mask(self, one_ab):
        input_ids = one_ab.input_ids.numpy()
        attention_mask_ids = one_ab.attention_mask[0].numpy()
        index_lst = np.where(attention_mask_ids == 1)[0].tolist()
        index_lst.pop(0)
        mask_index = random.sample(index_lst[15:45], int(self.mask_ratio*0.8 * len(index_lst)))
        input_ids[0][mask_index] = self.mask_token_id
        mask_index = random.sample(index_lst, int(self.mask_ratio*0.2 * len(index_lst)))
        input_ids[0][mask_index] = self.mask_token_id
        mask_lst = np.zeros(input_ids[0].shape)
        mask_lst[mask_index] = 1
        one_ab.input_ids = torch.tensor(input_ids)
        return one_ab, torch.tensor(mask_lst)

class DataLoad_downstream(data.Dataset):
    def __init__(self, data_path=str,
                 tokenizer_path=str,
                 mode='train',
                 max_length=32,
                 ):
        self.mode = mode
        self.max_length = max_length
        te_data = pd.read_csv(data_path)
        #if mode == 'train':
        te_data = te_data[te_data['Sequence'].str.len() <= 1024]
        seqs = te_data['Sequence'].values.tolist()
        seqs = [x.replace('<pad>', '') for x in seqs]
        seqs = [x.replace('T', 'U') for x in seqs]
        value = te_data['Value'].values.tolist()
        states = te_data['Split'].values.tolist()
        states = [x if x=='test' else 'train' for x in states ]
        self.data = [[x, y] for x, y, z in zip(seqs, value, states) if z == mode]
        print(len(self.data))
        self.tokenizer = DebertaTokenizerFast.from_pretrained(tokenizer_path, use_fast=True)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left"

    def __getitem__(self, index):
        no2 = random.choice(list(range(len(self.data))))
        seq, label = self.data[index]
        label = np.array(label, dtype=np.float32)
        seq = self.tokenizer(seq,
                              padding='max_length',
                              max_length=self.max_length,
                              truncation=True,
                              add_special_tokens=True,
                              return_tensors="pt",
                              )
        


        x = seq.input_ids[0].numpy()
        attention_mask = seq.attention_mask[0].numpy()


        return x,attention_mask,label

    def __len__(self):
        return len(self.data)
