import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from transformers import DebertaTokenizerFast, T5EncoderModel, T5Config
import argparse
import pickle
from functools import partial
from typing import Callable, Optional, Union, Tuple, List
from timm.layers import create_attn, get_act_layer, get_norm_layer, get_norm_act_layer, create_conv2d, create_pool2d
import timm
import pandas as pd
import torch.nn.functional as F
import transformers
from scipy.stats import pearsonr, spearmanr
from timm.layers import Mlp, DropPath, LayerNorm, ClassifierHead, NormMlpClassifierHead
from sklearn.metrics import accuracy_score
from model import mRNA2vec, T5_encoder,Regression_Model
from LoadData import DataLoad_downstream
SEED = 30

# Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(SEED)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='stability')
    parser.add_argument("--exp_name", type=str, default='unload')
    parser.add_argument("--data_path", type=str,default='data1.csv')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--load_model", type=str, default='False')
    parser.add_argument("--cuda_device", type=str, default='0')
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    #torch.cuda.set_device(int(args.cuda_device))
    task_name = args.task_name
    data_path = args.data_path
    model_name = args.model_name
    DATA_PATH = f'./data/downstream/{task_name}/{data_path}.csv'
    MODEL_PATH = f'./checkpoint/{model_name}'
    TOKENIZER_PATH = './tokenizer/'
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
   
    wandb.init(project=f"mRNA_exp_linear_{task_name}_Folders", dir='./', name=args.exp_name)
    scaler = torch.cuda.amp.GradScaler()
    
    model = Regression_Model(num_attention_heads=4, num_hidden_layers=4, pad_token_id=1, hidden_size=256).cuda()
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    if args.load_model == 'True':
        print('loading model---------')
        model.T5_encoder.load_state_dict(ckpt['encoder'], strict=True)

    train_db = DataLoad_downstream(mode='train', data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    valid_db = DataLoad_downstream(mode='test', data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH)
    val_loader = torch.utils.data.DataLoader(valid_db, batch_size=BATCH_SIZE, num_workers=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_decay = transformers.get_wsd_schedule(optimizer=optimizer,
                                             num_warmup_steps = len(train_loader) * 40,
                                             num_stable_steps = len(train_loader) * 20,
                                             num_decay_steps = len(train_loader) * 40,
                                             )

    best_spear = 0.
    for e in range(EPOCHS):
        loss_lst = []
        for no, batch_train in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x, attention_mask,label = [x.to('cuda') for x in batch_train]
                loss = model(x, attention_mask, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_decay.step()
                loss_lst.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train loss": np.mean(loss_lst),
                   'lr':current_lr,})
        model.eval()
        label_lst = []
        predict_lst = []
        for no, batch_test in enumerate(val_loader):
            x, attention_mask,label = [x.to('cuda') for x in batch_test]
            with torch.no_grad():
                outs = model.forward_logit_linear(x, attention_mask)
            label_lst.extend(label.reshape(-1).cpu().numpy().tolist())
            predict_lst.extend(outs.reshape(-1).cpu().numpy().tolist())
        spearman_corr = spearmanr(predict_lst, label_lst)[0].item()
        if best_spear < spearman_corr:
            best_spear = spearman_corr
        print(f'test on {task_name}', e, spearman_corr, best_spear)
        wandb.log({"eval_spearman": spearman_corr,
                   'best_spearman':best_spear,})
        model.train()

if __name__ == '__main__':
    main()

