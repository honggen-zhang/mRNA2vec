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
# Constants
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
    #python test_exp.py --task_name "HEK_TE" --exp_name "d2v" --data_path "data1" --model_name "model_d2v_mfe0_ss0_specific.pt" --load_model True --cuda_device "3"
    #os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    #torch.cuda.set_device(int(args.cuda_device))
    task_name = args.task_name
    data_path = args.data_path
    model_name = args.model_name
    DATA_PATH = f'./data/{task_name}/{data_path}.csv'
    MODEL_PATH = f'./models/{model_name}'
    TOKENIZER_PATH = './tokenizer/'
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
   
    wandb.init(project=f"mRNA_exp_linear_{task_name}_Folders", dir='./', name=args.exp_name)
    scaler = torch.cuda.amp.GradScaler()
    model = Regression_Model(num_attention_heads=4, num_hidden_layers=4, pad_token_id=1, hidden_size=256).cuda()
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    if args.load_model == 'True':
        print('loading model')
        encoder_state_dict = {k[8:]: v for k, v in ckpt['encoder'].items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state_dict, strict=True)

    train_db = DataLoad_downstream(mode='train', data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH)
    train_loader = torch.utils.data.DataLoader(train_db, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    valid_db = DataLoad_downstream(mode='test', data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH)
    val_loader = torch.utils.data.DataLoader(valid_db, batch_size=128, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_decay = transformers.get_wsd_schedule(optimizer=optimizer,
                                             num_warmup_steps = len(train_loader) * 40,
                                             num_stable_steps = len(train_loader) * 20,
                                             num_decay_steps = len(train_loader) * 40,
                                             )

    best_spear = 0.
    for e in range(EPOCHS):
        loss_lst = []
        for no, i in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                x1, x2, mask1, mask2, label1, label2 = [x.to('cuda') for x in i]
                l1, l2 = model(x1, x2, mask1, mask2, label1, label2)
                loss = l1 + l2
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_decay.step()
                loss_lst.append(loss.item())
        #print('train loss', e, np.mean(loss_lst))
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"train loss": np.mean(loss_lst),
                   'lr':current_lr,})
        #print(current_lr)
        model.eval()
        label_lst = []
        predict_lst = []
        for no, i in enumerate(val_loader):
            x1, x2, mask1, mask2, label1, label2 = [x.to('cuda') for x in i]
            with torch.no_grad():
                outs = model.forward_logit(x1, mask1)
            label_lst.extend(label1.reshape(-1).cpu().numpy().tolist())
            predict_lst.extend(outs.reshape(-1).cpu().numpy().tolist())
        spearman_corr = spearmanr(predict_lst, label_lst)[0].item()
        #spearman_corr = accuracy_score(predict_lst, label_lst)
        if best_spear < spearman_corr:
            best_spear = spearman_corr
        print('eval_EL', e, spearman_corr, best_spear)
        wandb.log({"eval_spearman": spearman_corr,
                   'best_spearman':best_spear,})
        model.train()

if __name__ == '__main__':
    main()

