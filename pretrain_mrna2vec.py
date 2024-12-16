from scipy.stats import pearsonr, spearmanr
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import tqdm
import numpy as np
import random
from torch.utils import data
import wandb
from model import mRNA2vec, T5_encoder
from LoadData import DataLoad

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['OMP_NUM_THREADS'] = '1'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

local_rank = int(os.environ.get('LOCAL_RANK', 0))
setup_seed(36)



if __name__ == '__main__':    

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = 'cuda'
    if local_rank == 0:
        wandb.init(
        project="mRNA_data2vec_mfe",
        dir = './',
        name = 'data2vec_mfe_model',
    
    )
    encoder = T5_encoder(
        hidden_size=256,
        num_attention_heads = 4,
        num_hidden_layers= 4,
    )
    print('loading encoder........')

    #ckpt = torch.load('./model_mfe_mse.pt', map_location='cpu')
    #encoder_state_dict = {k[8:]: v for k, v in ckpt['encoder'].items() if k.startswith('encoder.')}
    #encoder.load_state_dict(ckpt['encoder'], strict=True)

    model = mRNA2vec(encoder=encoder, cfg=cfg)
    

    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)

    
    criterion = nn.MSELoss()
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_cross = nn.CrossEntropyLoss(label_smoothing=0.2)
    criterion.to(device)
    criterion_binary.to(device)
    criterion_cross.to(device)
    print('loading model optimizer........')
    
    # Datasets & Data Loaders
    train_db = DataLoad(mask_ratio=0.15,
                    mode='train')
    val_db = DataLoad(mask_ratio=0.15,
                  mode='valid')
    

    bs = 256 #batch size
    epoch = 10
    param_mfe = 0.01
    param_ss = 0.001
    scaler = torch.cuda.amp.GradScaler()
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_db, shuffle=True, drop_last=False)

    train_loader = torch.utils.data.DataLoader(train_db,
                                               batch_size=bs,
                                               num_workers=8,
                                               drop_last=True,
                                               sampler=train_sampler,
                                               pin_memory=False,
                                               )

    val_loader = torch.utils.data.DataLoader(val_db,
                                             batch_size=bs,
                                             num_workers=8,
                                             drop_last=True,
                                             shuffle=False,
                                             pin_memory=False, )
    
    
    lr_decay = transformers.get_wsd_schedule(optimizer=optimizer,
                                         num_warmup_steps = len(train_loader) * 5,
                                         num_stable_steps = len(train_loader) * 0,
                                         num_decay_steps = len(train_loader) * 5,
                                         )
    i = 0
    best_loss  = 100
    targets = torch.tensor(range(bs))
    for e in range(0, epoch):
        train_sampler.set_epoch(e)
        if local_rank == 0:
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), position=0)
        else:
            loop = enumerate(train_loader)

        for no, batch in loop:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                #src, mask, trg,label_mask  = batch
                rna_masked, mfe_value, ss_label, attention_mask, rna_unmasked,label_mask  = batch
                mask_ss = ss_label > 0
                ss_label  = ss_label[mask_ss]
                #a,b = rna_masked.shape
                mfe_value,ss_label = mfe_value.to(device), ss_label.to(device)
                mfe_value.add_(100.0).div_(100.0)
                src, trg, attention_mask,label_mask = rna_masked.to(device), rna_unmasked.to(device), attention_mask.to(device),label_mask.to(device)
                
                x, y,logit, ss_pred = model(src,trg,attention_mask,label_mask)
                loss_lm = criterion(x.float(), y.float()).mean()
                loss_ss = criterion_cross(ss_pred,ss_)

                targets = torch.tensor(range(a))
                logit = -1*(logit.view(bs,1)-mfe_.to(device))**2
                loss_mfe =  criterion_cross(logit, targets.to(device))
    
                loss = loss_lm + param_mfe*loss_mfe + param_ss*loss_ss
                

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                lr_decay.step()
                model.module.ema_step()
                
                

            if local_rank == 0:
                
                loop.set_postfix(Epoch=e,
                                 loss=loss.item(),
                                 )
                current_lr = optimizer.param_groups[0]['lr']
                i=i+1
                if i%300 == 0:
                    print(loss.item())
                    wandb.log({"train_loss": loss.item(),
                               'lr':current_lr,
                               "loss_b":loss_mfe.item(),
                               "loss_ss":loss_ss.item(),
                               "loss_lm":loss_lm.item()
                               })

        model.eval()
        if local_rank == 0:
            label_lst = []
            res_lst = []
            valid_loss_list = []
            valid_loss_b_list = []
            valid_loss_lm_list = []
            valid_loss_ss_list = []
            spearman_corr_list = []
            for batch in val_loader:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        src,mfe_,ss_, mask, trg,label_mask  = batch
                        mask_ = ss_ > 0
                        ss_  = ss_[mask_]
                        mfe_ = mfe_.to(device)
                        mfe_.add_(100).div_(100)
                        ss_ = ss_.to(device)
                        a,b = src.shape
                        src, trg, mask,label_mask = src.to(device), trg.to(device), mask.to(device),label_mask.to(device)
                        x, y,logit,ss_pred = model(src,trg,mask,label_mask)
                        #x, y = model(src,trg,mask,label_mask)
                        loss_ss = criterion_cross(ss_pred,ss_)

                        spearman_corr = spearmanr(logit.squeeze().cpu().numpy(), mfe_.cpu().numpy())[0].item()
                        loss_lm = criterion(x.float(), y.float()).mean()
                        targets = torch.tensor(range(a))


                        logit = -1*(logit.view(bs,1)-mfe_.to(device))**2
                        loss_b =  criterion_cross(logit, targets.to(device))

                        valid_loss_lm_list.append(loss_lm.item())
                        valid_loss_b_list.append(loss_b.item())
                        valid_loss_list.append(loss_lm.item()+loss_b.item())
                        spearman_corr_list.append(spearman_corr)
                        valid_loss_ss_list.append(loss_ss.item())
               
            wandb.log({"spearman_corr": np.mean(spearman_corr_list),
                       "valid_loss_b": np.mean(valid_loss_b_list),
                       "valid_lm_loss": np.mean(valid_loss_lm_list),
                       "valid_ss_loss": np.mean(valid_loss_ss_list)
                       })
          
            if np.mean(valid_loss_lm_list)<best_loss:
                best_loss = np.mean(valid_loss_lm_list)
                checkpoint = {'encoder': model.module.ema.state_dict(),}
                torch.save(checkpoint, f'./models/model_d2v_mfe{param_mfe}_ss{param_ss}.pt')
        model.train()

                
