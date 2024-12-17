from transformers import AutoModel, AutoConfig, AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from EMA import EMA
import torch.optim as optim
import transformers
from typing import Callable, Optional, Union, Tuple, List
from transformers import T5EncoderModel, T5Config

class mRNA2vec(nn.Module):


    def __init__(self, encoder, **kwargs):
        super(mRNA2vec, self).__init__()
        self.embed_dim = 256
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.ema = EMA(self.encoder)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()

        #self.clf = nn.Linear(2048*2, 1)
        self.clf = nn.Sequential(nn.Linear(256, 200),
                                 nn.GELU(),
                                 nn.Linear(200, 1),)
        self.clf_ss = self._build_ss_head()

        #self.ema_decay = self.cfg.model.ema_decay
        #self.ema_end_decay = self.cfg.model.ema_end_decay
        #self.ema_anneal_end_step = self.cfg.model.ema_anneal_end_step

    def _build_ss_head(self):
        return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, 28))

    def _build_regression_head(self):
        return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim))

    def _clf(self,x):
        x = self.clf(x)
        return x

    def ema_step(self):
        self.ema.step(self.encoder)

    def forward(self, src, trg=None, mask=None,label_mask = None, **kwargs):
        # model forward in online mode (student)
        
        outputs = self.encoder(src, mask) # fetch the last layer outputs
        x = outputs['encoder_out']
        mask_ = src[:,1:-1].clone().gt(1)
        x_ss_clf  = x[:,1:-1,:].clone()
        x_ss_clf  = x_ss_clf[mask_]
        x_ss_clf = self.clf_ss(x_ss_clf)

        if trg is None:
            return x

        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(trg, mask)['encoder_states']
            y = y[-3:]  # take the last k transformer layers
            y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
            y = sum(y) / len(y)
            y = F.layer_norm(y.float(), y.shape[-1:])


        e = self._clf(x.mean(dim =1))
        

        masked_indices = src.eq(4)
        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        return x, y, e, x_ss_clf

class T5_encoder(nn.Module):
    def __init__(self,
                 vocab_size=69,
                 hidden_size=256,
                 num_hidden_layers=4,
                 num_attention_heads=8,
                 pad_token_id=1
                 ):
        super().__init__()
        self.embed_dim = hidden_size
        model_cofig = T5Config()
        model_cofig.d_model = hidden_size
        model_cofig.num_attention_heads = num_attention_heads
        model_cofig.d_kv = hidden_size//num_attention_heads
        model_cofig.pad_token_id = pad_token_id
        model_cofig.num_layers = num_hidden_layers
        model_cofig.d_ff = hidden_size * 4
        model_cofig.vocab_size = vocab_size
        self.encoder = T5EncoderModel(config=model_cofig)

    def forward(self, x, attention_mask):
        outputs = self.encoder(x,
                               attention_mask=attention_mask,
                               return_dict=True,
                               output_attentions = True,
                               output_hidden_states = True,
                               )
        encoder_states =outputs.hidden_states[:]
        encoder_out = outputs.hidden_states[-1]
        attentions = outputs.attentions
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }

class ConvMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,
            norm_layer=None,
            bias=True,
            drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=1, )
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=1,)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class LayerScale1d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma




class ConvNeXtBlock1D(nn.Module):
    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            conv_mlp: bool = True,
            drop_path: float = 0.
    ):
        super().__init__()
        out_chs = out_chs or in_chs
        act_layer = nn.GELU
        norm_layer = nn.LayerNorm
        mlp_layer = ConvMlp(in_features=out_chs, hidden_features=2 * out_chs, act_layer=act_layer)  # You will need to modify ConvMlp to be 1D as well
        self.use_conv_mlp = conv_mlp
        if stride == 2:
            self.shortcut = nn.AvgPool1d(3, stride=2, padding=1)
        elif in_chs != out_chs:
            self.shortcut = nn.Conv1d(in_chs, out_chs, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if stride == 2:
            self.down = nn.AvgPool1d(3, stride=2, padding=1)
        else:
            self.down = nn.Identity()

        self.conv_dw = nn.Conv1d(
            in_chs, out_chs, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=1)
        self.norm = norm_layer(normalized_shape=out_chs)
        self.mlp = mlp_layer  # Modify ConvMlp accordingly
        self.ls = LayerScale1d(out_chs, 1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        x = self.down(x)
        x = self.conv_dw(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)  # LayerNorm expects the channel as the last dimension
        x = self.mlp(x)
        x = self.ls(x)
        x = self.drop_path(x) + shortcut
        return x

class Regression_Model(nn.Module):
    def __init__(self,
                 vocab_size=69,
                 hidden_size=512,
                 num_hidden_layers=12,
                 num_attention_heads=8,
                 pad_token_id=1
                 ):
        super().__init__()
        self.T5_encoder = T5_encoder(hidden_size=hidden_size,
                                  num_attention_heads = num_attention_heads,
                                  num_hidden_layers= num_hidden_layers,)
        self.proj1 = nn.Linear(hidden_size, hidden_size // 32)

        self.proj2 = nn.Linear(32 * hidden_size // 32, 128)
        self.conv_blocks = nn.Sequential(
            ConvNeXtBlock1D(hidden_size, hidden_size // 2, stride=1),
            ConvNeXtBlock1D(hidden_size // 2, hidden_size // 2, stride=2),
            ConvNeXtBlock1D(hidden_size // 2, hidden_size // 4, stride=1),
            ConvNeXtBlock1D(hidden_size // 4, hidden_size // 4, stride=1),
            ConvNeXtBlock1D(hidden_size // 4, hidden_size // 4, stride=1),
            ConvNeXtBlock1D(hidden_size // 4, hidden_size // 4, stride=2)
        )

        self.cls = nn.Sequential(nn.Linear(hidden_size, 1),
                                 #nn.GELU(),
                                 #nn.Linear(200, 1),
                                 )
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.CrossEntropyLoss()
    def forward_logit_linear(self, x, mask):
        #with torch.no_grad():
        x = self.T5_encoder.encoder(x,attention_mask=mask,output_hidden_states = True,return_dict=True,)
        x = x.hidden_states[-2].mean(dim =1)  # take the last k transformer layers
        x = x.reshape(x.size(0), -1)
        x = self.cls(x)
        return x
    def forward_logit_cov(self, x, mask):
        x = self.T5_encoder.encoder(x,attention_mask=mask,output_hidden_states = True,return_dict=True,)
        x = x.hidden_states[-2]#[:,:12,:]
        x = x.permute(0,2,1)
        x = self.conv_blocks(x)
        x = x.reshape(x.size(0), -1)
        x = self.cls(x)
        return x

    def forward(self, x,mask,label):
        x = self.forward_logit_linear(x,
                          mask,
                          )

        loss = self.loss_fn(x.reshape(-1), label)

        return loss
