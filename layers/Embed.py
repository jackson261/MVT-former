import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        
        pe = torch.zeros(max_len, d_model).float()
        
        
        pe.requires_grad = False            
        

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
       
        w.requires_grad = False
       


        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.trunc_normal_(self.pe.weight, std=0.02)
    def forward(self, x):
        L = x.size(1)
        idx = torch.arange(L, device=x.device).long().unsqueeze(0)  # [1,L]
        return self.pe(idx)                                 


class DataEmbedding(nn.Module):
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 pos_type='sin', max_len=5000):
    
        super(DataEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

       
        self.pos_type = pos_type
        if pos_type == 'sin':
            self.position_embedding = PositionalEmbedding(d_model=d_model)
 
        elif pos_type == 'learned':
            self.position_embedding = LearnedPositionalEmbedding(max_len, d_model)

        elif pos_type in ('none', 'rope'):
            self.position_embedding = None

        else:
            raise ValueError("pos_type must be 'sin'|'learned'|'none'|'rope'")
        
       
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
                      
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)  
        return self.dropout(x)  





class DataEmbedding_inverted(nn.Module):
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,
                 n_vars=None, use_vpe=False):            
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        
        self.use_vpe = use_vpe
        self.n_vars = n_vars
        if use_vpe:
            assert n_vars is not None, 
            self.var_index_emb = nn.Embedding(n_vars, d_model)
            self.register_buffer("var_ids_buf", torch.arange(n_vars, dtype=torch.long), persistent=False)


    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [B, N, L]    
        x = self.value_embedding(x)  # [B, N, E] 
       
        
        
       
        if self.use_vpe:
            B, N, _ = x.shape
            var_ids = self.var_ids_buf.unsqueeze(0).expand(B, -1)  # [B,N]
            x = x + self.var_index_emb(var_ids)                    
        return self.dropout(x)       
       
