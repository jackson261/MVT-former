import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_inverted
from layers.rope.rope_encoder import RoPEEncoderLayer 
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        self.gate_proj = nn.Sequential(nn.Linear(2 * configs.d_model, configs.d_model),nn.Sigmoid())
       


        self.configs = configs 
        self.norm = nn.LayerNorm(configs.d_model)
        self.enc_embedding_feature = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout,n_vars=configs.enc_in, use_vpe=True)
        

        self.encoder_feature = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.enc_embedding_time = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, pos_type='rope')    # ★ 不在embedding里相加绝对PE
        
        
        
        
        self.encoder_time = Encoder(
            [
            
                 RoPEEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,   
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation         
            
                 ) for _ in range(configs.e_layers)   
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.cross_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            configs.d_model, configs.n_heads
        )

    
        self.ffn = nn.Sequential(
            
            nn.Linear(configs.d_model, configs.d_ff * 4),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff * 4, configs.d_model),
            
        )


        
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        _, _, N = x_enc.shape # B L N 
        
        enc_out_feature = self.enc_embedding_feature(x_enc, x_mark_enc)  # [B, N, E]
        enc_out_feature, _ = self.encoder_feature(enc_out_feature, attn_mask=None)  # [B, N, E]

        
        enc_out_time = self.enc_embedding_time(x_enc, x_mark_enc)  # [B, L, E]
        enc_out_time, _ = self.encoder_time(enc_out_time, attn_mask=None)  # [B, L, E]

        
        fused_out = self.cross_attention(
            enc_out_feature,  # queries [B, N, E]
            enc_out_time,     # keys [B, L, E]
            enc_out_time,     # values [B, L, E]
            attn_mask=None
        )[0]  # [B, N, E]

        
        gate = self.gate_proj(torch.cat([enc_out_feature, fused_out], dim=-1))  # [B,N,E] in (0,1)
        fused_out = gate * fused_out + (1.0 - gate) * enc_out_feature     
        
        
        

        
        ffn_input = self.norm(fused_out)          
        ffn_output = self.ffn(ffn_input)          
        fused_out = ffn_output + ffn_input        
        
        
        



        dec_out = self.projector(fused_out).permute(0, 2, 1)[:, :, :N]  # [B, pred_len, N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

   
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]