import torch
import torch.nn as nn
from einops import rearrange, repeat
from .losses import build_loss_evaluator

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot ProductAttention
    """
    def __init__(self,dropout=0.2):
        super(ScaledDotProductAttention,self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask=None):
        """
        q,k,v: tensor, [b,nhead,len,dim]

        """
        scale = q.shape[-1] ** 0.5 
        attn = torch.matmul(q/scale,k.transpose(2,3))  # [b,nhead,len,len]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
    
class MultiHeadAttention(nn.Module):
    """
    Contains attention operation and Add & Norm operation
    """

    def __init__(self,d_model, d_k, d_v, n_head = 4, dropout=0.2):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
       
    def forward(self, q, k, v, mask=None):
        # Here, Q,K,V is input features shape:[len,n_emb]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) #[b,len,n_head,d_k]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) #[b,len,n_head,d_k]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) #[b,len,n_head,d_v]

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,nhead,len,d_k]

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        #q (sz_b,len_q,n_head,N * d_k)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)
        return q, attn  

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.2):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.nn.functional.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class SlfAttnLayer(nn.Module):
    """ Compose with MultiHeadAttention and FeedForward layer """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(SlfAttnLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class CrossAttnLayer(nn.Module):
    """
    Compose with MultiHeadAttention(cross attention) and FeedForward layer
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.2):
        super(CrossAttnLayer, self).__init__()
        self.cross_attn = MultiHeadAttention(d_model, d_k, d_v, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_feat, dec_feat, dec_enc_attn_mask=None):
        output, cross_attn = self.cross_attn(enc_feat, dec_feat, dec_feat, dec_enc_attn_mask)
        output = self.pos_ffn(output)
        return output, cross_attn

class FusionBlock(nn.Module):
    """
    Compose with self AttnLayer and cross AttnLayer
    """
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.2, n_head=4):
        super(FusionBlock, self).__init__()
        self.slf_attn_layer = SlfAttnLayer(d_model, d_inner, n_head, d_k, d_v)
        self.cross_attn_layer = CrossAttnLayer(d_model, d_inner, n_head, d_k, d_v)

    def forward(self, emb_feat, dec_feat, slf_attn_mask=None, dec_enc_attn_mask=None):
        slf_attn_output, slf_attn = self.slf_attn_layer(emb_feat,slf_attn_mask)
        output, cross_attn = self.cross_attn_layer(slf_attn_output,dec_feat,dec_enc_attn_mask)
        return output, slf_attn, cross_attn

class SelfAttn(nn.Module):
    """
    Self attenion module
    """
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.2, n_layers=2, n_head=4):
        super(SelfAttn,self).__init__()
        self.layers = nn.ModuleList([
                      SlfAttnLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) 
                      for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, emb_feat, slf_attn_mask=None, return_attns=False):
        emb_feat = self.layer_norm(emb_feat)
        slf_attn_list = []
        for enc_layer in self.layers:
            emb_feat, enc_slf_attn = enc_layer(emb_feat,slf_attn_mask)
            slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return emb_feat, slf_attn_list
        return emb_feat

class CrossAttn(nn.Module):
    """
    Fusion module
    """
    def __init__(self, d_model, d_inner, d_k, d_v, dropout=0.2, n_layers=4, n_head=4):
        super(CrossAttn, self).__init__()
        self.fusionlayers = nn.ModuleList([FusionBlock(d_model,d_inner,d_k,d_v,dropout,n_head) 
                                           for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, emb_feat, dec_feat, slf_attn_mask=None, dec_enc_attn_mask=None, return_attns=False):
        slf_attn_list = []
        cross_attn_list = []
        dec_feat = self.layer_norm(dec_feat)
        for fusion_layer in self.fusionlayers:
            emb_feat, slf_attn, cross_attn = fusion_layer(emb_feat, dec_feat, slf_attn_mask, dec_enc_attn_mask)
            slf_attn_list += slf_attn
            cross_attn_list += cross_attn
        if return_attns:
            return emb_feat, slf_attn_list, cross_attn_list
        return emb_feat

class BaseSlfAttnModel(nn.Module):
    """
    Base model implement using self attention.
    """
    def __init__(self, cfg, emb_size=1280, d_inner=2048, 
                 n_slf_layers=2, n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(BaseSlfAttnModel, self).__init__()
        self.slf_attn_block = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                       dropout=dropout, n_layers=n_slf_layers, n_head=n_head)
        self.cls_head = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        if cfg.benchmark_name == 'amp_cls':
            self.mode = 'cls'
        elif cfg.benchmark_name == 'amp_reg':
            self.mode = 'reg'
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        esm_emb = input_data['batch_tokens']
        slf_attn_output = self.slf_attn_block(esm_emb)
        cls_token = slf_attn_output[:,0,:]
        output_logits = self.cls_head(cls_token)
        output_logits = output_logits.squeeze(dim=-1)
        gt = input_data['label'] if self.mode == 'cls' else input_data['mic']
        if self.training:
            loss_dict = self.loss_evaluator(output_logits,gt)
            return loss_dict
        else:
            results = dict(
                model_outputs = output_logits,
                labels = gt
            )
            return results

class BaseStcClsModel(nn.Module):
    """
    Base cls model using stc_info
    """
    def __init__(self, stc_size=676, emb_size=1280, n_classes=1):
        super(BaseStcClsModel,self).__init__()
        self.emb_layer = nn.Sequential(
            nn.Linear(stc_size, emb_size),
            )

        self.cls_head = nn.Sequential(
            
            nn.Linear(emb_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512, eps=1e-6),
            nn.Linear(512, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Linear(64,n_classes)
        )
        
    def forward(self, data):
        emb = self.emb_layer(data)
        pred_results = self.cls_head(emb)
        pred_results = pred_results.squeeze(dim=-1)
        return emb,pred_results

class BaseCrossAttnModel(nn.Module):
    """
    Base model implement using self attention & cross attention.
    """
    def __init__(self, cfg, stc_size=676, emb_size=1280, d_inner=2048, 
                 n_slf_layers=2, n_cross_layers=1, n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(BaseCrossAttnModel, self).__init__()

        self.linear_emb = nn.Linear(stc_size,emb_size)
        self.slf_attn_block = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                       dropout=dropout, n_layers=n_slf_layers, n_head=n_head)

        self.fusion_block = CrossAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)

        self.cls_head = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))

        if cfg.benchmark_name == 'amp_cls':
            self.mode = 'cls'
        elif cfg.benchmark_name == 'amp_reg':
            self.mode = 'reg'
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        esm_emb = input_data['batch_tokens']
        stc_info = input_data['stc']
        stc_emb = self.linear_emb(stc_info)
        stc_emb = stc_emb.unsqueeze(1) #[n, 1, emb_size]
        slf_attn_output = self.slf_attn_block(esm_emb)
        cross_attn_output = self.fusion_block(slf_attn_output,stc_emb)
        cls_token = cross_attn_output[:,0,:]
        output_logits = self.cls_head(cls_token)
        output_logits = output_logits.squeeze(dim=-1)
        gt = input_data['label'] if self.mode == 'cls' else input_data['mic']
        if self.training:
            loss_dict = self.loss_evaluator(output_logits,gt)
            return loss_dict
        else:
            results = dict(
                model_outputs = output_logits,
                labels = gt
            )
            return results

class SlfAttnEnsembelModel(nn.Module):
    """
    Ensemble model of slf attn
    """
    def __init__(self, cfg, emb_size=1280, d_inner=2048, 
                 n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(SlfAttnEnsembelModel, self).__init__()
        self.slf_attn_block1 = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                        dropout=dropout, n_layers=1, n_head=n_head)
        self.slf_attn_block2 = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                        dropout=dropout, n_layers=2, n_head=n_head)
        self.slf_attn_block3 = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                        dropout=dropout, n_layers=3, n_head=n_head)
        self.cls_head1 = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.cls_head2 = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))
        self.cls_head3 = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))

        if cfg.benchmark_name == 'amp_cls':
            self.mode = 'cls'
        elif cfg.benchmark_name == 'amp_reg':
            self.mode = 'reg'
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.loss_evaluator = build_loss_evaluator(cfg)
    
    def forward(self, input_data):
        esm_emb = input_data['batch_tokens']
        slf_attn_output1 = self.slf_attn_block1(esm_emb)
        slf_attn_output2 = self.slf_attn_block2(esm_emb)
        slf_attn_output3 = self.slf_attn_block3(esm_emb)
        cls_token1 = slf_attn_output1[:,0,:]
        cls_token2 = slf_attn_output2[:,0,:]
        cls_token3 = slf_attn_output3[:,0,:]
        output_logit1 = self.cls_head1(cls_token1)
        output_logit2 = self.cls_head2(cls_token2)
        output_logit3 = self.cls_head3(cls_token3)
        output_logit1 = output_logit1.squeeze(dim=-1)
        output_logit2 = output_logit2.squeeze(dim=-1)
        output_logit3 = output_logit3.squeeze(dim=-1)
        pred_output = (output_logit1+output_logit2+output_logit3)/3
        gt = input_data['label'] if self.mode == 'cls' else input_data['mic']
        if self.training:
            loss_dict = self.loss_evaluator(pred_output,gt)
            return loss_dict
        else:
            results = dict(
                model_outputs = pred_output,
                labels = gt
            )
            return results

class MultiModalFusionModel(nn.Module):
    """
    Use cross attention to fuse structure and esm pretraining info
    """
    def __init__(self, cfg, stc_size=676, emb_size=1280, d_inner=2048, 
                 n_slf_layers=3, n_cross_layers=3, n_head=4, d_k=1280, d_v=1280, dropout=0.2):
        super(MultiModalFusionModel, self).__init__()

        self.base_stc_model = BaseStcClsModel(stc_size=stc_size, emb_size=emb_size, n_classes=1)
        self.slf_attn_block = SelfAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v, 
                                       dropout=dropout, n_layers=n_slf_layers, n_head=n_head)
        self.fusion_block = CrossAttn(d_model=emb_size, d_inner=d_inner, d_k=d_k, d_v=d_v,
                                       dropout=dropout, n_layers=n_cross_layers, n_head=n_head)

        self.cls_head_final = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))

        self.cls_head = nn.Sequential(nn.Linear(emb_size, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, 1))

        if cfg.benchmark_name == 'amp_cls':
            self.mode = 'cls'
        elif cfg.benchmark_name == 'amp_reg':
            self.mode = 'reg'
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        self.loss_evaluator = build_loss_evaluator(cfg)

    def forward(self, input_data):
        esm_emb = input_data['batch_tokens']
        stc_info = input_data['stc']
        stc_emb,stc_pred = self.base_stc_model(stc_info)
        stc_emb = stc_emb.unsqueeze(1) #[n, 1, emb_size]
        slf_attn_output = self.slf_attn_block(esm_emb)
        # esm output 
        slf_cls_token = slf_attn_output[:,0,:]
        slf_output_logits = self.cls_head(slf_cls_token)
        slf_pred = slf_output_logits.squeeze(dim=-1)

        # final model output
        cross_attn_output = self.fusion_block(slf_attn_output,stc_emb)
        cls_token = cross_attn_output[:,0,:]
        output_logits = self.cls_head_final(cls_token)
        final_pred = output_logits.squeeze(dim=-1)
        gt = input_data['label'] if self.mode == 'cls' else input_data['mic']
        pred_output = {'stc_pred':stc_pred,'slf_pred':slf_pred,'final_pred':final_pred}
        if self.training:
            loss_dict = self.loss_evaluator(pred_output,gt)
            return loss_dict
        else:
            results = dict(
                # model_outputs = pred_logits,
                model_outputs = final_pred,
                labels = gt
            )
            return results