from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from data.batcher import pad_tensor
from sampler import get_sampler
from run_transformer import transformer_lm_head
from utils import flatten_list, chunks

from .collaborative_experts import CollaborativeExperts
from .ss_loss import calc_loss
from .transformer_dis import TransformerDis, FeatureEncoder
from .transformer_dis_group import TransformerDisGroup

from .modules import Residual, MLP, TopDownAttnLayer
from .NetVLAD import NetVLADWrapper
from .encoders import DeepEncoder



class TopdownAttn(TransformerDisGroup):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples, attn_type):
        super().__init__(transformer, tokenizer, dropout_before, num_samples)

        self.frame_start_linear = None
        self.frame_end_linear = None
        
        self.topDown_attn = TopDownAttnLayer(self.gpt_dim, attn_type)
        
        
    def prepare_group(self, batch, use_keyword=False):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        global_key = list(features.keys())[0]
        local_key = list(features.keys())[1]        
        assert 'swin_base' in features.keys()
        assert 'swin_base' == global_key
        assert ('swin_char_spec' == local_key or 'swin_object' == local_key)

        keywords = None
        B = features[list(features.keys())[0]].shape[0]
        G = features[list(features.keys())[0]].shape[1]
        L = features[list(features.keys())[0]].shape[2]
        C = features[list(features.keys())[0]].shape[3]

        features = self.process_features(features)
        features_merged = {k: rearrange(v.contiguous(), 'b g l c -> (b g) l c') for k, v in features.items()}
        
        # Handle top-down attention locally
        global_input = features_merged[global_key].squeeze(1)
        local_input = features_merged[local_key]
        
        attn_res_merged = self.topDown_attn(local_input, global_input, local_input)
#         print('global_input.shape', global_input.shape)
#         print('local_input.shape', local_input.shape)
#         print('attn_res.shape', attn_res_merged.shape)
        
        attn_res = rearrange(attn_res_merged.contiguous(), '(b g) l c -> b g l c', g=G)
        features[local_key] = attn_res
        features_merged[local_key] = attn_res_merged
        
        return features, features_merged, keywords, G
    
    

class EntityGrid(TransformerDisGroup):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples, attn_type):
        super().__init__(transformer, tokenizer, dropout_before, num_samples)

        self.frame_start_linear = None
        self.frame_end_linear = None
        
        self.coherence_dim = 2000
        
        self.topDown_attn = TopDownAttnLayer(self.gpt_dim, attn_type)
        self.W_x = nn.Linear(self.coherence_dim, self.gpt_dim, False)
        
    def get_reg_loss(self, fs, o, f, mask):
        return None, {}
    
    
    def prepare_group(self, batch, use_keyword=False):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        global_key = list(features.keys())[0]
        local_key = list(features.keys())[1]
        entity_grid_key = 'entity_grid'
        narrativity_key = 'narrativity'
        
        assert 'swin_base' in features.keys()
        assert 'swin_base' == global_key
        assert ('swin_char_spec' == local_key or 'swin_object' == local_key)
        keywords = None
        
        features = self.process_features(features)
        features_merged = {k: rearrange(v.contiguous(), 'b g l c -> (b g) l c') for k, v in features.items()}
        
        B = features[global_key].shape[0]
        G = features[global_key].shape[1]
        L = features[global_key].shape[2]
        C = features[global_key].shape[3]
        O = features[local_key].shape[2]
#         print('size', B, G, L, C)
        print(B, G, O, C)
        
        # entity grid
        if L != 1:
            global_in = features[global_key][:,:,5:,:]
            local_in = features[local_key][:,:,5:,:]
        else:
            global_in = features[global_key]
            local_in = features[local_key]
            
        global_GL = rearrange(global_in.contiguous(), 'b g l c -> b (g l) c')
        local_GL = rearrange(local_in.contiguous(), 'b g l c -> b (g l) c')
        coherence_mat = torch.einsum('bqc,bkc->bqk', local_GL, global_GL)
        coherence_vec = rearrange(coherence_mat.contiguous(), 'b q k -> b (q k)')
        pad_length = self.coherence_dim - coherence_vec.shape[1]
        pad_coherence_vec = F.pad(coherence_vec, (0, pad_length), mode='constant', value=0)
        x = self.W_x(pad_coherence_vec)
        
        coherence_out = x.unsqueeze(1).unsqueeze(1).expand(B, G, 1, -1)
        coherence_merged = rearrange(coherence_out.contiguous(), 'b g l c -> (b g) l c')
        features[entity_grid_key] = coherence_out
        features_merged[entity_grid_key] = coherence_merged   
        
        
        # compute narrativity loss
#         'narrativity'
        narrativity_loss = 0
        
        
        # compute coherence loss
        #if self.use_coherence_loss:
        print(B, G, O, C)
        print('local_in.shape', local_in.shape)
        local_in_norm = torch.norm(local_in, dim=3).unsqueeze(3).expand(B, G, O, C)
        norm_local_in = local_in / local_in_norm
        norm_local_GL = rearrange(norm_local_in.contiguous(), 'b g l c -> b (g l) c')
        
        coherence_Sp = torch.einsum('bglc,bkc->bglk', norm_local_in, norm_local_GL)
        coherence_S = coherence_Sp.mean(dim=2) # bgk
        coherence_S1 = coherence_S[:, :-1, :]
        coherence_S2 = coherence_S[:, 1:, :]
        coherence_So = coherence_S2 - coherence_S1
        coherence_vec = rearrange(coherence_So.contiguous(), 'b q k -> b (q k)')
        coherence_loss = torch.norm(coherence_vec, dim=1)

#         print('coherence_S.shape', coherence_S.shape)
#         print('coherence_So.shape', coherence_So.shape)
#         print('coherence_vec.shape', coherence_vec.shape)
#         print('coherence_loss.shape', coherence_loss.shape)
#         print('coherence_loss', coherence_loss)        
        
        self.coherence_loss = coherence_loss
        self.narrativity_loss = narrativity_loss
        
        return features, features_merged, keywords, G, #coherence_loss, narrativity_loss
    
    
    
class FullEntityGrid(TransformerDisGroup):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples, attn_type):
        super().__init__(transformer, tokenizer, dropout_before, num_samples)

        self.frame_start_linear = None
        self.frame_end_linear = None
        
        self.coherence_dim = 2000
        
        self.topDown_attn = TopDownAttnLayer(self.gpt_dim, attn_type)
        self.W_x = nn.Linear(self.coherence_dim, self.gpt_dim, False)
        
    def get_reg_loss(self, o, f, mask):
        return None, {}
    
    
    def prepare_group(self, batch, use_keyword=False):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        global_key = list(features.keys())[0]
        obj_key = list(features.keys())[1]
        char_key = list(features.keys())[2]
        
        entity_grid_key = 'EG'
        assert 'swin_base' in features.keys()
        assert 'swin_base' == global_key
        assert ('swin_char_spec' == char_key and 'swin_object' == obj_key)
        keywords = None
        
        features = self.process_features(features)
        features_merged = {k: rearrange(v.contiguous(), 'b g l c -> (b g) l c') for k, v in features.items()}
        
        B = features[global_key].shape[0]
        G = features[global_key].shape[1]
        L = features[global_key].shape[2]
        C = features[global_key].shape[3]
#         print('size', B, G, L, C)
        
        # entity grid
        global_in = features[global_key]
        obj_in = features[obj_key]
        char_in = features[char_key]
#         TODO: concat features
        local_in = torch.cat([obj_in, char_in], dim=2)
        
        global_GL = rearrange(global_in.contiguous(), 'b g l c -> b (g l) c')
        local_GL = rearrange(local_in.contiguous(), 'b g l c -> b (g l) c')
        coherence_mat = torch.einsum('bqc,bkc->bqk', local_GL, global_GL)
        coherence_vec = rearrange(coherence_mat.contiguous(), 'b q k -> b (q k)')
        pad_length = self.coherence_dim - coherence_vec.shape[1]
        pad_coherence_vec = F.pad(coherence_vec, (0, pad_length), mode='constant', value=0)
        x = self.W_x(pad_coherence_vec)
        
        coherence_out = x.unsqueeze(1).unsqueeze(1).expand(B, G, 1, -1)
        coherence_merged = rearrange(coherence_out.contiguous(), 'b g l c -> (b g) l c')
        features[entity_grid_key] = coherence_out
        features_merged[entity_grid_key] = coherence_merged   
        
        return features, features_merged, keywords, G
    
    
    
class NormalisedEntityGrid(EntityGrid):
    def get_reg_loss(self, o, f, mask):
        return None, {}
    
    
    def prepare_group(self, batch, use_keyword=False):
        features = {k: val for k, val \
                    in {f: getattr(batch, f) for f \
                        in self.feature_names}.items()}
        global_key = list(features.keys())[0]
        local_key = list(features.keys())[1]
        entity_grid_key = 'EG'
        assert 'swin_base' in features.keys()
        assert 'swin_base' == global_key
        assert ('swin_char_spec' == local_key or 'swin_object' == local_key)
        keywords = None
        
        features = self.process_features(features)
        features_merged = {k: rearrange(v.contiguous(), 'b g l c -> (b g) l c') for k, v in features.items()}
        
        B = features[global_key].shape[0]
        G = features[global_key].shape[1]
        L = features[global_key].shape[2]
        C = features[global_key].shape[3]
#         print('size', B, G, L, C)
        
        # Handle top-down attention locally
#         global_BG = features_merged[global_key].squeeze(1)
#         local_BG = features_merged[local_key]
#         attn_res_BG = self.topDown_attn(local_BG, global_BG, local_BG)
#         attn_res = rearrange(attn_res_BG.contiguous(), '(b g) l c -> b g l c', g=G)
#         features[local_key] = attn_res
#         features_merged[local_key] = attn_res_BG
        
        # entity grid
        global_in = features[global_key]
        local_in = features[local_key]
        global_GL = rearrange(global_in.contiguous(), 'b g l c -> b (g l) c')
        local_GL = rearrange(local_in.contiguous(), 'b g l c -> b (g l) c')
        coherence_mat = torch.einsum('bqc,bkc->bqk', local_GL, global_GL)
        coherence_vec = rearrange(coherence_mat.contiguous(), 'b q k -> b (q k)')
        pad_length = self.coherence_dim - coherence_vec.shape[1]
        pad_coherence_vec = F.pad(coherence_vec, (0, pad_length), mode='constant', value=0)
        x = self.W_x(pad_coherence_vec)
        
        coherence_out = x.unsqueeze(1).unsqueeze(1).expand(B, G, 1, -1)
        coherence_merged = rearrange(coherence_out.contiguous(), 'b g l c -> (b g) l c')
        features[entity_grid_key] = coherence_out
        features_merged[entity_grid_key] = coherence_merged   
        
#         print('global_input.shape', global_input.shape)
#         print('local_input.shape', local_input.shape)
#         print('coherence_mat.shape', coherence_mat.shape)
#         print('coherence_vec.shape', coherence_vec.shape)
#         print('pad_coherence_vec.shape', coherence_vec.shape)
#         print('x.shape', x.shape)
#         print('coherence_out.shape', coherence_out.shape)

        
        return features, features_merged, keywords, G
    
    