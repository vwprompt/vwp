from collections import OrderedDict, defaultdict
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from exp import ex

from utils import flatten_list, chunks

from .temporal_corr import TemporalCorrGlobal
from .pretrain_aux import PretrainAuxGroupRoll, ConcatAux
from .collaborative_experts import CollaborativeExperts
from .ss_loss import calc_loss
from .transformer_dis import TransformerDis, FeatureEncoder
from .transformer_dis_group import TransformerDisGroup

from .modules import Residual, MLP, TopDownAttnLayer
from .encoders import DeepEncoder



class ConcatNoGtSos(ConcatAux):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class NoGtSos(TemporalCorrGlobal):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class TAPM(NoGtSos):
    def _process_reg_loss(self, o, context, features, group_mask):
#         print('o.shape', o.shape)
#         print('context.shape', context.shape)
#         print('features.keys()', features.keys())
#         print('features.values()[0].shape', features['swin_base'].shape)
#         print('features.values()[1].shape', features['swin_object'].shape)
#         print('group_mask.shape', group_mask.shape)
#         print('group_mask', group_mask)
        
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(features, o.detach(), feature, group_mask)
        
        return reg_loss, stats
    
    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only
    
    
    
# TODO: need to refactor this part
# model for TACL submission
class TapmEntityGrid(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples, attn_type):
        super().__init__(transformer, tokenizer, dropout_before, num_samples)

        self.frame_start_linear = None
        self.frame_end_linear = None
        
        self.coherence_dim = 2000
        
        self.topDown_attn = TopDownAttnLayer(self.gpt_dim, attn_type)
        self.W_x = nn.Linear(self.coherence_dim, self.gpt_dim, False)
        
    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only
        
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
        # print('size', B, G, L, C)
        
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
        
        return features, features_merged, keywords, G
    
    
    
# TODO: need to refactor this part
class CharGridCoherence(NoGtSos):
    @ex.capture
    def __init__(self, transformer, tokenizer, dropout_before, num_samples, attn_type):
        super().__init__(transformer, tokenizer, dropout_before, num_samples)

        self.frame_start_linear = None
        self.frame_end_linear = None
        
        self.coherence_dim = 2000
        
        self.topDown_attn = TopDownAttnLayer(self.gpt_dim, attn_type)
        self.W_x = nn.Linear(self.coherence_dim, self.gpt_dim, False)
        
    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only
        
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
        # print('size', B, G, L, C)
        
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
        
        return features, features_merged, keywords, G
    
    
    
class NoGtSosOrig(PretrainAuxGroupRoll):
    def _process_reg_loss(self, o, context, features, group_mask, G):
        o = self.mean_pool_text(o)
        features = OrderedDict(sorted(features.items()))  # canonical ordering
        feature = self.mean_pool_features(list(features.values()))

        reg_loss, stats = self.get_reg_loss(o.detach(), feature, group_mask)
        return reg_loss, stats

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only


class NoGtGen(TemporalCorrGlobal):
    auxloss_generate = True


'''
class NoGtEos(TemporalCorrGlobal):
    def run_token(B, self, hypo, features, keywords):
        # hypo: (BG)L
        if self.training:
            if self.net.transformer.weight_freezed:
                hypo = torch.LongTensor(

        o, context = self.run_transformer(hypo, features, keywords)
        # (BG)LC
        logits = self.net.lm_head(o)

        return logits, o, context

    def mean_pool_text(self, o):
        # BGLC
        return o[:, :, 0]  # use the [sos] token only
'''
