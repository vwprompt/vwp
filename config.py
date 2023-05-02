default_args = {
    ## Mostly overwrited by command line input
    'model_name': 'no_gt_sos', # hybrid_dis
    'log_tag': 'global', # Brief descriptive tag for logdir readibility
    'feature_names': ['swin_base'],
    
    ## Datasets
    ### VWP_anonSq
    'vist_path': {
#         'train': 'data/database/vwp/VWP_anonSq_train_dict.pkl', 
#         'val': 'data/database/vwp/VWP_anonSq_val_dict.pkl',
#         'test': 'data/database/vwp/VWP_anonSq_test_dict.pkl',
        'train': 'data/database/vwp/VWP_anonSq_CharFaceOracle_ID_train_dict.pkl', 
        'val': 'data/database/vwp/VWP_anonSq_CharFaceOracle_ID_val_dict.pkl',
        'test': 'data/database/vwp/VWP_anonSq_CharFaceOracle_ID_test_dict.pkl',
#         'test': 'data/database/vwp/smallVIST_anonSq_smallTest_dict.pkl',

#        'test': 'data/database/vwp/mixed_anonSq_mixedTest_dict.pkl',
    },

    ### VWP_anonVIST ???
#     'vist_path': {
#         'train': 'data/database/vwp/VWP_anonSq_char_train_dict.pkl', 
#         'val': 'data/database/vwp/VWP_anonSq_char_val_dict.pkl',
#         'test': 'data/database/vwp/VWP_anonSq_char_test_dict.pkl',
#        'test': 'data/database/vwp/mixed_anon_mixedTest_dict.pkl',
#     },

    ### VIST_anonSq    
#     'vist_path': {
#         'train': 'data/database/vwp/VIST_anonSq_train_dict.pkl', 
#         'val': 'data/database/vwp/VIST_anonSq_val_dict.pkl',
#         #'test': 'data/database/vwp/VIST_anonSq_test_dict.pkl',
#         'test': 'data/database/vwp/VIST_anonSq_smallTest_dict.pkl',
# #        'test': 'data/database/vwp/mixed_anonSq_mixedTest_dict.pkl',
#     },
 
    ### VIST_anonSq    
#     'vist_path': {
#         'train': 'data/database/vwp/smallVIST_anonSq_train_dict.pkl', 
#         'val': 'data/database/vwp/smallVIST_anonSq_val_dict.pkl',
#         #'test': 'data/database/vwp/smallVIST_anonSq_test_dict.pkl',
# #         'test': 'data/database/vwp/smallVIST_anonSq_smallTest_dict.pkl',
#        'test': 'data/database/vwp/mixed_anonSq_mixedTest_dict.pkl',
#     },
    
    ### VIST
#     'vist_path': {
#         'train': 'data/database/vwp/VIST_train_dict.pkl', 
#         'val': 'data/database/vwp/VIST_val_dict.pkl',
#         'test': 'data/database/vwp/VIST_test_dict.pkl',
# #         'test': 'data/database/vwp/VIST_smallTest_dict.pkl',
# #        'test': 'data/database/vwp/mixed_anonVIST_mixedTest_dict.pkl',
#     },
    
    ### VIST original
#     'vist_path': {
#         'train': 'data/database/sis/train.story-in-sequence.json',
#         'val': 'data/database/sis/val.story-in-sequence.json',
#         'test': 'data/database/sis/test.story-in-sequence.json',
#     },

    'task_name': 'vwp',
    'fix_gpt_epoch': 0,
    'use_person_token': False,
    'use_model_tokenizer': False,
    'use_keyword': False,
    'attn_type' : 'dot', 
    'transformer_name': 'gpt2',  # 'gpt2' by default, 
    'sampling_method': 'greedy',  # ['beam', 'greedy', 'topk', 'nucleus', 'max_nucleus']
    'use_topdown': False, 
    'use_entity_grid': False, 
    'use_regression_loss': '', # ['C', 'N', 'CN']
    'coherence_loss_type': 'max', 
    'use_narrativity_loss': False, 
    
    'use_char_id_loss': False, 
    'reg_loss_before_lm': False, 
    
    'n_neg_samples' : 1,
    'use_bitfit': False, 
    
    'batch_sizes': {'train': 6, 'val': 6, 'test': 6}, 
    'max_epoch': 20,
    'store_ckpt': True,
    'num_workers': 0,
    'random_seed': 0,

    'log_keys': [
        'log_tag',
        'model_name',
#         'sample',
#         'task_name',
    ],
    
    'logger_type': ['tfboard'],
    'hold_after_finishing' : False,
    'reg_coeff': 1,
    'keyword_threshold': 1/3,
    'transformer_pool': 'mean',
    'threshold_gap': 3,
    'use_gt_keyword': False,
    'use_word_subset': False,
    'concat_group': False,
    'ss_loss_type': 'ranking',  # ['ranking', 'bce', 'l2']
    'feature_name_map': {
        'i3d_rgb': 'video',
        'i3d_flow': 'flow',
        'rcnn': 'box',
        'scene': 'scene',
        'vggish': 'sound',
        'human_i3d': 'human_i3d',
        'c3d': 'c3d',
        'vilbert': 'vilbert',
        'vilbert_tune_5': 'vilbert_tune_5',
        'lxmert': 'lxmert',
        
        'resnet': 'resnet',
        'rcnn_pad': 'object_feat',
        'resnet_char': 'resnet_char',
        
        'swin_base': 'swin_base',
        'swin_object': 'swin_object',
        'swin_char_spec': 'swin_char_spec',
        'swin_char_oracle': 'swin_char_oracle',
        
        'char_bodyF_or': 'char_bodyF_or',
        'char_upperBodyF_or': 'char_upperBodyF_or',
        'char_headF_or': 'char_headF_or',
        'char_face_or': 'char_face_or',
        
        'person_id_or': 'person_id_or',
    },
    'feature_dims': {
        'video': 1024,
        'flow': 1024,
        'box': 1600,
        'scene': 365,
        'sound': 128,
        'human_i3d' : 1536,
        'c3d': 500,
        'vilbert': 1024,
        'vilbert_tune_5': 1024,
        'lxmert': 768,
        
        'resnet': 2048,
        'object_feat': 2048,
        'resnet_char': 2048,
        'swin_base': 1024,
        'swin_object': 2048,
        'swin_char_spec': 2048,
        'swin_char_oracle': 2048,
        
        'char_bodyF_or': 2048,
        'char_upperBodyF_or': 2048,
        'char_headF_or': 2048,
        'char_face_or': 2048,
        'person_id_or': 1,
    },
    'cross_modal_names': ['vilbert', 'lxmert', 'vilbert_tune_5'],
    'cut_feature_temporal_dim': {},  # e.g. rcnn_pad: 18
    'keyword_ratio': 0.4,
    'latent_std': 1,
    'extract_keyword': False,
    'extraction_threshold': 1,
    'extraction_min_words': 2,
    'use_data': ['train', 'val', 'test'],
    'metrics': ['meteor', 'bleu', 'rouge', 'cider'],
    'segment_pool_type': 'interval',  # ['interval', 'boundary', None]
    'max_vocab': None,
    'pretrained_embedding': None,
    # 'pretrained_embedding': 'glove.840B.300d',

    'force_ascii': True,
    'reinforce_epoch': 1000,
    'reinforce_metrics': ['meteor'],
    'reinforce_group': False,  # reinforce concatenated group
    'reinforce': False,
    'length_normalize_beam': False,
    'sample_ema_coeff': 1,
    'sample_eval_at_last': True,
    'nvlad_size': 64,
    'concentration': 100,
    'transformer_size': 'small',
    'new_beam_seach': False,
    'postprocess_duplicates': 1,  # 1 indicated no postprocessing

    'max_segments': 3,
    'eval_after': 1,
    'eval_every': 1,
    'eval_every_iter': 1e+10,
    'eval_generate': True,
    'eval_metric': True,
    'max_sentence_tokens': 100,
    'max_target_len': 70,  # max bpe token num in target: 69
    'max_target_sent': 5,  # max sent for ActivityNet Captions
    'sampling_k': 8,
    'sampling_p': 0.9,
    'num_samples': 1,
    'normalizer_sparsity': None,

    'learning_rate': 5e-5,
    'dropout': 0.5,
    'visual_dropout': 0.0,
    'change_transformer_dropout': False,
    'label_smoothing': 0,  # 0.1 for Attention is All You Need
    'warmup_steps': 4000,
    'grad_clip': None,
    'grad_acc_steps': 1,
    'all_batch_size': None,
    'eval_subset': None,

    'eval_test': False,
    'eval_set': False,  # evaluate the group-concatenated text
    'use_lsmdc16': False,
    'use_vist': True,
    'use_fib': False,
    'use_multichoice': False,
    'repeat_vist_image': 1,  # previously 3, for unknown reasons
    'vist_sample_longest': False,
    'train_path': 'data/LSMDC/task1/LSMDC16_annos_training_val.csv',
    'val_path': 'data/LSMDC/task1/LSMDC16_annos_test.csv',
    'test_path': 'data/LSMDC/task1/LSMDC16_annos_test.csv',

    'fib_path': {
        'train': 'data/LSMDC/fib/LSMDC16_FIB_train.csv',
        'val': 'data/LSMDC/fib/LSMDC16_FIB_val.csv',
    },
    'multichoice_path': {
        'train': 'data/LSMDC/multichoice/LSMDC16_MC_train.csv',
        # 'val': 'data/LSMDC/multichoice/LSMDC16_MC_val.csv',
        'val': 'data/LSMDC/multichoice/LSMDC16_MC_test.csv',
        'test': 'data/LSMDC/multichoice/LSMDC16_MC_test.csv',
    },
    'add_target_to_pretrain': False,
    'train_pretrain_path': None,
    'val_pretrain_path': None,
    'test_pretrain_path': None,
    'use_actynetcap': False,
    'actynetcap_stride': 48,
    'actynetcap_path': {
        'train': 'data/ActyNetCap/actynetcap/train.json',
        'val': 'data/ActyNetCap/actynetcap/val_1.json',
        'test': 'data/ActyNetCap/actynetcap/val_2.json',
    },
    #'keyword_dir': 'keywords_top1000',
    #'keyword_name': 'keywords_gpt_top_1000.json',
    'keyword_name': None,

    'sample': False,
    'debug': False,
    'log_cmd': False,
    'log_multi': False, # Display multiple runs in a single board
    'log_path': 'data/log',
    'log_text_every': 2000,
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'database_path': 'data/database',
    'evals_path': 'data/eval_out',
    'hostname': 'localhost',

    'scripts': {
        'filename': 'publictest_results.json',
        'path': '../../data/LSMDC/task2/LSMDC16_annos_training_blank.csv',
        'relation_file': 'relation.pkl',
        'num_keywords': 1000,
        'logdir': '../../data/log/*',
    },
}

debug_args = {
    'num_workers': 1,
    'sample': False,
    'eval_after': 0,
    'eval_every': 1,
    'fix_gpt_epoch': 1,
    'logger_type': ['cmd'],
    'store_ckpt': False,
}

reinforce_args = {
    'reinforce_epoch': 0,
    'eval_after': 0,
    'eval_every': 1,
    'fix_gpt_epoch': 0,
}


vist_args = {
    'fix_gpt_epoch': 6,
}
