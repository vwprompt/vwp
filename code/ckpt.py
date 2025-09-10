import json
from pathlib import Path, PosixPath

import torch

from exp import ex
from model import get_model
from path import get_dirname_from_args
from data.dataloader import get_datasets




@ex.capture
def get_ckpt_path(epoch, key_dt, ckpt_path, isBest=False):
    ckpt_name = get_dirname_from_args()
    ckpt_path = ckpt_path / ckpt_name
    ckpt_path.mkdir(exist_ok=True, parents=True)
    key, val = list(key_dt.items())[0]
    val = '{:.2f}'.format(val*100)
    ckpt_name = f'{key[0]}_{val}_ep_{epoch:03}'
    ckpt_suffix = '.pickle'
    if isBest:
        ckpt_fname = ckpt_name + '_best' + ckpt_suffix
    else:
        ckpt_fname = ckpt_name + ckpt_suffix
    args_name = f'args.json'

    return ckpt_path, args_name, ckpt_fname


@ex.capture
def save_ckpt(epoch, key_dt, model, tokenizer, _config, isBest=False):
    print(f'saving epoch {epoch}')
    args = _config
    dt = {
        'epoch': epoch,
        **key_dt,
        'model': model.state_dict(),
        'tokenizer': tokenizer
    }

    ckpt_path, args_name, ckpt_name = get_ckpt_path(epoch, key_dt, isBest=isBest)
    print(f"Saving checkpoint {ckpt_path / ckpt_name}")
    torch.save(dt, ckpt_path / ckpt_name)
    args_path = ckpt_path / args_name
    if not args_path.is_file():
        with open(args_path, 'w') as f:
            json.dump({k: str(v) for k, v in args.items()}, f)


@ex.capture
def load_ckpt(ckpt_path, ckpt_name):
    root = Path('../').resolve()
    if str(root) not in str(ckpt_path):
        ckpt_path = root / ckpt_path
    ckpt_paths = sorted(ckpt_path.glob(f'{ckpt_name}*'), reverse=False)
    
    #print('ckpt_path', ckpt_path)
    #print('ckpt_paths', ckpt_paths)
    assert len(ckpt_paths) > 0, f"no ckpt candidate for {ckpt_path / ckpt_name}"
    
    # TH: always choose the last one
    ckpt_file_path = ckpt_paths[0]  # monkey patch for choosing the best ckpt; 
    
    print(f"loading from {ckpt_file_path}")
    dt = torch.load(ckpt_file_path)

    return dt, ckpt_file_path


@ex.capture
def get_model_ckpt(model_name, ckpt_name, data_path, pretrain_path,
                   use_data, device, feature_names,
                   transformer_name=None):
    ckpt_available = ckpt_name is not None
    if ckpt_available:
        dt, ckpt_path = load_ckpt()
    else:
        ckpt_path = None

#     print ('get_model_ckpt: ckpt_path', ckpt_path)
#     print ('get_model_ckpt: ckpt_name', ckpt_name)
#     print ('get_model_ckpt: data_path', data_path)
#     print ('get_model_ckpt: pretrain_path', pretrain_path)
#     print ('dt: dt.keys()', dt.keys())
#     print ('get_model_ckpt: dt["model"]', dt["model"].keys())
#     print ('get_model_ckpt: type(dt["model"])', type(dt["model"]))

    datasets = get_datasets(data_path, pretrain_path)
    
    model, tokenizer = get_model(model_name, datasets['target'][use_data[0]].data,
                                 transformer_name=transformer_name)
    model = model.to(device)
    epoch = 0
    
    print ('get_model_ckpt: model', model)
    print ('get_model_ckpt: type(model)', type(model))

    if ckpt_available:
        if 'net.transformer.word_embedding.weight' in dt['model']:
            model.net.transformer.word_embedding = model.net.transformer.wte
        elif hasattr(model.net.transformer, 'word_embedding'):
            del model.net.transformer.word_embedding
        
        '''
        # TH: add 'encoder' to state_dictj. But the purpose is not clear
        new_dt = {}
        remove_keys = []
        for name in feature_names:
            for key in dt['model'].keys():
                if key.startswith(f"{name}."):
                    new_key = key.split('.')
                    if len(new_key) > 2 and new_key[2] in ['linear_in', 'linear2', 'res_layers']:
                        new_key = '.'.join(new_key[:2] + ['encoder']  + new_key[2:])
                        new_dt[new_key] = dt['model'][key]
                        remove_keys.append(key)
        dt['model'] = {**dt['model'], **new_dt}
        for key in remove_keys:
            del dt['model'][key]
        '''
        
        model.load_state_dict(dt['model'])
        
        #TH: no need to load the tokenizer from dt 
        #tokenizer = dt['tokenizer']
        
        epoch = dt['epoch']
        model.ckpt_path = ckpt_path
        model.ckpt_name = ckpt_name
        
        
        
    return model, tokenizer, ckpt_path, datasets, epoch
