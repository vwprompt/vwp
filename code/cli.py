import os
import json
from pathlib import Path, PosixPath

from exp import ex

from args import get_args, _get_args, get_ckpt_args
from train import _train
from utils import wait_for_key, count_parameters
from evaluate import _evaluate
from infer import _infer
from vis_tsne import _tsne, _silhouette
from distance import _distance
from extract_keyword import extract_and_save_all
from model import get_model_options
from ckpt import get_model_ckpt
from loss.loss import get_loss
from optimizer import get_optimizer
from data.dataloader import get_dataloaders
from logger import get_logger
from scripts import run_script

from config import default_args


@ex.capture
def prepare_model(model_name, transformer_name, ckpt_name):
    if ckpt_name is not None:
        args, Model = _get_args(default_args)
        args.ckpt_name = ckpt_name
        ckpt_args = get_ckpt_args(args)
        #print ('prepare: args', ckpt_args)

        transformer_name = args.transformer_name
        print ('prepare: model_name', model_name)
        print ('prepare: transformer_name', transformer_name)
        print ('prepare: ckpt_name', ckpt_name)
        
        model, tokenizer, ckpt_path, datasets, epoch = get_model_ckpt(model_name, ckpt_name, transformer_name=transformer_name)
        #print ('prepare: tokenizer', tokenizer)
        print ('prepare: ckpt_path', ckpt_path)
        print ('prepare: epoch', epoch)
    else:
        print ('prepare: model_name', model_name)
        print ('prepare: transformer_name', transformer_name)
        
        model, tokenizer, ckpt_path, datasets, epoch = get_model_ckpt(model_name)
        #print ('prepare: tokenizer', tokenizer)
        print ('prepare: ckpt_path', ckpt_path)
        print ('prepare: epoch', epoch)
    
    return model, tokenizer, ckpt_path, datasets, epoch


@ex.capture
def prepare(no_logger=False):
    logger = get_logger(log_file=no_logger)
    
    model, tokenizer, ckpt_path, datasets, epoch = prepare_model()
    
    dataloaders = get_dataloaders(datasets, model.make_batch, tokenizer)
    '''
    args.batch_per_epoch = {}
    for key in dataloaders.keys():
        args.batch_per_epoch[key] = \
            math.ceil(len(dataloaders[key]) / args.batch_sizes[key])
    '''
    loss_fn = get_loss(padding_idx=tokenizer.pad_id)
    optimizers = get_optimizer(model, dataloaders)
    model.ckpt_epoch = epoch

    return model, loss_fn, optimizers, tokenizer, dataloaders, logger


@ex.command
def train(hold_after_finishing):
    all_args = prepare()
    res = _train(*all_args)

    logger = all_args[-1]
    # hold process to keep tensorboard alive
    if 'tfboard' in logger.logger_dests and hold_after_finishing:
        wait_for_key()

    return res


@ex.command
def evaluate_val(evals_path, vist_path, sampling_method, sampling_k, sampling_p):
    model, loss_fn, optimizers, tokenizer, dataloaders, logger = prepare(no_logger=True)
    assert hasattr(model, 'ckpt_path'), "no ckpt loaded"
    root = Path('../').resolve()
    path = model.ckpt_path
    parent = root / evals_path
    dir_name = path.parent.stem
    ckpt_name = path.stem
    split_name_list = vist_path['test'].split('/')[-1].split('.')[0].split('_')[0:2]
    split_name = '_'.join(split_name_list)
    
    stats, _, hypos, score_texts, batch_stats = _evaluate(model, loss_fn, None, tokenizer, dataloaders['target'], logger, key='val', eval_generate=True)
    print(stats)
    
    if sampling_method == 'beam':
        suffix = f'_{sampling_method}'
    elif sampling_method == 'topk':
        suffix = f'_{sampling_method}_{str(sampling_k)}'
    elif sampling_method == 'max_nucleus':
        suffix = f'_{sampling_method}_{str(sampling_p)}'
    else:
        suffix = ''
        
    parent = parent / dir_name / split_name
    os.makedirs(parent, exist_ok=True)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_val_stats.json', 'w') as f:
        json.dump(stats, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_val_hypo.json', 'w') as f:
        json.dump(hypos, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_val_text.json', 'w') as f:
        json.dump(score_texts, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_val_batch_stats.json', 'w') as f:
        json.dump(batch_stats, f)


@ex.command
def evaluate_test(evals_path, vist_path, sampling_method, sampling_k, sampling_p):
    model, loss_fn, optimizers, tokenizer, dataloaders, logger = prepare(no_logger=True)
    assert hasattr(model, 'ckpt_path'), "no ckpt loaded"
    root = Path('../').resolve()
    path = model.ckpt_path
    parent = root / evals_path
    dir_name = path.parent.stem
    ckpt_name = path.stem
    split_name_list = vist_path['test'].split('/')[-1].split('.')[0].split('_')[0:2]
    split_name = '_'.join(split_name_list)
    
    # DEBUG
#     print (model)
#     print ('path', path)
#     print ('parent', parent)
#     print ('dir_name', dir_name)
#     print ('ckpt_name', ckpt_name)
#     print ('split_name', split_name)
#     print ('data_name', data_name)
    
    stats, _, hypos, score_texts, batch_stats = _evaluate(model, loss_fn, None, tokenizer, dataloaders['target'], logger, key='test', eval_generate=True)
    #print(stats)
    #print(batch_stats)

    if sampling_method == 'beam':
        suffix = f'_{sampling_method}'
    elif sampling_method == 'topk':
        suffix = f'_{sampling_method}_{str(sampling_k)}'
    elif sampling_method == 'max_nucleus':
        suffix = f'_{sampling_method}_{str(sampling_p)}'
    else:
        suffix = ''
        
    parent = parent / dir_name / split_name
    os.makedirs(parent, exist_ok=True)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_test_stats.json', 'w') as f:
        json.dump(stats, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_test_hypo.json', 'w') as f:
        json.dump(hypos, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_test_text.json', 'w') as f:
        json.dump(score_texts, f)
    with open(parent / f'{ckpt_name}{suffix}_{split_name}_test_batch_stats.json', 'w') as f:
        json.dump(batch_stats, f)

        
@ex.command
def tsne(log_path, test_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _tsne(*all_args, key='test')


@ex.command
def silhouette(log_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _silhouette(*all_args, key='test')


@ex.command
def distance(log_path):
    # all_args = prepare({'use_data': 'val', 'sample': True})
    all_args = prepare()
    _distance(*all_args, key='val')


@ex.command
def infer():
    #all_args = prepare({'use_data': 'val'})
    all_args = prepare()

    texts = _infer(*all_args)


@ex.command
def model_stats():
    #all_args = prepare({'use_data': 'val'})
    all_args = prepare(no_logger=True)
    model = all_args[0]

    stats = {}
    stats['parameter_counts'] = count_parameters(model)

    print(stats)


@ex.command
def extract():
    model, _, _, tokenizer, \
        dataloaders, _ = prepare()
    for dataloader in dataloaders.values():
        dataloader.training = False

    extract_and_save_all(model, tokenizer, dataloaders)


@ex.command
def scripts(script):
    run_script(script)


@ex.command
def print_models():
    print(sorted(get_model_options()))


@ex.option_hook
def update_args(options):
    args = get_args(options)
    #print(sorted(args.items()))
    ex.add_config(args)


@ex.automain
def run():
    train()
