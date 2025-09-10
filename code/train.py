import sys
import json
from collections import defaultdict

import torch
from tqdm import tqdm
from torchinfo import summary 

from exp import ex
from path import get_dirname_from_args
from tensor_utils import move_device
from evaluate import _evaluate
from ckpt import save_ckpt
from extract_keyword import extract_and_save_all
from sampler import get_sampler
from loss.base import RLLoss


@ex.capture
def _train(model, loss_fn, optimizers, tokenizer, dataloaders, logger,
           data_path, max_epoch, device, reg_coeff, grad_acc_steps,
           eval_every, eval_subset, ckpt_path,\
           eval_after, eval_test, eval_every_iter,
           extract_keyword, use_vist, use_bitfit, 
           reinforce_epoch, reinforce_metrics, reinforce_group,
           store_ckpt, hold_after_finishing):
    print ("Print pytorch model: ")
    print (type(model))
    print ('use_vist', use_vist)
#     print (model)
    #print (summary(model))
    
    best_val_stats = {}
    best_test_stats = {}
    best_epoch = 0
    if model.task == 1:
        ckpt_criterion = {'key': 'CIDEr', 'compare': lambda prev, new: prev <= new}
        if use_vist:
            ckpt_criterion = {'key': 'METEOR', 'compare': lambda prev, new: prev <= new}
    elif model.task == 2:
        ckpt_criterion = {'key': 'final_loss', 'compare': lambda prev, new: prev >= new}

    def eval_data(key, epoch, eval_generate):
        eval_stats, _, texts, score_texts, batch_stats = _evaluate(model, loss_fn, None, tokenizer,
                                    dataloaders['target'], logger, epoch=epoch,
                                    subset=None if (epoch + 1) % eval_every == 0 else eval_subset,
                                    key=key, eval_generate=eval_generate)

        if texts is not None and len(texts) > 0:
            if key == 'test':
                filename = 'blindtest' if 'blind' in data_path['test'].name else 'publictest'
            else:
                filename = key
            task1_out_path = ckpt_path / get_dirname_from_args()
            task1_out_path.mkdir(parents=True, exist_ok=True)
            ckpt_key = ckpt_criterion['key']
            ckpt_res = round(eval_stats[ckpt_key]*100, 2)
            task1_out_path = task1_out_path / f'ep_{epoch:03}_{ckpt_key[0]}_{ckpt_res}_{filename}_text.json'
            # save file
            print(f"saving result to {task1_out_path}")
            with open(task1_out_path, 'w') as f:
                json.dump(texts, f, indent=4)

        for name, val in eval_stats.items():
            logger(f"{key}/epoch/{name}", val, epoch)

        return eval_stats, texts, score_texts, batch_stats

    def train_data(name, max_epoch_per_name, optimizer, dataset_idx=None):
        nonlocal best_val_stats
        nonlocal best_test_stats
        nonlocal best_epoch
        
        dataset_type = 'pretrain' if name in 'pretrain' else 'target'
        if dataset_idx is not None:
            name = "{}_{}".format(name, dataset_idx)
            dataloader = dataloaders[dataset_type]['train'][i]
        else:
            dataloader = dataloaders[dataset_type]['train']

        n_step = 0
        print('model.task: ', model.task)
        print(f"{name}: {dataloader.task} training steps: {len(dataloader)}")
        print(f"max_epoch: {max_epoch_per_name}")
        print(f"from {dataloader.path}")
        for epoch in range(max_epoch_per_name):
            epoch += model.ckpt_epoch
            print(f"training {epoch}th epoch")
            epoch_stats = defaultdict(float)
            epoch_stats['num'] = 0
            model.train()
            if hasattr(model, 'epoch_update') and name not in 'pretrain':
                model.epoch_update(epoch)

            optimizer.zero_grad()
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch = move_device(batch, to=device)
                
                B = batch['sentences'].shape[0] if torch.is_tensor(batch['sentences']) else len(batch['sentences'])
                targets = batch['targets']
                #print(targets.shape)

                _, logits, targets, reg_loss, stats, sampler_input = model(batch, epoch=epoch)
                                                                        # batch_per_epoch=batch_per_epoch['train'],
                
                if logits is not None:
                    #print('loss compute 1')
                    loss, stats = loss_fn(logits, targets, model)
                    stats = {'language_loss': loss.item(), **stats}
                    #print(stats)
                    if reg_loss is not None:
                        if reg_loss.shape[-1] != 1: 
                            final_loss = loss + reg_loss.mean() * reg_coeff
                        else:
                            final_loss = loss + reg_loss * reg_coeff
                        stats = {**stats, **{
                            'reg_loss': reg_loss.mean().item(),
                            'final_loss': final_loss.item()
                        }}
                    else:
                        final_loss = loss
                        stats = {'final_loss': loss.item(), **stats}
                    final_loss = final_loss / grad_acc_steps
                    final_loss.backward()
                elif reg_loss is not None:
                    #print('loss compute 2')
                    if reg_loss.shape[-1] != 1: 
                        final_loss = reg_loss.mean() * reg_coeff
                    else:
                        final_loss = reg_loss * reg_coeff
                    stats = {**stats, **{
                        'reg_loss': reg_loss.mean().item(),
                        'final_loss': final_loss.item()
                    }}
                    final_loss = final_loss / grad_acc_steps
                    final_loss.backward()
                else:
                    #print('loss compute 3')
                    continue
                #print(stats)
                
                if (n_step + 1) % grad_acc_steps == 0:
                    optimizer.clip_grad()
                    optimizer.step()
                    optimizer.scheduler.step()
                    optimizer.zero_grad()
                
                #if added_stats is not None:
                #    stats = {**stats, **added_stats}
                #print(stats)

                n_step += 1

                for k, v in stats.items():
                    if v is not None:
                        epoch_stats[k] = epoch_stats[k] + B * v
                epoch_stats['num'] = epoch_stats['num'] + B

                # log lr
                stats['lr'] = optimizer.get_lr()
                                
                for key, val in stats.items():
                    logger(f"{name}/train/iters/{key}", val, n_step)

                if n_step % eval_every_iter == 0:
                    eval_stats, _, _, _ = eval_data('val', epoch)
                    for key, val in eval_stats.items():
                        logger(f"{name}/eval/iters/{key}", val, n_step)

                del targets

            num = epoch_stats.pop('num')
            epoch_stats = {k: v / num for k, v in epoch_stats.items()}
            
            print(epoch_stats)
            
            for key, val in epoch_stats.items():
                logger(f"{name}/train/epoch/{key}", val, epoch)

            eval_generate = False
            if epoch >= eval_after and epoch % eval_every == 0:
                eval_generate = True
            if name in 'pretrain':
                eval_generate = False

            eval_name = 'test' if eval_test else 'val'
            eval_stats, _, _, _ = eval_data(eval_name, epoch, eval_generate=eval_generate)
            
            print(eval_stats)

            ckpt_key = ckpt_criterion['key']
            if ckpt_key in eval_stats:
                key_dt = {ckpt_key: eval_stats[ckpt_key]}
                
                # TODO: run test after training

                # store all ckpt
                if store_ckpt and name not in 'pretrain':
                    save_ckpt(epoch, key_dt, model, tokenizer)
                if not best_val_stats:
                    best_val_stats = eval_stats
                    best_test_stats, _, _, _ = eval_data('test', epoch, eval_generate=eval_generate)
                    best_epoch = epoch
                else:
                    if ckpt_criterion['compare'](best_val_stats[ckpt_key], eval_stats[ckpt_key]):
                        # save the best ckpt
                        save_ckpt(epoch, key_dt, model, tokenizer, isBest=True)
                        best_val_stats = eval_stats
                        best_test_stats, _, _, _ = eval_data('test', epoch, eval_generate=eval_generate)
                        best_epoch = epoch
            
            # clean shm to avoid bus error and crashing workers
            torch.cuda.empty_cache()

    #BitFit
    if use_bitfit:
        #print ('before BitFit', summary(led))
        for name ,para in model.named_parameters():
            if "bias" in name:
                para.requires_grad = True
            else:
                para.requires_grad = False
        #print ('after BitFit', summary(led))
        
    pretrain_epoch = 0
    if 'pretrain' in optimizers:
        optimizer = optimizers['pretrain']
        if isinstance(optimizer, list):
            pretrain_epoch = sum(op.max_epoch for op in optimizer)
        else:
            pretrain_epoch = optimizer.max_epoch
    if pretrain_epoch > 0:
        model.pretrain()
        if isinstance(dataloaders['pretrain']['train'], list):
            for i in range(len(dataloaders['pretrain']['train'])):
                optimizer = optimizers['pretrain'][i]
                train_data('pretrain', optimizer.max_epoch, optimizer,
                        dataset_idx=i)
        else:
            optimizer = optimizers['pretrain']
            train_data('pretrain', optimizer.max_epoch, optimizer)
        model.pretrain(False)
    
    optimizer = optimizers['target']
    train_data('target', optimizer.max_epoch, optimizer)
    
    print ('best_epoch: ', best_epoch)
    
    # hold process to keep tensorboard alive
    if 'tfboard' in logger.logger_dests and hold_after_finishing:
        wait_for_key()

    return best_val_stats
