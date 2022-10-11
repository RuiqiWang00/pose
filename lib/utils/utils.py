import os

import torch
import torch.optim as optim

def get_optimizer(cfg, parameters):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.AdamW(
            parameters,
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD
        )

    return optimizer

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, 'model', filename))
    epoch=states['epoch']
    name='epoch'+str(epoch)+'_model.pth.tar'
    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, name)
        )