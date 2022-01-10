import numpy as np

from torch import optim
from torch.optim import lr_scheduler

# Mostly Copied from https://github.com/jiwoon-ahn/irn/blob/master/misc/torchutils.py
class PolyOptimizer(optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.param_groups = params
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult
        super().step(closure)

        self.global_step += 1


def reduce_lr(epoch, optimizer, reduce_points, factor):
    values = reduce_points.strip().split(',')
    
    change_points = map(lambda x: int(x.strip()), values)

    if change_points is not None and epoch in change_points:
        for g in optimizer.param_groups:
            g['lr'] = g['lr']*factor
            print("Reduce Learning Rate : ", epoch, g['lr'])
        return True
 

# Return Optimizer and Scheduler
def get_finetune_optimzier(cfg, model):

    # Load Optimizer method
    if hasattr(optim, cfg['optimizer']['name']):
        optim_method = getattr(optim, cfg['optimizer']['name'])
    # scratch method
    else:
        optim_method = eval(optim, cfg['optimizer']['name'])

    # Load Scheduler method
    if 'scheduler' in cfg:
        if hasattr(lr_scheduler, cfg['scheduler']['name']):
            scheduler_method = getattr(lr_scheduler, cfg['scheduler']['name'])
        else:
            scheduler_method = eval(cfg['scheduler']['name'])
    else:
        scheduler_method = None

    # VGGs
    if cfg['network'].startswith('vgg'):
        param_groups = model.get_parameter_groups()

        parameters = [
            {'params': param_groups[0], 'lr': 1 * cfg['optimizer']['lr']},
            {'params': param_groups[1], 'lr': 2 * cfg['optimizer']['lr']},
            {'params': param_groups[2], 'lr': 10 * cfg['optimizer']['lr']},
            {'params': param_groups[3], 'lr': 20 * cfg['optimizer']['lr']}]

    # ResNets
    elif ('resnet' in cfg['network']) or ('resnext' in cfg['network']) or ('res2net' in cfg['network']):
        parameters = model.parameters()

    # Dino_ResNet
    elif cfg['network'] == 'dino_resnet50':
        parameters = model.parameters()

    # ViTs
    elif cfg['network'].startwith('vit'):
        parameters = model.parameters()

    # Custom hparams
    else:
        parameters = model.parameters()

    # Optimizer
    optimizer = optim_method(parameters, **cfg['optimizer']['kwargs'])
    # Scheduler
    scheduler = scheduler_method(optimizer, **cfg['scheduler']['kwargs']) if scheduler_method else None

    return optimizer, scheduler