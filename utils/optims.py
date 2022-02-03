import numpy as np

from torch import optim
from torch.optim import lr_scheduler


class PolyOptimizer(optim.SGD):
    # Mostly Copied from https://github.com/jiwoon-ahn/irn/blob/master/misc/torchutils.py
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
def get_finetune_optimzier(args, model):
    # Load Optimizer method
    if hasattr(optim, args.optim['name']):
        optim_method = getattr(optim, args.optim['name'])
    # scratch method (prevent injection)
    elif '.' not in args.optim['name'] and \
         '(' not in args.optim['name'] and \
         ' ' not in args.optim['name']:
        optim_method = eval(args.optim['name'])

    # Load Scheduler method
    if hasattr(args, 'scheduler'):
        if hasattr(lr_scheduler, args.scheduler['name']):
            scheduler_method = getattr(lr_scheduler, args.scheduler['name'])
        # for minimal security
        elif '.' not in args.optim['name'] and \
            '(' not in args.optim['name'] and \
            ' ' not in args.optim['name']:
            scheduler_method = eval(args.scheduler['name'])
    else:
        scheduler_method = None

    # VGGs
    if args.network.startswith('vgg'):
        param_groups = model.get_parameter_groups()

        parameters = [
            {'params': param_groups[0], 'lr': 1 * args.optim['lr']},
            {'params': param_groups[1], 'lr': 2 * args.optim['lr']},
            {'params': param_groups[2], 'lr': 10 * args.optim['lr']},
            {'params': param_groups[3], 'lr': 20 * args.optim['lr']}]

    # ResNets
    elif ('resnet' in args.network) or ('resnext' in args.network) or ('res2net' in args.network):
        if hasattr(model, 'get_parameter_groups'):
            param_groups = model.get_parameter_groups()

            parameters = [
                {'params': param_groups[0], 'lr': 1 * args.optim['kwargs']['lr'], 'weight_decay': args.optim['kwargs']['weight_decay']},
                {'params': param_groups[1], 'lr': 10 * args.optim['kwargs']['lr'], 'weight_decay': args.optim['kwargs']['weight_decay']}]
        else:
            parameters = model.parameters()

    # Dino_ResNet
    elif args.network == 'dino_resnet50':
        parameters = model.parameters()

    # ViTs
    elif args.network.startwith('vit'):
        parameters = model.parameters()

    # Custom hparams
    else:
        parameters = model.parameters()

    # Optimizer
    optimizer = optim_method(parameters, **args.optim['kwargs'])
    # Scheduler
    scheduler = scheduler_method(optimizer, **args.scheduler['kwargs']) if scheduler_method else None

    return optimizer, scheduler