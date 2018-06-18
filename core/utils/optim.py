import torch
import torch.optim


class Optim(object):
    def __init__(self, policies, config):
        if config.algorithm == 'SGD':
            self._optimizer = torch.optim.SGD(policies, config.args.base_lr,
                                              config.args.momentum,
                                              config.args.weight_decay)
            self._cfg = config
        else:
            raise ValueError(
                'Unknown optimizer algorithm: {}'.format(config.algorithm))

    def adjust_learning_rate(self, iter, epoch):
        if self._cfg.algorithm == 'SGD':
            if self._cfg.args.policy == 'step':
                decay = self._cfg.args.rate_decay_factor**(
                    iter // self._cfg.args.rate_decay_step)
                lr = self._cfg.args.base_lr * decay
                decay = self._cfg.args.weight_decay
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr * param_group['lr_mult']
                    param_group['weight_decay'] = decay * param_group[
                        'decay_mult']
                return lr

            elif self._cfg.args.policy == 'poly':
                decay = (1 - float(iter) / self._cfg.args.max_iter
                         )**self._cfg.args.learning_power
                lr = self._cfg.args.base_lr * decay
                decay = self._cfg.args.weight_decay
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr * param_group['lr_mult']
                    param_group['weight_decay'] = decay * param_group[
                        'decay_mult']
                return lr
            elif self._cfg.args.policy == 'poly_epoch':
                decay = (1 - float(epoch) / self._cfg.args.max_epoch
                         )**self._cfg.args.learning_power
                lr = self._cfg.args.base_lr * decay
                decay = self._cfg.args.weight_decay
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] = lr * param_group['lr_mult']
                    param_group['weight_decay'] = decay * param_group[
                        'decay_mult']
                return lr
            else:
                raise ValueError('Unknown optimizer policy: {}'.format(
                    self._cfg.args.policy))

    def __getattr__(self, key):
        return getattr(self._optimizer, key)
