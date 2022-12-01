import numpy as np
import torch


class Attack(object):
    def __init__(self, model, loss_fn, mean, std, norm_type, max_norm, targeted=False, device_ids=[0]):
        super(Attack, self).__init__()

        # mean = np.array(mean)
        # std = np.array(std)
        clip_min = (-1 - mean) / std
        clip_max = (1 - mean) / std

        # channel = mean.shape[0]

        # mean = torch.Tensor(mean).reshape([channel, 1, 1])
        # std = torch.Tensor(std).reshape([channel, 1, 1])
        # clip_min = torch.Tensor(clip_min).reshape([channel, 1, 1])
        # clip_max = torch.Tensor(clip_max).reshape([channel, 1, 1])
        expand_max_norm = max_norm / std

        # if next(model.parameters()).is_cuda:
        #     mean = mean.cuda(device=device_ids[0])
        #     std = std.cuda(device=device_ids[0])
        #     clip_min = clip_min.cuda(device=device_ids[0])
        #     clip_max = clip_max.cuda(device=device_ids[0])
        #     expand_max_norm = expand_max_norm.cuda(device=device_ids[0])


        self.model = model
        self.loss_fn = loss_fn
        self.mean = mean
        self.std = std
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm_type = norm_type
        self.max_norm = max_norm
        self.expand_max_norm = expand_max_norm
        self.targeted = targeted


    def attack(self, x, y):

        error = "Sub-classes must implement perturb."
        raise NotImplementedError(error)