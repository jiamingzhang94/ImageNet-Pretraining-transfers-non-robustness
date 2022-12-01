import torch
import os
# from utils.util import get_optimizer, get_backbone, load_state_dict
from torchvision.models.resnet import resnet18, resnet50, wide_resnet50_2, resnet101, wide_resnet101_2
from utils.dataset import data_process
import torchvision.transforms as transforms
from utils.resnet import ResnetGenerator
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore")


def get_optimizer(net, lr, weight_decay=0.0005, momentum=0.9, freeze_level=0):
    if freeze_level == 2:
        optimizer = torch.optim.Adam(net.fc.parameters(), lr[0])
    elif freeze_level == 1:
        para = net.parameters()
        fc_params = list(map(id, net.fc.parameters()))
        conv_params = filter(lambda p: id(p) not in fc_params, para)
        optimizer = torch.optim.Adam([
            {'params': conv_params, 'lr': lr[1]},
            {'params': net.fc.parameters()}], lr=lr[0])
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr[0], weight_decay=weight_decay,
                                    momentum=momentum, nesterov=True)
    return optimizer


def get_backbone(backbone):
    name_list = {'resnet18': resnet18, 'resnet50': resnet50, 'wide_resnet50_2': wide_resnet50_2,
                 'resnet101': resnet101, 'wide_resnet101_2': wide_resnet101_2}
    return name_list.get(backbone)


def load_state_dict(model, load):
    pretrained_dict = torch.load(load, map_location='cpu')['state_dict']
    model_dict = model.state_dict()

    for k, v in pretrained_dict.items():
        model_v = model_dict.get(k)
        if model_v is None or model_v.shape != v.shape:
            continue
        model_dict[k] = v

    model.load_state_dict(model_dict)


class UAPTrainer():
    def __init__(self, opt):
        self.root = opt.root
        self.dataset = opt.dataset
        self.device = opt.device
        self.backbone = opt.backbone
        self.num_classes = None
        self.lr = opt.lr
        self.num_epoch = opt.num_epoch
        self.batch_size = opt.batch_size
        self.save = opt.save
        self.load = opt.load
        self.target = opt.target
        self.epsilon = opt.epsilon


    def data_process(self):
        batch_size = self.batch_size
        train_loader, test_loader = data_process(root=self.root, dataset=self.dataset,
                                                 batch_size=batch_size, train=True)
        self.num_classes = test_loader.dataset.num_classes
        return train_loader, test_loader

    def net_process(self):
        net = get_backbone(self.backbone)
        if self.load == 'imagenet':
            net = net(pretrained=True)
        else:
            net = net(pretrained=False, num_classes=self.num_classes)
            load_state_dict(net, self.load)
        return net

    def get_optimizer(self, net):
        optimizer = get_optimizer(net, self.lr, 0, 0)
        schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.5)

        return optimizer, schedule

    def info(self):
        print('-------------uap train-------------')
        print('dataset: {}\tbackbone: {}\ttarget: {}'.format(self.dataset, self.backbone, self.target))
        print('load from:{}'.format(self.load))
        print('save to: {}'.format(self.save))

    def train(self, norm='l2'):
        self.info()
        train_loader, test_loader = self.data_process()
        base_net = self.net_process().to(self.device)

        uap_net = ResnetGenerator(3, 3, 64, norm_type='batch', act_type='relu', device_ids=[self.device]).to(self.device)

        optimizer, schedule = self.get_optimizer(uap_net)

        normalize = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

        if self.target is None:
            loss_func = torch.nn.MSELoss()
        else:
            loss_func = torch.nn.CrossEntropyLoss()

        noise_data = np.random.uniform(-1, 1, 224 * 224 * 3)
        im_noise = np.reshape(noise_data, (3, 224, 224))
        im_noise = im_noise[np.newaxis, :, :, :]
        im_noise_tr = np.tile(im_noise, (self.batch_size, 1, 1, 1))
        uap = torch.tensor(im_noise_tr).float().to(self.device)

        slide_loss = 0
        slide_acc = 0

        for epoch in range(self.num_epoch):

            uap_net.train()
            for i, (images, labels) in enumerate(train_loader):
                delta_im = uap_net(uap.cpu())
                if norm == 'l2':
                    temp = torch.norm(delta_im.view(self.batch_size, -1), dim=1).view(-1, 1, 1, 1)
                    delta_im = delta_im * self.epsilon / temp
                else:
                    delta_im = torch.clamp(delta_im, -self.epsilon, self.epsilon)

                images, labels = images.to(self.device), labels.to(self.device)
                images = normalize(images)
                x_adv = torch.clamp(images + delta_im[:images.size(0)], -1, 1)
                prediction = base_net(x_adv)
                if self.target is None:
                    loss_kl = loss_func(prediction, base_net(images))
                    loss = 10 - loss_kl
                else:
                    target_label = torch.ones_like(labels) * self.target
                    loss = loss_func(prediction, target_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct = (prediction.argmax(dim=1) == labels).sum().detach() / images.size(0)

                slide_loss = slide_loss * 0.7 + loss * 0.3
                slide_acc = slide_acc * 0.7 + correct * 0.3

                if i % 10 == 0:
                    print('epoch:{} loss:{} acc:{} '.format(epoch, slide_loss, slide_acc))
            schedule.step()
            torch.save({'state_dict': uap_net.state_dict(), 'uap': uap, 'delta_im': delta_im}, self.save)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='alphabet')
    parser.add_argument('--root', type=str, default='datasets')

    parser.add_argument('--load', type=str)
    parser.add_argument('--save', type=str)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--epsilon', type=float, default=40)
    parser.add_argument('--target', type=int, default=None)

    parser.add_argument('--lr', type=float, nargs='+', default=None)
    parser.add_argument('--num_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=20)

    opt = parser.parse_args()

    trainer = UAPTrainer(opt)
    trainer.train()
