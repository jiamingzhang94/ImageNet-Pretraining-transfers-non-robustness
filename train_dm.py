import torch
from utils.resnet import resnet18, wide_resnet50_2, resnet50
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as transforms
from customDataset import CommonDataset, DataLoader, DataLoaderX
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils.util import data_process, resume_model, fixed_feature
from utils.iterative_gradient_attack import *
import torch.nn.functional as F
from torch.autograd import Variable
import argparse


def dm_loss(model,
            x_natural,
            y,
            optimizer,
            device,
            step_size=0.003,
            epsilon=0.031,
            perturb_steps=10,
            beta=500.0,
            distance='l_inf'):
    """ based on TRADES, https://github.com/yaodongyu/TRADES """

    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_mse = nn.MSELoss()
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda(device=device[0]).detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_mse((model(x_adv)[0]),
                                        model(x_natural)[0])
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, -1.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    _, logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_mse(model(x_adv)[0],
                                                     model(x_natural)[0])
    # loss_robust = (1.0 / batch_size) * criterion_mse(F.log_softmax(model(x_adv)[1], dim=1),
    #                                                 F.softmax(model(x_natural)[1], dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust


def get_model(name):
    if name == 'RN18':
        model = resnet18(pretrained=True)
    elif name == 'RN50':
        model = resnet50(pretrained=True)
    elif name == 'wrn50':
        model = wide_resnet50_2(pretrained=True)
    else:
        print('please add {} to get_model function'.format(name))
        raise (f'Model {name} Not Found')
    return model


def get_optimizer(name, model, num_fixed, lr, cnn_lr):
    if name == 'Adam':
        if num_fixed == -1:
            optimizer = torch.optim.Adam(model.parameters(), lr)
        else:
            # res18_layers = [[0], [4,0], [4,1], [5,0], [5,1], [6,0], [6,1], [7,0], [7,1], [9]]
            para = []
            layers = model.children()
            for i, layer in enumerate(layers):
                if i < num_fixed:
                    continue
                para += layer.parameters()
            fc_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in fc_params, para)
            optimizer = torch.optim.Adam([
                {'params': base_params, 'lr': cnn_lr},
                {'params': model.fc.parameters()}], lr)

    elif name == 'SGD':
        if num_fixed == -1:
            optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
        else:
            # res18_layers = [[0], [4,0], [4,1], [5,0], [5,1], [6,0], [6,1], [7,0], [7,1], [9]]
            para = []
            layers = model.children()
            for i, layer in enumerate(layers):
                if i < num_fixed:
                    continue
                para += layer.parameters()
            fc_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in fc_params, para)
            optimizer = torch.optim.SGD([
                {'params': base_params, 'lr': cnn_lr},
                # {'params': model.fc.parameters()}], lr)
                {'params': model.fc.parameters()}], lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise (f'optimizer {name} Not defined')
    return optimizer


def train_md(dataset, model, num_epoch, device_ids, epsilon, n_d, optimizer, num_fixed, cnn_lr,
             lr, batch_size, num_iter, pre_ckpt):
    ckp_path = os.path.join('./checkpoint', dataset)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)
    train_dir = dataset + '/train'
    test_dir = dataset + '/test'

    # data load
    train_loader, test_loader, num_classes, _ = data_process(dataset=dataset,
                                                             train_dir=train_dir,
                                                             test_dir=test_dir,
                                                             batch_size=batch_size,
                                                             device_ids=device_ids,
                                                             is_train=True)
    # model init
    pre_trained_model = get_model(model)
    pre_ckpt = torch.load(pre_ckpt, map_location='cpu')
    pre_trained_model = resume_model(pre_trained_model, pre_ckpt)
    #
    model = pre_trained_model
    # model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(n_d, num_classes)
    model = model.cuda(device=device_ids[0])

    optimizer = get_optimizer(name=optimizer, model=model, lr=lr, cnn_lr=cnn_lr, num_fixed=num_fixed)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epoch, 1e-6)

    # optimizer = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0

    print(dataset + ' train start ')
    print('---------------')

    for epoch in range(num_epoch):
        losses_n = 0
        losses_r = 0
        corrects = 0
        total = 0
        # train
        model.train()
        for images, labels in train_loader:
            images, real_labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            optimizer.zero_grad()
            loss, l_n, l_r = dm_loss(model=model,
                                     x_natural=images,
                                     y=real_labels,
                                     device=device_ids,
                                     optimizer=optimizer,
                                     step_size=epsilon * 1.25 / num_iter,
                                     epsilon=epsilon,
                                     perturb_steps=num_iter,
                                     distance='l_inf')
            loss.backward()
            optimizer.step()
            schedule.step()
            losses_n += (l_n * len(images)).item()
            losses_r += (l_r * len(images)).item()
            _, prob = model(images)
            pred_idx = torch.argmax(prob.detach(), 1)
            corrects += (pred_idx == real_labels).sum().item()
            total += len(images)
        print('train epoch:{}\tloss_na:{:.4f}\tloss_ro:{:.4f}'.format(epoch, losses_n / total,
                                                                      losses_r / total))

        # eval
        model.eval()
        with torch.no_grad():
            corrects = 0
            total = 0

            for images, labels in test_loader:
                images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
                _, prediction = model(images)
                pred_idx = torch.argmax(prediction.detach(), 1)
                corrects += (pred_idx == labels).sum().item()
                total += len(images)

            # print('acc:{:.4f}'
            #       .format(corrects / total))

            if corrects / total > best_acc:
                best_acc = corrects / total
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                           os.path.join(ckp_path,
                                        'ckp_md.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/dataset/pets', help='dataset path')
    parser.add_argument('--pre_ckpt', type=str, default='pretrained/resnet18_rt.pt', help='the path of adversarially training model'
                                                                                          'downloaded from https://github.com/microsoft/robust-models-transfer')
    parser.add_argument('--model', type=str, default='RN18')
    parser.add_argument('--nd', type=int, default=512, help='number of dimension for logit')
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.5/255*2)
    parser.add_argument('--lr_fc', type=int, default=3e-4, help='lr for fc layer')
    parser.add_argument('--lr_cnn', type=int, default=3e-4, help='lr for cnn layer')
    parser.add_argument('--num_fixed', type=int, default=-1, help='-1: re-train all layer with a same lr'
                                                                  '0: re-train fc and cnn layer with different lrs')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    train_md(dataset=args.dataset, model=args.model, n_d=args.nd, num_epoch=args.epoch, batch_size=args.bs,
             lr=args.lr_fc, cnn_lr=args.lr_cnn, num_fixed=args.num_fixed, device_ids=args.device,
             optimizer=args.optimizer, epsilon=args.eps, num_iter=args.num_iter, pre_ckpt=args.pre_ckpt)
