import torch
from torchvision.models import resnet18, resnet50, wide_resnet50_2, densenet121
from torchvision.datasets import ImageFolder, CIFAR10
import torchvision.transforms as transforms
from customDataset import DataLoader, DataLoaderX
import os
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from PIL import Image
import argparse


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))  # args.lr = 0.1 ,
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CommonDataset(Dataset):

    def __init__(self, data_dir, is_train=True, img_size=224, corrupt=None, per_imgs=None, num_classes=None):
        "num_imgs: number of images per category"
        self.data = []
        self.labels = []
        self.is_train = is_train
        self.img_size = img_size
        if num_classes is None:
            self.classes_num = len(os.listdir(data_dir))
        else:
            self.classes_num = num_classes
        self.corrupt = corrupt
        self.per_imgs = per_imgs
        self.num_classes = num_classes

        if 'iid' in data_dir:
            distribution_option = True
        else:
            distribution_option = False

        if distribution_option:
            for class_id, dirs in enumerate(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, dirs)
                for property in os.listdir(class_dir):
                    property_dir = os.path.join(class_dir, property)
                    for img in os.listdir(property_dir):
                        self.data.append(os.path.join(class_dir, property, img))
                        self.labels.append(class_id)

        else:
            for class_id, dirs in enumerate(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, dirs)
                for i, basename in enumerate(os.listdir(class_dir)):
                    self.data.append(os.path.join(class_dir, basename))
                    self.labels.append(class_id)

        if is_train:
            self.data_transform = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size), 0),
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            self.data_transform = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size), 0),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

        self.t0 = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), 0),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data[item]
        image = Image.open(path).convert('RGB')

        image = self.data_transform(image)
        label = self.labels[item]
        return image, label


def fixed_feature(model, num_fixed, lr, cnn_lr):
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
    return optimizer


def get_model(name):
    if name == 'RN18':
        model = resnet18(pretrained=True)
    elif name == 'RN50':
        model = resnet50(pretrained=True)
    elif name == 'wrn50':
        model = wide_resnet50_2(pretrained=True)
    elif name == 'dense':
        model = densenet121(pretrained=True)
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


def train(dataset, model, n_d, num_epoch, batch_size, lr, cnn_lr, num_fixed, device_ids, optimizer):
    ckp_path = os.path.join('./checkpoint', dataset)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)
    train_dir = dataset + '/train'
    test_dir = dataset + '/test'

    train_dataset = CommonDataset(train_dir)
    train_loader = DataLoaderX(train_dataset,
                               batch_size=batch_size, shuffle=True,
                               num_workers=8,
                               pin_memory=True)
    test_dataset = CommonDataset(test_dir, is_train=False)
    test_loader = DataLoaderX(test_dataset,
                              batch_size=batch_size, shuffle=False,
                              num_workers=8,
                              pin_memory=True)

    num_classes = train_dataset.classes_num
    model = get_model(model)
    model.fc = torch.nn.Linear(n_d, num_classes)

    model = model.cuda(device=device_ids[0])

    optimizer = get_optimizer(name=optimizer, model=model, lr=lr, cnn_lr=cnn_lr, num_fixed=num_fixed)
    # optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * num_epoch, 1e-6)
    loss_func = torch.nn.CrossEntropyLoss()

    best_acc = 0

    print(dataset + ' train start ' + str(num_fixed))
    print('---------------')

    for epoch in range(num_epoch):
        losses = 0
        corrects = 0
        total = 0
        # adjust_learning_rate(lr, optimizer, epoch)

        # train
        model.train()
        for images, labels in train_loader:
            images, real_labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            prob = model(images)
            loss = loss_func(prob, real_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()

            losses += (loss * len(images)).item()
            pred_idx = torch.argmax(prob.detach(), 1)
            corrects += (pred_idx == real_labels).sum().item()

            total += len(images)

        print('train epoch:{}\tloss:{:.4f}\tacc:{:.4f}'.format(epoch, losses / total,
                                                               corrects / total))
        # eval
        model.eval()
        with torch.no_grad():
            losses = 0
            corrects = 0
            total = 0

            for images, labels in test_loader:
                images, real_labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])

                prob = model(images)
                loss = loss_func(prob, real_labels)

                losses += (loss * len(labels)).item()
                pred_idx = torch.argmax(prob.detach(), 1)
                corrects += (pred_idx == real_labels).sum().item()

                total += len(images)

            print('test epoch:{}\tloss:{:.4f}\tacc:{:.4f}'.format(epoch, losses / total,
                                                                  corrects / total))
            print('-----------------')

            if corrects / total > best_acc:
                best_acc = corrects / total
                print(best_acc)
                torch.save({'best_acc': best_acc, 'state_dict': model.state_dict()},
                           os.path.join(ckp_path,
                                        'ckp.pt'))

    # del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/dataset/pets', help='dataset path')
    parser.add_argument('--model', type=str, default='RN18')
    parser.add_argument('--nd', type=int, default=512, help='number of dimension for logit')
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--lr_fc', type=int, default=3e-4, help='lr for fc layer')
    parser.add_argument('--lr_cnn', type=int, default=3e-4, help='lr for cnn layer')
    parser.add_argument('--num_fixed', type=int, default=-1, help='-1: re-train all layer with a same lr'
                                                                  '0: re-train fc and cnn layer with different lrs')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    train(dataset=args.dataset, model=args.model, n_d=args.nd, num_epoch=args.epoch, batch_size=args.bs,
          lr=args.lr_fc, cnn_lr=args.lr_cnn, num_fixed=args.num_fixed, device_ids=args.device, optimizer=args.optimizer)
    # torch.cuda.empty_cache()
