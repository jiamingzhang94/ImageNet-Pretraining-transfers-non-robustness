from torchvision.models import resnet18
import torchvision
from customDataset import is_image_file
import os
import pickle
import numpy as np
from PIL import Image
import random
import torch
from torchvision.models import resnet18, alexnet, wide_resnet50_2, densenet121
from utils.cifar10 import CIFAR10
from torchvision.datasets import ImageFolder, SVHN, FashionMNIST
import torchvision.transforms as transforms
from customDataset import CommonDataset, DataLoader, DataLoaderX
import os
# from torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# def fixed_feature(model, num_fixed):
#     # res18_layers = [[0], [4,0], [4,1], [5,0], [5,1], [6,0], [6,1], [7,0], [7,1], [9]]
#     para = []
#     if model.module is None:
#         layers = model.children()
#     else:
#         layers = model.module.children()
#     for i, layer in enumerate(layers):
#         if i < num_fixed:
#             continue
#         para += layer.parameters()
#     return para

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
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': cnn_lr},
        {'params': model.fc.parameters()}], lr)
    return optimizer

# def fixed_feature_2(model, num_fixed):
#     res18_layers = [[0], [4,0], [4,1], [5,0], [5,1], [6,0], [6,1], [7,0], [7,1], [9]]
#     para = []
#     try:
#         layers = model.children()
#     except:
#         layers = model.module.children()
#
#     for i, layer in enumerate(layers):
#         try:
#             for j, item in enumerate(layer):
#                 if i*10+j < res18_layers[num_fixed][0]*10+res18_layers[num_fixed][1]:
#                     continue
#                 para += item.parameters()
#         except TypeError:
#             # single layer
#             if i < res18_layers[num_fixed][0]:
#                 continue
#             para += layer.parameters()
#     return para




def data_process(dataset, test_dir, batch_size, device_ids, test_batch_size=100, per_imgs=None, num_classes=None,
                 corrupt=None, train_dir=None, is_train=False, img_size=224):
    train_loader = None
    train_transform = transforms.Compose([
        # transforms.Resize((img_size, img_size), 0),

        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        # transforms.Resize((img_size, img_size), 0),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    if dataset == 'cifar1':
        if is_train:
            train_dataset = CIFAR10(root='/data/jiaming/datasets/cifar10', train=True, download=False, transform=train_transform)
            train_loader = DataLoaderX(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8 * len(device_ids),
                                       pin_memory=False)
        test_dataset = CIFAR10(root='/data/jiaming/datasets/cifar10', train=False, download=False, transform=test_transform)
        test_loader = DataLoaderX(test_dataset,
                                  batch_size=test_batch_size,
                                  num_workers=8 * len(device_ids),
                                  pin_memory=False)
        num_classes = 10

    else:
        if is_train:
            train_dataset = CommonDataset(train_dir, corrupt=corrupt, img_size=img_size,
                                          per_imgs=per_imgs, num_classes=num_classes)
            train_loader = DataLoaderX(train_dataset,
                                       batch_size=batch_size, shuffle=True,
                                       num_workers=8 * len(device_ids),
                                       pin_memory=False)
        test_dataset = CommonDataset(test_dir, corrupt=corrupt, is_train=False, num_classes=num_classes)
        test_loader = DataLoaderX(test_dataset,
                                  batch_size=test_batch_size,
                                  num_workers=8 * len(device_ids),
                                  pin_memory=False)
        num_classes = test_dataset.classes_num
    len_testset = len(test_dataset)
    return train_loader, test_loader, num_classes, len_testset


def ood_to_iid(src, dst, prob=0.7, seed=0):
    if not os.path.exists(dst):
        os.mkdir(dst)
    train_dir = os.path.join(dst, 'train')
    test_dir = os.path.join(dst, 'test')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for root, dirs, files in os.walk(dst):
        for f in files:
            os.remove(os.path.join(root, f))

    num = 0
    for root, dirs, files in os.walk(src):
        for dir in dirs:
            path = os.path.join(root, dir).replace(src, dst).replace('test', 'train')
            if not os.path.exists(path):
                os.mkdir(path)
            path = path.replace('train', 'test')
            if not os.path.exists(path):
                os.mkdir(path)

        for f in files:
            if not is_image_file(f):
                continue
            try:
                image = Image.open(os.path.join(root, f)).convert('RGB')
            except:
                continue
            property = os.path.basename(root)
            class_type = os.path.basename(os.path.dirname(root))
            random.seed(seed+num)
            if random.random() < prob:
                image.save(os.path.join(train_dir, class_type, property, f))
            else:
                image.save(os.path.join(test_dir, class_type, property, f))
            num += 1


import xlwt
import xlrd
import xlutils.copy
def save_finetune_acc(csv_file, load_tar, acc):
    lrs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    lrs = list(reversed(lrs))
    maps = {}
    for i, lr in enumerate(lrs):
        maps[lr] = i+1
    load_tar = os.path.splitext(os.path.basename(load_tar))[0]
    fclr, cnnlr = load_tar.split('_')
    fclr = float(fclr[4:])
    cnnlr = float(cnnlr[5:])
    if not os.path.exists(csv_file):
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('My Worksheet')
        for i, lr in enumerate(lrs):
            worksheet.write(i+1, 0, lr)
            worksheet.write(0, i+1, lr)
        worksheet.write(maps[cnnlr], maps[fclr], acc)
        workbook.save(csv_file)
    else:
        rd = xlrd.open_workbook(csv_file, formatting_info=True)  # 打开文件
        wt = xlutils.copy.copy(rd)  # 复制
        sheets = wt.get_sheet(0)  # 读取第一个工作表
        sheets.write(maps[cnnlr], maps[fclr], acc)
        wt.save(csv_file)  # 保存


def resume_model(model, checkpoint):
    #checkpoint = torch.load('/data/jiaming/code/exp_transfer/checkpoint_new/0/ILSVRC2012/resnet18_linf_eps0.5.ckpt')

    #model = resnet18()

    state_dict_path = 'model'
    if not ('model' in checkpoint):
        state_dict_path = 'state_dict'

    sd = checkpoint[state_dict_path]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    model_sd = {}
    for k, v, in sd.items():
        if k.startswith('model.'):
            model_sd[k[len('model.'):]] = v
    model.load_state_dict(model_sd)

    return model

def del_bad_img(src):
    for root, dirs, files in os.walk(src):
        for f in files:
            if not is_image_file(f):
                os.remove(os.path.join(root, f))
                continue
            try:
                Image.open(os.path.join(root, f)).convert('RGB')
            except:
                os.remove(os.path.join(root, f))
#
# import tensorflow.contrib.slim as slim
# class ImageNet_datastream:
#     def __init__(self, sess, train_batchsize, test_batchsize):
#         self.train_img_batch, self.train_label_batch = self.read_and_decode("/data/zhaoxian/dataset/imagenet-tfrecord", "train", train_batchsize)
#         self.val_img_batch, self.val_label_batch = self.read_and_decode("/data/zhaoxian/dataset/imagenet-tfrecord", "val", test_batchsize)
#         self.sess = sess
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#
#     def read_and_decode(self, path, type="train", batchsize=10, imgsize=224):
#         if type == "train":
#             file_path = os.path.join(path, "train-*")
#             num_samples = 1281167
#
#             dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
#             data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
#             image, label = data_provider.get(['image', 'label'])
#
#             image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)
#
#             image = tf.image.random_flip_left_right(image)
#             # image = tf.image.random_brightness(image, max_delta=0.1)
#             # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
#             image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.
#
#             img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=32,
#                                                             capacity=8192*16, min_after_dequeue=512)
#             # label_batch = tf.one_hot(label_batch, 1000)
#             img_batch = tf.transpose(img_batch, [0, 3, 1, 2])
#             return img_batch, label_batch-1
#         else:
#             file_path = os.path.join(path, "validation-*")
#             num_samples = 10000
#
#             dataset = self.get_record_dataset(file_path, num_samples=num_samples, num_classes=1000)
#             data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
#             image, label = data_provider.get(['image', 'label'])
#
#             image = self._fixed_sides_resize(image, output_height=imgsize, output_width=imgsize)
#
#             image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.
#
#             # img_batch, label_batch = tf.train.batch([image, label], batch_size=batchsize, allow_smaller_final_batch=True)
#             img_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batchsize, num_threads=2,
#                                                             capacity=4096*8, min_after_dequeue=512)
#             # label_batch = tf.one_hot(label_batch, 1000)
#             img_batch = tf.transpose(img_batch, [0, 3, 1, 2])
#             return img_batch, label_batch-1
#
#     def _fixed_sides_resize(self, image, output_height, output_width):
#         """Resize images by fixed sides.
#         Args:
#             image: A 3-D image `Tensor`.
#             output_height: The height of the image after preprocessing.
#             output_width: The width of the image after preprocessing.
#         Returns:
#             resized_image: A 3-D tensor containing the resized image.
#         """
#         output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
#         output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)
#
#         image = tf.expand_dims(image, 0)
#         resized_image = tf.image.resize_nearest_neighbor(
#             image, [output_height, output_width], align_corners=False)
#         resized_image = tf.squeeze(resized_image)
#         resized_image.set_shape([None, None, 3])
#         return resized_image
#
#     def get_record_dataset(self, record_path, reader=None, num_samples=1281167, num_classes=1000):
#         """Get a tensorflow record file.
#         Args:
#         """
#         if not reader:
#             reader = tf.TFRecordReader
#
#         keys_to_features = {
#             'image/encoded':
#                 tf.FixedLenFeature((), tf.string, default_value=''),
#             'image/format':
#                 tf.FixedLenFeature((), tf.string, default_value='jpeg'),
#             'image/class/label':
#                 tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1],
#                                                                          dtype=tf.int64))}
#
#         items_to_handlers = {
#             'image': slim.tfexample_decoder.Image(image_key='image/encoded',
#                                                   format_key='image/format'),
#             'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
#         decoder = slim.tfexample_decoder.TFExampleDecoder(
#             keys_to_features, items_to_handlers)
#
#         labels_to_names = None
#         items_to_descriptions = {
#             'image': 'An image with shape image_shape.',
#             'label': 'A single integer.'}
#         return slim.dataset.Dataset(
#             data_sources=record_path,
#             reader=reader,
#             decoder=decoder,
#             num_samples=num_samples,
#             num_classes=num_classes,
#             items_to_descriptions=items_to_descriptions,
#             labels_to_names=labels_to_names)
#
#     def get_one_batch_train(self):
#         image, label = self.sess.run([self.train_img_batch, self.train_label_batch])
#         image = torch.from_numpy(image)
#         label = torch.from_numpy(label)
#         return image, label
#
#     def get_one_batch_val(self):
#         image, label = self.sess.run([self.val_img_batch, self.val_label_batch])
#         image = torch.from_numpy(image)
#         label = torch.from_numpy(label)
#         return image, label

if __name__ == '__main__':
    save_finetune_acc('test.csv', 'fclr0.0005_cnnlr0.01.tar', acc=0.5)

