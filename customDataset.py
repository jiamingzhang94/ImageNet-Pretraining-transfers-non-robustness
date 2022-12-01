from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import torch
import json
from PIL import Image
from prefetch_generator import BackgroundGenerator
# from utils.corrupt import corrupt
corrupt_list = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)




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
                        if not is_image_file(img):
                            continue
                        self.data.append(os.path.join(class_dir, property, img))
                        self.labels.append(class_id)

        else:
            for class_id, dirs in enumerate(os.listdir(data_dir)):
                class_dir = os.path.join(data_dir, dirs)
                for i, basename in enumerate(os.listdir(class_dir)):
                    if not is_image_file(basename):
                        continue
                    self.data.append(os.path.join(class_dir, basename))
                    self.labels.append(class_id)


        if is_train:
            self.data_transform = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size), 0),
                transforms.Resize(256),
                transforms.RandomCrop(self.img_size),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            self.data_transform = transforms.Compose([
                # transforms.Resize((self.img_size, self.img_size), 0),
                transforms.Resize(256),
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

class AlphabetDataset(Dataset):
    def __init__(self, data_dir, is_train=True, img_size=80, use_meta=True, freq_rebuild=False, freq_dir=None, r=16, device_ids=[0]):
        self.data = []
        self.labels = []
        self.is_train = is_train
        self.freq_rebuild = freq_rebuild
        self.freq_dir = freq_dir
        self.device_ids = device_ids
        self.chars = ['F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T',
                      'U','V','W','X','Y','Z']

        if os.path.exists(os.path.join(data_dir, 'meta.json')) and use_meta:
            with open(os.path.join(data_dir, 'meta.json'), 'r') as f:
                meta = json.load(f)

            for path, label in meta.items():
                if not is_image_file(path):
                    continue
                self.data.append(path)
                self.labels.append(label)

        else:
            for class_id, char in enumerate(self.chars):
                class_dir = os.path.join(data_dir, str(char))
                for basename in os.listdir(class_dir):
                    self.data.append(os.path.join(class_dir, basename))
                    self.labels.append(class_id)

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.resize = transforms.Resize((img_size, img_size), 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data[item]
        img = Image.open(path).convert('RGB')
        img = self.resize(img)
        label = self.labels[item]

        if self.freq_rebuild:
            if self.freq_dir is None:
                # generate your low and high img
                img_low, img_high = generateDataWithDifferentFrequencies_3Channel(img, self.mask, self.device_ids[0])
            else:
                # read your low and high img
                assert ('Not implement. Read your low and high img here')

            img, img_low, img_high = self.data_transform(img), self.data_transform(img_low), self.data_transform(img_high)
            return img, img_low, img_high, label, path
        else:
            img = self.data_transform(img)
            return img, label, path



class ImageNetDataset(Dataset):

    def __init__(self, data_dir, is_train=True, img_size=224, ):
        "num_imgs: number of images per category"
        self.data = []
        self.labels = []
        self.is_train = is_train
        self.img_size = img_size

        temp = os.listdir(data_dir)

        for class_id, dirs in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, dirs)
            for i, basename in enumerate(os.listdir(class_dir)):
                if not is_image_file(basename):
                    continue
                self.data.append(os.path.join(class_dir, basename))
                self.labels.append(int(dirs))


        if is_train:
            self.data_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), 0),
                transforms.Resize(256),
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
        else:
            self.data_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size), 0),
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
