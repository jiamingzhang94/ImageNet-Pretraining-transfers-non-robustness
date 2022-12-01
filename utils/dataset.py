from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def data_process(root, dataset, batch_size, train=False):
    root = os.path.join(root, dataset)

    # train loader
    train_loader = None
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if train:
        train_dataset = CommonDataset(root, train=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # test loader
    test_transform = transforms.Compose([
        #transforms.Resize((224, 224), 0),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    test_dataset = CommonDataset(root, train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    return train_loader, test_loader


class CommonDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform

        if train:
            data_dir = os.path.join(root, 'train')
        else:
            data_dir = os.path.join(root, 'test')

        self.num_classes = len(os.listdir(data_dir))

        for class_id, dirs in enumerate(os.listdir(data_dir)):
            class_dir = os.path.join(data_dir, dirs)
            for basename in os.listdir(class_dir):
                if not is_image_file(basename):
                    continue
                self.paths.append(os.path.join(class_dir, basename))
                self.labels.append(class_id)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label

