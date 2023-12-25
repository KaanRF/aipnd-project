import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


class DataLoader:
    def __init__(self, path):
        super().__init__()
        self.train_dir = path + '/train'
        self.valid_dir = path + '/valid'
        self.test_dir = path + '/test'

        self.train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
        self.test_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])

        self.valid_transforms = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

    def get_datasets(self):
        train_datasets = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
        test_datasets = datasets.ImageFolder(self.test_dir, transform=self.test_transforms)
        valid_datasets = datasets.ImageFolder(self.valid_dir, transform=self.valid_transforms)

        return train_datasets, test_datasets, valid_datasets

    def get_loaders(self):
        train_loader = torch.utils.data.DataLoader(self.train_datasets, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_datasets, batch_size=64)
        valid_loader = torch.utils.data.DataLoader(self.valid_datasets, batch_size=64)

        return train_loader, test_loader, valid_loader


