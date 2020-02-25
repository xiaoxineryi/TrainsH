import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import FileLoader
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt



def default_loader(path):
    return Image.open(path)

class MyImageFloder(data.Dataset):
    def __init__(self,FileName,transform = None,target_transform = None,loader = default_loader):
        FilePlaces,LabelSet = FileLoader.filePlace_loader(FileName)

        LabelSet = [np.long(i) for i in LabelSet]
        LabelSet = torch.Tensor(LabelSet).long()

        self.imgs_place = FilePlaces
        self.LabelSet = LabelSet
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):
        img_place = self.imgs_place[item]
        label = self.LabelSet[item]
        img = self.loader(img_place)
        if self.transform is not None:
            img = self.transform(img)

        return img,label


    def __len__(self):
        return len(self.imgs_place)

