import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

def image_loader(image_name):
    loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)[0][0]

def file_loader(fileName):
    f = open(fileName)
    line = f.readline()
    labelSet = []
    dataSet = []
    i = 0
    while line:
        filePlace = line.split()[0]
        image = image_loader(filePlace)
        # print(image)
        dataSet.append(image)
        label = line.split()[1]
        labelSet.append(label)
        i = i+1
        line = f.readline()
    f.close()
    return dataSet,labelSet

def filePlace_loader(fileName):
    f= open(fileName)
    line = f.readline()
    labelSet = []
    filePlaces = []
    i = 0
    while line:
        filePlace = line.split()[0]
        filePlaces.append(filePlace)
        label = line.split()[1]
        labelSet.append(label)
        i = i +1
        line=f.readline()
    f.close()
    return filePlaces,labelSet



