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

#  公有矩阵和私有矩阵应该是从外面传入 否则的话测试和训练用的不是同一个了

class TrainsNetHLoader(data.Dataset):
    def __init__(self,FileName,noisyDimonsion,PublicMat,SelfMats,participantNum,Train=True,TestMat = None,transform = None,
                 target_transform = None,loader = default_loader):
        FilePlaces,LabelSet = FileLoader.filePlace_loader(FileName)

        LabelSet = [np.long(i) for i in LabelSet]
        LabelSet = torch.Tensor(LabelSet).long()
        #  原有数据集是  P：m*n 在加上噪音以后是 m*(n+d)  其中R是噪音 改完变成(P,R)d为噪音维度
        #  选取公共变换矩阵 A 维度是(n+d)*n
        #  每个私有变换矩阵是d*n

        # 初始化的时候 提供共有矩阵并且为每个参与者分配私有矩阵
        self.TestMat = TestMat
        self.Train = Train
        self.participantNum = participantNum
        self.PublicMat = PublicMat
        self.SelfMats = SelfMats
        self.Total = len(LabelSet)
        self.noisyDimonsion = noisyDimonsion
        self.imgs_place = FilePlaces
        self.LabelSet = LabelSet
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        print("初始化完成")

    def __getitem__(self, item):
        img_place = self.imgs_place[item]
        label = self.LabelSet[item]
        img = self.loader(img_place)
        if self.transform is not None:
            img = self.transform(img)
            img = img.float()
        #     将img 从28*28变成 1*784的矩阵
            img = img.view(1,-1)
        #     生成随机噪音 1*noisyDimonsion
            noisy = np.random.normal(0,1,(1,self.noisyDimonsion))
            noisy = torch.from_numpy(noisy)
            # 拼接在一起
            noisy = noisy.float()
            imgChange = torch.cat((img,noisy),1)
        #   到这一步 现在的img为1*884 已经完成了噪音的添加
        #   之后进行与公有矩阵相乘
            step1Mat = imgChange.mm(self.PublicMat)
            # img = step1Mat.view(1,28,-1)

          # 再加私有矩阵的那一部分
          # 假设每个人拥有的数据集个数
            if self.Train==True:
                everyVolume = self.Total/self.participantNum
        #   index表示这是第几个人的数据
                index = int(item/everyVolume)
                selfMat = self.SelfMats[index]
            else:
                selfMat = self.TestMat

            step2Mat = noisy.mm(selfMat)
            # step2Mat = noisy.mm(selfMat)
            #  两者相加就是最终的数据 也就是更正后的图片数据
            EndMat = step1Mat+step2Mat
            img = EndMat.view(1,28,-1)

        return img,label


    def __len__(self):
        return len(self.imgs_place)

