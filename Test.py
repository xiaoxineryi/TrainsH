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
import MyImageLoader
batch_size = 64


mytransform = transforms.Compose([
    transforms.ToTensor()
    ]
)
batch_size = 64

train_loader = DataLoader(MyImageLoader.MyImageFloder(FileName='data/MNIST/raw/train.txt',transform=mytransform), batch_size=batch_size,
                                           shuffle=True)

test_loader = DataLoader(MyImageLoader.MyImageFloder(FileName='data/MNIST/raw/test.txt',transform=mytransform), batch_size=batch_size,
                                           shuffle=True)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)#（in_features, out_features）

    def forward(self, x):
        # in_size = 64
        in_size = x.size(0) # one batch     此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  feature map =[(28-4)/2]^2=12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4
        x = F.relu(self.mp(self.conv3(x)))

        x = x.view(in_size, -1) # flatten the tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        return F.log_softmax(x)  #64*10

model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

losses =[]
test_losses = []
test_acces = []

def train(epoch):
    loss=0;

    for batch_idx, (data, target) in enumerate(train_loader):#batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]

        output = model(data)
        #output:64*10


        # target = [np.long(i) for i in target]
        # target = torch.Tensor(target).long()


        loss = F.nll_loss(output, target)

        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        optimizer.zero_grad()   # 所有参数的梯度清零
        loss.backward()         #即反向传播求梯度
        optimizer.step()        #调用optimizer进行梯度下降更新参数
        # 每次训练完一整轮，记录一次
    losses.append(loss.item())




def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # target = [np.long(i) for i in target]
        # target = torch.Tensor(target).long()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]

            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_losses.append(test_loss)
    print(correct.item())
    test_acces.append(correct.item() / len(test_loader.dataset))

for epoch in range(1, 5):
    train(epoch)
    test()



plt.figure(22)

x_loss = list(range(len(losses)))
x_acc = list(range(len(test_acces)))

plt.subplot(221)
plt.title('train loss ')
plt.plot(x_loss, losses)

plt.subplot(222)
plt.title('test loss')
plt.plot(x_loss, test_losses)

plt.subplot(212)
plt.plot(x_acc, test_acces)
plt.title('test acc')
plt.show()


