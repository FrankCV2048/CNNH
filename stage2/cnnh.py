import torch.nn as nn
import torch
import torch.optim as optim
from utill import data_load,getsimilarity
import math

class CNNH(nn.Module):
    def __init__(self,numbers_bit):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc=nn.Sequential(
            nn.Linear(128 * 3 * 3, 500),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(500, numbers_bit)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def hash_loss(output,h,batch_size):
    """
    calculate the loss
    :param output: the output of the model
    :param h: the approximate hash
    :return: loss
    """
    criterion = nn.CrossEntropyLoss()
    h=torch.as_tensor(h)
    # loss=criterion(output,h)
    loss_1=torch.sum(torch.pow(output-h,2))/batch_size
    return loss_1
    pass

def train(batch_size,hash_bit):
    """
    
    :param batch_size: the size of batch image into model
    :param hash_bit: the length of hash
    :return: None
    """
    model=CNNH(hash_bit)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    data_train_iter, data_test_iter = data_load('/Users/Dream/PycharmProjects/CNNH/data/cifar',
                                                batch_size)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(math.ceil(50000/batch_size)):
            image_train,label_train=data_train_iter.next()
            H=getsimilarity(label_train,numbers_bit=hash_bit)
            optimizer.zero_grad()
            outputs = model(image_train)
            loss = hash_loss(outputs, H,batch_size)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    pass

def test():
    pass

train(100,10)

