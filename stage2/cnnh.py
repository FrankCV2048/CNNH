import torch.nn as nn
import torch
import torch.optim as optim
from utill import data_load,getsimilarity
import torch.nn.init as init

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
            # nn.Dropout(0.5),
            nn.ReLU(inplace=True),

            nn.Linear(500, numbers_bit),
        )
        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

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
    h=torch.as_tensor(h)
    loss_1=torch.sqrt(torch.sum(torch.pow(output-h,2)))/batch_size
    return loss_1
    pass

def train(batch_size,hash_bit):
    """
    train
    :param batch_size: the size of batch image into model
    :param hash_bit: the length of hash
    :return: None
    """
    model=CNNH(hash_bit)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    data_train_load, data_test_load = data_load('/Users/Dream/PycharmProjects/CNNH/data/cifar',
                                                batch_size)
    for epoch in range(4):
        for i, (imgs, labels) in enumerate(data_train_load):
            H=getsimilarity(labels,numbers_bit=hash_bit)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = hash_loss(outputs, H,batch_size)
            print(loss.item())
            loss.backward()
            optimizer.step()
    pass

def test():
    pass

train(1000,10)

