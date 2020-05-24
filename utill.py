import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10
from stage1.lahc import learning_hash
def data_load(path,batch_size):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #be careful
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_loader = DataLoader(CIFAR10(path, train=True, download=True, transform=transform_train),
                              batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print('train set: {}'.format(len(train_loader.dataset)))
    test_loader = DataLoader(CIFAR10(path, train=False, download=True, transform=transform_test),
                             batch_size=batch_size * 8, shuffle=False, num_workers=0, pin_memory=True)
    print('val set: {}'.format(len(test_loader.dataset)))

    data_train_iter=iter(train_loader)
    data_test_iter=iter(test_loader)

    # image_train,label_train=data_train_iter.next()
    # image_test, label_test = data_test_iter.next()

    return data_train_iter,data_test_iter
    pass


def getsimilarity(label,numbers_bit):
    """
    get the similarity matrix by the array of label
    :param label: the labels of image
    :return: the similarity matrix 
    """
    label=label.numpy()
    n=len(label)
    similarity=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            similarity[i][j]=1 if label[i]==label[j] else 0
    h = learning_hash(similarity, numbers_bit, 10, 0.2)
    return h
# data_train_iter,data_test_iter = data_load('/Users/Dream/PycharmProjects/CNNH/data/cifar', 1001)
# for i in range(50):
#     image_train,label_train=next(data_train_iter)
#     print(len(image_train))
# image_train,image_test,label_train,label_test=dataLoad('/Users/Dream/PycharmProjects/CNNH/data/cifar',64)
# similarity=getSimilarity(label_train)
# h=learning_hash(similarity,32,10,0.2)