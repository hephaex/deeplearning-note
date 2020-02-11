import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets as dsets
import matplotlib.pyplot as plt

batch_size = 100
train_data = dsets.CIFAR10(root='./tmp/cifar-10', train=True, download=False, transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.33, 3.0))]))
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_data = dsets.CIFAR10(root='./tmp/cifar-10', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),]))
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False)

def image_show(data_loader,n):

  tmp = iter(data_loader)
  images,labels = tmp.next()

  images = images.numpy()

  for i in range(n):
    image = np.transpose(images[i],[1,2,0])
    plt.imshow(image)
    plt.show()

image_show(train_loader,10)
