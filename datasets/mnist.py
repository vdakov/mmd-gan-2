from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

<<<<<<< HEAD
def load_MNIST(save_path="datasets/data", batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),  
=======

def load_MNIST(save_path="datasets/data", batch_size=64, size=28, flatten=False, normalize=True):
    transform_list = [
        transforms.Resize(size),
>>>>>>> 933045324926cf4e4b968fed2d21cb2ee376298d
        transforms.ToTensor(),
    ]

    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    trainset = MNIST(root=save_path, train=True, download=True, transform=transform)
    testset = MNIST(root=save_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

<<<<<<< HEAD
    return trainloader, testloader, 1
=======
    return trainloader, testloader, 0.5, 0.5
>>>>>>> 933045324926cf4e4b968fed2d21cb2ee376298d
