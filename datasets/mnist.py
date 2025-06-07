from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

def load_MNIST(save_path="datasets/data", batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    trainset = MNIST(root=save_path, train=True, download=True, transform=transform)
    testset = MNIST(root=save_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, 1