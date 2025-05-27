from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch

def load_CIFAR(save_path="datasets/data", batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])

    trainset = CIFAR10(root=save_path, train=True, download=True, transform=transform)
    testset = CIFAR10(root=save_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader
