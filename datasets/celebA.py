from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import torch


def load_CELEB_A(save_path="datasets/data", batch_size=64):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # Optional: crop faces at center
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CelebA(root=save_path, split='train', download=True, transform=transform)
    valset = CelebA(root=save_path, split='valid', download=False, transform=transform)
    testset = CelebA(root=save_path, split='test', download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader
