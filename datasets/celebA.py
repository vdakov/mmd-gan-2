from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import torch
from datasets.normalization import compute_min_max

<<<<<<< HEAD

def load_CELEB_A(save_path="datasets/data", batch_size=64):
    transform = transforms.Compose([ 
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
=======
def load_CELEB_A(save_path="datasets/data", batch_size=64, size=64, normalize=True):
    raw_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
>>>>>>> 933045324926cf4e4b968fed2d21cb2ee376298d
    ])
    
    temp_train = CelebA(root=save_path, split='train', download=True, transform=raw_transform)

    transform_list = [
        transforms.CenterCrop(178),  # Optional: crop faces at center
        transforms.Resize(size),
        transforms.ToTensor(),
    ]

    if normalize:
        min_val, max_val = compute_min_max(temp_train)
        transform_list.append(
            transforms.Lambda(lambda x: (x - min_val[:, None, None]) / (max_val[:, None, None] - min_val[:, None, None]))
        )

    transform = transforms.Compose(transform_list)

    trainset = CelebA(root=save_path, split='train', download=True, transform=transform)
    valset = CelebA(root=save_path, split='valid', download=False, transform=transform)
    testset = CelebA(root=save_path, split='test', download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

<<<<<<< HEAD
    return trainloader, valloader, testloader, 3
=======
    if normalize:
        return trainloader, valloader, testloader, min_val, max_val
    else:
        return trainloader, valloader, testloader
>>>>>>> 933045324926cf4e4b968fed2d21cb2ee376298d
