from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.normalization import compute_min_max
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.normalization import compute_min_max


def load_CIFAR(save_path="datasets/data", batch_size=64, size=64, flatten=False, normalize=True):
    raw_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    temp_train = CIFAR10(root=save_path, train=True, download=True, transform=raw_transform)

    if normalize:
        min_val, max_val = compute_min_max(temp_train)

    transform_list = [
        transforms.Resize(size),
        transforms.ToTensor()
    ]

    if normalize:
        transform_list.append(
            transforms.Lambda(lambda x: (x - min_val[:, None, None]) / (max_val[:, None, None] - min_val[:, None, None]))
        )

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    trainset = CIFAR10(root=save_path, train=True, download=False, transform=transform)
    testset = CIFAR10(root=save_path, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if normalize:
        return trainloader, testloader, min_val, max_val
    else:
        return trainloader, testloader, None, None

