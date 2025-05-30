from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch

def load_MNIST(save_path="datasets/data", batch_size=64, size=28, flatten=False):
    """
    Loads MNIST dataset with optional flattening.

    Args:
        save_path (str): Directory where the dataset should be downloaded/loaded from.
        batch_size (int): Batch size for the DataLoaders.
        flatten (bool): If True, flattens the 28x28 images into a 1D vector of 784.

    Returns:
        tuple: (trainloader, testloader)
    """
    transform_list = [
        transforms.Resize(size),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    if flatten:
        transform_list.append(transforms.Lambda(lambda x: x.view(-1)))

    transform = transforms.Compose(transform_list)

    trainset = MNIST(root=save_path, train=True, download=True, transform=transform)
    testset = MNIST(root=save_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

