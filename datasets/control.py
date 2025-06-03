import numpy as np
import sklearn
import sklearn.model_selection
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets.normalization import MinMaxNormalize

def get_mixture_distribution(mixture_list, weights, size=(10000, 2)):
    assert len(mixture_list) == len(weights), "Distributions and weights must match!"
    for i, entry in enumerate(mixture_list):
        assert isinstance(entry, tuple), f"mixture_list[{i}] must be a tuple, got {type(entry)}"
    
    components = np.random.choice(len(weights), size=size[0], p=weights)
    mixture_data = np.empty(size)
    n_dims = size[1]
    
    for comp_id in range(len(weights)):
        mask = (components == comp_id)
        n_samples = np.sum(mask)
        
        distribution_type = mixture_list[comp_id][0]
        args = mixture_list[comp_id][1:]
        
        
        if n_samples == 0:
            continue
        
        if distribution_type == "uniform":
            low, high = args
            assert low.size == n_dims, f"Uniform 'low' param must have exactly {size[1]} elements at mixture_list[{i}]"
            assert high.size == n_dims, f"Uniform 'high' param must have exactly {size[1]} elements at mixture_list[{i}]"
            mixture_data[mask] = np.random.uniform(low, high, size=(n_samples, size[1]))
        elif distribution_type == "normal":
            loc, scale = args
            assert loc.size == n_dims, f"Normal 'loc' param must have exactly {n_dims} elements at mixture_list[{i}]"
            assert scale.size == n_dims, f"Normal 'scale' param must have exactly {n_dims} elements at mixture_list[{i}]"
            mixture_data[mask] = np.random.normal(loc, scale, size=(n_samples, size[1]))
        elif distribution_type == "laplace":
            loc, scale = args
            assert loc.size == n_dims, f"Laplace 'loc' param must have exactly {n_dims} elements at mixture_list[{i}]"
            assert scale.size == n_dims, f"Laplace 'scale' param must have exactly {n_dims} elements at mixture_list[{i}]"
            mixture_data[mask] = np.random.laplace(loc, scale, size=(n_samples, size[1]))
        elif distribution_type == "exponential":
            scale = args[0]
            samples = np.random.exponential(scale, size=(n_samples, size[1]))
            assert scale.size == n_dims, f"Exponential 'scale' param must have exactly {n_dims} elements at mixture_list[{i}]"
            if len(args) > 1:
                shift = np.array(args[1])
                assert shift.size == n_dims, f"Exponential 'shift' param must have exactly {n_dims} elements at mixture_list[{i}]"
            mixture_data[mask] = samples
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}. Check for typos and such!")
    return mixture_data

def load_control(data, batch_size=64, num_classes=2, label_type="random", normalize=True):
    """
    label_type: "random" | "zeros" | "ones"
    num_classes: only used if label_type == "random"
    """
    train_data, test_data = sklearn.model_selection.train_test_split(data, test_size=0.2)
    
    # Convert data to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Create fake labels
    if label_type == "random":
        train_labels = torch.randint(0, num_classes, (train_data.shape[0],))
        test_labels = torch.randint(0, num_classes, (test_data.shape[0],))
    elif label_type == "zeros":
        train_labels = torch.zeros(train_data.shape[0], dtype=torch.long)
        test_labels = torch.zeros(test_data.shape[0], dtype=torch.long)
    elif label_type == "ones":
        train_labels = torch.ones(train_data.shape[0], dtype=torch.long)
        test_labels = torch.ones(test_data.shape[0], dtype=torch.long)
    else:
        raise ValueError(f"Unsupported label_type '{label_type}'")
    
        
    transform = None
    if normalize:
        min_vals = train_data.min(dim=0)[0]
        max_vals = train_data.max(dim=0)[0]
        transform = MinMaxNormalize(min_vals, max_vals)
    else:
        min_vals = max_vals = None
        

    # Wrap in TensorDataset
    trainset = TensorDataset(train_data, train_labels)
    testset = TensorDataset(test_data, test_labels)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, min_vals, max_vals

