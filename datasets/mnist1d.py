import torch
from torch.utils.data import TensorDataset, DataLoader
from mnist1d.data import make_dataset, get_dataset_args

def load_MNIST1D(batch_size=64, train_split=0.8, seed=42):
    args = get_dataset_args()
    args.seed = seed
    data = make_dataset(args)

    x, y, t = data['x'], data['y'], data['t']  # t is the 1D template (not used in training)
    
    x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
    
    num_samples = x.shape[0]
    num_train = int(train_split * num_samples)

    x_train = torch.tensor(x[:num_train], dtype=torch.float32).unsqueeze(1)  # (N, 1, 40)
    y_train = torch.tensor(y[:num_train], dtype=torch.long)

    x_test = torch.tensor(x[num_train:], dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y[num_train:], dtype=torch.long)

    # Create TensorDatasets
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
