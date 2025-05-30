#Taken from https://github.com/Abhipanda4/GMMN-Pytorch 
import torch.nn as nn
import torch.nn.functional as F
import torch

class GMMN(nn.Module):
    def __init__(self, n_start, n_out):
        super(GMMN, self).__init__()
        self.fc1 = nn.Linear(n_start, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 784)
        self.fc5 = nn.Linear(784, n_out)

    def forward(self, samples):
        x = F.relu(self.fc1(samples))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return x
    
    
    
def generate_gmmn_samples(model, autoencoder, NOISE_SIZE, number, device='cpu'):
    model.eval()
    model.to(device) 

    generated_samples = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for _ in range(number):
            # Generate uniform random noise between [-1, 1]
            # Use '1' for batch size as we're generating one sample at a time
            # and then concatenating. If 'number' is large, you could generate
            # in batches for efficiency.
            noise = torch.rand((1, NOISE_SIZE)) * 2 - 1
            noise = noise.to(device) # Move noise to the same device as the model

            # Generate sample
            sample = model(noise)
            sample = autoencoder.decode(sample)
            generated_samples.append(sample.cpu()) # Move generated sample to CPU for storage

    # Concatenate all generated samples into a single tensor
    return torch.squeeze(torch.cat(generated_samples, dim=0)).numpy()
