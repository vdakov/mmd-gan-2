from models.gmmn.gmmn import GMMN
import torch 
from tqdm import tqdm
import os 
import torch.optim as optim
from torch.autograd import Variable


def get_scale_matrix(M, N, device):
    s1 = (torch.ones((N, 1)) * 1.0 / N).to(device)
    s2 = (torch.ones((M, 1)) * -1.0 / M).to(device)
    return torch.cat((s1, s2), 0)

def train_one_step(x, net, samples, optimizer, device, sigma=[1]):
    samples = Variable(samples).to(device)
    gen_samples = net(samples)
    X = torch.cat((gen_samples, x), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()

    M = gen_samples.size()[0]
    N = x.size()[0]
    s = get_scale_matrix(M, N, device)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(exp / v)
        loss += torch.sum(S * kernel_val)

    loss = torch.sqrt(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_gmmn(trainloader, autoencoder, data_size, noise_size, batch_size, num_epochs, device, save_path):

    gmm_net = GMMN(noise_size, data_size).to(device)
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        gmm_net.load_state_dict(checkpoint['model_state_dict'])
        losses = checkpoint.get('losses', []) # Use .get() with a default to handle older checkpoints
        print("Loaded previously saved GMM Network and losses from checkpoint.")
    else:
        gmmn_optimizer = optim.Adam(gmm_net.parameters(), lr=0.001)
        losses = []

        epoch_pbar = tqdm(range(num_epochs), desc="GMMN Training Progress")

        for ep in epoch_pbar:
            avg_loss = 0
            for idx, (x, _) in enumerate(tqdm(trainloader, desc=f"Epoch {ep+1}/{num_epochs} (Batch)", leave=False)):
                x = x.view(x.size()[0], -1)
                with torch.no_grad():
                    x = Variable(x).to(device)
                    encoded_x = autoencoder.encode(x)

                # uniform random noise between [-1, 1]
                random_noise = torch.rand((batch_size, noise_size)) * 2 - 1
                loss = train_one_step(encoded_x, gmm_net, random_noise, gmmn_optimizer, device)
                avg_loss += loss.item()

            avg_loss /= (idx + 1)
            losses.append(avg_loss)
            epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "Epoch": f"{ep+1}/{num_epochs}"})

        torch.save({
        'model_state_dict': gmm_net.state_dict(),
        'losses': losses,
        }, save_path)
    
    return gmm_net, losses