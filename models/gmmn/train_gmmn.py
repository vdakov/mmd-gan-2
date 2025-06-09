from models.gmmn.gmmn import GMMN
import torch 
from tqdm import tqdm
import os 
import torch.optim as optim
from torch.autograd import Variable

def gaussian_kernel_matrix(x, y, sigma):
    """Compute full Gaussian kernel matrix between x and y."""
    # x: (n, d), y: (m, d)
    x_norm = torch.sum(x**2, dim=1, keepdim=True)  # (n, 1)
    y_norm = torch.sum(y**2, dim=1, keepdim=True)  # (m, 1)
    dist_sq = x_norm - 2 * torch.mm(x, y.t()) + y_norm.t()  # (n, m)
    return torch.exp(-dist_sq / (2 * sigma**2))

def mmd_loss(x_real, x_fake, sigmas=[1, 5, 10]):
    """Unbiased MMD loss with full kernel matrix computation."""
    loss = 0
    n = x_real.size(0)
    
    for sigma in sigmas:
        # Compute kernel matrices
        k_real_real = gaussian_kernel_matrix(x_real, x_real, sigma)
        k_fake_fake = gaussian_kernel_matrix(x_fake, x_fake, sigma)
        k_real_fake = gaussian_kernel_matrix(x_real, x_fake, sigma)
        
        # Remove diagonals for unbiased estimate
        real_real = (k_real_real.sum() - k_real_real.trace()) / (n*(n-1))
        fake_fake = (k_fake_fake.sum() - k_fake_fake.trace()) / (n*(n-1))
        real_fake = k_real_fake.mean()
        
        loss += real_real + fake_fake - 2 * real_fake
        
    return loss / len(sigmas) # so it is characteristic

# UPDATED TRAINING STEP
def train_one_step(x, net, samples, optimizer, device, sigmas, clip_value=1.0):
    samples = samples.to(device)
    gen_samples = net(samples)
    
    loss = mmd_loss(x, gen_samples, sigmas)
    # loss = torch.sqrt(loss)
    
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
    optimizer.step()
    
    return loss.item()



def train_gmmn(trainloader, autoencoder, sigmas, data_size, noise_size, batch_size, num_epochs, device, save_path):
    

    gmm_net = GMMN(noise_size, data_size).to(device)    
    gmmn_optimizer = optim.Adam(gmm_net.parameters(), lr=0.001)
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        gmm_net.load_state_dict(checkpoint['model_state_dict'])
        losses = checkpoint.get('losses', []) # Use .get() with a default to handle older checkpoints
        print("Loaded previously saved GMM Network and losses from checkpoint.")
    else:
        
        losses = []
        epoch_pbar = tqdm(range(num_epochs), desc="GMMN Training Progress")

        for ep in epoch_pbar:
            avg_loss = 0
            for idx, (x, _) in enumerate(trainloader):
                x = x.view(x.size()[0], -1)
                with torch.no_grad():
                    x = x.to(device)
                    encoded_x = autoencoder.encode(x)

                # uniform random noise between [-1, 1]
                random_noise = torch.rand((batch_size, noise_size)) * 2 - 1
                loss = train_one_step(encoded_x, gmm_net, random_noise, gmmn_optimizer, device, sigmas)
                avg_loss += loss

            avg_loss /= (idx + 1)
            losses.append(avg_loss)
            epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "Epoch": f"{ep+1}/{num_epochs}"})

        torch.save({
        'model_state_dict': gmm_net.state_dict(),
        'losses': losses,
        }, save_path)
    
    return gmm_net, losses