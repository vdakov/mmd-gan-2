from models.gmmn.autoencoder import Autoencoder
import torch 
from tqdm import tqdm
import os 
import torch.optim as optim
from torch.autograd import Variable


def train_autoencoder(trainloader, input_size, encoded_size, num_epochs, device, save_path):
    encoder_net = Autoencoder(input_size, encoded_size).to(device)
    encoder_optim = optim.Adam(encoder_net.parameters())
    losses = []
    
    epoch_pbar = tqdm(range(num_epochs), desc="Autoencoder Training Progress")

    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        encoder_net.load_state_dict(checkpoint['model_state_dict'])
        losses = checkpoint.get('losses', []) # Use .get() with a default to handle older checkpoints
        print("Loaded saved autoencoder model")
    else:
        for ep in epoch_pbar:
            avg_loss = 0
            for idx, (x, _) in enumerate(tqdm(trainloader, desc=f"Epoch {ep+1}/{num_epochs} (Batch); Avg Loss: {avg_loss:.4f}", leave=False)):
                x = x.view(x.size()[0], -1)
                x = Variable(x).to(device)
                _, decoded = encoder_net(x)
                loss = torch.sum((x - decoded) ** 2)
                encoder_optim.zero_grad()
                loss.backward()
                encoder_optim.step()
                avg_loss += loss.item()
            avg_loss /= (idx + 1)
            losses.append(avg_loss)
            
        # epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "Epoch": f"{ep+1}/{num_epochs}"})

        torch.save({
        'model_state_dict': encoder_net.state_dict(),
        'losses': losses,
        }, save_path)

    print("Autoencoder has been successfully trained")
    return encoder_net, losses 