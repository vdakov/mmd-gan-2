import matplotlib.pyplot as plt

def plot_loss(training_losses, test_losses=None, title="Training Loss Over Epochs", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss", color='blue')

    if test_losses is not None:
        if len(test_losses) != len(training_losses):
            print("Warning: Length of test_losses does not match training_losses. Plotting may be misaligned.")
        plt.plot(test_losses, label="Test Loss", color='red', linestyle='--')

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()



def plot_multiple_losses(losses_dict, title="Multiple Loss Curves", xlabel="Epoch", ylabel="Loss", save_path=None):
    plt.figure(figsize=(12, 7))

    colors = plt.cm.get_cmap('tab10', len(losses_dict)) # 'tab10' is a good categorical colormap

    for i, (label, loss_values) in enumerate(losses_dict.items()):
        plt.plot(loss_values, label=label, color=colors(i))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()
