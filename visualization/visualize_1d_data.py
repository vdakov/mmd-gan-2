from matplotlib import pyplot as plt
import numpy as np
import torch


def visualize_mnist1d(data_samples, labels=None, title_prefix="MNIST-1D Samples",
                      num_samples_to_show=5, figsize_per_sample=(10, 2),
                      ylim=(-1.1, 1.1)):

    if isinstance(data_samples, torch.Tensor):
        data_samples = data_samples.squeeze().cpu().numpy()
    
    is_single_sample = data_samples.ndim == 1 or \
                       (data_samples.ndim == 2 and data_samples.shape[0] == 1)

    if is_single_sample:
        data_samples = data_samples.reshape(1, -1)
        num_samples_to_show = 1
        if labels is not None and not isinstance(labels, (list, np.ndarray, torch.Tensor)):
            labels = [labels]

    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.squeeze().cpu().numpy()

    num_samples_to_display = min(num_samples_to_show, data_samples.shape[0])

    _, axes = plt.subplots(num_samples_to_display, 1,
                             figsize=(figsize_per_sample[0], figsize_per_sample[1] * num_samples_to_display))

    # Ensure axes is an array even if num_samples_to_display is 1
    if num_samples_to_display == 1:
        axes = [axes]

    for i in range(num_samples_to_display):
        ax = axes[i]
        sample = data_samples[i].flatten() # Ensure sample is 1D
        label = labels[i] if labels is not None and i < len(labels) else None

        ax.plot(sample)
        current_title = f"{title_prefix} {i+1}" if not is_single_sample else title_prefix
        ax.set_title(f"{current_title}{f' (Label: {label})' if label is not None else ''}", fontsize=10)
        ax.set_xlabel("Position / Feature Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=7)

    plt.tight_layout()
    plt.show()