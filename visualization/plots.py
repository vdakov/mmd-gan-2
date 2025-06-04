from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

def plot_3d_histogram(data, bins=20):
    """
    Plots a 3D histogram of the input data.

    Parameters:
        data (ndarray): Input data as a 2D array of shape (n_samples, D), where D >= 2.
                        The first two columns represent X and Y.
        bins (int): Number of bins for the histogram.
    """
    # Ensure the data has at least two dimensions
    if data.shape[1] < 2:
        raise ValueError("Input data must have at least two columns: X and Y.")

    # Extract X and Y from the data
    X = data[:, 0]
    Y = data[:, 1]

    # Compute the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(X, Y, bins=bins)

    # Construct the grid for the histogram
    x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
    x_pos = x_pos.ravel()
    y_pos = y_pos.ravel()
    z_pos = np.zeros_like(x_pos)

    # Flatten the histogram values
    dx = dy = (x_edges[1] - x_edges[0]) * 0.9  # Width of the bars
    dz = hist.ravel()

    # Plot the 3D histogram
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, shade=True, cmap='viridis')
    ax.set_title('3D Histogram')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Frequency')
    plt.show()
    
    
def plot_3d_kde(data, title="Sample title"):
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        xlabel, ylabel = "PC1", "PC2"
    else:
        data_2d = data
        xlabel, ylabel = "X", "Y"

    kde = gaussian_kde(data_2d.T)
    x = np.linspace(data_2d[:, 0].min(), data_2d[:, 0].max(), 100)
    y = np.linspace(data_2d[:, 1].min(), data_2d[:, 1].max(), 100)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title("3D KDE Surface of Mixture Distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Density")
    plt.title(title)
    plt.show()
