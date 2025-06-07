import matplotlib.pyplot as plt 
import numpy as np 

def visualize_image_grid(data, n_rows, n_cols, title):
  
    
    assert len(data) >= n_cols * n_rows, "Ensure you have enough data with your rows and columns!"
    
    subset = np.random.choice(data, n_rows * n_cols)
    i = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, r * n_cols + c + 1)
            x = subset[i]
            i +=1

            plt.imshow(x.reshape(28, 28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()