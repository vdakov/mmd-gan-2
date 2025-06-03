import numpy as np 
from scipy.stats import f 

def calculate_hotelling_t_squared(X, Y):
    """
    Calculates Hotelling's T-squared statistic for two samples X and Y.
    X and Y are assumed to be 2D arrays where rows are observations and columns are dimensions.
    """
    n_x, p_x = X.shape # n_x: number of observations, p_x: number of dimensions
    n_y, p_y = Y.shape

    if p_x != p_y:
        raise ValueError("Number of dimensions must be the same for both datasets.")
    p = p_x # Number of dimensions

    mean_x = np.mean(X, axis=0) # Mean vector for X
    mean_y = np.mean(Y, axis=0) # Mean vector for Y

    # Calculate pooled covariance matrix
    cov_x = np.cov(X, rowvar=False) # rowvar=False means columns are variables
    cov_y = np.cov(Y, rowvar=False)


    try:
        pooled_cov = ((n_x - 1) * cov_x + (n_y - 1) * cov_y) / (n_x + n_y - 2)
        pooled_cov_inv = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        # Handle singular matrix case - for simplicity here, we'll return a large value.
        # In a real application, you might want to log this or raise a more specific error.
        print("Warning: Pooled covariance matrix is singular. Hotelling's T-squared might be unreliable.")
        return np.inf

    mean_diff = mean_x - mean_y

    t_squared = (n_x * n_y) / (n_x + n_y) * (mean_diff @ pooled_cov_inv @ mean_diff.T)
    return t_squared

def bootstrap_hypothesis_test(original_data, generating_function, generating_function_args, alpha=0.05, num_iterations=1000): 
    generated_data = generating_function(*generating_function_args)
    generated_mean = np.mean(generated_data)
    generated_var = np.var(generated_data)
    n_x, p = original_data.shape # n_x: number of observations, p: number of dimensions
    n_y = generated_data.shape[0]

    
    original_mean = np.mean(original_data)
    original_var = np.var(original_data)

    z = np.mean(np.concatenate((original_data, generated_data), axis=0), axis=0)

    
    # t = calculate_t(original_mean, generated_mean, original_var, generated_var, n_x, n_y)
    t = calculate_hotelling_t_squared(original_data, generated_data)
    
    zero_centered_original = np.add(np.subtract(original_data, original_mean), z) 
    zero_centered_generated = np.add(np.subtract(generated_data, generated_mean), z) 
    
    t_s = bootstrap_multidimensional(zero_centered_original, zero_centered_generated, n_x, n_y, num_iterations)
    print(t_s)
    
    p_value = np.sum(np.abs(t_s) >= np.abs(t)) / num_iterations
    
    if p_value < alpha:
        print(f"Reject H_0: Conclusion: Distributions A and B are significantly different (p={p_value:.4f}).")
    else:
        print(f"Conclusion: We fail to reject the H_0. Distributions A and B are not significantly different (p={p_value:.4f}).")

    return p_value

def bootstrap_multidimensional(zero_centered_x, zero_centered_y, n_x, n_y, num_iterations):
    """
    Performs bootstrap resampling for multidimensional data and calculates Hotelling's T-squared.
    zero_centered_x and zero_centered_y are 2D arrays (rows are observations, columns are dimensions).
    """
    t_squared_s = []
    p = zero_centered_x.shape[1] # Number of dimensions

    for _ in range(num_iterations):
        sample_x_indices = np.random.choice(n_x, n_x, replace=True)
        sample_y_indices = np.random.choice(n_y, n_y, replace=True)

        sample_x = zero_centered_x[sample_x_indices, :]
        sample_y = zero_centered_y[sample_y_indices, :]
        
        t_squared_s.append(calculate_hotelling_t_squared(sample_x, sample_y))
        
    return t_squared_s
    
    
    
    
def calculate_t(mean_x, mean_y, var_x, var_y, n, m):
    denominator = np.sqrt(var_x / n + var_y / m)

    if denominator == 0:
        return np.inf if (mean_x - mean_y) > 0 else (-np.inf if (mean_x - mean_y) < 0 else 0)
    else:
        t = (mean_x - mean_y) / denominator
        return t
    
def bootstrap(zero_centered_x, zero_centered_y, n, m, num_iterations):
    t_s = []
    B = num_iterations 
    
    for _ in range(B):
        sample_x = np.random.choice(zero_centered_x, n, replace=True)
        sample_y = np.random.choice(zero_centered_y, m, replace=True)
        mean_x = np.mean(sample_x)
        mean_y = np.mean(sample_y)
        var_x = np.var(sample_x)
        var_y = np.var(sample_y)
        
        t_s.append(calculate_t(mean_x, mean_y, var_x, var_y, n, m))
        
    return t_s
    