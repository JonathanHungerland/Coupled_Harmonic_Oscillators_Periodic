import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import re

mpl.rcParams['font.size']=16


def plot_means_of_arrays(file_paths):
    plt.figure(figsize=(6, 6))
    
    for file_path in file_paths:
        # Load the numpy array
        array = np.load(file_path)
        
        # Reshape the array to (N, 100)
        reshaped_array = array.reshape((-1, 100))
        
        # Compute the mean over the rows (mean of each column)
        mean_values = np.mean(reshaped_array, axis=0)
        
        c = re.search(r'\d+\.\d+', file_path).group()

        # Plot the mean values
        plt.plot([i+1 for i in range(100)], mean_values, label=f'c={c}', linewidth=2)
    
    plt.xlabel('Principal Component')
    plt.ylabel('Normalized Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.xlim(1, 100)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('means_of_PCAs.png')

if __name__ == "__main__":
    # Check if file paths are provided as arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_means.py <file1.npy> <file2.npy> ...")
    else:
        # Collect file paths from command line arguments
        file_paths = sys.argv[1:]
        plot_means_of_arrays(file_paths)

