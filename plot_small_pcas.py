import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

mpl.rcParams['font.size']=16

def read_number_of_small_pcas(file_path, treshold=0.001):
    pcas = np.load(file_path)
    reshaped_array = pcas.reshape((-1, 100))
    mean_values = np.mean(reshaped_array, axis=0)
    c = re.search(r'\d+\.\d+', file_path).group()
    return c, len([i for i in mean_values if i < treshold])




if __name__ == "__main__":
    # Check if file paths are provided as arguments
    if len(sys.argv) < 2:
        print("Usage: python plot_means.py <file1.npy> <file2.npy> ...")
    else:
        # Collect file paths from command line arguments
        
        plt.figure(figsize=(6, 6))
        couplings, small_pcas = zip(*[read_number_of_small_pcas(file_path, treshold=1E-2) for file_path in sys.argv[1:]])
        relative_couplings = np.array(couplings, dtype=float)
        plt.scatter(relative_couplings, small_pcas, label='treshold=1E-2')

        couplings, small_pcas = zip(*[read_number_of_small_pcas(file_path, treshold=1E-4) for file_path in sys.argv[1:]])
        relative_couplings = np.array(couplings, dtype=float)
        plt.scatter(relative_couplings, small_pcas, label='treshold=1E-4')

        couplings, small_pcas = zip(*[read_number_of_small_pcas(file_path, treshold=1E-5) for file_path in sys.argv[1:]])
        relative_couplings = np.array(couplings, dtype=float)
        plt.scatter(relative_couplings, small_pcas, label='treshold=1E-5')

        couplings, small_pcas = zip(*[read_number_of_small_pcas(file_path, treshold=1E-6) for file_path in sys.argv[1:]])
        relative_couplings = np.array(couplings, dtype=float)
        plt.scatter(relative_couplings, small_pcas, label='treshold=1E-6')

        plt.xlabel(r'$c$')
        plt.xlim(0, max(relative_couplings))
        plt.ylim(0,100)
        plt.ylabel('Number of\nnormalized eigenvalues < threshold')
        plt.legend(framealpha=1, loc='center')
        plt.tight_layout()
        plt.savefig('small_pcas.png')
