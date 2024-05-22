from MyOscillators.Oscillators import CoupledOscillators
import matplotlib.pyplot as plt
from MyOscillators.MyPCA import MyPCA
import time
import numpy as np
import concurrent.futures

def compute_pca(N_oscillators, N_frozen, k, c, periods=125, store_per_period=5):
    oscillators = CoupledOscillators('PCA_vs_Dimension/PCA_vs_Dimension',N_oscillators, k, c, verbose=False)
    oscillators.run(periods=periods, store_per_period=store_per_period, store_on_disk=False)
    pca = MyPCA(oscillators.get_position_trajectory())
    return pca.eigenvalues()

def compute_convergence(pcas):
    #Split the computations into four sets
    sets = np.array_split(pcas,4)
    means = [np.mean(s, axis=0) for s in sets]
    means_matrix = np.vstack(means)
    max_abs_deviations = np.max(np.abs(means_matrix - np.mean(means_matrix, axis=0)), axis=0)
    #print("Current max abs deviation: ", np.max(max_abs_deviations), flush=True)
    return np.max(max_abs_deviations)

def converge_pca(name, N_oscillators, N_frozen, k=np.arange(0.1,10.1,0.1), c=0, periods=125, store_per_period=5):
    for i in range(N_frozen):
        k[(i+1)] = k[0]

    rng=np.random.default_rng(seed=0)
    rng.shuffle(k)
    pcas = np.array([])
    for i in range(12):
        rng.shuffle(k)
        pcas = np.append(pcas,(compute_pca(N_oscillators, N_frozen, k, c)))
    i = 1
    while compute_convergence(pcas) > 0.001:
        for _ in range(4):
            rng.shuffle(k)
            pcas = np.append(pcas,(compute_pca(N_oscillators, N_frozen, k, c)))
        i+=1
    np.save(name+".npy", pcas)
    return "Computation for N=%d completed after %d iterations"%(N_oscillators-N_frozen, i)
    
num_tasks = 64
N_frozens = [i for i in range(100)]
with concurrent.futures.ProcessPoolExecutor(max_workers=num_tasks) as executor:
    futures = {executor.submit(converge_pca, 'PCA_vs_Dimension/N%d'%(100-N), 100, N, periods=125, store_per_period=5): N for N in N_frozens}
    for future in concurrent.futures.as_completed(futures):
        task_id = futures[future]
        try:
            converged_pca = future.result()
            print(converged_pca)
        except Exception as e:
            print(f"Task {task_id} generated an exception: {e}")