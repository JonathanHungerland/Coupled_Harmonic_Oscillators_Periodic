from MyOscillators.Oscillators import CoupledOscillators
import matplotlib.pyplot as plt
from MyOscillators.MyPCA import MyPCA
import time
import numpy as np
import concurrent.futures

def compute_pca(N_oscillators, k, c, periods=125, store_per_period=5):
    oscillators = CoupledOscillators('PCA_vs_Coupling2/PCA_vs_Coupling2',N_oscillators, k, c, verbose=False)
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

def converge_pca(name, c, N_oscillators=100, k=np.arange(0.1,10.1,0.1), periods=125, store_per_period=5):
    rng=np.random.default_rng(seed=0)
    rng.shuffle(k)
    pcas = np.array([])
    for i in range(4):
        rng.shuffle(k)
        pcas = np.append(pcas,(compute_pca(N_oscillators, k, c)))
    i = 1
    while compute_convergence(pcas) > 0.001:
        for _ in range(4):
            rng.shuffle(k)
            pcas = np.append(pcas,(compute_pca(N_oscillators, k, c)))
        i+=1
    np.save(name+".npy", pcas)
    return "Computation for c=%.2f completed after %d iterations"%(c, i)
    
num_tasks = 64
#c_values = np.concatenate((np.arange(0,0.1,0.01), np.arange(0.1,1.0,0.1), np.arange(1,10.0,0.5),np.arange(10,50,10)))
c_values = np.concatenate((np.arange(15, 55, 10), np.arange(60, 150, 10)))
with concurrent.futures.ProcessPoolExecutor(max_workers=num_tasks) as executor:
    futures = {executor.submit(converge_pca, 'PCA_vs_Coupling2/c%2.2f'%c, c, periods=125, store_per_period=5): c for c in c_values}
    for future in concurrent.futures.as_completed(futures):
        task_id = futures[future]
        try:
            converged_pca = future.result()
            print(converged_pca)
        except Exception as e:
            print(f"Task {task_id} generated an exception: {e}")



