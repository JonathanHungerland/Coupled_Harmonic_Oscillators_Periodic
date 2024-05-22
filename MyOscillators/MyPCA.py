import numpy as np
from scipy.linalg import eig, eigvals
from scipy.optimize import curve_fit

class MyPCA:
    def __init__(self, trajectory: list):
        self.trajectory = np.array(trajectory)
        self.length = self.trajectory.shape[0]
        self.shape = self.trajectory.shape[1]

    def covariance_matrix(self,start_slice=None, end_slice=-1):
        if start_slice is None:
            return np.cov(self.trajectory, rowvar=False,dtype=np.float128)
        if type(start_slice) == int:
            return np.cov(self.trajectory[start_slice, end_slice], rowvar=False,dtype=np.float128)
        if type(start_slice) == float and type(end_slice) == float:
            first_timestep = int(start_slice * self.length)
            last_timestep = int(end_slice * self.length)
            return np.cov(self.trajectory[first_timestep:last_timestep], rowvar=False,dtype=np.float128)
        print("Invalid slicing definition! Expected int or float.")
        exit()

    def is_converged(self, threshold=1e-4):
        first_batch = self.covariance_matrix(start_slice=0.2, end_slice=0.6)
        second_batch = self.covariance_matrix(start_slice=0.6, end_slice=1.0)
        
        deviation = np.abs(first_batch - second_batch)
        max_deviation = np.max(deviation)
        if max_deviation < threshold:
            return True
        else:
            return max_deviation 
    
    def eigen(self):
        return eig(self.covariance_matrix(start_slice=0.2, end_slice=1.0))
    
    def eigenvalues(self):
        values = eigvals(self.covariance_matrix(start_slice=0.2, end_slice=1.0))
        values = values.real.astype(np.float64)
        values = np.sort(values)[::-1]
        values = values/values[0]
        return values
    
    def pca_slope(self):
        eigenvalues = self.eigenvalues().astype(np.float64)
        eigenvalues = eigenvalues/eigenvalues[0]
        
        def exponential_decay(x,a,b,c):
            return a * np.exp(-b * x) + c

        x = np.arange(len(eigenvalues))
        initial_a = eigenvalues[0]
        initial_b = eigenvalues[0] / eigenvalues[1]
        initial_c = 0
        popt, pcov = curve_fit(exponential_decay, x, eigenvalues, \
                               p0=(initial_a, initial_b, initial_c))
        return popt[1]

