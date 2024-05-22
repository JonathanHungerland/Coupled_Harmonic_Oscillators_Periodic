from MyOscillators.Oscillators import CoupledOscillators
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

N_oscillators = 5
k = 1
c = 0
target_time = 100

def plot_qp_diagram(name, trajectory, velocity_trajectory):
    plt.figure(figsize=(10,10))
    plt.plot(trajectory, velocity_trajectory, 'b-')
    min_max = max(np.abs([np.min(trajectory), np.max(trajectory), np.min(velocity_trajectory), np.max(velocity_trajectory)]))
    plt.xlim(-min_max, min_max)
    plt.ylim(-min_max, min_max)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title(name)
    plt.savefig(name+'.png')
    plt.close()

def plot_energy_over_time(name, times, energies):
    plt.figure(figsize=(10,10))
    plt.plot(times, energies, 'b-')
    plt.xlim(0, times[-1])
    plt.ylim(np.min(energies), np.max(energies))
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(name)
    plt.savefig(name+'.png')
    plt.close()

print("deltaT\t\tEnergySlope\tSlopeStderr\tEnergyStdDev\tEnergyMinMax", flush=True)
for timestep in [1E-7, 1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1][::-1]:
    required_steps = int(target_time/timestep)
    store_frequency = 0.1/timestep

    oscillators = CoupledOscillators('TimeStep_Convergence/Uncoupled_Oscillators_fourth_order/ConvergeTimestep_%.2E'%timestep,N_oscillators, k, c, dt=timestep)
    oscillators.randomize_positions(target_energy=10, seed=1)
    oscillators.run(required_steps, store_frequency=store_frequency,method='symplectic-4th-order')

    energies = oscillators.get_energy_trajectory()
    times = np.arange(len(energies))*timestep*store_frequency

    slope, intercept, r_value, p_value, std_err = linregress(times.astype(np.float64), energies.astype(np.float64))

    print("%.2E\t%.2E\t%.2E\t%.2E\t%.2E"%(timestep, slope, std_err, np.std(energies), np.max(energies)-np.min(energies)), flush=True)
    
    plot_qp_diagram('TimeStep_Convergence/Uncoupled_Oscillators_fourth_order/ConvergeTimestep_%.2E_pq'%timestep, oscillators.get_position_trajectory()[:,0], oscillators.get_velocity_trajectory()[:,0])
    plot_energy_over_time('TimeStep_Convergence/Uncoupled_Oscillators_fourth_order/ConvergeTimestep_%.2E_energy'%timestep, times, energies)
    
    