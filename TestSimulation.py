from Oscillators import CoupledOscillators
import numpy as np

N_oscillators = 10
k = 10
c = 10
dt = 1e-4

oscillators = CoupledOscillators('TestSimulation',N_oscillators, k, c, dt=dt)
oscillators.initialize(np.random.rand(N_oscillators))
oscillators.run(1E5, store_frequency=10, report_energies=False)