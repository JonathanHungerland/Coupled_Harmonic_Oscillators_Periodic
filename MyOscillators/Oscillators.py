import numpy as np
from MyOscillators.animation import create_animation
from typing import Union

class CoupledOscillators:
    # Class attributes for the 4th order symplectic integrator constants
    c1 = 0.5153528374311229364
    c2 = -0.085782019412973646
    c3 = 0.4415830236164665242
    d1 = 0.1344961992774310892
    d2 = 0.2927356636812486763
    d3 = 0.6233367090322121420

    def __init__(self,name:str, N_oscillators:int, k: Union[float,np.ndarray], c:float, dt:float=None, verbose:bool=True):
        self.name = name
        self.N_oscillators = N_oscillators
        self.k = k
        self.fastest_period = np.pi/np.sqrt(np.max(self.k))
        self.slowest_period = np.pi/np.sqrt(np.min(self.k))
        self.c = c
        if dt is not None:
            self.dt = dt
        else:
            #1/10 T needed, use 1/100 T for stability
            self.dt = self.fastest_period/100
            if verbose:
                print("Fastest period of %.2Es dictates a timestep of at least %.2Es. Will use %.2Es."%(self.fastest_period, self.fastest_period/10, self.dt),flush=True)
        self.verbose = verbose
        self.positions = np.ones(N_oscillators,dtype=np.longdouble)
        self.old_positions = None
        self.velocities = np.zeros(N_oscillators,dtype=np.longdouble)
        self.accelerations = np.zeros(N_oscillators,dtype=np.longdouble)
        self.old_accelerations = None
        self.position_trajectory = np.array([])
        self.velocity_trajectory = np.array([])
        self.energy_trajectory = np.array([])

    def initialize(self,positions):
        self.positions = positions

    def randomize_positions(self, target_energy=None, addition_increment=0.1, seed=None):
        if target_energy is None:
            target_energy = self.N_oscillators
        rng1 = np.random.default_rng(seed=seed)
        rng2 = np.random.default_rng(seed=seed)
        while self.get_energy() < target_energy:
            self.positions[rng1.integers(0,self.N_oscillators)]+=rng2.uniform(-1,1)*addition_increment

    def calculate_accelerations(self):
        self.accelerations = - self.k*self.positions - 0.5*self.c*(2*self.positions-np.roll(self.positions,1)-np.roll(self.positions,-1))

    def calculate_velocities(self):
        if self.old_positions is None:
            return np.zeros(self.N_oscillators)
        self.calculate_accelerations()
        new_positions = 2*self.positions - self.old_positions + self.accelerations*self.dt**2
        return (new_positions - self.old_positions)/(2*self.dt)

    def initial_step(self):
        self.calculate_accelerations()
        self.old_positions = self.positions.copy()
        self.positions += self.velocities * self.dt + 0.5*self.accelerations*self.dt**2

    def basic_verlet_step(self, initial=False):
        if initial:
            self.calculate_accelerations()
            self.old_positions = self.positions.copy()
            self.velocities = np.zeros(self.N_oscillators)
            self.positions += self.velocities * self.dt + 0.5*self.accelerations*self.dt**2
            return
        self.calculate_accelerations()
        new_positions = 2*self.positions - self.old_positions + self.accelerations*self.dt**2
        self.old_positions = self.positions.copy()
        self.positions = new_positions

    def velocity_verlet_step(self, initial=False):      
        if initial:
            self.calculate_accelerations()
            self.velocities=np.zeros(self.N_oscillators)
        self.positions += self.velocities*self.dt+0.5*self.accelerations*self.dt**2
        self.velocities+=0.5*self.accelerations*self.dt
        self.calculate_accelerations()
        self.velocities+=0.5*self.accelerations*self.dt

    def symplectic_4th_order_step(self, initial=False):
        if initial:
            self.velocities = np.zeros(self.N_oscillators)
        
        #First step
        self.calculate_accelerations()
        self.velocities += 0.5*self.d1*self.accelerations*self.dt
        self.positions += self.c1*self.dt*self.velocities
        self.calculate_accelerations()
        self.velocities += 0.5*self.d1*self.accelerations*self.dt
        #Second step
        self.calculate_accelerations()
        self.velocities += 0.5*self.d2*self.accelerations*self.dt
        self.positions += self.c2*self.dt*self.velocities
        self.calculate_accelerations()
        self.velocities += 0.5*self.d2*self.accelerations*self.dt
        #Third step
        self.calculate_accelerations()
        self.velocities += 0.5*self.d3*self.accelerations*self.dt
        self.positions += self.c3*self.dt*self.velocities
        self.calculate_accelerations()
        self.velocities += 0.5*self.d3*self.accelerations*self.dt

    def take_step(self, step_method, initial=False):
        step_method(initial=initial)

    def run(self, steps=None, periods=None, store_frequency=100, store_per_period=5,method='basic-verlet', store_on_disk=True):
        if periods is None and steps is None:
            raise ValueError("Must specify either steps or periods.")
        if periods is not None:
            steps = int(periods*self.slowest_period/self.dt)
            store_frequency = int(self.fastest_period/self.dt/store_per_period)
            if self.verbose:
                print("Requested %d periods of slowest oscillation. Will simulate for %.2E steps."%(periods,steps),flush=True)
                print("Will store %d times per fastest period, resuling in %d store_frequency."%(store_per_period,store_frequency),flush=True)
            steps = steps - (steps % store_frequency)

        match method:
            case 'basic-verlet':
                step_method=self.basic_verlet_step
            case 'velocity-verlet':
                step_method=self.velocity_verlet_step
            case 'symplectic-4th-order':
                step_method=self.symplectic_4th_order_step

        array_length=int(steps/store_frequency)
        new_trajectory = np.zeros((array_length,self.N_oscillators))
        new_velocity_trajectory = np.zeros((array_length,self.N_oscillators))
        new_energy_trajectory = np.zeros(array_length)

        if self.position_trajectory.size == 0:
            self.take_step(step_method=step_method,initial=True)
            start=1
        else:
            start=0

        for i in range(int(start),int(steps)):
            self.take_step(step_method=step_method)
            if i % store_frequency == 0:
                new_trajectory[i//store_frequency] = self.positions.astype(np.float64)
                new_velocity_trajectory[i//store_frequency] = self.get_velocities().astype(np.float64)
                new_energy_trajectory[i//store_frequency] = self.get_energy()

        if self.position_trajectory.size == 0:
            self.position_trajectory = new_trajectory
            self.velocity_trajectory = new_velocity_trajectory
            self.energy_trajectory = new_energy_trajectory
        else:
            self.position_trajectory = np.vstack((self.position_trajectory, new_trajectory))
            self.velocity_trajectory = np.vstack((self.velocity_trajectory, new_velocity_trajectory))
            self.energy_trajectory = np.append(self.energy_trajectory, new_energy_trajectory)
        if store_on_disk:
            np.save(self.name+'_energy_trajectory.npy', self.energy_trajectory)
            np.save(self.name+'_trajectory.npy', self.position_trajectory)
            np.save(self.name+'_velocity_trajectory.npy', self.velocity_trajectory)
    
    def get_energy(self):
        energy =0
        energy += np.sum(self.k*np.power(self.positions,2))
        #only in one direction to avoid double counting
        energy += self.c*np.sum(np.power(self.positions-np.roll(self.positions,1),2))
        energy += np.sum(np.power(self.get_velocities(),2))
        energy *= 0.5
        return energy

    def get_positions(self):
        return self.positions

    def get_velocities(self):
        if self.velocities is not None:
            return self.velocities
        return self.calculate_velocities()
    
    def get_position_trajectory(self):
        return self.position_trajectory
    
    def get_velocity_trajectory(self):
        return self.velocity_trajectory

    def get_energy_trajectory(self):
        return self.energy_trajectory
    


    


        
            




        
