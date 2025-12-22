import numpy as np
from Backend.particle_system import Particles
from typing import Tuple, Optional
from Config.py import *

class Environment:

    def __init__(self):
<<<<<<< HEAD
        self._interactionmatrix: np.ndarray = np.array([[0, 1, 2, 3, 4],
                                                       [1, 1, -1, -1, 1],
                                                       [2, -1, 1, -1, 1],
                                                       [3, -1, -1, 1, 1],
                                                       [4, 1, 1, 1, -1]])
        self._particles: Particles = Particles()
        self._checked_particles: np.ndarray= np.zeros(self._particles.shape)

	@numba.jit(nopython = True)
    def check_interactions(self, position_x, position_y, radius, index) -> np.ndarray:
		#positionen aller Particles im Radius herausfinden
		maske_x = self._particles.x >= position_x & self._particles.x <= position_x + radius
		maske_y = self._particles.y >= position_y & self._paritcles.y <= position.y + radius
		maske_n = maske_x & maske_y
		maske_n[index] = False
		if sum(maske_n) == 0:
			return 0
		neighbours_x = self._particles.x[maske_n]
		neighbours_y = self._particles.y[maske_n]
		
		#typen der Benachbarten Particles herausfinden
		n_types: np.ndarray = self._particles.type[maske_n]
		interactions = np.array([self._particles.types[index, x] for x in n_types])
		
		indices = np.where(mask)
		return np.array([neighbours_x, neigbours_y, interactions, indices])

	@numba.jit(nopython = True)
	def calc_velocity(self, position_x: np.ndarray, position_y: np.ndarray, neigbours_x: np.ndarray, neighbours_y: np.ndarray, interactions: np.ndarray, index: int, indices: np.ndarray ) -> np.ndarray:
		new_x: np.zeros(neighbour_x.shape[0])
		new_y: np.zeros(neighbour_y.shape[0])
		r1: np.ndarray = np.array([position_x, poition_y])
		k: int = 1
		m1: int = 1
		m2: int = 1
		t: float = 0.01
		for i in range(neighbours_x.shape[0]):
			r2 = np.array([neighbours_x, neighbours_y ])
			r = r1 - r2
			r_abs = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2)
			r_norm: np.ndarray = r / r_abs
			f1: np.ndarray = k * (np.prod(self._interactionmatrix[interactions])/r**2) * r_norm
			f2: np.ndarray = f1 * -1

            #mit Reibungskraft verechnen
            f1 = f1 - gamma * self._particles.velocity_x[indices[i]]
            f2 = f2 - gamma * self._particles.velocity_y[indices[i]]

			a1: np.ndarray = f1 / m1
			a2: np.ndarray = f2 / m2
			self._particles.velocity_x[index] += a1[0] * t
			self._particles.velocity_y[index] += a1[1] * t
			self._particles.velocity_x[indices[i]] += a2[0] * t
			self._particles.velocity_y[indices[i]] += a2[1] * t
			
			#Position ändern
			self._particles.x[index] += self._particles.velocity_x[index] * t
            self._particles.y[index] += self._particles.velocity_x[index] * t
			self._particles.x[indices[i]] += self._particles.velocity_x[indices[i]]  * t
			self._particles.y[indices[i]] += self._particles.velocity_x[indices[i]]  * t

    
    def diffuse():
        pass
       

    def get_particles_x(self):
        return self.particles.x 

    def get_particles_y(self):
        return self.particles.y 
