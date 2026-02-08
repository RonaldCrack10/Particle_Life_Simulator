import numpy as np
import numba
from .particle_system import Particles
from typing import Tuple, Optional
from Config.config import *

class Environment:

    def __init__(self):
        self._interactionmatrix: np.ndarray = np.array([[0, 1, 2, 3, 4],
                                                       [1, 1, -1, -1, 1],
                                                       [2, -1, 1, -1, 1],
                                                       [3, -1, -1, 1, 1],
                                                       [4, 1, 1, 1, -1]])
        self._particles: Particles = Particles()
        particles = Particles()
        self._checked_particles: np.ndarray= np.zeros((self._particles.shape()), dtype = int)
        self._particles = particles
        self._particles_x = particles.x
        self._particles_y = particles.y

    
    
    def check_interactions(self, position_x, position_y, radius, index) -> np.ndarray:
		#positionen aller Particles im Radius herausfinden
        dx = self._particles_x - position_x
        dy = self._particles_y - position_y
        maske_n = dx*dx + dy*dy <= radius*radius
        maske_n[index] = False

        # maske_x = (self._particles_x >= position_x) & (self._particles_x <= position_x + radius)
        # maske_y = (self._particles_y >= position_y) & (self._particles_y <= position_y + radius)
        # maske_n = maske_x & maske_y
        # maske_n[index] = False
        
        if sum(maske_n) == 0:
            return  (
        np.empty(0),
        np.empty(0),
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
    )
        neighbours_x = self._particles_x[maske_n]
        neighbours_y = self._particles_y[maske_n]
		
		#typen der Benachbarten Particles herausfinden
        type_i = self._particles.types[index]
        n_types: np.ndarray = self._particles.types[maske_n]
        interactions = np.array([self._interactionmatrix[type_i, t] for t in n_types])
		
        indices = np.where(maske_n)[0]
        return neighbours_x, neighbours_y, interactions, indices

    
    
    def calc_velocity(self, position_x: np.ndarray, position_y: np.ndarray, neighbours_x: np.ndarray, neighbours_y: np.ndarray, interactions: np.ndarray, index: int, indices: np.ndarray ):
        new_x: np.array = np.zeros(neighbours_x.shape[0])
        new_y: np.array =  np.zeros(neighbours_y.shape[0])
        r1: np.ndarray = np.array([position_x, position_y])
        k: int = 1
        m1: int = 1
        m2: int = 1
        t: float = 0.01
        for i in range(neighbours_x.shape[0]):
            r2 = np.array([neighbours_x[i], neighbours_y[i]])
            r = r1 - r2
            r_abs = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2)
            r_norm: np.ndarray = r / r_abs
            # f1: np.ndarray = k * (np.prod(self._interactionmatrix[interactions])/r**2) * r_norm
            # f2: np.ndarray = f1 * -1
            # type_i = self._particles.types[index]
            # type_j = self._particles.types[indices[i]]

            #interaction = self._interactionmatrix[type_i, type_j]
            force_mag = k * interactions[i] / (r_abs * r_abs)
            if r_abs < 1e-5:  # Vermeidung von Singularität
                continue

            f1 = force_mag * r_norm
            f2 = -f1


            #mit Reibungskraft verechnen
            # f1 = f1 - FRICTION * self._particles.velocity_x[indices[i]] # Reibungskraft gamma  = Friction
            # f2 = f2 - FRICTION * self._particles.velocity_y[indices[i]]

            a1: np.ndarray = f1 / m1
            a2: np.ndarray = f2 / m2
            self._particles.velocity_x[index] += a1[0] * t
            self._particles.velocity_y[index] += a1[1] * t
            self._particles.velocity_x[indices[i]] += a2[0] * t
            self._particles.velocity_y[indices[i]] += a2[1] * t
			
			#Position ändern
        self._particles.x[index] += self._particles.velocity_x[index] * t
        self._particles.y[index] += self._particles.velocity_y[index] * t
        for n in range(neighbours_x.shape[0]):
            j = indices[n]
            self._particles.x[j] += self._particles.velocity_x[j] * t
            self._particles.y[j] += self._particles.velocity_y[j] * t
    
    def diffuse(self):
	
		# Startwerte für die Schleife
        i = 0
		# Für jedes Partikel die Nachbarn prüfen und Geschwindigkeit berechnen
        for i in range(NUM_PARTICLES):
            neighbours_x, neighbours_y, interactions, indices = self.check_interactions(self._particles.x[i], self._particles.y[i], PARTICLE_RADIUS, i)
           
            if indices.shape[0] > 0:
                self.calc_velocity(self._particles.x[i], self._particles.y[i], neighbours_x, neighbours_y, interactions, i, indices)
            else:
				# No interactions, just update position
                t = 0.01
                self._particles.x[i] += self._particles.velocity_x[i] * t
                self._particles.y[i] += self._particles.velocity_y[i] * t

			


    def get_particles_x(self):
        return self._particles.x 

    def get_particles_y(self):
        return self._particles.y 
