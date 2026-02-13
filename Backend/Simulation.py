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
        num = self._particles.x.shape[0]
        self._checked_particles: np.ndarray= np.zeros(num, dtype = int)
        self._particles = particles
        self._particles_x = particles.x
        self._particles_y = particles.y

    
    
    def check_interactions(self, position_x, position_y, radius, index) -> np.ndarray:
		#positionen aller Particles im Radius herausfinden
        
        sliced_x = self._particles_x[index + 1:] # Nur die Partikel nach dem aktuellen Index betrachten, um doppelte Berechnungen zu vermeiden
        sliced_y = self._particles_y[index + 1:]
        dx = sliced_x - position_x
        dy = sliced_y - position_y
        maske_n = (dx*dx + dy*dy) <= radius*radius
        
        
        if sum(maske_n) == 0:
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int),
                )
        neighbours_x = sliced_x[maske_n]
        neighbours_y = sliced_y[maske_n]
		
		#typen der Benachbarten Particles herausfinden
        indices = np.where(maske_n)[0] + (index + 1) # Korrektur der Indizes, da wir nur die Partikel nach dem aktuellen Index betrachten
        type_i = self._particles.types[index]
        n_types: np.ndarray = self._particles.types[indices]
        interactions = self._interactionmatrix[type_i, n_types]
		
        
        return neighbours_x, neighbours_y, interactions, indices

    
    
    def calc_velocity(self, position_x: np.ndarray, position_y: np.ndarray, neighbours_x: np.ndarray, neighbours_y: np.ndarray, interactions: np.ndarray, index: int, indices: np.ndarray ):
        
        # Konstanten für bessere Sichtbarkeit
        k: float = 20.0     # Kraft-Skalierung 
        t: float = 0.05     # Zeitschritt 
        min_dist: float = 10.0 # Radius
        m1 = m2 = 1.0

        for i in range(neighbours_x.shape[0]):
            j = indices[i]
            
            # 1. Abstandsvektor berechnen 
            dx = position_x - neighbours_x[i]
            dy = position_y - neighbours_y[i]
            r_abs = np.sqrt(dx**2 + dy**2)

            if r_abs < 1e-5:
                continue

            # Kraft berechnen
            if r_abs < min_dist:
                # UNABHÄNGIG von der Matrix: Harte Abstoßung im Nahbereich
                # Wir nutzen eine negative Kraft, damit sie sich wegdrücken
                force_mag = (r_abs / min_dist - 1) * k * 5.0
            else:
                # Normale Interaktion laut Matrix
                
                force_mag = k * interactions[i] * (1 - r_abs / PARTICLE_RADIUS)

            # Kraft auf die Achsen verteilen (nx = dx/r_abs) nx = Einheitsvektor in Richtung der Kraft
            fx = force_mag * (dx / r_abs)
            fy = force_mag * (dy / r_abs)

            # Geschwindigkeit anpassen (Newton 3: Action = Reaction)
            # Partikel 1 (index) bekommt die Kraft direkt
            self._particles.velocity_x[index] += (fx / m1) * t
            self._particles.velocity_y[index] += (fy / m1) * t
            # Partikel 2 (j) bekommt die exakt entgegengesetzte Kraft
            self._particles.velocity_x[j] -= (fx / m2) * t
            self._particles.velocity_y[j] -= (fy / m2) * t

        # 5. Positionen updaten 
        
        self._particles.x[index] += self._particles.velocity_x[index] * t
        self._particles.y[index] += self._particles.velocity_y[index] * t
        
        for n in range(neighbours_x.shape[0]):
            idx_j = indices[n]
            self._particles.x[idx_j] += self._particles.velocity_x[idx_j] * t
            self._particles.y[idx_j] += self._particles.velocity_y[idx_j] * t
           
    def diffuse(self):
	
		
        n = self._particles.num_particles
		# Für jedes Partikel die Nachbarn prüfen und Geschwindigkeit berechnen
        for i in range(n):
            
            neighbours_x, neighbours_y, interactions, indices = self.check_interactions(self._particles.x[i],
                                                                                         self._particles.y[i],
                                                                                           PARTICLE_RADIUS,
                                                                                             i)
           
            if indices.shape[0] > 0:
                self.calc_velocity(self._particles.x[i],
                                    self._particles.y[i],
                                    neighbours_x, 
                                    neighbours_y, 
                                    interactions, 
                                    i, 
                                    indices)
            
            # Reibung anwenden und Positionen updaten 
        
        self._particles.velocity_x *= FRICTION
        self._particles.velocity_y *= FRICTION

       
        t = 0.5 # Zeitschritt für die Bewegung 
        self._particles.x += self._particles.velocity_x * t
        self._particles.y += self._particles.velocity_y * t
			


    def get_particles_x(self):
        return self._particles.x 

    def get_particles_y(self):
        return self._particles.y 
