import numpy as np
import numba
from .Particles import Particles
from typing import Tuple, Optional
from Config.constants import *

class Simulation:

    def __init__(self,
        interactionmatrix: np.ndarray = np.array([
            [4, -2, -1, 2, 3],      # Type 0 (RED) interactions
            [1, 5, -1, -3, 1],      # Type 1 (GREEN) interactions
            [-1, 1, 1, -1, 1],      # Type 2 (BLUE) interactions
            [-1, -2, 1, 1, 6],      # Type 3 (YELLOW) interactions
            [1, 4, 1, 1, -5],       # Type 4 (MAGENTA) interactions
        ], dtype=float),
        particles: Particles = None):
        
        self._interactionmatrix = interactionmatrix
        self._particles = particles if particles is not None else Particles()
        self._particles_x = self._particles.x
        self._particles_y = self._particles.y
        
        num = self._particles.x.shape[0]
        self._checked_particles: np.ndarray = np.zeros(num, dtype=int)
        

    
    
    def check_interactions(self, position_x, position_y, radius, index) -> np.ndarray:
		#positionen aller Particles im Radius herausfinden
        
        sliced_x = self._particles_x[index + 1:] # Nur die Partikel nach dem aktuellen Index betrachten, um doppelte Berechnungen zu vermeiden
        sliced_y = self._particles_y[index + 1:]
        dx = sliced_x - position_x
        dy = sliced_y - position_y
        maske_n = (dx*dx + dy*dy) <= radius*radius
        
        
        if not np.any(maske_n):
            return None
        
        neighbours_x = sliced_x[maske_n]
        neighbours_y = sliced_y[maske_n]
		
		#typen der Benachbarten Particles herausfinden
        indices = np.where(maske_n)[0] + (index + 1) # Korrektur der Indizes, da wir nur die Partikel nach dem aktuellen Index betrachten
        type_i = self._particles.types[index]
        n_types: np.ndarray = self._particles.types[indices]
        interactions = self._interactionmatrix[type_i, n_types]
		
        
        return neighbours_x, neighbours_y, interactions, indices

    
    
    def calc_velocity(
        self,
        position_x: float,
        position_y: float,
        neighbours_x: np.ndarray,     # (N,)
        neighbours_y: np.ndarray,     # (N,)
        interactions: np.ndarray,     # (N,2) -> [current_type, neighbour_type]
        index: int,
        indices: np.ndarray,          # (N,) -> originale Nachbar-Indizes
        ):
        # Konstanten (bei dir ggf. als Attribute speichern)
        k: float = 3.0
        m1: float = 1.0
        m2: float = 1.0
        t: float = 1
        gamma: float = 0.001
        eps: float = 1e-12

        N:int = neighbours_x.shape[0]
        if N == 0:
            return

        # --- Geometrie (alles vektorisiert) ---
        # Vektoren vom aktuellen Partikel zu allen Nachbarn
        dx: np.ndarray = position_x - neighbours_x                # (N,)
        dy: np.ndarray = position_y - neighbours_y                # (N,)
        r2: np.ndarray = dx * dx + dy * dy + eps                  # (N,)   (Abstand^2)
        r_abs: np.ndarray = np.sqrt(r2)                           # (N,)

        # Einheitsrichtungen (N,2)
        r_hat: np.ndarray = np.column_stack((dx / r_abs, dy / r_abs))  # (N,2)

        # --- k_ij holen (vektorisiert) ---
        # kij: np.ndarray = self._interactionmatrix[interactions[:, 0], interactions[:, 1]]  # (N,)
        kij: np.ndarray = interactions  # (N,) - da interactions bereits die k_ij Werte enthält
        # --- Kräfte pro Nachbar (N,2) ---
        # inverse-square: 1/r^2 (hier r2 ist schon Abstand^2)
        F_pairs: np.ndarray = (k * kij / r2)[:, None] * r_hat     # (N,2)

        # Gesamtkraft auf das aktuelle Partikel (2,)
        F1: np.ndarray = F_pairs.sum(axis=0)

        # Reibung auf aktuelles Partikel
        v1: np.ndarray = np.array([self._particles.velocity_x[index], self._particles.velocity_y[index]], dtype=np.float64)
        F1 = F1 - gamma * v1

        # Beschleunigung + Update für aktuelles Partikel
        a1: np.ndarray = F1 / m1
        self._particles.velocity_x[index] += a1[0] * t
        self._particles.velocity_y[index] += a1[1] * t
        self._particles.x[index] += self._particles.velocity_x[index] * t
        self._particles.y[index] += self._particles.velocity_y[index] * t

        # --- Gegenkräfte auf Nachbarn (ohne Loop) ---
        # Newton III: Nachbar bekommt -F_pair
        F2_pairs: np.ndarray = -F_pairs                            # (N,2)

        # Reibung auf Nachbarn (N,2)
        v2: np.ndarray = np.column_stack((
            self._particles.velocity_x[indices],
            self._particles.velocity_y[indices],
        )).astype(np.float64)                           # (N,2)
        F2_pairs = F2_pairs - gamma * v2

        # Beschleunigung + Update für Nachbarn (vektorisiert)
        a2: np.ndarray = F2_pairs / m2                              # (N,2)
        self._particles.velocity_x[indices] += a2[:, 0] * t
        self._particles.velocity_y[indices] += a2[:, 1] * t
        self._particles.x[indices] += self._particles.velocity_x[indices] * t
        self._particles.y[indices] += self._particles.velocity_y[indices] * t
                
    def diffuse(self):
        for i in range(self._particles.x.shape[0]):
            check = self.check_interactions(self._particles.x[i], self._particles.y[i], self._particles._radius, i)
            if check is not None:
                
                neighbours_x, neighbours_y, interactions, indices = check
                self.calc_velocity(self._particles.x[i], self._particles.y[i],neighbours_x, neighbours_y, interactions, i, indices)
            t = 1
            self._particles.x += self._particles.velocity_x * t
            self._particles.y += self._particles.velocity_y * t
            
            return self._particles.x, self._particles.y