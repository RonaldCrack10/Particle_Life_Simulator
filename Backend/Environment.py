import numpy as np
from Backend.particle_system import Particles
from typing import Tuple, Optional
from Config.constants import FRICTION
import numba

class Simulation:

    def __init__(self):

        # interaction matrix should be a 2D array (rows = types, cols = types)
        # Example: 5 types -> a 5x5 matrix. Use a list of lists to construct.
        self._interactionmatrix: np.ndarray = np.array([
            [0, 1, 2, 3, 4],
            [1, 1, -1, -1, 1],
            [2, -1, 1, -1, 1],
            [3, -1, -1, 1, 1],
            [4, 1, 1, 1, -1],
        ], dtype=float)

        # Create a Particles instance. If Particles expects initialization args,
        # the caller should pass them or modify this accordingly.
        self._particles: Particles = Particles()
        # default boolean mask sized to number of particles if available
        try:
            n = self._particles.x.shape[0]
        except Exception:
            n = 0
        self._checked_particles: np.ndarray = np.zeros(n, dtype=bool)


    @numba.jit(nopython = True)
    def check_interactions(self, 
                           position_x, 
                           position_y, 
                           radius, 
                           index) -> np.ndarray:
        
        # positionen aller Particles im Radius herausfinden
        maske_x = self._particles.x >= position_x & self._particles.x <= position_x + radius
        maske_y = self._particles.y >= position_y & self._particles.y <= position_y + radius
        maske_n = maske_x & maske_y
        maske_n[index] = False
        if sum(maske_n) == 0:
            return 0
        neighbours_x = self._particles.x[maske_n]
        neighbours_y = self._particles.y[maske_n]
        
        # typen der Benachbarten Particles herausfinden
        n_types: np.ndarray = self._particles.type[maske_n]
        interactions = np.array([self._particles.types[index, x] for x in n_types])
        return np.array([neighbours_x, neighbours_y, interactions])

    @numba.jit(nopython = True)
    def calc_velocity(self, position_x, position_y, neigbours_x, neighbours_y) -> np.ndarray:
        pass
       



    def diffuse(
        self,
    ):
        """
         Compute total forces (Fx_total, Fy_total) for particles.

        Parameters
        - index: index of the particle for which forces are computed
        - neigh_idx: iterable of neighbour indices (use -1 to indicate missing)
        - position: Nx2 array of particle positions
        - type_idx: array of length N with integer type indices
        - interaction: matrix mapping (type_i, type_j) -> interaction scalar
        - velocity_x, velocity_y: optional velocity arrays; if omitted, zeros are used

        Returns:
            (Fx_total, Fy_total) arrays with length N. Only the entry at `index`
            is modified in this function (consistent with prior behavior).

            schleife über for i in range(particle) array check_interactions für alle partikel und calc_velocity
            position_x = self._particles.x[i]
            position_y = self._particles.y[i]
            check_interactions(position_x, position_y) gibt nachbarn zurück
        """
        # + diffuse
        pass
    

        
