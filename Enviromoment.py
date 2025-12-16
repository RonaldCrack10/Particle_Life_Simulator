import numpy as np
import numba
from Particles.py import Particles

class Environment:

    def __init__(self):
        self._interactionmatrix: np.ndarray = np.array([0, 1, 2, 3, 4],
                                                       [1, 1, -1, -1, 1],
                                                        [2, -1, 1, -1, 1],
                                                       [3, -1, -1, 1, 1],
                                                       [4, 1, 1, 1, -1])
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
		return np.array([neighbours_x, neigbours_y, interactions])

	@numba.jit(nopython = True)
	def calc_velocity(self, position_x, position_y, neigbours_x, neighbours_y) -> np.ndarray:
		pass
