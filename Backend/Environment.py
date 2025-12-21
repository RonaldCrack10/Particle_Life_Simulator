import numpy as np
import numba
from Particles.py import Particles
from typing import Tuple, Optional



class Environment:

    def __init__(self):
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
            f1 = f1 - gamma * 

			a1: np.ndarray = f1 / m1
			a2: np.ndarray = f2 / m2
			self._particles.velocity_x[index] = self._particles.velocity_x[index] + a1[0] * t
			self._particles.velocity_y[index] = self._particles.velocity_y[index] + a1[1] * t
			self._particles.velocity_x[indices[i]] = self._particles.velocity_x[indices[i]] + a2[0] * t
			self._particles.velocity_y[indices[i]] = self._particles.velocity_y[indices[i]] + a2[1] * t
			
			#Position ändern
			self._particles.x[index] = self._particles.x[index] + self._particles.velocity_x[index] * t
            self._particles.y[index] = self._particles.y[index] + self._particles.velocity_x[index] * t
			self._particles.x[indices[i]] = self._particles.x[indices[i]] + self._particles.velocity_x[indices[i]]  * t
			self._particles.y[indices[i]] = self._particles.y[indices[i]] + self._particles.velocity_x[indices[i]]  * t

    
    def diffuse():
        pass


    def calc_force(
        self,
        index: int,
        neigh_idx,
        position: np.ndarray,
        type_idx,
        interaction: np.ndarray,
        velocity_x: Optional[np.ndarray] = None,
        velocity_y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        """

        # Ensure velocities are available; if not provided, default to zeros
        n_particles = position.shape[0]
        if velocity_x is None or velocity_y is None:
            velocity_x = np.zeros(n_particles) if velocity_x is None else velocity_x
            velocity_y = np.zeros(n_particles) if velocity_y is None else velocity_y

        # Calculate friction forces (signed to oppose motion)
        friction_x, friction_y = self.calc_friction(velocity_x, velocity_y)

        # Prepare output arrays (one entry per particle)
        Fx_total = np.zeros(n_particles, dtype=position.dtype)
        Fy_total = np.zeros(n_particles, dtype=position.dtype)

        # Convert neighbour list to array for masking
        neigh_arr = np.asarray(neigh_idx)
        # Mask out invalid neighbours (marked with -1)
        valid_mask = neigh_arr != -1
        if valid_mask.any():
            # Only iterate over valid neighbour indices
            neighbours = neigh_arr[valid_mask]
            px = position[index, 0]
            py = position[index, 1]
            t_i = type_idx[index]
            # Small epsilon to avoid division by zero
            eps = 1e-12
            for j in neighbours:
                dx = position[j, 0] - px
                dy = position[j, 1] - py
                distance = np.hypot(dx, dy)
                if distance > eps:
                    force_scalar = interaction[t_i, type_idx[j]]
                    Fx_total[index] += force_scalar * (dx / distance)
                    Fy_total[index] += force_scalar * (dy / distance)

        # Subtract (apply) friction for this particle
        Fx_total[index] += friction_x[index]
        Fy_total[index] += friction_y[index]

        return Fx_total, Fy_total
    

    def get_particles_x(self):
        return self.particles.x 

    def get_particles_y(self):
        return self.particles.y 
