import numpy as np
from Backend.particle_system import Particles
from typing import Tuple, Optional
from Config.config import FRICTION
import numba

class Environment:

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
       


    def calc_friction(self, velocity_x: np.ndarray, velocity_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate friction forces for all particles.

        Returns:
            Tuple of numpy arrays representing friction forces in x and y directions.

        Note: friction is returned with a sign that *opposes* the velocity.
        i.e. friction_x = -FRICTION * velocity_x
        """

        friction_x = -FRICTION * velocity_x
        friction_y = -FRICTION * velocity_y

        return friction_x, friction_y


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
    

    def diffuse(self):
        pass

        
