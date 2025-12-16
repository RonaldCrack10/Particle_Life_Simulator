import numpy as np
from typing import Tuple, Optional
from Config.config import FRICTION


class Environment:

    def __init__(self):
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

        