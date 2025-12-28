import numpy as np
from typing import Tuple, Optional
from Backend.particle_system import Particles
from Config.config import (
    NUM_PARTICLES, FRICTION, INTERACTION_RADIUS,
    PARTICLE_RADIUS, WIDTH, HEIGHT, INTERACTION_MATRIX
)


class Simulation:
    """Berechnet Interaktionen zwischen Partikeln"""

    def __init__(self):
        self._interactionmatrix: np.ndarray = INTERACTION_MATRIX
        self._particles: Particles = Particles()
        self._checked_particles: np.ndarray = np.zeros(NUM_PARTICLES, dtype=bool)

    @property
    def particles(self) -> Particles:
        """Zugriff auf Partikel-Objekt"""
        return self._particles

    def check_interactions(self, position_x, position_y, radius, index) -> Optional[Tuple]:
        """Findet Nachbar-Partikel im Radius"""
        maske_x = (self._particles.x >= position_x - radius) & (self._particles.x <= position_x + radius)
        maske_y = (self._particles.y >= position_y - radius) & (self._particles.y <= position_y + radius)
        maske_n = maske_x & maske_y
        maske_n[index] = False

        if np.sum(maske_n) == 0:
            return None

        neighbours_x = self._particles.x[maske_n]
        neighbours_y = self._particles.y[maske_n]
        n_types = self._particles.types[maske_n]
        my_type = self._particles.types[index]
        interactions = self._interactionmatrix[my_type, n_types]

        return (neighbours_x, neighbours_y, interactions, maske_n)

    def calc_friction(self, velocity_x: np.ndarray, velocity_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Berechnet Reibungskraft"""
        friction_x = -FRICTION * velocity_x
        friction_y = -FRICTION * velocity_y
        return friction_x, friction_y

    def calc_force(self, index: int) -> Tuple[float, float]:
        """Berechnet Kraft für ein Partikel"""
        px = self._particles.x[index]
        py = self._particles.y[index]

        result = self.check_interactions(px, py, INTERACTION_RADIUS, index)

        if result is None:
            return 0.0, 0.0

        neighbours_x, neighbours_y, interactions, mask = result

        dx = neighbours_x - px
        dy = neighbours_y - py
        distances = np.sqrt(dx.astype(float)**2 + dy.astype(float)**2)

        eps = 1e-12
        unit_x = dx / (distances + eps)
        unit_y = dy / (distances + eps)

        MIN_DIST = 2 * PARTICLE_RADIUS
        repulsion = np.where(distances < MIN_DIST, (MIN_DIST - distances)**2 * 10.0, 0.0)

        force_magnitude = interactions / (distances + eps)

        fx = -np.sum(force_magnitude * unit_x) - np.sum(repulsion * unit_x)
        fy = -np.sum(force_magnitude * unit_y) - np.sum(repulsion * unit_y)

        return fx, fy

    def step(self, dt: float = 0.01) -> None:
        """Führt einen Simulationsschritt durch."""
        n = NUM_PARTICLES

        force_x = np.zeros(n)
        force_y = np.zeros(n)

        for i in range(n):
            fx, fy = self.calc_force(i)
            force_x[i] = fx
            force_y[i] = fy

        self._particles.velocity_x = self._particles.velocity_x + force_x * dt
        self._particles.velocity_y = self._particles.velocity_y + force_y * dt

        self._particles.velocity_x *= FRICTION
        self._particles.velocity_y *= FRICTION

        self._particles.x = self._particles.x + self._particles.velocity_x * dt
        self._particles.y = self._particles.y + self._particles.velocity_y * dt

        self._particles.x = self._particles.x % WIDTH
        self._particles.y = self._particles.y % HEIGHT

    def diffuse(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper für step()"""
        self.step()
        return self._particles.x, self._particles.y