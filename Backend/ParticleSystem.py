import numpy as np
from Config.config import NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT

class Particles:
    """Verwaltet alle Partikel als NumPy Arrays."""

    def __init__(self):
        self._x = np.random.rand(NUM_PARTICLES) * WIDTH
        self._y = np.random.rand(NUM_PARTICLES) * HEIGHT
        self._velocity_x = np.zeros(NUM_PARTICLES)
        self._velocity_y = np.zeros(NUM_PARTICLES)
        self._types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def velocity_x(self):
        return self._velocity_x

    @velocity_x.setter
    def velocity_x(self, value):
        self._velocity_x = value

    @property
    def velocity_y(self):
        return self._velocity_y

    @velocity_y.setter
    def velocity_y(self, value):
        self._velocity_y = value

    @property
    def types(self):
        return self._types

    @types.setter
    def types(self, value):
        self._types = value