import numpy as np
from Config.constants import NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT


class Particles:
    def __init__(self, 
                 x: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=1000),
                 y: np.ndarray = np.random.normal(loc=0.0, scale=10.0, size=1000), 
                 velocity_x: np.ndarray = np.zeros(1000), 
                 velocity_y: np.ndarray = np.zeros(1000),  
                 types: np.ndarray = np.clip(np.rint(np.random.normal(loc=0, scale=1.0, size=1000)), 0, 4).astype(int),
                 radius: int = 15):
            # keine Parameter
        self._x = x
        self._y = y
        self._velocity_x = velocity_x
        self._velocity_y = velocity_y
        self._types = types 
        self._radius = radius
        

    def shape(self) -> tuple[int]:
        return self._x.shape

    # x - Getter gibt Array zurück, Setter gibt nichts zurück
    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._x = value

    # y
    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        self._y = value

    # velocity_x
    @property
    def velocity_x(self) -> np.ndarray:
        return self._velocity_x

    @velocity_x.setter
    def velocity_x(self, value: np.ndarray) -> None:
        self._velocity_x = value

    # velocity_y
    @property
    def velocity_y(self) -> np.ndarray:
        return self._velocity_y

    @velocity_y.setter
    def velocity_y(self, value: np.ndarray) -> None:
        self._velocity_y = value

    # types
    @property
    def types(self) -> np.ndarray:
        return self._types

    @types.setter
    def types(self, value: np.ndarray) -> None:
        self._types = value

    # Read-only
    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def num_types(self) -> int:
        return self._num_types