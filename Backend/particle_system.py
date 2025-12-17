import numpy as np
from Config.config import NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT


class Particles:
    """Verwaltet alle Partikel als NumPy Arrays."""
    
    def __init__(self, num_particles: int = NUM_PARTICLES, num_types: int = NUM_TYPES) -> None:
        self._num_particles = num_particles
        self._num_types = num_types
        
        self._x = np.random.rand(num_particles) * WIDTH
        self._y = np.random.rand(num_particles) * HEIGHT
        self._velocity_x = np.zeros(num_particles)
        self._velocity_y = np.zeros(num_particles)
        self._types = np.random.randint(0, num_types, num_particles)
    
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