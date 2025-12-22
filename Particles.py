import numpy as np
from Config.config import NUM_PARTICLES, NUM_TYPES, WIDTH, HEIGHT

class Particles:
    """
    Verwaltet alle Partikel als NumPy Arrays.
    
    """
    """
        Erstellt ein neues Partikelsystem.
        
        Args:
            num_particles: Anzahl der Partikel (default aus config.py)
            num_types: Anzahl der Typen, z.B. 4 für Rot/Grün/Blau/Gelb
    """

  
    """
    Verwaltet alle Partikel als NumPy Arrays, ergänzt um @property
    """
    
    def __init__(self):
            # keine Parameter
            self.x = np.random.rand(NUM_PARTICLES) * WIDTH
            self.y = np.random.rand(NUM_PARTICLES) * HEIGHT
            self.velocity_x = np.zeros(NUM_PARTICLES)
            self.velocity_y = np.zeros(NUM_PARTICLES)
            self.types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES)
        
            # mit property
            @property
            def x(self) -> np.ndarray: 
                return self._x
            @x.setter
            def x(self, value: np.ndarray) -> None: 
                self._x = value
            
            @property
            def y(self) -> np.ndarray: 
                return self._y
            @y.setter
            def y(self, value: np.ndarray) -> None: 
                self._y = value
        
            @property
            def velocity_x(self) -> np.ndarray: 
                return self._velocity_x
            @velocity_x.setter
            def velocity_x(self, value: np.ndarray) -> None: 
                self._velocity_x = value
            
            @property
            def velocity_y(self) -> np.ndarray: 
                return self._velocity_y
            @velocity_y.setter
            def velocity_y(self, value: np.ndarray) -> None: 
                self._velocity_y = value
            
            @property
            def types(self) -> np.ndarray: 
                return self._types
            @types.setter
            def types(self, value: np.ndarray) -> None: 
                self._types = value