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

    def __init__(self):
        
        # Position: Zufällig im Fenster verteilt (0 bis WIDTH/HEIGHT)
        self.x = np.random.rand(NUM_PARTICLES) * WIDTH
        self.y = np.random.rand(NUM_PARTICLES) * HEIGHT
        
        # Geschwindigkeit: Startet bei 0 für alle Partikel
        self.velocity_x = np.zeros(NUM_PARTICLES)
        self.velocity_y = np.zeros(NUM_PARTICLES)
        
        # Typ: Zufällig zwischen 0 und num_types-1 (bestimmt Farbe & Verhalten)
        self.types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES)
        