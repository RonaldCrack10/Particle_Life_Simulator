import numpy as np

# Fenstergröße
WIDTH, HEIGHT = 800, 800

# Simulationsparameter
NUM_PARTICLES = 5000 
PARTICLE_RADIUS = 3
FRICTION = 0.95  # Dämpfung der Geschwindigkeit (0.0 bis 1.0)
MAX_SPEED = 5
INTERACTION_RADIUS = 100
ATTRACTION_STRENGTH = 0.05
RANDOM_FORCE_STRENGTH = 0.5 # Stärke der zufälligen Anfangsbewegung

# Farben und Partikeltypen
COLORS = {
    'RED': (255, 60, 60),
    'GREEN': (60, 255, 60),
    'BLUE': (60, 60, 255),
    'YELLOW': (255, 255, 60)
}
COLOR_LIST = list(COLORS.keys())

# NumPy-Array der Farbwerte für schnellen Zugriff
COLOR_VALUES = np.array(list(COLORS.values())) 
NUM_TYPES = len(COLOR_LIST)