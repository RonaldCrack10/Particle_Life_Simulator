import numpy as np

"""
Config - Parameter für Particle Life Simulator
"""

# Fenstergröße
WIDTH, HEIGHT = 800, 800

# Simulationsparameter
NUM_PARTICLES = 100  # Für Test weniger (später: 5000)
PARTICLE_RADIUS = 3
FRICTION = 0.95
MAX_SPEED = 5
INTERACTION_RADIUS = 100
ATTRACTION_STRENGTH = 0.05
RANDOM_FORCE_STRENGTH = 0.5

# Farben und Partikeltypen
COLORS = {
    'RED': (255, 60, 60),
    'GREEN': (60, 255, 60),
    'BLUE': (60, 60, 255),
    'YELLOW': (255, 255, 60)
}
COLOR_LIST = list(COLORS.keys())
COLOR_VALUES = np.array(list(COLORS.values()))
NUM_TYPES = len(COLOR_LIST)

# Farben für VisPy (0.0-1.0 statt 0-255)
COLORS_VISPY = np.array([
    [1.0, 0.2, 0.2, 1.0],   # RED
    [0.2, 1.0, 0.2, 1.0],   # GREEN
    [0.2, 0.2, 1.0, 1.0],   # BLUE
    [1.0, 1.0, 0.2, 1.0],   # YELLOW
])

# INTERACTION_MATRIX
# Positive Werte = Abstoßung, Negative Werte = Anziehung
INTERACTION_MATRIX = np.array([
    [+1.0, -1.0, +0.5, -0.5],   # RED
    [-1.0, +1.0, -0.5, +0.5],   # GREEN
    [+0.5, -0.5, +1.0, -1.0],   # BLUE
    [-0.5, +0.5, -1.0, +1.0],   # YELLOW
])