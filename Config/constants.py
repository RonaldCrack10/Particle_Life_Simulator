"""
Constants for Particle Life Simulator
"""
import numpy as np

# Window size
WIDTH, HEIGHT = 800, 800

# Particle configuration
NUM_TYPES = 4
NUM_PARTICLES = 2000  # For testing (later: 5000)

# Physics parameters
FRICTION = 0.95  # Friction coefficient (0-1, where 1 = no friction)
INTERACTION_RADIUS = 100  # Radius for particle interactions
PARTICLE_RADIUS = 5  # Radius of visual particle representation

# Interaction matrix (5x5 for 5 particle types)
INTERACTION_MATRIX = np.array([
    [0, 1, 2, 3, 4],
    [1, 1, -1, -1, 1],
    [2, -1, 1, -1, 1],
    [3, -1, -1, 1, 1],
    [4, 1, 1, 1, -1],
], dtype=float)

# Colors for VisPy (0.0-1.0 instead of 0-255)
COLORS_VISPY = np.array([
    [1.0, 0.2, 0.2, 1.0],   # RED
    [0.2, 1.0, 0.2, 1.0],   # GREEN
    [0.2, 0.2, 1.0, 1.0],   # BLUE
    [1.0, 1.0, 0.2, 1.0],   # YELLOW
    [1.0, 0.2, 1.0, 1.0]  # MAGENTA 
])
