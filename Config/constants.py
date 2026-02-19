"""
Constants for Particle Life Simulator
"""
import numpy as np

# Window size
WIDTH, HEIGHT = 800, 800

# Particle configuration
NUM_TYPES = 5
NUM_PARTICLES = 2000  # For testing (later: 5000)

# Physics parameters
FRICTION = 0.95  # Friction coefficient (0-1, where 1 = no friction)
INTERACTION_RADIUS = 100  # Radius for particle interactions
PARTICLE_RADIUS = 5  # Radius of visual particle representation

# Interaction matrix (5x5 for 5 particle types)
# Each row i represents interactions of type i with types 0-4
# Positive values = attraction, Negative values = repulsion
INTERACTION_MATRIX = np.array([
    [1, -1, -1, 1, 1],      # Type 0 (RED) interactions
    [1, 1, -1, -1, 1],      # Type 1 (GREEN) interactions
    [-1, 1, 1, -1, 1],      # Type 2 (BLUE) interactions
    [-1, -1, 1, 1, 1],      # Type 3 (YELLOW) interactions
    [1, 1, 1, 1, -1],       # Type 4 (MAGENTA) interactions
], dtype=float)

# Colors for VisPy (0.0-1.0 instead of 0-255)
COLORS_VISPY = np.array([
    [1.0, 0.2, 0.2, 1.0],   # RED
    [0.2, 1.0, 0.2, 1.0],   # GREEN
    [0.2, 0.2, 1.0, 1.0],   # BLUE
    [1.0, 1.0, 0.2, 1.0],   # YELLOW
    [1.0, 0.2, 1.0, 1.0]  # MAGENTA 
])
