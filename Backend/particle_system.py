
from config import NUM_PARTICLES
import numpy as np

def generate_particle() -> None:
    """
    Generates a specified number of particles of a given type.

    Args:
        number_of_particles (int): The number of particles to generate.
        particle_type (str): The type of particles to generate (e.g., 'electron', 'proton').

    Returns:
        None
    """
    position = np.random.rand(NUM_PARTICLES, 2)
    velocity = np.zeros(NUM_PARTICLES, 2) # Initial velocity is zero
    types = np.random.randint(0, 4, NUM_PARTICLES) # Randomly assign one of 4 types
    return position, velocity, types  