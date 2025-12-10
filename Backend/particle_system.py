
from Config.config import NUM_PARTICLES
import numpy as np

def generate_particle() -> None:
    """
    Generates a specified number of particles of a given type.

    Args:
        number_of_particles (int): The number of particles to generate.
        particle_type (str): The type of particles to generate (e.g., 'RED', 'GREEN', etc.).

    Returns:
        None
    """
    position = np.random.rand(NUM_PARTICLES, 2)
    velocity = np.zeros(NUM_PARTICLES, 2) # Initial velocity is zero
    types = np.random.randint(0, 4, NUM_PARTICLES) # Randomly assign one of 4 types
    return position, velocity, types  


def update_velocity():
    """
    Updates the velocity of particles based on their interactions.

    Returns:
        None
    """
    pass  

def update_position():
    """
    Updates the position of particles based on their velocity.

    Returns:
        None
    """
    pass