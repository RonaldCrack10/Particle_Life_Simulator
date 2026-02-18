
from Backend.Particles import Particles


def test_particles_system():
    particles = Particles()
    assert particles.x.shape == (1000,)
    assert particles.y.shape == (1000,)
    assert particles.velocity_x.shape == (1000,)
    assert particles.velocity_y.shape == (1000,)
    assert particles.types.shape == (1000,)