from Backend.Environment import calc_friction
from Backend.particle_system import Particles
import numpy as np
from Backend.Environment import Environment
from Config.config import FRICTION

def test_particle_funtion():
    particles = Particles()
    env = Environment()

    # Teste, ob die Partikel innerhalb der Fenstergrenzen liegen
    assert np.all(particles.x >= 0) and np.all(particles.x <= env.width)
    assert np.all(particles.y >= 0) and np.all(particles.y <= env.height)

    # Teste, ob die Anfangsgeschwindigkeiten 0 sind
    assert np.all(particles.velocity_x == 0)
    assert np.all(particles.velocity_y == 0)

    # Teste, ob die Partikeltypen im gültigen Bereich liegen
    assert np.all(particles.types >= 0) and np.all(particles.types < env.num_types)

    # Teste die Reibungsberechnung
    test_velocity = 10.0
    expected_friction = test_velocity * FRICTION
    calculated_friction = calc_friction(test_velocity)
    assert np.isclose(expected_friction, calculated_friction)