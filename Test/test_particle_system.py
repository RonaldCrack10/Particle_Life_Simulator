
from Backend.particle_system import Particles
import numpy as np
from Backend.Environment import Environment
from Config.config import FRICTION

import numpy as np
from Backend.particle_system import Particles
from Backend.Environment import Environment

from Config.config import FRICTION, WIDTH, HEIGHT, NUM_TYPES

def test_particle_system():
    particles = Particles()
    env = Environment()

    # 1. Teste Fenstergrenzen (Nutze die globalen WIDTH/HEIGHT, 
    # da die Environment diese intern verwendet)
    assert np.all(particles.x >= 0) and np.all(particles.x <= WIDTH)
    assert np.all(particles.y >= 0) and np.all(particles.y <= HEIGHT)

    # 2. Teste Anfangsgeschwindigkeiten
    assert np.all(particles.velocity_x == 0)
    assert np.all(particles.velocity_y == 0)

    # 3. Teste Partikeltypen (NUM_TYPES statt env.num_types)
    assert np.all(particles.types >= 0) and np.all(particles.types < NUM_TYPES)

    # 4. Teste die Reibungsberechnung
    # In deinem Code oben war calc_friction: velocity * FRICTION
    test_vel_x = np.array([10.0, -5.0])
    test_vel_y = np.array([0.0, 8.0])
    
    # Deine Environment.calc_friction gibt (vx, vy) zurück
    calc_v_x, calc_v_y = env.calc_friction(test_vel_x, test_vel_y)
    
    # WICHTIG: In deinem Code oben ist calc_friction = -FRICTION * velocity
    # Prüfe, ob die Richtung stimmt (Dämpfung)
    expected_x = -FRICTION * test_vel_x
    assert np.allclose(calc_v_x, expected_x)