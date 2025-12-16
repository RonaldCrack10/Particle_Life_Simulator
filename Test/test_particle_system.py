from Backend.Environment import calc_friction
import numpy as np
from Backend.Environment import Environment
from Config.config import FRICTION

def test_calc_friction():
    env = Environment()
    velocity_x = np.array([1.0, -2.0, 0.0])
    velocity_y = np.array([0.5, -1.5, 0.0])
    
    friction_x, friction_y = env.calc_friction(velocity_x, velocity_y)
    
    expected_friction_x = -FRICTION * velocity_x
    expected_friction_y = -FRICTION * velocity_y
    
    assert (friction_x == expected_friction_x).all(), "Friction X calculation is incorrect"
    assert (friction_y == expected_friction_y).all(), "Friction Y calculation is incorrect"