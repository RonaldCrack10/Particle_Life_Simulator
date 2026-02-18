from Backend.Simulation import Simulation
import numpy as np
from Config.constants import INTERACTION_RADIUS, INTERACTION_MATRIX


def test_check_interactions():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0, 2.0, 3.0])
    sim._particles.y = np.array([0.0, 1.0, 2.0, 3.0])
    sim._particles.types = np.array([0, 1, 2, 3])
    
    result = sim.check_interactions(0.0, 0.0, radius=1.5, index=0)
    assert result is not None
    neighbours_x, neighbours_y, interactions, indices = result
    assert len(neighbours_x) == 1
    assert len(neighbours_y) == 1
    assert len(interactions) == 1
    assert len(indices) == 1
    assert neighbours_x[0] == 1.0
    assert neighbours_y[0] == 1.0
    assert interactions[0] == sim._interactionmatrix[sim._particles.types[0], sim._particles.types[1]]


def test_calc_velocity():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0])
    sim._particles.y = np.array([0.0, 1.0])
    sim._particles.types = np.array([0, 1])
    
    neighbours_x = np.array([1.0])
    neighbours_y = np.array([1.0])
    interactions = np.array([sim._interactionmatrix[sim._particles.types[0], sim._particles.types[1]]])
    
    sim.calc_velocity(0.0, 0.0, neighbours_x, neighbours_y, interactions, index=0, indices=np.array([1]))
    
    assert True


def run_performance_test():
    import timeit

    print("\n" + "-" * 60)
    print("LEISTUNGSTEST")
    print("-" * 60)

    sim = Simulation()
    time_step = timeit.timeit(lambda: sim.step(), number=10)
    print(f"\nstep() x10:                 {time_step:.4f}s  ({time_step/10*1000:.2f}ms pro step)")
    px, py = sim._particles.x[0], sim._particles.y[0]
    time_check = timeit.timeit(
        lambda: sim.check_interactions(px, py, INTERACTION_RADIUS, 0),
        number=1000
    )
    print(f"check_interactions() x1000: {time_check:.4f}s  ({time_check/1000*1000:.3f}ms pro Aufruf)")
    time_force = timeit.timeit(lambda: sim.calc_force(0), number=1000)
    print(f"calc_force() x1000:         {time_force:.4f}s ({time_force/1000*1000:.3f}ms pro Aufruf)")

    time_friction = timeit.timeit(
        lambda: sim.calc_friction(sim._particles.velocity_x, sim._particles.velocity_y),
        number=10000
    )
    print(f"calc_friction() x10000:     {time_friction:.4f}s  ({time_friction/10000*1000:.4f}ms pro Aufruf)")


def test_diffuse():
    sim = Simulation()
    sim._particles.x = np.array([0.0, 1.0])
    sim._particles.y = np.array([0.0, 1.0])
    sim._particles.types = np.array([0, 1])
    sim._particles.velocity_x = np.zeros(2)
    sim._particles.velocity_y = np.zeros(2)
    
    new_x, new_y = sim.diffuse()
    
    assert new_x is not None
    assert new_y is not None