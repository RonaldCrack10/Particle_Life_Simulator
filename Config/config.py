import numpy as np
from Backend.particle_system import Particles
from Frontend.visualize import Visualize
from Backend.Simulation import Simulation
from Config.constants import WIDTH, HEIGHT, NUM_TYPES, NUM_PARTICLES, COLORS_VISPY

"""
Config - Parameter für Particle Life Simulator
"""
def config():
    #intanzierungen und configurierung
    particles = Particles(x = np.random.normal(loc=0.0, scale=10.0, size=10000),
                 y = np.random.normal(loc=0.0, scale=10.0, size=10000), 
                 velocity_x = np.zeros(10000), 
                 velocity_y = np.zeros(10000),  
                 types = np.clip(np.rint(np.random.normal(loc=0, scale=1.0, size=10000)), 0, 4).astype(int),
                 radius = 20)
    simulation = Simulation(particles=particles)
    visualize = Visualize(simulation=simulation, particles=particles)
    visualize.start()



