import numpy as np
from vispy import app, scene
from Backend.Simulation import Environment
from Config.config import *

"""
GUI - VisPy Visualisierung

"""

env = Environment()

canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
x_min, x_max = env.get_particles_x().min(), env.get_particles_x().max()
y_min, y_max = env.get_particles_y().min(), env.get_particles_y().max()
padding = 5  # Abstand zum Rand, damit Partikel nicht abgeschnitten werden
view.camera = scene.cameras.PanZoomCamera(rect=(-60, -60, 120, 120))  
view.camera.set_range(x=(-60,60), y=(-60,60))

# Initial random positions for particles
scatter = scene.visuals.Markers()
initial_pos = np.column_stack((env.get_particles_x(), env.get_particles_y()))

COLOR_MAP = {
    0: COLORS_VISPY[0],  # RED
    1: COLORS_VISPY[1],  # GREEN
    2: COLORS_VISPY[2],  # BLUE
    3: COLORS_VISPY[3],  # YELLOW
}
colors = np.array([COLOR_MAP[t] for t in env._particles.types])
scatter.set_data(initial_pos, face_color=colors, size=10)
view.add(scatter)

def update(event):


    env.diffuse()
    new_pos = np.array([env.get_particles_x(), env.get_particles_y()]).T
    scatter.set_data(new_pos, face_color= colors, size=10)
 
timer = app.Timer(interval= 1/60, connect=update, start=True)
 

 