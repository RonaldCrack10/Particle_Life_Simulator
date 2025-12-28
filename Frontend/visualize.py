
"""
GUI - Einfachste VisPy Visualisierung
"""
import numpy as np
from vispy import app, scene
from Backend.Simulation import Simulation
from Config.config import WIDTH, HEIGHT, COLORS_VISPY

"""
GUI - VisPy Visualisierung
"""

# Fenster
canvas = scene.SceneCanvas(size=(WIDTH, HEIGHT), title='Particle Life')
view = canvas.central_widget.add_view()

# Simulation
sim = Simulation()

# Punkte
scatter = scene.visuals.Markers(parent=view.scene)


def update(event):
    sim.step()
    pos = np.column_stack([sim.particles.x, sim.particles.y])
    colors = COLORS_VISPY[sim.particles.types]
    scatter.set_data(pos, face_color=colors, size=10)


# 60 FPS
timer = app.Timer(interval=1/60, connect=update, start=True)

# Starten
canvas.show()
app.run()
