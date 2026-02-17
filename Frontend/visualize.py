import numpy as np
from vispy import app, scene
from Backend.Simulation import Simulation
from Config.constants import *

"""
GUI - VisPy Visualisierung

"""
class Visualize:

   
    def __init__(self, simulation, particles):
        self._simulation = simulation
        self._particles = particles

        self._canvas = scene.SceneCanvas(keys="interactive", show=True)
        self._view = self._canvas.central_widget.add_view()
        self._view.camera = scene.cameras.PanZoomCamera(aspect=1)

        self._scatter = scene.visuals.Markers(parent=self._view.scene)

        # initial plot
        pos = np.c_[particles.x, particles.y]  # shape (N,2)
        COLOR_MAP = {
        0: COLORS_VISPY[0],  # RED
        1: COLORS_VISPY[1],  # GREEN
        2: COLORS_VISPY[2],  # BLUE
        3: COLORS_VISPY[3],
        4: COLORS_VISPY[4]  # YELLOW
        
    }
        colors = np.array([COLOR_MAP[t] for t in self._particles.types])
       
        self._scatter.set_data(pos, face_color=colors, size=6)

        

        # make sure points are in view
        self._view.camera.set_range(x=(pos[:, 0].min(), pos[:, 0].max()),
                                    y=(pos[:, 1].min(), pos[:, 1].max()))

        # timer calls self.update
        self._timer = app.Timer(interval=0.05, connect=self.update, start=True)

    def update(self, event):
        x, y = self._simulation.diffuse()
        pos = np.c_[x, y]
        COLOR_MAP = {
        0: COLORS_VISPY[0],  # RED
        1: COLORS_VISPY[1],  # GREEN
        2: COLORS_VISPY[2],  # BLUE
        3: COLORS_VISPY[3],  # YELLOW
        4: COLORS_VISPY[4]  # MAGENTA
        
    }
        colors = np.array([COLOR_MAP[t] for t in self._particles.types])
        # colors = np.array([COLOR_MAP[t] for t in self._particles.types])
        self._scatter.set_data(pos, face_color=colors, size=6)

        # optional: keep camera updated if points move a lot
        # self._view.camera.set_range(x=(pos[:,0].min(), pos[:,0].max()),
        #                             y=(pos[:,1].min(), pos[:,1].max()))

    def start(self):
        app.run()


if __name__ == "__main__":
    app.use_app("pyqt5")  # set backend BEFORE creating canvas

    # TODO: hier musst du deine echten Objekte einsetzen:
    simulation = ...   # hat .diffuse() -> (x, y)
    particles = ...    # hat .x und .y arrays

    viz = Visualize(simulation, particles)
    viz.start()


