"""
Particle Life Simulator - Konsolen-Test
Milestone 2: Grundlegende Simulation funktioniert über Konsole

Integriert habe ich: 
- Backend/particle_system.py (mit @property ergänzt)
- Config/config.py (mit INTERACTION_MATRIX ergänzt)
- Backend/Environment.py (Code mit Bugfixes)
- Ergänzungen (calc_friction, calc_force)
- damit Partikel sich nicht überlagern fixes2 und 3 etc. s.u.
"""

import numpy as np
from typing import Tuple, Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QGroupBox, QGridLayout, QTabWidget, QMessageBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPainter, QColor, QPen
import sys

# ==================== CONFIG  ====================
# Fenstergröße
WIDTH, HEIGHT = 800, 800

# Simulationsparameter
NUM_PARTICLES = 100  # Für Test weniger (im Repo: 5000)
PARTICLE_RADIUS = 3
FRICTION = 0.95  # Wie im Repo
MAX_SPEED = 5
INTERACTION_RADIUS = 100
ATTRACTION_STRENGTH = 0.05
RANDOM_FORCE_STRENGTH = 0.5  # Wie im Repo

# Farben und Partikeltypen
COLORS = {
    'RED': (255, 60, 60),
    'GREEN': (60, 255, 60),
    'BLUE': (60, 60, 255),
    'YELLOW': (255, 255, 60)
}
COLOR_LIST = list(COLORS.keys())
COLOR_VALUES = np.array(list(COLORS.values()))  # Wie im Repo
NUM_TYPES = len(COLOR_LIST)

# INTERACTION_MATRIX (muss in Config/config.py ergänzt werden)
# Positive Werte = Abstoßung, Negative Werte = Anziehung
INTERACTION_MATRIX = np.array([
    [+1.0, -1.0, +0.5, -0.5],   # RED
    [-1.0, +1.0, -0.5, +0.5],   # GREEN
    [+0.5, -0.5, +1.0, -1.0],   # BLUE
    [-0.5, +0.5, -1.0, +1.0],   # YELLOW
])


# ==================== PARTICLES  ====================
class Particles:
    """
    Verwaltet alle Partikel als NumPy Arrays, ergänzt um @property
    """
    
    def __init__(self):
        # keine Parameter
        self._x = np.random.rand(NUM_PARTICLES) * WIDTH
        self._y = np.random.rand(NUM_PARTICLES) * HEIGHT
        self._velocity_x = np.zeros(NUM_PARTICLES)
        self._velocity_y = np.zeros(NUM_PARTICLES)
        self._types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES)
    
    # mit property
    @property
    def x(self) -> np.ndarray: 
        return self._x
    @x.setter
    def x(self, value: np.ndarray) -> None: 
        self._x = value
    
    @property
    def y(self) -> np.ndarray: 
        return self._y
    @y.setter
    def y(self, value: np.ndarray) -> None: 
        self._y = value
  
    @property
    def velocity_x(self) -> np.ndarray: 
        return self._velocity_x
    @velocity_x.setter
    def velocity_x(self, value: np.ndarray) -> None: 
        self._velocity_x = value
    
    @property
    def velocity_y(self) -> np.ndarray: 
        return self._velocity_y
    @velocity_y.setter
    def velocity_y(self, value: np.ndarray) -> None: 
        self._velocity_y = value
    
    @property
    def types(self) -> np.ndarray: 
        return self._types
    @types.setter
    def types(self, value: np.ndarray) -> None: 
        self._types = value


# ==================== ENVIRONMENT ====================
class Environment:
    """
    Berechnet Interaktionen zwischen Partikeln.
    
    """

    def __init__(self):
        # Interaktionsmatrix (aber jetzt halt aus Config)
        self._interactionmatrix: np.ndarray = INTERACTION_MATRIX
        
      
        self._particles: Particles = Particles()
        
        # TODO: _checked_particles wird deklariert, aber nie verwendet
        # Bugfix: .shape gibt es nicht, nutze NUM_PARTICLES
        self._checked_particles: np.ndarray = np.zeros(NUM_PARTICLES, dtype=bool)

    #  @numba.jit(nopython=True) hat nicht mit self funktioniert
    # evtl als separa Funktion nachher 
    
    def check_interactions(self, position_x, position_y, radius, index) -> Optional[np.ndarray]:
        """
        Bugfixes: Klammern und Tippfehler
        """
        # Masken-Konzept
        
        # Bugfix: Klammern, und Radius in beide Richtungen

        maske_x = (self._particles.x >= position_x - radius) & (self._particles.x <= position_x + radius)
        maske_y = (self._particles.y >= position_y - radius) & (self._particles.y <= position_y + radius)
        maske_n = maske_x & maske_y
        maske_n[index] = False
        
        if np.sum(maske_n) == 0:
            return None  # Keine Nachbarn 
        
        neighbours_x = self._particles.x[maske_n]
        neighbours_y = self._particles.y[maske_n]
        
        # ypen der benachbarten Particles herausfinden
        n_types: np.ndarray = self._particles.types[maske_n]
        
        # Interaktionswerte holen
        my_type = self._particles.types[index]
        interactions = self._interactionmatrix[my_type, n_types]
        
        # Return als Tuple statt np.array 

        return (neighbours_x, neighbours_y, interactions, maske_n)

    ## @numba.jit
    ## def calc_velocity(self, position_x, position_y, neigbours_x, neighbours_y) -> np.ndarray:
    ##     pass

    def calc_friction(self, velocity_x: np.ndarray, velocity_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        friction forces für alle Partikel
        
        Wichtig: Diese Methode wird nicht mehr in step() verwendet, Reibung jetzt als direkter Dämpfungsfaktor 
        """
        friction_x = -FRICTION * velocity_x
        friction_y = -FRICTION * velocity_y
        return friction_x, friction_y

    def calc_force(
        self,
        index: int,
        ## weitere Parameter erstmal rausgelassen
        ## neigh_idx,, position: np.ndarray, type_idx, interaction: np.ndarray,
        ## velocity_x: Optional[np.ndarray] = None,
        ## velocity_y: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Berechnet Kraft für einen Partikel
        
       vereinfacht für Simulation, aber mir check_interactions() für Nachbarsuche
        """
        px = self._particles.x[index]
        py = self._particles.y[index]
        
        result = self.check_interactions(px, py, INTERACTION_RADIUS, index)
        
        if result is None:
            return 0.0, 0.0
        
        neighbours_x, neighbours_y, interactions, mask = result
        
        # Richtung zu Nachbarn
        dx = neighbours_x - px
        dy = neighbours_y - py
        
        # Abstand
        distances = np.sqrt(dx.astype(float)**2 + dy.astype(float)**2)
        
        # Epsilon gegen Division durch 0
      
        eps = 1e-12
        
        # Einheitsvektor (Richtung)
        unit_x = dx / (distances + eps)
        unit_y = dy / (distances + eps)
        
        # FIX 3: Nahbereichs-Abstoßung - Partikel sollen sich nie überlagern
        # Mindestabstand = 2 * PARTICLE_RADIUS (beide Radien zusammen)
        MIN_DIST = 2 * PARTICLE_RADIUS
        # Starke Abstoßung wenn Distanz < MIN_DIST, quadratisch für harte Abstoßung
        repulsion = np.where(distances < MIN_DIST,
                             (MIN_DIST - distances)**2 * 10.0,
                             0.0)
        
        # Kraft berechnen
        # Original: force_scalar * (dx / distance) - also lineare Kraft
        # statt Coulomb /r² hier
        force_magnitude = interactions / (distances + eps)
        
        # Summe aller Kräfte
        # FIX 1: Negatives Vorzeichen, damit positive Interaktion = Abstoßung
        # Repulsion wird subtrahiert (zeigt weg vom Nachbarn)
        fx = -np.sum(force_magnitude * unit_x) - np.sum(repulsion * unit_x)
        fy = -np.sum(force_magnitude * unit_y) - np.sum(repulsion * unit_y)
        
        return fx, fy

    def step(self, dt: float = 0.01) -> None:
        """
        Führt nen Simulationsschritt durch
        """
        n = NUM_PARTICLES
        
        # Kräfte für alle Partikel berechnen
        force_x = np.zeros(n)
        force_y = np.zeros(n)
        
        for i in range(n):
            fx, fy = self.calc_force(i)
            force_x[i] = fx
            force_y[i] = fy
        
        # Geschwindigkeit ändern (nur mit Kräften)
        self._particles.velocity_x = self._particles.velocity_x + force_x * dt
        self._particles.velocity_y = self._particles.velocity_y + force_y * dt
        
        # FIX 2: Reibung als direkter Dämpfungsfaktor anwenden
        self._particles.velocity_x *= FRICTION
        self._particles.velocity_y *= FRICTION
        
        # Position ändern
        self._particles.x = self._particles.x + self._particles.velocity_x * dt
        self._particles.y = self._particles.y + self._particles.velocity_y * dt
        
        # Wrapping am Rand 
        self._particles.x = self._particles.x % WIDTH
        self._particles.y = self._particles.y % HEIGHT

    def diffuse(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        diffuse-Methode - ruft step() auf.
        """
        self.step()
        return self._particles.x, self._particles.y


# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("PARTICLE LIFE SIMULATOR")
    print("=" * 60)
    print()
    
    # Environment erstellen (enthält Particles)
    env = Environment()
    particles = env._particles
    
    # Statistik
    print(f"Partikel erstellt: {NUM_PARTICLES}")
    print(f"Config: FRICTION={FRICTION}, INTERACTION_RADIUS={INTERACTION_RADIUS}")
    print()
    print(f"Typen-Verteilung:")
    for i, name in enumerate(COLOR_LIST):
        count = np.sum(particles.types == i)
        print(f"  {name}: {count}")
    print()
    
    # Anfangszustand
    print("Anfangszustand (Partikel 0-4):")
    print("-" * 50)
    for i in range(min(5, NUM_PARTICLES)):
        t = COLOR_LIST[particles.types[i]]
        print(f"  [{i}] {t:6s}: pos=({particles.x[i]:6.1f}, {particles.y[i]:6.1f})")
    print()
    
    # Simulation laufen lassen, steps Zahl
    num_steps = 50
    print(f"Starte Simulation ({num_steps} Schritte)...")
    print("-" * 50)
    
    for step in range(num_steps):
        env.diffuse() 
        
        if step % 10 == 0:
            avg_speed = np.mean(np.sqrt(particles.velocity_x**2 + particles.velocity_y**2))
            print(f"  Step {step:3d}: Partikel 0 bei ({particles.x[0]:6.1f}, {particles.y[0]:6.1f}), avg speed: {avg_speed:.4f}")
    
    print()
    print("-" * 50)
    
    print("Endzustand (Partikel 0-4):")
    print("-" * 50)
    for i in range(min(5, NUM_PARTICLES)):
        t = COLOR_LIST[particles.types[i]]
        vx, vy = particles.velocity_x[i], particles.velocity_y[i]
        print(f"  [{i}] {t:6s}: pos=({particles.x[i]:6.1f}, {particles.y[i]:6.1f}), vel=({vx:6.3f}, {vy:6.3f})")
    
    print()
    print("=" * 60)
    print("Simulation funktioniert")
    print("=" * 60)

if __name__ == "__main__":
    main()

