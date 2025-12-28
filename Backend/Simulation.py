"""
Simulation - Berechnet Interaktionen zwischen Partikel
FIXES:

- Tippfehler korrigiert
- gamma → FRICTION
- numba entfernt (funktioniert nicht mit self!!)
- Imports korrigiert
- step() Methode hinzugefügt
"""
import numpy as np
from typing import Tuple, Optional
from Backend.particle_system import Particles
from Config.config import (
    NUM_PARTICLES, FRICTION, INTERACTION_RADIUS,
    PARTICLE_RADIUS, WIDTH, HEIGHT, INTERACTION_MATRIX
)


class Simulation:
    """
    Berechnet Interaktionen zwischen Partikeln.
    
   Coulomb-Kraft mit Reibung
    """

    def __init__(self):
        # Interaktionsmatrix aus Config (nicht mehr hardcoded)
        self._interactionmatrix: np.ndarray = INTERACTION_MATRIX
        
        # Partikel erstellen
        self._particles: Particles = Particles()
        
        # Für Nachbarschaftsprüfung (Michaels Original)
        self._checked_particles: np.ndarray = np.zeros(NUM_PARTICLES, dtype=bool)

    @property
    def particles(self) -> Particles:
        """Zugriff auf Partikel-Objekt für GUI"""
        return self._particles

    def check_interactions(self, position_x: float, position_y: float, radius: float, index: int) -> Optional[Tuple]:
        """
        Findet alle Nachbarn im Radius.
        
        Masken-Konzept mit Bugfixes:
        - Klammern bei & Operator hinzugefügt
        - Radius in beide Richtungen (-radius und +radius)
        - Tippfehler korrigiert
        
        Returns:
            Tuple (neighbours_x, neighbours_y, interactions, indices) oder None
        """
        # FIX: Klammern um Vergleiche (& hat höhere Priorität als >= und <=)
        # FIX: Radius in beide Richtungen (>= px-r UND <= px+r)
        maske_x = (self._particles.x >= position_x - radius) & (self._particles.x <= position_x + radius)
        maske_y = (self._particles.y >= position_y - radius) & (self._particles.y <= position_y + radius)
        maske_n = maske_x & maske_y
        
        # Sich selbst ausschließen
        maske_n[index] = False
        
        # Keine Nachbarn gefunden
        if np.sum(maske_n) == 0:
            return None
        
        # Nachbar-Positionen
        neighbours_x = self._particles.x[maske_n]
        neighbours_y = self._particles.y[maske_n]
        
        # Typen der Nachbarn herausfinden
        n_types: np.ndarray = self._particles.types[maske_n]
        
        # Interaktionswerte aus Matrix holen
        my_type = self._particles.types[index]
        interactions = self._interactionmatrix[my_type, n_types]
        
        # Indizes der Nachbarn (für calc_velocity)
        indices = np.where(maske_n)[0]
        
        return (neighbours_x, neighbours_y, interactions, indices)

    def calc_velocity(self, position_x: float, position_y: float, 
                      neighbours_x: np.ndarray, neighbours_y: np.ndarray, 
                      interactions: np.ndarray, index: int, indices: np.ndarray) -> None:
        """
        Coulomb-Kraft Berechnung mit Reibung.
        
        Fixes:
        - gamma → FRICTION (aus Config)
        - Tippfehler korrigiert
        - Division durch 0 verhindert
        """
        r1: np.ndarray = np.array([position_x, position_y])
        k: float = 1.0       # Coulomb-Konstante (vereinfacht)
        m1: float = 1.0      # Masse Partikel 1
        m2: float = 1.0      # Masse Partikel 2
        t: float = 0.01      # Zeitschritt
        gamma: float = FRICTION  # FIX: gamma war nicht definiert
        
        for i in range(neighbours_x.shape[0]):
            r2 = np.array([neighbours_x[i], neighbours_y[i]])
            r = r1 - r2
            
            # Abstand berechnen
            r_abs = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2)
            
            # FIX: Division durch 0 verhindern
            if r_abs < 1e-10:
                continue
            
            # Einheitsvektor (Richtung)
            r_norm: np.ndarray = r / r_abs
            
            # Coulomb-Kraft: F = k * q1*q2 / r²
            # interactions[i] enthält q1*q2 aus der Matrix
            f1: np.ndarray = k * (interactions[i] / r_abs**2) * r_norm
            f2: np.ndarray = f1 * -1  # Gegenkraft
            
            # Mit Reibungskraft verrechnen
            # FIX: gamma → FRICTION
            f1 = f1 - gamma * self._particles.velocity_x[index]
            f2 = f2 - gamma * self._particles.velocity_y[indices[i]]
            
            # Beschleunigung (a = F / m)
            a1: np.ndarray = f1 / m1
            a2: np.ndarray = f2 / m2
            
            # Geschwindigkeit ändern (v = v + a * t)
            self._particles.velocity_x[index] += a1[0] * t
            self._particles.velocity_y[index] += a1[1] * t
            self._particles.velocity_x[indices[i]] += a2[0] * t
            self._particles.velocity_y[indices[i]] += a2[1] * t
            
            # Position ändern (x = x + v * t)
            self._particles.x[index] += self._particles.velocity_x[index] * t
            self._particles.y[index] += self._particles.velocity_y[index] * t
            self._particles.x[indices[i]] += self._particles.velocity_x[indices[i]] * t
            self._particles.y[indices[i]] += self._particles.velocity_y[indices[i]] * t

    def calc_friction(self, velocity_x: np.ndarray, velocity_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Berechnet Reibungskraft.
        
        Wird in calc_velocity schon angewendet, aber hier für Kompatibilität.
        """
        friction_x = -FRICTION * velocity_x
        friction_y = -FRICTION * velocity_y
        return friction_x, friction_y

    def diffuse(self) -> Tuple[np.ndarray, np.ndarray]:
        """
         diffuse() Methode
        
        1. Nachbarn finden (check_interactions)
        2. Geschwindigkeit berechnen (calc_velocity)
        3. Position aktualisieren
        """
        for i in range(NUM_PARTICLES):
            result = self.check_interactions(
                self._particles.x[i], 
                self._particles.y[i], 
                INTERACTION_RADIUS, 
                i
            )
            
            if result is not None:
                neighbours_x, neighbours_y, interactions, indices = result
                self.calc_velocity(
                    self._particles.x[i], 
                    self._particles.y[i], 
                    neighbours_x, 
                    neighbours_y, 
                    interactions, 
                    i, 
                    indices
                )
            else:
                # Keine Nachbarn - nur Position mit aktueller Geschwindigkeit ändern
                self._particles.x[i] += self._particles.velocity_x[i] * 0.01
                self._particles.y[i] += self._particles.velocity_y[i] * 0.01
        
        # Wrapping am Rand
        self._particles.x = self._particles.x % WIDTH
        self._particles.y = self._particles.y % HEIGHT
        
        return self._particles.x, self._particles.y

    def step(self, dt: float = 0.01) -> None:
        """
        Einfacher Wrapper für diffuse().
        
        Wird von der GUI aufgerufen (sim.step()).
        """
        self.diffuse()

    def get_particles_x(self) -> np.ndarray:
        """Original-Getter."""
        return self._particles.x
    
    def get_particles_y(self) -> np.ndarray:
        """Original-Getter."""
        return self._particles.y