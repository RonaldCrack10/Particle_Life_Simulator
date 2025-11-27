# Particle_Life_Simulator

Eine Python-Simulation eines dynamischen Partikelsystems, bei dem tausende Partikel basierend auf vordefinierten Regeln interagieren und emergentes Verhalten zeigen.


## Was macht das Programm?

- Tausende farbige Punkte (Partikel) bewegen sich auf dem Bildschirm
- Jede Farbe reagiert anders auf andere Farben (zieht an, stößt ab, ignoriert)
- Daraus entstehen Muster, Cluster und Bewegungen – ohne dass wir sie programmiert haben
- Die Parameter können live während der Simulation angepasst werden

## Projektziel 

Erstellung einer visuell ansprechenden und recheneffizienten Partikelsimulation mit:

- Mehreren Partikeltypen mit einzigartigen Interaktionsmustern
- Echtzeit-Visualisierung von ≥2.000 Partikeln
- Live-Anpassung der Simulationsparameter während der Simulation
- Emergentes Verhalten durch einfache Interaktionsregeln

Inspiriert von: [Particle Life WebGL Demo](https://webgl-particle-life.netlify.app)

## Features

- **4 Partikeltypen** mit individuellen Farben (Rot, Gelb, Grün, Blau)
- **Interaktionsmatrix** für Anziehung/Abstoßung zwischen allen Typkombinationen
- **Echtzeit-Rendering** mit VisPy (OpenGL-basiert)
- **Interaktive GUI** zur Live-Anpassung aller Parameter
- **Performance-optimiert** mit NumPy und Numba (JIT-Kompilierung)
- **Professionelle Codequalität** mit Linting, Testing und CI/CD

## Installation

### Voraussetzungen

- Python 3.10 oder höher
- Git

### Setup

git clone [https://github.com/RonaldCrack10/Particle_Life_Simulator.git]
cd particle-life

- evtl. virtuelle Umgebung erstellen
- Pakete installieren
- Programm starten

python -m particle_life

### Simulation starten mit benutzdefinierten Parametern
#### Parameter, Beschreibung und Standardwert

## Wie funktioniert die Simulation?

### Partikel

Jeder Partikel hat:
- **Position** (x, y) – wo er ist
- **Geschwindigkeit** (vx, vy) – wohin er sich bewegt
- **Typ** (0, 1, 2, 3) – bestimmt die Farbe und das Verhalten

### Interaktionsmatrix
#### Die Interaktionswerte werden durch eine Matrix bestimmt.
### Rot, Gelb, Grün, Blau x Rot, Gelb, Grün, Blau

**Positive Werte**: Anziehung
- **Negative Werte**: Abstoßung
- **Null**: Neutral

### Simulationsschleife

In jedem Frame passiert:
1. Für jeden Partikel: Berechne Kräfte von allen anderen Partikeln in der Nähe
2. Aktualisiere Geschwindigkeit basierend auf den Kräften
3. Aktualisiere Position basierend auf der Geschwindigkeit
4. Zeichne alle Partikel neu

## Parameter
- `num_particles`: Anzahl der Partikel (mehr = langsamer)
- `friction`: Reibung, bremst die Partikel ab (0 = keine, 1 = sofort stoppen)
- `influence_radius`: Wie weit Partikel sich gegenseitig "sehen" können
- `dt`: Zeitschritt: wie schnell die Simulation läuft

## Tests implementieren

## Code-Qualität prüfen

## Ordner

particle-life/
├── .github/workflows/ci.yml   # Automatische Tests (GitHub Actions)
├── src/particle_life/         # Unser Code
│   ├── __init__.py
│   ├── __main__.py            # Startpunkt des Programms
│   ├── particles.py           # Partikel-Daten (Position, Geschwindigkeit, Typ)
│   ├── simulation.py          # Berechnung der Bewegungen
│   ├── renderer.py            # Visualisierung (was wir sehen)
│   └── config.py              # Einstellungen (Farben, Parameter)
├── tests/                     # Unsere Tests
│   ├── test_particles.py
│   └── test_simulation.py
├── pyproject.toml             # Projekt-Konfiguration & Dependencies
├── README.md                  # Diese Datei
└── .gitignore                 # Dateien, die Git ignorieren soll


## Architektur

┌─────────────────────────────────────────────────────────────────────┐
│                           Application                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐      ┌────────────────────┐      ┌─────────────┐  │
│  │ ControlPanel │─────▶│    Simulation      │─────▶│  Renderer   │  │
│  │              │      │                    │      │             │  │
│  │              │      │  ┌──────────────┐  │      │             │  │
│  │ - Sliders    │      │  │ParticleSystem│  │      │ - Canvas    │  │
│  │ - Matrix UI  │      │  │              │  │      │ - Markers   │  │
│  │ - Buttons    │      │  │ - positions  │  │      │ - Colors    │  │
│  └──────────────┘      │  │ - velocities │  │      └─────────────┘  │
│         │              │  │ - types      │  │            ▲          │
│         │              │  └──────────────┘  │            │          │
│         │              │                    │            │          │
│         │              │  ┌──────────────┐  │            │          │
│         └─────────────▶│  │ Interaction- │  │────────────┘          │
│                        │  │    Matrix    │  │                       │
│                        │  │              │  │                       │
│                        │  │ - forces     │  │                       │
│                        │  │ - radius     │  │                       │
│                        │  │ - friction   │  │                       │
│                        │  └──────────────┘  │                       │
│                        └────────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

### Wie wir arbeiten
1. Fork des Respository erstellen
2. Neues Issue auf GitHub erstellen
3. Branch erstellen: `git checkout -b feature/mein-feature`
4. Code schreiben + Tests
5. Committen: `git commit -m "Add: Beschreibung"`
6. Pushen: `git push origin feature/mein-feature`
7. Pull Request auf GitHub erstellen
8. Code Review, dann Merge

## Roadmap / Milestones

### Milestone 1: Kernlogik

 Projektstruktur & Setup
 ParticleSystem Klasse
 InteractionMatrix Klasse
 Simulation Klasse mit Update-Loop
 Konsolen-Output zum Testen

### Milestone 2: Visualisierung & GUI

 Renderer für Partikel Zeichnen
 Echtzeit-Darstellung
 4 Farben für 4 Typen
 GUI für Parameter
 ControlPanel mit Live-Parameteranpassung

### Milestone 3: Testing & CI

 Unit Tests schreiben (pytest)
 >70% Code Coverage
 GitHub Actions CI 
 Linting im CI

### Milestone 4: Optimierung

 Profiling & Bottleneck-Analyse
 Numba o.ä. für JIT-Optimierung
 ≥1.000 Partikel performant
 Dokumentation zuende schreiben

## Sinnvolle Bibliotheken
- NumPy
- Numba
- VisPy
- PyQt5
- pytest
- Ruff
- weitere ergänzen

## Autoren
- **Ronald**
- **Katharina**
- **Michael**
- **Alina**
  
Erstellt als Studienprojekt zur Demonstration professioneller Python-Entwicklung mit Clean Code, Testing, CI/CD und Performance-Optimierung.
