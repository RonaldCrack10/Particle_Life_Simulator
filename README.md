# Particle Life Simulator

A Python simulation of a dynamic particle system where thousands of particles interact based on predefined rules and exhibit emergent behavior.

## What does the program do?

- Thousands of colored dots (particles) move across the screen
- Each color reacts differently to other colors (attracts, repels, ignores)
- This creates patterns, clusters, and movements – without us explicitly programming them
- Parameters can be adjusted live during the simulation

## Project Goal

Create a visually appealing and computationally efficient particle simulation with:

- Multiple particle types with unique interaction patterns
- Real-time visualization of ≥2,000 particles
- Live adjustment of simulation parameters during runtime
- Emergent behavior through simple interaction rules

Inspired by: [Particle Life WebGL Demo](https://particle-life.com/)

## Features

- 4 particle types with individual colors (Red, Yellow, Green, Blue)
- Interaction matrix for attraction/repulsion between all type combinations
- Real-time rendering with VisPy
- Interactive GUI for live parameter adjustment (optional)
- Performance-optimized with NumPy (and Numba?)
- Professional code quality with linting, testing, and CI/CD

## Installation

### Prerequisites

- Python
- Git

### Setup

```bash
git clone https://github.com/RonaldCrack10/Particle_Life_Simulator.git
cd particle-life

# optionally create virtual environment

# install packages

# start program
python -m particle_life
```

## How does the simulation work?

### Particles

Each particle has:

- **Position (x, y)** – where it is located
- **Velocity (vx, vy)** – where it's moving
- **Type (0, 1, 2, 3)** – determines color and behavior

### Interaction Matrix

Interaction values are determined by a matrix:

|       | Red | Yellow | Green | Blue |
|-------|-----|--------|-------|------|
| Red   | ... | ...    | ...   | ...  |
| Yellow| ... | ...    | ...   | ...  |
| Green | ... | ...    | ...   | ...  |
| Blue  | ... | ...    | ...   | ...  |

- **Positive values**: Attraction
- **Negative values**: Repulsion
- **Zero**: Neutral

### Simulation Loop

In each frame:

1. For each particle: Calculate forces from all other nearby particles
2. Update velocity based on forces
3. Update position based on velocity
4. Redraw all particles

## Parameters

- **num_particles**: Number of particles (more = slower)
- **friction**: Friction that slows down particles (0 = none, 1 = instant stop)
- **influence_radius**: How far particles can "see" each other
- **dt**: Time step: how fast the simulation runs

## Project Structure

```
particle-life/
├── .github/workflows/ci.yml   # Automated tests (GitHub Actions)
├── src/particle_life/         # Our code
│   ├── __init__.py
│   ├── __main__.py            # Entry point of the program
│   ├── particles.py           # Particle data (position, velocity, type)
│   ├── simulation.py          # Movement calculations
│   ├── renderer.py            # Visualization (what we see)
│   └── config.py              # Settings (colors, parameters)
├── tests/                     # Our tests
│   ├── test_particles.py
│   └── test_simulation.py
├── pyproject.toml             # Project configuration & dependencies
├── README.md                  # This file
└── .gitignore                 # Files Git should ignore
```

## Architecture

```
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
```

## Workflow

1. Create fork of the repository
2. Create new issue on GitHub
3. Create branch: `git checkout -b feature/my-feature`
4. Write code + tests
5. Commit: `git commit -m "Add: description"`
6. Push: `git push origin feature/my-feature`
7. Create Pull Request on GitHub
8. Code review, then merge

## Roadmap / Milestones

### Milestone 1: Core Logic
- Project structure & setup
- ParticleSystem class
- InteractionMatrix class
- Simulation class with update loop
- Console output for testing

### Milestone 2: Visualization & GUI
- Renderer for drawing particles
- Real-time display
- 4 colors for 4 types
- GUI for parameters
- ControlPanel with live parameter adjustment

### Milestone 3: Testing & CI
- Write unit tests
- 70% code coverage
- GitHub Actions CI
- Linting in CI

### Milestone 4: Optimization
- Profiling & bottleneck analysis
- Numba or similar for JIT optimization
- ≥1,000 particles performing efficiently
- Complete documentation

## Useful Libraries

- NumPy
- Numba
- VisPy
- PyQt5
- pytest
- Ruff
- (more to be added)

## Authors

- Ronald
- Katharina
- Michael
- Alina

Created as a student project to demonstrate professional Python development with Clean Code, Testing, CI/CD, and performance optimization.

## License

MIT License

---

**About**: A demonstration of emergent behavior through particle interactions  
**Stars**: ⭐ 1  
**Language**: Python 100.0%
