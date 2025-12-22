"""
Performance Tests für Particle Life Simulator

Ausführen:
    python Test/test_particle_system.py

Nur Performance (timeit):
    python Test/test_particle_system.py --perf

Nur cProfile:
    python Test/test_particle_system.py --profile

Mit snakeviz visualisieren:
    pip install snakeviz
    python Test/test_particle_system.py --snakeviz

Profil speichern:
    python Test/test_particle_system.py --save-profile output.prof
"""
import numpy as np
import timeit
import cProfile
import pstats
import io
import sys
import os
import argparse

# Pfad für Imports anpassen
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Config.config import (
    WIDTH, HEIGHT, NUM_PARTICLES, NUM_TYPES,
    FRICTION, INTERACTION_RADIUS, PARTICLE_RADIUS, INTERACTION_MATRIX
)
from Backend.particle_system import Particles
from Backend.Environment import Environment


# ============================================================================
# PERFORMANCE TESTS (timeit)
# ============================================================================

def run_timeit_tests():
    """Performance-Tests mit timeit"""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS (timeit)")
    print("=" * 60)

    env = Environment()

    time_step = timeit.timeit(lambda: env.step(), number=10)
    print(f"\nstep() x10:                 {time_step:.4f}s  ({time_step/10*1000:.2f}ms pro step)")

    px, py = env._particles.x[0], env._particles.y[0]
    time_check = timeit.timeit(
        lambda: env.check_interactions(px, py, INTERACTION_RADIUS, 0),
        number=1000
    )
    print(f"check_interactions() x1000: {time_check:.4f}s  ({time_check/1000*1000:.3f}ms pro Aufruf)")

    time_force = timeit.timeit(lambda: env.calc_force(0), number=1000)
    print(f"calc_force() x1000:         {time_force:.4f}s  ({time_force/1000*1000:.3f}ms pro Aufruf)")

    time_friction = timeit.timeit(
        lambda: env.calc_friction(env._particles.velocity_x, env._particles.velocity_y),
        number=10000
    )
    print(f"calc_friction() x10000:     {time_friction:.4f}s  ({time_friction/10000*1000:.4f}ms pro Aufruf)")


def run_scaling_test():
    """Testet O(n²) Skalierung"""
    print("\n" + "-" * 60)
    print("SKALIERUNG (O(n²) Problem)")
    print("-" * 60)
    print(f"{'n':>6} | {'Zeit (5 steps)':>14} | {'ms/step':>10} | {'Faktor':>8}")
    print("-" * 50)

    times = []
    ns = [50, 100, 200, 400]

    for n in ns:
        env = Environment()
        env._particles._x = np.random.rand(n) * WIDTH
        env._particles._y = np.random.rand(n) * HEIGHT
        env._particles._velocity_x = np.zeros(n)
        env._particles._velocity_y = np.zeros(n)
        env._particles._types = np.random.randint(0, NUM_TYPES, n)

        def run_steps():
            for _ in range(5):
                force_x = np.zeros(n)
                force_y = np.zeros(n)
                for i in range(n):
                    fx, fy = env.calc_force(i)
                    force_x[i] = fx
                    force_y[i] = fy
                env._particles.velocity_x = env._particles.velocity_x + force_x * 0.01
                env._particles.velocity_y = env._particles.velocity_y + force_y * 0.01
                env._particles.velocity_x *= FRICTION
                env._particles.velocity_y *= FRICTION
                env._particles.x = (env._particles.x + env._particles.velocity_x * 0.01) % WIDTH
                env._particles.y = (env._particles.y + env._particles.velocity_y * 0.01) % HEIGHT

        time_n = timeit.timeit(run_steps, number=1)
        times.append(time_n)

        faktor = times[-1] / times[0] if len(times) > 0 else 1.0
        print(f"{n:>6} | {time_n:>14.4f}s | {time_n/5*1000:>10.2f} | {faktor:>8.2f}x")

    print("-" * 50)
    print("Bei O(n²): Verdopplung von n → 4x Zeit")
    print(f"Gemessen: n×8 → {times[-1]/times[0]:.1f}x Zeit")


# ============================================================================
# CPROFILE
# ============================================================================

def run_cprofile(num_steps=20, top_n=20):
    """Findet Bottlenecks mit cProfile"""
    print("\n" + "=" * 60)
    print(f"CPROFILE ANALYSE ({num_steps} steps)")
    print("=" * 60)

    env = Environment()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(num_steps):
        env.step()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)

    print(stream.getvalue())

    print("-" * 60)
    print("SPALTEN ERKLÄRT:")
    print("-" * 60)
    print("ncalls  = Anzahl Aufrufe")
    print("tottime = Zeit NUR in dieser Funktion")
    print("cumtime = Gesamtzeit (inkl. Unterfunktionen)")
    print("-" * 60)


def run_cprofile_to_file(filename="profile_output.prof", num_steps=50):
    """Speichert cProfile Ergebnis in Datei für snakeviz"""
    print("\n" + "=" * 60)
    print(f"CPROFILE → {filename}")
    print("=" * 60)

    env = Environment()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(num_steps):
        env.step()
    profiler.disable()

    profiler.dump_stats(filename)
    print(f"Profil gespeichert: {filename}")
    print(f"\nZum Visualisieren:")
    print(f"  pip install snakeviz")
    print(f"  snakeviz {filename}")

    return filename


def run_snakeviz(filename="profile_output.prof", num_steps=50):
    """Erstellt Profil und öffnet snakeviz"""
    import subprocess
    import shutil

    if shutil.which("snakeviz") is None:
        print("\n" + "=" * 60)
        print("SNAKEVIZ NICHT INSTALLIERT")
        print("=" * 60)
        print("Installieren mit: pip install snakeviz")
        print("\nErstelle trotzdem Profil-Datei...")
        run_cprofile_to_file(filename, num_steps)
        return

    print("\n" + "=" * 60)
    print("SNAKEVIZ VISUALISIERUNG")
    print("=" * 60)

    env = Environment()

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(num_steps):
        env.step()
    profiler.disable()

    profiler.dump_stats(filename)
    print(f"Profil erstellt: {filename}")
    print("Öffne snakeviz im Browser...")

    subprocess.run(["snakeviz", filename])


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Tests für Particle Life Simulator")
    parser.add_argument("--perf", action="store_true", help="Nur Performance Tests (timeit)")
    parser.add_argument("--profile", action="store_true", help="Nur cProfile")
    parser.add_argument("--snakeviz", action="store_true", help="cProfile mit snakeviz visualisieren")
    parser.add_argument("--save-profile", type=str, help="Profil in Datei speichern")

    args = parser.parse_args()

    run_all = not any([args.perf, args.profile, args.snakeviz, args.save_profile])

    if args.perf or run_all:
        run_timeit_tests()
        run_scaling_test()

    if args.profile or run_all:
        run_cprofile()

    if args.save_profile:
        run_cprofile_to_file(args.save_profile)

    if args.snakeviz:
        run_snakeviz()