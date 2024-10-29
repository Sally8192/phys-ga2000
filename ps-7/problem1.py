import numpy as np

# Constants (in kg and meters)
EARTH_MASS = 5.974e24
MOON_MASS = 7.348e22
SUN_MASS = 1.989e30
JUPITER_MASS = 1.898e27

EARTH_MOON_DIST = 3.844e8  # Distance from Earth to Moon (in meters)
SUN_EARTH_DIST = 1.5e11     # Distance from Sun to Earth (in meters)

def f(r_prime, m_prime):
    a = (1 - r_prime) ** 2
    return -r_prime ** 3 * a + a - m_prime * r_prime ** 2

def f_prime(r_prime, m_prime):
    return -5 * r_prime ** 4 + 8 * r_prime ** 3 - 3 * r_prime ** 2 + (2 - 2 * m_prime) * r_prime - 2

def Newton(initial, func, func_prime, m_prime, iters=100):
    r_prime = initial
    for _ in range(iters):
        r_new = r_prime - func(r_prime, m_prime) / func_prime(r_prime, m_prime)
        if abs(r_new - r_prime) < 1e-10:  # Convergence criterion
            break
        r_prime = r_new
    return r_prime

if __name__ == "__main__":
    # Use initial guesses
    r_L1_moon = EARTH_MOON_DIST * Newton(0.9, f, f_prime, MOON_MASS / EARTH_MASS)
    distance_from_earth_to_L1_moon = EARTH_MOON_DIST - r_L1_moon
    print(f"Earth and Moon: Distance from Earth to L1 point: {distance_from_earth_to_L1_moon:.4f} m")

    r_L1_sun = SUN_EARTH_DIST * Newton(0.9, f, f_prime, EARTH_MASS / SUN_MASS)
    distance_from_earth_to_L1_sun = SUN_EARTH_DIST - r_L1_sun
    print(f"Earth and Sun: Distance from Earth to L1 point: {distance_from_earth_to_L1_sun:.4f} m")

    # Update to calculate L1 for Jupiter mass at Earth's distance
    r_L1_jupiter = SUN_EARTH_DIST * Newton(0.9, f, f_prime, JUPITER_MASS / SUN_MASS)
    distance_from_earth_to_L1_jupiter = SUN_EARTH_DIST - r_L1_jupiter
    print(f"Jupiter mass planet at Earth's distance to Sun: Distance from Earth to L1 point: {distance_from_earth_to_L1_jupiter:.4f} m")
