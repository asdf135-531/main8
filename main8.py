import random
import math
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

# -----------------------------
# Источник
# -----------------------------
class Source:
    def __init__(self, x0, y0, z0):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0


# -----------------------------
# Случайное направление
# -----------------------------
def ray():
    while True:
        l = random.uniform(-1, 1)
        n = random.uniform(-1, 1)
        m = random.uniform(-1, 1)
        norm = math.sqrt(l*l + n*n + m*m)
        if norm > 0:
            break
    return l/norm, n/norm, m/norm


# -----------------------------
# Коэффициенты
# -----------------------------
def func_sigmaPh(E, Z):
    return 6.651e-25 * 4 * math.sqrt(2) * (Z**5) / (137**4) * (0.511/E)**(3.5)

def func_sigmaK(E, Z):
    g = E / 0.511
    return 6.651e-25 * 3 * Z / (8*g) * (
        (1 - 2*(g+1)/(g**2)) * math.log(2*g+1)
        + 0.5 + 4/g - 1/(2*(2*g+1)**2)
    )


# -----------------------------
# Комптоновское рассеяние
# -----------------------------
def compton(E, l, n, m):
    l2, n2, m2 = ray()

    cos = (l*l2 + n*n2 + m*m2) / (
        math.sqrt(l*l + n*n + m*m) * math.sqrt(l2*l2 + n2*n2 + m2*m2)
    )

    E_new = E / (1 + (E/0.511)*(1 - cos))
    E_loss = E - E_new

    return E_new, l2, n2, m2, E_loss


# -----------------------------
# Геометрия цилиндра
# -----------------------------
d = 10
r = 5

# -----------------------------
# Основной Monte Carlo
# -----------------------------
def cylinder(args):
    s, N = args

    N_CHANNELS = 1024
    spectrum = np.zeros(N_CHANNELS)

    E0 = 0.662  # MeV
    E_max = E0

    bins = np.linspace(0, E0, N_CHANNELS + 1)

    for _ in range(N):

        E = E0
        x, y, z = s.x0, s.y0, s.z0
        l, n, m = ray()

        deposited_energy = 0.0

        while E > 0.01:

            # свободный пробег (упрощённо)
            sigma = 1.0  # можно заменить на реальный σ
            step = -math.log(random.random()) / sigma

            # движение
            x += l * step
            y += n * step
            z += m * step

            # проверка выхода из цилиндра
            if x*x + y*y > r*r or abs(z) > d/2:
                break

            # взаимодействие
            if random.random() < 0.5:  # Compton
                E_new, l, n, m, dE = compton(E, l, n, m)
                deposited_energy += dE
                E = E_new
            else:  # фотоэффект
                deposited_energy += E
                E = 0

        # -------------------------
        # ПРАВИЛЬНЫЙ БИННИНГ
        # -------------------------
        idx = int(deposited_energy / E0 * N_CHANNELS)
        if 0 <= idx < N_CHANNELS:
            spectrum[idx] += 1

    return spectrum


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    N_total = 100000
    n_proc = 6

    s = Source(0, 0, 0)

    with Pool(n_proc) as p:
        results = p.map(cylinder, [(s, N_total // n_proc)] * n_proc)

    spectrum = np.sum(results, axis=0)

    E_axis = np.linspace(0, 0.662, len(spectrum))

    plt.figure(figsize=(12, 6))
    plt.bar(E_axis, spectrum, width=E_axis[1]-E_axis[0])

    plt.yscale("log")
    plt.xlabel("Energy deposited (MeV)")
    plt.ylabel("Counts")
    plt.grid(True)
    plt.show()