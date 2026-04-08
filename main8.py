import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

NUM_PROCESSES = 4

R = 2.0
D = 4.0
half_height = D / 2.0

X0, Y0, Z0 = 4.0, 0.0, 0.0
N = 1000000
NBINS = 1024

m_e = 0.511
r_e = 2.818e-13
NA = 6.022e23
rho = 3.67
M_NaI = 149.89

n_mol = rho * NA / M_NaI
n_Na = n_mol
n_I = n_mol


def get_source_spectrum(source_name):
    sources = {
        'Cs137': [(0.662, 1.0)],
        'Co60': [(1.173, 1.0), (1.333, 1.0)],
        'Na22': [(0.511, 0.5), (1.274, 0.5)],
        'Ba133': [(0.356, 0.62), (0.081, 0.34), (0.303, 0.19), (0.276, 0.07)],
        'Am241': [(0.060, 1.0)],
        'Eu152': [(0.122, 0.28), (0.344, 0.27), (0.964, 0.15), (1.112, 0.14),
                  (1.408, 0.21), (0.245, 0.07), (0.444, 0.03)]
    }
    lines = sources.get(source_name, [(0.662, 1.0)])
    total_p = sum(p for _, p in lines)
    if total_p > 0:
        lines = [(e, p / total_p) for e, p in lines]
    return lines


def sample_initial_energy(lines):
    r = random.random()
    cum = 0.0
    for e, p in lines:
        cum += p
        if r < cum:
            return e
    return lines[-1][0]


def ray():
    while True:
        l = random.uniform(-1, 1)
        m = random.uniform(-1, 1)
        n = random.uniform(-1, 1)
        norm = math.sqrt(l * l + m * m + n * n)
        if norm != 0:
            l /= norm
            m /= norm
            n /= norm
            return l, m, n


def crossFlat(RAY, Ps, F):
    l, m, n = RAY
    xs, ys, zs = Ps
    A, B, C, D = F
    denom = A * l + B * m + C * n
    if abs(denom) < 1e-12:
        return None
    t = -(A * xs + B * ys + C * zs + D) / denom
    if t < 0:
        return None
    x = xs + l * t
    y = ys + m * t
    z = zs + n * t
    return x, y, z


def insideFlat(R, Pcross):
    x, y, _ = Pcross
    return x * x + y * y <= R * R


def crossCil(RAY, R):
    l, m, n = RAY
    xs, ys, zs = X0, Y0, Z0
    a = l * l + m * m
    b = 2 * (xs * l + ys * m)
    c = xs * xs + ys * ys - R * R
    if a == 0:
        return None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = None
    for ti in (t1, t2):
        if ti > 0:
            if t is None or ti < t:
                t = ti
    if t is None:
        return None
    x = xs + l * t
    y = ys + m * t
    z = zs + n * t
    return t, (x, y, z)


def insideCil(d, Pcross):
    _, _, z = Pcross
    return abs(z) <= d


def sigmaPh(E, Z):
    C_photo = 6.651e-25 * 4 * math.sqrt(2) / (137 ** 4) * (m_e ** 3.5)
    return C_photo * (Z ** 5) / (E ** 3.5)


def sigmaK(E, Z):
    gamma = E / m_e
    sigma = 6.651e-25 * 3 * Z / (8 * gamma) * (
            (1 - (2 * (gamma + 1) / (gamma ** 2))) * math.log(2 * gamma + 1) +
            0.5 + (4 / gamma) - (1 / (2 * (2 * (gamma + 1) ** 2)))
    )
    return sigma


def Eloss(cost_val, E):
    return E - E / (1 + (E / m_e) * (1 - cost_val))


def Interaction(Pcur, l, m, n, L):
    x, y, z = Pcur
    return (x + l * L, y + m * L, z + n * L)


def Lottery(sigma_ph_total, sigma_k_total):
    total = sigma_ph_total + sigma_k_total
    if total == 0:
        return None
    r = random.random()
    if r < sigma_ph_total / total:
        return 'ph'
    else:
        return 'k'


def Length(Sigma_total):
    return -math.log(random.random()) / Sigma_total


def sample_compton_cos(E):
    alpha = E / m_e
    r_e_sq = r_e * r_e

    # Максимальное значение дифференциального сечения (при cos_theta = 1)
    max_dsigma = (r_e_sq / 2) * (1 + alpha + alpha ** 2) / (1 + alpha) ** 2 * (1 + alpha ** 2 / (1 + alpha) ** 2)
    # Упрощенно: max при forward рассеянии
    max_dsigma = r_e_sq * (1 + alpha)  # приближение

    while True:
        cos_theta = random.uniform(-1, 1)
        E_prime = E / (1 + alpha * (1 - cos_theta))
        ratio = E_prime / E

        # Дифференциальное сечение Клейна-Нишины
        dsigma = (r_e_sq / 2) * (ratio ** 2) * (ratio + 1 / ratio - (1 - cos_theta ** 2))

        # Более точная оценка максимума
        if dsigma > max_dsigma:
            max_dsigma = dsigma

        if random.random() < dsigma / max_dsigma:
            return cos_theta


def rotate_dir(l, m, n, cos_theta, phi):
    w = np.array([l, m, n])
    if abs(l) < 0.999:
        u = np.cross([1, 0, 0], w)
    else:
        u = np.cross([0, 1, 0], w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    sin_theta = math.sqrt(1 - cos_theta * cos_theta)
    new_dir = cos_theta * w + sin_theta * math.cos(phi) * u + sin_theta * math.sin(phi) * v
    return new_dir[0], new_dir[1], new_dir[2]


def simulate_part(start_idx, count, source_name):
    lines = get_source_spectrum(source_name)
    E_max = max(e for e, _ in lines)
    E_max_keV = E_max * 1000 * 1.05
    bin_width = E_max_keV / NBINS
    hist = np.zeros(NBINS)

    F_top = (0.0, 0.0, 1.0, -half_height)
    F_bottom = (0.0, 0.0, 1.0, half_height)

    for _ in range(count):
        l, m, n = ray()

        best_t = float('inf')
        entry_point = None

        p_top = crossFlat((l, m, n), (X0, Y0, Z0), F_top)
        if p_top is not None and insideFlat(R, p_top):
            t = math.sqrt((p_top[0] - X0) ** 2 + (p_top[1] - Y0) ** 2 + (p_top[2] - Z0) ** 2)
            if t > 0 and t < best_t:
                best_t = t
                entry_point = p_top

        p_bottom = crossFlat((l, m, n), (X0, Y0, Z0), F_bottom)
        if p_bottom is not None and insideFlat(R, p_bottom):
            t = math.sqrt((p_bottom[0] - X0) ** 2 + (p_bottom[1] - Y0) ** 2 + (p_bottom[2] - Z0) ** 2)
            if t > 0 and t < best_t:
                best_t = t
                entry_point = p_bottom

        cil_res = crossCil((l, m, n), R)
        if cil_res is not None:
            t, p_side = cil_res
            if insideCil(half_height, p_side):
                if t > 0 and t < best_t:
                    best_t = t
                    entry_point = p_side

        if entry_point is None:
            continue

        x, y, z = entry_point
        E_ph = sample_initial_energy(lines)
        total_deposited = 0.0

        while True:
            # Вычисление сечений для текущей энергии фотона
            sigma_ph_Na = sigmaPh(E_ph, 11)
            sigma_ph_I = sigmaPh(E_ph, 53)
            sigma_k_Na = sigmaK(E_ph, 11)
            sigma_k_I = sigmaK(E_ph, 53)

            mu_ph_total = sigma_ph_Na * n_Na + sigma_ph_I * n_I
            mu_k_total = sigma_k_Na * n_Na + sigma_k_I * n_I
            mu_total = mu_ph_total + mu_k_total

            # Если сечение нулевое - фотон не взаимодействует, вылетает
            if mu_total <= 0:
                if total_deposited > 0:
                    deposited_keV = total_deposited * 1000.0
                    bin_idx = min(int(deposited_keV / bin_width), NBINS - 1)
                    hist[bin_idx] += 1
                break

            # Длина свободного пробега до следующего взаимодействия
            s = Length(mu_total)
            x_new, y_new, z_new = Interaction((x, y, z), l, m, n, s)

            # Проверка: не вылетел ли фотон из детектора?
            if not (abs(z_new) <= half_height and x_new * x_new + y_new * y_new <= R * R):
                # Фотон покинул детектор - регистрируем накопленную энергию
                if total_deposited > 0:
                    deposited_keV = total_deposited * 1000.0
                    bin_idx = min(int(deposited_keV / bin_width), NBINS - 1)
                    hist[bin_idx] += 1
                break

            # Определяем тип взаимодействия (фотоэффект или комптон)
            event = Lottery(mu_ph_total, mu_k_total)

            if event == 'ph':
                # Фотоэффект - вся оставшаяся энергия поглощается
                total_deposited += E_ph
                deposited_keV = total_deposited * 1000.0
                bin_idx = min(int(deposited_keV / bin_width), NBINS - 1)
                hist[bin_idx] += 1
                break

            else:  # Комптоновское рассеяние
                # Сэмплируем угол рассеяния по Клейну-Нишине
                cos_theta = sample_compton_cos(E_ph)
                phi = random.uniform(0, 2 * math.pi)

                # Вычисляем энергию, переданную электрону
                alpha = E_ph / m_e
                E_loss = E_ph - E_ph / (1 + alpha * (1 - cos_theta))

                # Добавляем переданную энергию в накопленную
                total_deposited += E_loss

                # Новая энергия фотона после рассеяния
                E_ph_new = E_ph - E_loss
                E_ph = E_ph_new

                # Если фотон стал очень низкоэнергетичным - считаем, что он поглотится
                if E_ph < 0.01:  # порог в keV
                    total_deposited += E_ph
                    deposited_keV = total_deposited * 1000.0
                    bin_idx = min(int(deposited_keV / bin_width), NBINS - 1)
                    hist[bin_idx] += 1
                    break

                # Поворачиваем направление фотона
                l, m, n = rotate_dir(l, m, n, cos_theta, phi)

                # Обновляем позицию (точка взаимодействия)
                x, y, z = x_new, y_new, z_new

                # Продолжаем цикл - фотон может рассеяться снова
                # (без break!)

    return hist, bin_width, E_max_keV


def run_simulation(num_photons, source_name, num_processes=NUM_PROCESSES):
    chunk_size = num_photons // num_processes
    extra = num_photons % num_processes

    with mp.Pool(processes=num_processes) as pool:
        results = []
        start = 0
        for i in range(num_processes):
            cnt = chunk_size + (1 if i < extra else 0)
            results.append(pool.apply_async(simulate_part, (start, cnt, source_name)))
            start += cnt
        pool.close()
        pool.join()

        hists = []
        bin_width = None
        E_max_keV = None
        for r in results:
            hist, bw, emax = r.get()
            hists.append(hist)
            if bin_width is None:
                bin_width = bw
                E_max_keV = emax

        total_hist = np.sum(hists, axis=0)
    return total_hist, bin_width, E_max_keV


def main():
    source = 'Cs137'

    start_time = time.time()
    hist, bin_width, E_max_keV = run_simulation(N, source, NUM_PROCESSES)
    elapsed = time.time() - start_time

    energies_keV = np.linspace(0, E_max_keV, NBINS)

    plt.figure(figsize=(10, 6))

    plt.bar(energies_keV, hist, width=bin_width, align='center', edgecolor='black')

    plt.xlabel('Energy, keV')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title(f'Gamma spectrum of {source} in NaI detector (N={N} photons)')
    plt.grid(alpha=0.3)
    plt.show()

    print(f"Simulated photons: {N}")
    print(f"Total registered events: {np.sum(hist)}")
    print(f"Maximum energy in histogram: {E_max_keV:.0f} keV")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Number of processes used: {NUM_PROCESSES}")


if __name__ == "__main__":
    mp.freeze_support()
    main()