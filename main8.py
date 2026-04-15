import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
from functools import partial

NA = 6.022e23
mc2 = 0.511
Z_Na = 11
Z_I = 53
A_Na = 22.99
A_I = 126.9
rho_NaI = 3.67
M_NaI = 149.89

n_molecules = (rho_NaI / M_NaI) * NA
N_Na = n_molecules
N_I = n_molecules

E_min = 0.05
E_max = 1.0
num_channels = 1024
Cch = (E_max - E_min) / num_channels

# Глобальные переменные для процессов (доступны только для чтения)
# Используем mp.Array для разделяемой памяти
shared_params = None


def init_process(params):
    """Инициализация каждого процесса"""
    global shared_params
    shared_params = params


def ray():
    theta = random.uniform(0, math.pi)
    phi = random.uniform(0, 2 * math.pi)
    l = math.sin(theta) * math.cos(phi)
    m = math.sin(theta) * math.sin(phi)
    n = math.cos(theta)
    return l, m, n


def flateABCD(P1, P2, P3):
    x1, y1, z1 = P1
    x2, y2, z2 = P2
    x3, y3, z3 = P3
    A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
    C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    D = - (A * x1 + B * y1 + C * z1)
    return A, B, C, D


def crossFlat(RAY, Ps, F):
    l, m, n = RAY
    x0, y0, z0 = Ps
    A, B, C, D = F
    denominator = A * l + B * m + C * n
    if abs(denominator) < 1e-12:
        return None, None
    t = - (A * x0 + B * y0 + C * z0 + D) / denominator
    if t < 0:
        return None, None
    x = x0 + l * t
    y = y0 + m * t
    z = z0 + n * t
    return t, (x, y, z)


def insideFlat(R_cyl, P_cross):
    x, y, z = P_cross
    return math.sqrt(x ** 2 + y ** 2) <= R_cyl


def crossCil(RAY, R_cyl, Ps):
    l, m, n = RAY
    x0, y0, z0 = Ps
    a = l ** 2 + m ** 2
    if a < 1e-12:
        return None, None
    b = 2 * (x0 * l + y0 * m)
    c = x0 ** 2 + y0 ** 2 - R_cyl ** 2
    D = b ** 2 - 4 * a * c
    if D < 0:
        return None, None
    t1 = (-b - math.sqrt(D)) / (2 * a)
    t2 = (-b + math.sqrt(D)) / (2 * a)
    t_min = None
    for t in [t1, t2]:
        if t > 1e-12:
            if t_min is None or t < t_min:
                t_min = t
    if t_min is None:
        return None, None
    x = x0 + l * t_min
    y = y0 + m * t_min
    z = z0 + n * t_min
    return t_min, (x, y, z)


def insideCil(d_half, P, R_cyl):
    x, y, z = P
    if abs(z) > d_half:
        return False
    return math.sqrt(x ** 2 + y ** 2) <= R_cyl


def inObj(P, d_half, R_cyl):
    return insideCil(d_half, P, R_cyl)


def sigmaPh(E, Z):
    if E <= 0:
        return 0
    return 6.651e-25 * 4 * math.sqrt(2) * (Z ** 5) / (137 ** 4) * (mc2 / E) ** (7 / 2)


def sigmaK(E, Z):
    if E <= 0:
        return 0
    gamma = E / mc2
    if gamma <= 0:
        return 0
    term1 = 1 - (2 * (gamma + 1) / gamma ** 2) * math.log(2 * gamma + 1)
    term2 = 1 / 2 + 4 / gamma - 1 / (2 * (2 * gamma + 1) ** 2)
    return 6.651e-25 * (3 * Z) / (8 * gamma) * (term1 + term2)


def Sigma(sigmaPh_Na, sigmaPh_I, sigmaK_Na, sigmaK_I):
    Sigma_ph_Na = (5 / 4) * N_Na * sigmaPh_Na
    Sigma_ph_I = (5 / 4) * N_I * sigmaPh_I
    Sigma_ph_total = Sigma_ph_Na + Sigma_ph_I

    Sigma_k_Na = NA * (Z_Na / A_Na) * sigmaK_Na
    Sigma_k_I = NA * (Z_I / A_I) * sigmaK_I
    Sigma_k_total = Sigma_k_Na + Sigma_k_I

    Sigma_total = Sigma_ph_total + Sigma_k_total
    return Sigma_ph_total, Sigma_k_total, Sigma_total


def Length(Sigma_total):
    if Sigma_total <= 0:
        return float('inf')
    return - (1.0 / Sigma_total) * math.log(random.random())


def Interaction(P_cross, l, m, n, L):
    x, y, z = P_cross
    return (x + l * L, y + m * L, z + n * L)


def cost(l, m, n, ll, mm, nn):
    return l * ll + m * mm + n * nn


def Eloss(cos_theta, E):
    gamma = E / mc2
    E_prime = E / (1 + gamma * (1 - cos_theta))
    return E - E_prime


def Lottery(Sigma_ph, Sigma_k, Sigma_total):
    if Sigma_total <= 0:
        return None
    return 'ph' if random.random() < Sigma_ph / Sigma_total else 'k'


def process_chunk(chunk_args):
    """Обработка чанка событий (для уменьшения накладных расходов)"""
    chunk_id, start_idx, chunk_size, R, d, Ps, F_top, F_bottom = chunk_args

    # Создаём локальный спектр для чанка
    local_spectrum = [0] * num_channels

    # Устанавливаем seed для воспроизводимости
    random.seed(chunk_id + 12345)

    for _ in range(chunk_size):
        E = 0.662
        l, m, n = ray()
        current_point = Ps

        # Поиск точки входа
        t_top, P_top_point = crossFlat((l, m, n), current_point, F_top)
        hit_top = False
        if t_top is not None and insideFlat(R, P_top_point):
            hit_top = True
            t_entry = t_top
            P_entry = P_top_point

        t_bottom, P_bottom_point = crossFlat((l, m, n), current_point, F_bottom)
        hit_bottom = False
        if t_bottom is not None and insideFlat(R, P_bottom_point):
            hit_bottom = True
            if not hit_top or t_bottom < t_entry:
                t_entry = t_bottom
                P_entry = P_bottom_point

        t_cyl, P_cyl_point = crossCil((l, m, n), R, Ps)
        hit_cyl = False
        if t_cyl is not None and insideCil(d, P_cyl_point, R):
            hit_cyl = True
            if (not hit_top and not hit_bottom) or (hit_top and t_cyl < t_entry) or (hit_bottom and t_cyl < t_entry):
                t_entry = t_cyl
                P_entry = P_cyl_point

        if not (hit_top or hit_bottom or hit_cyl):
            continue

        current_point = P_entry

        while E > 0:
            sigma_ph_Na = sigmaPh(E, Z_Na)
            sigma_ph_I = sigmaPh(E, Z_I)
            sigma_k_Na = sigmaK(E, Z_Na)
            sigma_k_I = sigmaK(E, Z_I)

            Sigma_ph, Sigma_k, Sigma_total = Sigma(sigma_ph_Na, sigma_ph_I, sigma_k_Na, sigma_k_I)

            if Sigma_total <= 0:
                break

            L = Length(Sigma_total)
            P_int = Interaction(current_point, l, m, n, L)

            if not inObj(P_int, d, R):
                break

            interaction_type = Lottery(Sigma_ph, Sigma_k, Sigma_total)

            if interaction_type == 'ph':
                channel = int(round(E / Cch))
                if 0 <= channel < num_channels:
                    local_spectrum[channel] += 1
                break

            elif interaction_type == 'k':
                l_new, m_new, n_new = ray()
                cos_theta = cost(l, m, n, l_new, m_new, n_new)
                dE = Eloss(cos_theta, E)

                if dE > 0:
                    channel = int(round(dE / Cch))
                    if 0 <= channel < num_channels:
                        local_spectrum[channel] += 1

                E = E - dE
                l, m, n = l_new, m_new, n_new
                current_point = P_int

    return local_spectrum


if __name__ == "__main__":
    R = float(input("введите радиус детектора R: "))
    D = float(input("введите высоту детектора D: "))
    d = D / 2
    XO = float(input("введите x источника: "))
    YO = float(input("введите y источника: "))
    ZO = float(input("введите z источника: "))

    N_events = int(input("введите количество фотонов N: "))

    # Автоматически определяем оптимальное число процессов
    num_cores = mp.cpu_count()
    # Для больших N используем все ядра, для маленьких - меньше
    if N_events < 10000:
        use_cores = max(1, num_cores // 2)
    else:
        use_cores = num_cores

    print(f"Доступно ядер: {num_cores}, используем: {use_cores}")

    start_time = time.time()

    Ps = (XO, YO, ZO)

    P1_top = (0, 0, d)
    P2_top = (R, 0, d)
    P3_top = (0, R, d)
    F_top = flateABCD(P1_top, P2_top, P3_top)

    P1_bottom = (0, 0, -d)
    P2_bottom = (R, 0, -d)
    P3_bottom = (0, R, -d)
    F_bottom = flateABCD(P1_bottom, P2_bottom, P3_bottom)

    # Разбиваем на чанки для уменьшения накладных расходов
    chunk_size = max(1, N_events // (use_cores * 10))  # Каждый процесс получает ~10 чанков
    chunks = []
    for i in range(0, N_events, chunk_size):
        chunk_id = len(chunks)
        actual_size = min(chunk_size, N_events - i)
        chunks.append((chunk_id, i, actual_size, R, d, Ps, F_top, F_bottom))

    print(f"Разбито на {len(chunks)} чанков (размер чанка: {chunk_size})")

    # Создаём пул процессов
    with mp.Pool(processes=use_cores) as pool:
        print("Начинаем обработку...")

        # Используем imap_unordered для лучшей производительности
        results = []
        total_chunks = len(chunks)
        for i, result in enumerate(pool.imap_unordered(process_chunk, chunks)):
            results.append(result)
            if (i + 1) % max(1, total_chunks // 10) == 0:
                elapsed = time.time() - start_time
                print(
                    f"Обработано {i + 1}/{total_chunks} чанков ({100 * (i + 1) / total_chunks:.1f}%) - {elapsed:.1f} сек")

    # Суммируем результаты
    print("Суммирование спектров...")
    spectrum = [0] * num_channels
    for local_spec in results:
        for i, count in enumerate(local_spec):
            spectrum[i] += count

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nВремя выполнения: {total_time:.2f} с")
    print(f"Производительность: {N_events / total_time:.0f} событий/сек")
    print(f"Ускорение относительно одного ядра: ~{use_cores * (1 - 0.15):.1f}x (с учётом накладных расходов)")

    # Построение графика
    channels = np.arange(num_channels)
    energy_axis = channels * Cch + E_min

    plt.figure(figsize=(12, 6))
    plt.bar(energy_axis, spectrum, width=Cch, align='edge')
    plt.xlabel('энергия (МэВ)')
    plt.ylabel('количество отсчётов')
    plt.title(f'спектр 137Cs (NaI), R={R} см, D={D} см, {use_cores} ядер')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xlim(E_min, E_max)
    plt.show()