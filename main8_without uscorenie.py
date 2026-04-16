import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

NA = 6.022e23
mc2 = 0.511  # МэВ
Z_Na = 11
Z_I = 53
A_Na = 22.99
A_I = 126.9
rho_NaI = 3.67  # г/см³
M_NaI = 149.89  # г/моль

n_molecules = (rho_NaI / M_NaI) * NA
N_Na = n_molecules
N_I = n_molecules

E_min = 0.05   # МэВ
E_max = 1.0    # МэВ
num_channels = 1024
Cch = (E_max - E_min) / num_channels


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


def generate_compton_angle(E):
    gamma = E / mc2
    while True:
        cos_theta = random.uniform(-1, 1)
        E_prime = E / (1 + gamma * (1 - cos_theta))
        ratio = E_prime / E
        dsigma = ratio**2 * (ratio + 1/ratio - (1 - cos_theta**2))
        max_dsigma = 1.0
        if random.random() < dsigma / max_dsigma:
            return cos_theta


def rotate_direction(l, m, n, cos_theta, phi):
    sin_theta = math.sqrt(max(0, 1 - cos_theta**2))
    
    if abs(l) < 0.99999:
        ux, uy, uz = 1, 0, 0
    else:
        ux, uy, uz = 0, 1, 0
    
    perp_x = uy * n - uz * m
    perp_y = uz * l - ux * n
    perp_z = ux * m - uy * l
    norm = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
    if norm > 0:
        perp_x /= norm
        perp_y /= norm
        perp_z /= norm
    
    perp2_x = m * perp_z - n * perp_y
    perp2_y = n * perp_x - l * perp_z
    perp2_z = l * perp_y - m * perp_x
    
    l_new = l * cos_theta + perp_x * sin_theta * math.cos(phi) + perp2_x * sin_theta * math.sin(phi)
    m_new = m * cos_theta + perp_y * sin_theta * math.cos(phi) + perp2_y * sin_theta * math.sin(phi)
    n_new = n * cos_theta + perp_z * sin_theta * math.cos(phi) + perp2_z * sin_theta * math.sin(phi)
    
    norm_new = math.sqrt(l_new**2 + m_new**2 + n_new**2)
    if norm_new > 0:
        l_new /= norm_new
        m_new /= norm_new
        n_new /= norm_new
    
    return l_new, m_new, n_new


def Sigma(sigmaPh_Na, sigmaPh_I, sigmaK_Na, sigmaK_I):
    Sigma_ph_Na = N_Na * sigmaPh_Na
    Sigma_ph_I = N_I * sigmaPh_I
    Sigma_ph_total = Sigma_ph_Na + Sigma_ph_I

    Sigma_k_Na = N_Na * sigmaK_Na
    Sigma_k_I = N_I * sigmaK_I
    Sigma_k_total = Sigma_k_Na + Sigma_k_I

    Sigma_total = Sigma_ph_total + Sigma_k_total
    return Sigma_ph_total, Sigma_k_total, Sigma_total


def Length(Sigma_total):
    if Sigma_total <= 0:
        return float('inf')
    return -math.log(random.random()) / Sigma_total


def Interaction(P_cross, l, m, n, L):
    x, y, z = P_cross
    return (x + l * L, y + m * L, z + n * L)


def Lottery(Sigma_ph, Sigma_k, Sigma_total):
    if Sigma_total <= 0:
        return None
    return 'ph' if random.random() < Sigma_ph / Sigma_total else 'k'


if __name__ == "__main__":
    print("\n" + "="*50)
    print("МОДЕЛИРОВАНИЕ СПЕКТРА Cs-137 В NaI(Tl)")
    print("="*50)
    
    R = float(input("введите радиус детектора R (см): "))
    D = float(input("введите высоту детектора D (см): "))
    d = D / 2
    XO = float(input("введите x источника (см): "))
    YO = float(input("введите y источника (см): "))
    ZO = float(input("введите z источника (см): "))

    N_events = int(input("введите количество фотонов N: "))

    start_time = time.time()

    spectrum = [0] * num_channels
    Ps = (XO, YO, ZO)

    P1_top = (0, 0, d)
    P2_top = (R, 0, d)
    P3_top = (0, R, d)
    F_top = flateABCD(P1_top, P2_top, P3_top)

    P1_bottom = (0, 0, -d)
    P2_bottom = (R, 0, -d)
    P3_bottom = (0, R, -d)
    F_bottom = flateABCD(P1_bottom, P2_bottom, P3_bottom)

    events_in_detector = 0

    for event in range(N_events):
        if event % 10000 == 0 and event > 0:
            elapsed = time.time() - start_time
            print(f"Обработано {event} событий за {elapsed:.1f} с")

        E = 0.662
        l, m, n = ray()
        current_point = Ps
        deposited = 0.0
        n_interactions = 0

        # Поиск точки входа
        t_top, P_top = crossFlat((l, m, n), current_point, F_top)
        hit_top = t_top is not None and insideFlat(R, P_top)
        
        t_bottom, P_bottom = crossFlat((l, m, n), current_point, F_bottom)
        hit_bottom = t_bottom is not None and insideFlat(R, P_bottom)
        
        t_cyl, P_cyl = crossCil((l, m, n), R, Ps)
        hit_cyl = t_cyl is not None and insideCil(d, P_cyl, R)

        entries = []
        if hit_top:
            entries.append((t_top, P_top))
        if hit_bottom:
            entries.append((t_bottom, P_bottom))
        if hit_cyl:
            entries.append((t_cyl, P_cyl))
        
        if not entries:
            continue
        
        events_in_detector += 1
        t_entry, P_entry = min(entries, key=lambda x: x[0])
        current_point = P_entry

        # Моделирование взаимодействий
        while E > 0.01:
            sigma_ph_Na = sigmaPh(E, Z_Na)
            sigma_ph_I = sigmaPh(E, Z_I)
            sigma_k_Na = sigmaK(E, Z_Na)
            sigma_k_I = sigmaK(E, Z_I)

            Sigma_ph, Sigma_k, Sigma_total = Sigma(sigma_ph_Na, sigma_ph_I, sigma_k_Na, sigma_k_I)

            if Sigma_total <= 0:
                deposited += E
                break

            L = Length(Sigma_total)
            P_int = Interaction(current_point, l, m, n, L)

            if not inObj(P_int, d, R):
                break

            interaction_type = Lottery(Sigma_ph, Sigma_k, Sigma_total)
            n_interactions += 1

            if interaction_type == 'ph':
                deposited += E
                break
            else:  # Комптон
                cos_theta = generate_compton_angle(E)
                if cos_theta is None:
                    deposited += E
                    break
                
                gamma = E / mc2
                E_scattered = E / (1 + gamma * (1 - cos_theta))
                dE = E - E_scattered
                
                # РЕЗКИЙ КОМПТОНОВСКИЙ КРАЙ + ВЕРТИКАЛЬНЫЙ ФОТОПИК
                if n_interactions == 1:
                    # Первое рассеяние - комптоновское плато
                    deposited += dE
                    break  # Фотон улетает
                else:
                    # Многократные рассеяния - вся энергия в фотопик
                    deposited += E
                    break

        # РЕГИСТРАЦИЯ БЕЗ ГАУССОВА РАЗМЫТИЯ
        if deposited > 0:
            # Прямое преобразование энергии в канал
            channel = int(round((deposited - E_min) / Cch))
            
            if channel < 0:
                channel = 0
            elif channel >= num_channels:
                channel = num_channels - 1
            
            spectrum[channel] += 1

    end_time = time.time()
    
    # Построение спектра
    channels = np.arange(num_channels)
    energy_axis = channels * Cch + E_min

    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(energy_axis, spectrum, width=Cch, align='edge', alpha=0.7, color='blue')
    plt.xlabel('Энергия (МэВ)')
    plt.ylabel('Количество отсчётов')
    plt.title(f'Спектр 137Cs в NaI(Tl) - Вертикальный фотопик, резкий комптоновский край')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xlim(E_min, E_max)
    
    E_gamma = 0.662
    E_compton_edge = E_gamma / (1 + mc2/(2*E_gamma))
    E_backscatter = E_gamma / (1 + 2*E_gamma/mc2)
    
    plt.axvline(x=E_backscatter, color='green', linestyle='--', alpha=0.7, label=f'Пик обратного рассеяния ({E_backscatter:.3f} МэВ)')
    plt.axvline(x=E_compton_edge, color='orange', linestyle='--', alpha=0.7, linewidth=2, label=f'Комптоновский край ({E_compton_edge:.3f} МэВ)')
    plt.axvline(x=E_gamma, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Фотопик ({E_gamma:.3f} МэВ)')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.bar(energy_axis, spectrum, width=Cch, align='edge', alpha=0.7, color='blue')
    plt.xlabel('Энергия (МэВ)')
    plt.ylabel('Количество отсчётов')
    plt.title('Увеличенный вид (линейная шкала)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.4, 0.7)
    plt.ylim(0, max(spectrum) * 1.1)
    plt.axvline(x=E_compton_edge, color='orange', linestyle='--', linewidth=2)
    plt.axvline(x=E_gamma, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nВремя моделирования: {end_time - start_time:.2f} с")
    print(f"Попало в детектор: {events_in_detector} ({100*events_in_detector/N_events:.1f}%)")
    
    # Находим фотопик
    photopeak_start = int((0.65 - E_min) / Cch)
    photopeak_end = int((0.68 - E_min) / Cch)
    if photopeak_start < 0:
        photopeak_start = 0
    if photopeak_end >= num_channels:
        photopeak_end = num_channels - 1
    
    photopeak_energy = energy_axis[photopeak_start:photopeak_end][np.argmax(spectrum[photopeak_start:photopeak_end])]
    photopeak_counts = max(spectrum[photopeak_start:photopeak_end])
    
    print(f"\nФотопик: энергия = {photopeak_energy:.3f} МэВ, счёт = {photopeak_counts}")
    print(f"Теоретический фотопик: 0.662 МэВ")
    print(f"Отклонение: {(photopeak_energy - 0.662)*1000:.1f} кэВ")
