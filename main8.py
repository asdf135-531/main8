import math
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def simulate_chunk(args):
    detector, n_events = args

    local_spectrum = [0] * detector.num_channels

    for _ in range(n_events):
        E = detector.source_energy
        l, m, n = detector.ray()
        P_entry = detector.find_entry_point((l, m, n))
        if P_entry is None:
            continue

        current_point = P_entry

        while E > 0:
            sigma_ph_Na = detector.sigmaPh(E, detector.Z_Na)
            sigma_ph_I = detector.sigmaPh(E, detector.Z_I)
            sigma_k_Na = detector.sigmaK(E, detector.Z_Na)
            sigma_k_I = detector.sigmaK(E, detector.Z_I)

            Sigma_ph, Sigma_k, Sigma_total = detector.Sigma(
                [sigma_ph_Na, sigma_ph_I],
                [sigma_k_Na, sigma_k_I]
            )

            if Sigma_total <= 0:
                break

            L = detector.Length(Sigma_total)
            P_int = detector.Interaction(current_point, l, m, n, L)

            if not detector.insideCil(P_int):
                break

            interaction_type = detector.Lottery(Sigma_ph, Sigma_k, Sigma_total)

            if interaction_type == 'ph':
                channel = int((E - detector.E_min) / detector.Cch)
                if 0 <= channel < detector.num_channels:
                    local_spectrum[channel] += 1
                break

            elif interaction_type == 'k':
                l_new, m_new, n_new = detector.ray()
                cos_theta = detector.cost(l, m, n, l_new, m_new, n_new)
                dE = detector.Eloss(cos_theta, E)

                if dE > 0 and dE <= detector.source_energy:
                    channel = int((dE - detector.E_min) / detector.Cch)
                    if 0 <= channel < detector.num_channels:
                        local_spectrum[channel] += 1

                E -= dE
                l, m, n = l_new, m_new, n_new
                current_point = P_int

    return local_spectrum
class Gamma:

    def __init__(self):
        self.NA = 6.022e23
        self.mc2 = 0.511
        self.Z_Na = 11
        self.Z_I = 53
        self.A_Na = 22.99
        self.A_I = 126.9
        self.rho_NaI = 3.67
        self.M_NaI = 149.89
        n_molecules = (self.rho_NaI / self.M_NaI) * self.NA
        self.N_Na = n_molecules
        self.N_I = n_molecules
        self.R = float(input("Радиус цилиндра R (см): "))
        self.D = float(input("Высота цилиндра D (см): "))
        self.d = self.D / 2
        self.XO = float(input("XO (см): "))
        self.YO = float(input("YO (см): "))
        self.ZO = float(input("ZO (см): "))
        self.N_events = int(input("\nКоличество событий: "))
        self.E_min = None
        self.E_max = None
        self.num_channels = 1024
        self.Cch = None
        self.spectrum = None
        self.source_energy = None
        self.source_name = None
        self._init_planes()

    @staticmethod
    def ray():
        while True:
            l = random.uniform(-1, 1)
            m = random.uniform(-1, 1)
            n = random.uniform(-1, 1)
            length_sq = l ** 2 + m ** 2 + n ** 2
            if length_sq <= 1:
                break
        length = math.sqrt(length_sq)
        return (l / length, m / length, n / length)

    def _init_planes(self):
        P1_top = (0, 0, self.d)
        P2_top = (self.R, 0, self.d)
        P3_top = (0, self.R, self.d)
        self.F_top = self.flateABCD(P1_top, P2_top, P3_top)

        P1_bottom = (0, 0, -self.d)
        P2_bottom = (self.R, 0, -self.d)
        P3_bottom = (0, self.R, -self.d)
        self.F_bottom = self.flateABCD(P1_bottom, P2_bottom, P3_bottom)

        self.Ps = (self.XO, self.YO, self.ZO)

    def flateABCD(self, P1, P2, P3):
        x1, y1, z1 = P1
        x2, y2, z2 = P2
        x3, y3, z3 = P3

        A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        B = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        C = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        D = - (A * x1 + B * y1 + C * z1)
        return A, B, C, D

    def crossFlat(self, RAY, Ps, F):
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

    def insideFlat(self, R_cyl, P_cross):
        x, y, z = P_cross
        r = math.sqrt(x ** 2 + y ** 2)
        return r <= R_cyl

    def crossCil(self, RAY):
        l, m, n = RAY
        x0, y0, z0 = self.XO, self.YO, self.ZO

        a = l ** 2 + m ** 2
        if a < 1e-12:
            return None, None

        b = 2 * (x0 * l + y0 * m)
        c = x0 ** 2 + y0 ** 2 - self.R ** 2

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

    def insideCil(self, P):
        x, y, z = P

        if abs(z) > self.d:
            return False

        if math.sqrt(x ** 2 + y ** 2) > self.R:
            return False
        return True

    def sigmaPh(self, E, Z):
        if E <= 0:
            return 0
        return 6.651e-25 * 4 * math.sqrt(2) * (Z ** 5) / (137 ** 4) * (self.mc2 / E) ** (7 / 2)

    def sigmaK(self, E, Z):
        if E <= 0:
            return 0
        gamma = E / self.mc2
        if gamma <= 0:
            return 0

        term1 = 1 - (2 * (gamma + 1) / gamma ** 2) * math.log(2 * gamma + 1)
        term2 = 1 / 2 + 4 / gamma - 1 / (2 * (2 * gamma + 1) ** 2)

        return 6.651e-25 * (3 * Z) / (8 * gamma) * (term1 + term2)

    def Sigma(self, sigmaPh_values, sigmaK_values):
        Sigma_ph_Na = (5 / 4) * self.N_Na * sigmaPh_values[0]
        Sigma_ph_I = (5 / 4) * self.N_I * sigmaPh_values[1]
        Sigma_ph_total = Sigma_ph_Na + Sigma_ph_I
        Sigma_k_Na = self.NA * (self.Z_Na / self.A_Na) * sigmaK_values[0] * 1e-24
        Sigma_k_I = self.NA * (self.Z_I / self.A_I) * sigmaK_values[1]
        Sigma_k_total = Sigma_k_Na + Sigma_k_I
        Sigma_total = Sigma_ph_total + Sigma_k_total

        return Sigma_ph_total, Sigma_k_total, Sigma_total

    def Length(self, Sigma_total):
        if Sigma_total <= 0:
            return float('inf')
        return - (1.0 / Sigma_total) * math.log(random.random())

    def Interaction(self, P_cross, l, m, n, L):
        x, y, z = P_cross
        x_int = x + l * L
        y_int = y + m * L
        z_int = z + n * L
        return (x_int, y_int, z_int)

    def cost(self, l, m, n, ll, mm, nn):
        return l * ll + m * mm + n * nn

    def Eloss(self, cos_theta, E):
        gamma = E / self.mc2
        E_prime = E / (1 + gamma * (1 - cos_theta))
        return E - E_prime

    def Lottery(self, Sigma_ph, Sigma_k, Sigma_total):
        if Sigma_total <= 0:
            return None

        rand = random.random()
        p_ph = Sigma_ph / Sigma_total

        if rand < p_ph:
            return 'ph'
        else:
            return 'k'

    def find_entry_point(self, ray_direction):
        l, m, n = ray_direction
        t_top, P_top = self.crossFlat(ray_direction, self.Ps, self.F_top)
        hit_top = False
        t_entry = None
        P_entry = None

        if t_top is not None and self.insideFlat(self.R, P_top):
            hit_top = True
            t_entry = t_top
            P_entry = P_top

        t_bottom, P_bottom = self.crossFlat(ray_direction, self.Ps, self.F_bottom)
        hit_bottom = False
        if t_bottom is not None and self.insideFlat(self.R, P_bottom):
            hit_bottom = True
            if not hit_top or t_bottom < t_entry:
                t_entry = t_bottom
                P_entry = P_bottom

        t_cyl, P_cyl = self.crossCil(ray_direction)
        hit_cyl = False
        if t_cyl is not None and self.insideCil(P_cyl):
            hit_cyl = True
            if (not hit_top and not hit_bottom) or \
                    (hit_top and t_cyl < t_entry) or \
                    (hit_bottom and t_cyl < t_entry):
                t_entry = t_cyl
                P_entry = P_cyl

        if not (hit_top or hit_bottom or hit_cyl):
            return None
        return P_entry

    def set_source(self, source_type):
        """
        Установка типа источника и настройка энергетического диапазона
        Доступные источники: Am-241, Cs-137, Mn-54, Zn-65, Na-22, Co-60
        """
        sources = {
            'Am-241': 0.0595,
            'Cs-137': 0.662,
            'Mn-54': 0.835,
        }

        if source_type in sources:
            self.source_energy = sources[source_type]
            self.source_name = source_type
        else:
            print(f"Источник {source_type} не найден. Используется Cs-137.")
            self.source_energy = 0.662
            self.source_name = "Cs-137"

        self.E_min = max(0.01, self.source_energy * 0.1)
        self.E_max = min(1.5, self.source_energy * 1.2)

        if source_type == 'Am-241':
            self.E_min = 0.01
            self.E_max = 0.15

        self.Cch = (self.E_max - self.E_min) / self.num_channels
        self.spectrum = [0] * self.num_channels

    def simulate(self, source_type='Cs-137'):
        self.set_source(source_type)

        n_cores = mp.cpu_count()
        events_per_core = self.N_events // n_cores

        pool = mp.Pool(n_cores)

        tasks = [(self, events_per_core) for _ in range(n_cores)]

        results = pool.map(simulate_chunk, tasks)

        pool.close()
        pool.join()

        for res in results:
            for i in range(self.num_channels):
                self.spectrum[i] += res[i]

        peak_channel = int((self.source_energy - self.E_min) / self.Cch)
        for i in range(peak_channel + 1, self.num_channels):
            self.spectrum[i] = 0

    def plot_spectrum(self):
      plt.figure(figsize=(12, 8))

      if self.source_name == 'Am-241':
          start_channel = 0
      else:
          start_channel = int(0.07 / self.Cch)
          if start_channel < 0:
              start_channel = 0

      energies = [self.E_min + i * self.Cch for i in range(start_channel, self.num_channels)]
      spectrum_cropped = self.spectrum[start_channel:]
      spectrum_cropped = [max(1, x) for x in spectrum_cropped]

      plt.bar(energies, spectrum_cropped, width=self.Cch, align='edge')

      plt.xlabel('Энергия (МэВ)', fontsize=12)
      plt.ylabel('Количество отсчетов', fontsize=12)
      plt.title(f'Спектр {self.source_name} в NaI(Tl)', fontsize=14)

      plt.grid(True, which='both', linestyle='--', alpha=0.4)
      plt.tight_layout()
      plt.show()
if __name__ == "__main__":
    detector = Gamma()
    print("\nДоступные источники: Cs-137, Am-241, Mn-54")
    source = input("Введите тип источника: ") or "Cs-137"
    start_time = time.perf_counter()
    detector.simulate(source_type=source)
    end_time = time.perf_counter()
    print(f"\nВремя расчёта: {end_time - start_time:.4f} сек")
    detector.plot_spectrum()