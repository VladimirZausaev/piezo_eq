import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from anaflow import get_lap_inv
from scipy.special import iv, kv

# ============================================================
# 1) Входные данные
# ============================================================
k = 50e-15  # m^2
phi = 0.20
mu = 10e-3  # Pa*s
ct = 5e-10  # 1/Pa
B = 1.0
h = 10.0  # m

rw = 0.1  # m
re = 300.0  # m

p_i = 250e5  # Pa
b = 150e5  # Pa

a = 2e10  # Pa / (m^3/s)

Cs_list = [0.0, 1e-6, 5e-6, 2e-5]  # m^3/Pa

# ============================================================
# 2) Производные параметры и сетки
# ============================================================
kappa = k / (phi * mu * ct * B)  # m^2/s
P0 = p_i - b
r_eD = re / rw
z_e = np.log(r_eD)

N = 400
z = np.linspace(0.0, z_e, N + 1)
dz = z[1] - z[0]

t_days = np.logspace(-8, 3, 500)
t_sec = t_days * 86400.0
tD = kappa * t_sec / rw ** 2

atm = 101325.0


# ============================================================
# 3) Метод прогонки
# ============================================================
def tdma(A, B, C, F):
    n = len(B)
    alpha = np.zeros(n)
    beta = np.zeros(n)
    x = np.zeros(n)

    denom = B[0]
    alpha[0] = C[0] / denom
    beta[0] = F[0] / denom
    for i in range(1, n):
        denom = B[i] - A[i] * alpha[i - 1]
        alpha[i] = C[i] / denom if i < n - 1 else 0.0
        beta[i] = (F[i] - A[i] * beta[i - 1]) / denom

    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = beta[i] - alpha[i] * x[i + 1]
    return x

# ============================================================
# 4) Численное решение
# ============================================================
def solve_for_Cs(Cs):
    D1 = mu * B / (2 * np.pi * k * h * a)
    D2 = mu * B * Cs * kappa / (2 * np.pi * k * h * rw ** 2)

    P = np.zeros(N + 1)

    profiles = {}
    t_profile_sec = [t * 86400 for t in [0.01, 0.1, 1, 10, 100]]
    for t_sec_val in t_profile_sec:
        profiles[t_sec_val] = None

    PDw = np.zeros_like(tD)
    pw = np.zeros_like(tD)
    qs = np.zeros_like(tD)
    qr = np.zeros_like(tD)

    # Начальные условия
    PDw[0] = 0.0
    pw[0] = p_i
    qs[0] = (p_i - b) / a
    qr[0] = 0.0 if Cs > 0 else qs[0]

    # Основной цикл
    for n in range(1, len(tD)):
        dt_sec = t_sec[n] - t_sec[n - 1]
        dtD = tD[n] - tD[n - 1]
        if dtD <= 0:
            raise ValueError("Non-increasing tD grid.")

        A = np.zeros(N + 1)
        Bm = np.zeros(N + 1)
        C = np.zeros(N + 1)
        F = np.zeros(N + 1)

        # Внутренняя граница (скважина)
        A[0] = 0.0
        Bm[0] = -(1.0 / dz + D1 + D2 / dtD)
        C[0] = 1.0 / dz
        F[0] = -D1 - (D2 / dtD) * P[0]

        # Внутренние узлы
        for i in range(1, N):
            lam = (dtD / dz**2) * np.exp(-2.0 * z[i])
            A[i] = -lam
            Bm[i] = 1.0 + 2.0 * lam
            C[i] = -lam
            F[i] = P[i]

        # Внешняя граница
        A[N] = -1.0
        Bm[N] = 1.0
        C[N] = 0.0
        F[N] = 0.0

        # Решение системы (метод прогонки)
        P_new = tdma(A, Bm, C, F)
        P = P_new

        PDw[n] = P[0]
        pw[n] = p_i - P0 * PDw[n]
        qs[n] = (pw[n] - b) / a

        dPDw_dt = (PDw[n] - PDw[n - 1]) / dt_sec

        # Приток из пласта
        if Cs == 0:
            qr[n] = qs[n]
        else:
            qr_calc = qs[n] - Cs * P0 * dPDw_dt
            qr[n] = 0.0 if n == 1 else max(qr_calc, 0.0)

        # Сохранение профилей давления в заданные моменты
        for t_sec_val in t_profile_sec:
            if profiles[t_sec_val] is None:
                idx = np.argmin(np.abs(t_sec - t_sec_val))
                if n == idx or abs(t_sec[n] - t_sec_val) < 0.1 * (t_sec[n] - t_sec[n - 1]):
                    profiles[t_sec_val] = P.copy()

    return PDw, pw, qs, qr, profiles

# ============================================================
# 5) Аналитика (взял для сравнения)
# ============================================================
def laplace_solution_Cs0(t_sec_array):
    """
    Аналитическое решение через преобразование Лапласа для Cs=0
    Используется как эталон для валидации конечно-разностной схемы
    """
    # Символьные переменные
    s, r = sp.symbols("s r", positive=True, real=True)

    # Параметры
    kappa = k / (phi * mu * ct * B)
    lam = sp.sqrt(s / kappa)
    dp0 = p_i - b

    # Коэффициенты
    alpha = mu * B / (2 * np.pi * k * h * rw * a)

    # Базисная функция (нулевой градиент на re)
    Phi = (sp.besseli(0, lam * r) * sp.besselk(1, lam * re) +
           sp.besselk(0, lam * r) * sp.besseli(1, lam * re))
    Phi_w = Phi.subs(r, rw)
    dPhi_w = lam * (sp.besseli(1, lam * rw) * sp.besselk(1, lam * re) -
                    sp.besselk(1, lam * rw) * sp.besseli(1, lam * re))

    # Депрессия на забое в пространстве Лапласа
    A = -alpha * dp0 / (s * (dPhi_w - alpha * Phi_w))
    Dp_w_lap = sp.simplify(A * Phi_w)

    # Lambdify для численных вычислений
    bessel_map = {"besseli": iv, "besselk": kv}
    Dp_w_fun = sp.lambdify(s, Dp_w_lap, modules=[bessel_map, "numpy"])

    # Инверсия Лапласа (Стефест)
    def inv(fun):
        return get_lap_inv(fun, method="stehfest")

    f_dp = inv(lambda ss: Dp_w_fun(ss))

    # Вычисление для всех времён
    dpw = np.array([np.asarray(f_dp(t)).item() for t in t_sec_array])

    return dpw


# ============================================================
# 6) Расчет
# ============================================================
print("Расчёт конечно-разностным методом...")
results = {}
all_profiles = {}

for Cs in Cs_list:
    print(f"  Cs = {Cs:g} м³/Па...")
    PDw, pw, qs, qr, profiles = solve_for_Cs(Cs)
    results[Cs] = (PDw, pw, qs, qr)
    all_profiles[Cs] = profiles

print("\nРасчёт аналитического решения (Лаплас) для Cs=0...")
dpw_laplace = laplace_solution_Cs0(t_sec)
print("  Готово.")

# ============================================================
# 7) Графики
# ============================================================
plt.style.use('default')

# ---- График 1: Δp_w(t) с аналитическим решением для Cs=0 ----
plt.figure(figsize=(12, 8))

# Конечно-разностные решения для всех Cs
for Cs in Cs_list:
    PDw, pw, qs, qr = results[Cs]
    dpw = P0 * PDw
    if Cs == 0.0:
        # Для Cs=0: сплошная линия + маркеры
        plt.semilogx(t_days, dpw / atm, 'b-', label="FD: C_s=0", linewidth=2.5)
    else:
        plt.semilogx(t_days, dpw / atm, label=f"FD: C_s={Cs:g}", linewidth=2, alpha=0.8)

# Аналитическое решение Лапласа для Cs=0 (пунктир для сравнения)
plt.semilogx(t_days, dpw_laplace / atm, 'r--', label="Laplace: C_s=0 (эталон)",
             linewidth=2.5, alpha=0.9)

plt.xlabel("Время, дни", fontsize=12)
plt.ylabel("Δp_w, атм", fontsize=12)
plt.title("Депрессия на забое: сравнение конечно-разностного и аналитического решений",
          fontsize=14, fontweight='bold')
plt.grid(True, which="both", alpha=0.7, linewidth=0.5)
plt.legend(fontsize=10, loc='best')
plt.xlim(10e-3,10e3)
plt.tight_layout()
# plt.savefig('drawdown_with_laplace.png', dpi=300, bbox_inches='tight')
plt.show()

# ---- Проверка погрешности для Cs=0 ----
PDw_fd, _, _, _ = results[0.0]
dpw_fd = P0 * PDw_fd
error_rel_raw = np.abs(dpw_fd - dpw_laplace) / np.abs(dpw_laplace) * 100
finite_mask = np.isfinite(error_rel_raw)
error_rel = error_rel_raw[finite_mask]
t_days_finite = t_days[finite_mask]

print(f"\nПогрешность конечно-разностного решения (Cs=0) относительно Лапласа:")
print(f"  Средняя относительная погрешность: {np.mean(error_rel):.3f}%")
print(f"  Максимальная относительная погрешность: {np.max(error_rel):.3f}%")
print(f"  Погрешность на ранних временах (t<1 сут): {np.mean(error_rel[t_days_finite < 1]):.3f}%")
# ---- График 2: q_s(t) ----
plt.figure(figsize=(12, 8))
for Cs in Cs_list:
    PDw, pw, qs, qr = results[Cs]
    plt.semilogx(t_days, qs * 86400, label=f"C_s={Cs:g}", linewidth=2)
plt.xlabel("Время, дни", fontsize=12)
plt.ylabel("q_s, м³/сут", fontsize=12)
plt.title("Поверхностный дебит", fontsize=14)
plt.grid(True, which="both", alpha=0.7)
plt.xlim(10e-3,10e3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ---- График 3: q_r(t) ----
plt.figure(figsize=(12, 8))
for Cs in Cs_list:
    PDw, pw, qs, qr = results[Cs]
    plt.semilogx(t_days, qr * 86400, label=f"C_s={Cs:g}", linewidth=2)
plt.xlabel("Время, дни", fontsize=12)
plt.ylabel("q_res, м³/сут", fontsize=12)
plt.title("Приток из пласта", fontsize=14)
plt.grid(True, which="both", alpha=0.7)
plt.legend(fontsize=10)
plt.xlim(10e-3,10e3)
plt.tight_layout()
plt.show()

# ---- График 4: q_s(p_w) ----
plt.figure(figsize=(12, 8))
for Cs in Cs_list:
    PDw, pw, qs, qr = results[Cs]
    plt.plot(pw / atm, qs * 86400, label=f"C_s={Cs:g}", linewidth=2)
plt.xlabel("p_w, атм", fontsize=12)
plt.ylabel("q_s, м³/сут", fontsize=12)
plt.title("Индикаторная диаграмма", fontsize=14)
plt.grid(True, alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# ---- График 5: Профили давления p(r,t) ----
Cs_show = 0.0
PDw, pw, qs, qr = results[Cs_show]
profiles = all_profiles[Cs_show]

r_plot = np.geomspace(rw, re, 200)
z_plot = np.log(r_plot / rw)
t_profile_days = [0.01, 0.1, 1, 10, 100]
t_profile_sec = [t * 86400 for t in t_profile_days]

plt.figure(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(t_profile_days)))

for idx, t_sec_val in enumerate(t_profile_sec):
    P_prof = profiles[t_sec_val]
    if P_prof is not None:
        P_interp = np.interp(z_plot, z, P_prof)
        p_prof = p_i - P0 * P_interp
        plt.semilogx(r_plot, p_prof / atm, label=f"t = {t_profile_days[idx]:g} сут",
                     color=colors[idx], linewidth=2.5)

plt.axvline(rw, color='gray', linestyle='--', alpha=0.7, label='r_w')
plt.axvline(re, color='gray', linestyle=':', alpha=0.7, label='r_e')
plt.axhline(p_i / atm, color='red', linestyle='-', alpha=0.4, label='p_i')
plt.axhline(b / atm, color='blue', linestyle='-', alpha=0.4, label='b')

plt.xlabel("Радиус r, м", fontsize=12)
plt.ylabel("Давление p(r,t), атм", fontsize=12)
plt.title(f"Профили давления (C_s = {Cs_show:g} м³/Па)", fontsize=14)
plt.grid(True, which="both", alpha=0.7)
plt.legend(fontsize=10, loc='best')
plt.xlim([rw, re])
plt.tight_layout()
# plt.savefig('pressure_profiles.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Расчёт завершён!")
print("=" * 70)