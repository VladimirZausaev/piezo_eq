# ============================================================
# Нестационарная радиальная фильтрация к скважине (конечный радиус)
# - уравнение пьезопроводности
# - внешняя граница: непроницаемая (dp/dr|re = 0)
# - внутренняя граница: линейная связь q~p_w + послеприток (wellbore storage Cs)
# - решение в Лапласе без асимптотик (I,K Бессели)
# - численное ОПЛ: anaflow.get_lap_inv, метод Talbot (быстро, векторно)
#
# Графики:
#  1) Δp_w(t)
#  2) dΔp_w/d ln t
#  3) Δp(r,t) профили
#  4) q(t) и q_s(t)
#  5) q(Δp_w)  <-- как вы просили
# ============================================================

from math import pi
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from anaflow import get_lap_inv
from scipy.special import iv, kv

# ============================================================
# 0) ВХОДНЫЕ ДАННЫЕ (поменяйте под себя)
# ============================================================
# Пласт / флюид
k   = 50e-15          # м^2
phi = 0.20            # -
mu  = 10e-3           # Па*с
ct  = 5e-10           # 1/Па
B   = 1.0             # -
h   = 10.0            # м

rw  = 0.1             # м
re  = 300.0           # м

# Давления
p_i = 250e5           # Па
b   = 150e5           # Па (давление отсечки)

# Линейная связь дебит–давление: q(t) = (p_w(t) - b)/a
# a [Па/(м^3/с)]
a = 2.0e10            # пример, замените

# Послеприток Cs [м^3/Па]
Cs_list = [0.0, 1e-6, 5e-6, 2e-5]

# Время (сутки -> секунды)
t_days = np.logspace(-3, 3, 220)   # 0.001 .. 1000 суток
t_sec  = t_days * 86400.0

# Радиусы для профилей
r_grid = np.geomspace(rw, re, 140)

# Времена для профилей
t_show_days = [0.01, 0.1, 1, 10, 100]
t_show_sec  = [x * 86400.0 for x in t_show_days]

# Метод ОПЛ в рамках anaflow
LAP_METHOD = "stehfest"

# ============================================================
# 1) Символьное решение в Лапласе
# ============================================================
s, r = sp.symbols('s r', positive=True, real=True)
Cs_sym = sp.Symbol('Cs', nonnegative=True, real=True)

kappa = k / (phi * mu * ct * B)     # коэффициент пьезопроводности
lam = sp.sqrt(s / kappa)

alpha = mu * B / (2*pi*k*h*rw*a)
beta  = mu * B * Cs_sym / (2*pi*k*h*rw)
dp0   = (p_i - b)

# Устойчивая комбинация, автоматически удовлетворяющая dp/dr|re = 0:
# Phi(r) = I0(lam r) K1(lam re) + K0(lam r) I1(lam re)
Phi_r = sp.besseli(0, lam*r) * sp.besselk(1, lam*re) + sp.besselk(0, lam*r) * sp.besseli(1, lam*re)
Phi_w = Phi_r.subs(r, rw)

# dPhi/dr на rw: d/dr I0 = lam I1, d/dr K0 = -lam K1
dPhi_w = lam * (sp.besseli(1, lam*rw) * sp.besselk(1, lam*re) - sp.besselk(1, lam*rw) * sp.besseli(1, lam*re))

# Решение для депрессии Δp = p_i - p:
# (∂Δp/∂r)|rw = (alpha + beta*s)*Δp_w - alpha*(p_i-b)/s
A = -alpha * dp0 / (s * (dPhi_w - (alpha + beta*s)*Phi_w))
Dp_lap   = sp.simplify(A * Phi_r)           # \bar{Δp}(r,s)
Dp_w_lap = sp.simplify(Dp_lap.subs(r, rw))  # \bar{Δp}_w(s)

# Дебиты:
# q(t) = (p_w-b)/a = (dp0 - Δp_w)/a
# => \bar q(s) = (dp0/s - \barΔp_w)/a
q_lap  = (dp0/s - Dp_w_lap) / a
# q_s - q = Cs dp_w/dt = -Cs dΔp_w/dt
# => \bar q_s(s) = \bar q(s) - Cs*s*\barΔp_w(s)
qs_lap = sp.simplify(q_lap - Cs_sym*s*Dp_w_lap)

# ============================================================
# 2) lambdify (векторно, через scipy.special)
# ============================================================
bessel_map = {"besseli": iv, "besselk": kv}

Dp_lap_fun   = sp.lambdify((s, r, Cs_sym), Dp_lap,   modules=[bessel_map, "numpy"])
Dp_w_lap_fun = sp.lambdify((s, Cs_sym),    Dp_w_lap, modules=[bessel_map, "numpy"])
q_lap_fun    = sp.lambdify((s, Cs_sym),    q_lap,    modules=[bessel_map, "numpy"])
qs_lap_fun   = sp.lambdify((s, Cs_sym),    qs_lap,   modules=[bessel_map, "numpy"])

# ============================================================
# 3) Обратное преобразование Лапласа (anaflow, Talbot)
# ============================================================
def inv_Dp(r_val, Cs_val):
    return get_lap_inv(lambda ss: Dp_lap_fun(ss, r_val, Cs_val), method=LAP_METHOD)

def inv_Dp_w(Cs_val):
    return get_lap_inv(lambda ss: Dp_w_lap_fun(ss, Cs_val), method=LAP_METHOD)

def inv_q(Cs_val):
    return get_lap_inv(lambda ss: q_lap_fun(ss, Cs_val), method=LAP_METHOD)

def inv_qs(Cs_val):
    return get_lap_inv(lambda ss: qs_lap_fun(ss, Cs_val), method=LAP_METHOD)

# ============================================================
# 4) Δp_w(t)
# ============================================================
DpW = {}

plt.figure(figsize=(9, 6))
for Cs_val in Cs_list:
    f = inv_Dp_w(Cs_val)
    DpW[Cs_val] = np.array([np.asarray(f(tt)).item() for tt in t_sec])  # Па
    plt.semilogx(t_days, DpW[Cs_val] / 1e5, label=f"C_s = {Cs_val:g} м³/Па")

plt.grid(True, which="both")
plt.xlabel("Время t, сут")
plt.ylabel("Депрессия на забое Δp_w, бар")
plt.title("Депрессия на забое во времени (учёт послепритока)")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 6) Профили Δp(r,t) для одного Cs
# ============================================================
Cs_show = Cs_list[min(1, len(Cs_list)-1)]

plt.figure(figsize=(9, 6))
for tt, tday in zip(t_show_sec, t_show_days):
    prof = np.array([np.asarray(inv_Dp(rr, Cs_show)(tt)).item() for rr in r_grid])
    plt.semilogx(r_grid, prof / 1e5, label=f"t = {tday:g} сут")

plt.grid(True, which="both")
plt.xlabel("Радиус r, м")
plt.ylabel("Депрессия Δp(r,t), бар")
plt.title(f"Пространственные профили депрессии (C_s = {Cs_show:g} м³/Па)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 6b) Пространственные профили ДАВЛЕНИЯ p(r,t) = p_i - Δp(r,t)
# ============================================================

plt.figure(figsize=(9, 6))

for tt, tday in zip(t_show_sec, t_show_days):
    # сначала считаем депрессию
    dp_prof = np.array([
        np.asarray(inv_Dp(rr, Cs_show)(tt)).item()
        for rr in r_grid
    ])  # Па

    # давление
    p_prof = p_i - dp_prof  # Па

    plt.semilogx(
        r_grid,
        p_prof / 1e5,        # бар
        label=f"t = {tday:g} сут"
    )

plt.grid(True, which="both")
plt.xlabel("Радиус r, м")
plt.ylabel("Давление p(r,t), бар")
plt.title(
    "Пространственные профили давления\n"
    r"$p(r,t) = p_i - \Delta p(r,t)$"
    f"\n(C_s = {Cs_show:g} м³/Па)"
)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7) q(t), q_s(t) и баланс
# ============================================================
rates_cache = {}

plt.figure(figsize=(9, 6))
for Cs_val in Cs_list:
    fq  = inv_q(Cs_val)
    fqs = inv_qs(Cs_val)

    q_t  = np.array([np.asarray(fq(tt)).item()  for tt in t_sec])   # м3/с
    qs_t = np.array([np.asarray(fqs(tt)).item() for tt in t_sec])   # м3/с

    rates_cache[Cs_val] = (q_t, qs_t)

    plt.semilogx(t_days, q_t*86400,  label=f"q, C_s={Cs_val:g}")
    plt.semilogx(t_days, qs_t*86400, linestyle="--", label=f"q_s, C_s={Cs_val:g}")

plt.grid(True, which="both")
plt.xlabel("Время t, сут")
plt.ylabel("Дебит, м³/сут")
plt.title("Дебиты: приток из пласта q(t) и поверхностный дебит q_s(t)")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

print("\nПроверка баланса: (q_s - q) + C_s * dΔp_w/dt  ≈ 0")
for Cs_val in Cs_list:
    q_t, qs_t = rates_cache[Cs_val]
    y = DpW[Cs_val]
    dy_dt = np.gradient(y, t_sec)

    resid = (qs_t - q_t) + Cs_val * dy_dt
    # если где-то появятся nan/inf (редко, но бывает), просто игнорируем их при оценке
    resid = resid[np.isfinite(resid)]
    max_res = np.max(np.abs(resid)) if resid.size else np.nan
    print(f"  C_s={Cs_val:g}: max|невязка| = {max_res:.3e} м³/с")

# ============================================================
# 8) График, который вы просили: q от Δp_w (не q_s!)
# ============================================================
plt.figure(figsize=(9, 6))
for Cs_val in Cs_list:
    dpw_bar = DpW[Cs_val] / 1e5     # бар
    q_t, _ = rates_cache[Cs_val]
    q_day = q_t * 86400.0           # м3/сут

    plt.plot(dpw_bar, q_day, label=f"C_s = {Cs_val:g} м³/Па")

plt.grid(True)
plt.xlabel("Депрессия на забое Δp_w, бар")
plt.ylabel("Приток из пласта q, м³/сут")
plt.title("Зависимость притока из пласта от депрессии (q от Δp_w)")
plt.legend()
plt.tight_layout()
plt.show()


