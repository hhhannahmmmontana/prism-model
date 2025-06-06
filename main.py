# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import sys


def validate_input(n_550, angle_inc_deg, A_cauchy):
    """Проверка корректности входных данных"""
    errors = []

    if not 1.0 < n_550 < 3.0:
        errors.append(f"Показатель преломления n_550 = {n_550} нереалистичен. Должен быть в диапазоне 1.0 < n < 3.0")

    if not 0 <= angle_inc_deg < 90:
        errors.append(f"Угол падения {angle_inc_deg}° вне допустимого диапазона [0°, 90°)")

    if not 1.0 <= A_cauchy <= 2.0:
        errors.append(f"Параметр Коши A = {A_cauchy} выходит за разумные пределы (1.0-2.0)")

    lambda0 = 550e-9
    B_cauchy = (n_550 - A_cauchy) * (lambda0 ** 2)
    if B_cauchy <= 0:
        errors.append(
            f"Некорректная дисперсия: параметр B = {B_cauchy:.2e} ≤ 0. "
            f"Убедитесь, что A < n_550 (A = {A_cauchy}, n_550 = {n_550})"
        )

    if errors:
        print("Ошибки во входных данных:", file=sys.stderr)
        for error in errors:
            print(f"• {error}", file=sys.stderr)
        sys.exit(1)


# Запрос и проверка входных данных
try:
    n_550 = float(input("Введите показатель преломления призмы при 550 нм: "))
    angle_inc_deg = float(input("Введите угол падения белого света на призму (в градусах): "))
    A_cauchy = float(input("Введите параметр A для уравнения Коши (зависит от материала): "))

    validate_input(n_550, angle_inc_deg, A_cauchy)

except ValueError:
    print("Ошибка: все параметры должны быть числами!", file=sys.stderr)
    sys.exit(1)

# Задаём исходные параметры
wavelengths_nm = np.linspace(400, 700, 10000)
norm = Normalize(vmin=400, vmax=700)
cmap = plt.cm.rainbow
angle_inc_rad = np.radians(angle_inc_deg)
h = 5.0

# Модель дисперсии: Коши
λ0 = 550e-9
B_cauchy = (n_550 - A_cauchy) * λ0 ** 2


def n_of_lambda(lambda_nm):
    λ_m = lambda_nm * 1e-9
    return A_cauchy + B_cauchy / (λ_m ** 2)


n_values = n_of_lambda(wavelengths_nm)


# Функции Снелла и пересечения
def snell_refraction_direction(v_in, n1, n2, normal):
    v = v_in / np.linalg.norm(v_in)
    n_hat = normal / np.linalg.norm(normal)

    cos_theta_i = np.dot(v, n_hat)
    sin_theta_i_sq = max(0.0, 1.0 - cos_theta_i ** 2)
    sin_theta_i = np.sqrt(sin_theta_i_sq)

    if (n1 / n2) * sin_theta_i > 1.0:
        return None

    sin_theta_t = (n1 / n2) * sin_theta_i
    cos_theta_t = np.sqrt(max(0.0, 1.0 - sin_theta_t ** 2))
    v_out = (n1 / n2) * (v - cos_theta_i * n_hat) + cos_theta_t * n_hat
    return v_out / np.linalg.norm(v_out)


def intersect_with_hypotenuse(p0, v, h):
    x0, y0 = p0
    vx, vy = v

    denom = vx + vy
    if abs(denom) < 1e-8:
        return None

    t = (h - x0 - y0) / denom
    if t <= 0:
        return None

    x_int = x0 + t * vx
    y_int = y0 + t * vy
    return np.array([x_int, y_int]), t


# Отрисовка графика
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
ax.set_xlim(-2.0, 8.0)
ax.set_ylim(-1.0, 5.0)
ax.set_title("Разложение белого света в призме", fontsize=16)
ax.set_xlabel("x")
ax.set_ylabel("y")

prism_coords = np.array([[0, 0], [0, h], [h, 0]])
ax.fill(prism_coords[:, 0], prism_coords[:, 1], color="lightgray", edgecolor="black", zorder=0)

entry_y = h / 2
entry_point = np.array([0.0, entry_y])

x_start = -2.0
y_start = entry_y - (0 - x_start) * np.tan(angle_inc_rad)
start_point = np.array([x_start, y_start])

ax.plot([start_point[0], entry_point[0]], [start_point[1], entry_point[1]],
        color="black", linestyle="--", linewidth=2)

normal_in = np.array([1.0, 0.0])

# Градиентное построение лучей
for i, wl in enumerate(wavelengths_nm):
    n = n_values[i]
    color = cmap(norm(wl))

    v_incident = np.array([np.cos(angle_inc_rad), np.sin(angle_inc_rad)])
    v_inside = snell_refraction_direction(v_incident, 1.0, n, normal_in)
    if v_inside is None:
        continue

    result = intersect_with_hypotenuse(entry_point, v_inside, h)
    if result is None:
        continue

    exit_point, t_hit = result



    normal_exit = np.array([1.0, 1.0]) / np.sqrt(2)
    v_out = snell_refraction_direction(v_inside, n, 1.0, normal_exit)
    if v_out is None:
        continue

    length_out = 3.0
    out_end = exit_point + v_out * length_out

    ax.plot([entry_point[0], exit_point[0]], [entry_point[1], exit_point[1]], color=color, linewidth=2)
    ax.plot([exit_point[0], out_end[0]], [exit_point[1], out_end[1]], color=color, linewidth=2)

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Длина волны, нм')

ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
