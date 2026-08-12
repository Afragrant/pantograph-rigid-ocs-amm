import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.signal import butter, sosfilt, sosfilt_zi
from tqdm import tqdm

PANTOGRAPH = 3  # 1 = DSA380; 2 = DSA250; 3 = TSG18F
RIGID_OVERHEAD_CONTACT_SYSTEM = 2
SPEED_KMH = 120  # [km/h]；陈龙论文表2-3验证工况为 120 km/h
NM = 200
DT_BASE = 5e-5  # [s]
G = 9.8  # [m/s²]

# KS 为受电弓–接触网之间的接触弹簧刚度
KS = 82300  # 接触刚度 [N/m]
ALPHA_C = 0.0125
BETA_C = 0.0001

N_SKIP_SPANS = 8  # 统计稳定段时剔除首尾各 N_SKIP_SPANS 跨

LOWPASS_HZ = 20.0  # 低通滤波截止频率 [Hz], 按 EN 50317:2012
LOWPASS_ORDER = 6  # Butterworth 滤波器阶数, EN 50317:2012 要求 6 阶
LOWPASS_ENABLED = True  # 是否启用低通滤波; False 时统计与绘图均为原始未滤波数据

MAX_NEWTON_ITER = 10  # 每步接触 Newton / 有效集迭代最大次数

# 'woodbury' = 预分解 + Sherman-Morrison 秩-1 优化(默认, 快)
# 'direct'   = 每步重新组装并分解刚度矩阵(慢, 用于对比验证)
LINEAR_SOLVER = 'woodbury'

# 接触线波磨不平顺默认参数，参考范围：幅值 [0.2, 3] mm，波长 [0.4, 1.2] m
WEAR_AMPLITUDE = 1.0e-3  # [m]
WEAR_WAVELENGTH = 0.6  # [m]

# 汇流排接头从模型起点开始布置；实际线路可按施工布置覆盖此参数.
BUSBAR_JOINT_OFFSET = 0.0  # [m]


def contact_wire_wear(x, A_w: float = WEAR_AMPLITUDE, lambda_w: float = WEAR_WAVELENGTH):
    """接触线磨耗深度 W_cw = A_w/2 · (1 - cos(2πl/λ_w))；l 为接触线沿线长度坐标。"""
    return 0.5 * A_w * (1.0 - np.cos(2.0 * np.pi * x / lambda_w))


def contact_penetration(y_pantograph: float, y_contact_wire: float, wear_depth: float = 0.0) -> float:
    """穿透量 = 弓头位移 - 接触线位移 - 磨耗深度，向上为正。"""
    return y_pantograph - y_contact_wire - wear_depth


def rigid_overhead_contact_system_params(rigid_overhead_contact_system: int, N_spans: int | None = None):
    """返回指定刚性接触网预设的结构参数。

    返回 (L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ)：
    L 跨距，N 跨数，rhoA 线密度，EI 抗弯刚度，
    KEQ 悬挂刚度，MEQ 悬挂质量，MZ 汇流排接头等效质量，L_MZ 汇流排长度。
    """
    table = {
        # L,    N,  rhoA, EI,       KEQ,  MEQ, MZ,   L_MZ
        1: (
            8.0,
            30,
            8.1,
            1.7e5,
            6.7e7,
            7.0,
            2.84,
            12.0,
        ),  # surrogate-model paper, OCR system case 1
        2: (
            8.5,
            30,
            7.1,
            2.7e5,
            6e5,
            7.0,
            2.84,
            12.0,
        ),  # surrogate-model paper, OCR system case 2
        3: (8.0, 30, 7.25, 2.69e5, 6e4, 7.0, 2.84, 12.0),  # 陈龙 西南交大博士大论文
    }
    if rigid_overhead_contact_system not in table:
        raise ValueError(
            f'RIGID_OVERHEAD_CONTACT_SYSTEM must be 1-3, the current value is {rigid_overhead_contact_system}.'
        )
    L, N_default, rhoA, EI, KEQ, MEQ, MZ, L_MZ = table[rigid_overhead_contact_system]
    N = N_spans if N_spans is not None else N_default
    return L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ


def pantograph_params(ptype: int):
    """返回所选受电弓的 (m1,m2,m3, k1,k2,k3, c1,c2,c3, F0)。

    m/k/c 下标 1→3 依次对应弓头、上框架、下框架；F0 为抬升力。
    """
    table = {
        1: (7.12, 6.00, 5.80, 9430.0, 14100.0, 0.1, 0, 0, 70.0, 120.0),  # DSA380
        2: (7.51, 5.855, 4.645, 8380.0, 6200.0, 80.0, 0, 0, 70, 120.0),  # DSA250
        3: (10.31, 6.15, 15.30, 53415.8, 6754.2, 40687.0, 29.60, 0.20, 1.14, 110.0),
    }
    if ptype not in table:
        raise ValueError(f'PANTOGRAPH must be 1-3, the current value is {ptype}')
    return table[ptype]


def compute_busbar_positions(LS: float, l_mz: float, offset: float = BUSBAR_JOINT_OFFSET) -> np.ndarray:
    """返回等间距布置下的汇流排接头位置。"""
    if l_mz <= 0:
        raise ValueError(f'l_mz must be positive, got {l_mz}')
    if not 0 <= offset < LS:
        raise ValueError(f'offset must satisfy 0 <= offset < LS, got offset={offset}, LS={LS}')
    return np.arange(offset, LS, l_mz, dtype=float)


def lowpass_filter(sig: np.ndarray, fs: float, fc: float = LOWPASS_HZ, order: int = LOWPASS_ORDER) -> np.ndarray:
    """按 EN 50317:2012 的 6 阶 Butterworth 20 Hz 低通滤波；单向滤波保持标准规定的幅频特性。

    使用 sosfilt_zi 将初始状态设为与 sig[0] 匹配的稳态，避免零初始条件引起的启动振铃。
    """
    sos = butter(order, fc, fs=fs, output='sos')
    zi = sosfilt_zi(sos) * sig[0]
    return sosfilt(sos, sig, zi=zi)[0]


def run_simulation(
    rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,
    pantograph: int = PANTOGRAPH,
    speed_kmh: float = SPEED_KMH,
    NM: int = NM,
    N_spans: int = 30,
    dt_base: float = DT_BASE,
    irregularity: bool = False,
    wear_amplitude: float = WEAR_AMPLITUDE,
    wear_wavelength: float = WEAR_WAVELENGTH,
    busbar_joint_offset: float = BUSBAR_JOINT_OFFSET,
    lowpass: bool = LOWPASS_ENABLED,
    verbose: bool = True,
    max_newton_iter: int = MAX_NEWTON_ITER,
    method: str = LINEAR_SOLVER,
):
    if method not in ('woodbury', 'direct'):
        raise ValueError(f"method must be 'woodbury' or 'direct', got {method!r}")

    L, N, rhoA, EI, KEQ, MEQ, MZ, L_MZ = rigid_overhead_contact_system_params(rigid_overhead_contact_system, N_spans)
    m1, m2, m3, k1, k2, k3, c1, c2, c3, F0 = pantograph_params(pantograph)

    v = speed_kmh / 3.6  # [m/s]
    dt = dt_base
    LS = L * N

    t_total = LS / v
    n_steps = int(np.floor(t_total / dt)) + 1  # 接触点只在梁范围内移动，步数按全程时间取整
    t_vec = np.arange(n_steps, dtype=float) * dt
    x_vec = v * t_vec

    if verbose:
        print('=' * 60)
        print('POCS Simulation')
        print('=' * 60)
        print(f'  Catenary preset  : {rigid_overhead_contact_system}   (L={L} m, N={N} spans, LS={LS} m)')
        print(f'  Pantograph type  : {pantograph}')
        print(f'  Speed            : {speed_kmh} km/h  ({v:.2f} m/s)')
        print(f'  Time step dt     : {dt:.6f} s')
        print(f'  Total steps      : {n_steps:,}')
        print(f'  Retained modes   : {NM}')
        print(f'  Linear solver    : {method}')
        if irregularity:
            print(f'  Irregularity     : ON  (A_w={wear_amplitude * 1e3:.3f} mm, λ_w={wear_wavelength:.3f} m)')
        else:
            print('  Irregularity     : OFF')
        if lowpass:
            print(f'  Low-pass filter  : {LOWPASS_HZ:.1f} Hz (Butterworth order {LOWPASS_ORDER}, single-pass)')
        else:
            print('  Low-pass filter  : OFF')

    M_p = np.diag([m1, m2, m3])
    C_p = np.array(
        [
            [c1, -c1, 0],
            [-c1, c1 + c2, -c2],
            [0, -c2, c2 + c3],
        ],
        dtype=float,
    )
    K_p = np.array(
        [
            [k1, -k1, 0],
            [-k1, k1 + k2, -k2],
            [0, -k2, k2 + k3],
        ],
        dtype=float,
    )
    F_p = np.array([0.0, 0.0, F0])

    modes = np.arange(1, NM + 1, dtype=float)
    norm_factor = np.sqrt(2.0 / (rhoA * LS))
    omega_n = (modes * np.pi / LS) ** 2 * np.sqrt(EI / rhoA)

    x_j = L * np.arange(1, N, dtype=float)
    Phi_sup = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_j))  # (NM, N-1)
    M_add_sup = MEQ * Phi_sup @ Phi_sup.T
    K_add_sup = KEQ * Phi_sup @ Phi_sup.T

    x_mz = compute_busbar_positions(LS, l_mz=L_MZ, offset=busbar_joint_offset)
    Phi_mz = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_mz))  # (NM, Nmz)
    M_add_mz = MZ * Phi_mz @ Phi_mz.T

    M_cat = np.eye(NM) + M_add_sup + M_add_mz
    K_cat = np.diag(omega_n**2) + K_add_sup
    # 瑞利阻尼C = α·M + β·K, 含支撑/接头的满矩阵.
    C_cat = ALPHA_C * M_cat + BETA_C * K_cat

    int_sin = (LS / (modes * np.pi)) * (1.0 - np.cos(modes * np.pi))
    F_grav_beam = -rhoA * G * norm_factor * int_sin
    F_grav_sup = -MEQ * G * Phi_sup.sum(axis=1)
    F_grav_mz = -MZ * G * Phi_mz.sum(axis=1)
    F_gravity = F_grav_beam + F_grav_sup + F_grav_mz

    # 找形等效：悬挂点静态位移归零
    # 约束悬挂点处模态位移 Phi_sup.T @ q = 0，使支撑点在静态平衡态位于设计高度。
    # 这相当于调节悬挂弹簧原长，动力学中仍保留 KEQ 弹簧。
    n_sup = Phi_sup.shape[1]
    KKT = np.block(
        [
            [np.diag(omega_n**2), Phi_sup],
            [Phi_sup.T, np.zeros((n_sup, n_sup))],
        ]
    )
    sol = np.linalg.solve(KKT, np.concatenate([F_gravity, np.zeros(n_sup)]))
    q_static = sol[:NM]
    # 悬挂预应力：使 K_cat q_static = F_gravity + F_pre（弹簧在 q_static 处应变为零）
    F_pre = np.diag(omega_n**2) @ q_static - F_gravity
    F_gravity = F_gravity + F_pre

    n_dof = 3 + NM
    M_sys = np.zeros((n_dof, n_dof))
    C_sys = np.zeros((n_dof, n_dof))
    K_sys = np.zeros((n_dof, n_dof))
    M_sys[:3, :3] = M_p
    M_sys[3:, 3:] = M_cat
    C_sys[:3, :3] = C_p
    C_sys[3:, 3:] = C_cat
    K_sys[:3, :3] = K_p
    K_sys[3:, 3:] = K_cat

    F_base = np.zeros(n_dof)
    F_base[3:] = F_gravity

    # Woodbury 预分解：P 的 Cholesky 分解只做一次，接触秩-1 更新用 Sherman-Morrison 出解
    beta_nm, gamma_nm = 0.25, 0.5
    a0 = 1.0 / (beta_nm * dt * dt)
    a1 = gamma_nm / (beta_nm * dt)
    a2 = 1.0 / (beta_nm * dt)
    a3 = 1.0 / (2.0 * beta_nm) - 1.0
    a4 = gamma_nm / beta_nm - 1.0
    a5 = dt * (gamma_nm / (2.0 * beta_nm) - 1.0)
    a6 = dt * (1.0 - gamma_nm)
    a7 = gamma_nm * dt
    P = K_sys + a0 * M_sys + a1 * C_sys
    cP = cho_factor(P, check_finite=True) if method == 'woodbury' else None
    w_buf = np.zeros(n_dof)
    w_buf[0] = 1.0  # w = [1, 0, 0, -φ]; 接触步内原地写入 w[3:]

    def phi_at(x):
        return norm_factor * np.sin(modes * np.pi * x / LS)

    # 受电弓装配预应力：把稳定段静态接触力均值锚定到 F0。
    # 串联链模型中，导线静态高度 u(x) 会通过接地弹簧 k3 调制 Fc；
    # 加入常量基底预应力 P3 后，mean(Fc) = F0，而动态波动特性不变。
    # S_p 包含接触弹簧、受电弓弹簧以及接触网平均局部柔度（trace(K_inv)/(rhoA*LS)）。
    K_cat_inv = np.linalg.inv(K_cat)
    c_cat_mean = np.trace(K_cat_inv) / (rhoA * LS)
    S_p = 1.0 / KS + 1.0 / k1 + 1.0 / k2 + c_cat_mean
    i_start_pre = np.searchsorted(x_vec, N_SKIP_SPANS * L, side='left')
    i_end_pre = np.searchsorted(x_vec, LS - N_SKIP_SPANS * L, side='right')
    if i_start_pre < i_end_pre:
        u_mean = np.mean(np.array([phi_at(x) @ q_static for x in x_vec[i_start_pre:i_end_pre]]))
    else:
        # 跨数不足时退化为全线均值，避免空窗口导致 NaN
        u_mean = np.mean(np.array([phi_at(x) @ q_static for x in x_vec]))
    P3 = k3 * (F0 * S_p + u_mean)
    F_p_eff = F_p.copy()
    F_p_eff[2] += P3
    F_base[:3] = F_p_eff

    Y = np.zeros(n_dof)
    V = np.zeros(n_dof)

    # 接触网从重力静平衡位形起步
    Y[3:] = q_static

    # 受电弓初值：x=0 处挂接触弹簧求静平衡
    phi_0 = phi_at(0.0)
    u_c_0 = phi_0 @ q_static
    K_p_static = K_p.copy()
    K_p_static[0, 0] += KS
    F_p_static = F_p_eff + np.array([KS * u_c_0, 0.0, 0.0])
    Y[:3] = np.linalg.solve(K_p_static, F_p_static)

    # 初始加速度按含接触弹簧的装配体求解
    Kc0 = np.zeros((n_dof, n_dof))
    Kc0[0, 0] = KS
    Kc0[0, 3:] = -KS * phi_0
    Kc0[3:, 0] = -KS * phi_0
    Kc0[3:, 3:] = KS * np.outer(phi_0, phi_0)
    A = np.linalg.solve(M_sys, F_base - C_sys @ V - (K_sys + Kc0) @ Y)

    contact_force = np.zeros(n_steps)
    y_pantograph = np.zeros(n_steps)
    y_rigid_overhead_contact_system = np.zeros(n_steps)

    rel_0 = Y[0] - u_c_0
    contact_force[0] = KS * rel_0 if rel_0 > 0 else 0.0
    y_pantograph[0] = Y[0]
    y_rigid_overhead_contact_system[0] = u_c_0

    t_start = time.perf_counter()
    newton_failures = 0
    solve_time = 0.0
    for k in tqdm(range(1, n_steps), desc='Simulating', unit='step', disable=not verbose):
        xc = x_vec[k]
        phi = phi_at(xc)

        # 接触线波磨磨耗深度 w_cw(x)，关闭不平顺时恒为 0。
        # 接触线位于汇流排下方，磨耗使工作面向上移动，因此 gap = y_p - y_c - w_cw.
        w_cw = contact_wire_wear(xc, wear_amplitude, wear_wavelength) if irregularity else 0.0

        # Newmark 有效载荷中不随接触状态改变的部分 (Y, V, A 来自上一步).
        Ft_base = F_base + M_sys @ (a0 * Y + a2 * V + a3 * A) + C_sys @ (a1 * Y + a4 * V + a5 * A)

        # Newton / 有效集迭代: 找到与位移自洽的接触状态.
        # 罚函数接触在固定接触集下是线性的, 因此有效集稳定后 Sherman-Morrison 解即精确.
        Y_new = Y + dt * V + (0.5 - beta_nm) * dt * dt * A
        in_contact = contact_penetration(Y_new[0], phi @ Y_new[3:], w_cw) > 0.0
        z = None

        for _ in range(max_newton_iter):
            if in_contact:
                w_buf[3:] = -phi
                # 磨耗等效为接触弹簧的常量预压量, 仅接触时计入右端项.
                if w_cw != 0.0:
                    Ft = Ft_base.copy()
                    Ft[0] += KS * w_cw
                    Ft[3:] -= KS * phi * w_cw
                else:
                    Ft = Ft_base
                t_solve = time.perf_counter()
                if method == 'direct':
                    Kt = P + KS * np.outer(w_buf, w_buf)
                    Y_new = cho_solve(cho_factor(Kt, check_finite=True), Ft, check_finite=True)
                else:
                    if z is None:
                        z = cho_solve(cP, w_buf, check_finite=True)
                    u = cho_solve(cP, Ft, check_finite=True)
                    Y_new = u - KS * (w_buf @ u) / (1.0 + KS * (w_buf @ z)) * z
                solve_time += time.perf_counter() - t_solve
            else:
                Ft = Ft_base
                t_solve = time.perf_counter()
                if method == 'direct':
                    Y_new = cho_solve(cho_factor(P, check_finite=True), Ft, check_finite=True)
                else:
                    Y_new = cho_solve(cP, Ft_base, check_finite=True)
                solve_time += time.perf_counter() - t_solve

            # 检查新位移下的接触状态是否与假设一致.
            gap = contact_penetration(Y_new[0], phi @ Y_new[3:], w_cw)
            in_contact_new = gap > 0.0
            if in_contact == in_contact_new:
                break
            in_contact = in_contact_new
        else:
            newton_failures += 1  # 达到最大迭代次数仍未稳定 (极少发生)

        u_c_new = phi @ Y_new[3:]
        A_new = a0 * (Y_new - Y) - a2 * V - a3 * A
        V = V + a6 * A + a7 * A_new
        A = A_new
        Y = Y_new

        rel = contact_penetration(Y[0], u_c_new, w_cw)
        contact_force[k] = KS * rel if rel > 0 else 0.0
        y_pantograph[k] = Y[0]
        y_rigid_overhead_contact_system[k] = u_c_new

    elapsed_s = time.perf_counter() - t_start
    if verbose:
        print(f'\nSimulation complete – {elapsed_s:.1f} s wall time')
        if newton_failures:
            print(f'  Warning: Newton/active-set did not converge at {newton_failures} step(s)')

    # 低通滤波开关: 启用时统计与绘图全部使用滤波后信号, 关闭时为原始信号.
    # 例外：离线率必须用未滤波的原始接触力统计，低通会把 F=0 的采样平滑为非零，导致离线被掩盖。
    contact_force_raw = contact_force  # lowpass_filter 返回新数组, 此引用零拷贝保留原始信号
    if lowpass:
        fs = 1.0 / dt
        contact_force = lowpass_filter(contact_force, fs)
        y_pantograph = lowpass_filter(y_pantograph, fs)
        y_rigid_overhead_contact_system = lowpass_filter(y_rigid_overhead_contact_system, fs)

    def compute_stats(fc_arr):
        if len(fc_arr) == 0:
            return {}
        return {
            'mean_N': float(fc_arr.mean()),
            'std_N': float(fc_arr.std(ddof=1)),
            'max_N': float(fc_arr.max()),
            'min_N': float(fc_arr.min()),
        }

    stats_full = compute_stats(contact_force)

    i_start = np.searchsorted(x_vec, N_SKIP_SPANS * L, side='left')
    i_end = np.searchsorted(x_vec, LS - N_SKIP_SPANS * L, side='right')
    stable_label = f'first/last {N_SKIP_SPANS} spans skipped'
    contact_force_stable = contact_force[i_start:i_end]
    x_stable = x_vec[i_start:i_end]
    y_panto_stable = y_pantograph[i_start:i_end]
    y_cat_stable = y_rigid_overhead_contact_system[i_start:i_end]
    stats_stable = compute_stats(contact_force_stable)
    stats_stable_raw = compute_stats(contact_force_raw[i_start:i_end])
    stats_full['loss_of_contact_pct'] = float(100 * (contact_force_raw == 0).mean())
    if len(contact_force_stable):
        stats_stable['loss_of_contact_pct'] = float(100 * (contact_force_raw[i_start:i_end] == 0).mean())

    if verbose:
        print(f'\n--- Contact force statistics (stable window: {stable_label}) ---')
        for k, v in stats_stable.items():
            print(f'  {k:<28}: {v:.3f}')
        if lowpass and stats_stable_raw:
            print(f'\n--- Raw (unfiltered) contact force statistics (stable window: {stable_label}) ---')
            for k, v in stats_stable_raw.items():
                print(f'  {k:<28}: {v:.3f}')
        print()

    return {
        'x_vec': x_vec,
        't_vec': t_vec,
        'contact_force': contact_force,
        'y_pantograph': y_pantograph,
        'y_rigid_overhead_contact_system': y_rigid_overhead_contact_system,
        'x_stable': x_stable,
        't_stable': t_vec[i_start:i_end],
        'contact_force_stable': contact_force_stable,
        'y_pantograph_stable': y_panto_stable,
        'y_rigid_overhead_contact_system_stable': y_cat_stable,
        'stats_full': stats_full,
        'stats_stable': stats_stable,
        'stats_stable_raw': stats_stable_raw,
        'stable_label': stable_label,
        'lowpass_hz': LOWPASS_HZ if lowpass else None,
        'elapsed_s': elapsed_s,
        'solve_time_s': solve_time,
    }


def plot_results(
    results: dict,
    pantograph: int = PANTOGRAPH,
    rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,
    speed_kmh: float = SPEED_KMH,
    out_dir: str | Path = './result/pc_plots',
    show: bool = True,
):
    """绘制接触力、弓头位移、接触网位移（全程 + 稳定段）。"""
    stable_label = results.get('stable_label', f'first/last {N_SKIP_SPANS} spans skipped')
    lp_hz = results.get('lowpass_hz')
    lp_note = f' · {lp_hz:.0f} Hz low-pass' if lp_hz else ''

    col_titles = ('Full run', f'Stable window ({stable_label})')
    xlabel = 'Position x [m]'
    rows = [
        ('Contact force [N]', 'contact_force', 'contact_force_stable'),
        ('Pantograph disp. [m]', 'y_pantograph', 'y_pantograph_stable'),
        ('OCS disp. [m]', 'y_rigid_overhead_contact_system', 'y_rigid_overhead_contact_system_stable'),
    ]
    sup = f'Pantograph {pantograph} · OCS {rigid_overhead_contact_system} · {round(speed_kmh)} km/h' + lp_note

    x_full = results['x_vec']
    x_stab = results['x_stable']

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex='col')
    for r, (ylabel, key_full, key_stab) in enumerate(rows):
        axes[r, 0].plot(x_full, results[key_full], lw=0.7, color='C0')
        axes[r, 1].plot(x_stab, results[key_stab], lw=0.7, color='C1')
        axes[r, 0].set_ylabel(ylabel)
        for c in (0, 1):
            axes[r, c].grid(alpha=0.3)
    for c in (0, 1):
        axes[0, c].set_title(col_titles[c])
        axes[-1, c].set_xlabel(xlabel)
    fig.suptitle(sup)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'p{pantograph}_c{rigid_overhead_contact_system}_{round(speed_kmh)}kmh'
    fig_path = out_dir / f'pc_response_{suffix}.png'
    fig.savefig(fig_path, dpi=150)
    print(f'Figure saved → {fig_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig_path


def plot_initial_catenary_shape(
    rigid_overhead_contact_system: int = RIGID_OVERHEAD_CONTACT_SYSTEM,
    N_spans: int = 30,
    NM_plot: int = NM,
    busbar_joint_offset: float = BUSBAR_JOINT_OFFSET,
    out_dir: str | Path = './result/pc_plots',
    show: bool = True,
):
    """绘制刚性接触网初始静态位形，标出吊点与跨中接头位置。"""
    L, N, rhoA, EI, _KEQ, MEQ, MZ, L_MZ = rigid_overhead_contact_system_params(rigid_overhead_contact_system, N_spans)
    LS = L * N

    modes = np.arange(1, NM_plot + 1, dtype=float)
    norm_factor = np.sqrt(2.0 / (rhoA * LS))
    omega_n = (modes * np.pi / LS) ** 2 * np.sqrt(EI / rhoA)

    x_j = L * np.arange(1, N, dtype=float)
    Phi_sup = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_j))

    x_mz = compute_busbar_positions(LS, l_mz=L_MZ, offset=busbar_joint_offset)
    Phi_mz = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_mz))

    int_sin = (LS / (modes * np.pi)) * (1.0 - np.cos(modes * np.pi))
    F_grav_beam = -rhoA * G * norm_factor * int_sin
    F_grav_sup = -MEQ * G * Phi_sup.sum(axis=1)
    F_grav_mz = -MZ * G * Phi_mz.sum(axis=1)
    F_gravity = F_grav_beam + F_grav_sup + F_grav_mz

    # 找形等效：悬挂点静态位移归零
    n_sup = Phi_sup.shape[1]
    KKT = np.block(
        [
            [np.diag(omega_n**2), Phi_sup],
            [Phi_sup.T, np.zeros((n_sup, n_sup))],
        ]
    )
    sol = np.linalg.solve(KKT, np.concatenate([F_gravity, np.zeros(n_sup)]))
    q_static = sol[:NM_plot]

    x_fine = np.linspace(0, LS, 2000)
    phi_fine = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_fine))
    u_static = phi_fine.T @ q_static

    all_sup = np.concatenate([[0.0], x_j, [LS]])
    phi_sup_all = norm_factor * np.sin(np.outer(modes * np.pi / LS, all_sup))
    u_sup_all = phi_sup_all.T @ q_static

    phi_mz = norm_factor * np.sin(np.outer(modes * np.pi / LS, x_mz))
    u_mz = phi_mz.T @ q_static

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_fine, u_static * 1e3, lw=1.2, color='C0', label='Catenary vertical disp.')

    ax.scatter(all_sup, u_sup_all * 1e3, color='red', s=50, zorder=5, marker='o', label='Suspension point')
    ax.scatter(x_mz, u_mz * 1e3, color='green', s=50, zorder=5, marker='^', label='Mid-span joint')

    ax.set_xlabel('Position x [m]')
    ax.set_ylabel('Vertical disp. [mm]')
    title = f'Initial catenary static shape (preset {rigid_overhead_contact_system}, spans={N}, span={L} m)'
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f'initial_catenary_shape_c{rigid_overhead_contact_system}.png'
    fig.savefig(fig_path, dpi=150)
    print(f'Figure saved → {fig_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig_path


if __name__ == '__main__':
    plot_initial_catenary_shape(
        rigid_overhead_contact_system=RIGID_OVERHEAD_CONTACT_SYSTEM,
    )
    results = run_simulation(
        rigid_overhead_contact_system=RIGID_OVERHEAD_CONTACT_SYSTEM,
        pantograph=PANTOGRAPH,
        speed_kmh=SPEED_KMH,
        NM=NM,
        verbose=True,
        lowpass=True,
    )
    plot_results(
        results,
        pantograph=PANTOGRAPH,
        rigid_overhead_contact_system=RIGID_OVERHEAD_CONTACT_SYSTEM,
        speed_kmh=SPEED_KMH,
    )
