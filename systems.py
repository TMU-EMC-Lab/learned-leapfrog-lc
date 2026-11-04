#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""調和振動子以外の力学系（2026-09-30 追加）：単振り子・ファンデルポール方程式・ダフィング方程式・ローレンツ方程式.

各系は次を持つ。
- rhs(s)：ベクトル場 ds/dt（s は (N, d)）
- 参照解：刻みの細かい古典的 RK4（numpy でベクトル化）。integrate() が dt ごとの状態を返す
- 初期条件の引き方（sample_ics）：学習・検証・内挿と外挿で領域を分ける
- 誤差・損失を割る尺度（scales）：変数ごと・初期条件ごと

2 階の系（位置 q と速度 v を持つ単振り子・ファンデルポール・ダフィング）は LF 系のモデルにも使える。
ローレンツ方程式は 1 階の 3 変数系で，今は状態入力の 1 段系だけに対応する（LF 系の割り当ては未定）。

2026-10-15 追加（記憶なし 3 段階の比較を保存系で広げる）:
- 二重井戸（DoubleWell）・モース型ポテンシャル（Morse）：1 自由度の保存系
- 磁場中の荷電粒子（ChargedParticle）：2 自由度。保存系だが力が速度に依存する（ローレンツ力）
- 質点の列（FPUChain）：n 自由度（既定 4）。両端固定の β-FPU
多自由度の系は状態を s = (q_1..q_k, v_1..v_k) と並べ，n_dof = k を持つ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def rk4_integrate(
    rhs: Callable[[np.ndarray], np.ndarray],
    s0: np.ndarray,
    dt: float,
    steps: int,
    h_max: float = 1e-3,
) -> np.ndarray:
    """s0 (N, d) から dt ごとに steps 点（s0 を含む）を返す (N, steps, d). dt < 0 で時間を戻す.

    1 区間 dt を ceil(|dt| / h_max) 個の RK4 で積分する（参照解用。刻み 1e-3 で局所誤差は 1e-15 程度）。
    """
    s = np.asarray(s0, dtype=np.float64).copy()
    n_sub = max(1, int(math.ceil(abs(dt) / h_max)))
    h = dt / n_sub
    out = np.empty((s.shape[0], steps, s.shape[1]), dtype=np.float64)
    out[:, 0] = s
    for k in range(1, steps):
        for _ in range(n_sub):
            k1 = rhs(s)
            k2 = rhs(s + 0.5 * h * k1)
            k3 = rhs(s + 0.5 * h * k2)
            k4 = rhs(s + h * k3)
            s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[:, k] = s
    return out


@dataclass
class System:
    name: str
    dim: int
    var_names: tuple[str, ...]
    second_order: bool  # True なら s = (q, v)（位置 1 つ・速度 1 つ）で LF 系にも使える
    params: dict = field(default_factory=dict)
    t_end: float = 10.0  # 学習に使う軌道の長さ [s]
    dt_direct: float = 0.025  # 1 段系の刻み
    dt_lf: float | None = 0.05  # LF 系の刻み（None なら LF 系は使えない）
    t_evals: tuple[float, ...] = (10.0, 20.0, 50.0)
    omega: float = 1.0  # 代表の角振動数（損失の v の規格化などで使う。既存コードとの互換）
    n_dof: int = 1  # 2 階の系の自由度 k（状態は (q_1..q_k, v_1..v_k)）

    # --- 系ごとに上書きする ---
    def rhs(self, s: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def energy(self, s: np.ndarray) -> np.ndarray | None:
        return None

    def sample_ics(self, n: int, lo: float, hi: float, rng: np.random.Generator, **kw) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def region_value(self, ics: np.ndarray) -> np.ndarray:
        """初期条件の「大きさ」（内挿・外挿の境界に使う量）."""
        raise NotImplementedError  # pragma: no cover

    def scales(self, ics: np.ndarray, floor: float = 0.05, **kw) -> np.ndarray:
        """誤差・損失を割る尺度 (N, d). 既定は参照解の各変数の最大絶対値（学習長まで）."""
        ref = self.integrate(ics, self.dt_direct, int(round(self.t_end / self.dt_direct)) + 1)
        return np.maximum(np.abs(ref).max(axis=1), floor)

    # --- 共通 ---
    def integrate(self, ics: np.ndarray, dt: float, steps: int) -> np.ndarray:
        return rk4_integrate(self.rhs, ics, dt, steps)

    def half_step_velocity(self, ics: np.ndarray, dt: float, steps: int) -> np.ndarray:
        """LF 系の教師：v[n] = v((n − 1/2) dt)（n = 0..steps−1）. 2 階の系のみ.

        1 自由度は (N, steps)，多自由度は (N, steps, k) を返す。
        """
        if not self.second_order:
            raise ValueError(f"{self.name} は 2 階の系ではない（LF 系は使えない）")
        s_half = rk4_integrate(self.rhs, ics, -0.5 * dt, 2)[:, 1]  # t = −dt/2
        v = self.integrate(s_half, dt, steps)[:, :, self.n_dof:]
        return v[:, :, 0] if self.n_dof == 1 else v

    def position_record(self, ics: np.ndarray, dt: float, steps: int) -> np.ndarray:
        """位置だけの計測（2026-10-17）：t = −dt, 0, dt, …, steps·dt の位置 (N, steps + 2, k). 2 階の系のみ.

        P[:, j] = x_{j−1}。差分の速度を作るのに，始めと終わりに 1 点ずつ余分に記録する。
        """
        if not self.second_order:
            raise ValueError(f"{self.name} は 2 階の系ではない（位置だけの計測は使えない）")
        s_m1 = rk4_integrate(self.rhs, ics, -dt, 2)[:, 1]  # t = −dt
        return self.integrate(s_m1, dt, steps + 2)[:, :, : self.n_dof]

    def fd_teacher(self, ics: np.ndarray, dt: float, steps: int, family: str) -> tuple[np.ndarray, np.ndarray]:
        """位置だけの計測から作る教師（teacher=fdpos）. 戻り値 (x, v)：x (N, steps, k)，v (N, steps, k).

        LF 系：v[:, n] = (x_n − x_{n−1})/dt = v_{n−1/2} の差分版（位置の段の答えは厳密に v_new になる）。
        1 段系：v[:, n] = (x_{n+1} − x_{n−1})/(2 dt)（中心差分，整数時刻）。
        """
        P = self.position_record(ics, dt, steps)
        x = P[:, 1 : steps + 1]
        if family == "lf":
            v = (P[:, 1 : steps + 1] - P[:, 0:steps]) / dt
        elif family == "direct_state":
            v = (P[:, 2 : steps + 2] - P[:, 0:steps]) / (2.0 * dt)
        else:
            raise ValueError(family)
        return x, v

    def manifest_params(self) -> dict:
        return dict(self.params)


class Harmonic(System):
    """調和振動子 dx/dt = v, dv/dt = −ω² x（汎用の経路の確認用。既存の調和振動子の実験は harmonic_ref を使う）."""

    def __init__(self, omega: float = 1.0) -> None:
        super().__init__("harmonic", 2, ("x", "v"), True, {"omega": omega}, omega=omega)

    def rhs(self, s):
        w2 = self.params["omega"] ** 2
        return np.stack([s[:, 1], -w2 * s[:, 0]], axis=1)

    def energy(self, s):
        w = self.params["omega"]
        return 0.5 * s[..., 1] ** 2 + 0.5 * w * w * s[..., 0] ** 2

    def region_value(self, ics):
        w = self.params["omega"]
        return np.sqrt(ics[:, 0] ** 2 + (ics[:, 1] / w) ** 2)

    def sample_ics(self, n, lo, hi, rng, **kw):
        # (x, v/ω) 平面の円環 lo < A ≤ hi から面積一様
        w = self.params["omega"]
        r = np.sqrt(rng.uniform(lo * lo, hi * hi, size=n))
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([r * np.cos(th), w * r * np.sin(th)], axis=1)

    def scales(self, ics, floor=0.05, **kw):
        A = np.maximum(self.region_value(ics), floor)
        return np.stack([A, self.params["omega"] * A], axis=1)


class Pendulum(System):
    """単振り子 dθ/dt = ω, dω/dt = −ω0² sin θ（振動する軌道のみ，最大振れ角 θ_max < π）.

    振れ角の範囲は以前の batch_pendulum_ic と同じ（内挿 θ_max ≤ 60°，外挿 60°〜90°）。
    ω0 は既定 1（調和振動子と時間の尺度をそろえる。旧実験は g = 9.81，L = 1）。
    """

    def __init__(self, omega0: float = 1.0) -> None:
        super().__init__("pendulum", 2, ("theta", "omega"), True, {"omega0": omega0}, omega=omega0)

    def rhs(self, s):
        w2 = self.params["omega0"] ** 2
        return np.stack([s[:, 1], -w2 * np.sin(s[:, 0])], axis=1)

    def energy(self, s):
        w2 = self.params["omega0"] ** 2
        return 0.5 * s[..., 1] ** 2 + w2 * (1.0 - np.cos(s[..., 0]))

    def theta_max(self, ics):
        w2 = self.params["omega0"] ** 2
        c = 1.0 - self.energy(ics) / w2
        return np.arccos(np.clip(c, -1.0, 1.0))

    def region_value(self, ics):
        return self.theta_max(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        # (θ, ω/ω0) 平面で θ_max ∈ (lo, hi] の領域から面積一様（棄却法）
        if hi >= math.pi:
            raise ValueError("θ_max は π 未満（振動する軌道のみ）")
        w0 = self.params["omega0"]
        wmax = 2.0 * math.sin(hi / 2.0)  # θ = 0 での最大の ω/ω0
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-hi, hi, m), w0 * rng.uniform(-wmax, wmax, m)], axis=1)
            tm = self.theta_max(cand)
            out.extend(cand[(tm > lo) & (tm <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        tm = np.maximum(self.theta_max(ics), floor)
        return np.stack([tm, self.params["omega0"] * 2.0 * np.sin(tm / 2.0)], axis=1)


class VanDerPol(System):
    """ファンデルポール方程式 dx/dt = v, dv/dt = μ(1 − x²) v − x（既定 μ = 1，リミットサイクルの振幅は約 2）.

    初期条件は (x, v) 平面の円環から面積一様（既定：学習・内挿は半径 0.05〜2，外挿 2〜3）。
    尺度は参照解（学習長まで）の各変数の最大絶対値。
    """

    def __init__(self, mu: float = 1.0) -> None:
        super().__init__("vanderpol", 2, ("x", "v"), True, {"mu": mu}, omega=1.0)

    def rhs(self, s):
        mu = self.params["mu"]
        x, v = s[:, 0], s[:, 1]
        return np.stack([v, mu * (1.0 - x * x) * v - x], axis=1)

    def region_value(self, ics):
        return np.sqrt(ics[:, 0] ** 2 + ics[:, 1] ** 2)

    def sample_ics(self, n, lo, hi, rng, **kw):
        r = np.sqrt(rng.uniform(lo * lo, hi * hi, size=n))
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


class Duffing(System):
    """外力のないダフィング方程式 dx/dt = v, dv/dt = −δ v − α x − β x³（既定 α = 1，β = 1，δ = 0）.

    δ = 0 ではエネルギー H = v²/2 + α x²/2 + β x⁴/4 が保存される（硬化型：振幅が大きいほど周期が短い）。
    δ > 0 にするとエネルギーが減る散逸系になる。外力（γ cos Ωt）は時刻に依存するので今は扱わない。
    振幅 A は折り返し点（v = 0）での |x|：β A⁴/4 + α A²/2 = H。初期条件は (x, v) 平面で A ∈ (lo, hi] の領域から面積一様。
    α > 0，β ≥ 0（単一の井戸）を想定。
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, delta: float = 0.0) -> None:
        if alpha <= 0 or beta < 0:
            raise ValueError("Duffing は α > 0，β ≥ 0（単一の井戸）のみ対応")
        super().__init__("duffing", 2, ("x", "v"), True, {"alpha": alpha, "beta": beta, "delta": delta},
                         omega=math.sqrt(alpha))

    def rhs(self, s):
        p = self.params
        x, v = s[:, 0], s[:, 1]
        return np.stack([v, -p["delta"] * v - p["alpha"] * x - p["beta"] * x ** 3], axis=1)

    def energy(self, s):
        p = self.params
        x, v = s[..., 0], s[..., 1]
        return 0.5 * v * v + 0.5 * p["alpha"] * x * x + 0.25 * p["beta"] * x ** 4

    def amplitude(self, ics):
        a, b = self.params["alpha"], self.params["beta"]
        E = self.energy(ics)
        if b == 0:
            return np.sqrt(2 * E / a)
        return np.sqrt((-a + np.sqrt(a * a + 4 * b * E)) / b)

    def region_value(self, ics):
        return self.amplitude(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        a, b = self.params["alpha"], self.params["beta"]
        vmax = math.sqrt(a * hi * hi + 0.5 * b * hi ** 4)  # A = hi のときの最大速度（x = 0）
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-hi, hi, m), rng.uniform(-vmax, vmax, m)], axis=1)
            A = self.amplitude(cand)
            out.extend(cand[(A > lo) & (A <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        A = np.maximum(self.amplitude(ics), floor)
        a, b = self.params["alpha"], self.params["beta"]
        return np.stack([A, np.sqrt(a * A * A + 0.5 * b * A ** 4)], axis=1)  # 速度の尺度は最大速度


class Lorenz(System):
    """ローレンツ方程式 dx/dt = σ(y − x), dy/dt = x(ρ − z) − y, dz/dt = xy − βz（既定 σ = 10, ρ = 28, β = 8/3）.

    カオス系なので軌道そのものの長期予測はできない（最大リャプノフ指数 ≈ 0.9，リャプノフ時間 ≈ 1.1）。
    初期条件：学習・内挿はアトラクタ上の点，外挿はアトラクタ上の点をアトラクタの広がり（標準偏差の平均）の
    lo〜hi 倍だけずらした点。尺度はアトラクタの変数ごとの標準偏差（全初期条件で共通）。
    """

    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> None:
        super().__init__("lorenz", 3, ("x", "y", "z"), False, {"sigma": sigma, "rho": rho, "beta": beta},
                         t_end=5.0, dt_direct=0.01, dt_lf=None, t_evals=(1.0, 2.0, 5.0), omega=1.0)
        self._attractor: np.ndarray | None = None

    def rhs(self, s):
        p = self.params
        x, y, z = s[:, 0], s[:, 1], s[:, 2]
        return np.stack([p["sigma"] * (y - x), x * (p["rho"] - z) - y, x * y - p["beta"] * z], axis=1)

    def attractor(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """アトラクタ上の点列（(1,1,1) から 20 時間単位を捨て，その後 400 時間単位を 0.01 ごと）."""
        if self._attractor is None:
            s0 = np.array([[1.0, 1.0, 1.0]])
            burn = rk4_integrate(self.rhs, s0, 20.0, 2, h_max=1e-3)[:, 1]
            self._attractor = rk4_integrate(self.rhs, burn, 0.01, 40001, h_max=1e-3)[0]
        return self._attractor

    def attractor_std(self) -> np.ndarray:
        return self.attractor().std(axis=0)

    def region_value(self, ics):
        # アトラクタ上の最も近い点からの距離（アトラクタの広がりで規格化）
        a = self.attractor()[::10]
        d = np.sqrt(((ics[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return d / float(self.attractor_std().mean())

    def sample_ics(self, n, lo, hi, rng, exclude: set | None = None, **kw):
        a = self.attractor()
        idx = rng.choice(len(a), size=n, replace=False)
        if exclude is not None:
            idx = np.array([i for i in rng.permutation(len(a)) if i not in exclude][:n])
            exclude.update(int(i) for i in idx)
        pts = a[idx].copy()
        if hi > 0:
            # アトラクタ上の点をずらし，アトラクタまでの距離（広がりで規格化）が (lo, hi] に入るものだけ採る
            scale = float(self.attractor_std().mean())
            out = []
            while len(out) < n:
                m = 4 * (n - len(out)) + 16
                base = a[rng.choice(len(a), size=m)]
                u = rng.normal(size=(m, 3))
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                cand = base + rng.uniform(lo, 2 * hi, size=(m, 1)) * scale * u
                rv = self.region_value(cand)
                out.extend(cand[(rv > lo) & (rv <= hi)].tolist())
            pts = np.array(out[:n])
        return pts

    def scales(self, ics, floor=0.05, **kw):
        return np.tile(np.maximum(self.attractor_std(), floor), (len(ics), 1))


def _sample_energy_shell(n: int, lo: float, hi: float, rng: np.random.Generator, quad, quart, dim: int) -> np.ndarray:
    """H(s) = quad(s) + quart(s)（2 次と 4 次の同次式）の系で，エネルギー E ∈ (lo, hi] を一様に引き，
    状態空間の向きを一様に引いて，H = E になる長さに伸ばす（調和振動子の円環から面積一様と同じく E が一様）."""
    E = rng.uniform(lo, hi, size=n)
    z = rng.normal(size=(n, dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    a, b = quad(z), quart(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.where(b > 0, (-a + np.sqrt(a * a + 4.0 * b * E)) / (2.0 * b), E / a)
    return z * np.sqrt(s2)[:, None]


class _MultiDOF(System):
    """多自由度の 2 階の系の共通部分：尺度は位置・速度それぞれで共通（座標ごとに割ると振れの小さい座標が効きすぎる）."""

    def quad(self, s):  # pragma: no cover
        raise NotImplementedError

    def quart(self, s):  # pragma: no cover
        raise NotImplementedError

    def energy(self, s):
        return self.quad(s) + self.quart(s)

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        return _sample_energy_shell(n, lo, hi, rng, self.quad, self.quart, self.dim)

    def scales(self, ics, floor=0.05, **kw):
        k = self.n_dof
        ref = self.integrate(ics, self.dt_direct, int(round(self.t_end / self.dt_direct)) + 1)
        m = np.abs(ref).max(axis=1)  # (N, d)
        sq = np.maximum(m[:, :k].max(axis=1), floor)
        sv = np.maximum(m[:, k:].max(axis=1), floor)
        return np.concatenate([np.repeat(sq[:, None], k, 1), np.repeat(sv[:, None], k, 1)], axis=1)


class DoubleWell(System):
    """二重井戸 dx/dt = v, dv/dt = a x − b x³（V = −a x²/2 + b x⁴/4，既定 a = b = 1）.

    井戸の底 x = ±√(a/b)，V_min = −a²/(4b)，山の頂上 V(0) = 0。E < 0 は片方の井戸の中の振動，E > 0 は両方の井戸をまたぐ。
    領域の量はエネルギー E。初期条件は (x, v) 平面で E ∈ (lo, hi] の領域から面積一様。
    尺度：位置は折り返し点の |x|（b x⁴/4 − a x²/2 = E の大きい方の根），速度は最大速度 √(2(E − V_min))。
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__("doublewell", 2, ("x", "v"), True, {"a": a, "b": b}, omega=math.sqrt(2.0 * a))

    def rhs(self, s):
        a, b = self.params["a"], self.params["b"]
        x, v = s[:, 0], s[:, 1]
        return np.stack([v, a * x - b * x ** 3], axis=1)

    def energy(self, s):
        a, b = self.params["a"], self.params["b"]
        x, v = s[..., 0], s[..., 1]
        return 0.5 * v * v - 0.5 * a * x * x + 0.25 * b * x ** 4

    def v_min(self) -> float:
        return -self.params["a"] ** 2 / (4.0 * self.params["b"])

    def x_turn(self, E):
        a, b = self.params["a"], self.params["b"]
        return np.sqrt((a + np.sqrt(np.maximum(a * a + 4.0 * b * E, 0.0))) / b)

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        if lo < self.v_min():
            raise ValueError("E の下限は V_min 以上")
        xmax, vmax = float(self.x_turn(hi)), math.sqrt(2.0 * (hi - self.v_min()))
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-xmax, xmax, m), rng.uniform(-vmax, vmax, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        E = self.energy(ics)
        return np.stack([np.maximum(self.x_turn(E), floor),
                         np.maximum(np.sqrt(2.0 * np.maximum(E - self.v_min(), 0.0)), floor)], axis=1)


class Morse(System):
    """モース型ポテンシャル dx/dt = v, dv/dt = −2 D a e^{−a x}(1 − e^{−a x})（V = D(1 − e^{−a x})²，既定 D = a = 1）.

    左右非対称（x > 0 側が緩い）。E < D で有界な振動。小振幅では角振動数 √(2D) a。
    領域の量はエネルギー E。初期条件は (x, v) 平面で E ∈ (lo, hi] の領域から面積一様。
    尺度：位置は右の折り返し点 x_+ = −ln(1 − √(E/D))/a（|x_−| より大きい），速度は √(2E)。
    """

    def __init__(self, D: float = 1.0, a: float = 1.0) -> None:
        super().__init__("morse", 2, ("x", "v"), True, {"D": D, "a": a}, omega=math.sqrt(2.0 * D) * a)

    def rhs(self, s):
        D, a = self.params["D"], self.params["a"]
        x, v = s[:, 0], s[:, 1]
        e = np.exp(-a * x)
        return np.stack([v, -2.0 * D * a * e * (1.0 - e)], axis=1)

    def energy(self, s):
        D, a = self.params["D"], self.params["a"]
        x, v = s[..., 0], s[..., 1]
        return 0.5 * v * v + D * (1.0 - np.exp(-a * x)) ** 2

    def x_turn(self, E):
        D, a = self.params["D"], self.params["a"]
        r = np.sqrt(np.clip(E / D, 0.0, None))
        return -np.log(1.0 - r) / a, -np.log(1.0 + r) / a  # (x_+, x_−)

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        if hi >= self.params["D"]:
            raise ValueError("E の上限は D 未満（有界な振動のみ）")
        xp, xm = (float(v) for v in self.x_turn(hi))
        vmax = math.sqrt(2.0 * hi)
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(xm, xp, m), rng.uniform(-vmax, vmax, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        E = self.energy(ics)
        return np.stack([np.maximum(self.x_turn(E)[0], floor), np.maximum(np.sqrt(2.0 * E), floor)], axis=1)


class NonlinearCapLC(System):
    """電圧に依存する容量 C(V) = C0/(1 + (V/V0)²) と線形のインダクタ L の LC 回路（2026-10-18，ELEX）.

    回路の変数 (V, I)（コンデンサの電圧・インダクタの電流）で書く：dV/dt = I/C(V) = (1 + V²) I，dI/dt = −V（C0 = L = V0 = 1）。
    電荷は Q = arctan V なので，エネルギー H = ln(1 + V²)/2 + I²/2 が保存される。強誘電体の容量や逆直列の可変容量ダイオードのように，
    |V| とともに容量が下がる。電圧の更新は I/C(V) で運動学 dx/dt = v ではないので，位置の更新を x + Δt v に固定する LF-K・LF-S は
    この系を表せない（リープフロッグの回路の過渡解析と同じく，電圧と電流を半ステップずらして交互に更新する）。
    位置だけの計測の教師（fd_teacher）は dx/dt = v を仮定するので，この系には使わない。
    領域の量はエネルギー E。初期条件は (V, I) 平面で E ∈ (lo, hi] の領域から面積一様。尺度：電圧は折り返し点 √(e^{2E} − 1)，電流は √(2E)。
    """

    def __init__(self) -> None:
        super().__init__("nlcap", 2, ("V", "I"), True, {}, omega=1.0)

    def rhs(self, s):
        V, I = s[:, 0], s[:, 1]
        return np.stack([(1.0 + V * V) * I, -V], axis=1)

    def energy(self, s):
        V, I = s[..., 0], s[..., 1]
        return 0.5 * np.log1p(V * V) + 0.5 * I * I

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        vmax, imax = math.sqrt(math.expm1(2.0 * hi)), math.sqrt(2.0 * hi)
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-vmax, vmax, m), rng.uniform(-imax, imax, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        E = self.energy(ics)
        return np.stack([np.maximum(np.sqrt(np.expm1(2.0 * E)), floor), np.maximum(np.sqrt(2.0 * E), floor)], axis=1)


class NonlinearLC(System):
    """電圧に依存する容量と電流に依存するインダクタンスの LC 回路（2026-10-19，ELEX）.

    C(V) = C0/(1 + V²)，L(I) = L0/(1 + I²)（C0 = L0 = 1）。回路の変数 (V, I) で dV/dt = (1 + V²) I，dI/dt = −(1 + I²) V。
    エネルギー H = ln(1 + V²)/2 + ln(1 + I²)/2 が保存される。電圧の更新（位置側）も電流の更新（力）も非線形で，
    電圧の更新は運動学 dx/dt = v ではない（LF-K・LF-S は表せない）。力が非線形なので，リープフロッグの並びが効くはずの系
    （非線形容量だけの NonlinearCapLC は力が線形で，並びは効かなかった）。領域の量はエネルギー E。
    """

    def __init__(self) -> None:
        super().__init__("nllc", 2, ("V", "I"), True, {}, omega=1.0)

    def rhs(self, s):
        V, I = s[:, 0], s[:, 1]
        return np.stack([(1.0 + V * V) * I, -(1.0 + I * I) * V], axis=1)

    def energy(self, s):
        V, I = s[..., 0], s[..., 1]
        return 0.5 * np.log1p(V * V) + 0.5 * np.log1p(I * I)

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        m_ = math.sqrt(math.expm1(2.0 * hi))
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-m_, m_, m), rng.uniform(-m_, m_, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        E = self.energy(ics)
        a = np.maximum(np.sqrt(np.expm1(2.0 * E)), floor)
        return np.stack([a, a], axis=1)


class NonlinearLCChargeFlux(System):
    """NonlinearLC と同じ回路を，電荷 q と磁束 φ の変数で書いた版（2026-10-21，ELEX）.

    C(V) = C0/(1 + V²)，L(I) = L0/(1 + I²)（C0 = L0 = 1）。電荷 q = ∫C dV = arctan V，磁束 φ = ∫L dI = arctan I なので，
    dq/dt = I = tan φ，dφ/dt = −V = −tan q。エネルギー H = −ln cos q − ln cos φ（= NonlinearLC の H）が保存される。
    位置側（電荷）の更新は磁束だけの非線形な関数 tan φ で，ẋ = v ではない（LF-K・LF-S・RK4-K は表せない）が，
    位置の網に中点の磁束を渡せば力の項（τΔt² tan′φ · tan q）は消える（2 節の議論がそのまま当てはまる，分離型のハミルトン系）。
    初期条件は NonlinearLC.sample_ics と同じ乱数の引き方で (V, I) を作り，(arctan V, arctan I) に変える
    （同じ seed・同じ領域なら，電圧・電流の版と物理的に同じ初期条件・同じ軌道になる）。領域の量はエネルギー E。
    """

    def __init__(self) -> None:
        super().__init__("nllcqf", 2, ("q", "phi"), True, {}, omega=1.0)

    def rhs(self, s):
        q, f = s[:, 0], s[:, 1]
        return np.stack([np.tan(f), -np.tan(q)], axis=1)

    def energy(self, s):
        q, f = s[..., 0], s[..., 1]
        return -np.log(np.cos(q)) - np.log(np.cos(f))

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        vi = NonlinearLC().sample_ics(n, lo, hi, rng)
        return np.arctan(vi)

    def scales(self, ics, floor=0.05, **kw):
        a = np.maximum(np.arccos(np.exp(-self.energy(ics))), floor)  # 折り返し点の |q|（φ = 0）。φ も同じ
        return np.stack([a, a], axis=1)


class CubicChargeFluxLC(System):
    """3 次の特性の容量とインダクタの LC 回路を，電荷 q と磁束 φ の変数で書いた版（2026-10-21 午後，ELEX の確認の実験）.

    容量の電圧 v(q) = q + q³（電荷とともに容量が下がる），インダクタの電流 i(φ) = φ + φ³（磁束とともにインダクタンスが下がる）。
    dq/dt = i(φ) = φ + φ³，dφ/dt = −v(q) = −(q + q³)。H = q²/2 + q⁴/4 + φ²/2 + φ⁴/4 が保存される（分離型のハミルトン系）。
    電荷の更新は磁束の非線形な関数で，ẋ = v ではない（LF-K・LF-S・RK4-K は表せない）。nllcqf（tan の特性）とは非線形の形が違う。
    エネルギー 1 までで，折り返し点の振幅は √(√5 − 1) ≈ 1.11，剛性 1 + 3q²・1 + 3φ² は 1〜約 4.7 倍に変わる。
    初期条件は (q, φ) 平面でエネルギー E ∈ (lo, hi] の領域から面積一様。領域の量はエネルギー E。尺度は折り返し点の振幅（q・φ とも）。
    """

    def __init__(self) -> None:
        super().__init__("nlqf3", 2, ("q", "phi"), True, {}, omega=1.0)

    def rhs(self, s):
        q, f = s[:, 0], s[:, 1]
        return np.stack([f + f ** 3, -(q + q ** 3)], axis=1)

    def energy(self, s):
        q, f = s[..., 0], s[..., 1]
        return 0.5 * q * q + 0.25 * q ** 4 + 0.5 * f * f + 0.25 * f ** 4

    def region_value(self, ics):
        return self.energy(ics)

    @staticmethod
    def _amp(E):
        return np.sqrt(np.sqrt(1.0 + 4.0 * np.asarray(E)) - 1.0)  # q²/2 + q⁴/4 = E の正の解

    def sample_ics(self, n, lo, hi, rng, **kw):
        a = float(self._amp(hi))
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-a, a, m), rng.uniform(-a, a, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        a = np.maximum(self._amp(self.energy(ics)), floor)
        return np.stack([a, a], axis=1)


class NonlinearCapLCChargeFlux(System):
    """NonlinearCapLC（1 つ目の回路）を電荷 q と磁束 φ の変数で書いた版（2026-10-22，ELEX：同じ回路を両方の変数で比べる）.

    C(V) = 1/(1 + V²)，L = 1。q = arctan V，φ = I なので dq/dt = φ，dφ/dt = −tan q（電荷の更新は運動学の形 ẋ = v）。
    H = −ln cos q + φ²/2（= NonlinearCapLC の H）。初期条件は NonlinearCapLC.sample_ics と同じ乱数の引き方で (V, I) を作り
    (arctan V, I) に変える（同じ seed・同じ領域なら電圧・電流の版と同じ初期条件・同じ軌道）。領域の量はエネルギー E。
    """

    def __init__(self) -> None:
        super().__init__("nlcapqf", 2, ("q", "phi"), True, {}, omega=1.0)

    def rhs(self, s):
        q, f = s[:, 0], s[:, 1]
        return np.stack([f, -np.tan(q)], axis=1)

    def energy(self, s):
        q, f = s[..., 0], s[..., 1]
        return -np.log(np.cos(q)) + 0.5 * f * f

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        vi = NonlinearCapLC().sample_ics(n, lo, hi, rng)
        return np.stack([np.arctan(vi[:, 0]), vi[:, 1]], axis=1)

    def scales(self, ics, floor=0.05, **kw):
        E = self.energy(ics)
        return np.stack([np.maximum(np.arccos(np.exp(-E)), floor), np.maximum(np.sqrt(2.0 * E), floor)], axis=1)


class NonlinearIndLCChargeFlux(System):
    """NonlinearIndLC（3 つ目の回路，C0 = 2，L0 = I0 = 1）を電荷 q と磁束 φ の変数で書いた版（2026-10-22，同じ回路を両方の変数で比べる）.

    q = C0 V，φ = L0 I0 arctan(I/I0) なので dq/dt = I = I0 tan(φ/(L0 I0))，dφ/dt = −V = −q/C0。
    H = q²/(2 C0) − L0 I0² ln cos(φ/(L0 I0))（= NonlinearIndLC の H）。初期条件は NonlinearIndLC.sample_ics と同じ乱数の引き方で作って変える。
    """

    def __init__(self, C0: float = 2.0, L0: float = 1.0, I0: float = 1.0) -> None:
        super().__init__("nlindqf", 2, ("q", "phi"), True, {"C0": C0, "L0": L0, "I0": I0}, omega=math.sqrt(1.0 / (C0 * L0)))

    def rhs(self, s):
        p = self.params
        q, f = s[:, 0], s[:, 1]
        return np.stack([p["I0"] * np.tan(f / (p["L0"] * p["I0"])), -q / p["C0"]], axis=1)

    def energy(self, s):
        p = self.params
        q, f = s[..., 0], s[..., 1]
        return q * q / (2.0 * p["C0"]) - p["L0"] * p["I0"] ** 2 * np.log(np.cos(f / (p["L0"] * p["I0"])))

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        p = self.params
        vi = NonlinearIndLC(p["C0"], p["L0"], p["I0"]).sample_ics(n, lo, hi, rng)
        return np.stack([p["C0"] * vi[:, 0], p["L0"] * p["I0"] * np.arctan(vi[:, 1] / p["I0"])], axis=1)

    def scales(self, ics, floor=0.05, **kw):
        p = self.params
        E = self.energy(ics)
        qm = np.sqrt(2.0 * p["C0"] * E)
        fm = p["L0"] * p["I0"] * np.arccos(np.exp(-E / (p["L0"] * p["I0"] ** 2)))
        return np.stack([np.maximum(qm, floor), np.maximum(fm, floor)], axis=1)


def _cubic_inv(y):
    """x + x³ = y の実数解（単調なので 1 つ。カルダノの式）."""
    y = np.asarray(y, dtype=np.float64)
    r = np.sqrt(0.25 * y * y + 1.0 / 27.0)
    return np.cbrt(0.5 * y + r) + np.cbrt(0.5 * y - r)


class CubicLCVoltageCurrent(System):
    """CubicChargeFluxLC（3 次の回路）を電圧 V と電流 I の変数で書いた版（2026-10-22，同じ回路を両方の変数で比べる）.

    V = q + q³，I = φ + φ³ なので dV/dt = (1 + 3q²) I，dI/dt = −(1 + 3φ²) V（q，φ は 3 次式の逆関数で V，I から求める）。
    それぞれの式が自分の変数にも依る（C(V) = 1/(1 + 3q(V)²)，L(I) = 1/(1 + 3φ(I)²)）。H は CubicChargeFluxLC と同じ。
    初期条件は CubicChargeFluxLC.sample_ics と同じ乱数の引き方で (q, φ) を作り (V, I) に変える。
    """

    def __init__(self) -> None:
        super().__init__("nlqf3vi", 2, ("V", "I"), True, {}, omega=1.0)

    def rhs(self, s):
        V, I = s[:, 0], s[:, 1]
        q, f = _cubic_inv(V), _cubic_inv(I)
        return np.stack([(1.0 + 3.0 * q * q) * I, -(1.0 + 3.0 * f * f) * V], axis=1)

    def energy(self, s):
        q, f = _cubic_inv(s[..., 0]), _cubic_inv(s[..., 1])
        return 0.5 * q * q + 0.25 * q ** 4 + 0.5 * f * f + 0.25 * f ** 4

    def region_value(self, ics):
        return self.energy(ics)

    def sample_ics(self, n, lo, hi, rng, **kw):
        qf = CubicChargeFluxLC().sample_ics(n, lo, hi, rng)
        return qf + qf ** 3

    def scales(self, ics, floor=0.05, **kw):
        a = CubicChargeFluxLC._amp(self.energy(ics))
        m = np.maximum(a + a ** 3, floor)
        return np.stack([m, m], axis=1)


class NonlinearIndLC(System):
    """線形の容量と飽和するインダクタの LC 回路（2026-10-19，ELEX）.

    C0 dV/dt = I，L(I) dI/dt = −V，L(I) = L0/(1 + I²/I0²)。回路の変数 (V, I) で dV/dt = I/C0，dI/dt = −(1 + I²/I0²) V/L0。
    エネルギー H = C0 V²/2 + (L0 I0²/2) ln(1 + I²/I0²) が保存される。電圧の更新（位置側）は I に比例する運動学の形だが，
    係数 1/C0 はモデルに知らせない（既定 C0 = 2）。そのため ẋ = v を仮定する LF-K・LF-S・RK4-K は表せず，位置の更新を学ぶ必要がある。
    力（電流の更新）は I について非線形で，学習の範囲（H ≤ 1）で ∂a/∂V = −(1 + I²/I0²)/L0 は約 7.4 倍変わる。領域の量はエネルギー E。
    """

    def __init__(self, C0: float = 2.0, L0: float = 1.0, I0: float = 1.0) -> None:
        super().__init__("nlind", 2, ("V", "I"), True, {"C0": C0, "L0": L0, "I0": I0}, omega=math.sqrt(1.0 / (C0 * L0)))

    def rhs(self, s):
        p = self.params
        V, I = s[:, 0], s[:, 1]
        return np.stack([I / p["C0"], -(1.0 + (I / p["I0"]) ** 2) * V / p["L0"]], axis=1)

    def energy(self, s):
        p = self.params
        V, I = s[..., 0], s[..., 1]
        return 0.5 * p["C0"] * V * V + 0.5 * p["L0"] * p["I0"] ** 2 * np.log1p((I / p["I0"]) ** 2)

    def region_value(self, ics):
        return self.energy(ics)

    def _amps(self, E):
        p = self.params
        vmax = np.sqrt(2.0 * E / p["C0"])
        imax = p["I0"] * np.sqrt(np.expm1(2.0 * E / (p["L0"] * p["I0"] ** 2)))
        return vmax, imax

    def sample_ics(self, n, lo, hi, rng, **kw):
        vm, im = self._amps(hi)
        out = []
        while len(out) < n:
            m = 4 * (n - len(out)) + 16
            cand = np.stack([rng.uniform(-vm, vm, m), rng.uniform(-im, im, m)], axis=1)
            E = self.energy(cand)
            out.extend(cand[(E > lo) & (E <= hi)].tolist())
        return np.array(out[:n])

    def scales(self, ics, floor=0.05, **kw):
        vm, im = self._amps(self.energy(ics))
        return np.stack([np.maximum(vm, floor), np.maximum(im, floor)], axis=1)


class ChargedParticle(_MultiDOF):
    """一様な磁場（z 方向，強さ B）の中で非調和な井戸に閉じ込めた荷電粒子（質量・電荷 1，平面内の運動）.

    dq/dt = v，dv/dt = −∇V(q) + B (v_y, −v_x)，V = (x² + κ y²)/2 + β (x² + y²)²/4（既定 B = 1，κ = 2，β = 1）。
    ローレンツ力は仕事をしないので H = |v|²/2 + V は保存されるが，力が速度に依存する（力が位置だけの形では表せない）。
    状態 s = (x, y, v_x, v_y)。領域の量はエネルギー E。
    """

    def __init__(self, B: float = 1.0, kappa: float = 2.0, beta: float = 1.0) -> None:
        super().__init__("charged", 4, ("x", "y", "vx", "vy"), True, {"B": B, "kappa": kappa, "beta": beta},
                         omega=1.0, n_dof=2)

    def rhs(self, s):
        p = self.params
        x, y, vx, vy = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        r2 = x * x + y * y
        ax = -x - p["beta"] * r2 * x + p["B"] * vy
        ay = -p["kappa"] * y - p["beta"] * r2 * y - p["B"] * vx
        return np.stack([vx, vy, ax, ay], axis=1)

    def quad(self, s):
        k = self.params["kappa"]
        return 0.5 * (s[..., 2] ** 2 + s[..., 3] ** 2 + s[..., 0] ** 2 + k * s[..., 1] ** 2)

    def quart(self, s):
        return 0.25 * self.params["beta"] * (s[..., 0] ** 2 + s[..., 1] ** 2) ** 2


class FPUChain(_MultiDOF):
    """両端固定の β-FPU 鎖（質点 n 個，質量 1）: V = Σ_{i=0..n} [ (d_i)²/2 + β (d_i)⁴/4 ]，d_i = q_{i+1} − q_i，q_0 = q_{n+1} = 0.

    既定 n = 4，β = 1。線形の角振動数は 2 sin(jπ / (2(n+1)))（n = 4 で最大 1.90）。
    状態 s = (q_1..q_n, v_1..v_n)。領域の量は全エネルギー E。
    """

    def __init__(self, n: int = 4, beta: float = 1.0) -> None:
        n = int(n)
        names = tuple(f"q{i + 1}" for i in range(n)) + tuple(f"v{i + 1}" for i in range(n))
        super().__init__("fpu", 2 * n, names, True, {"n": n, "beta": beta}, omega=1.0, n_dof=n)

    def _diffs(self, q):
        z = np.zeros(q.shape[:-1] + (1,))
        return np.diff(np.concatenate([z, q, z], axis=-1), axis=-1)  # (..., n+1)

    def rhs(self, s):
        n, b = self.params["n"], self.params["beta"]
        q, v = s[:, :n], s[:, n:]
        d = self._diffs(q)
        f = d + b * d ** 3  # ばねの張力
        return np.concatenate([v, f[:, 1:] - f[:, :-1]], axis=1)

    def quad(self, s):
        n = self.params["n"]
        d = self._diffs(s[..., :n])
        return 0.5 * (s[..., n:] ** 2).sum(-1) + 0.5 * (d * d).sum(-1)

    def quart(self, s):
        n = self.params["n"]
        d = self._diffs(s[..., :n])
        return 0.25 * self.params["beta"] * (d ** 4).sum(-1)


class ChargeFluxLine(_MultiDOF):
    """3 次の容量とインダクタを環状に並べた非線形伝送線路（電荷・磁束の変数，2026-10-21 夕，ELEX）.

    段 k の容量の電荷 q_k と，段 k−1 と k をつなぐインダクタの磁束 φ_k（k は n で回る）。
    dq_k/dt = i(φ_k) − i(φ_{k+1})，dφ_k/dt = v(q_{k−1}) − v(q_k)，v(q) = q + q³（容量の電圧），i(φ) = φ + φ³（インダクタの電流）。
    H = Σ_k (q²/2 + q⁴/4 + φ²/2 + φ⁴/4) と，全電荷 Σq・全磁束 Σφ が保存される（分離型のハミルトン系）。
    初期条件：q と φ のそれぞれを，環に乗る波数 m = 1..n/2（空間の角周波数 κ = 2πm/n）の正弦波の和にする。係数は標準正規で，
    振幅の重み exp(−κ²/(4κ0²))（分散では exp(−κ²/(2κ0²))），κ0 = 1（波長 6 段ほど）を掛ける。1 段あたりの統計は段数に依らず，長い線路には学習の線路にない長い波も入る。
    最後に，1 段あたりのエネルギー E/n が (lo, hi] の一様乱数になるように縮める。領域の量は E/n。
    """

    KAPPA0 = 1.0

    def __init__(self, n: int = 8) -> None:
        n = int(n)
        if n < 4 or n % 2:
            raise ValueError("段数 n は 4 以上の偶数にする")
        names = tuple(f"q{k}" for k in range(n)) + tuple(f"phi{k}" for k in range(n))
        super().__init__("nltl", 2 * n, names, True, {"n": n}, omega=1.0, n_dof=n)

    def rhs(self, s):
        n = self.params["n"]
        q, f = s[:, :n], s[:, n:]
        i, v = f + f ** 3, q + q ** 3
        dq = i - np.roll(i, -1, axis=1)  # i(φ_k) − i(φ_{k+1})
        df = np.roll(v, 1, axis=1) - v  # v(q_{k−1}) − v(q_k)
        return np.concatenate([dq, df], axis=1)

    def quad(self, s):
        return 0.5 * (s * s).sum(-1)

    def quart(self, s):
        return 0.25 * (s ** 4).sum(-1)

    def region_value(self, ics):
        return self.energy(ics) / self.params["n"]

    def sample_ics(self, n, lo, hi, rng, **kw):
        cells = self.params["n"]
        k = np.arange(cells)
        m = np.arange(1, cells // 2 + 1)
        kappa = 2.0 * np.pi * m / cells
        w = np.sqrt(np.exp(-kappa ** 2 / (2.0 * self.KAPPA0 ** 2)))  # 振幅の重み
        out = np.zeros((n, 2 * cells))
        for j in range(2):
            a = rng.standard_normal((n, len(m))) * w
            b = rng.standard_normal((n, len(m))) * w
            out[:, j * cells:(j + 1) * cells] = a @ np.cos(np.outer(kappa, k)) + b @ np.sin(np.outer(kappa, k))
        target = rng.uniform(lo, hi, n) * cells  # 全エネルギーの目標
        q2, q4 = self.quad(out), self.quart(out)
        s2 = (-q2 + np.sqrt(q2 * q2 + 4.0 * q4 * target)) / (2.0 * q4)  # s² q2 + s⁴ q4 = target
        return out * np.sqrt(s2)[:, None]


class ChargeFluxLossyLine(ChargeFluxLine):
    """ChargeFluxLine の各段の容量に漏れ（並列のコンダクタンス G）を足した損失のある線路（2026-10-21 夜，ELEX：
    LF が要る場面を試す）.

    dq_k/dt = i(φ_k) − i(φ_{k+1}) − G u(q_k)，dφ_k/dt = u(q_{k−1}) − u(q_k)，u(q) = q + q³（容量の電圧），i(φ) = φ + φ³。
    電荷の更新が自分の電荷にも依るので，それぞれの網に相手の変数だけを渡す LF-Sep は損失を表せない。
    エネルギー H（ChargeFluxLine と同じ式）は dH/dt = −G Σ_k u(q_k)² で減る。初期条件の作り方と領域（初期の 1 段あたりのエネルギー）は同じ。
    """

    def __init__(self, n: int = 8, G: float = 0.03) -> None:
        super().__init__(n)
        self.name = "nltlg"
        self.params = {"n": int(n), "G": float(G)}

    def rhs(self, s):
        out = super().rhs(s)
        n = self.params["n"]
        q = s[:, :n]
        out[:, :n] = out[:, :n] - self.params["G"] * (q + q ** 3)
        return out


class VaractorLine(ChargeFluxLine):
    """バラクタ（逆バイアスの接合容量）と線形のインダクタを環状に並べた線路（2026-10-22，ELEX：物理の単位の例）.

    物理の値：接合容量 C_j0 = 1 pF，内蔵電位 φ_bi = 0.7 V，指数 m = 0.5（階段接合），逆バイアス V_b = 2 V で C_b = C_j0/√(1 + V_b/φ_bi)
    = 0.509 pF，インダクタ L = 1.27 nH（Z0 = √(L/C_b) = 50 Ω）。単位：電圧 1 V，電荷 C_b·1 V，時間 t0 = √(L C_b) = 25.4 ps，電流 1 V/Z0 = 20 mA。
    バイアス点からの電荷 q と電圧 u の関係は，m = 0.5 の接合の式 Q(V) = 2 C_j0 φ_bi (√(1 + V/φ_bi) − 1) を逆に解いて
    u(q) = q + α q²，α = (C_b/C_j0)² (1 V)/(4 φ_bi) = 0.0925。インダクタは線形で i(φ) = φ。
    dq_k/dt = i(φ_k) − i(φ_{k+1})，dφ_k/dt = u(q_{k−1}) − u(q_k)。H = Σ_k (q²/2 + α q³/3 + φ²/2)（分離型のハミルトン系）。
    初期条件：パルスが通る線路の状態（バイアス点から電圧が 0〜3 V 上がり，前に進む波で磁束も 0〜3）を覆うように，直流の偏り
    q_dc ∈ [−0.5, 2.6]，φ_dc ∈ [−1, 3.2]（一様）に，ChargeFluxLine と同じ形の正弦波の和を 1 段あたりの二乗平均平方根 a（(lo, hi] の一様，
    q と φ で別々）で足す。領域の量は a ではなく，初期の電圧の最大 max_k |u(q_k)|（報告用）。
    u(q) > −2.7（V_b + u > −φ_bi，順バイアスにならない）は q > −3.5 に当たり，この範囲の初期条件はそれを満たす。
    """

    def __init__(self, n: int = 8, alpha: float = 0.0925) -> None:
        super().__init__(n)
        self.name = "nltlv"
        self.params = {"n": int(n), "alpha": float(alpha)}

    def u(self, q):
        return q + self.params["alpha"] * q * q

    def rhs(self, s):
        n = self.params["n"]
        q, f = s[:, :n], s[:, n:]
        i, v = f, self.u(q)
        dq = i - np.roll(i, -1, axis=1)
        df = np.roll(v, 1, axis=1) - v
        return np.concatenate([dq, df], axis=1)

    def quad(self, s):
        return 0.5 * (s * s).sum(-1)

    def quart(self, s):  # 名前は基底の都合（ここでは 3 次の項）
        n = self.params["n"]
        q = s[..., :n]
        return (self.params["alpha"] / 3.0) * (q ** 3).sum(-1)

    def region_value(self, ics):
        n = self.params["n"]
        return np.abs(self.u(ics[:, :n])).max(axis=1)

    def sample_ics(self, n, lo, hi, rng, **kw):
        cells = self.params["n"]
        k = np.arange(cells)
        m = np.arange(1, cells // 2 + 1)
        kappa = 2.0 * np.pi * m / cells
        w = np.sqrt(np.exp(-kappa ** 2 / (2.0 * self.KAPPA0 ** 2)))
        out = np.zeros((n, 2 * cells))
        for j in range(2):
            a = rng.standard_normal((n, len(m))) * w
            b = rng.standard_normal((n, len(m))) * w
            shape = a @ np.cos(np.outer(kappa, k)) + b @ np.sin(np.outer(kappa, k))
            shape /= np.sqrt((shape ** 2).mean(axis=1, keepdims=True)) + 1e-12  # 1 段あたりの二乗平均平方根を 1 に
            out[:, j * cells:(j + 1) * cells] = rng.uniform(lo, hi, n)[:, None] * shape
        out[:, :cells] += rng.uniform(-0.5, 2.6, n)[:, None]
        out[:, cells:] += rng.uniform(-1.0, 3.2, n)[:, None]
        return out


# 既定の領域（lo, hi）：学習・検証・内挿と外挿
DEFAULT_REGIONS = {
    "harmonic": {"train": (0.05, 1.0), "extrap": (1.0, 1.2)},
    "pendulum": {"train": (0.05, math.pi / 3), "extrap": (math.pi / 3, math.pi / 2)},
    "vanderpol": {"train": (0.05, 2.0), "extrap": (2.0, 3.0)},
    "duffing": {"train": (0.05, 1.0), "extrap": (1.0, 1.2)},
    "lorenz": {"train": (0.0, 0.0), "extrap": (0.5, 1.0)},
    # 以下は領域の量がエネルギー E。調和振動子の振幅 0.05〜1.0（E = A²/2 で 0.00125〜0.5），外挿 1.0〜1.2（0.5〜0.72）に合わせた
    "doublewell": {"train": (-0.24, 0.5), "extrap": (0.5, 0.72)},  # 井戸の底 −0.25 の近くから山（0）を越えるまで
    "morse": {"train": (0.00125, 0.5), "extrap": (0.5, 0.72)},
    "charged": {"train": (0.00125, 0.5), "extrap": (0.5, 0.72)},
    "nlcap": {"train": (0.00125, 0.5), "extrap": (0.5, 0.72)},  # 電圧に依存する容量の LC 回路（2026-10-18）
    "nllc": {"train": (0.00125, 0.5), "extrap": (0.5, 0.72)},  # 容量もインダクタンスも非線形の LC 回路（2026-10-19）
    "nlind": {"train": (0.0025, 1.0), "extrap": (1.0, 1.44)},  # 線形の容量（値は未知）と飽和するインダクタの LC 回路（2026-10-19）
    "nllcqf": {"train": (0.0025, 1.0), "extrap": (1.0, 1.44)},
    "nlqf3": {"train": (0.0025, 1.0), "extrap": (1.0, 1.44)},
    "nlcapqf": {"train": (0.00125, 0.5), "extrap": (0.5, 0.72)},  # nlcap を電荷・磁束で（2026-10-22）。同じ初期条件
    "nlindqf": {"train": (0.0025, 1.0), "extrap": (1.0, 1.44)},  # nlind を電荷・磁束で（2026-10-22）
    "nlqf3vi": {"train": (0.0025, 1.0), "extrap": (1.0, 1.44)},  # nlqf3 を電圧・電流で（2026-10-22）
    "nltl": {"train": (0.0025, 0.5), "extrap": (0.5, 0.72)},
    "nltlg": {"train": (0.0025, 0.5), "extrap": (0.5, 0.72)},  # 損失のある線路（2026-10-21 夜），初期の 1 段あたりのエネルギー  # 非線形伝送線路，1 段あたりのエネルギー（2026-10-21 夕）  # 3 次の容量とインダクタ，電荷・磁束の変数（2026-10-21 午後，確認の実験）  # nllc を電荷・磁束の変数で（2026-10-21）。電圧・電流の版も同じ領域で比べる
    "fpu": {"train": (0.005, 2.0), "extrap": (2.0, 2.88)},  # 質点 4 個の合計（1 個あたり 0.00125〜0.5）
    "nltlv": {"train": (0.05, 1.0), "extrap": (1.0, 1.44)},  # バラクタの線路（2026-10-22）：直流の偏りに足す波の 1 段あたりの二乗平均平方根
}


def make_system(name: str, **params) -> System:
    name = name.lower()
    if name == "harmonic":
        return Harmonic(**params)
    if name == "pendulum":
        return Pendulum(**params)
    if name in ("vanderpol", "vdp"):
        return VanDerPol(**params)
    if name == "lorenz":
        return Lorenz(**params)
    if name == "duffing":
        return Duffing(**params)
    if name == "doublewell":
        return DoubleWell(**params)
    if name == "morse":
        return Morse(**params)
    if name == "charged":
        return ChargedParticle(**params)
    if name == "fpu":
        return FPUChain(**params)
    if name == "nlcap":
        return NonlinearCapLC(**params)
    if name == "nllc":
        return NonlinearLC(**params)
    if name == "nlind":
        return NonlinearIndLC(**params)
    if name == "nllcqf":
        return NonlinearLCChargeFlux(**params)
    if name == "nlqf3":
        return CubicChargeFluxLC(**params)
    if name == "nlcapqf":
        return NonlinearCapLCChargeFlux(**params)
    if name == "nlindqf":
        return NonlinearIndLCChargeFlux(**params)
    if name == "nlqf3vi":
        return CubicLCVoltageCurrent(**params)
    if name == "nltl":
        return ChargeFluxLine(**params)
    if name == "nltlg":
        return ChargeFluxLossyLine(**params)
    if name == "nltlv":
        return VaractorLine(**params)
    raise ValueError(f"unknown system: {name}")


def system_from_manifest(manifest: dict) -> System:
    return make_system(manifest.get("system", "harmonic"), **manifest.get("params", {}))
