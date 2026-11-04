#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_harmonic モデル定義（state_dict 互換・IC汎化学習用）."""

from __future__ import annotations

import torch
import torch.nn as nn

from harmonic_common import LEGACY_MODEL_ALIASES, canonical_model_name


# LTC 更新則の種類（2026-09-28 追加）
#   legacy:          旧実装。分母に τ_sys = τ/(1+τf) を代入して 1 + Δt(1/τ + 2f) になり、τ = 1 固定。
#                    g = 2f, A' = A/2 と置けば原著と同じ関数族（表現力は同じ）。
#   paper_fixed_tau: 分母を原著 (Hasani et al. 2021, 式(3)) の 1 + Δt(1/τ + f) に修正、τ = 1 固定。
#   paper:           paper_fixed_tau に加え、τ をユニットごとの学習パラメータにする
#                    （τ = softplus(ρ)、初期値 τ = 1）。
LTC_VARIANTS = ("legacy", "paper_fixed_tau", "paper")
# f の活性化関数。原著の定理（τ_sys の上下界）は有界・非負の sigmoid を仮定している。
LTC_ACTIVATIONS = ("relu", "sigmoid")
DEFAULT_LTC_VARIANT = "paper"
DEFAULT_LTC_ACTIVATION = "relu"

_SOFTPLUS_INV_ONE = 0.5413248546129181  # log(e - 1): softplus(ρ) = 1


class _LTCMixin:
    """LTC 系セルの共通部分（更新則・活性化関数の切り替え）."""

    def _init_ltc(
        self,
        hidden: int,
        ltc_variant: str,
        ltc_activation: str,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
    ) -> None:
        if ltc_variant not in LTC_VARIANTS:
            raise ValueError(f"unknown ltc_variant: {ltc_variant}")
        if ltc_activation not in LTC_ACTIVATIONS:
            raise ValueError(f"unknown ltc_activation: {ltc_activation}")
        self.ltc_variant = ltc_variant
        self.ltc_activation = ltc_activation
        # 1 入力あたりの展開回数 L と積分時間 T（2026-09-30 追加）。
        # 既定（L = 1，T = None）は従来どおり物理の刻み Δt を 1 回で積分する。
        # 原著（Hasani et al. 2021 付録 表 S1，公式実装の ode_solver_unfolds = 6）は L = 6，T = 入力の周期 = 1。
        if int(ltc_unfolds) < 1:
            raise ValueError(f"ltc_unfolds must be >= 1: {ltc_unfolds}")
        self.ltc_unfolds = int(ltc_unfolds)
        self.ltc_int_time = None if ltc_int_time is None else float(ltc_int_time)
        if ltc_variant == "paper":
            self.tau_raw = nn.Parameter(torch.full((hidden,), _SOFTPLUS_INV_ONE))
        else:
            self.tau_fixed = 1.0

    @property
    def tau(self):
        if self.ltc_variant == "paper":
            return nn.functional.softplus(self.tau_raw)
        return self.tau_fixed

    def _ltc_act(self, z):
        return torch.sigmoid(z) if self.ltc_activation == "sigmoid" else torch.relu(z)

    def _ltc_integrate(self, h, u, W_hh, dt):
        """1 回の入力の間に LTC の状態を進める. u は入力側の前活性（W·x + b）で，展開の間は固定.

        T = ltc_int_time（未指定なら物理の刻み dt）を L 回に分け，毎回その時点の h から f を計算し直す
        （公式実装と同じ）。KDK3 の半ステップのように dt が delta_t の定数倍のときは T も同じ倍率にする。
        """
        T = dt if self.ltc_int_time is None else self.ltc_int_time * (dt / self.delta_t)
        d = T / self.ltc_unfolds
        for _ in range(self.ltc_unfolds):
            f = self._ltc_act(u + W_hh(h))
            h = self._ltc_step(h, f, d)
        return h

    def _ltc_step(self, h, f, dt):
        tau = self.tau
        if self.ltc_variant == "legacy":
            tau_dynamic = tau / (1 + tau * f)
            return (h + dt * f * self.A) / (1 + dt * ((1 / tau_dynamic) + f))
        return (h + dt * f * self.A) / (1 + dt * ((1 / tau) + f))


class LF_RNN(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.W_xh = nn.Linear(2, hidden)
        self.W_xhh = nn.Linear(hidden, hidden)
        self.W_xout = nn.Linear(hidden, 1)
        self.W_vh = nn.Linear(2, hidden)
        self.W_vhh = nn.Linear(hidden, hidden)
        self.W_vout = nn.Linear(hidden, 1)
        self.delta_t = 0.05

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros_like(h_prev)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        if not is_velocity_step:
            f = torch.relu(self.W_xh(x) + self.W_xhh(h_prev_prev))
        else:
            f = torch.relu(self.W_vh(x) + self.W_vhh(h_prev_prev))
        h_new = h_prev_prev + self.delta_t * f
        output = self.W_vout(h_new) if is_velocity_step else self.W_xout(h_new)
        return output, h_new, h_prev


class LF_LTC(_LTCMixin, nn.Module):
    def __init__(
        self,
        hidden: int,
        ltc_variant: str = DEFAULT_LTC_VARIANT,
        ltc_activation: str = DEFAULT_LTC_ACTIVATION,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
    ) -> None:
        super().__init__()
        self.W_xh = nn.Linear(2, hidden)
        self.W_xhh = nn.Linear(hidden, hidden)
        self.W_xout = nn.Linear(hidden, 1)
        self.W_vh = nn.Linear(2, hidden)
        self.W_vhh = nn.Linear(hidden, hidden)
        self.W_vout = nn.Linear(hidden, 1)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 0.05
        self._init_ltc(hidden, ltc_variant, ltc_activation, ltc_unfolds, ltc_int_time)

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros_like(h_prev)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        if not is_velocity_step:
            h_new = self._ltc_integrate(h_prev_prev, self.W_xh(x), self.W_xhh, self.delta_t)
        else:
            h_new = self._ltc_integrate(h_prev_prev, self.W_vh(x), self.W_vhh, self.delta_t)
        output = self.W_vout(h_new) if is_velocity_step else self.W_xout(h_new)
        return output, h_new, h_prev


class DirectRNN(nn.Module):
    """1 段更新 RNN（W_hh は標準 Elman 型の Linear(hidden, hidden)）.

    2026-07-10 以前は W_hh: Linear(hidden, 1) で、隠れ状態をスカラー1個に
    潰してブロードキャストする rank-1 ボトルネックだった（実装バグと判断）。
    現行の full 結合では hs=24 で総パラメータ 673 と LF_RNN(hs=16) の 674 に
    ほぼ一致する（等パラメータ比較）。旧チェックポイント（W_hh が (1, hs)）は
    本クラスでは load できない点に注意。
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.W_xh = nn.Linear(1, hidden)
        self.W_hh = nn.Linear(hidden, hidden)
        self.W_hout = nn.Linear(hidden, 1)
        self.delta_t = 0.025

    def forward(self, x, h):
        f = torch.relu(self.W_xh(x) + self.W_hh(h))
        h_new = h + self.delta_t * f
        return self.W_hout(h_new), h_new


class DirectLTC(_LTCMixin, nn.Module):
    """1 段更新 LTC（W_hh は Linear(hidden, hidden)。旧 rank-1 実装の注意は DirectRNN と同じ）.

    hs=24 で総パラメータ 697 と LF_LTCs(hs=16) の 690 にほぼ一致する.
    """

    def __init__(
        self,
        hidden: int,
        ltc_variant: str = DEFAULT_LTC_VARIANT,
        ltc_activation: str = DEFAULT_LTC_ACTIVATION,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
    ) -> None:
        super().__init__()
        self.W_xh = nn.Linear(1, hidden)
        self.W_hh = nn.Linear(hidden, hidden)
        self.W_hout = nn.Linear(hidden, 1)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 0.025
        self._init_ltc(hidden, ltc_variant, ltc_activation, ltc_unfolds, ltc_int_time)

    def forward(self, x, h):
        h_new = self._ltc_integrate(h, self.W_xh(x), self.W_hh, self.delta_t)
        return self.W_hout(h_new), h_new


class DirectStateRNN(nn.Module):
    """1 段更新 RNN の状態入力版（2026-09-29 追加）: (x_n, v_n) → (x_{n+1}, v_{n+1}).

    DirectRNN は x だけを入出力にするため初速 v0 を受け取れない。LF 系と同じ情報
    （初期条件 (x0, v0)）を与えたうえで、更新の骨格だけが違う対照群にする。
    hs=23 で総パラメータ 669（LF_RNN hs=16 の 674 とほぼ同じ）。
    """

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__()
        self.W_xh = nn.Linear(dim, hidden)
        self.W_hh = nn.Linear(hidden, hidden)
        self.W_hout = nn.Linear(hidden, dim)
        self.delta_t = 0.025

    def forward(self, xv, h):
        f = torch.relu(self.W_xh(xv) + self.W_hh(h))
        h_new = h + self.delta_t * f
        return self.W_hout(h_new), h_new


class DirectStateLTC(_LTCMixin, nn.Module):
    """1 段更新 LTC の状態入力版: (x_n, v_n) → (x_{n+1}, v_{n+1}).

    hs=23 で総パラメータ 715（原著の更新式。LF_LTCs hs=16 の 706 とほぼ同じ）。
    """

    def __init__(
        self,
        hidden: int,
        ltc_variant: str = DEFAULT_LTC_VARIANT,
        ltc_activation: str = DEFAULT_LTC_ACTIVATION,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
        dim: int = 2,
    ) -> None:
        super().__init__()
        self.W_xh = nn.Linear(dim, hidden)
        self.W_hh = nn.Linear(hidden, hidden)
        self.W_hout = nn.Linear(hidden, dim)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 0.025
        self._init_ltc(hidden, ltc_variant, ltc_activation, ltc_unfolds, ltc_int_time)

    def forward(self, xv, h):
        h_new = self._ltc_integrate(h, self.W_xh(xv), self.W_hh, self.delta_t)
        return self.W_hout(h_new), h_new


class SplitStateRNN(nn.Module):
    """LF の効果を切り分ける対照（2026-10-02 追加）: LF_RNN と同じ 2 つのセル（v 用・x 用，それぞれの隠れ状態）を
    同時に更新する。どちらのセルも (x_n, v_n) を読み，x のセルは更新後の v を使わない（蛙跳びの順番がない）。
    刻みは状態入力の 1 段系と同じ（v も整数時刻）。パラメータは LF_RNN と同じ数（hs=16 で 674）。
    隠れ状態は [h_x, h_v] をつないだ幅 2·hidden のベクトルで持つ。
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.W_xh = nn.Linear(2, hidden)
        self.W_xhh = nn.Linear(hidden, hidden)
        self.W_xout = nn.Linear(hidden, 1)
        self.W_vh = nn.Linear(2, hidden)
        self.W_vhh = nn.Linear(hidden, hidden)
        self.W_vout = nn.Linear(hidden, 1)
        self.delta_t = 0.025
        self.cell_hidden = hidden
        self.hidden_size = 2 * hidden

    def forward(self, xv, h):
        hx, hv = h[:, : self.cell_hidden], h[:, self.cell_hidden :]
        hx_new = hx + self.delta_t * torch.relu(self.W_xh(xv) + self.W_xhh(hx))
        hv_new = hv + self.delta_t * torch.relu(self.W_vh(xv) + self.W_vhh(hv))
        return torch.cat([self.W_xout(hx_new), self.W_vout(hv_new)], dim=1), torch.cat([hx_new, hv_new], dim=1)


class SplitStateLTC(_LTCMixin, nn.Module):
    """SplitStateRNN の LTC 版。LF_LTC と同じパラメータ（A・τ は 2 つのセルで共有，hs=16 で 706）."""

    def __init__(
        self,
        hidden: int,
        ltc_variant: str = DEFAULT_LTC_VARIANT,
        ltc_activation: str = DEFAULT_LTC_ACTIVATION,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
    ) -> None:
        super().__init__()
        self.W_xh = nn.Linear(2, hidden)
        self.W_xhh = nn.Linear(hidden, hidden)
        self.W_xout = nn.Linear(hidden, 1)
        self.W_vh = nn.Linear(2, hidden)
        self.W_vhh = nn.Linear(hidden, hidden)
        self.W_vout = nn.Linear(hidden, 1)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 0.025
        self._init_ltc(hidden, ltc_variant, ltc_activation, ltc_unfolds, ltc_int_time)
        self.cell_hidden = hidden
        self.hidden_size = 2 * hidden

    def forward(self, xv, h):
        hx, hv = h[:, : self.cell_hidden], h[:, self.cell_hidden :]
        hx_new = self._ltc_integrate(hx, self.W_xh(xv), self.W_xhh, self.delta_t)
        hv_new = self._ltc_integrate(hv, self.W_vh(xv), self.W_vhh, self.delta_t)
        return torch.cat([self.W_xout(hx_new), self.W_vout(hv_new)], dim=1), torch.cat([hx_new, hv_new], dim=1)


# ---------------------------------------------------------------------------
# 記憶なし・増分型のモデル（2026-10-09 追加）：LF の有無だけを比べるための 3 段階 × 2 部品
#   ① 1 段（Res_*_state）   ：(x, v) ← (x, v) + Δt·F(x, v)                       （陽的 Euler 型，シンプレクティックでない）
#   ② LF・構造だけ（LF_Free_*）：v ← v + Δt·F(x, v)，続いて x ← x + Δt·G(x, v)    （LF の順番，シンプレクティックでない）
#   ③ LF・シンプレクティック（LF_Sym_*）：v ← v + Δt·F(x)，続いて x ← x + Δt·v （せん断の合成，厳密にシンプレクティック）
# 部品 F, G は MLP（2 層，tanh）か LTC 型の関数（毎回 h = 0 から LTC の ODE を L = 6 回の展開・積分時間 T = 1 で積分。
# 入力で時定数が変わる性質は残り，ステップをまたぐ記憶はない）。隠れ状態は持たない（hidden_size = 1 の 0 を受け渡すだけ）。
# LF 系の v は半整数時刻（v_{n−1/2} → v_{n+1/2}）なので，③ は (x_n, v_{n−1/2}) → (x_{n+1}, v_{n+1/2}) の写像として
# 厳密にシンプレクティック（自由度 1 では F が何であってもヤコビ行列の行列式が 1）。
# 多自由度（2026-10-15）では ③ の力を F = −∇V(x)（V はスカラーの MLP）にする。② と ① は出力の次元を増やすだけ。
# ---------------------------------------------------------------------------


class _MLPFunc(nn.Module):
    """2 層の MLP（tanh）: in_dim → width → width → out_dim."""

    def __init__(self, in_dim: int, width: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, out_dim))

    def forward(self, u):
        return self.net(u)


class _LTCFunc(_LTCMixin, nn.Module):
    """LTC 型の関数：毎回 h = 0 から LTC の ODE（原著式，τ 学習）を積分時間 1，6 回の展開で積分して読み出す."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, unfolds: int = 6) -> None:
        super().__init__()
        self.W_in = nn.Linear(in_dim, hidden)
        self.W_hh = nn.Linear(hidden, hidden)
        self.W_out = nn.Linear(hidden, out_dim)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 1.0
        self._init_ltc(hidden, DEFAULT_LTC_VARIANT, DEFAULT_LTC_ACTIVATION, unfolds, 1.0)
        self.hidden = hidden

    def forward(self, u):
        h = torch.zeros(u.shape[0], self.hidden, device=u.device, dtype=u.dtype)
        return self.W_out(self._ltc_integrate(h, self.W_in(u), self.W_hh, self.delta_t))


def _make_func(kind: str, in_dim: int, width: int, out_dim: int) -> nn.Module:
    if kind == "mlp":
        return _MLPFunc(in_dim, width, out_dim)
    if kind == "ltc":
        return _LTCFunc(in_dim, width, out_dim)
    raise ValueError(kind)


class _MemorylessLF(nn.Module):
    """記憶なし・増分型の LF（② と ③）。LF 系と同じ呼び出し方（is_velocity_step で速度と位置の段を切り替える）."""

    kind = "mlp"
    symplectic = True
    learned_drift = True  # False なら位置の段は x ← x + Δt·v_new に固定（速度の段は F(x, v)）

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__()
        k = dim // 2  # 自由度（2026-10-15：多自由度の系。状態は (q_1..q_k, v_1..v_k)）
        if self.symplectic:
            # 力は x だけの関数。多自由度ではポテンシャル V(x)（スカラー）の勾配 F = −∇V にする
            # （力が勾配でないとキックがシンプレクティックにならない。1 自由度はどんな F でも勾配なので従来どおり）
            self.F = _make_func(self.kind, k, hidden, k if k == 1 else 1)
        else:
            self.F = _make_func(self.kind, 2 * k, hidden, k)  # F(x, v)
            if self.learned_drift:
                self.G = _make_func(self.kind, 2 * k, hidden, k)  # G(x, v)
        self.n_dof = k
        self.delta_t = 0.05
        self.hidden_size = 1

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k = self.n_dof
        pos, vel = x[:, :k], x[:, k:2 * k]
        dt = self.delta_t
        if is_velocity_step:
            out = vel + dt * (self._force(pos) if self.symplectic else self.F(x))
        else:
            out = pos + dt * (vel if self.symplectic or not self.learned_drift else self.G(x))
        return out, torch.zeros_like(h_prev_prev), h_prev


    def _force(self, pos):
        if self.n_dof == 1:
            return self.F(pos)
        create = torch.is_grad_enabled()  # 学習中は勾配の勾配を残す。評価（no_grad）では残さない
        with torch.enable_grad():
            q = pos if pos.requires_grad else pos.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self.F(q).sum(), q, create_graph=create)
        return -g


class LFSymMLP(_MemorylessLF):
    kind, symplectic = "mlp", True


class LFSymLTC(_MemorylessLF):
    kind, symplectic = "ltc", True


class LFFreeMLP(_MemorylessLF):
    kind, symplectic = "mlp", False


class LFFreeLTC(_MemorylessLF):
    kind, symplectic = "ltc", False


class LFKickMLP(_MemorylessLF):
    """② の位置の段を x ← x + Δt·v_new に固定した版（2026-10-16）：v ← v + Δt·F(x, v)，x ← x + Δt·v_new.

    GNS（Sanchez-Gonzalez ら 2020）と同じ形。力は一般の F(x, v) なので速度に依存する力も表せるが，
    シンプレクティックではない。② との差は「位置の段を学習するか」だけ。
    """

    kind, symplectic, learned_drift = "mlp", False, False


class LFParMLP(_MemorylessLF):
    """② から LF の順番だけを外した対照（2026-10-16）：v ← v + Δt·F(x, v)，x ← x + Δt·G(x, v_old).

    データ（半ステップずれた速度）・損失・ネットワーク・パラメータ数は ② と同じで，位置の更新が
    更新前の速度を見る点だけが違う。更新前の速度は，速度の段の出力 h_new として位置の段へ渡す
    （同じ時刻ステップの中だけで使い，ステップをまたぐ記憶はない）。
    """

    kind, symplectic = "mlp", False

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__(hidden, dim)
        self.hidden_size = self.n_dof

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        k = self.n_dof
        pos, vel = x[:, :k], x[:, k:2 * k]
        if h_prev_prev is None:
            h_prev_prev = torch.zeros_like(pos)
        if is_velocity_step:
            return vel + self.delta_t * self.F(x), vel, h_prev  # 更新前の速度を位置の段へ
        v_old = h_prev if h_prev is not None else vel
        return pos + self.delta_t * self.G(torch.cat([pos, v_old], dim=1)), torch.zeros_like(pos), h_prev


class LFSepMLP(_MemorylessLF):
    """リープフロッグの 2 つの更新を，それぞれ相手の変数だけを受け取る網で学ぶ版（2026-10-21，ELEX）.

    v_{n+1/2} = v_{n-1/2} + Δt·F(x_n)，x_{n+1} = x_n + Δt·G(v_{n+1/2})。電荷・磁束の変数の回路（q̇ = i(φ)，φ̇ = −v(q)）のように
    分離型の形が分かっているときの学習型のシュテルマー・ベルレ法（SRNN・SympNets と同じ組み立てで，網は勾配に限らない）。
    1 自由度ではどんな F・G でも面積を保つ（ずれの合成）。LF（LF_Free_MLP）と同じ幅でパラメータ数はほぼ同じ（入力が半分）。
    """

    kind, symplectic = "mlp", False

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        k = dim // 2
        self.F = _make_func("mlp", k, hidden, k)  # 速度の増分（位置だけ）
        self.G = _make_func("mlp", k, hidden, k)  # 位置の増分（速度だけ）
        self.n_dof = k
        self.delta_t = 0.05
        self.hidden_size = 1

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k = self.n_dof
        pos, vel = x[:, :k], x[:, k:2 * k]
        if is_velocity_step:
            out = vel + self.delta_t * self.F(pos)
        else:
            out = pos + self.delta_t * self.G(vel)
        return out, torch.zeros_like(h_prev_prev), h_prev


class LFCenMLP(_MemorylessLF):
    """LF の 2 つの更新に，自分自身の変数も中点で渡す版（2026-10-21 夕，ELEX）.

    v_{n+1/2} = v_{n-1/2} + Δt·F(x_n, (v_{n-1/2} + v_{n+1/2})/2)，x_{n+1} = x_n + Δt·G((x_n + x_{n+1})/2, v_{n+1/2})。
    どちらも陰的なので，LF の陽的な更新を初期値にした不動点の反復（n_iter 回）で解く。学習（厳密な状態からの 1 ステップの誤差）も
    評価のロールアウトも同じ反復を通す（反復を展開して微分する）。電圧・電流の変数の回路（V̇ = I/C(V)，İ = −V/L(I)）のように，
    更新が自分自身の変数にも依るときの中点からのずれを消す（陰的中点則と同じ考え）。網・幅・パラメータ数は LF（LF_Free_MLP）と同じで，
    1 回の更新の網の評価は n_iter + 1 回。
    """

    kind, symplectic = "mlp", False
    n_iter = 3

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k, dt = self.n_dof, self.delta_t
        pos, vel = x[:, :k], x[:, k:2 * k]
        if is_velocity_step:
            new = vel + dt * self.F(x)
            for _ in range(self.n_iter):
                new = vel + dt * self.F(torch.cat([pos, 0.5 * (vel + new)], dim=1))
        else:
            new = pos + dt * self.G(x)
            for _ in range(self.n_iter):
                new = pos + dt * self.G(torch.cat([0.5 * (pos + new), vel], dim=1))
        return new, torch.zeros_like(h_prev_prev), h_prev


# ---------------------------------------------------------------------------
# 段ごとに同じ網を共有する版（2026-10-21 夕，非線形伝送線路）：環状に並んだ N 段の各段に，前後 LOCAL_R 段の窓を入力にした
# 同じ MLP を当てる。重みは段数 N に依らないので，短い線路で学習して長い線路で試せる（set_length で段数を変える）。
# 状態の並びは (q_0..q_{N-1}, φ_0..φ_{N-1})（チャンネルごとに N 段が続く）。
# ---------------------------------------------------------------------------
LOCAL_R = 2
# 2026-10-23 夜（ソリトンの試験）：段ごとの網の入力を何倍かしてから MLP に入れ，出力を同じ数で割る（関数の集まりは同じで初期値の尺度だけが変わる）。
# 振幅 0.05〜0.3 のデータでは入力が小さく，汎用の網が全部同じ損失（約 3e-9）で止まった。既定 1 は従来どおり。環境変数で与え，学習の config に残す。
import os as _os  # noqa: E402

LOCAL_INPUT_SCALE = float(_os.environ.get("BHIC_LOCAL_INPUT_SCALE", "1"))


class _LocalMLP(nn.Module):
    """入力 (B, in_ch·N) の各段に，前後 r 段の窓（環状）で同じ MLP を当て，(B, out_ch·N) を返す."""

    def __init__(self, in_ch: int, out_ch: int, width: int, r: int = LOCAL_R) -> None:
        super().__init__()
        self.in_ch, self.out_ch, self.r = in_ch, out_ch, r
        self.net = nn.Sequential(nn.Linear(in_ch * (2 * r + 1), width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, out_ch))

    def forward(self, u):
        b, n = u.shape[0], u.shape[1] // self.in_ch
        z = u.reshape(b, self.in_ch, n)
        z = torch.cat([z[:, :, n - self.r:], z, z[:, :, :self.r]], dim=2)  # 環状に前後 r 段を足す
        w = z.unfold(2, 2 * self.r + 1, 1)  # (b, in_ch, n, 2r+1)
        w = w.permute(0, 2, 1, 3).reshape(b, n, self.in_ch * (2 * self.r + 1))
        if LOCAL_INPUT_SCALE != 1.0:
            return (self.net(w * LOCAL_INPUT_SCALE) / LOCAL_INPUT_SCALE).permute(0, 2, 1).reshape(b, self.out_ch * n)
        return self.net(w).permute(0, 2, 1).reshape(b, self.out_ch * n)


class _LocalMixin:
    """段数に依らない（length_agnostic）。評価で線路の段数を変えるときは set_length を呼ぶ."""

    length_agnostic = True

    def set_length(self, n_cells: int) -> None:
        self.n_dof = int(n_cells)
        if getattr(self, "hidden_is_dof", False):
            self.hidden_size = self.n_dof


class LFFreeLoc(_LocalMixin, LFFreeMLP):
    """LF（LF_Free_MLP）の段ごとの網の版：v ← v + Δt·F(窓の x, v)，x ← x + Δt·G(窓の x, v_new)."""

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F, self.G = _LocalMLP(2, 1, hidden), _LocalMLP(2, 1, hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1


class LFParLoc(_LocalMixin, LFParMLP):
    """LF-U（LF_Par_MLP）の段ごとの網の版."""

    hidden_is_dof = True

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F, self.G = _LocalMLP(2, 1, hidden), _LocalMLP(2, 1, hidden)
        self.n_dof, self.delta_t = dim // 2, 0.05
        self.hidden_size = self.n_dof


class LFSepLoc(_LocalMixin, LFSepMLP):
    """LF-Sep の段ごとの網の版：F は x の窓だけ，G は v の窓だけを受け取る."""

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F, self.G = _LocalMLP(1, 1, hidden), _LocalMLP(1, 1, hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1


class LFSepDLoc(_LocalMixin, _MemorylessLF):
    """LF-Sep に損失の項を足した構造つきの版（2026-10-21 夜，ELEX：構造を持つ損失のモデルと比べる）.

    磁束（速度）の更新 φ_{n+1/2} = φ_{n−1/2} + Δt·F(q_n の窓)（LF-Sep と同じ），
    電荷（位置）の更新 q_{n+1} = q_n + Δt·[G(φ_{n+1/2} の窓) + D(q̄ の窓)]，q̄ = (q_n + q_{n+1})/2。
    損失の項 D は中点の電荷で取る（損失のある FDTD の更新と同じ考え）。陰的なので，D(q_n) から始めた不動点の反復 n_iter 回で解き，
    学習も評価も同じ反復を通す。網は 3 つ（F，G，D，どれも入力 1 チャンネルの 5 段の窓）で，幅はパラメータ数がほかの型にそろうように選ぶ。
    """

    kind, symplectic = "mlp", False
    n_iter = 3

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F, self.G, self.D = _LocalMLP(1, 1, hidden), _LocalMLP(1, 1, hidden), _LocalMLP(1, 1, hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k, dt = self.n_dof, self.delta_t
        pos, vel = x[:, :k], x[:, k:2 * k]
        if is_velocity_step:
            return vel + dt * self.F(pos), torch.zeros_like(h_prev_prev), h_prev
        base = pos + dt * self.G(vel)
        new = base + dt * self.D(pos)
        for _ in range(self.n_iter):
            new = base + dt * self.D(0.5 * (pos + new))
        return new, torch.zeros_like(h_prev_prev), h_prev


class LFSepDExpLoc(LFSepDLoc):
    """LF_SepD_Loc の損失の項を区間の始めの電荷 q_n で取る陽的な版（中点にそろえることが効くかを切り分ける対照，2026-10-21 夜）.

    q_{n+1} = q_n + Δt·[G(φ_{n+1/2} の窓) + D(q_n の窓)]。網・幅・パラメータ数は LF_SepD_Loc と同じ。
    """

    n_iter = 0


class _ScalarMLP(nn.Module):
    """全段で共有するスカラーの関数 f: R → R（tanh の隠れ層 2 層）を段ごとに当てる。(B, N) → (B, N)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, u):
        b, n = u.shape
        if LOCAL_INPUT_SCALE != 1.0:
            return (self.net(u.reshape(b * n, 1) * LOCAL_INPUT_SCALE) / LOCAL_INPUT_SCALE).reshape(b, n)
        return self.net(u.reshape(b * n, 1)).reshape(b, n)


class LFKCLLoc(_LocalMixin, _MemorylessLF):
    """素子の特性だけを学び，キルヒホッフの法則でつなぐ版（2026-10-22，ELEX：保存する形の対照）.

    容量の電圧 u_θ(q) とインダクタの電流 i_ψ(φ)（どちらも全段で共有するスカラーの MLP）を学び，環状の線路を古典の
    リープフロッグと同じ組み立てで進める：φ_k ← φ_k + Δt [u_θ(q_{k−1}) − u_θ(q_k)]，q_k ← q_k + Δt [i_ψ(φ_k) − i_ψ(φ_{k+1})]。
    網が出すのは節点の電圧と枝の電流（端子の量）なので，総電荷 Σq と総磁束 Σφ は網が何を学んでも厳密に保存される。
    """

    kind, symplectic = "mlp", False

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.U, self.I = _ScalarMLP(hidden), _ScalarMLP(hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k, dt = self.n_dof, self.delta_t
        q, f = x[:, :k], x[:, k:2 * k]
        if is_velocity_step:
            u = self.U(q)
            return f + dt * (torch.roll(u, 1, dims=1) - u), torch.zeros_like(h_prev_prev), h_prev
        i = self.I(f)
        return q + dt * (i - torch.roll(i, -1, dims=1)), torch.zeros_like(h_prev_prev), h_prev


class LFKCLDLoc(LFKCLLoc):
    """LF_KCL_Loc に，各節点から地への漏れの電流 d_χ(q̄)（全段で共有するスカラーの MLP，中点の電荷）を足した版（2026-10-22）.

    q_k ← q_k + Δt [i_ψ(φ_k) − i_ψ(φ_{k+1}) − d_χ(q̄_k)]，q̄ = (q_n + q_{n+1})/2。陰的なので LF_SepD_Loc と同じく
    d_χ(q_n) から始めた不動点の反復 3 回で解き，学習も評価も同じ反復を通す。総磁束は保存し，総電荷は漏れの分だけ減る。
    """

    n_iter = 3

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__(hidden, dim)
        self.D = _ScalarMLP(hidden)

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        if is_velocity_step:
            return super().forward(x, h_prev_prev, h_prev, True)
        if h_prev_prev is None:
            h_prev_prev = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        k, dt = self.n_dof, self.delta_t
        q, f = x[:, :k], x[:, k:2 * k]
        i = self.I(f)
        base = q + dt * (i - torch.roll(i, -1, dims=1))
        new = base - dt * self.D(q)
        for _ in range(self.n_iter):
            new = base - dt * self.D(0.5 * (q + new))
        return new, torch.zeros_like(h_prev_prev), h_prev



def _init_two_net_local(self, hidden: int, dim: int) -> None:
    nn.Module.__init__(self)
    self.F, self.G = _LocalMLP(2, 1, hidden), _LocalMLP(2, 1, hidden)
    self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1


class _ResidualState(nn.Module):
    """記憶なし・増分型の 1 段系（①）：(x, v) ← (x, v) + Δt·F(x, v)."""

    kind = "mlp"

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__()
        self.F = _make_func(self.kind, dim, hidden, dim)
        self.delta_t = 0.05
        self.hidden_size = 1

    def forward(self, xv, h):
        return xv + self.delta_t * self.F(xv), torch.zeros_like(h)


class ResMLPState(_ResidualState):
    kind = "mlp"


class ResLTCState(_ResidualState):
    kind = "ltc"


class RK4MLPState(_ResidualState):
    """外部の比較相手（2026-10-16）：一般のベクトル場 f(x, v) を RK4 で 1 ステップ進める（ニューラル ODE）.

    ① と同じネットワーク・パラメータ数で，積分法だけを 4 次にした。1 ステップで f を 4 回評価する。
    """

    kind = "mlp"

    def forward(self, xv, h):
        dt, f = self.delta_t, self.F
        k1 = f(xv)
        k2 = f(xv + 0.5 * dt * k1)
        k3 = f(xv + 0.5 * dt * k2)
        k4 = f(xv + dt * k3)
        return xv + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), torch.zeros_like(h)


class RK4KinMLPState(nn.Module):
    """2 階の Neural ODE を RK4 で 1 ステップ進める対照（2026-10-18，ELEX）.

    位置の微分は速度に固定し（x′ = v，Gruver ら 2022 の 2 階の運動学），加速度 v′ = F(x, v) だけを学ぶ。
    LF_Kick_MLP と同じ網・同じ幅・同じパラメータ数で，積分法だけが違う（整数時刻の (x_n, v_n)，RK4 で F を 4 回評価）。
    """

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__()
        k = dim // 2
        self.F = _make_func("mlp", 2 * k, hidden, k)  # 加速度
        self.n_dof = k
        self.delta_t = 0.05
        self.hidden_size = 1

    def forward(self, xv, h):
        k, dt = self.n_dof, self.delta_t

        def f(z):
            return torch.cat([z[:, k:2 * k], self.F(z)], dim=1)

        k1 = f(xv)
        k2 = f(xv + 0.5 * dt * k1)
        k3 = f(xv + 0.5 * dt * k2)
        k4 = f(xv + dt * k3)
        return xv + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), torch.zeros_like(h)


class _TwoNetState(nn.Module):
    """同時刻の (x_n, v_n) を 2 つの網で進める対照（2026-10-18，ELEX）.

    LF_Par_MLP・LF_Free_MLP と同じ 2 つの網 F(x, v)・G(x, v)（同じ幅・同じパラメータ数）を，半ステップずらさない
    速度で使う。① 対 Res2 は「網を 2 つに分ける」効果，Res2 対 LF_Par は「速度を半ステップずらす」効果，
    SIE 対 Res2 は「速度をずらさずに順番だけ入れる」効果。
    """

    ordered = False
    mid = False

    def __init__(self, hidden: int, dim: int = 2) -> None:
        super().__init__()
        k = dim // 2
        self.F = _make_func("mlp", 2 * k, hidden, k)  # 速度の増分
        self.G = _make_func("mlp", 2 * k, hidden, k)  # 位置の増分
        self.n_dof = k
        self.delta_t = 0.05
        self.hidden_size = 1

    def forward(self, xv, h):
        k, dt = self.n_dof, self.delta_t
        x, v = xv[:, :k], xv[:, k:2 * k]
        v_new = v + dt * self.F(xv)
        x_new = x + dt * self.G(self._g_input(x, v, v_new, xv))
        return torch.cat([x_new, v_new], dim=1), torch.zeros_like(h)

    def _g_input(self, x, v, v_new, xv):
        """位置の網に渡す速度：順番なしは v_n，順番ありは v_{n+1}，中点版は (v_n + v_{n+1})/2."""
        if self.mid:
            return torch.cat([x, 0.5 * (v + v_new)], dim=1)
        return torch.cat([x, v_new], dim=1) if self.ordered else xv

    def teacher_step(self, xv, xv_next):
        """学習の 1 ステップ（教師強制）. 順番ありの位置の段には参照の新しい速度を入れる.

        LF 系の学習（lf_one_step_loss）は位置の段に参照の v_{n+1/2} を入れるので，速度の段と位置の段の損失が分かれる。
        SIE も同じにする（予測した v_{n+1} を入れると位置の損失が F にも流れ，LF と学習のしかたが変わる。2026-10-18）。
        rollout（forward）では予測した v_{n+1} を使う。
        """
        k, dt = self.n_dof, self.delta_t
        x, v = xv[:, :k], xv[:, k:2 * k]
        v_new = v + dt * self.F(xv)
        x_new = x + dt * self.G(self._g_input(x, v, xv_next[:, k:2 * k], xv))
        return torch.cat([x_new, v_new], dim=1)


class ResSplitMLPState(_TwoNetState):
    """① を 2 つの網に分けた版：x ← x + Δt·G(x_n, v_n)，v ← v + Δt·F(x_n, v_n)（順番なし・速度は同時刻）."""

    ordered = False


class SIEMLPState(_TwoNetState):
    """順番だけを入れた版（半陰的オイラー型）：v ← v + Δt·F(x_n, v_n)，x ← x + Δt·G(x_n, v_{n+1})（速度は同時刻）."""

    ordered = True


class SIEMidMLPState(_TwoNetState):
    """位置の網に中点の速度を渡す版（2026-10-19，ELEX の仕組みの検証）：v ← v + Δt·F(x_n, v_n)，x ← x + Δt·G(x_n, (v_n + v_{n+1})/2).

    速度は整数の時刻のまま（ずらさない）。厳密な解では x_{n+1} − x_n = Δt (v_n + v_{n+1})/2 + O(Δt³) なので，位置の網は
    LF と同じく力をほとんど学ばなくてよい。LF との違いは速度の段（整数の時刻の更新で，中心に置かれていない）だけ。
    学習（teacher_step）では参照の v_{n+1} との平均を入れる。
    """

    ordered = True
    mid = True


class Res2Loc(_LocalMixin, ResSplitMLPState):
    __init__ = _init_two_net_local


class SIELoc(_LocalMixin, SIEMLPState):
    __init__ = _init_two_net_local


class SIEMidLoc(_LocalMixin, SIEMidMLPState):
    __init__ = _init_two_net_local


class ResLoc(_LocalMixin, ResMLPState):
    """Res の段ごとの網の版：1 つの網が各段の (Δq, Δφ) を出す."""

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F = _LocalMLP(2, 2, hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1


class RK4Loc(_LocalMixin, RK4MLPState):
    """RK4 の段ごとの網の版（Res と同じ網を RK4 で 1 ステップ積分）."""

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.F = _LocalMLP(2, 2, hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1


class RK4KCLLoc(_LocalMixin, RK4MLPState):
    """LF-KCL と同じスカラーの特性 u_θ(q)・i_ψ(φ) とキルヒホッフの法則のベクトル場を，整数の時刻の状態 (q_n, φ_n) で RK4 で進める対照
    （2026-10-22 早朝，ELEX：別の説明「特性を学ぶ型の利点は構造から来て，中点の配置からではない」を確かめる）.

    q̇_k = i_ψ(φ_k) − i_ψ(φ_{k+1})，φ̇_k = u_θ(q_{k−1}) − u_θ(q_k)。パラメータは LF_KCL_Loc と同じ（同じ幅なら同数）。
    1 ステップで特性を 4 回ずつ評価する。各段の増分が差の形なので，総電荷・総磁束は RK4 でも保存される。
    """

    kind = "mlp"

    def __init__(self, hidden: int, dim: int = 2) -> None:
        nn.Module.__init__(self)
        self.U, self.I = _ScalarMLP(hidden), _ScalarMLP(hidden)
        self.n_dof, self.delta_t, self.hidden_size = dim // 2, 0.05, 1

    def F(self, z):
        k = self.n_dof
        q, f = z[:, :k], z[:, k:2 * k]
        u, i = self.U(q), self.I(f)
        return torch.cat([i - torch.roll(i, -1, dims=1), torch.roll(u, 1, dims=1) - u], 1)


# ---------------------------------------------------------------------------
# 記憶あり・増分型（2026-10-12 追加）：記憶なし・増分型との差を「記憶」と「出力の形」に分けるための対照
#   LF_LTCs・LTCs_state は次の状態を隠れ状態から直接読み出す（x_{n+1} = W_out(h)）。
#   ここでは隠れ状態の更新（記憶）はそのままにして，読み出しだけを増分にする（x_{n+1} = x_n + Δt·W_out(h)）。
#   パラメータ・幅・LTC の積分は元のモデルと同じ。Δt はデータの刻み（学習時に model.delta_t がデータに合わせられる）。
# ---------------------------------------------------------------------------


class LF_LTCInc(LF_LTC):
    """LF_LTCs の増分版：v ← v + Δt·W_vout(h_v)，x ← x + Δt·W_xout(h_x)（h は元と同じくステップをまたいで持ち越す）."""

    def forward(self, x, h_prev_prev, h_prev, is_velocity_step):
        out, h_new, h_prev = super().forward(x, h_prev_prev, h_prev, is_velocity_step)
        base = x[:, 1:2] if is_velocity_step else x[:, :1]
        return base + self.delta_t * out, h_new, h_prev


class DirectStateLTCInc(DirectStateLTC):
    """LTCs_state の増分版：(x, v) ← (x, v) + Δt·W_hout(h)（h は元と同じくステップをまたいで持ち越す）."""

    def forward(self, xv, h):
        out, h_new = super().forward(xv, h)
        return xv + self.delta_t * out, h_new


class KDK3_RNN(nn.Module):
    """Kick₁ / Drift / Kick₂ で独立な重みを持つ 3 セル構成の Leapfrog 型 RNN.

    以前の振り子の研究のコードの KDK3_RNN の移植（2026-07-13）。
    1 タイムステップ = 3 substep（step_id 0: kick1 で v_half, 1: drift で x,
    2: kick2 で v を予測）。隠れ状態更新の刻みは dt/2, dt, dt/2。
    hs=13 で総パラメータ 705（LF_RNN(hs16) の 674 とほぼ等パラメータ）。
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        in_dim = 2
        self.W_k1_h = nn.Linear(in_dim, hidden)
        self.W_k1_hh = nn.Linear(hidden, hidden)
        self.W_k1_out = nn.Linear(hidden, 1)
        self.W_d_h = nn.Linear(in_dim, hidden)
        self.W_d_hh = nn.Linear(hidden, hidden)
        self.W_d_out = nn.Linear(hidden, 1)
        self.W_k2_h = nn.Linear(in_dim, hidden)
        self.W_k2_hh = nn.Linear(hidden, hidden)
        self.W_k2_out = nn.Linear(hidden, 1)
        self.delta_t = 0.05

    def forward(self, x, h_prev_prev, h_prev, step_id):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros_like(h_prev)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        if step_id == 0:
            f = torch.relu(self.W_k1_h(x) + self.W_k1_hh(h_prev_prev))
            h_new = h_prev_prev + 0.5 * self.delta_t * f
            out = self.W_k1_out(h_new)
        elif step_id == 1:
            f = torch.relu(self.W_d_h(x) + self.W_d_hh(h_prev_prev))
            h_new = h_prev_prev + self.delta_t * f
            out = self.W_d_out(h_new)
        else:
            f = torch.relu(self.W_k2_h(x) + self.W_k2_hh(h_prev_prev))
            h_new = h_prev_prev + 0.5 * self.delta_t * f
            out = self.W_k2_out(h_new)
        return out, h_new, h_prev


class KDK3_LTC(_LTCMixin, nn.Module):
    """KDK3_RNN の LTC 版（時定数ベクトル A は 3 セルで共有）.

    hs=13 で総パラメータ 718（LF_LTCs(hs16) の 690 とほぼ等パラメータ）。
    ltc_variant=paper では τ も 3 セルで共有する。
    """

    def __init__(
        self,
        hidden: int,
        ltc_variant: str = DEFAULT_LTC_VARIANT,
        ltc_activation: str = DEFAULT_LTC_ACTIVATION,
        ltc_unfolds: int = 1,
        ltc_int_time: float | None = None,
    ) -> None:
        super().__init__()
        in_dim = 2
        self.W_k1_h = nn.Linear(in_dim, hidden)
        self.W_k1_hh = nn.Linear(hidden, hidden)
        self.W_k1_out = nn.Linear(hidden, 1)
        self.W_d_h = nn.Linear(in_dim, hidden)
        self.W_d_hh = nn.Linear(hidden, hidden)
        self.W_d_out = nn.Linear(hidden, 1)
        self.W_k2_h = nn.Linear(in_dim, hidden)
        self.W_k2_hh = nn.Linear(hidden, hidden)
        self.W_k2_out = nn.Linear(hidden, 1)
        self.A = nn.Parameter(torch.randn(hidden))
        self.delta_t = 0.05
        self._init_ltc(hidden, ltc_variant, ltc_activation, ltc_unfolds, ltc_int_time)

    def forward(self, x, h_prev_prev, h_prev, step_id):
        if h_prev_prev is None:
            h_prev_prev = torch.zeros_like(h_prev)
        if h_prev is None:
            h_prev = torch.zeros_like(h_prev_prev)
        if step_id == 0:
            h_new = self._ltc_integrate(h_prev_prev, self.W_k1_h(x), self.W_k1_hh, 0.5 * self.delta_t)
            out = self.W_k1_out(h_new)
        elif step_id == 1:
            h_new = self._ltc_integrate(h_prev_prev, self.W_d_h(x), self.W_d_hh, self.delta_t)
            out = self.W_d_out(h_new)
        else:
            h_new = self._ltc_integrate(h_prev_prev, self.W_k2_h(x), self.W_k2_hh, 0.5 * self.delta_t)
            out = self.W_k2_out(h_new)
        return out, h_new, h_prev


def hidden_size_of(model: nn.Module) -> int:
    """隠れ状態の幅（損失・評価で初期状態を作るのに使う）."""
    if hasattr(model, "hidden_size"):
        return int(model.hidden_size)
    for name, attr in (("W_xhh", "out_features"), ("W_k1_hh", "out_features"), ("W_hout", "in_features")):
        if hasattr(model, name):
            return int(getattr(getattr(model, name), attr))
    raise AttributeError("hidden size unknown")



# モデル名（2026-07 リネーム後）:
#   Leapfrog 系:          LF_LTCs (旧 LF_Harmonic_LTCNNs) / LF_RNN (旧 LF_Harmonic_RNN)
#   陽的オイラー更新系:   LTCs (旧 Direct_Harmonic_LTCNNs_equal) / RNN (旧 Direct_Harmonic_RNN_equal)
#   2026-07-10: W_hh を Linear(hidden, hidden) に修正し RNN / LTCs のデフォルトへ昇格。
#               移行期の RNN_full / LTCs_full はエイリアスとして受け付ける。
#   2026-07-13: KDK3_RNN / KDK3_LTCs（3 セル KDK 型、hs13 で LF 系と等パラ）を追加。
#   2026-09-29: RNN_state / LTCs_state（1 段系の状態入力版、(x, v) → (x, v)、hs23 で LF 系と等パラ）を追加。
MODEL_REGISTRY: dict[str, dict] = {
    "LF_RNN": {
        "cls": LF_RNN,
        "family": "lf",
        "default_hs": 16,
    },
    "LF_LTCs": {
        "cls": LF_LTC,
        "family": "lf",
        "default_hs": 16,
    },
    "RNN": {
        "cls": DirectRNN,
        "family": "direct",
        "default_hs": 24,
    },
    "LTCs": {
        "cls": DirectLTC,
        "family": "direct",
        "default_hs": 24,
    },
    "RNN_state": {
        "cls": DirectStateRNN,
        "family": "direct_state",
        "default_hs": 23,
    },
    "LTCs_state": {
        "cls": DirectStateLTC,
        "family": "direct_state",
        "default_hs": 23,
    },
    "Split_RNN_state": {
        "cls": SplitStateRNN,
        "family": "direct_state",
        "default_hs": 16,
    },
    "Split_LTCs_state": {
        "cls": SplitStateLTC,
        "family": "direct_state",
        "default_hs": 16,
    },
    "LF_Sym_MLP": {"cls": LFSymMLP, "family": "lf", "default_hs": 25},
    "LF_Sym_LTC": {"cls": LFSymLTC, "family": "lf", "default_hs": 24},
    "LF_Free_MLP": {"cls": LFFreeMLP, "family": "lf", "default_hs": 16},
    "LF_Free_LTC": {"cls": LFFreeLTC, "family": "lf", "default_hs": 16},
    "LF_Par_MLP": {"cls": LFParMLP, "family": "lf", "default_hs": 16},
    "LF_Kick_MLP": {"cls": LFKickMLP, "family": "lf", "default_hs": 24},
    "LF_Sep_MLP": {"cls": LFSepMLP, "family": "lf", "default_hs": 16},
    "LF_Cen_MLP": {"cls": LFCenMLP, "family": "lf", "default_hs": 16},
    # 2026-10-21 夕：段ごとに同じ網を共有する版（非線形伝送線路）
    "LF_Free_Loc": {"cls": LFFreeLoc, "family": "lf", "default_hs": 16},
    "LF_Par_Loc": {"cls": LFParLoc, "family": "lf", "default_hs": 16},
    "LF_Sep_Loc": {"cls": LFSepLoc, "family": "lf", "default_hs": 16},
    "LF_SepD_Loc": {"cls": LFSepDLoc, "family": "lf", "default_hs": 16},  # 2026-10-21 夜：LF-Sep＋中点の損失の項
    "LF_SepDx_Loc": {"cls": LFSepDExpLoc, "family": "lf", "default_hs": 16},  # 2026-10-21 夜：同じ損失の項を q_n で取る対照
    "LF_KCL_Loc": {"cls": LFKCLLoc, "family": "lf", "default_hs": 17},  # 2026-10-22：素子の特性を学びキルヒホッフの法則でつなぐ（保存する形）
    "LF_KCLD_Loc": {"cls": LFKCLDLoc, "family": "lf", "default_hs": 14},
    "RK4_KCL_Loc": {"cls": RK4KCLLoc, "family": "direct_state", "default_hs": 17},  # 2026-10-22 早朝：LF-KCL と同じ特性を RK4 で進める対照  # 2026-10-22：LF_KCL_Loc＋節点の漏れの電流（中点）
    "Res2_Loc_state": {"cls": Res2Loc, "family": "direct_state", "default_hs": 16},
    "SIE_Loc_state": {"cls": SIELoc, "family": "direct_state", "default_hs": 16},
    "SIEMid_Loc_state": {"cls": SIEMidLoc, "family": "direct_state", "default_hs": 16},
    "Res_Loc_state": {"cls": ResLoc, "family": "direct_state", "default_hs": 20},
    "RK4_Loc_state": {"cls": RK4Loc, "family": "direct_state", "default_hs": 20},
    "RK4_MLP_state": {"cls": RK4MLPState, "family": "direct_state", "default_hs": 24},
    "RK4Kin_MLP_state": {"cls": RK4KinMLPState, "family": "direct_state", "default_hs": 24},
    "Res2_MLP_state": {"cls": ResSplitMLPState, "family": "direct_state", "default_hs": 16},
    "SIE_MLP_state": {"cls": SIEMLPState, "family": "direct_state", "default_hs": 16},
    "SIEMid_MLP_state": {"cls": SIEMidMLPState, "family": "direct_state", "default_hs": 16},
    "Res_MLP_state": {"cls": ResMLPState, "family": "direct_state", "default_hs": 24},
    "Res_LTC_state": {"cls": ResLTCState, "family": "direct_state", "default_hs": 23},
    "LF_LTCs_inc": {"cls": LF_LTCInc, "family": "lf", "default_hs": 16},
    "LTCs_state_inc": {"cls": DirectStateLTCInc, "family": "direct_state", "default_hs": 23},
    "KDK3_RNN": {
        "cls": KDK3_RNN,
        "family": "kdk3",
        "default_hs": 13,
    },
    "KDK3_LTCs": {
        "cls": KDK3_LTC,
        "family": "kdk3",
        "default_hs": 13,
    },
}

# CLI の --model で受け付ける名称（新名称 + 旧名エイリアス）
MODEL_CHOICES: list[str] = list(MODEL_REGISTRY.keys()) + list(LEGACY_MODEL_ALIASES.keys())


def is_ltc_model(name: str) -> bool:
    return issubclass(resolve_model_spec(name)[1]["cls"], _LTCMixin)


def build_model(
    name: str,
    hidden: int,
    *,
    ltc_variant: str = DEFAULT_LTC_VARIANT,
    ltc_activation: str = DEFAULT_LTC_ACTIVATION,
    ltc_unfolds: int = 1,
    ltc_int_time: float | None = None,
    dim: int | None = None,
) -> nn.Module:
    """モデルを生成する。LTC 系にだけ LTC の設定を渡す.

    dim：状態の次元（状態入力の 1 段系だけが使う。None なら 2。調和振動子以外の系で 3 次元のローレンツ方程式など）。
    """
    _, spec = resolve_model_spec(name)
    cls = spec["cls"]
    kw = {}
    if dim is not None and dim != 2:
        if spec["family"] != "direct_state" and not issubclass(cls, _MemorylessLF):
            raise ValueError(f"{name}（{spec['family']} 系）は 2 次元の系にしか使えない（dim={dim}）")
        kw["dim"] = dim
    if issubclass(cls, _LTCMixin):
        return cls(hidden, ltc_variant=ltc_variant, ltc_activation=ltc_activation,
                   ltc_unfolds=ltc_unfolds, ltc_int_time=ltc_int_time, **kw)
    return cls(hidden, **kw)


# 記憶なし・増分型の幅（2026-10-09）：パラメータ数を LF_LTCs（tiny 322 / small 706 / medium 2434）にそろえた
MEMLESS_W2 = {
    "LF_Sym_MLP": {"tiny": 16, "small": 25, "medium": 47}, "LF_Sym_LTC": {"tiny": 15, "small": 24, "medium": 46},
    "LF_Free_MLP": {"tiny": 10, "small": 16, "medium": 32}, "LF_Free_LTC": {"tiny": 10, "small": 16, "medium": 32},
    "Res_MLP_state": {"tiny": 15, "small": 24, "medium": 46}, "Res_LTC_state": {"tiny": 14, "small": 23, "medium": 45},
    # 2026-10-16 の対照：順番を外した ②（② と同じ幅）と RK4（① と同じ幅）
    "LF_Par_MLP": {"tiny": 10, "small": 16, "medium": 32}, "RK4_MLP_state": {"tiny": 15, "small": 24, "medium": 46},
    "LF_Kick_MLP": {"tiny": 16, "small": 24, "medium": 47},  # 位置の段を固定した ②（GNS 型）
    "LF_Sep_MLP": {"tiny": 10, "small": 16, "medium": 32},  # 2026-10-21：入力を 1 つに絞った学習型のシュテルマー・ベルレ法
    "LF_Cen_MLP": {"tiny": 10, "small": 16, "medium": 32},  # 2026-10-21 夕：自分の変数も中点で渡す LF（LF と同じ網）
    # 2026-10-18 の対照：網を 2 つに分けた ①（Res2）と順番だけを入れた ①（SIE）。どちらも LF_Par と同じ 2 つの網・同じ幅
    "Res2_MLP_state": {"tiny": 10, "small": 16, "medium": 32}, "SIE_MLP_state": {"tiny": 10, "small": 16, "medium": 32},
    # 2026-10-19 の対照：位置の網に中点の速度 (v_n + v_{n+1})/2 を渡す ①（SIE と同じ網・同じ幅）
    "SIEMid_MLP_state": {"tiny": 10, "small": 16, "medium": 32},
    # 2026-10-18 の対照：2 階の Neural ODE＋RK4（LF_Kick_MLP と同じ網）
    "RK4Kin_MLP_state": {"tiny": 16, "small": 24, "medium": 47},
}
# large（2026-10-18）は medium の 4 倍。幅が表に無いときはパラメータ数から決める（memoryless_width）
MEMLESS_TARGET_PARAMS = {"tiny": 322, "small": 706, "medium": 2434, "large": 9736}


def memoryless_width(name: str, size: str, dim: int = 2) -> int:
    """記憶なし・増分型の幅。2 次元の系は既存の表，多自由度の系（2026-10-15）は同じパラメータ数に最も近い幅."""
    if dim == 2 and size in MEMLESS_W2.get(name, {}):
        return MEMLESS_W2[name][size]
    target = MEMLESS_TARGET_PARAMS[size]
    best = None
    for w in range(2, 256):
        n = sum(p.numel() for p in build_model(name, w, dim=dim).parameters())
        if best is None or abs(n - target) < best[0]:
            best = (abs(n - target), w)
    return best[1]


def ltc_options_from_config(config: dict) -> dict:
    """config.json から LTC の設定を読む（キーが無い 2026-09 以前の run は legacy / relu）."""
    return {
        "ltc_variant": config.get("ltc_variant", "legacy"),
        "ltc_activation": config.get("ltc_activation", "relu"),
        "ltc_unfolds": int(config.get("ltc_unfolds", 1)),
        "ltc_int_time": config.get("ltc_int_time"),
    }


def resolve_model_spec(name: str) -> tuple[str, dict]:
    """新旧どちらの名称でも (canonical_name, spec) を返す."""
    canonical = canonical_model_name(name)
    return canonical, MODEL_REGISTRY[canonical]
