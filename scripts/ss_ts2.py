"""ss_ts2.py - Exact-ML (Kalman) SARIMA estimation, ADF and KPSS stationarity tests.
JMP: Time Series (ARIMA/Seasonal ARIMA).  SPSS: Forecasting (ARIMA).  ADF/KPSS: the usual stationarity check before differencing / ARIMA.
numpy / scipy only.
"""
from __future__ import annotations
import math
import numpy as np, pandas as pd
from scipy import stats, optimize, linalg
import matplotlib.pyplot as plt
from ss_common import HE, Result, Lang, save_fig, PALETTE

HE.update({"Stationarity tests": "בדיקות סטציונריות", "Unit-root / stationarity tests": "בדיקות שורש יחידה / סטציונריות"})


# ----------------------------------------------------------------------------
# Exact Gaussian likelihood of an ARMA process by the Kalman filter (Harvey / Gardner et al.)
# ----------------------------------------------------------------------------
def arma_exact_nll(z, phi, theta):
    """Concentrated exact -loglik of zero-mean ARMA(phi,theta) data z (sigma^2 profiled out).
    Returns (nll, sigma2_hat, standardized_innovations)."""
    z = np.asarray(z, float); n = len(z); p, q = len(phi), len(theta); r = max(p, q + 1)
    T = np.zeros((r, r)); T[:p, 0] = phi; T[: r - 1, 1:] = np.eye(r - 1)
    R = np.zeros(r); R[0] = 1.0; R[1: q + 1] = theta
    RR = np.outer(R, R)
    try: P = linalg.solve_discrete_lyapunov(T, RR)
    except Exception: return 1e12, np.nan, None
    a = np.zeros(r); ssq = 0.0; sumlogF = 0.0; std = np.empty(n)
    for t in range(n):
        F = P[0, 0]
        if not np.isfinite(F) or F <= 1e-12: return 1e12, np.nan, None
        v = z[t] - a[0]; std[t] = v / math.sqrt(F); ssq += v * v / F; sumlogF += math.log(F)
        k = (T @ P[:, 0]) / F
        a = T @ a + k * v
        P = T @ P @ T.T + RR - np.outer(k, k) * F
    s2 = ssq / n
    return 0.5 * (n * math.log(2 * math.pi * s2) + n + sumlogF), s2, std


def _pacf_to_coef(r):
    phi = np.zeros(0)
    for rk in r: phi = np.concatenate([phi - rk * phi[::-1], [rk]])
    return phi


def _coef_to_pacf(phi):
    phi = np.array(phi, float); out = np.zeros(len(phi))
    for k in range(len(phi) - 1, -1, -1):
        out[k] = phi[k]
        if abs(out[k]) >= 1: return None
        if k: phi = (phi[:k] + out[k] * phi[:k][::-1]) / (1 - out[k] ** 2)
    return out


def _expand(ar, sar, ma, sma, m):
    """Multiplicative seasonal polynomials -> expanded AR (phi) and MA (theta) coefficient vectors (z_t = sum phi z_{t-i} + e_t + sum theta e_{t-j})."""
    arp = np.concatenate([[1.0], -np.asarray(ar, float)]); map_ = np.concatenate([[1.0], np.asarray(ma, float)])
    for j, c in enumerate(sar, 1):
        s = np.zeros(j * m + 1); s[0] = 1; s[j * m] = -c; arp = np.convolve(arp, s)
    for j, c in enumerate(sma, 1):
        s = np.zeros(j * m + 1); s[0] = 1; s[j * m] = c; map_ = np.convolve(map_, s)
    return -arp[1:], map_[1:]


def sarima_ml(y, p=0, d=0, q=0, P=0, D=0, Q=0, m=1, with_mean=None, start=None):
    """Exact-ML SARIMA(p,d,q)(P,D,Q)_m.  Returns a dict compatible with ss_models2.arima_forecast."""
    from ss_models2 import _diff_poly
    y = np.asarray(y, float); dp = _diff_poly(d, D, m); w = np.convolve(y, dp[::-1], mode="valid") if len(dp) > 1 else y.copy(); nw = len(w)
    if with_mean is None: with_mean = (d + D == 0)
    npar = p + q + P + Q; wm = float(w.mean())

    def unpack(par):
        i = 0; ar = par[i:i + p]; i += p; ma = par[i:i + q]; i += q; sar = par[i:i + P]; i += P; sma = par[i:i + Q]; i += Q
        mu = par[i] if with_mean else 0.0
        return ar, ma, sar, sma, mu

    def nll_native(par):
        ar, ma, sar, sma, mu = unpack(par); phi, th = _expand(ar, sar, ma, sma, m)
        return arma_exact_nll(w - mu, phi, th)[0]

    def to_native(x):                       # unconstrained x -> stationary/invertible native parameters
        i = 0; out = []
        for k, sign in ((p, 1), (q, -1), (P, 1), (Q, -1)):
            if k: out.append(sign * _pacf_to_coef(np.tanh(x[i:i + k])))
            i += k
        if with_mean: out.append(np.array([x[i]]))
        return np.concatenate(out) if out else np.array([])

    def f(x): return nll_native(to_native(x))
    nx = npar + (1 if with_mean else 0); best = None
    starts = [np.concatenate([np.zeros(npar), [wm] if with_mean else []])]
    if start is not None: starts.append(np.asarray(start, float))
    rng = np.random.default_rng(0)
    if npar: starts.append(np.concatenate([rng.normal(0, .3, npar), [wm] if with_mean else []]))
    for x0 in starts:
        if nx == 0: break
        r = optimize.minimize(f, x0, method="BFGS", options={"gtol": 1e-7, "maxiter": 400})
        if best is None or r.fun < best.fun: best = r
    par = to_native(best.x) if nx else np.array([]); nll = float(best.fun) if nx else nll_native(par)
    ar, ma, sar, sma, mu = unpack(par); phi, th = _expand(ar, sar, ma, sma, m); _, s2, std = arma_exact_nll(w - mu, phi, th)
    # standard errors: numerical Hessian of the exact nll in native parameters
    se = np.full(len(par), np.nan)
    if len(par):
        h = 1e-4 * np.maximum(np.abs(par), 0.1); H = np.zeros((len(par), len(par)))
        for i in range(len(par)):
            for j in range(i, len(par)):
                def g(di, dj):
                    x = par.copy(); x[i] += di; x[j] += dj; return nll_native(x)
                H[i, j] = H[j, i] = (g(h[i], h[j]) - g(h[i], -h[j]) - g(-h[i], h[j]) + g(-h[i], -h[j])) / (4 * h[i] * h[j])
        try:
            C = np.linalg.inv(H); se = np.sqrt(np.where(np.diag(C) > 0, np.diag(C), np.nan))
        except np.linalg.LinAlgError: pass
    k = npar + (1 if with_mean else 0) + 1; aic = 2 * nll + 2 * k
    return dict(phi=phi, theta=th, mu=mu, sigma2=s2, resid=std * math.sqrt(s2), std_resid=std, aic=aic, aicc=aic + 2 * k * (k + 1) / max(nw - k - 1, 1), n=nw, dp=dp, w=w, y=y,
                with_mean=with_mean, p=len(phi), q=len(th), d=d, D=D, m=m, loglik=-nll, native=par, native_se=se, orders=(p, d, q, P, D, Q, m),
                names=[f"AR{i+1}" for i in range(p)] + [f"MA{i+1}" for i in range(q)] + [f"SAR{i+1}" for i in range(P)] + [f"SMA{i+1}" for i in range(Q)] + (["Mean"] if with_mean else []))


# ----------------------------------------------------------------------------
# ADF and KPSS
# ----------------------------------------------------------------------------
def _adf_stat(y, k, reg):
    dy = np.diff(y); n = len(dy); rows = n - k
    ylag = y[k:n]                                                      # y_{t-1} aligned with dy[k:]
    Xc = [ylag]
    for i in range(1, k + 1): Xc.append(dy[k - i: n - i])
    if reg in ("c", "ct"): Xc.append(np.ones(rows))
    if reg == "ct": Xc.append(np.arange(1, rows + 1, dtype=float))
    X = np.column_stack(Xc); z = dy[k:]
    beta, res, rk, _ = np.linalg.lstsq(X, z, rcond=None); e = z - X @ beta; s2 = e @ e / (rows - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X); return float(beta[0] / math.sqrt(cov[0, 0])), float(e @ e), rows, X.shape[1]


def adf_test(y, lags=None, reg="c", maxlag=None, ic="aic", reps=3000, seed=11):
    """Augmented Dickey-Fuller.  H0: unit root.  p-value and critical values by Monte-Carlo under a driftless random walk with the same n, lag order and regression type."""
    y = np.asarray(y, float); n = len(y)
    if maxlag is None: maxlag = int(math.ceil(12 * (n / 100) ** 0.25))
    maxlag = min(maxlag, n // 2 - 5)
    if lags is None:
        best = None
        for k in range(0, maxlag + 1):                                 # AIC on a common estimation sample
            dy = np.diff(y); nn = len(dy) - maxlag; z = dy[maxlag:]; Xc = [y[maxlag: len(dy)]]
            for i in range(1, k + 1): Xc.append(dy[maxlag - i: len(dy) - i])
            if reg in ("c", "ct"): Xc.append(np.ones(nn))
            if reg == "ct": Xc.append(np.arange(1, nn + 1, dtype=float))
            X = np.column_stack(Xc); b = np.linalg.lstsq(X, z, rcond=None)[0]; e = z - X @ b; kk = X.shape[1]
            crit = nn * math.log(e @ e / nn) + (2 * kk if ic == "aic" else kk * math.log(nn))
            if best is None or crit < best[0]: best = (crit, k)
        lags = best[1]
    stat, sse, rows, kpar = _adf_stat(y, lags, reg)
    rng = np.random.default_rng(seed); sims = np.empty(reps)
    for i in range(reps):
        sims[i] = _adf_stat(np.cumsum(rng.normal(size=n)), lags, reg)[0]
    pval = (1 + np.sum(sims <= stat)) / (reps + 1); cv = {c: float(np.percentile(sims, c)) for c in (1, 5, 10)}
    return dict(stat=stat, p=float(pval), lags=lags, crit=cv, n_used=rows, reg=reg)


_KPSS = {"c": [(0.10, 0.347), (0.05, 0.463), (0.025, 0.574), (0.01, 0.739)], "ct": [(0.10, 0.119), (0.05, 0.146), (0.025, 0.176), (0.01, 0.216)]}


def kpss_test(y, reg="c", lags=None):
    """KPSS (Kwiatkowski, Phillips, Schmidt, Shin 1992).  H0: stationary around a level (reg='c') or deterministic trend (reg='ct').
    Critical values are the published asymptotic values; p-value is interpolated and bounded to [0.01, 0.10]."""
    y = np.asarray(y, float); n = len(y); t = np.arange(1, n + 1, dtype=float)
    X = np.ones((n, 1)) if reg == "c" else np.column_stack([np.ones(n), t]); e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    if lags is None: lags = int(4 * (n / 100) ** 0.25)
    S = np.cumsum(e); s2 = e @ e / n
    for j in range(1, lags + 1): s2 += 2 * (1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    stat = float((S @ S) / (n * n * s2)); tab = _KPSS[reg]
    crit = {pp: c for pp, c in tab}; cs = [c for _, c in tab]; ps = [pp for pp, _ in tab]
    if stat <= cs[0]: p = 0.10
    elif stat >= cs[-1]: p = 0.01
    else: p = float(np.interp(stat, cs, ps))
    return dict(stat=stat, p=p, lags=lags, crit=crit, reg=reg, bounded=(stat <= cs[0] or stat >= cs[-1]))


def stationarity(df, params, L: Lang, outdir):
    """column; trend ('c' constant | 'ct' constant+trend, default c); lags (ADF, default AIC-selected); kpss_lags; difference (0/1/2: apply before testing)."""
    col = params["column"]; y = df[col].dropna().values.astype(float); k = int(params.get("difference", 0))
    for _ in range(k): y = np.diff(y)
    reg = params.get("trend", "c"); n = len(y); res = Result(L.t("Unit-root / stationarity tests"))
    a = adf_test(y, params.get("lags"), reg, reps=int(params.get("mc_reps", 3000))); kp = kpss_test(y, reg, params.get("kpss_lags"))
    res.table("ADF test (H0: unit root)", pd.DataFrame([{"Regression": {"c": "constant", "ct": "constant + trend"}[reg], "Lags (AIC)" if params.get("lags") is None else "Lags": a["lags"], "n used": a["n_used"], "ADF statistic": a["stat"], "p-value (Monte-Carlo)": a["p"], "Crit 1%": a["crit"][1], "Crit 5%": a["crit"][5], "Crit 10%": a["crit"][10]}]))
    res.table("KPSS test (H0: stationary)", pd.DataFrame([{"Regression": {"c": "level", "ct": "trend"}[reg], "Bandwidth": kp["lags"], "KPSS statistic": kp["stat"], "p-value (interpolated, bounded 0.01-0.10)": kp["p"], "Crit 10%": kp["crit"][0.10], "Crit 5%": kp["crit"][0.05], "Crit 1%": kp["crit"][0.01]}]))
    adf_rej, kp_rej = a["p"] < 0.05, kp["p"] < 0.05
    verdict = ("Both tests agree: STATIONARY (ADF rejects unit root, KPSS does not reject stationarity)" if adf_rej and not kp_rej else
               "Both tests agree: NON-STATIONARY / unit root (ADF does not reject, KPSS rejects) - difference the series" if (not adf_rej) and kp_rej else
               "Both reject: inconclusive - often a structural break or a fractionally integrated series" if adf_rej and kp_rej else
               "Neither rejects: inconclusive - low power / short series")
    res.table("Verdict (5% level)", pd.DataFrame([{"ADF rejects unit root": adf_rej, "KPSS rejects stationarity": kp_rej, "Verdict": verdict}]))
    fig, axs = plt.subplots(2, 1, figsize=(8, 6)); axs[0].plot(y); axs[0].set_title(col + (f" (differenced x{k})" if k else ""))
    from ss_models2 import acf
    r = acf(y, min(30, n // 3)); axs[1].bar(range(len(r)), r, width=.3); b = 1.96 / math.sqrt(n); axs[1].axhline(b, color=PALETTE[3], ls="--"); axs[1].axhline(-b, color=PALETTE[3], ls="--"); axs[1].set_title(L.fig("Autocorrelation"))
    res.fig("stationarity", save_fig(fig, outdir, "stationarity"))
    res.summary.update({"adf_stat": a["stat"], "adf_p": a["p"], "adf_lags": a["lags"], "kpss_stat": kp["stat"], "kpss_p": kp["p"], "verdict": verdict})
    res.note("ADF p-value and critical values are simulated (Monte-Carlo, driftless random walk, same n/lags/regression), so they are exact for this sample size; they differ slightly from MacKinnon's response-surface values. KPSS uses the published asymptotic critical values (KPSS 1992); with short series it can over-reject.")
    return res
