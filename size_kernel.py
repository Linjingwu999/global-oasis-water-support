import hashlib
import numpy as np
import pandas as pd

YEARS = np.arange(1984, 2022, dtype=int)
SCOPES = ['GLOBAL', 'Africa', 'Asia', 'North America', 'Oceania', 'South America']
QUINTILES = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
B = 2000
MIN_VALID = 1900
BASE_SEED = 20260820

def seed_for(analysis_id: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{analysis_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big", signed=False)

def slope_1d(y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    x = YEARS[ok].astype(float) - 2002.5
    yy = y[ok]
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    return float(np.dot(xc, yy - yy.mean()) / denom) if denom > 0 else np.nan

def row_slopes(y: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[0], np.nan, dtype=float)
    x0 = YEARS.astype(float) - 2002.5
    for i in range(y.shape[0]):
        ok = np.isfinite(y[i])
        if ok.sum() >= 2:
            x = x0[ok]
            yy = y[i, ok]
            xc = x - x.mean()
            d = np.dot(xc, xc)
            if d > 0:
                out[i] = np.dot(xc, yy - yy.mean()) / d
    return out

def interval(draws: np.ndarray) -> tuple[float, float, int, str]:
    finite = draws[np.isfinite(draws)]
    n = int(finite.size)
    if n < 2:
        return np.nan, np.nan, n, "INTERVAL_UNAVAILABLE"
    lo, hi = np.percentile(finite, [2.5, 97.5])
    quality = "INTERVAL_AVAILABLE" if n >= MIN_VALID else "INTERVAL_LOW_VALID_REPLICATES"
    return float(lo), float(hi), n, quality

def group_ids(area: pd.DataFrame, scope: str, q: str) -> pd.DataFrame:
    z = area[area["global_area_quintile"].eq(q)]
    if scope != "GLOBAL":
        z = z[z["continent"].eq(scope)]
    return z.sort_values("OasisID", kind="mergesort").reset_index(drop=True)

def strata_positions(members: pd.DataFrame, scope: str) -> list[np.ndarray]:
    if scope != "GLOBAL":
        return [np.arange(len(members), dtype=int)]
    return [
        np.flatnonzero(members["continent"].to_numpy() == continent)
        for continent in sorted(members["continent"].unique())
    ]

def group_matrices(members: pd.DataFrame, water_indexed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = members["OasisID"].tolist()
    sub = water_indexed.loc[pd.IndexSlice[ids, YEARS], ["eligible", "value"]].reset_index()
    eligible = sub.pivot(index="OasisID", columns="calendar_year", values="eligible").reindex(index=ids, columns=YEARS).to_numpy(bool)
    value = sub.pivot(index="OasisID", columns="calendar_year", values="value").reindex(index=ids, columns=YEARS).to_numpy(float)
    area = members["area_km2"].to_numpy(float)[:, None]
    num = np.where(eligible & np.isfinite(value), area * value, 0.0)
    den = np.where(eligible & np.isfinite(value), area, 0.0)
    return num, den

def bootstrap_series(
    num: np.ndarray,
    den: np.ndarray,
    strata: list[np.ndarray],
    analysis_id: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(seed_for(analysis_id)))
    total_num = np.zeros((B, len(YEARS)), dtype=float)
    total_den = np.zeros((B, len(YEARS)), dtype=float)
    chunk = 100
    for pos in strata:
        n = len(pos)
        for start in range(0, B, chunk):
            stop = min(start + chunk, B)
            idx = pos[rng.integers(0, n, size=(stop - start, n))]
            total_num[start:stop] += num[idx, :].sum(axis=1)
            total_den[start:stop] += den[idx, :].sum(axis=1)
    return np.divide(total_num, total_den, out=np.full_like(total_num, np.nan), where=total_den > 0)

def bootstrap_annual(num: np.ndarray, den: np.ndarray, strata: list[np.ndarray], year_index: int, analysis_id: str) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed_for(analysis_id)))
    total_num = np.zeros(B, dtype=float)
    total_den = np.zeros(B, dtype=float)
    for pos in strata:
        n = len(pos)
        idx = pos[rng.integers(0, n, size=(B, n))]
        total_num += num[idx, year_index].sum(axis=1)
        total_den += den[idx, year_index].sum(axis=1)
    return np.divide(total_num, total_den, out=np.full(B, np.nan), where=total_den > 0)
